from contextlib import nullcontext
from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.utils.data

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmpose.apis.inference import dataset_meta_from_config
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy
from PIL import Image
import torch.multiprocessing as mp

from pose_engine.log import logger


class PoseEstimator:
    def __init__(
        self,
        model_config_path: str = None,
        checkpoint: str = None,
        deployment_config_path: str = None,
        device: str = "cuda:1",
        max_boxes_per_inference: int = 75,
        use_fp_16: bool = False,
    ):
        logger.info("Initializing pose estimator...")

        if model_config_path is None or checkpoint is None:
            raise ValueError(
                "Pose estimator must be initialized by providing a config + checkpoint pair"
            )

        self.model_config_path = model_config_path
        self.checkpoint = checkpoint
        self.deployment_config_path = deployment_config_path
        self.deployment_config = None

        self.max_boxes_per_inference = max_boxes_per_inference

        self.use_fp_16 = use_fp_16

        self.lock = mp.Lock()

        self.pose_estimator = None
        self.task_processor = None

        self.using_tensort = self.deployment_config_path is not None
        if not self.using_tensort:  # Standard PYTorch
            pose_estimator = init_pose_estimator(
                config=self.model_config_path, checkpoint=self.checkpoint, device=device
            )
            pose_estimator.share_memory()

            if self.use_fp_16:
                self.pose_estimator = pose_estimator.half().to(device)
            else:
                self.pose_estimator = pose_estimator

            self.model_config = pose_estimator.cfg

            logger.info("Compiling pose estimator model...")
            self.pose_estimator = torch.compile(
                self.pose_estimator, mode="max-autotune"
            )
            logger.info("Finished compiling pose estimator model")

        else:  # TensorRT
            self.model_config, self.deployment_config = load_config(
                self.model_config_path, self.deployment_config_path
            )
            self.task_processor = build_task_processor(
                model_cfg=self.model_config,
                deploy_cfg=self.deployment_config,
                device=device,
            )
            self.pose_estimator = self.task_processor.build_backend_model(
                [self.checkpoint]
            )

        self.pipeline = Compose(self.model_config.test_dataloader.dataset.pipeline)

        self.dataset_meta = dataset_meta_from_config(
            self.model_config, dataset_mode="train"
        )
        # TODO: Solve issue with compiled model not being serializable
        # self.pose_estimator = torch.compile(self.pose_estimator)
        self._frame_count = mp.Value(
            "i", 0
        )  # Each frame contains multiple boxes, so we track frames separately
        self._inference_count = mp.Value("i", 0)

        logger.info("Finished initializing pose estimator")

    @property
    def frame_count(self):
        return self._frame_count.value

    @property
    def inference_count(self):
        return self._inference_count.value

    def inference_topdown(
        self,
        imgs: Union[list[np.ndarray], list[str]],
        bboxes: Optional[Union[List[List], List[np.ndarray]]] = None,
        meta: List[dict] = None,
        bbox_format: str = "xyxy",
    ) -> List[PoseDataSample]:
        """Inference image with a top-down pose estimator.

        Args:
            model (nn.Module): The top-down pose estimator
            img (np.ndarray | str): The loaded image or image file to inference
            bboxes (np.ndarray, optional): The bboxes in shape (N, 5), each row
                represents a bbox. If not given, the entire image will be regarded
                as a single bbox area. Defaults to ``None``
            bbox_format (str): The bbox format indicator. Options are ``'xywh'``
                and ``'xyxy'``. Defaults to ``'xyxy'``

        Returns:
            List[:obj:`PoseDataSample`]: The inference results. Specifically, the
            predicted keypoints and scores are saved at
            ``data_sample.pred_instances.keypoints`` and
            ``data_sample.pred_instances.keypoint_scores``.
        """
        scope = self.model_config.get("default_scope", "mmpose")
        if scope is not None:
            init_default_scope(scope)

        # construct batch data samples
        data_list = []
        meta_mapping = []
        for img_idx, img in enumerate(imgs):
            img_bboxes = bboxes[img_idx]
            if img_bboxes is None:
                # get bbox from the image size
                if isinstance(img, str):
                    w, h = Image.open(img).size
                else:
                    h, w = img.shape[:2]

                img_bboxes = np.array([[0, 0, w, h, 1]], dtype=np.float32)
            elif len(img_bboxes) == 0:
                continue
            else:
                if isinstance(img_bboxes, list):
                    img_bboxes = np.array(img_bboxes)

                assert bbox_format in {
                    "xyxy",
                    "xywh",
                }, f'Invalid bbox_format "{bbox_format}".'

                if bbox_format == "xywh":
                    img_bboxes = bbox_xywh2xyxy(img_bboxes)

            for bbox in img_bboxes:
                # meta_mapping.extend([meta[img_idx]] * len(img_bboxes))
                meta_mapping.append(meta[img_idx])

                if isinstance(img, str):
                    data_info = {"img_path": img}
                else:
                    data_info = {"img": img}
                data_info["bbox"] = bbox[None, :4]  # shape (1, 4)
                data_info["bbox_score"] = bbox[None, 4]  # shape (1,)

                data_info.update(self.dataset_meta)
                data_list.append(self.pipeline(data_info))

        results = []
        if len(data_list) > 0:
            # collate data list into a batch, which is a dict with following keys:
            # batch['inputs']: a list of input images
            # batch['data_samples']: a list of :obj:`PoseDataSample`
            chunk_size = self.max_boxes_per_inference
            for chunk_ii in range(0, len(data_list), chunk_size):
                sub_data_list = data_list[chunk_ii : chunk_ii + chunk_size]

                logger.debug(
                    f"Running pose estimation against {len(sub_data_list)} data samples"
                )

                batch = pseudo_collate(sub_data_list)
                with torch.no_grad():
                    #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                    if self.using_tensort:
                        input_shape = get_input_shape(self.deployment_config)
                        model_inputs, _ = self.task_processor.create_input(
                            imgs[0], input_shape
                        )
                        # When trying to use TensorRT I get an invalid memory access error

                        # Look at context creation in mmdeploy/backend/tensort/wrapper.py, seems like the problem is somewhere in there
                        # Other ideas: there's a post here about running two tensorrt models at the same time: https://forums.developer.nvidia.com/t/can-i-inference-two-engine-simultaneous-on-jetson-using-tensorrt/64396/3
                        # Here's one solution which is to share (or maybe separate) contexts: https://forums.developer.nvidia.com/t/unable-to-run-two-tensorrt-models-in-a-cascade-manner/145274/3b

                        # The actual error is raised in the to_numpy call in this file: /home/ben/.cache/pypoetry/virtualenvs/wf-pose-engine-_vUPQedX-py3.11/lib/python3.11/site-packages/mmpose/models/heads/base_head.py

                        # Ultimately, the solution is probably some sort of version mismatch: https://github.com/pytorch/pytorch/issues/21819
                        # I think I want:
                        # CUDA 11.8       (current 12.2)
                        # cuDNN 8.9.0     (current 8.9.7.29_cuda12)
                        # pytorch 2.1.2   (current 2.1.0)
                        # tensorRT 8.6.x  (current 8.6.1.6)
                        results.extend(self.pose_estimator.test_step(model_inputs))
                    else:
                        results.extend(self.pose_estimator.test_step(batch))

        for res_idx, _result in enumerate(results):
            results[res_idx].pred_instances["custom_metadata"] = [meta_mapping[res_idx]]

        return results

    def iter_dataloader(self, loader: torch.utils.data.DataLoader):
        """Runs pose estimation against all items in the provided loader

        :param loader: The dataloader object
        :type loader: torch.utils.data.DataLoader
        :returns: a generator of tuples (poses, meta)
        :rtype: generator
        """
        for batch_idx, (bboxes, frames, meta) in enumerate(loader):
            try:
                self._frame_count.value += len(frames)

                logger.debug(
                    f"Processing pose estimation batch #{batch_idx} - Includes {len(frames)} frames"
                )
                # meta_mapping = []
                imgs = []
                for img_idx, img in enumerate(frames):
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()

                    imgs.append(img)

                    # if bboxes is None or len(bboxes[idx]) == 0:
                    #     meta_mapping.extend([meta])
                    # else:
                    #     meta_mapping.extend([meta] * len(bboxes[img_idx]))

                    if bboxes is not None and isinstance(bboxes[img_idx], torch.Tensor):
                        bboxes[img_idx] = bboxes[img_idx].detach().cpu().numpy()

                # TODO: Update inference_topdown to work with Tensors

                # self.lock.acquire()
                with torch.cuda.amp.autocast() if self.use_fp_16 and not self.using_tensort else nullcontext():
                    pose_results = self.inference_topdown(
                        # model=self.pose_estimator,
                        imgs=imgs,
                        bboxes=bboxes,
                        meta=meta,
                    )
                # self.lock.release()

                if bboxes is not None:
                    for img_idx, img in enumerate(imgs):
                        self._inference_count.value += len(bboxes[img_idx])

                # data_samples = merge_data_samples(pose_results)

                if pose_results and len(pose_results) > 0:
                    for idx, pose_result in enumerate(pose_results):
                        if pose_result is None:
                            continue

                        pose_result_keypoints = pose_result.pred_instances["keypoints"][
                            0
                        ]
                        pose_result_keypoint_visible = pose_result.pred_instances[
                            "keypoints_visible"
                        ][0]
                        pose_result_keypoint_scores = pose_result.pred_instances[
                            "keypoint_scores"
                        ][0]
                        pose_result_bboxes = pose_result.pred_instances["bboxes"][0]
                        pose_result_bbox_scores = pose_result.pred_instances[
                            "bbox_scores"
                        ][0]
                        pose_result_metadata = pose_result.pred_instances[
                            "custom_metadata"
                        ][0]

                        pose_prediction = np.concatenate(
                            (
                                pose_result_keypoints,  # 0, 1 = X, Y,
                                np.full_like(
                                    np.expand_dims(
                                        pose_result_keypoint_visible, axis=1
                                    ),
                                    -1,
                                ),  # 2 = visibility - mmPose doesn't produce actual visibility values, it simply duplicates scores. For now default the value to -1.
                                np.expand_dims(
                                    pose_result_keypoint_scores, axis=1
                                ),  # 3 = confidence
                            ),
                            axis=1,
                        )
                        box_prediction = np.concatenate(
                            (
                                pose_result_bboxes,  # 0, 1, 2, 3 = X1, Y1, X2, Y2
                                np.expand_dims(
                                    pose_result_bbox_scores, axis=0
                                ),  # 5 = confidence
                            )
                        )
                        yield pose_prediction, box_prediction, pose_result_metadata
            finally:
                del bboxes
                del frames
                del meta

    def __del__(self):
        if self.pose_estimator is not None:
            del self.pose_estimator
