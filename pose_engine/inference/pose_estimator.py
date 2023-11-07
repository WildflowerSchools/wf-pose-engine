from typing import List, Optional, Union

import numpy as np
import torch.nn as nn
import torch.utils.data

from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.structures import merge_data_samples, PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy
from PIL import Image

from pose_engine.log import logger


class PoseEstimator:
    def __init__(
        self,
        preset_model: str = None,
        config: str = None,
        checkpoint: str = None,
        device: str = "cuda:1",
    ):
        logger.info("Initializing pose estimator...")

        if preset_model is None and (config is None or checkpoint is None):
            raise ValueError(
                "Pose estimator must be initialized with a default_model setting or by providing a config + checkpoint pair"
            )

        if preset_model == "small":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-s_8xb256-420e_body8-256x192.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth"
        elif preset_model == "medium_256":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
        elif preset_model == "medium_384":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-384x288.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth"
        elif preset_model == "large_256":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"
        elif preset_model == "large_384":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-384x288.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth"

        pose_config = config
        pose_checkpoint = checkpoint
        pose_estimator = init_pose_estimator(
            config=pose_config, checkpoint=pose_checkpoint, device=device
        )

        pose_estimator.share_memory()
        self.pose_estimator = pose_estimator

    def inference_topdown(
        self,
        model: nn.Module,
        imgs: Union[list[np.ndarray], list[str]],
        bboxes: Optional[Union[List[List], List[np.ndarray]]] = None,
        bbox_format: str = "xyxy",
    ) -> List[PoseDataSample]:
        """Inference image with a top-down pose estimator.

        Args:
            model (nn.Module): The top-down pose estimator
            img (np.ndarray | str): The loaded image or image file to inference
            bboxes (np.ndarray, optional): The bboxes in shape (N, 4), each row
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
        scope = model.cfg.get("default_scope", "mmpose")
        if scope is not None:
            init_default_scope(scope)
        pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

        # construct batch data samples
        data_list = []
        for img_idx, img in enumerate(imgs):
            img_bboxes = bboxes[img_idx]
            if img_bboxes is None:
                # get bbox from the image size
                if isinstance(img, str):
                    w, h = Image.open(img).size
                else:
                    h, w = img.shape[:2]

                img_bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
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
                if isinstance(img, str):
                    data_info = dict(img_path=img)
                else:
                    data_info = dict(img=img)
                data_info["bbox"] = bbox[None]  # shape (1, 4)
                data_info["bbox_score"] = np.ones(1, dtype=np.float32)  # shape (1,)
                data_info.update(model.dataset_meta)
                data_list.append(pipeline(data_info))

        if len(data_list) > 0:
            logger.info(
                f"Running pose estimation against {len(data_list)} data samples"
            )
            # collate data list into a batch, which is a dict with following keys:
            # batch['inputs']: a list of input images
            # batch['data_samples']: a list of :obj:`PoseDataSample`
            batch = pseudo_collate(data_list)
            with torch.no_grad():
                results = model.test_step(batch)
        else:
            results = []

        return results

    def iter_dataloader(self, loader: torch.utils.data.DataLoader):
        """Runs pose estimation against all items in the provided loader

        :param loader: The dataloader object
        :type loader: torch.utils.data.DataLoader
        :returns: a generator of tuples (poses, meta)
        :rtype: generator
        """
        logger.info("Running pose estimation against dataloader object...")

        total_frame_count = 0
        for batch_idx, (bboxes, frames, meta) in enumerate(loader):
            total_frame_count += len(frames)

            logger.info(
                f"Processing pose estimation batch #{batch_idx} - Includes {len(frames)} frames"
            )
            imgs = []
            for idx, img in enumerate(frames):
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()

                imgs.append(img)

                if isinstance(bboxes[idx], torch.Tensor):
                    bboxes[idx] = bboxes[idx].detach().cpu().numpy()

            # TODO: Update inference_topdown to work with Tensors
            pose_results = self.inference_topdown(
                model=self.pose_estimator, imgs=imgs, bboxes=bboxes
            )
            # data_samples = merge_data_samples(pose_results)

            yield pose_results
