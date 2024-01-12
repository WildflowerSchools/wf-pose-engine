from contextlib import nullcontext
import time
from typing import Optional, Union

from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import load_config
from mmdet.apis import init_detector
from mmdet.apis.inference import ImagesType
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
from mmpose.utils import adapt_mmdet_pipeline
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.ops

from pose_engine.log import logger


class Detector:
    def __init__(
        self,
        model_config_path: str = None,
        checkpoint: str = None,
        deployment_config_path: str = None,
        device="cpu",
        nms_iou_threshold=0.3,
        bbox_threshold=0.3,
        use_fp_16=False,
    ):
        logger.info("Initializing object detector...")

        if model_config_path is None or checkpoint is None:
            raise ValueError(
                "Detector must be initialized by providing a config + checkpoint pair"
            )

        self.model_config_path = model_config_path
        self.checkpoint = checkpoint
        self.deployment_config_path = deployment_config_path
        self.deployment_config = None

        self.use_fp_16 = use_fp_16

        self.lock = mp.Lock()

        self.using_tensort = self.deployment_config_path is not None
        if not self.using_tensort:  # Standard PYTorch
            detector = init_detector(
                config=self.model_config_path, checkpoint=self.checkpoint, device=device
            )
            detector.cfg = adapt_mmdet_pipeline(detector.cfg)
            detector.share_memory()

            if self.use_fp_16:
                self.detector = detector.half().to(device)
            else:
                self.detector = detector

            self.model_config = detector.cfg

            logger.info("Compiling detector model...")
            self.detector = torch.compile(self.detector, mode="max-autotune")
            logger.info("Finished compiling detector model")

        else:  # TensorRT
            self.model_config, self.deployment_config = load_config(
                self.model_config_path, self.deployment_config_path
            )
            task_processor = build_task_processor(
                model_cfg=self.model_config,
                deploy_cfg=self.deployment_config,
                device=device,
            )
            self.detector = task_processor.build_backend_model([self.checkpoint])

        cfg = self.model_config.copy()
        self.pipeline = get_test_pipeline_cfg(cfg)
        # if isinstance(imgs[0], np.ndarray):
        # Calling this method across libraries will result
        # in module unregistered error if not prefixed with mmdet.
        self.pipeline[0].type = "mmdet.LoadImageFromNDArray"
        self.pipeline = Compose(self.pipeline)

        # TODO: Solve issue with compiled model not being serializable
        # self.detector = torch.compile(self.detector)

        self.nms_iou_threshold = nms_iou_threshold
        self.bbox_threshold = bbox_threshold

        self._inference_count = mp.Value("i", 0)
        self._processing_start_time = mp.Value("d", -1.0)
        self._first_inference_time = mp.Value("d", -1.0)
        self._time_waiting = mp.Value("d", 0)

        logger.info("Finished initializing detector")

    @property
    def inference_count(self) -> int:
        return self._inference_count.value

    @property
    def processing_start_time(self) -> float:
        return self._processing_start_time.value

    @property
    def first_inference_time(self) -> float:
        return self._first_inference_time.value

    @property
    def time_waiting(self) -> int:
        return self._time_waiting.value

    def inference_detector(
        self,
        # model: nn.Module,
        imgs: ImagesType,
        test_pipeline: Optional[Compose] = None,
        text_prompt: Optional[str] = None,
        custom_entities: bool = False,
    ) -> Union[DetDataSample, SampleList]:
        """Custom inference image(s) with the detector. This allows true batch processing.

        Args:
            model (nn.Module): The loaded detector.
            imgs (str, ndarray, Sequence[str/ndarray]):
            Either image files or loaded images.
            test_pipeline (:obj:`Compose`): Test pipeline.

        Returns:
            :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
            If imgs is a list or tuple, the same length list type results
            will be returned, otherwise return the detection results directly.
        """

        if isinstance(imgs, (list, tuple)):
            is_batch = True
        else:
            imgs = [imgs]
            is_batch = False

        if self.detector.data_preprocessor.device.type == "cpu":
            for m in self.detector.modules():
                assert not isinstance(
                    m, RoIPool
                ), "CPU inference with RoIPool is not supported currently."

        all_inputs = []
        all_data_samples = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # TODO: remove img_id.
                data_ = {"img": img, "img_id": 0}
            else:
                # TODO: remove img_id.
                data_ = {"img_path": img, "img_id": 0}

            if text_prompt:
                data_["text"] = text_prompt
                data_["custom_entities"] = custom_entities

            # build the data pipeline
            data_ = self.pipeline(data_)
            all_inputs.append(data_["inputs"])
            all_data_samples.append(data_["data_samples"])

        data_ = {"inputs": all_inputs, "data_samples": all_data_samples}

        # forward the model
        with torch.no_grad():
            # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            # start = time.time()
            result_list = self.detector.test_step(data_)
            # end = time.time()
            # logger.info(f"Detector inference time: {len(imgs)} images in {end - start} seconds")

        if not is_batch:
            return result_list[0]

        return result_list

    def iter_dataloader(self, loader: torch.utils.data.DataLoader):
        """Runs detection against all items in the provided loader

        :param loader: The dataloader object
        :type loader: torch.utils.data.DataLoader
        :returns: a generator of tuples (bboxes, frame, meta)
        :rtype: generator
        """
        last_loop_start_time = time.time()
        self._processing_start_time.value = last_loop_start_time
        for batch_idx, (frames, meta) in enumerate(loader):
            current_loop_time = time.time()
            self._time_waiting.value += current_loop_time - last_loop_start_time

            if self._first_inference_time.value == -1:
                self._first_inference_time.value = current_loop_time

            try:
                logger.debug(
                    f"Processing detector batch #{batch_idx} - Includes {len(frames)} frames"
                )
                meta_as_list_of_dicts = []
                for idx in range(len(frames)):
                    meta_item = {}
                    for key in meta.keys():
                        meta_item[key] = meta[key][idx]

                    meta_as_list_of_dicts.append(meta_item)

                list_np_imgs = list(
                    frames.cpu().detach().numpy()
                )  # Annoying we have to move frames/images to the CPU to run detector

                self.lock.acquire()
                with torch.cuda.amp.autocast() if self.use_fp_16 else nullcontext():
                    det_results = self.inference_detector(
                        # model=self.detector,
                        imgs=list_np_imgs
                    )
                self.lock.release()

                self._inference_count.value += len(frames)

                for idx, det_result in enumerate(det_results):
                    frame = frames[idx]
                    meta_dictionary = meta_as_list_of_dicts[idx]

                    bboxes_and_scores = torch.concatenate(
                        (
                            det_result.pred_instances.bboxes,
                            det_result.pred_instances.labels[:, None],
                            det_result.pred_instances.scores[:, None],
                        ),
                        axis=1,
                    )

                    # Filter by label (person == 0) and bbox_threshold
                    bboxes_and_scores = bboxes_and_scores[
                        torch.logical_and(
                            bboxes_and_scores[:, 4] == 0,
                            bboxes_and_scores[:, 5] > self.bbox_threshold,
                        )
                    ]

                    # Filter by NMS iou_threshold
                    nms_filter_idxs = torchvision.ops.nms(
                        boxes=bboxes_and_scores[:, :4],
                        scores=bboxes_and_scores[:, 5],
                        iou_threshold=self.nms_iou_threshold,
                    )
                    retained_bboxes = bboxes_and_scores[nms_filter_idxs]

                    # Remove label id but retain score
                    retained_bboxes = torch.concatenate(
                        (
                            retained_bboxes[:, :4],
                            retained_bboxes[:, 5, None],
                        ),
                        axis=1,
                    )  # [x, y, [x|w], [y|h], score]

                    yield retained_bboxes, frame, meta_dictionary
            finally:
                del meta
                del frames

            last_loop_start_time = time.time()

    def __del__(self):
        del self.detector
