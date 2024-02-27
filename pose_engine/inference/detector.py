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
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.ops

# import torch_tensorrt

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
        use_fp16=False,
        batch_size=100,
        compile_engine: Optional[str] = None,
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

        self.use_fp16 = use_fp16

        self.batch_size = batch_size

        self.compile_engine = compile_engine

        self.lock = mp.Lock()

        self.using_tensort = self.deployment_config_path is not None
        if not self.using_tensort:  # Standard PYTorch
            self.detector = init_detector(
                config=self.model_config_path, checkpoint=self.checkpoint, device=device
            )
            self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
            self.detector.share_memory()

            self.model_config = self.detector.cfg

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

        if self.use_fp16:
            self.detector = self.detector.half().to(device)
        else:
            self.detector = self.detector

        if self.compile_engine is not None:
            logger.info("Compiling detector model...")

            if self.compile_engine == "inductor":
                self.detector = torch.compile(
                    self.detector, dynamic=False, mode="max-autotune"
                )
            elif self.compile_engine == "tensorrt":
                # Attempted to use torch_tensorrt backend on 2/14/2024, observed no speed up, retaining stub for future use/testing
                self.detector = torch.compile(
                    self.detector,
                    backend="torch_tensorrt",
                    dynamic=False,
                    options={
                        "truncate_long_and_double": True,
                        "precision": torch.half if self.use_fp16 else torch.float,
                        # "debug": True,
                        "min_block_size": 10,
                        # "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
                        "optimization_level": 3,
                        "use_python_runtime": False,
                        "device": self.device,
                    },
                )
            logger.info("Finished compiling detector model")

        cfg = self.model_config.copy()
        self.pipeline = get_test_pipeline_cfg(cfg)
        # if isinstance(imgs[0], np.ndarray):
        # Calling this method across libraries will result
        # in module unregistered error if not prefixed with mmdet.
        self.pipeline[0].type = "mmdet.LoadImageFromNDArray"
        self.pipeline = Compose(self.pipeline)

        self.nms_iou_threshold = nms_iou_threshold
        self.bbox_threshold = bbox_threshold

        self._inference_count = mp.Value("i", 0)
        self._start_time = mp.Value("d", -1.0)
        self._stop_time = mp.Value("d", -1.0)
        self._first_inference_time = mp.Value("d", -1.0)
        self._time_waiting = mp.Value("d", 0)

        logger.info("Finished initializing detector")

    @property
    def inference_count(self) -> int:
        return self._inference_count.value

    @property
    def start_time(self) -> float:
        return self._start_time.value

    @property
    def stop_time(self) -> float:
        return self._stop_time.value

    @property
    def first_inference_time(self) -> float:
        return self._first_inference_time.value

    @property
    def running_time_from_start(self) -> float:
        if self.start_time == -1:
            return 0

        current_time_or_stop_time = time.time()
        if self.stop_time > -1.0:
            current_time_or_stop_time = self.stop_time

        return current_time_or_stop_time - self.start_time

    @property
    def running_time_from_first_inference(self) -> float:
        if self.first_inference_time == -1:
            return 0

        current_time_or_stop_time = time.time()
        if self.stop_time > -1.0:
            current_time_or_stop_time = self.stop_time

        return current_time_or_stop_time - self.first_inference_time

    @property
    def time_waiting(self) -> int:
        return self._time_waiting.value

    def inference_detector(
        self,
        imgs: ImagesType,
        text_prompt: Optional[str] = None,
        custom_entities: bool = False,
    ) -> Union[DetDataSample, SampleList]:
        """Custom inference image(s) with the detector. This allows true batch processing.

        Args:
            imgs (str, ndarray, Sequence[str/ndarray]):
            Either image files or loaded images.

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
        self._start_time.value = last_loop_start_time
        for batch_idx, (frames, meta) in enumerate(loader):
            current_loop_time = time.time()
            self._time_waiting.value += current_loop_time - last_loop_start_time

            if self._first_inference_time.value == -1:
                self._first_inference_time.value = current_loop_time

            try:
                all_det_results = []
                meta_as_list_of_dicts = []
                batch_chunk_size = self.batch_size
                for chunk_ii in range(0, len(frames), batch_chunk_size):
                    logger.debug(
                        f"Processing detector batch #{batch_idx}:{chunk_ii} - Includes {len(frames)} frames"
                    )

                    sub_frames = frames[chunk_ii : chunk_ii + batch_chunk_size]
                    logger.debug(
                        f"Running detections against {len(sub_frames)} frames..."
                    )
                    for idx in range(len(sub_frames)):
                        meta_item = {}
                        for key in meta.keys():
                            meta_item[key] = meta[key][idx]

                        meta_as_list_of_dicts.append(meta_item)

                    list_np_imgs = list(
                        frames.cpu().detach().numpy()
                    )  # Annoying we have to move frames/images to the CPU to run detector

                    self.lock.acquire()
                    with torch.cuda.amp.autocast() if self.use_fp16 else nullcontext():
                        det_results = self.inference_detector(
                            # model=self.detector,
                            imgs=list_np_imgs
                        )
                        all_det_results = [*all_det_results, *det_results]
                    self.lock.release()

                    self._inference_count.value += len(sub_frames)

                for idx, det_result in enumerate(all_det_results):
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

        self._stop_time.value = time.time()

    def __del__(self):
        del self.detector
