from multiprocessing import sharedctypes
from typing import Optional

import torch.multiprocessing as mp

from . import inference
from .known_inference_models import DetectorModel
from .log import logger
from .loaders.bounding_boxes_dataset import BoundingBoxesDataset
from .loaders.video_frames_dataloader import VideoFramesDataLoader


class ProcessDetection:
    def __init__(
        self,
        detector_model: DetectorModel,
        input_video_frames_loader: VideoFramesDataLoader,
        output_bbox_dataset: BoundingBoxesDataset,
        use_fp16: bool = False,
        device: str = "cpu",
        batch_size: int = 100,
        compile_model: bool = False,
    ):
        self.detector_model: DetectorModel = detector_model
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.compile_model = compile_model

        self.input_video_frames_loader: VideoFramesDataLoader = (
            input_video_frames_loader
        )
        self.output_bbox_dataset: BoundingBoxesDataset = output_bbox_dataset

        self.process: Optional[mp.Process] = None

        self._inference_count: sharedctypes.Synchronized = mp.Value("i", 0)
        self._start_time: sharedctypes.Synchronized = mp.Value("d", -1.0)
        self._stop_time: sharedctypes.Synchronized = mp.Value("d", -1.0)
        self._running_time_from_start: sharedctypes.Synchronized = mp.Value("d", -1.0)
        self._running_time_from_first_inference: sharedctypes.Synchronized = mp.Value(
            "d", -1.0
        )
        self._first_inference_time: sharedctypes.Synchronized = mp.Value("d", -1.0)
        self._time_waiting: sharedctypes.Synchronized = mp.Value("d", 0)

    @property
    def inference_count(self) -> int:
        return self._inference_count.value

    @property
    def first_inference_time(self) -> float:
        return self._first_inference_time.value

    @property
    def start_time(self) -> float:
        return self._start_time.value

    @property
    def stop_time(self) -> float:
        return self._stop_time.value

    @property
    def running_time_from_start(self) -> float:
        return self._running_time_from_start.value

    @property
    def running_time_from_first_inference(self) -> float:
        return self._running_time_from_first_inference.value

    @property
    def time_waiting(self) -> int:
        return self._time_waiting.value

    def add_data_objects(self, data_objects=None):
        if data_objects is None:
            data_objects = []

        for data_object in data_objects:
            self.input_video_frames_loader.dataset.add_data_object(data_object)

    def start(self):
        if self.process is None:
            self.process = mp.Process(target=self._run, args=())
            self.process.start()

    def wait(self):
        self.process.join()
        logger.info(f"ProcessDetection service finished: {self.process.exitcode}")

    def stop(self):
        self.process.close()
        self.process = None

    def _run(self):
        detector = None
        try:
            detector = inference.Detector(
                model_config_path=self.detector_model.model_config,
                checkpoint=self.detector_model.checkpoint,
                deployment_config_path=self.detector_model.deployment_config,
                device=self.device,
                use_fp16=self.use_fp16,
                batch_size=self.batch_size,
                compile_model=self.compile_model,
            )

            logger.info("Running ProcessDetection service...")
            for bbox_tuple in detector.iter_dataloader(
                loader=self.input_video_frames_loader
            ):
                self._inference_count.value = detector.inference_count
                self._start_time.value = detector.start_time
                self._stop_time.value = detector.stop_time
                self._running_time_from_first_inference.value = (
                    detector.running_time_from_first_inference
                )
                self._running_time_from_start.value = detector.running_time_from_start
                self._first_inference_time.value = detector.first_inference_time
                self._time_waiting.value = detector.time_waiting

                self.output_bbox_dataset.add_data_object(bbox_tuple)
        except Exception as e:
            logger.error(e)
            raise e
        finally:
            if detector is not None:
                del detector
            logger.info("ProcessDetection service loop ended")

    def __del__(self):
        del self.input_video_frames_loader
