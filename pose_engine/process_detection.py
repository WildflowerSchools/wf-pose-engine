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
        # detector: inference.Detector,
        input_video_frames_loader: VideoFramesDataLoader,
        output_bbox_dataset: BoundingBoxesDataset,
        device: str = "cpu",
    ):
        # self.detector: inference.Detector = detector
        self.detector = None
        self.detector_model: DetectorModel = detector_model
        self.device = device

        self.input_video_frames_loader: VideoFramesDataLoader = (
            input_video_frames_loader
        )
        self.output_bbox_dataset: BoundingBoxesDataset = output_bbox_dataset

        self.process: Optional[mp.Process] = None

        self._inference_count = mp.Value("i", 0)

    @property
    def inference_count(self) -> int:
        return self._inference_count.value

    def add_video_objects(self, video_objects=None):
        if video_objects is None:
            video_objects = []

        for video_object in video_objects:
            self.input_video_frames_loader.dataset.add_video_object(video_object)

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
        # TODO: Consider instantiating detector here, inside the new process. If we do this, we may be able to compile the model and speed up inference.

        detector = None
        try:
            detector = inference.Detector(
                model_config_path=self.detector_model.model_config,
                checkpoint=self.detector_model.checkpoint,
                deployment_config_path=self.detector_model.deployment_config,
                device=self.device,
                # use_fp_16=self.use_fp_16,
            )

            logger.info("Running ProcessDetection service...")
            for bbox_tuple in detector.iter_dataloader(
                loader=self.input_video_frames_loader
            ):
                self.output_bbox_dataset.add_bboxes(bbox_tuple)
                self._inference_count.value = detector.inference_count
        except Exception as e:
            logger.error(e)
            raise e
        finally:
            if detector is not None:
                del detector
            logger.info("ProcessDetection service loop ended")

    def __del__(self):
        del self.input_video_frames_loader
