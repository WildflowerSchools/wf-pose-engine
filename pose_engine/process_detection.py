from typing import Optional

import torch.multiprocessing as mp

from . import inference
from .log import logger
from .loaders.bounding_boxes_dataset import BoundingBoxesDataset
from .loaders.video_frames_dataloader import VideoFramesDataLoader


class ProcessDetection:
    def __init__(
        self,
        detector: inference.Detector,
        input_video_frames_loader: VideoFramesDataLoader,
        output_bbox_dataset: BoundingBoxesDataset,
    ):
        self.detector: inference.Detector = detector

        self.input_video_frames_loader: VideoFramesDataLoader = (
            input_video_frames_loader
        )
        self.output_bbox_dataset: BoundingBoxesDataset = output_bbox_dataset

        self.process: Optional[mp.Process] = None

    def add_video_objects(self, video_objects=None):
        if video_objects is None:
            video_objects = []

        for video_object in video_objects:
            self.input_video_frames_loader.dataset.add_video_object(video_object)

    def total_video_files_processed(self) -> int:
        return 0

    def total_video_frames_processed(self) -> int:
        return 0

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
        logger.info("Running ProcessDetection service...")
        for bbox_tuple in self.detector.iter_dataloader(
            loader=self.input_video_frames_loader
        ):
            self.output_bbox_dataset.add_bboxes(bbox_tuple)

        logger.info("ProcessDetection service loop ended")

    def __del__(self):
        del self.input_video_frames_loader
