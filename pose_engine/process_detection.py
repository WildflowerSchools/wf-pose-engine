import torch.multiprocessing as mp

from . import inference
from .log import logger
from .loaders.bounding_boxes_dataloader import BoundingBoxesDataLoader
from .loaders.bounding_boxes_dataset import BoundingBoxesDataset
from .loaders.video_frames_dataloader import VideoFramesDataLoader
from .loaders.video_frames_dataset import VideoFramesDataset


class ProcessDetection:
    def __init__(
        self,
        detector: inference.Detector,
        input_video_frames_loader: VideoFramesDataLoader,
        output_bbox_dataset: BoundingBoxesDataset,
    ):
        self.detector = detector

        self.input_video_frames_loader = input_video_frames_loader
        self.output_bbox_dataset = output_bbox_dataset

        self.process = mp.Process(target=self._run, args=())

    def add_video_files(self, files=[]):
        for file in files:
            self.input_video_frames_loader.dataset.add_video_path(file)

    def start(self):
        self.process.start()

    def wait(self):
        self.process.join()
        logger.info(f"ProcessDetection service finished: {self.process.exitcode}")

    def _run(self):
        logger.info("Running ProcessDetection service...")
        for bbox_tuple in self.detector.iter_dataloader(
            loader=self.input_video_frames_loader
        ):
            self.output_bbox_dataset.add_bboxes(bbox_tuple)

        logger.info("ProcessDetection service loop ended")
