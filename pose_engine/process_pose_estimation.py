import torch.multiprocessing as mp

from . import inference
from .log import logger
from .loaders.bounding_boxes_dataloader import BoundingBoxesDataLoader
from .loaders.bounding_boxes_dataset import BoundingBoxesDataset
from .loaders.video_frames_dataloader import VideoFramesDataLoader
from .loaders.video_frames_dataset import VideoFramesDataset


class ProcessPoseEstimation:
    def __init__(
        self,
        pose_estimator: inference.PoseEstimator,
        input_bboxes_loader: BoundingBoxesDataLoader,
    ):
        self.pose_estimator = pose_estimator
        self.input_bboxes_loader = input_bboxes_loader

        self.process = mp.Process(target=self._run, args=())

    def start(self):
        self.process.start()

    def wait(self):
        self.process.join()
        logger.info(f"ProcessPoseEstimation service finished: {self.process.exitcode}")

    def mark_detector_completed(self):
        self.input_bboxes_loader.dataset.done_loading()

    def _run(self):
        logger.info("Running ProcessPoseEstimation service...")

        pose_data = []
        for pose_tuple in self.pose_estimator.iter_dataloader(
            loader=self.input_bboxes_loader
        ):
            pose_data.append(pose_tuple)
