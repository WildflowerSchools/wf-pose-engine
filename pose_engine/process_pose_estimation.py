from typing import Optional

import torch.multiprocessing as mp

from . import inference
from .log import logger
from .loaders.bounding_boxes_dataloader import BoundingBoxesDataLoader
from .loaders.poses_dataset import PosesDataset


class ProcessPoseEstimation:
    def __init__(
        self,
        pose_estimator: inference.PoseEstimator,
        input_bboxes_loader: BoundingBoxesDataLoader,
        output_poses_dataset: PosesDataset,
    ):
        self.pose_estimator: inference.PoseEstimator = pose_estimator
        self.input_bboxes_loader: BoundingBoxesDataLoader = input_bboxes_loader
        self.output_poses_dataset: PosesDataset = output_poses_dataset

        self.process: Optional[mp.Process] = None

    def start(self):
        if self.process is None:
            self.process = mp.Process(target=self._run, args=())
            self.process.start()

    def wait(self):
        self.process.join()
        logger.info(f"ProcessPoseEstimation service finished: {self.process.exitcode}")

    def stop(self):
        self.process.close()
        self.process = None

    def mark_detector_drained(self):
        self.input_bboxes_loader.dataset.done_loading()

    def _run(self):
        logger.info("Running ProcessPoseEstimation service...")

        for pose_tuple in self.pose_estimator.iter_dataloader(
            loader=self.input_bboxes_loader
        ):
            self.output_poses_dataset.add_pose(pose_tuple)

        logger.info("ProcessPoseEstimation service loop ended")
