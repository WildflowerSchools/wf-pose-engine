from typing import Optional

import torch.multiprocessing as mp

from . import inference
from .known_inference_models import PoseModel
from .log import logger
from .loaders.bounding_boxes_dataloader import BoundingBoxesDataLoader
from .loaders.poses_dataset import PosesDataset


class ProcessPoseEstimation:
    def __init__(
        self,
        # pose_estimator: inference.PoseEstimator,
        pose_estimator_model: PoseModel,
        input_bboxes_loader: BoundingBoxesDataLoader,
        output_poses_dataset: PosesDataset,
        device: str = "cpu",
    ):
        # self.pose_estimator: inference.PoseEstimator = pose_estimator
        self.pose_estimator = None
        self.pose_estimator_model: PoseModel = pose_estimator_model
        self.device = device

        self.input_bboxes_loader: BoundingBoxesDataLoader = input_bboxes_loader
        self.output_poses_dataset: PosesDataset = output_poses_dataset

        self.process: Optional[mp.Process] = None

        self._frame_count = mp.Value(
            "i", 0
        )  # Each frame contains multiple boxes, so we track frames separately
        self._inference_count = mp.Value("i", 0)

    @property
    def frame_count(self):
        return self._frame_count.value

    @property
    def inference_count(self):
        return self._inference_count.value

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

        pose_estimator = None
        try:
            pose_estimator = inference.PoseEstimator(
                model_config_path=self.pose_estimator_model.model_config,
                checkpoint=self.pose_estimator_model.checkpoint,
                deployment_config_path=self.pose_estimator_model.deployment_config,
                device=self.device,
                max_boxes_per_inference=40,  # 70,
                # use_fp_16=self.use_fp_16,
            )

            for pose_tuple in pose_estimator.iter_dataloader(
                loader=self.input_bboxes_loader
            ):
                self.output_poses_dataset.add_pose(pose_tuple)
                self._frame_count.value = pose_estimator.frame_count
                self._inference_count.value = pose_estimator.inference_count
        except Exception as e:
            logger.error(e)
            raise e
        finally:
            if pose_estimator is not None:
                del pose_estimator
            logger.info("ProcessPoseEstimation service loop ended")

    def __del__(self):
        del self.input_bboxes_loader
