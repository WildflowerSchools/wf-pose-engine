from typing import Optional, Union

import torch.multiprocessing as mp

from . import inference
from .known_inference_models import PoseModel
from .log import logger
from .loaders.bounding_boxes_dataloader import BoundingBoxesDataLoader
from .loaders.video_frames_dataloader import VideoFramesDataLoader
from .loaders.poses_dataset import PosesDataset


class ProcessPoseEstimation:
    def __init__(
        self,
        pose_estimator_model: PoseModel,
        input_data_loader: Union[BoundingBoxesDataLoader, VideoFramesDataLoader],
        output_poses_dataset: PosesDataset,
        device: str = "cpu",
        use_fp_16: bool = False,
        run_distributed: bool = False,
        max_objects_per_inference: int = 100,
    ):
        self.pose_estimator_model: PoseModel = pose_estimator_model
        self.device: str = device
        self.use_fp_16: bool = use_fp_16
        self.run_distributed: bool = run_distributed
        self.max_objects_per_inference: int = max_objects_per_inference

        self.data_loader: Union[
            BoundingBoxesDataLoader, VideoFramesDataLoader
        ] = input_data_loader
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

    def add_data_objects(self, data_objects=None):
        if data_objects is None:
            data_objects = []

        for data_object in data_objects:
            self.data_loader.dataset.add_data_object(data_object)

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
        self.data_loader.dataset.done_loading()

    def _run(self):
        logger.info("Running ProcessPoseEstimation service...")

        pose_estimator = None
        try:
            pose_estimator = inference.PoseEstimator(
                model_config_path=self.pose_estimator_model.model_config,
                checkpoint=self.pose_estimator_model.checkpoint,
                deployment_config_path=self.pose_estimator_model.deployment_config,
                device=self.device,
                max_objects_per_inference=self.max_objects_per_inference,
                use_fp_16=self.use_fp_16,
                run_distributed=self.run_distributed,
            )

            for pose_tuple in pose_estimator.iter_dataloader(loader=self.data_loader):
                self.output_poses_dataset.add_data_object(pose_tuple)
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
        del self.data_loader
