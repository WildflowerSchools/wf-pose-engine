import concurrent.futures
from ctypes import c_bool
import queue
from typing import Optional, Union

import torch.multiprocessing as mp

from pose_db_io.handle.models import pose_2d

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
        run_parallel: bool = False,
        distributed_rank: Optional[int] = None,
        max_objects_per_inference: int = 100,
    ):
        self.pose_estimator_model: PoseModel = pose_estimator_model
        self.device: str = device
        self.use_fp_16: bool = use_fp_16
        self.run_parallel: bool = run_parallel
        self.distributed_rank: int = distributed_rank
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
        self._processing_start_time = mp.Value("d", -1.0)
        self._first_inference_time = mp.Value("d", -1.0)
        self._time_waiting = mp.Value("d", 0)
        self._dataloader_exhausted = mp.Value(c_bool, False)

    @property
    def frame_count(self) -> int:
        return self._frame_count.value

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

    @property
    def dataloader_exhausted(self) -> bool:
        return self._dataloader_exhausted.value

    def is_topdown(self):
        return (
            self.pose_estimator_model.pose_estimator_type
            == pose_2d.PoseEstimatorType.top_down
        )

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
                run_parallel=self.run_parallel,
                distributed_rank=self.distributed_rank,
            )

            for pose_tuple in pose_estimator.iter_dataloader(loader=self.data_loader):
                self._frame_count.value = pose_estimator.frame_count
                self._inference_count.value = pose_estimator.inference_count
                self._processing_start_time.value = pose_estimator.processing_start_time
                self._first_inference_time.value = pose_estimator.first_inference_time
                self._time_waiting.value = pose_estimator.time_waiting

                self.output_poses_dataset.add_data_object(pose_tuple)

        except Exception as e:
            logger.error(e)
            raise e
        finally:
            if pose_estimator is not None:
                del pose_estimator
            logger.info("ProcessPoseEstimation service loop ended")

    def __del__(self):
        del self.data_loader
