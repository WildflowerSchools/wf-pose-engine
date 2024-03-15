import functools
from multiprocessing import sharedctypes
from typing import Optional, Union

import torch.distributed
import torch.multiprocessing as mp

from pose_db_io.handle.models import pose_2d
from pose_engine.torch import distributed_data_parallel as ddp

from . import inference
from .known_inference_models import PoseModel
from .log import logger
from .loaders.bounding_boxes_dataloader import BoundingBoxesDataLoader
from .loaders.video_frames_dataloader import VideoFramesDataLoader
from .loaders.poses_dataset import PosesDataset


class PoseEstimationInstanceStatus:
    def __init__(self, rank: int):
        self.rank = rank

        self._frame_count: sharedctypes.Synchronized = mp.Value(
            "i", 0
        )  # Each frame contains multiple boxes, so we track frames separately
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
    def frame_count(self) -> int:
        return self._frame_count.value

    @frame_count.setter
    def frame_count(self, value: int):
        self._frame_count.value = value

    @property
    def inference_count(self) -> int:
        return self._inference_count.value

    @inference_count.setter
    def inference_count(self, value: int):
        self._inference_count.value = value

    @property
    def start_time(self) -> float:
        return self._start_time.value

    @start_time.setter
    def start_time(self, value: float):
        self._start_time.value = value

    @property
    def stop_time(self) -> float:
        return self._stop_time.value

    @stop_time.setter
    def stop_time(self, value: float):
        self._stop_time.value = value

    @property
    def running_time_from_start(self) -> float:
        return self._running_time_from_start.value

    @running_time_from_start.setter
    def running_time_from_start(self, value: float):
        self._running_time_from_start.value = value

    @property
    def running_time_from_first_inference(self) -> float:
        return self._running_time_from_first_inference.value

    @running_time_from_first_inference.setter
    def running_time_from_first_inference(self, value: float):
        self._running_time_from_first_inference.value = value

    @property
    def first_inference_time(self) -> float:
        return self._first_inference_time.value

    @first_inference_time.setter
    def first_inference_time(self, value: float):
        self._first_inference_time.value = value

    @property
    def time_waiting(self) -> int:
        return self._time_waiting.value

    @time_waiting.setter
    def time_waiting(self, value: int):
        self._time_waiting.value = value


class ProcessPoseEstimation:
    def __init__(
        self,
        pose_estimator_model: PoseModel,
        input_data_loader: Union[BoundingBoxesDataLoader, VideoFramesDataLoader],
        output_poses_dataset: PosesDataset,
        device: str = "cpu",
        use_fp16: bool = False,
        run_parallel: bool = False,
        run_distributed: bool = False,
        batch_size: int = 100,
        compile_engine: Optional[str] = None,
    ):
        self.pose_estimator_model: PoseModel = pose_estimator_model
        self.device: str = device
        self.use_fp16: bool = use_fp16
        self.run_parallel: bool = run_parallel
        self.run_distributed: bool = run_distributed
        self.batch_size: int = batch_size
        self.compile_engine: bool = compile_engine

        self.data_loader: Union[BoundingBoxesDataLoader, VideoFramesDataLoader] = (
            input_data_loader
        )
        self.output_poses_dataset: PosesDataset = output_poses_dataset

        self.process: Optional[Union[mp.Process, mp.ProcessContext]] = None

        if self.run_distributed:
            self.world_size = torch.cuda.device_count()
        else:
            self.world_size = 1
        self.pose_estimation_instance_statuses: list[PoseEstimationInstanceStatus] = [
            PoseEstimationInstanceStatus(rank=rank) for rank in range(self.world_size)
        ]

    @property
    def total_frame_count(self) -> int:
        return functools.reduce(
            lambda total, instance_status: total + instance_status.frame_count,
            self.pose_estimation_instance_statuses,
            0,
        )

    @property
    def total_inference_count(self) -> int:
        return functools.reduce(
            lambda total, instance_status: total + instance_status.inference_count,
            self.pose_estimation_instance_statuses,
            0,
        )

    @property
    def earliest_processing_start_time(self) -> float:
        return min(
            list(
                map(
                    lambda instance_status: instance_status.start_time,
                    self.pose_estimation_instance_statuses,
                )
            )
        )

    @property
    def latest_processing_stop_time(self) -> float:
        return max(
            list(
                map(
                    lambda instance_status: instance_status.stop_time,
                    self.pose_estimation_instance_statuses,
                )
            )
        )

    @property
    def earliest_first_inference_time(self) -> float:
        return min(
            list(
                map(
                    lambda instance_status: instance_status.first_inference_time,
                    self.pose_estimation_instance_statuses,
                )
            )
        )

    @property
    def total_time_running_from_start(self) -> int:
        return functools.reduce(
            lambda total, instance_status: total
            + instance_status.running_time_from_start,
            self.pose_estimation_instance_statuses,
            0,
        )

    @property
    def total_time_running_from_first_inference(self) -> int:
        return functools.reduce(
            lambda total, instance_status: total
            + instance_status.running_time_from_first_inference,
            self.pose_estimation_instance_statuses,
            0,
        )

    @property
    def total_time_waiting(self) -> int:
        return functools.reduce(
            lambda total, instance_status: total + instance_status.time_waiting,
            self.pose_estimation_instance_statuses,
            0,
        )

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
            if self.run_distributed:
                self.process: mp.ProcessContext = mp.spawn(
                    self._run,
                    args=(self.world_size,),
                    nprocs=self.world_size,
                    daemon=False,
                    join=False,
                )
            else:
                self.process: mp.Process = mp.Process(target=self._run, args=())
                self.process.start()

    def wait(self):
        self.process.join()
        if isinstance(self.process, mp.ProcessContext):
            # pylint: disable=E1101
            exitcodes = list(map(lambda p: p.exitcode, self.process.processes))
        else:
            exitcodes = [self.process.exitcode]
        logger.info(f"ProcessPoseEstimation service finished: {exitcodes}")

    def stop(self):
        if not isinstance(self.process, mp.ProcessContext):
            self.process.close()
        self.process = None

    def mark_detector_drained(self):
        self.data_loader.dataset.done_loading()

    def _run(self, rank=None, world_size=None):
        logger.info("Running ProcessPoseEstimation service...")
        pose_estimator_device: str = self.device
        if self.run_distributed and rank is not None:
            ddp.ddp_setup(rank=rank, world_size=world_size)
            pose_estimator_device = f"cuda:{rank}"
        else:
            rank = 0

        pose_estimator = None
        try:
            pose_estimator = inference.PoseEstimator(
                model_config_path=self.pose_estimator_model.model_config,
                checkpoint=self.pose_estimator_model.checkpoint,
                deployment_config_path=self.pose_estimator_model.deployment_config,
                device=pose_estimator_device,
                batch_size=self.batch_size,
                use_fp16=self.use_fp16,
                run_parallel=self.run_parallel,
                run_distributed=self.run_distributed,
                compile_engine=self.compile_engine,
            )

            for pose_tuple in pose_estimator.iter_dataloader(loader=self.data_loader):
                self.pose_estimation_instance_statuses[rank].frame_count = (
                    pose_estimator.frame_count
                )
                self.pose_estimation_instance_statuses[rank].inference_count = (
                    pose_estimator.inference_count
                )
                self.pose_estimation_instance_statuses[rank].start_time = (
                    pose_estimator.start_time
                )
                self.pose_estimation_instance_statuses[rank].stop_time = (
                    pose_estimator.stop_time
                )
                self.pose_estimation_instance_statuses[rank].running_time_from_start = (
                    pose_estimator.running_time_from_start
                )
                self.pose_estimation_instance_statuses[
                    rank
                ].running_time_from_first_inference = (
                    pose_estimator.running_time_from_first_inference
                )
                self.pose_estimation_instance_statuses[rank].first_inference_time = (
                    pose_estimator.first_inference_time
                )
                self.pose_estimation_instance_statuses[rank].time_waiting = (
                    pose_estimator.time_waiting
                )

                self.output_poses_dataset.add_data_object(pose_tuple)

        except Exception as e:
            logger.error(e)
            raise e
        finally:
            if pose_estimator is not None:
                pose_estimator.stop_pre_processor()
                del pose_estimator

            if self.run_distributed and rank is not None:
                torch.distributed.destroy_process_group()

            logger.info("ProcessPoseEstimation service loop ended")

    def __del__(self):
        del self.data_loader
