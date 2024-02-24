from datetime import datetime
import time
from typing import Optional
import uuid
from zoneinfo import ZoneInfo

import torch.multiprocessing as mp

import honeycomb_io
from pose_db_io.handle.models import pose_2d

from .known_inference_models import DetectorModel, PoseModel
from . import loaders
from .process_detection import ProcessDetection
from .process_pose_estimation import ProcessPoseEstimation
from .process_status_poll import ProcessStatusPoll
from .process_store_poses import ProcessStorePoses
from .video import VideoFetch

DEFAULT_DETECTOR_BATCH_SIZE = 100
DEFAULT_POSE_ESTIMATOR_BATCH_SIZE = 350


class Pipeline:
    def __init__(
        self,
        detector_model: Optional[DetectorModel],
        pose_estimator_model: PoseModel,
        mp_manager: Optional[mp.Manager] = None,
        detector_device: str = "cpu",
        pose_estimator_device: str = "cpu",
        use_fp16: bool = False,
        run_parallel: bool = False,
        run_distributed: bool = False,
        compile_models: bool = True,
        detector_batch_size: Optional[int] = DEFAULT_DETECTOR_BATCH_SIZE,
        pose_estimator_batch_size: Optional[int] = DEFAULT_POSE_ESTIMATOR_BATCH_SIZE,
    ):
        if (
            detector_model is None
            and pose_estimator_model.pose_estimator_type
            == pose_2d.PoseEstimatorType.top_down
        ):
            raise ValueError(
                "Pipeline object requires a DetectorModel to be supplied when the PoseModel is a top down model type"
            )

        self.detector_model: DetectorModel = detector_model
        self.pose_estimator_model: PoseModel = pose_estimator_model
        self.detector_device: str = detector_device
        self.pose_estimator_device: str = pose_estimator_device
        self.use_fp16: bool = use_fp16
        self.run_distributed: bool = run_distributed
        self.compile_models: bool = compile_models

        if detector_batch_size is None:
            self.detector_batch_size = DEFAULT_DETECTOR_BATCH_SIZE
        else:
            self.detector_batch_size = detector_batch_size

        if pose_estimator_batch_size is None:
            self.pose_estimator_batch_size = DEFAULT_POSE_ESTIMATOR_BATCH_SIZE
        else:
            self.pose_estimator_batch_size = pose_estimator_batch_size

        if (
            self.pose_estimator_model.pose_estimator_type
            == pose_2d.PoseEstimatorType.top_down
        ) or self.run_distributed:
            # Don't allow models to run parallel when using topdown or the model is being run distributed
            self.run_parallel = False
        else:
            self.run_parallel = run_parallel

        self.mp_manager: mp.Manager = mp_manager
        if self.mp_manager is None:
            self.mp_manager = mp.Manager()

        self.environment: Optional[str] = None
        self.environment_id: Optional[str] = None
        self.environment_name: Optional[str] = None
        self.environment_timezone: Optional[ZoneInfo] = None

        self.start_datetime: Optional[datetime] = None
        self.end_datetime: Optional[datetime] = None

        self.current_run_start_time: Optional[datetime] = None
        self.current_run_inference_id: Optional[uuid.UUID] = None

        self.video_frame_dataset: Optional[loaders.VideoFramesDataset] = None
        self.video_frames_loader: Optional[loaders.VideoFramesDataLoader] = None
        self.bbox_dataset: Optional[loaders.BoundingBoxesDataset] = None
        self.bboxes_loader: Optional[loaders.BoundingBoxesDataLoader] = None
        self.poses_dataset: Optional[loaders.PosesDataset] = None
        self.poses_loader: Optional[loaders.PosesDataLoader] = None

        self.detection_process: Optional[ProcessDetection] = None
        self.pose_estimation_process: Optional[ProcessPoseEstimation] = None
        self.store_poses_process: Optional[ProcessStorePoses] = None
        self.status_poll_process: Optional[ProcessStatusPoll] = None

    def _load_environment(self):
        df_all_environments = honeycomb_io.fetch_all_environments(
            output_format="dataframe"
        )

        df_all_environments = df_all_environments.reset_index()
        df_environment = df_all_environments[
            (df_all_environments["environment_id"] == self.environment)
            | (df_all_environments["environment_name"] == self.environment)
        ]

        if len(df_environment) == 0:
            raise ValueError(
                f"Unable to determing environment from '{self.environment}'"
            )

        if len(df_environment) > 1:
            raise ValueError(f"Multiple environments match '{self.environment}'")

        self.environment_id = df_environment.iloc[0]["environment_id"]
        self.environment_name = df_environment.iloc[0]["environment_name"]
        self.environment_timezone = ZoneInfo(
            df_environment.iloc[0]["environment_timezone_name"]
        )

    def _init_dataloaders(self):
        # Top-down model's detector is responsible for handling images
        # Bottom-up and one-stage model's pose estimator is responsible for handling images
        if (
            self.pose_estimator_model.pose_estimator_type
            == pose_2d.PoseEstimatorType.top_down
        ):
            video_frame_dataloader_batch_size = self.detector_batch_size
        else:
            video_frame_dataloader_batch_size = self.pose_estimator_batch_size

        bouding_box_dataloader_batch_size = self.pose_estimator_batch_size

        self.video_frame_dataset = loaders.VideoFramesDataset(
            frame_queue_maxsize=1000,
            wait_for_video_files=True,
            mp_manager=self.mp_manager,
            filter_min_datetime=self.start_datetime,
            filter_max_datetime=self.end_datetime,
        )

        self.video_frames_loader = loaders.VideoFramesDataLoader(
            dataset=self.video_frame_dataset,
            device="cpu",  # This should be "cuda:0", but need to wait until image pre-processing doesn't require moving frames back to CPU
            shuffle=False,
            num_workers=2,
            batch_size=video_frame_dataloader_batch_size,
            pin_memory=True,
        )

        self.bbox_dataset = loaders.BoundingBoxesDataset(
            bbox_queue_maxsize=600, wait_for_bboxes=True, mp_manager=self.mp_manager
        )
        self.bboxes_loader = loaders.BoundingBoxesDataLoader(
            dataset=self.bbox_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=bouding_box_dataloader_batch_size,
            pin_memory=True,
        )

        self.poses_dataset = loaders.PosesDataset(
            pose_queue_maxsize=1000, wait_for_poses=True, mp_manager=self.mp_manager
        )
        self.poses_loader = loaders.PosesDataLoader(
            dataset=self.poses_dataset,
            shuffle=False,
            num_workers=1,
            batch_size=200,
            pin_memory=False,
        )

    def _init_processes(self):
        self.detection_process = None

        # When using topdown models, feed video frames to the detector
        # But when using bottomup or onestage, feed video frames to the estimator
        if (
            self.pose_estimator_model.pose_estimator_type
            == pose_2d.PoseEstimatorType.top_down
        ):
            pose_estimator_data_loader = self.bboxes_loader
            self.detection_process = ProcessDetection(
                detector_model=self.detector_model,
                input_video_frames_loader=self.video_frames_loader,
                output_bbox_dataset=self.bbox_dataset,
                device=self.detector_device,
                use_fp16=self.use_fp16,
                batch_size=self.detector_batch_size,
                compile_model=self.compile_models,
            )
        else:
            pose_estimator_data_loader = self.video_frames_loader

        self.pose_estimation_process = ProcessPoseEstimation(
            pose_estimator_model=self.pose_estimator_model,
            input_data_loader=pose_estimator_data_loader,
            output_poses_dataset=self.poses_dataset,
            device=self.pose_estimator_device,
            use_fp16=self.use_fp16,
            run_parallel=self.run_parallel,
            run_distributed=self.run_distributed,
            batch_size=self.pose_estimator_batch_size,
            compile_model=self.compile_models,
        )

        self.store_poses_process = ProcessStorePoses(
            input_poses_loader=self.poses_loader
        )

        self.status_poll_process = ProcessStatusPoll(
            video_frame_dataset=self.video_frame_dataset,
            bounding_box_dataset=self.bbox_dataset,
            poses_dataset=self.poses_dataset,
            detection_process=self.detection_process,
            pose_estimation_process=self.pose_estimation_process,
        )

    def _fetch_videos(self):
        # TODO: Make video fetching and loading onto the pipeline asynch
        raw_videos = VideoFetch().fetch(
            environment=self.environment_id,
            start=self.start_datetime,
            end=self.end_datetime,
        )

        if raw_videos is None:
            raw_videos = []

        for raw_video in raw_videos:
            self.video_frames_loader.dataset.add_data_object(raw_video)

        # Load the frames
        # self.video_frames_loader.dataset.start_video_loader()
        # for _ in self.video_frames_loader:

    def start(self):
        bounding_box_format_enum = (
            self.detector_model.bounding_box_format_enum
            if self.detector_model
            else self.pose_estimator_model.bounding_box_format_enum
        )

        common_metadata = pose_2d.Pose2dMetadataCommon(
            inference_run_id=uuid.uuid4(),
            inference_run_created_at=datetime.utcnow(),
            environment_id=self.environment_id,
            classroom_date=self.start_datetime.astimezone(
                self.environment_timezone
            ).strftime("%Y-%m-%d"),
            bounding_box_format=bounding_box_format_enum,
            detection_model_config=(
                self.detector_model.model_config_enum if self.detector_model else None
            ),
            detection_model_checkpoint=(
                self.detector_model.checkpoint_enum if self.detector_model else None
            ),
            detection_model_deployment_config=(
                self.detector_model.deployment_config_enum
                if self.detector_model
                else None
            ),
            keypoints_format=self.pose_estimator_model.keypoint_format_enum,
            pose_model_config=self.pose_estimator_model.model_config_enum,
            pose_model_checkpoint=self.pose_estimator_model.checkpoint_enum,
            pose_model_deployment_config=self.pose_estimator_model.deployment_config_enum,
        )

        self.store_poses_process.start(common_metadata=common_metadata)
        self.pose_estimation_process.start()

        if self.detection_process is not None:
            self.detection_process.start()

        self.status_poll_process.start()

        self.current_run_inference_id = uuid.uuid4()
        self.current_run_start_time = time.time()

    def wait(self):
        if self.detection_process is not None:
            self.detection_process.wait()
            self.pose_estimation_process.mark_detector_drained()

        self.pose_estimation_process.wait()

        self.store_poses_process.mark_poses_drained()
        self.store_poses_process.wait()

    def cleanup(self):
        self.status_poll_process.stop()
        self.store_poses_process.stop()
        self.pose_estimation_process.stop()
        if self.detection_process is not None:
            self.detection_process.stop()

        self.video_frame_dataset.cleanup()
        self.bbox_dataset.cleanup()
        self.poses_dataset.cleanup()

    def run(self, environment: str, start_datetime: datetime, end_datetime: datetime):
        self.environment: str = environment
        self.start_datetime: datetime = start_datetime
        self.end_datetime: datetime = end_datetime

        self._load_environment()
        self._init_dataloaders()
        self._init_processes()

        self._fetch_videos()
        self.start()
        self.wait()
        self.cleanup()

    def __del__(self):
        if self.status_poll_process is not None:
            del self.status_poll_process
        if self.store_poses_process is not None:
            del self.store_poses_process
        if self.pose_estimation_process is not None:
            del self.pose_estimation_process
        if self.detection_process is not None:
            del self.detection_process
