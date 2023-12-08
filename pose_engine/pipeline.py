from datetime import datetime
import time
from typing import Optional
import uuid
from zoneinfo import ZoneInfo

import torch.multiprocessing as mp

import honeycomb_io
from pose_db_io.handle.models.pose_2d import Pose2dMetadataCommon

from . import inference
from .known_inference_models import DetectorModel, PoseModel
from .log import logger
from . import loaders
from .process_detection import ProcessDetection
from .process_pose_estimation import ProcessPoseEstimation
from .process_status_poll import ProcessStatusPoll
from .process_store_poses import ProcessStorePoses
from .video import VideoFetch


class Pipeline:
    def __init__(
        self,
        detector_model: DetectorModel,
        pose_estimator_model: PoseModel,
        mp_manager: mp.Manager,
        detector_device: str = "cpu",
        pose_estimator_device: str = "cpu",
        use_fp_16: bool = False,
    ):
        self.detector_model: DetectorModel = detector_model
        self.pose_estimator_model: PoseModel = pose_estimator_model
        self.detector_device: str = detector_device
        self.pose_estimator_device: str = pose_estimator_device
        self.use_fp_16 = use_fp_16

        self.mp_manager: mp.Manager = mp_manager

        self.environment_id: Optional[str] = None
        self.environment_name: Optional[str] = None
        self.environment_timezone: Optional[ZoneInfo] = None

        self.current_run_start_time: Optional[datetime] = None
        self.current_run_inference_id: Optional[uuid.UUID] = None

        self._init_models()

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

    def _init_models(self):
        self.detector = inference.Detector(
            config=self.detector_model.config,
            checkpoint=self.detector_model.checkpoint,
            device=self.detector_device,
            use_fp_16=self.use_fp_16,
        )

        self.pose_estimator = inference.PoseEstimator(
            config=self.pose_estimator_model.config,
            checkpoint=self.pose_estimator_model.checkpoint,
            device=self.pose_estimator_device,
            max_boxes_per_inference=70,
            use_fp_16=self.use_fp_16,
        )

    def _init_dataloaders(self):
        self.video_frame_dataset = loaders.VideoFramesDataset(
            frame_queue_maxsize=180,
            wait_for_video_files=False,
            mp_manager=self.mp_manager,
            filter_min_datetime=self.start_datetime,
            filter_max_datetime=self.end_datetime,
        )
        self.video_frames_loader = loaders.VideoFramesDataLoader(
            dataset=self.video_frame_dataset,
            device="cpu",  # This should be "cuda:0", but need to wait until image pre-processing doesn't require moving frames back to CPU
            shuffle=False,
            num_workers=0,
            batch_size=60,
            pin_memory=True,
        )

        self.bbox_dataset = loaders.BoundingBoxesDataset(
            bbox_queue_maxsize=150, wait_for_bboxes=True, mp_manager=self.mp_manager
        )
        self.bboxes_loader = loaders.BoundingBoxesDataLoader(
            dataset=self.bbox_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=60,
            pin_memory=True,
        )

        self.poses_dataset = loaders.PosesDataset(
            pose_queue_maxsize=200, wait_for_poses=True, mp_manager=self.mp_manager
        )
        self.poses_loader = loaders.PosesDataLoader(
            dataset=self.poses_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=100,
            pin_memory=False,
        )

    def _init_processes(self):
        self.detection_process = ProcessDetection(
            detector=self.detector,
            input_video_frames_loader=self.video_frames_loader,
            output_bbox_dataset=self.bbox_dataset,
        )

        self.pose_estimation_process = ProcessPoseEstimation(
            pose_estimator=self.pose_estimator,
            input_bboxes_loader=self.bboxes_loader,
            output_poses_dataset=self.poses_dataset,
        )

        self.store_poses_process = ProcessStorePoses(
            input_poses_loader=self.poses_loader
        )

        self.status_poll_process = ProcessStatusPoll(
            video_frame_dataset=self.video_frame_dataset,
            bounding_box_dataset=self.bbox_dataset,
            poses_dataset=self.poses_dataset,
            detector=self.detector,
            pose_estimator=self.pose_estimator,
        )

    def _fetch_videos(self):
        # TODO: Make video fetching and loading onto the pipeline asynch
        raw_videos = VideoFetch().fetch(
            environment=self.environment_id,
            start=self.start_datetime,
            end=self.end_datetime,
        )

        self.detection_process.add_video_objects(video_objects=raw_videos)

    def start(self):
        common_metadata = Pose2dMetadataCommon(
            inference_run_id=uuid.uuid4(),
            inference_run_created_at=datetime.utcnow(),
            environment_id=self.environment_id,
            classroom_date=self.start_datetime.astimezone(
                self.environment_timezone
            ).strftime("%Y-%m-%d"),
            bounding_box_format=self.detector_model.bounding_box_format_enum,
            detection_model_config=self.detector_model.config_enum,
            detection_model_checkpoint=self.detector_model.checkpoint_enum,
            keypoints_format=self.pose_estimator_model.keypoint_format_enum,
            pose_model_config=self.pose_estimator_model.config_enum,
            pose_model_checkpoint=self.pose_estimator_model.checkpoint_enum,
        )

        self.status_poll_process.start()

        self.store_poses_process.start(common_metadata=common_metadata)
        self.pose_estimation_process.start()
        self.detection_process.start()

        self.current_run_start_time = time.time()

    def wait(self):
        self.detection_process.wait()

        self.pose_estimation_process.mark_detector_drained()
        self.pose_estimation_process.wait()

        total_time = time.time() - self.current_run_start_time
        logger.info(
            f"Finished running pose estimation ({total_time:.3f} seconds - {(self.video_frames_loader.total_video_files_queued() * 100) / total_time} fps)"
        )

        self.store_poses_process.mark_poses_drained()
        self.store_poses_process.wait()

    def cleanup(self):
        self.status_poll_process.stop()
        self.store_poses_process.stop()
        self.pose_estimation_process.stop()
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
        del self.status_poll_process
        del self.store_poses_process
        del self.pose_estimation_process
        del self.detection_process

        del self.pose_estimator
        del self.detector
