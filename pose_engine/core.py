from datetime import datetime
import time

import torch.multiprocessing as mp

from . import inference
from .log import logger
from .loaders.bounding_boxes_dataloader import BoundingBoxesDataLoader
from .loaders.bounding_boxes_dataset import BoundingBoxesDataset
from .loaders.video_frames_dataloader import VideoFramesDataLoader
from .loaders.video_frames_dataset import VideoFramesDataset
from .process_detection import ProcessDetection
from .process_pose_estimation import ProcessPoseEstimation
from .video import VideoFetch


def run(environment: str, start: datetime, end: datetime):
    raw_videos = VideoFetch().fetch(environment=environment, start=start, end=end)

    mp_manager = mp.Manager()

    ################################################################
    # 1. Prepare models
    ################################################################
    detector = inference.Detector(
        preset_model="medium",
        device="cuda:0",
    )
    pose_estimator = inference.PoseEstimator(
        preset_model="medium_384",
        device="cuda:1",
    )

    ################################################################
    # 2. Prepare dataloaders
    ################################################################
    video_frames_loader = VideoFramesDataLoader(
        dataset=VideoFramesDataset(
            frame_queue_maxsize=300, wait_for_video_files=False, mp_manager=mp_manager
        ),
        device="cpu",  # This should be "cuda:0", but need to wait until image pre-processing doesn't require moving frames back to CPU
        shuffle=False,
        num_workers=0,
        batch_size=60,
        pin_memory=True,
    )

    bboxes_loader = BoundingBoxesDataLoader(
        dataset=BoundingBoxesDataset(bbox_queue_maxsize=300, mp_manager=mp_manager),
        shuffle=False,
        num_workers=0,
        batch_size=100,
        pin_memory=True,
    )

    ################################################################
    # 3. Prepare model services/processes
    ################################################################
    detection_process = ProcessDetection(
        detector=detector,
        input_video_frames_loader=video_frames_loader,
        output_bbox_dataset=bboxes_loader.dataset,
    )
    # video_files = [
    #     "./input/test_video/output000.mp4",
    #     "./input/test_video/output001.mp4",
    #     "./input/test_video/output002.mp4",
    #     "./input/test_video/output003.mp4",
    #     "./input/test_video/output004.mp4",
    # ]
    # detection_process.add_video_files(video_files)
    detection_process.add_video_files(
        files=list(map(lambda v: v["video_local_path"], raw_videos))
    )

    pose_estimation_process = ProcessPoseEstimation(
        pose_estimator=pose_estimator, input_bboxes_loader=bboxes_loader
    )

    ################################################################
    # 4. Start processing!
    ################################################################
    pose_estimation_process.start()
    detection_process.start()

    start_timer = time.time()

    detection_process.wait()
    pose_estimation_process.mark_detector_completed()
    pose_estimation_process.wait()

    total_time = time.time() - start_timer
    logger.info(
        f"Finished running pose estimation ({total_time:.3f} seconds - {(len(raw_videos) * 100) / total_time} fps)"
    )
