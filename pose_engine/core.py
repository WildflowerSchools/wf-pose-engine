import time

import torch.multiprocessing as mp

from mmpose.apis import init_model as init_pose_estimator

from . import inference
from .log import logger
from .loaders.bounding_boxes_dataloader import BoundingBoxesDataLoader
from .loaders.bounding_boxes_dataset import BoundingBoxesDataset
from .loaders.video_frames_dataloader import VideoFramesDataLoader
from .loaders.video_frames_dataset import VideoFramesDataset
from .process_detection import ProcessDetection
from .process_pose_estimation import ProcessPoseEstimation


def run():
    mp_manager = mp.Manager()

    ################################################################
    # 1. Prepare models
    ################################################################
    detector = inference.Detector(
        config="./configs/mmdet/rtmdet_m_640-8xb32_coco-person.py",
        checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
        device="cuda:0",
    )
    pose_estimator = inference.PoseEstimator(
        config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py",
        checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
        device="cuda:1",
    )

    ################################################################
    # 2. Prepare dataloaders
    ################################################################
    video_frames_loader = VideoFramesDataLoader(
        dataset=VideoFramesDataset(
            frame_queue_maxsize=200, wait_for_video_files=False, mp_manager=mp_manager
        ),
        device="cpu",  # This should be "cuda:0", but need to wait until image pre-processing doesn't require moving frames back to CPU
        shuffle=False,
        num_workers=0,
        batch_size=55,
        pin_memory=True,
    )

    bboxes_loader = BoundingBoxesDataLoader(
        dataset=BoundingBoxesDataset(bbox_queue_maxsize=200, mp_manager=mp_manager),
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
    video_files = [
        "./input/test_video/output000.mp4",
        "./input/test_video/output001.mp4",
        "./input/test_video/output002.mp4",
        "./input/test_video/output003.mp4",
        "./input/test_video/output004.mp4",
        "./input/test_video/output000.mp4",
        "./input/test_video/output001.mp4",
        "./input/test_video/output002.mp4",
        "./input/test_video/output003.mp4",
        "./input/test_video/output004.mp4",
        "./input/test_video/output000.mp4",
        "./input/test_video/output001.mp4",
        "./input/test_video/output002.mp4",
        "./input/test_video/output003.mp4",
        "./input/test_video/output004.mp4",
        "./input/test_video/output000.mp4",
        "./input/test_video/output001.mp4",
        "./input/test_video/output002.mp4",
        "./input/test_video/output003.mp4",
        "./input/test_video/output004.mp4",
    ]
    detection_process.add_video_files(video_files)

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
        f"Finished running pose estimation ({total_time:.3f} seconds - {(len(video_files) * 100) / total_time} fps)"
    )
