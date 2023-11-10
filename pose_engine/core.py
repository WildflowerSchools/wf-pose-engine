from datetime import datetime

# import time

# import torch.multiprocessing as mp

from .known_inference_models import DetectorModel, PoseModel
from .pipeline import Pipeline

# from . import inference
# from .log import logger
# from . import loaders
# from .process_detection import ProcessDetection
# from .process_pose_estimation import ProcessPoseEstimation
# from .process_status_poll import ProcessStatusPoll
# from .process_store_poses import ProcessStorePoses
# from .video import VideoFetch


def run(environment: str, start: datetime, end: datetime):
    detector_model = DetectorModel.rtmdet_medium()
    pose_model = PoseModel.rtmpose_large_384()

    p = Pipeline(
        environment=environment,
        start_datetime=start,
        end_datetime=end,
        detector_model=detector_model,
        detector_device="cuda:0",
        pose_estimator_model=pose_model,
        pose_estimator_device="cuda:1",
    )
    p.run()

    # raw_videos = VideoFetch().fetch(environment=environment, start=start, end=end)

    # mp_manager = mp.Manager()

    # ################################################################
    # # 1. Prepare models
    # ################################################################
    # detector = inference.Detector(
    #     preset_model="medium",
    #     device="cuda:0",
    # )
    # pose_estimator = inference.PoseEstimator(
    #     preset_model="large_384",
    #     device="cuda:1",
    # )

    # ################################################################
    # # 2. Prepare dataloaders
    # ################################################################
    # video_frame_dataset = loaders.VideoFramesDataset(
    #     frame_queue_maxsize=180,
    #     wait_for_video_files=False,
    #     mp_manager=mp_manager
    # )
    # video_frames_loader = loaders.VideoFramesDataLoader(
    #     dataset=video_frame_dataset,
    #     device="cpu",  # This should be "cuda:0", but need to wait until image pre-processing doesn't require moving frames back to CPU
    #     shuffle=False,
    #     num_workers=0,
    #     batch_size=70,
    #     pin_memory=True,
    # )

    # bbox_dataset = loaders.BoundingBoxesDataset(bbox_queue_maxsize=180, wait_for_bboxes=True, mp_manager=mp_manager)
    # bboxes_loader = loaders.BoundingBoxesDataLoader(
    #     dataset=bbox_dataset,
    #     shuffle=False,
    #     num_workers=0,
    #     batch_size=25,
    #     pin_memory=True,
    # )

    # poses_dataset = loaders.PosesDataset(pose_queue_maxsize=180, wait_for_poses=True)
    # poses_loader = loaders.PosesDataLoader(
    #     dataset=poses_dataset,
    #     shuffle=False,
    #     num_workers=0,
    #     batch_size=25,
    #     pin_memory=False,
    # )

    # ################################################################
    # # 3. Prepare model services/processes
    # ################################################################
    # detection_process = ProcessDetection(
    #     detector=detector,
    #     input_video_frames_loader=video_frames_loader,
    #     output_bbox_dataset=bboxes_loader.dataset,
    # )
    # # video_files = [
    # #     "./input/test_video/output000.mp4",
    # #     "./input/test_video/output001.mp4",
    # #     "./input/test_video/output002.mp4",
    # #     "./input/test_video/output003.mp4",
    # #     "./input/test_video/output004.mp4",
    # # ]
    # # detection_process.add_video_files(video_files)
    # detection_process.add_video_files(
    #     files=list(map(lambda v: v["video_local_path"], raw_videos))
    # )

    # pose_estimation_process = ProcessPoseEstimation(
    #     pose_estimator=pose_estimator,
    #     input_bboxes_loader=bboxes_loader,
    #     output_poses_dataset=poses_dataset
    # )

    # store_poses_process = ProcessStorePoses(
    #     input_poses_loader=poses_loader
    # )

    # status_poll_process = StatusPoll(
    #     video_frame_dataset=video_frame_dataset,
    #     bounding_box_dataset=bbox_dataset,
    #     poses_dataset=poses_dataset
    # )

    # ################################################################
    # # 4. Start processing!
    # ################################################################
    # status_poll_process.start()
    # store_poses_process.start()
    # pose_estimation_process.start()
    # detection_process.start()

    # start_timer = time.time()

    # detection_process.wait()
    # pose_estimation_process.mark_detector_drained()
    # pose_estimation_process.wait()

    # total_time = time.time() - start_timer
    # logger.info(
    #     f"Finished running pose estimation ({total_time:.3f} seconds - {(len(raw_videos) * 100) / total_time} fps)"
    # )

    # store_poses_process.mark_poses_drained()
    # store_poses_process.wait()

    # status_poll_process.stop()
