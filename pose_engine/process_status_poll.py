import multiprocessing as mp
from threading import Thread
import time
from typing import Optional

from pose_engine import inference
from pose_engine import loaders
from pose_engine.process_detection import ProcessDetection
from pose_engine.process_pose_estimation import ProcessPoseEstimation
from pose_engine.log import logger


class ProcessStatusPoll:
    def __init__(
        self,
        video_frame_dataset: loaders.VideoFramesDataset,
        bounding_box_dataset: loaders.BoundingBoxesDataset,
        poses_dataset: loaders.PosesDataset,
        # detector: inference.Detector,
        # pose_estimator: inference.PoseEstimator,
        pose_estimation_process: ProcessPoseEstimation,
        detection_process: Optional[ProcessDetection] = None,
        poll: int = 10,
    ):
        self.video_frame_dataset: loaders.VideoFramesDataset = video_frame_dataset
        self.bounding_box_dataset: loaders.BoundingBoxesDataset = bounding_box_dataset
        self.poses_dataset: loaders.PosesDataset = poses_dataset
        # self.detector = detector
        # self.pose_estimator = pose_estimator
        self.detection_process: ProcessDetection = detection_process
        self.pose_estimation_process: ProcessPoseEstimation = pose_estimation_process
        self.poll: int = poll

        self.stop_event: mp.Event = mp.Event()

        self.polling_thread: Optional[Thread] = None

    def start(self):
        if self.polling_thread is not None:
            return

        self.stop_event.clear()

        self.polling_thread = Thread(target=self._run, args=())
        self.polling_thread.daemon = True
        self.polling_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.polling_thread is not None:
            self.polling_thread.join()

            self.polling_thread = None
            del self.polling_thread

    def _run(self):
        start_run_time = time.time()
        start_detector_inference_time = None
        start_pose_inference_time = None

        current_detector_inference_count = 0
        current_pose_frame_count = 0
        current_pose_inference_count = 0
        last_detector_inference_count = 0
        last_pose_frame_count = 0
        last_pose_inference_count = 0

        while True:
            current_run_time = time.time()
            seconds_running = round((current_run_time - start_run_time), 2)

            if seconds_running == 0:
                continue

            if self.detection_process is not None:
                current_detector_inference_count = (
                    self.detection_process.inference_count
                )

            if self.pose_estimation_process is not None:
                current_pose_frame_count = self.pose_estimation_process.frame_count
                current_pose_inference_count = (
                    self.pose_estimation_process.inference_count
                )

            seconds_detector_running = 0
            if current_detector_inference_count > 0:
                if start_detector_inference_time is None:
                    start_detector_inference_time = time.time()

                seconds_detector_running = round(
                    (current_run_time - start_detector_inference_time), 2
                )

            seconds_pose_inference_running = 0
            if current_pose_frame_count > 0:
                if start_pose_inference_time is None:
                    start_pose_inference_time = time.time()

                seconds_pose_inference_running = round(
                    (current_run_time - start_pose_inference_time), 2
                )

            logger.info(
                f"Status: Running time: {seconds_running} seconds ({round(seconds_running / 60 / 60, 2)} hours)"
            )
            logger.info(
                f"Status: Video frame queue size: {self.video_frame_dataset.size()}/{self.video_frame_dataset.maxsize()}"
            )
            if self.detection_process is not None:
                logger.info(
                    f"Status: Bounding box queue size: {self.bounding_box_dataset.size()}/{self.bounding_box_dataset.maxsize()}"
                )
            if self.pose_estimation_process is not None:
                logger.info(
                    f"Status: Poses queue size: {self.poses_dataset.size()}/{self.poses_dataset.maxsize()}"
                )
            if self.detection_process is not None:
                frames_last_x = (
                    current_detector_inference_count - last_detector_inference_count
                )
                fps_last_x = round(
                    (current_detector_inference_count - last_detector_inference_count)
                    / self.poll,
                    2,
                )
                logger.info(
                    f"Status: Detector (last {self.poll} seconds): {fps_last_x} FPS {frames_last_x} frames"
                )
                if seconds_detector_running > 0:
                    frames_overall = current_detector_inference_count
                    fps_overall = round(frames_overall / seconds_detector_running, 2)
                    logger.info(
                        f"Status: Detector (overall - {seconds_detector_running} seconds): {fps_overall} FPS {frames_overall} frames"
                    )
            if self.pose_estimation_process is not None:
                frames_last_x = current_pose_frame_count - last_pose_frame_count
                bb_last_x = current_pose_inference_count - last_pose_inference_count
                fps_last_x = round(frames_last_x / self.poll, 2)
                bbps_last_x = round(bb_last_x / self.poll, 2)
                logger.info(
                    f"Status: Pose Inference (last {self.poll} seconds): {fps_last_x} FPS {bbps_last_x} BBPS (bounding boxes per second) {frames_last_x} frames {bb_last_x} bounding boxes"
                )
                if seconds_pose_inference_running > 0:
                    frames_overall = current_pose_frame_count
                    bb_overall = current_pose_inference_count
                    fps_overall = round(
                        frames_overall / seconds_pose_inference_running, 2
                    )
                    bbps_overall = round(bb_overall / seconds_pose_inference_running, 2)

                    logger.info(
                        f"Status: Pose Inference (overall - {seconds_pose_inference_running} seconds): {fps_overall} FPS {bbps_overall} BBPS (bounding boxes per second) {frames_overall} frames {bb_overall} bounding boxes"
                    )

            last_detector_inference_count = current_detector_inference_count
            last_pose_frame_count = current_pose_frame_count
            last_pose_inference_count = current_pose_inference_count

            if self.stop_event.is_set():
                break

            time.sleep(self.poll)
