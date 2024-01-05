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
        detection_process: ProcessDetection,
        pose_estimation_process: ProcessPoseEstimation,
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
        current_detector_inference_count = 0
        current_pose_frame_count = 0
        current_pose_inference_count = 0
        last_detector_inference_count = 0
        last_pose_frame_count = 0
        last_pose_inference_count = 0

        while not self.stop_event.is_set():
            if self.detection_process is not None:
                current_detector_inference_count = (
                    self.detection_process.inference_count
                )

            if self.pose_estimation_process is not None:
                current_pose_frame_count = self.pose_estimation_process.frame_count
                current_pose_inference_count = (
                    self.pose_estimation_process.inference_count
                )

            logger.info(
                f"Video frame queue size: {self.video_frame_dataset.size()}/{self.video_frame_dataset.maxsize()}"
            )
            logger.info(
                f"Bounding box queue size: {self.bounding_box_dataset.size()}/{self.bounding_box_dataset.maxsize()}"
            )
            logger.info(
                f"Poses queue size: {self.poses_dataset.size()}/{self.poses_dataset.maxsize()}"
            )
            logger.info(
                f"Overall Detector FPS: {(current_detector_inference_count - last_detector_inference_count) / self.poll}"
            )
            logger.info(
                f"Overall Pose Inference FPS: {(current_pose_frame_count - last_pose_frame_count) / self.poll}"
            )
            logger.info(
                f"Overall Pose Inference BBPS (bounding box per second): {(current_pose_inference_count - last_pose_inference_count) / self.poll}\n"
            )

            last_detector_inference_count = current_detector_inference_count
            last_pose_frame_count = current_pose_frame_count
            last_pose_inference_count = current_pose_inference_count

            time.sleep(self.poll)
