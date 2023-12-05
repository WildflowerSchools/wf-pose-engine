from datetime import datetime, timedelta
from threading import Thread
import time
import queue

import torch.multiprocessing as mp
import torch.utils.data

from cv_utils import VideoInput

from pose_engine.log import logger


class VideoFramesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        video_objects: list[dict] = None,
        frame_queue_maxsize: int = 120,
        wait_for_video_files: bool = True,
        wait_for_video_frames: bool = True,
        filter_min_datetime: datetime = None,
        filter_max_datetime: datetime = None,
        mp_manager=None,
    ):
        super().__init__()

        if video_objects is None:
            video_objects = []

        self.frame_queue_maxsize = frame_queue_maxsize
        self.wait_for_video_files = wait_for_video_files
        self.wait_for_video_frames = wait_for_video_frames
        self.filter_min_datetime = filter_min_datetime
        self.filter_max_datetime = filter_max_datetime

        self._total_video_files_queued = mp.Value("i", 0)
        self._total_video_frames_queued = mp.Value("i", 0)

        self.video_object_queue = mp.JoinableQueue()
        for v in video_objects:
            self.video_object_queue.put(v)

        if mp_manager is None:
            mp_manager = mp.Manager()
        self.video_frame_queue = mp_manager.Queue(maxsize=frame_queue_maxsize)

        self.video_loader_thread_stopped = False
        self.video_loader_thread = None

    def add_video_object(self, video_object):
        self.video_object_queue.put(video_object)
        self._total_video_files_queued.value += 1

    def size(self):
        return self.video_frame_queue.qsize()

    def maxsize(self):
        return self.frame_queue_maxsize

    def total_video_files_queued(self):
        return self._total_video_files_queued.value

    def total_video_frames_queued(self):
        return self._total_video_frames_queued.value

    def _start_video_loader(self):
        time.sleep(1)  # Hacky, but whatever

        while not self.video_loader_thread_stopped:
            try:
                video_object = self.video_object_queue.get(block=False)
                video_path = video_object["video_local_path"]

                logger.info(f"Loading frames from {video_path}...")
                video_reader = VideoInput(input_path=video_path, queue_frames=False)

                frame_idx = 0
                while True:
                    frame = video_reader.get_frame()
                    if frame is None:
                        logger.info(f"Exhausted all frames in {video_path}")
                        break

                    frame_idx += 1
                    frame_time = video_object["video_timestamp"] + (
                        (frame_idx - 1) * timedelta(seconds=1 / video_object["fps"])
                    )

                    if self.filter_min_datetime is not None:
                        if frame_time < self.filter_min_datetime:
                            logger.warning(
                                f"Skipping frame in {video_path} at frame time ({frame_time}), less than minimum datetime ({self.filter_min_datetime})"
                            )
                            continue

                    if self.filter_max_datetime is not None:
                        if frame_time > self.filter_max_datetime:
                            logger.warning(
                                f"Skipping frame in {video_path} at frame time ({frame_time}) greater than maximum datetime ({self.filter_max_datetime})"
                            )
                            continue

                    frame_written = False
                    while not frame_written and not self.video_loader_thread_stopped:
                        try:
                            logger.debug(
                                f"Putting '{video_path}' frame index {frame_idx} on frame queue"
                            )

                            self.video_frame_queue.put(
                                (
                                    frame,
                                    {
                                        "video_path": video_path,
                                        "frame_index": frame_idx,
                                        "frame_timestamp": frame_time.timestamp(),
                                        "camera_device_id": video_object["device_id"],
                                    },
                                ),
                                timeout=0.5,
                            )
                            self._total_video_frames_queued.value += 1
                            frame_written = True
                        except queue.Full:
                            logger.debug(
                                f"Reached frame queue max size '{self.frame_queue_maxsize}', waiting to load additional frames..."
                            )

                    logger.debug(
                        "Frame written or video_loader_thread_stopped, reading next frame"
                    )
                self.video_object_queue.task_done()

                logger.info(f"Finished loading frames from {video_path}")
            except queue.Empty:
                if not self.wait_for_video_files:
                    logger.info(
                        "Video file queue empty and wait_for_video_files set to False, killing video loader"
                    )
                    self.video_loader_thread_stopped = True

                logger.debug("Video file queue empty, sleeping for 1 second")
                time.sleep(1)

    def __getitem__(self, _idx):
        raise NotImplementedError(
            "Attempted to use __getitem__, but VideoFramesDataset is an IterableDataset so __getitem__ is intentionally not implemented"
        )

    def __iter__(self):
        self.video_loader_thread = Thread(target=self._start_video_loader, args=())
        self.video_loader_thread.daemon = True
        self.video_loader_thread.start()

        while True:
            try:
                frame, meta = self.video_frame_queue.get(block=False, timeout=0.5)
                if frame is not None:
                    yield (frame, meta)
            except queue.Empty:
                # DO NOT REMOVE: the "qsize()" assertion, this is important as the queue.Empty exception doesn't necessarily mean the queue is empty
                if self.video_frame_queue.qsize() == 0:
                    if not self.wait_for_video_frames:
                        logger.info(
                            "Nothing to read from frame queue, terminating iterator"
                        )
                        break

                    if self.video_loader_thread_stopped:
                        logger.info(
                            "Video loader is terminated and no more items in the frame queue, terminating iterator"
                        )
                        break

                # logger.debug(f"Nothing to read from frame queue, waiting...")

        logger.debug("Video frames dataset iter finished")

    def stop_video_loader(self):
        if self.video_loader_thread is not None:
            self.video_loader_thread_stopped = True
            self.video_loader_thread.join()
            self.video_loader_thread

    def cleanup(self):
        try:
            while True:
                item = self.video_object_queue.get_nowait()
                del item
        except queue.Empty:
            pass
        finally:
            del self.video_object_queue

        try:
            while True:
                item = self.video_frame_queue.get_nowait()
                del item
        except queue.Empty:
            pass
        finally:
            del self.video_frame_queue

        self.stop_video_loader()
