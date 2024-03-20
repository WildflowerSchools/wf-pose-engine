from ctypes import c_bool
from datetime import datetime, timedelta
from multiprocessing import sharedctypes
from threading import Thread
import time
import queue

import torch.distributed
import torch.multiprocessing as mp
import torch.utils.data

from cv_utils import VideoInput
from faster_fifo import Queue as ffQueue

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

        self._queue_wait_time: sharedctypes.Synchronized = mp.Value("d", 0)

        self._total_video_files_queued: sharedctypes.Synchronized = mp.Value("i", 0)
        self._total_video_files_completed: sharedctypes.Synchronized = mp.Value("i", 0)
        self._total_video_frames_queued: sharedctypes.Synchronized = mp.Value("i", 0)

        self.video_object_queue = mp.JoinableQueue()
        for v in video_objects:
            self.video_object_queue.put(v)

        # if mp_manager is None:
        #     mp_manager = mp.Manager()

        self.video_frame_queue = ffQueue(
            max_size_bytes=3 * 640 * 480 * frame_queue_maxsize
        )

        self._video_loader_thread_stopped: sharedctypes.Synchronized = mp.Value(
            c_bool, False
        )
        self._video_loader_thread_initialized: sharedctypes.Synchronized = mp.Value(
            c_bool, False
        )
        self.video_loader_thread: Thread = None

        self._video_loader_start_time: sharedctypes.Synchronized = mp.Value("d", -1.0)
        self._video_loader_stop_time: sharedctypes.Synchronized = mp.Value("d", -1.0)
        self._video_loader_first_frame_read_time: sharedctypes.Synchronized = mp.Value(
            "d", -1.0
        )

    def add_data_object(self, data_object):
        self.video_object_queue.put(data_object)
        with self._total_video_files_queued.get_lock():
            self._total_video_files_queued.value += 1

    def size(self):
        return self.video_frame_queue.qsize()

    def maxsize(self):
        return self.frame_queue_maxsize

    @property
    def video_loader_start_time(self) -> float:
        return self._video_loader_start_time.value

    @property
    def video_loader_stop_time(self) -> float:
        return self._video_loader_stop_time.value

    @property
    def first_frame_read_time(self) -> float:
        return self._video_loader_first_frame_read_time.value

    @property
    def video_loader_running_time_from_start(self) -> float:
        if self.video_loader_start_time == -1:
            return 0

        current_time_or_stop_time = time.time()
        if self.video_loader_stop_time > -1.0:
            current_time_or_stop_time = self.video_loader_stop_time

        return current_time_or_stop_time - self.video_loader_start_time

    @property
    def video_loader_running_time_from_first_frame_read(self) -> float:
        if self.first_frame_read_time == -1:
            return 0

        current_time_or_stop_time = time.time()
        if self.video_loader_stop_time > -1.0:
            current_time_or_stop_time = self.video_loader_stop_time

        return current_time_or_stop_time - self.first_frame_read_time

    @property
    def queue_wait_time(self):
        return self._queue_wait_time.value

    @property
    def total_video_files_queued(self):
        return self._total_video_files_queued.value

    @property
    def total_video_files_completed(self):
        return self._total_video_files_completed.value

    @property
    def total_video_frames_queued(self):
        return self._total_video_frames_queued.value

    @property
    def video_loader_thread_stopped(self):
        return self._video_loader_thread_stopped.value

    @property
    def video_loader_thread_initialized(self):
        return self._video_loader_thread_initialized.value

    def _start_video_loader(self):
        logger.info("Starting the video loader thread...")

        with self._video_loader_start_time.get_lock():
            if self._video_loader_start_time.value == -1.0:
                self._video_loader_start_time.value = time.time()

        # time.sleep(1)  # Hacky, but whatever

        # ii = 0

        while not self.video_loader_thread_stopped:
            # ii += 1

            # if ii > 60:  # 60 loops/videos is equal to 10 minutes of video
            #     self._video_loader_thread_stopped.value = True

            try:
                video_object = self.video_object_queue.get(block=False)
                video_path = video_object["video_local_path"]

                logger.info(
                    f"Loading frames from {video_path}... (Queue size: {self.size()}, Total queued: {self.total_video_frames_queued})"
                )
                video_reader = VideoInput(
                    input_path=video_path, queue_frames=True, queue_size=128
                )

                frame_idx = 0
                while True:
                    frame = video_reader.get_frame()
                    if frame is None:
                        logger.info(
                            f"Exhausted all frames in {video_path} (Queue size: {self.size()}, Total queued: {self.total_video_frames_queued})"
                        )
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
                    while not frame_written and (
                        not self.video_loader_thread_stopped
                        or self.wait_for_video_frames
                    ):
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

                            with self._total_video_frames_queued.get_lock():
                                self._total_video_frames_queued.value += 1

                            with self._video_loader_first_frame_read_time.get_lock():
                                if (
                                    self._video_loader_first_frame_read_time.value
                                    == -1.0
                                ):
                                    self._video_loader_first_frame_read_time.value = (
                                        time.time()
                                    )

                            frame_written = True
                        except queue.Full:
                            logger.debug(
                                f"Reached frame queue max size '{self.frame_queue_maxsize}', waiting to load additional frames..."
                            )

                    logger.debug(
                        "Frame written or video_loader_thread_stopped, reading next frame"
                    )
                self.video_object_queue.task_done()

                logger.info(
                    f"Finished loading frames from {video_path} - Frames loaded {frame_idx}"
                )
                with self._total_video_files_completed.get_lock():
                    self._total_video_files_completed.value += 1
            except queue.Empty:
                if not self.wait_for_video_files:
                    logger.info(
                        "Video file queue empty and wait_for_video_files set to False. Killing video loader thread"
                    )
                    self._video_loader_thread_stopped.value = True

                if (
                    self.wait_for_video_files
                    and self._total_video_files_completed.value
                    == self._total_video_files_queued.value
                ):
                    logger.info(
                        "Video file queue empty, wait_for_video_files set to True, and all queued video files have been processed. Killing video loader thread"
                    )
                    self._video_loader_thread_stopped.value = True

                logger.debug("Video file queue empty, sleeping for 1 second")
                time.sleep(1)

        self._video_loader_stop_time.value = time.time()

    def __getitem__(self, _idx):
        raise NotImplementedError(
            "Attempted to use __getitem__, but VideoFramesDataset is an IterableDataset so __getitem__ is intentionally not implemented"
        )

    def start_video_loader(self):
        # if self.video_loader_thread is None and not self.video_loader_thread_initialized:
        self.video_loader_thread = Thread(target=self._start_video_loader, args=())
        self.video_loader_thread.daemon = True
        self.video_loader_thread.start()

        # video_frame_loader_thread = Thread(target=self._start_video_loader, args=())
        # video_frame_loader_thread.daemon = True
        # video_frame_loader_thread.start()

        # self.video_frame_loader_processes.append(video_frame_loader_thread)

        # self._video_loader_thread_initialized.value = True

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # pass

        # self.video_loader_lock.acquire()
        # if not self.video_loader_thread_initialized:

        self.start_video_loader()
        # self.video_loader_thread = mp.Process(target=self._start_video_loader, args=())
        # self.video_loader_thread.daemon = False
        # self.video_loader_thread.start()

        #
        # self.video_loader_lock.release()

        while True:
            start_wait = time.time()

            try:
                frame, meta = self.video_frame_queue.get(block=False, timeout=0.5)
                if frame is not None:
                    yield (frame, meta)
            except queue.Empty:
                # logger.info("Video frames queue empty")

                end_wait = time.time() - start_wait
                with self._queue_wait_time.get_lock():
                    self._queue_wait_time.value += end_wait

                # DO NOT REMOVE: the "qsize()" assertion, this is important as the queue.Empty exception doesn't necessarily mean the queue is empty
                if self.video_frame_queue.qsize() == 0:
                    if not self.wait_for_video_frames:
                        logger.info(
                            "Nothing to read from frame queue and wait_for_video_frames set to False, terminating video frame iterator"
                        )
                        break

                    if self.video_loader_thread_stopped:
                        logger.info(
                            "Video loader thread killed. Terminating video frame iterator"
                        )
                        break

                # logger.debug(f"Nothing to read from frame queue, waiting...")

        logger.debug("Video frames dataset iter finished")

    def stop_video_loader(self):

        # if len(self.video_frame_loader_processes) > 0:
        self._video_loader_thread_stopped.value = True
        # for frame_loader in self.video_frame_loader_processes:
        #     frame_loader.join()
        if self.video_loader_thread is not None:
            self.video_loader_thread.join()

        # self.video_frame_loader_processes.clear()
        self.video_loader_thread = None
        # self._video_loader_thread_initialized.value = False

    def cleanup(self):
        self.stop_video_loader()
        # del self.video_frame_loader_processes

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
