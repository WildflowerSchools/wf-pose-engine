from threading import Thread
import time
import queue

import torch.multiprocessing as mp
import torch.utils.data

from cv_utils import VideoInput

from pose_engine.log import logger


# mp.set_start_method("spawn", force=True)


class VideoFramesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        video_paths: list[str] = [],
        frame_queue_maxsize: int = 120,
        wait_for_video_files: bool = True,
        wait_for_video_frames: bool = True,
        mp_manager=None,
    ):
        super(VideoFramesDataset).__init__()

        self.frame_queue_maxsize = frame_queue_maxsize
        self.wait_for_video_files = wait_for_video_files
        self.wait_for_video_frames = wait_for_video_frames

        self.video_path_queue = mp.JoinableQueue()
        for v in video_paths:
            self.video_path_queue.put(v)

        if mp_manager is None:
            mp_manager = mp.Manager()
        self.video_frame_queue = mp_manager.Queue(maxsize=frame_queue_maxsize)

        self.video_loader_thread_stopped = False
        self.video_loader_thread = None

    def add_video_path(self, video_path):
        self.video_path_queue.put(video_path)

    def size(self):
        return self.video_frame_queue.qsize()

    def maxsize(self):
        return self.frame_queue_maxsize

    def _start_video_loader(self):
        time.sleep(1)  # Hacky, but whatever

        while not self.video_loader_thread_stopped:
            try:
                video_path = self.video_path_queue.get(block=False)

                logger.info(f"Loading frames from {video_path}...")
                video_reader = VideoInput(input_path=video_path, queue_frames=True)

                frame_idx = 0
                while True:
                    frame = video_reader.get_frame()
                    if frame is None:
                        logger.info(f"Exhausted all frames in {video_path}")
                        break

                    frame_idx += 1

                    frame_written = False
                    while not frame_written and not self.video_loader_thread_stopped:
                        try:
                            logger.debug(
                                f"Putting '{video_path}' frame index {frame_idx} on frame queue"
                            )
                            self.video_frame_queue.put(
                                (frame, {"path": video_path, "frame_index": frame_idx}),
                                timeout=0.5,
                            )
                            frame_written = True
                        except queue.Full:
                            logger.debug(
                                f"Reached frame queue max size '{self.frame_queue_maxsize}', waiting to load additional frames..."
                            )

                    logger.debug(
                        "Frame written or video_loader_thread_stopped, reading next frame"
                    )
                self.video_path_queue.task_done()

                logger.info(f"Finished loading frames from {video_path}")
            except queue.Empty:
                if not self.wait_for_video_files:
                    logger.info(
                        "Video file queue empty and wait_for_video_files set to False, killing video loader"
                    )
                    self.video_loader_thread_stopped = True

                logger.debug("Video file queue empty, sleeping for 1 second")
                time.sleep(1)

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
                            f"Nothing to read from frame queue, terminating iterator"
                        )
                        break

                    if self.video_loader_thread_stopped:
                        logger.info(
                            f"Video loader is terminated and no more items in the frame queue, terminating iterator"
                        )
                        break

                # logger.debug(f"Nothing to read from frame queue, waiting...")

        logger.debug("Video frames dataset iter finished")

    def stop_video_loader(self):
        self.video_loader_thread_stopped = True
        self.video_loader_thread.join()
