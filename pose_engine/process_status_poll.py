import multiprocessing as mp
from threading import Thread
import time
from typing import Optional

from pose_engine import loaders
from pose_engine.log import logger


class ProcessStatusPoll:
    def __init__(
        self,
        video_frame_dataset: loaders.VideoFramesDataset,
        bounding_box_dataset: loaders.BoundingBoxesDataset,
        poses_dataset: loaders.PosesDataset,
        poll: int = 10,
    ):
        self.video_frame_dataset: loaders.VideoFramesDataset = video_frame_dataset
        self.bounding_box_dataset: loaders.BoundingBoxesDataset = bounding_box_dataset
        self.poses_dataset: loaders.PosesDataset = poses_dataset
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
        while not self.stop_event.is_set():
            logger.info(
                f"Video frame queue size: {self.video_frame_dataset.size()}/{self.video_frame_dataset.maxsize()}"
            )
            logger.info(
                f"Bounding box queue size: {self.bounding_box_dataset.size()}/{self.bounding_box_dataset.maxsize()}"
            )
            logger.info(
                f"Poses queue size: {self.poses_dataset.size()}/{self.poses_dataset.maxsize()}"
            )

            time.sleep(self.poll)
