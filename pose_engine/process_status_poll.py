import multiprocessing as mp
from threading import Thread
import time

import torch

from pose_engine.log import logger


class StatusPoll:
    def __init__(
        self,
        video_frame_dataset: torch.utils.data.IterableDataset,
        bounding_box_dataset: torch.utils.data.IterableDataset,
        poll: int = 5,
    ):
        self.video_frame_dataset = video_frame_dataset
        self.bounding_box_dataset = bounding_box_dataset
        self.poll = poll

        # self.process = mp.Process(target=self._run, args=())
        self.stop_event = mp.Event()

        self.polling_thread = None

    def start(self):
        if self.polling_thread is not None:
            return

        self.stop_event.clear()

        self.polling_thread = Thread(target=self._run, args=())
        self.polling_thread.daemon = True
        self.polling_thread.start()

    def stop(self):
        self.stop_event.set()
        self.polling_thread = None

    def _run(self):
        while not self.stop_event.is_set():
            logger.info(
                f"Video frame queue size: {self.video_frame_dataset.size()}/{self.video_frame_dataset.maxsize()}"
            )
            logger.info(
                f"Bounding box queue size: {self.bounding_box_dataset.size()}/{self.bounding_box_dataset.maxsize()}"
            )

            time.sleep(self.poll)
