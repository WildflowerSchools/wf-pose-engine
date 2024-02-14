from ctypes import c_bool
from multiprocessing import sharedctypes
import queue
import time

import torch
import torch.multiprocessing as mp
import torch.utils.data

from pose_engine.log import logger


class PosesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        pose_queue_maxsize: int = 120,
        wait_for_poses: bool = True,
        mp_manager=None,
    ):
        super().__init__()

        self.done_loading_dataset: sharedctypes.Synchronized = mp.Value(c_bool, False)

        self.pose_queue_maxsize = pose_queue_maxsize
        self.wait_for_poses = wait_for_poses

        self._queue_wait_time: sharedctypes.Synchronized = mp.Value("d", 0)

        if mp_manager is None:
            mp_manager = mp.Manager()

        self.pose_queue = mp_manager.Queue(maxsize=pose_queue_maxsize)

    def add_data_object(self, data_object):
        self.pose_queue.put(data_object)

    @property
    def queue_wait_time(self):
        return self._queue_wait_time.value

    def size(self):
        return self.pose_queue.qsize()

    def maxsize(self):
        return self.pose_queue_maxsize

    def done_loading(self):
        self.done_loading_dataset.value = True

    def __getitem__(self, _idx):
        raise NotImplementedError(
            "Attempted to use __getitem__, but PosesDataset is an IterableDataset so __getitem__ is intentionally not implemented"
        )

    def __iter__(self):
        while True:
            start_wait = time.time()

            try:
                pose_queue_item = self.pose_queue.get(block=False, timeout=0.5)
                if isinstance(pose_queue_item, tuple):
                    yield self.pose_queue.get(block=False, timeout=0.5)
                elif isinstance(pose_queue_item, list):
                    for t in pose_queue_item:
                        yield t
            except queue.Empty:
                end_wait = time.time() - start_wait
                with self._queue_wait_time.get_lock():
                    self._queue_wait_time.value += end_wait

                # DO NOT REMOVE: the "qsize()" assertion, this is important as the queue.Empty exception doesn't necessarily mean the queue is empty
                if self.pose_queue.qsize() == 0:
                    if not self.wait_for_poses:
                        logger.info(
                            "Nothing to read from pose queue, terminating iterator"
                        )
                        break

                    if self.done_loading_dataset.value:
                        logger.info(
                            "Stopping bounding pose dataset iteration, pose queue is empty and dataset has been set as having exhausted all poses"
                        )
                        break

                # logger.debug(f"Nothing to read from pose queue, waiting...")

    def cleanup(self):
        try:
            while True:
                item = self.pose_queue.get_nowait()
                del item
        except queue.Empty:
            pass
        finally:
            del self.pose_queue
