from ctypes import c_bool
from multiprocessing import sharedctypes
import multiprocessing.context
import time
from typing import Optional
import queue

import torch
import torch.multiprocessing as mp
import torch.utils.data

from faster_fifo import Queue as ffQueue

from pose_engine.log import logger


class RTMOImagesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        queue_maxsize: int = 2,
        wait_for_images: bool = True,
        mp_manager: Optional[mp.Manager] = None,
    ):
        super().__init__()

        self.done_loading_dataset: sharedctypes.Synchronized = mp.Value(c_bool, False)

        self.queue_maxsize: int = queue_maxsize
        self.wait_for_images: bool = wait_for_images

        self._queue_wait_time: sharedctypes.Synchronized = mp.Value("d", 0)

        if mp_manager is None:
            mp_manager = mp.Manager()

        self.queue = mp_manager.Queue(maxsize=queue_maxsize)
        # self.queue = ffQueue(max_size_bytes=3 * 640 * 480 * queue_maxsize)

    def add_data_object(self, pre_processed_data_samples):
        # move_to_numpy = True  # TODO: Figure out whey I can't share tensors across processes. Unless I move tensors to the CPU, I get the error "RuntimeError: Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending."
        # if move_to_numpy:
        #     for key in meta.keys():
        #         if isinstance(meta[key], torch.Tensor):
        #             meta[key] = meta[key].cpu()

        #     self.queue.put((bboxes.cpu(), frame.cpu(), meta))
        # else:
        #     # Attempts at making tensors shareable across processes
        #     for key in meta.keys():
        #         if isinstance(meta[key], torch.Tensor):
        #             meta[key] = meta[key].clone().share_memory_()

        #     self.queue.put(
        #         (bboxes.clone().share_memory_(), frame.clone().share_memory_(), meta)
        #     )

        # for sample in pre_processed_data_samples:
        #     sample['inputs'] = sample['inputs'].detach().clone()

        s = time.time()
        for item in pre_processed_data_samples:
            if item["inputs"].device.type == "cuda":
                item["inputs"] = item["inputs"].to("cpu")

        self.queue.put(pre_processed_data_samples)
        # if isinstance(pre_processed_data_samples, list):
        #     for p in pre_processed_data_samples:
        #         self.queue.put(p)
        # else:
        #     self.queue.put(pre_processed_data_samples)

        logger.info(
            f"Added {len(pre_processed_data_samples)} items to pre-processed queue, seconds to add items: {round(time.time() - s, 2)} seconds"
        )

    @property
    def queue_wait_time(self):
        return self._queue_wait_time.value

    def size(self):
        return self.queue.qsize()

    def maxsize(self):
        return self.queue_maxsize

    def done_loading(self):
        self.done_loading_dataset.value = True

    def __getitem__(self, _idx):
        raise NotImplementedError(
            "Attempted to use __getitem__, but RTMOImagesDataset is an IterableDataset so __getitem__ is intentionally not implemented"
        )

    def __iter__(self):
        while True:
            start_wait = time.time()

            try:
                pose_data_sample = self.queue.get(block=False, timeout=0.5)
                if pose_data_sample is not None:
                    yield pose_data_sample
            except queue.Empty:
                end_wait = time.time() - start_wait
                with self._queue_wait_time.get_lock():
                    self._queue_wait_time.value += end_wait

                # DO NOT REMOVE: the "qsize()" assertion, this is important as the queue.Empty exception doesn't necessarily mean the queue is empty
                if self.queue.qsize() == 0:
                    if not self.wait_for_images:
                        logger.info(
                            "Nothing to read from RTMO images queue, terminating iterator"
                        )
                        break

                    if self.done_loading_dataset.value:
                        logger.info(
                            "Stopping RTMO images dataset iteration, RTMO images queue is empty and dataset has been set as having exhausted all boxes"
                        )
                        break

    def cleanup(self):
        try:
            if self.queue is not None:
                while True:
                    item = self.queue.get_nowait()
                    del item
        except queue.Empty:
            pass
        finally:
            del self.queue
