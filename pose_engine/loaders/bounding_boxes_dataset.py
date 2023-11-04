from ctypes import c_bool
from threading import Thread
import time
import queue

import torch
import torch.multiprocessing as mp
import torch.utils.data

from cv_utils import VideoInput

from pose_engine.log import logger


# mp.set_start_method("spawn", force=True)


class BoundingBoxesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        bbox_queue_maxsize: int = 120,
        wait_for_bboxes: bool = True,
        mp_manager=None,
    ):
        super(BoundingBoxesDataset).__init__()

        self.done_loading_dataset = mp.Value(c_bool, False)

        self.bbox_queue_maxsize = bbox_queue_maxsize
        self.wait_for_bboxes = wait_for_bboxes

        if mp_manager is None:
            mp_manager = mp.Manager()

        self.bbox_queue = mp_manager.Queue(maxsize=bbox_queue_maxsize)

    def add_bboxes(self, bbox_records):
        (bboxes, frame, meta) = bbox_records

        move_to_numpy = (
            True  # TODO: Figure out whey I can't share tensors across processes
        )
        if move_to_numpy:
            for key in meta.keys():
                if isinstance(meta[key], torch.Tensor):
                    meta[key] = meta[key].cpu().numpy()

            self.bbox_queue.put((bboxes.cpu().numpy(), frame.cpu().numpy(), meta))

        # Attempts at making tensors shareable across processes
        # for key in meta.keys():
        #     if isinstance(meta[key], torch.Tensor):
        #         meta[key] = meta[key].clone().share_memory_()

        # self.bbox_queue.put((bboxes.clone().share_memory_(), frame.clone().share_memory_(), meta))

    def size(self):
        return self.bbox_queue.qsize()

    def done_loading(self):
        self.done_loading_dataset.value = True

    def __iter__(self):
        while True:
            try:
                bbox, image, meta = self.bbox_queue.get(block=False, timeout=0.5)
                if bbox is not None:
                    yield (bbox, image, meta)
            except queue.Empty:
                # DO NOT REMOVE: the "qsize()" assertion, this is important as the queue.Empty exception doesn't necessarily mean the queue is empty
                if self.bbox_queue.qsize() == 0:
                    if not self.wait_for_bboxes:
                        logger.info(
                            f"Nothing to read from bbox queue, terminating iterator"
                        )
                        break

                    if self.done_loading_dataset.value:
                        logger.info(
                            f"Stopping bounding box dataset iteration, bbox queue is empty and dataset has been set as having exhausted all boxes"
                        )
                        break

                # logger.debug(f"Nothing to read from bbox queue, waiting...")
