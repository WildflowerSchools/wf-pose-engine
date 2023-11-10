from ctypes import c_bool
import queue

import torch
import torch.multiprocessing as mp
import torch.utils.data

from pose_engine.log import logger


class BoundingBoxesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        bbox_queue_maxsize: int = 120,
        wait_for_bboxes: bool = True,
        mp_manager=None,
    ):
        super().__init__()

        self.done_loading_dataset = mp.Value(c_bool, False)

        self.bbox_queue_maxsize = bbox_queue_maxsize
        self.wait_for_bboxes = wait_for_bboxes

        if mp_manager is None:
            mp_manager = mp.Manager()

        self.bbox_queue = mp_manager.Queue(maxsize=bbox_queue_maxsize)

    def add_bboxes(self, bbox_records):
        (bboxes, frame, meta) = bbox_records

        move_to_numpy = True  # TODO: Figure out whey I can't share tensors across processes. Unless I move tensors to the CPU, I get the error "RuntimeError: Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending."
        if move_to_numpy:
            for key in meta.keys():
                if isinstance(meta[key], torch.Tensor):
                    meta[key] = meta[key].cpu()

            self.bbox_queue.put((bboxes.cpu(), frame.cpu(), meta))
        else:
            # Attempts at making tensors shareable across processes
            for key in meta.keys():
                if isinstance(meta[key], torch.Tensor):
                    meta[key] = meta[key].clone().share_memory_()

            self.bbox_queue.put(
                (bboxes.clone().share_memory_(), frame.clone().share_memory_(), meta)
            )

    def size(self):
        return self.bbox_queue.qsize()

    def maxsize(self):
        return self.bbox_queue_maxsize

    def done_loading(self):
        self.done_loading_dataset.value = True

    def __iter__(self):
        while True:
            try:
                bboxes, images, meta = self.bbox_queue.get(block=False, timeout=0.5)
                if bboxes is not None:
                    yield (bboxes, images, meta)
            except queue.Empty:
                # DO NOT REMOVE: the "qsize()" assertion, this is important as the queue.Empty exception doesn't necessarily mean the queue is empty
                if self.bbox_queue.qsize() == 0:
                    if not self.wait_for_bboxes:
                        logger.info(
                            "Nothing to read from bbox queue, terminating iterator"
                        )
                        break

                    if self.done_loading_dataset.value:
                        logger.info(
                            "Stopping bounding box dataset iteration, bbox queue is empty and dataset has been set as having exhausted all boxes"
                        )
                        break

                # logger.debug(f"Nothing to read from bbox queue, waiting...")
