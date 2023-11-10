from ctypes import c_bool
import queue

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
        super(PosesDataset).__init__()

        self.done_loading_dataset = mp.Value(c_bool, False)

        self.pose_queue_maxsize = pose_queue_maxsize
        self.wait_for_poses = wait_for_poses

        if mp_manager is None:
            mp_manager = mp.Manager()

        self.pose_queue = mp_manager.Queue(maxsize=pose_queue_maxsize)

    def add_pose(self, pose_record):
        (pose, bbox, meta) = pose_record

        move_to_numpy = True  # TODO: Figure out whey I can't share tensors across processes. Unless I move tensors to the CPU, I get the error "RuntimeError: Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending."
        if move_to_numpy:
            if isinstance(pose, torch.Tensor):
                pose = pose.cpu()

            if isinstance(bbox, torch.Tensor):
                bbox = bbox.cpu()

            for key in meta.keys():
                if isinstance(meta[key], torch.Tensor):
                    meta[key] = meta[key].cpu()

            self.pose_queue.put((pose, bbox, meta))
        else:
            # Attempts at making tensors shareable across processes
            if isinstance(pose, torch.Tensor):
                pose = pose.clone().share_memory_()

            if isinstance(bbox, torch.Tensor):
                bbox = bbox.clone().share_memory_()

            for key in meta.keys():
                if isinstance(meta[key], torch.Tensor):
                    meta[key] = meta[key].clone().share_memory_()

            self.pose_queue.put((pose, bbox, meta))

    def size(self):
        return self.pose_queue.qsize()

    def maxsize(self):
        return self.pose_queue_maxsize

    def done_loading(self):
        self.done_loading_dataset.value = True

    def __iter__(self):
        while True:
            try:
                pose, bbox, meta = self.pose_queue.get(block=False, timeout=0.5)
                if pose is not None:
                    yield (pose, bbox, meta)
            except queue.Empty:
                # DO NOT REMOVE: the "qsize()" assertion, this is important as the queue.Empty exception doesn't necessarily mean the queue is empty
                if self.pose_queue.qsize() == 0:
                    if not self.wait_for_poses:
                        logger.info(
                            f"Nothing to read from pose queue, terminating iterator"
                        )
                        break

                    if self.done_loading_dataset.value:
                        logger.info(
                            f"Stopping bounding pose dataset iteration, pose queue is empty and dataset has been set as having exhausted all poses"
                        )
                        break

                # logger.debug(f"Nothing to read from pose queue, waiting...")
