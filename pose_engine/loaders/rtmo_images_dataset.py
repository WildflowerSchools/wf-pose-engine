import concurrent.futures
from ctypes import c_bool
from multiprocessing import sharedctypes
import multiprocessing.context
from threading import Thread
import time
from typing import List, Optional, Union
import queue

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data

from mmengine.registry import init_default_scope
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy

from PIL import Image

# from faster_fifo import Queue as ffQueue

from pose_engine.loaders import VideoFramesDataLoader, BoundingBoxesDataLoader
from pose_engine.log import logger
from pose_engine.mmlab.mmpose.transforms import BatchBottomupResize


class RTMOImagesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        raw_images_loader: torch.utils.data.DataLoader,
        pipeline,
        model_config,
        batch_bottomup_resize_transform: Optional[BatchBottomupResize] = None,
        queue_maxsize: int = 10,
        wait_for_images: bool = True,
        mp_manager: Optional[mp.Manager] = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.raw_images_loader = raw_images_loader
        self.pipeline = pipeline
        self.model_config = model_config
        self.batch_bottomup_resize_transform = batch_bottomup_resize_transform

        self.done_loading_dataset: sharedctypes.Synchronized = mp.Value(c_bool, False)

        self.queue_maxsize: int = queue_maxsize
        self.wait_for_images: bool = wait_for_images

        self._queue_wait_time: sharedctypes.Synchronized = mp.Value("d", 0)

        if mp_manager is None:
            mp_manager = mp.Manager()

        self.device = device

        self.queue = mp_manager.Queue(maxsize=queue_maxsize)
        # self.queue = ffQueue(max_size_bytes=3 * 640 * 480 * queue_maxsize)

    def add_data_object(self, pre_processed_data_samples):
        # move_to_numpy = True  # TODO: Figure out why I can't share tensors across processes. Unless I move tensors to the CPU, I get the error "RuntimeError: Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending."
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

    def pre_process(
        self,
        imgs: Union[list[np.ndarray], list[str]],
        inference_mode: str = "topdown",
        bboxes: Optional[Union[List[List], List[np.ndarray]]] = None,
        meta: List[dict] = None,
        bbox_format: str = "xyxy",
    ) -> List[PoseDataSample]:
        logger.info("PreProcessor::pre_process: start...")
        scope = self.model_config.get("default_scope", "mmpose")
        if scope is not None:
            init_default_scope(scope)

        # construct batch data samples
        raw_data_for_pre_processing_list = []
        meta_mapping = []
        pre_processed_data_list = []
        total_pre_processing_time = 0

        s = time.time()
        logger.info("PreProcessor::pre_process: prep...")
        for img_idx, img in enumerate(imgs):
            if inference_mode == "topdown":
                img_bboxes = bboxes[img_idx]
                if img_bboxes is None:
                    # get bbox from the image size
                    if isinstance(img, str):
                        w, h = Image.open(img).size
                    else:
                        h, w = img.shape[:2]

                    img_bboxes = np.array([[0, 0, w, h, 1]], dtype=np.float32)
                elif len(img_bboxes) == 0:
                    continue
                else:
                    if isinstance(img_bboxes, list):
                        img_bboxes = np.array(img_bboxes)

                    assert bbox_format in {
                        "xyxy",
                        "xywh",
                    }, f'Invalid bbox_format "{bbox_format}".'

                    if bbox_format == "xywh":
                        img_bboxes = bbox_xywh2xyxy(img_bboxes)

                for bbox in img_bboxes:
                    meta_mapping.append(meta[img_idx])

                    if isinstance(img, str):
                        data_for_pre_processing = {"img_path": img}
                    else:
                        data_for_pre_processing = {"img": img}
                    data_for_pre_processing["bbox"] = bbox[None, :4]  # shape (1, 4)
                    data_for_pre_processing["bbox_score"] = bbox[None, 4]  # shape (1,)
                    raw_data_for_pre_processing_list.append(data_for_pre_processing)
            else:
                meta_item = {}
                for key in meta.keys():
                    meta_item[key] = meta[key][img_idx]

                meta_mapping.append(meta_item)
                data_for_pre_processing = {"img": img}
                raw_data_for_pre_processing_list.append(data_for_pre_processing)
        total_pre_processing_time += time.time() - s

        logger.info("PreProcessor::pre_process: start pipeline processing...")

        s_prep = time.time()

        futures = []
        n_threads = 4
        # raw_data_list_chunk_size = (len(raw_data_for_pre_processing_list) // n_threads) + 1
        with concurrent.futures.ThreadPoolExecutor(n_threads) as executor:
            # for ii in range(0, len(raw_data_for_pre_processing_list), raw_data_list_chunk_size):
            for idx, data_for_pre_processing in enumerate(
                raw_data_for_pre_processing_list
            ):
                # chunk = raw_data_for_pre_processing_list[
                #     ii : ii + raw_data_list_chunk_size
                # ]
                futures.append(executor.submit(self.pipeline, data_for_pre_processing))

            for future in concurrent.futures.as_completed(futures):
                processed_pipeline_data = future.result()
                processed_pipeline_data["meta_mapping"] = meta_mapping[idx]
                pre_processed_data_list.append(processed_pipeline_data)
        total_pre_processing_time += time.time() - s_prep
        logger.info("PreProcessor::pre_process: finish pipeline processing...")

        logger.info(
            f"Pose estimator data pipeline pre-processing prep performance (device: {self.device}): {len(pre_processed_data_list)} records {round(time.time() - s_prep, 3)} seconds"
        )

        # if self.device == 'cpu':
        list(map(lambda r: r["inputs"].pin_memory(), pre_processed_data_list))
        if self.batch_bottomup_resize_transform is not None:
            logger.info("PreProcessor::pre_process: start resize processing...")

            s_resize = time.time()
            # Batch resizing consumes an outsized chunk of CUDA memory, so split this step across two passes
            # data_list = self.batch_bottomup_resize_transform.transform(
            #     data_list=data_list,
            #     device=self.device,
            # )
            data_list_chunk_size = (len(pre_processed_data_list) // 2) + 1
            for ii in range(0, len(pre_processed_data_list), data_list_chunk_size):
                pre_processed_data_list[ii : ii + data_list_chunk_size] = (
                    self.batch_bottomup_resize_transform.transform(
                        data_list=pre_processed_data_list[
                            ii : ii + data_list_chunk_size
                        ],
                        device=self.device,
                    )
                )
            total_pre_processing_time += time.time() - s_resize
            logger.info(
                f"Pose estimator data pipeline pre-processing resize performance (device: {self.device}): {len(pre_processed_data_list)} records {round(time.time() - s_resize, 3)} seconds"
            )
        # total_pre_processing_time += time.time() - s

        if total_pre_processing_time == 0:
            records_per_second = "N/A"
        else:
            records_per_second = round(
                len(pre_processed_data_list) / total_pre_processing_time, 3
            )

        logger.info(
            f"Pre-processing pipeline performance (device: {self.device}): {len(pre_processed_data_list)} records {round(total_pre_processing_time, 3)} seconds {records_per_second} records/second"
        )

        return pre_processed_data_list

    def _start_pre_processing_loader(self):
        is_topdown = False

        last_loop_start_time = None
        # self._start_time.value = time.time()
        for instance_batch_idx, data in enumerate(self.raw_images_loader):
            current_loop_time = time.time()
            seconds_between_loops = 0
            if last_loop_start_time is not None:
                seconds_between_loops = current_loop_time - last_loop_start_time

            global_batch_idx = instance_batch_idx

            # with self._batch_count.get_lock():
            #     self._batch_count.value += 1
            #     global_batch_idx = instance_batch_idx

            # with self._time_waiting.get_lock():
            #     self._time_waiting.value += seconds_between_loops

            # with self._first_inference_time.get_lock():
            #     if self._first_inference_time.value == -1:
            #         self._first_inference_time.value = current_loop_time

            if isinstance(self.raw_images_loader, BoundingBoxesDataLoader):
                is_topdown = True
                (bboxes, frames, meta) = data
            elif isinstance(self.raw_images_loader, VideoFramesDataLoader):
                bboxes = None
                (frames, meta) = data
            else:
                raise ValueError(
                    f"Unknown dataloader ({type(self.raw_images_loader)}) provided to RTMOImagesDataset, accepted loaders are any of [BoundingBoxesDataLoader, VideoFramesDataLoader]"
                )

            logger.info(
                f"PreProcessor::loop: handle new batch of size {len(frames)}..."
            )

            try:
                logger.info(
                    f"Pre-processing pose estimation batch #{global_batch_idx} (device: {self.device}) - Includes {len(frames)} frames - Seconds since last batch {round(seconds_between_loops, 3)}"
                )

                # with self._frame_count.get_lock():
                #     self._frame_count.value += len(frames)

                imgs = []
                for img_idx, img in enumerate(frames):
                    if isinstance(img, torch.Tensor):
                        img = img.numpy()

                    imgs.append(img)

                    if bboxes is not None and isinstance(bboxes[img_idx], torch.Tensor):
                        bboxes[img_idx] = bboxes[img_idx].numpy()

                inference_mode = "topdown" if is_topdown else "onestage"
                logger.info(f"PreProcessor::loop: starting pre_process...")

                pre_processed_data_samples = []
                futures = []
                n_threads = 4
                chunk_size = (len(imgs) // n_threads) + 1
                with concurrent.futures.ThreadPoolExecutor(n_threads) as executor:
                    for ii in range(0, len(imgs), chunk_size):
                        imgs_chunk = imgs[ii : ii + chunk_size]
                        bboxes_chunk = None
                        if bboxes is not None:
                            bboxes_chunk = bboxes[ii : ii + chunk_size]
                        futures.append(
                            executor.submit(
                                self.pre_process,
                                **dict(
                                    inference_mode=inference_mode,
                                    imgs=imgs_chunk,
                                    bboxes=bboxes_chunk,
                                    meta=meta,
                                ),
                            )
                        )

                    for future in concurrent.futures.as_completed(futures):
                        pre_processed_data_samples.extend(future.result())
                        # processed_pipeline_data["meta_mapping"] = meta_mapping[idx]
                        # pre_processed_data_list.append(processed_pipeline_data)

                        # pre_processed_data_samples = self.pre_process(
                        #     inference_mode=inference_mode,
                        #     imgs=imgs,
                        #     bboxes=bboxes,
                        #     meta=meta,
                        # )
                logger.info(f"PreProcessor::loop: done pre_process")

                start_add_items_to_queue = time.time()
                self.add_data_object(pre_processed_data_samples)
                logger.info(
                    f"Pre-processing pose estimation batch #{global_batch_idx} add to pre-process queue performance (device: {self.device}): {round(time.time() - start_add_items_to_queue, 3)} seconds"
                )
                logger.info(
                    f"Pre-processing pose estimation batch #{global_batch_idx} overall performance (device: {self.device}) - Includes {len(frames)} frames: {round(time.time() - current_loop_time, 3)} seconds - {round(len(frames) / (time.time() - current_loop_time), 2)} FPS"
                )
            finally:
                del bboxes
                del frames
                del meta

            last_loop_start_time = current_loop_time

        self.done_loading()

    def __getitem__(self, _idx):
        raise NotImplementedError(
            "Attempted to use __getitem__, but RTMOImagesDataset is an IterableDataset so __getitem__ is intentionally not implemented"
        )

    def start_pre_processing_loader(self):
        # if self.video_loader_thread is None and not self.video_loader_thread_initialized:
        self.video_loader_thread = Thread(
            target=self._start_pre_processing_loader, args=()
        )
        self.video_loader_thread.daemon = False
        self.video_loader_thread.start()
        # self.video_loader_thread = mp.Process(target=self._start_pre_processing_loader, args=())
        # self.video_loader_thread.daemon = False
        # self.video_loader_thread.start()

    def __iter__(self):
        self.start_pre_processing_loader()

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
        if self.video_loader_thread is not None:
            self.video_loader_thread.join()

            # self.video_loader_thread.close()

        # self.video_frame_loader_processes.clear()
        self.video_loader_thread = None

        try:
            if self.queue is not None:
                while True:
                    item = self.queue.get_nowait()
                    del item
        except queue.Empty:
            pass
        finally:
            del self.queue
