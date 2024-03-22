import concurrent.futures
from contextlib import nullcontext
from ctypes import c_bool
from multiprocessing import sharedctypes
import pathlib
import queue
from random import randrange
import sys
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torch._dynamo as torchdynamo
import torch.distributed
import torch.multiprocessing as mp
import torch.utils.data

# import onnxconverter_common

import torch_tensorrt  # DO NOT DELETE

import cv2
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmengine.dataset import Compose, default_collate
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis.inference import dataset_meta_from_config
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy
import mmpose.models.heads.hybrid_heads.rtmo_head
import mmpose.evaluation.functional
import mmpose.evaluation.functional.nms

# from mmpose.evaluation.functional import nearby_joints_nms

# from faster_fifo import Queue as ffQueue
from PIL import Image

from pose_engine.loaders import (
    VideoFramesDataLoader,
    BoundingBoxesDataLoader,
    # RTMOImagesDataLoader,
    # RTMOImagesDataset,
)
from pose_engine.mmlab.mmpose.transforms import BatchBottomupResize
from pose_engine.torch.mmlab_compatible_data_parallel import MMLabCompatibleDataParallel
from pose_engine.log import logger
from pose_engine.mmlab.mmpose.misc import nms_torch, nearby_joints_nms
from pose_engine.util import is_valid_url

# Override the RTMO head's NMS algorithm with our custom NMS alorithm
mmpose.models.heads.hybrid_heads.rtmo_head.nms_torch = nms_torch


class PoseEstimatorPreProcessor:
    def __init__(
        self,
        model_config,
        dataset_meta,
        device: str = "cpu",
        run_in_background: bool = False,
    ) -> None:
        self.model_config = model_config
        self.dataset_meta = dataset_meta
        self.device = device
        self.run_in_background = run_in_background

        self.batch_bottomup_resize_transform = None
        self.rtmo_pre_processed_images_dataloader = None

        queue_maxsize = 50
        mp_manager = mp.Manager()
        self.queue = mp_manager.Queue(maxsize=queue_maxsize)
        # self.queue = ffQueue(max_size_bytes=3 * 640 * 640 * queue_maxsize)

        self.process = None
        self.stop_event: mp.Event = mp.Event()

        self._queue_wait_time: sharedctypes.Synchronized = mp.Value("d", 0)
        self._frame_count: sharedctypes.Synchronized = mp.Value(
            "i", 0
        )  # Each frame contains multiple boxes, so we track frames separately from inferences

        self._init_pipeline()

    @property
    def queue_wait_time(self):
        return self._queue_wait_time.value

    @property
    def frame_count(self) -> int:
        return self._frame_count.value

    def size(self):
        return self.queue.qsize()

    def __iter__(self):
        while True:
            start_wait = time.time()

            try:
                yield self.queue.get(block=False, timeout=0.5)
            except queue.Empty:
                end_wait = time.time() - start_wait
                with self._queue_wait_time.get_lock():
                    self._queue_wait_time.value += end_wait

                if self.stop_event.is_set():
                    break

    def _init_pipeline(self):
        self.pipeline = self.model_config.test_dataloader.dataset.pipeline

        # Modify the pipeline by slicing out the default BottomupResize method
        # BottomupResize does not batch process. Below, we substitute in a
        # torchvision resize method that can take advantage of batching
        modified_pipeline = []
        for stage in self.pipeline:
            if False and stage["type"] == "BottomupResize":
                self.batch_bottomup_resize_transform = BatchBottomupResize(**stage)
            else:
                modified_pipeline.append(stage)

        self.pipeline = Compose(modified_pipeline)

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

        # futures = []
        # n_threads = 1
        # # raw_data_list_chunk_size = (len(raw_data_for_pre_processing_list) // n_threads) + 1
        # with concurrent.futures.ThreadPoolExecutor(n_threads) as executor:
        #     # for ii in range(0, len(raw_data_for_pre_processing_list), raw_data_list_chunk_size):
        #     for idx, data_for_pre_processing in enumerate(
        #         raw_data_for_pre_processing_list
        #     ):
        #         # chunk = raw_data_for_pre_processing_list[
        #         #     ii : ii + raw_data_list_chunk_size
        #         # ]
        #         futures.append(executor.submit(self.pipeline, data_for_pre_processing))

        #     for future in concurrent.futures.as_completed(futures):
        #         processed_pipeline_data = future.result()
        #         processed_pipeline_data["meta_mapping"] = meta_mapping[idx]
        #         pre_processed_data_list.append(processed_pipeline_data)

        for idx, data_for_pre_processing in enumerate(raw_data_for_pre_processing_list):
            processed_pipeline_data = self.pipeline(data_for_pre_processing)
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

    def _preprocessing_process(self, process_index: int, loader=None):
        is_topdown = False

        last_loop_start_time = None
        # self._start_time.value = time.time()
        for instance_batch_idx, data in enumerate(loader):
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

            if isinstance(loader, BoundingBoxesDataLoader):
                is_topdown = True
                (bboxes, frames, meta) = data
            elif isinstance(loader, VideoFramesDataLoader):
                bboxes = None
                (frames, meta) = data
            else:
                raise ValueError(
                    f"Unknown dataloader ({type(loader)}) provided to PoseEstimator, accepted loaders are any of [BoundingBoxesDataLoader, VideoFramesDataLoader]"
                )

            logger.info(
                f"PreProcessor::loop: handle new batch of size {len(frames)}..."
            )

            try:
                logger.info(
                    f"Pre-processing pose estimation batch #{global_batch_idx} (device: {self.device}) - Includes {len(frames)} frames - Seconds since last batch {round(seconds_between_loops, 3)}"
                )

                with self._frame_count.get_lock():
                    self._frame_count.value += len(frames)

                imgs = []
                for img_idx, img in enumerate(frames):
                    if isinstance(img, torch.Tensor):
                        img = img.numpy()

                    imgs.append(img)

                    if bboxes is not None and isinstance(bboxes[img_idx], torch.Tensor):
                        bboxes[img_idx] = bboxes[img_idx].numpy()

                inference_mode = "topdown" if is_topdown else "onestage"
                logger.info(f"PreProcessor::loop: starting pre_process...")

                pre_processed_data_samples = self.pre_process(
                    inference_mode=inference_mode,
                    imgs=imgs,
                    bboxes=bboxes,
                    meta=meta,
                )
                # pre_processed_data_samples = []
                # futures = []
                # n_threads = 2
                # chunk_size = (len(imgs) // n_threads) + 1
                # with concurrent.futures.ThreadPoolExecutor(n_threads) as executor:
                #     for ii in range(0, len(imgs), chunk_size):
                #         imgs_chunk = imgs[ii : ii + chunk_size]
                #         bboxes_chunk = None
                #         if bboxes is not None:
                #             bboxes_chunk = bboxes[ii : ii + chunk_size]
                #         futures.append(
                #             executor.submit(
                #                 self.pre_process,
                #                 **dict(
                #                     inference_mode=inference_mode,
                #                     imgs=imgs_chunk,
                #                     bboxes=bboxes_chunk,
                #                     meta=meta,
                #                 ),
                #             )
                #         )

                #     for future in concurrent.futures.as_completed(futures):
                #         pre_processed_data_samples.extend(future.result())
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
                self.queue.put(pre_processed_data_samples)
                # self.rtmo_pre_processed_images_dataloader.dataset.add_data_object(
                #     pre_processed_data_samples
                # )
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

        self.stop_event.set()
        # self.rtmo_pre_processed_images_dataloader.dataset.done_loading()

    def start_preprocessing_process(self, loader):
        # rtmo_pre_processed_images_dataset = RTMOImagesDataset(
        #     queue_maxsize=10,
        #     pipeline=self.pipeline,
        #     model_config=self.model_config,
        #     wait_for_images=True,
        #     raw_images_loader=loader,
        #     batch_bottomup_resize_transform=self.batch_bottomup_resize_transform,
        #     device=self.device
        # )

        # self.rtmo_pre_processed_images_dataloader = RTMOImagesDataLoader(
        #     dataset=rtmo_pre_processed_images_dataset,
        #     num_workers=1,
        #     pin_memory=True,
        #     batch_size=1,
        # )
        if self.process is not None:
            return

        self.stop_event.clear()

        n_processes = mp.multiprocessing.cpu_count()
        if n_processes > 1:
            n_processes -= 1
        if n_processes > 2:
            n_processes = 2

        n_processes = 1
        # for _ in range(n_processes):
        #     process = mp.Process(target=self._preprocessing_process, args=(loader,), daemon=False)
        #     self.processes.append(process)
        #     process.start()

        self.process: mp.ProcessContext = mp.spawn(
            self._preprocessing_process,
            args=(loader,),
            nprocs=n_processes,
            daemon=False,
            join=False,
        )

    def stop(self):
        self.stop_event.set()
        # if self.rtmo_pre_processed_images_dataloader.dataset is not None:
        #     self.rtmo_pre_processed_images_dataloader.dataset.cleanup()

        # if len(self.processes) > 0:
        #     for p in self.processes:
        #         p.join()
        #         if isinstance(p, mp.Process):
        #             p.close()
        #         # self.process = None
        #     self.processes = []
        self.process.join()
        if isinstance(self.process, mp.Process):
            self.process.close()
        self.process = None


class PoseEstimator:
    def __init__(
        self,
        model_config_path: str = None,
        model_runtime: str = "pytorch",
        checkpoint: str = None,
        deployment_config_path: str = None,
        device: str = "cuda:1",
        batch_size: int = 75,
        use_fp16: bool = False,
        run_parallel: bool = False,
        run_distributed: bool = False,
        compile_engine: Optional[str] = None,
    ):
        torchdynamo.config.cache_size_limit = 512
        # torch._logging.set_logs(recompiles=True, dynamo=logging.DEBUG) #dynamo=logging.DEBUG
        # torch._dynamo.config.suppress_errors = True
        # torch.compiler.disable(fn=bbox_overlaps, recursive=True)

        if model_config_path is None or checkpoint is None:
            raise ValueError(
                "Pose estimator must be initialized by providing a config + checkpoint pair"
            )

        self.model_runtime = model_runtime
        self.checkpoint = checkpoint

        self.model_config_path = model_config_path
        self.model_config = load_config(self.model_config_path)[0]
        self.dataset_meta = dataset_meta_from_config(
            self.model_config, dataset_mode="train"
        )

        self.is_topdown = self.model_config["data_mode"] == "topdown"

        self.deployment_config_path = deployment_config_path
        self.deployment_config = None
        if self.deployment_config_path is None:
            self.deployment_config = None
        else:
            self.deployment_config = load_config(self.deployment_config_path)[0]

        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.run_parallel = run_parallel
        self.run_distributed = run_distributed
        self.compile_engine = compile_engine
        self.was_model_compiled = False

        self.distributed_rank: int = -1
        self.distributed_world_size: int = -1
        if self.run_distributed:
            self.distributed_rank = torch.distributed.get_rank()
            self.distributed_world_size = torch.distributed.get_world_size()

            self.device = f"cuda:{self.distributed_rank}"

        logger.info(f"Initializing pose estimator (device: {self.device})...")

        self.lock = mp.Lock()
        self._batch_count = mp.Value("i", 0)
        # self._frame_count = mp.Value(
        #     "i", 0
        # )  # Each frame contains multiple boxes, so we track frames separately from inferences
        self._inference_count = mp.Value("i", 0)
        self._frame_count = mp.Value("i", 0)
        self._start_time = mp.Value("d", -1.0)
        self._stop_time = mp.Value("d", -1.0)
        self._first_inference_time = mp.Value("d", -1.0)
        self._time_waiting = mp.Value("d", 0)

        # self.rtmo_pre_processed_images_dataset = RTMOImagesDataset(
        #     queue_maxsize=1000,
        #     wait_for_images=True
        # )
        # self.rtmo_pre_processed_images_dataloader = None  # Set when the iter_dataloader method is called

        self.pose_estimator = None
        self.task_processor = None
        self.pipeline = None
        self.batch_bottomup_resize_transform = None

        self._init_model()
        # self._init_pipeline()
        self.pre_processor: PoseEstimatorPreProcessor = None
        self._init_preprocessor()

        if self.use_fp16:
            self.pose_estimator = self.pose_estimator.half()
        self.pose_estimator = self.pose_estimator.to(memory_format=torch.channels_last)

        self._compile_model()
        self._warmup_model()

        self._start_time.value = time.time()

        logger.info(f"Finished initializing pose estimator (device: {self.device})")

    @property
    def frame_count(self) -> int:
        # return self.pre_processor.frame_count
        return self._frame_count.value

    @property
    def inference_count(self) -> int:
        return self._inference_count.value

    @property
    def start_time(self) -> float:
        return self._start_time.value

    @property
    def stop_time(self) -> float:
        return self._stop_time.value

    @property
    def first_inference_time(self) -> float:
        return self._first_inference_time.value

    @property
    def running_time_from_start(self) -> float:
        if self.start_time == -1:
            return 0

        current_time_or_stop_time = time.time()
        if self.stop_time > -1.0:
            current_time_or_stop_time = self.stop_time

        return current_time_or_stop_time - self.start_time

    @property
    def running_time_from_first_inference(self) -> float:
        if self.first_inference_time == -1:
            return 0

        current_time_or_stop_time = time.time()
        if self.stop_time > -1.0:
            current_time_or_stop_time = self.stop_time

        return current_time_or_stop_time - self.first_inference_time

    @property
    def time_waiting(self) -> int:
        return self._time_waiting.value

    def _init_model(self):
        if self.model_runtime == "pytorch":
            self.pose_estimator = init_pose_estimator(
                config=self.model_config_path,
                checkpoint=self.checkpoint,
                device=self.device,
            )
            self.pose_estimator.share_memory()

            # self.model_config = self.pose_estimator.cfg

            if self.run_parallel:
                self.pose_estimator = MMLabCompatibleDataParallel(
                    self.pose_estimator,
                    device_ids=list(range(torch.cuda.device_count())),
                    output_device=self.device,  # torch.device("cpu")
                )
            elif self.run_distributed:
                logger.info(
                    f"Configuring pose estimator model for DDP (DistributedDataParallel) on device {self.distributed_rank}..."
                )
                self.pose_estimator = MMDistributedDataParallel(
                    self.pose_estimator,
                    device_ids=[self.distributed_rank],
                    output_device=self.distributed_rank,
                    static_graph=True,
                )
                logger.info(
                    f"Finished configuring pose estimator model for DDP on device {self.distributed_rank}"
                )

        else:  # TensorRT or ONNX
            # from pose_engine.mmlab.mmdeploy.custom_ort_wrapper import CustomORTWrapper
            # from mmdeploy.backend.base import BACKEND_WRAPPER
            # register_module
            # @BACKEND_MANAGERS.register('onnxruntime')

            self.task_processor = build_task_processor(
                model_cfg=self.model_config,
                deploy_cfg=self.deployment_config,
                device=self.device,
            )

            # from mmpose.registry import MODELS
            # MODELS.build(self.model_config)

            if is_valid_url(self.checkpoint):
                if self.model_runtime == "onnx":
                    file_type = "end2end.onnx"
                else:
                    file_type = "end2end.engine"
                filename = (
                    f"{pathlib.Path(self.model_config.filename).stem}.{file_type}"
                )
                # checkpoint_model_file = Runner.from_cfg(self.model_config).load_checkpoint(filename=self.checkpoint, map_location="cpu")
                # checkpoint_model_file = load_checkpoint

                from pose_engine.mmlab.mmengine import hub

                # CheckpointLoader.load_checkpoint(filename=filename, map_location="cpu")
                # loaded_checkpoint = load_from_http(filename=self.checkpoint, map_location="cpu")
                checkpoint_path = hub.download_url(
                    url=self.checkpoint, file_name=filename
                )
            else:
                checkpoint_path = self.checkpoint

            self.pose_estimator = self.task_processor.build_backend_model(
                [checkpoint_path]
            )

    def _init_preprocessor(self):
        self.pre_processor = PoseEstimatorPreProcessor(
            model_config=self.model_config,
            dataset_meta=self.dataset_meta,
            device=self.device,
        )

    def _compile_model(self):
        if self.model_runtime == "pytorch" and self.compile_engine is not None:
            logger.info(
                f"Compiling pose estimator model (device: {self.device}) with '{self.compile_engine}' engine..."
            )

            compiled_backbone_and_neck_module = None
            compileable_backbone_and_neck_module = None
            compileable_backbone_and_neck_module_method = "extract_feat"  # "forward"
            if self.run_parallel or self.run_distributed:
                compileable_backbone_and_neck_module = self.pose_estimator.module
            else:
                compileable_backbone_and_neck_module = self.pose_estimator

            compiled_data_preprocessor_module = None
            compileable_data_preprocessor_module = None
            compileable_data_preprocessor_module_method = "forward"
            if (
                False
            ):  # DISABLED INTENTIONALLY: No benefit, in fact this negatively impacts throughput
                if self.run_parallel or self.run_distributed:
                    compileable_data_preprocessor_module = (
                        self.pose_estimator.module.data_preprocessor
                    )
                else:
                    compileable_data_preprocessor_module = (
                        self.pose_estimator.data_preprocessor
                    )

            # Compile the model's head, compiling the full model causes way too many graph recompilations
            compiled_head_module = None
            compileable_head_module = None
            compileable_head_module_method = "forward"
            if self.run_parallel or self.run_distributed:
                compileable_head_module = self.pose_estimator.module.head
            else:
                compileable_head_module = self.pose_estimator.head

            compiled_dcc_module = None
            compileable_dcc_module = None
            compileable_dcc_module_method = "forward_test"
            if self.run_parallel or self.run_distributed:
                compileable_dcc_module = self.pose_estimator.module.head.dcc
            else:
                compileable_dcc_module = self.pose_estimator.head.dcc

            if self.compile_engine == "inductor":
                if (
                    self.compile_engine is not None
                    and self.compile_engine == "inductor"
                ):
                    mmpose.models.heads.hybrid_heads.rtmo_head.nms_torch = (
                        torch.compile(
                            mmpose.models.heads.hybrid_heads.rtmo_head.nms_torch,
                            dynamic=True,
                            mode="max-autotune-no-cudagraphs",
                        )
                    )

                if compileable_backbone_and_neck_module is not None:
                    compiled_backbone_and_neck_module = torch.compile(
                        getattr(
                            compileable_backbone_and_neck_module,
                            compileable_backbone_and_neck_module_method,
                        ),
                        dynamic=False,
                        mode="default",  # Must be "default" to compile on AWS V100 instances
                    )

                if compileable_data_preprocessor_module is not None:
                    compiled_data_preprocessor_module = torch.compile(
                        getattr(
                            compileable_data_preprocessor_module,
                            compileable_data_preprocessor_module_method,
                        ),
                        dynamic=True,
                        mode="max-autotune-no-cudagraphs",
                    )

                if compileable_head_module is not None:
                    compiled_head_module = torch.compile(
                        getattr(
                            compileable_head_module, compileable_head_module_method
                        ),
                        dynamic=False,
                        mode="default",  # Must be "default" to compile on AWS V100 instances
                    )

                if compileable_dcc_module is not None:
                    compiled_dcc_module = torch.compile(
                        getattr(
                            compileable_dcc_module,
                            compileable_dcc_module_method,
                        ),
                        dynamic=True,
                        mode="default",  # Must be "default" to compile on AWS V100 instances
                    )
            elif self.compile_engine == "tensorrt":
                # Attempted to use torch_tensorrt backend on 2/14/2024, observed no speed up, retaining stub for future use/testing
                if compileable_backbone_and_neck_module is not None:
                    compiled_backbone_and_neck_module = torch.compile(
                        getattr(
                            compileable_backbone_and_neck_module,
                            compileable_backbone_and_neck_module_method,
                        ),
                        backend="torch_tensorrt",
                        dynamic=False,
                        options={
                            "truncate_long_and_double": True,
                            "precision": torch.half if self.use_fp16 else torch.float,
                            # "debug": True,
                            "min_block_size": 10,
                            "torch_executed_ops": {"torch.ops.aten.sort.default"},
                            "optimization_level": 1,
                            "use_python_runtime": False,
                            "device": self.device,
                        },
                    )

                if compileable_head_module is not None:
                    compiled_head_module = torch.compile(
                        getattr(
                            compileable_head_module, compileable_head_module_method
                        ),
                        backend="torch_tensorrt",
                        dynamic=False,
                        options={
                            "truncate_long_and_double": True,
                            "precision": torch.half if self.use_fp16 else torch.float,
                            # "debug": True,
                            "min_block_size": 10,
                            "torch_executed_ops": {"torch.ops.aten.sort.default"},
                            "optimization_level": 1,
                            "use_python_runtime": False,
                            "device": self.device,
                        },
                    )

            if (
                compiled_backbone_and_neck_module is not None
                and compileable_backbone_and_neck_module is not None
            ):
                setattr(
                    compileable_backbone_and_neck_module,
                    compileable_backbone_and_neck_module_method,
                    compiled_backbone_and_neck_module,
                )
                self.was_model_compiled = True

            if (
                compiled_data_preprocessor_module is not None
                and compileable_data_preprocessor_module is not None
            ):
                setattr(
                    compileable_data_preprocessor_module,
                    compileable_data_preprocessor_module_method,
                    compiled_data_preprocessor_module,
                )
                self.was_model_compiled = True

            if compiled_head_module is not None and compileable_head_module is not None:
                setattr(
                    compileable_head_module,
                    compileable_head_module_method,
                    compiled_head_module,
                )
                self.was_model_compiled = True

            if compiled_dcc_module is not None and compileable_dcc_module is not None:
                setattr(
                    compileable_dcc_module,
                    compileable_dcc_module_method,
                    compiled_dcc_module,
                )
                self.was_model_compiled = True

    def _warmup_model(self):
        logger.info(f"Warming up pose estimator model (device: {self.device})")

        with (
            torch.cuda.amp.autocast()
            if self.use_fp16 and self.model_runtime == "pytorch"
            else nullcontext()
        ):
            example_image_path = str(
                pathlib.Path(
                    pathlib.Path(sys.modules["pose_engine"].__file__).parent,
                    "assets/coco_image_example.jpg",
                )
            )
            np_example_image = cv2.imread(example_image_path)
            np_example_image = cv2.resize(np_example_image, (1296, 972))
            imgs = list(
                np.repeat(
                    np_example_image[np.newaxis, :, :, :], self.batch_size, axis=0
                )
            )

            inference_mode = None
            bboxes = None
            if self.is_topdown:
                inference_mode = "topdown"
                # imgs = list() # list(np.random.rand(self.batch_size, 972, 1296, 3))

                bboxes = []
                for _ in range(self.batch_size):
                    img_boxes = np.random.rand(
                        randrange(0, 18), 5
                    )  # For test inference, generate upto 18 bboxes (i.e. people) for each images
                    img_boxes[:, :4] = np.asarray([0.0, 0.0, 972.0, 1296.0])
                    bboxes.append(img_boxes)

                meta = list(map(lambda x: {"idx": x}, range(self.batch_size)))
            else:
                inference_mode = "onestage"
                # imgs = list(np.random.rand(self.batch_size, 972, 1296, 3))
                meta = {"idx": list(map(lambda x: x, range(self.batch_size)))}

            # Run 10 rounds of warmup
            for _ in range(10):
                pre_processed_data_list = self.pre_processor.pre_process(
                    inference_mode=inference_mode,
                    imgs=imgs,
                    bboxes=bboxes,
                    meta=meta,
                )

                pose_results = self.inference(
                    pre_processed_data_list=pre_processed_data_list
                )

                if not self.is_topdown:
                    self.apply_nearby_joints_nms(pose_results)

        logger.info(f"Finished warming up pose estimator model (device: {self.device})")

    def inference(self, pre_processed_data_list) -> List[PoseDataSample]:
        pose_results = []
        if len(pre_processed_data_list) > 0:
            # collate data list into a batch, which is a dict with following keys:
            # batch['inputs']: a list of input images
            # batch['data_samples']: a list of :obj:`PoseDataSample`
            batch_chunk_size = self.batch_size
            for chunk_ii in range(0, len(pre_processed_data_list), batch_chunk_size):
                s = time.time()
                sub_data_list = pre_processed_data_list[
                    chunk_ii : chunk_ii + batch_chunk_size
                ]

                logger.info(
                    f"Pose inference - Running pose estimation against {len(sub_data_list)} data samples (device: {self.device})..."
                )

                batch = default_collate(sub_data_list)  # pseudo_collate(sub_data_list)

                # batch = torch.utils.data.default_collate(list(map(lambda l: {'inputs': l['inputs'], 'meta_mapping': l['meta_mapping']}, sub_data_list)))
                # batch['data_samples'] = list(map(lambda l: l['data_samples'], sub_data_list))

                # Make sure the batch size is consistent. If any items are missing, duplicate the last item in the batch to fill up the batch.
                # We do this because torch.compile freaks out if it seems a different batch size. It triggers a graph recompilation that kills
                # the application silently. No idea why that happens.
                batch_fill = self.batch_size - len(sub_data_list)
                if batch_fill > 0:
                    logger.info(
                        f"Pose inference - Batch not full, filling with redundant {batch_fill} frames"
                    )
                    # Extend the img inputs
                    repeated_input = (
                        batch["inputs"][-1].unsqueeze(0).repeat(batch_fill, 1, 1, 1)
                    )
                    if isinstance(batch["inputs"], list):
                        batch["inputs"].extend([repeated_input])
                    else:
                        batch["inputs"] = torch.cat((batch["inputs"], repeated_input))

                    # Extend the data_samples list
                    repeated_data_samples = [batch["data_samples"][-1]] * batch_fill
                    batch["data_samples"].extend(repeated_data_samples)
                logger.info(
                    f"Pose inference - Preparing batch for GPU: {len(sub_data_list)} records {round(time.time() - s, 2)} seconds {round(len(sub_data_list) / (time.time() - s), 2)} records/second"
                )

                s = time.time()
                batch["inputs"] = batch["inputs"].to(
                    self.device, memory_format=torch.channels_last, non_blocking=True
                )
                logger.info(
                    f"Pose inference - Sending batch to GPU: {len(sub_data_list)} records {round(time.time() - s, 2)} seconds {round(len(sub_data_list) / (time.time() - s), 2)} records/second"
                )

                s = time.time()
                with torch.no_grad():
                    if self.model_runtime == "pytorch":
                        # s = time.time()
                        # batch['inputs'] = list(map(lambda img_tensor: img_tensor.to(self.device), batch['inputs']))
                        # logger.info(f"Pose estimator pre-processing, move image tensors to GPU timing: {round(time.time() - s, 2)}")

                        if self.was_model_compiled:
                            torch.compiler.cudagraph_mark_step_begin()

                        # for idx, _ in enumerate(batch['inputs']):
                        #     batch['inputs'][idx] = batch['inputs'][idx].to(memory_format=torch.channels_last)
                        inference_results = self.pose_estimator.test_step(batch)
                    else:
                        # input_shape = get_input_shape(self.deployment_config)
                        # model_inputs, _ = self.task_processor.create_input(
                        #     batch, input_shape
                        # )

                        # input_shape = get_input_shape(self.deployment_config)
                        # model_inputs, _ = self.task_processor.create_input(
                        #     pre_processed_data_list[0]["inputs"], input_shape
                        # )

                        # When trying to use a modile pre-compiled to execute with TensorRT I get an invalid memory access error

                        # Look at context creation in mmdeploy/backend/tensort/wrapper.py, seems like the problem is somewhere in there
                        # Other ideas: there's a post here about running two tensorrt models at the same time: https://forums.developer.nvidia.com/t/can-i-inference-two-engine-simultaneous-on-jetson-using-tensorrt/64396/3
                        # Here's one solution which is to share (or maybe separate) contexts: https://forums.developer.nvidia.com/t/unable-to-run-two-tensorrt-models-in-a-cascade-manner/145274/3b

                        # The actual error is raised in the to_numpy call in this file: /home/ben/.cache/pypoetry/virtualenvs/wf-pose-engine-_vUPQedX-py3.11/lib/python3.11/site-packages/mmpose/models/heads/base_head.py

                        # Ultimately, the solution is probably some sort of version mismatch: https://github.com/pytorch/pytorch/issues/21819
                        # I think I want:
                        # CUDA 11.8       (current 12.2)
                        # cuDNN 8.9.0     (current 8.9.7.29_cuda12)
                        # pytorch 2.1.2   (current 2.1.0)
                        # tensorRT 8.6.x  (current 8.6.1.6)
                        #
                        # As of 1/12/2024, the problem is when running the model of GPU device 1! Running the detector on device 1 causes the same issue.
                        # I might need to set CUDA_VISIBLE_DEVICES on each process that's running the respective model: https://github.com/NVIDIA/TensorRT/issues/322
                        inference_results = self.pose_estimator.test_step(batch)
                    logger.info(
                        f"Pose inference - Pose estimator inference performance (device: {self.device}): {len(sub_data_list)} records {round(time.time() - s, 2)} seconds {round(len(sub_data_list) / (time.time() - s), 2)} records/second"
                    )
                    pose_results.extend(
                        inference_results[0 : self.batch_size - batch_fill]
                    )

        # meta_mapping = data_list['meta_mapping']
        for res_idx, _result in enumerate(pose_results):
            _result.set_field(
                name="custom_metadata",
                value=pre_processed_data_list[res_idx]["meta_mapping"],
            )

        if pose_results and len(pose_results) > 0:
            for pose_result in pose_results:
                if pose_result is None or len(pose_result.pred_instances) == 0:
                    continue

                if isinstance(pose_result.pred_instances.keypoints, torch.Tensor):
                    pose_result.pred_instances.keypoints = (
                        pose_result.pred_instances.keypoints.detach().to("cpu")
                    ).numpy()
                if isinstance(pose_result.pred_instances.keypoint_scores, torch.Tensor):
                    pose_result.pred_instances.keypoint_scores = (
                        pose_result.pred_instances.keypoint_scores.detach()
                        .to("cpu")
                        .numpy()
                    )
                if hasattr(pose_result.pred_instances, "keypoints_visible"):
                    if isinstance(
                        pose_result.pred_instances.keypoints_visible,
                        torch.Tensor,
                    ):
                        pose_result.pred_instances.keypoints_visible = (
                            pose_result.pred_instances.keypoints_visible.detach()
                            .to("cpu")
                            .numpy()
                        )

                if isinstance(pose_result.pred_instances.bboxes, torch.Tensor):
                    pose_result.pred_instances.bboxes = (
                        pose_result.pred_instances.bboxes.detach().to("cpu").numpy()
                    )
                if isinstance(pose_result.pred_instances.bbox_scores, torch.Tensor):
                    pose_result.pred_instances.bbox_scores = (
                        pose_result.pred_instances.bbox_scores.detach()
                        .to("cpu")
                        .numpy()
                    )

        return pose_results

    def apply_nearby_joints_nms(self, pose_results):
        for pose_result in pose_results:
            if len(pose_result.pred_instances) == 0:
                continue

            pose_result_keypoints = pose_result.pred_instances.get("keypoints")
            pose_result_scores = pose_result.pred_instances.get("bbox_scores")
            num_keypoints = pose_result_keypoints.shape[-2]

            prediction_indicies = nearby_joints_nms(
                [
                    {
                        "keypoints": pose_result_keypoints[i],
                        "score": pose_result_scores[i],
                    }
                    for i in range(len(pose_result_keypoints))
                ],
                num_nearby_joints_thr=num_keypoints // 3,
            )
            pose_result.pred_instances = pose_result.pred_instances[prediction_indicies]

        return pose_results

    def iter_dataloader(self, loader: torch.utils.data.DataLoader):
        """Runs pose estimation against all items in the provided loader

        :param loader: The dataloader object
        :type loader: torch.utils.data.DataLoader
        :returns: a generator of tuples (poses, meta)
        :rtype: generator
        """
        self.pre_processor.start_preprocessing_process(loader=loader)

        imgs = []
        bboxes = []
        last_loop_start_time = time.time()
        for instance_batch_idx, pre_processed_data_list in enumerate(
            # for instance_batch_idx, pre_processed_data_list_of_lists in enumerate(
            # self.pre_processor.rtmo_pre_processed_images_dataloader
            self.pre_processor
        ):
            logger.info(
                f"Size of the pre-processed images queue: {self.pre_processor.size()}"
            )
            current_loop_time = time.time()
            seconds_between_loops = 0
            if last_loop_start_time is not None:
                seconds_between_loops = current_loop_time - last_loop_start_time

            navigating_locks_time = time.time()
            with self._first_inference_time.get_lock():
                if self._first_inference_time.value == -1:
                    self._first_inference_time.value = current_loop_time

            # for pre_processed_data_list in pre_processed_data_list_of_lists:
            # TODO: Need more accurate way to get Frame count
            with self._frame_count.get_lock():
                self._frame_count.value += len(pre_processed_data_list)

            with self._batch_count.get_lock():
                self._batch_count.value += 1
                global_batch_idx = instance_batch_idx

            # with self._frame_count.get_lock():
            #     self._frame_count.value += len(data_samples_list)

            with self._time_waiting.get_lock():
                self._time_waiting.value += seconds_between_loops

            global_batch_idx = instance_batch_idx

            logger.info(
                f"Pose estimation batch #{global_batch_idx} navigating locks performance (device: {self.device}): {round(time.time() - navigating_locks_time, 2)} seconds"
            )

            actual_inference_time = time.time()
            with (
                torch.cuda.amp.autocast()
                if self.use_fp16 and self.model_runtime == "pytorch"
                else nullcontext()
            ):
                pose_results = self.inference(
                    pre_processed_data_list=pre_processed_data_list
                )
                logger.info(
                    f"Pose estimation batch #{global_batch_idx} actual inference performance (device: {self.device}): {round(time.time() - actual_inference_time, 2)} seconds"
                )

                if self.is_topdown:
                    for img_idx, img in enumerate(imgs):
                        self._inference_count.value += len(bboxes[img_idx])
                else:
                    if pose_results and len(pose_results) > 0:
                        s = time.time()

                        pose_results = self.apply_nearby_joints_nms(pose_results)
                        for pose_result in pose_results:
                            if len(pose_result.pred_instances) == 0:
                                continue

                            pose_result_bboxes = pose_result.pred_instances.get(
                                "bboxes"
                            )
                            self._inference_count.value += len(pose_result_bboxes)

                        logger.info(
                            f"Pose estimation batch #{global_batch_idx} nearby_joints_nms performance (device: {self.device}): {len(pose_results)} records {round(time.time() - s, 2)} seconds {round(len(pose_results) / (time.time() - s), 2)} records/second"
                        )

            poses_to_yield = []
            # TODO: Optimize this method for extracting results and preparing them for the pose queue
            start_prepare_results_for_yield_time = time.time()
            if pose_results and len(pose_results) > 0:
                for _, pose_result in enumerate(pose_results):
                    if pose_result is None or len(pose_result.pred_instances) == 0:
                        continue

                    for pred_instance_idx in range(len(pose_result.pred_instances)):
                        pose_result_keypoints = pose_result.pred_instances.keypoints[
                            pred_instance_idx
                        ]
                        pose_result_keypoint_scores = (
                            pose_result.pred_instances.keypoint_scores[
                                pred_instance_idx
                            ]
                        )

                        if hasattr(pose_result, "keypoints_visible"):
                            pose_result_keypoint_visible = (
                                pose_result.pred_instances.keypoints_visible[
                                    pred_instance_idx
                                ]
                            )
                        else:
                            pose_result_keypoint_visible = pose_result_keypoint_scores

                        pose_result_bboxes = pose_result.pred_instances.bboxes[
                            pred_instance_idx
                        ]
                        pose_result_bbox_scores = (
                            pose_result.pred_instances.bbox_scores[pred_instance_idx]
                        )

                        pose_result_metadata = pose_result.get("custom_metadata")

                        pose_prediction = np.concatenate(
                            (
                                pose_result_keypoints,  # 0, 1 = X, Y,
                                np.full_like(
                                    np.expand_dims(
                                        pose_result_keypoint_visible, axis=1
                                    ),
                                    -1,
                                ),  # 2 = visibility - mmPose doesn't produce actual visibility values, it simply duplicates scores. For now default the value to -1.
                                np.expand_dims(
                                    pose_result_keypoint_scores, axis=1
                                ),  # 3 = confidence
                            ),
                            axis=1,
                        )
                        box_prediction = np.concatenate(
                            (
                                pose_result_bboxes,  # 0, 1, 2, 3 = X1, Y1, X2, Y2
                                np.expand_dims(
                                    pose_result_bbox_scores, axis=0
                                ),  # 5 = confidence
                            )
                        )
                        poses_to_yield.append(
                            (pose_prediction, box_prediction, pose_result_metadata)
                        )

                logger.info(
                    f"Pose estimation batch #{global_batch_idx} prepare results for yield time performance (device: {self.device}): {round(time.time() - start_prepare_results_for_yield_time, 2)} seconds"
                )

                start_yield_results_time = time.time()
                yield poses_to_yield
                logger.info(
                    f"Pose estimation batch #{global_batch_idx} yield time performance (device: {self.device}): {round(time.time() - start_yield_results_time, 2)} seconds"
                )

                logger.info(
                    f"Completed pose estimation batch #{global_batch_idx} overall performance (device: {self.device}) - Includes {len(pre_processed_data_list)} frames - {round(time.time() - current_loop_time, 2)} seconds - {round(len(pre_processed_data_list) / (time.time() - current_loop_time), 2)} FPS"
                )

            last_loop_start_time = time.time()

        self._stop_time.value = time.time()

    def stop_pre_processor(self):
        if self.pre_processor is not None:
            self.pre_processor.stop()

    def __del__(self):
        if hasattr(self, "pose_estimator") and self.pose_estimator is not None:
            del self.pose_estimator
