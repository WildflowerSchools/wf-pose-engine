from contextlib import nullcontext

import time
from random import randrange
from typing import List, Optional, Union

import numpy as np
import torch
import torch._dynamo as torchdynamo
import torch.multiprocessing as mp
import torch.utils.data

import torch_tensorrt  # DO NOT DELETE

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmengine.dataset import Compose, default_collate
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.registry import init_default_scope
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis.inference import dataset_meta_from_config
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy
import mmpose.models.heads.hybrid_heads.rtmo_head
import mmpose.evaluation.functional
import mmpose.evaluation.functional.nms
from mmpose.evaluation.functional import nearby_joints_nms

from PIL import Image

from pose_engine.mmpose.transforms import BatchBottomupResize
from pose_engine.torch.mmlab_compatible_data_parallel import MMLabCompatibleDataParallel
from pose_engine.loaders import VideoFramesDataLoader, BoundingBoxesDataLoader
from pose_engine.log import logger

from pose_engine.mmpose.misc import nms_torch

mmpose.models.heads.hybrid_heads.rtmo_head.nms_torch = nms_torch


class PoseEstimator:
    def __init__(
        self,
        model_config_path: str = None,
        checkpoint: str = None,
        deployment_config_path: str = None,
        device: str = "cuda:1",
        batch_size: int = 75,
        use_fp16: bool = False,
        run_parallel: bool = False,
        run_distributed: bool = False,
        compile_engine: Optional[str] = None,
    ):
        # torch._logging.set_logs(recompiles=True, dynamo=logging.DEBUG) #dynamo=logging.DEBUG
        # torch._dynamo.config.suppress_errors = True
        torchdynamo.config.cache_size_limit = 512
        # torch.compiler.disable(fn=bbox_overlaps, recursive=True)

        if model_config_path is None or checkpoint is None:
            raise ValueError(
                "Pose estimator must be initialized by providing a config + checkpoint pair"
            )

        self.checkpoint = checkpoint

        self.model_config_path = model_config_path
        self.model_config = load_config(self.model_config_path)[0]
        self.dataset_meta = dataset_meta_from_config(
            self.model_config, dataset_mode="train"
        )

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
        self._frame_count = mp.Value(
            "i", 0
        )  # Each frame contains multiple boxes, so we track frames separately from inferences
        self._inference_count = mp.Value("i", 0)
        self._start_time = mp.Value("d", -1.0)
        self._stop_time = mp.Value("d", -1.0)
        self._first_inference_time = mp.Value("d", -1.0)
        self._time_waiting = mp.Value("d", 0)

        self.pose_estimator = None
        self.task_processor = None
        self.pipeline = None
        self.batch_bottomup_resize_transform = None

        self._init_model()
        self._init_pipeline()

        if self.use_fp16:
            self.pose_estimator = self.pose_estimator.half()
        self.pose_estimator = self.pose_estimator.to(memory_format=torch.channels_last)

        self._compile_model()
        self._warmup_model()

        logger.info(f"Finished initializing pose estimator (device: {self.device})")

    @property
    def frame_count(self) -> int:
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

    def _init_model(
        self
    ):
        self.using_tensort = self.deployment_config_path is not None
        if not self.using_tensort:  # Standard PYTorch
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

        else:  # TensorRT
            self.task_processor = build_task_processor(
                model_cfg=self.model_config,
                deploy_cfg=self.deployment_config,
                device=self.device,
            )
            self.pose_estimator = self.task_processor.build_backend_model(
                [self.checkpoint]
            )

    def _init_pipeline(
        self
    ):
        self.pipeline = self.model_config.test_dataloader.dataset.pipeline

        # Modify the pipeline by slicing out the default BottomupResize method
        # BottomupResize does not batch process. Below, we substitute in a
        # torchvision resize method that can take advantage of batching
        modified_pipeline = []
        for stage in self.pipeline:
            if stage["type"] == "BottomupResize":
                self.batch_bottomup_resize_transform = BatchBottomupResize(**stage)
            else:
                modified_pipeline.append(stage)

        self.pipeline = Compose(modified_pipeline)
    
    def _compile_model(
        self
    ):
        if self.compile_engine is not None:
            logger.info(
                f"Compiling pose estimator model (device: {self.device}) with '{self.compile_engine}' engine..."
            )

            compileable_backbone_and_neck_module = None
            compileable_backbone_and_neck_module_method = "extract_feat"  # "forward"
            if self.run_parallel or self.run_distributed:
                compileable_backbone_and_neck_module = self.pose_estimator.module
            else:
                compileable_backbone_and_neck_module = self.pose_estimator

            # Compile the model's head, compiling the full model causes way too many graph recompilations
            compileable_head_module = None
            compileable_head_module_method = "forward"
            if self.run_parallel or self.run_distributed:
                compileable_head_module = self.pose_estimator.module.head
            else:
                compileable_head_module = self.pose_estimator.head
            # compileable_head_module = None

            if self.compile_engine == "inductor":
                if self.compile_engine is not None and self.compile_engine == "inductor":
                    mmpose.models.heads.hybrid_heads.rtmo_head.nms_torch = torch.compile(
                        mmpose.models.heads.hybrid_heads.rtmo_head.nms_torch,
                        dynamic=True,
                        # DO NOT ENABLE max-autotune mode, it enables triton.cudagraphs which leads to an Exception. Must set max_autotune in inductor options.
                        #   ERROR: These live storage data ptrs are in the cudagraph pool but not accounted for as an output of cudagraph trees:
                        # mode="max-autotune",
                        options={
                            "max_autotune": True,
                            "triton.cudagraphs": False,  # This must be set to False
                        },
                    )

                compiled_backbone_and_neck_module = torch.compile(
                    getattr(
                        compileable_backbone_and_neck_module,
                        compileable_backbone_and_neck_module_method,
                    ),
                    dynamic=False,
                    mode="default",
                )

                if compileable_head_module is not None:
                    compiled_head_model = torch.compile(
                        getattr(
                            compileable_head_module, compileable_head_module_method
                        ),
                        dynamic=False,
                        mode="default",
                    )
            elif self.compile_engine == "tensorrt":
                # Attempted to use torch_tensorrt backend on 2/14/2024, observed no speed up, retaining stub for future use/testing
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

                compiled_head_model = torch.compile(
                    getattr(compileable_head_module, compileable_head_module_method),
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

            if compileable_backbone_and_neck_module is not None:
                setattr(
                    compileable_backbone_and_neck_module,
                    compileable_backbone_and_neck_module_method,
                    compiled_backbone_and_neck_module,
                )
                self.was_model_compiled = True

            if compileable_head_module is not None:
                setattr(
                    compileable_head_module,
                    compileable_head_module_method,
                    compiled_head_model,
                )
                self.was_model_compiled = True

    def _warmup_model(self):
        logger.info(
            f"Warming up pose estimator model (device: {self.device})"
        )

        with (
                torch.cuda.amp.autocast()
                if self.use_fp16 and not self.using_tensort
                else nullcontext()
        ):
            inference_mode = None
            bboxes = None
            if self.model_config["data_mode"] == "topdown":
                inference_mode = "topdown"
                imgs = list(np.random.rand(self.batch_size, 972, 1296, 3))

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
                imgs = list(np.random.rand(self.batch_size, 972, 1296, 3))
                meta = {"idx": list(map(lambda x: x, range(self.batch_size)))}

            self.inference(
                inference_mode=inference_mode,
                imgs=imgs,
                bboxes=bboxes,
                meta=meta,
            )

        logger.info(
            f"Finished warming up pose estimator model (device: {self.device})"
        )
        
    def inference(
        self,
        imgs: Union[list[np.ndarray], list[str]],
        inference_mode: str = "topdown",
        bboxes: Optional[Union[List[List], List[np.ndarray]]] = None,
        meta: List[dict] = None,
        bbox_format: str = "xyxy",
    ) -> List[PoseDataSample]:
        """Inference image with a top-down pose estimator.

        Args:
            model (nn.Module): The top-down pose estimator
            img (np.ndarray | str): The loaded image or image file to inference
            bboxes (np.ndarray, optional): The bboxes in shape (N, 5), each row
                represents a bbox. If not given, the entire image will be regarded
                as a single bbox area. Defaults to ``None``
            bbox_format (str): The bbox format indicator. Options are ``'xywh'``
                and ``'xyxy'``. Defaults to ``'xyxy'``

        Returns:
            List[:obj:`PoseDataSample`]: The inference results. Specifically, the
            predicted keypoints and scores are saved at
            ``data_sample.pred_instances.keypoints`` and
            ``data_sample.pred_instances.keypoint_scores``.
        """
        scope = self.model_config.get("default_scope", "mmpose")
        if scope is not None:
            init_default_scope(scope)

        # construct batch data samples
        data_list = []
        meta_mapping = []
        total_pre_processing_time = 0

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
                        data_info = {"img_path": img}
                    else:
                        data_info = {"img": img}
                    data_info["bbox"] = bbox[None, :4]  # shape (1, 4)
                    data_info["bbox_score"] = bbox[None, 4]  # shape (1,)
            else:
                meta_item = {}
                for key in meta.keys():
                    meta_item[key] = meta[key][img_idx]

                meta_mapping.append(meta_item)
                data_info = {"img": img}

            data_info.update(self.dataset_meta)
            s = time.time()
            processed_pipeline_data = self.pipeline(data_info)
            processed_pipeline_data["inputs"] = processed_pipeline_data["inputs"]
            total_pre_processing_time += time.time() - s

            data_list.append(processed_pipeline_data)

        # Batch resizing consumes an outsized chunk of CUDA memory, so split this step across two passes
        data_list_chunk_size = (len(data_list) // 2) + 1
        for ii in range(0, len(data_list), data_list_chunk_size):
            data_list[ii : ii + data_list_chunk_size] = (
                self.batch_bottomup_resize_transform.transform(
                    data_list=data_list[ii : ii + data_list_chunk_size],
                    device=self.device,
                )
            )

        if total_pre_processing_time == 0:
            records_per_second = "N/A"
        else:
            records_per_second = round(len(data_list) / total_pre_processing_time, 2)

        logger.info(
            f"Pose estimator data pipeline pre-processing performance (device: {self.device}): {len(data_list)} records {round(total_pre_processing_time, 3)} seconds {records_per_second} records/second"
        )

        results = []
        if len(data_list) > 0:
            # collate data list into a batch, which is a dict with following keys:
            # batch['inputs']: a list of input images
            # batch['data_samples']: a list of :obj:`PoseDataSample`
            batch_chunk_size = self.batch_size
            for chunk_ii in range(0, len(data_list), batch_chunk_size):
                sub_data_list = data_list[chunk_ii : chunk_ii + batch_chunk_size]

                logger.debug(
                    f"Running pose estimation against {len(sub_data_list)} data samples (device: {self.device})..."
                )

                batch = default_collate(sub_data_list)  # pseudo_collate(sub_data_list)

                # Make sure the batch size is consistent. If any items are missing, duplicate the last item in the batch to fill up the batch.
                # We do this because torch.compile freaks out if it seems a different batch size. It triggers a graph recompilation that kills
                # the application silently. No idea why that happens.
                batch_fill = self.batch_size - len(sub_data_list)
                if batch_fill > 0:
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

                batch["inputs"] = (
                    batch["inputs"]
                    .to(memory_format=torch.channels_last)
                    .to(self.device)
                )

                s = time.time()
                with torch.no_grad():
                    if self.using_tensort:
                        input_shape = get_input_shape(self.deployment_config)
                        model_inputs, _ = self.task_processor.create_input(
                            imgs[0], input_shape
                        )
                        # When trying to use TensorRT I get an invalid memory access error

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
                        inference_results = self.pose_estimator.test_step(model_inputs)
                    else:
                        # s = time.time()
                        # batch['inputs'] = list(map(lambda img_tensor: img_tensor.to(self.device), batch['inputs']))
                        # logger.info(f"Pose estimator pre-processing, move image tensors to GPU timing: {round(time.time() - s, 2)}")

                        if self.was_model_compiled:
                            torch.compiler.cudagraph_mark_step_begin()

                        # for idx, _ in enumerate(batch['inputs']):
                        #     batch['inputs'][idx] = batch['inputs'][idx].to(memory_format=torch.channels_last)
                        inference_results = self.pose_estimator.test_step(batch)

                    logger.info(
                        f"Pose estimator inference performance (device: {self.device}): {len(sub_data_list)} records {round(time.time() - s, 2)} seconds {round(len(sub_data_list) / (time.time() - s), 2)} records/second"
                    )
                    results.extend(inference_results[0 : self.batch_size - batch_fill])

        for res_idx, _result in enumerate(results):
            _result.set_field(name="custom_metadata", value=meta_mapping[res_idx])

        return results

    def iter_dataloader(self, loader: torch.utils.data.DataLoader):
        """Runs pose estimation against all items in the provided loader

        :param loader: The dataloader object
        :type loader: torch.utils.data.DataLoader
        :returns: a generator of tuples (poses, meta)
        :rtype: generator
        """

        is_topdown = False

        last_loop_start_time = None
        self._start_time.value = time.time()
        for instance_batch_idx, data in enumerate(loader):
            current_loop_time = time.time()
            seconds_between_loops = 0
            if last_loop_start_time is not None:
                seconds_between_loops = current_loop_time - last_loop_start_time

            global_batch_idx = instance_batch_idx
            with self._batch_count.get_lock():
                self._batch_count.value += 1
                global_batch_idx = instance_batch_idx

            with self._time_waiting.get_lock():
                self._time_waiting.value += seconds_between_loops

            with self._first_inference_time.get_lock():
                if self._first_inference_time.value == -1:
                    self._first_inference_time.value = current_loop_time

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

            try:
                logger.info(
                    f"Processing pose estimation batch #{global_batch_idx} (device: {self.device}) - Includes {len(frames)} frames - Seconds since last batch {round(seconds_between_loops, 3)}"
                )

                self._frame_count.value += len(frames)

                imgs = []
                for img_idx, img in enumerate(frames):
                    if isinstance(img, torch.Tensor):
                        # img = img.detach().cpu().numpy()
                        img = img.numpy()

                    imgs.append(img)

                    if bboxes is not None and isinstance(bboxes[img_idx], torch.Tensor):
                        # bboxes[img_idx] = bboxes[img_idx].detach().cpu().numpy()
                        bboxes[img_idx] = bboxes[img_idx].numpy()

                # TODO: Update inference_topdown to work with Tensors

                with (
                    torch.cuda.amp.autocast()
                    if self.use_fp16 and not self.using_tensort
                    else nullcontext()
                ):
                    inference_mode = "topdown" if is_topdown else "onestage"
                    pose_results = self.inference(
                        inference_mode=inference_mode,
                        imgs=imgs,
                        bboxes=bboxes,
                        meta=meta,
                    )

                if is_topdown:
                    for img_idx, img in enumerate(imgs):
                        self._inference_count.value += len(bboxes[img_idx])
                else:
                    if pose_results and len(pose_results) > 0:
                        s = time.time()

                        for pose_result in pose_results:
                            if len(pose_result.pred_instances) == 0:
                                continue

                            pose_result_keypoints = pose_result.pred_instances.get(
                                "keypoints"
                            )
                            pose_result_scores = pose_result.pred_instances.get(
                                "bbox_scores"
                            )
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
                            pose_result.pred_instances = pose_result.pred_instances[
                                prediction_indicies
                            ]

                            pose_result_bboxes = pose_result.pred_instances.get(
                                "bboxes"
                            )
                            self._inference_count.value += len(pose_result_bboxes)

                        logger.info(
                            f"Pose estimation batch #{global_batch_idx} nearby_joints_nms performance (device: {self.device}): {len(pose_results)} records {round(time.time() - s, 2)} seconds {round(len(pose_results) / (time.time() - s), 2)} records/second"
                        )

                # data_samples = merge_data_samples(pose_results)

                poses_to_yield = []
                # TODO: Optimize this method for extracting results and preparing them for the pose queue
                start_prepare_results_for_yield_time = time.time()
                if pose_results and len(pose_results) > 0:
                    for _, pose_result in enumerate(pose_results):
                        if pose_result is None or len(pose_result.pred_instances) == 0:
                            continue

                        for pred_instance_idx in range(len(pose_result.pred_instances)):
                            pose_result_keypoints = (
                                pose_result.pred_instances.keypoints[pred_instance_idx]
                            )
                            pose_result_keypoint_visible = (
                                pose_result.pred_instances.keypoints_visible[
                                    pred_instance_idx
                                ]
                            )
                            pose_result_keypoint_scores = (
                                pose_result.pred_instances.keypoint_scores[
                                    pred_instance_idx
                                ]
                            )
                            pose_result_bboxes = pose_result.pred_instances.bboxes[
                                pred_instance_idx
                            ]
                            pose_result_bbox_scores = (
                                pose_result.pred_instances.bbox_scores[
                                    pred_instance_idx
                                ]
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
                    f"Completed pose estimation batch #{global_batch_idx} overall performance (device: {self.device}) - Includes {len(frames)} frames - {round(time.time() - current_loop_time, 2)} seconds - {round(len(frames) / (time.time() - current_loop_time), 2)} FPS"
                )
            finally:
                del bboxes
                del frames
                del meta

            last_loop_start_time = current_loop_time

        self._stop_time.value = time.time()

    def __del__(self):
        if self.pose_estimator is not None:
            del self.pose_estimator
