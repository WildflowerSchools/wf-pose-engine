from datetime import datetime
from typing import Optional, Union

from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmdet.apis.inference import ImagesType
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
from mmpose.utils import adapt_mmdet_pipeline
import numpy as np
import torch
from torch import nn
import torch.utils.data
import torchvision.ops

from pose_engine.log import logger


class Detector:
    def __init__(
        self,
        config: str = None,
        checkpoint: str = None,
        device="cpu",
        nms_iou_threshold=0.3,
        bbox_threshold=0.3,
    ):
        logger.info("Initializing object detector...")

        if config is None or checkpoint is None:
            raise ValueError(
                "Detector must be initialized by providing a config + checkpoint pair"
            )

        self.config = config
        self.checkpoint = checkpoint

        detector = init_detector(
            config=self.config, checkpoint=self.checkpoint, device=device
        )
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        detector.share_memory()

        self.detector = detector
        self.nms_iou_threshold = nms_iou_threshold
        self.bbox_threshold = bbox_threshold

    def inference_detector(
        self,
        model: nn.Module,
        imgs: ImagesType,
        test_pipeline: Optional[Compose] = None,
        text_prompt: Optional[str] = None,
        custom_entities: bool = False,
    ) -> Union[DetDataSample, SampleList]:
        """Custom inference image(s) with the detector. This allows true batch processing.

        Args:
            model (nn.Module): The loaded detector.
            imgs (str, ndarray, Sequence[str/ndarray]):
            Either image files or loaded images.
            test_pipeline (:obj:`Compose`): Test pipeline.

        Returns:
            :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
            If imgs is a list or tuple, the same length list type results
            will be returned, otherwise return the detection results directly.
        """

        if isinstance(imgs, (list, tuple)):
            is_batch = True
        else:
            imgs = [imgs]
            is_batch = False

        cfg = model.cfg

        if test_pipeline is None:
            cfg = cfg.copy()
            test_pipeline = get_test_pipeline_cfg(cfg)
            if isinstance(imgs[0], np.ndarray):
                # Calling this method across libraries will result
                # in module unregistered error if not prefixed with mmdet.
                test_pipeline[0].type = "mmdet.LoadImageFromNDArray"

            test_pipeline = Compose(test_pipeline)

        if model.data_preprocessor.device.type == "cpu":
            for m in model.modules():
                assert not isinstance(
                    m, RoIPool
                ), "CPU inference with RoIPool is not supported currently."

        all_inputs = []
        all_data_samples = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # TODO: remove img_id.
                data_ = {"img": img, "img_id": 0}
            else:
                # TODO: remove img_id.
                data_ = {"img_path": img, "img_id": 0}

            if text_prompt:
                data_["text"] = text_prompt
                data_["custom_entities"] = custom_entities

            # build the data pipeline
            data_ = test_pipeline(data_)
            all_inputs.append(data_["inputs"])
            all_data_samples.append(data_["data_samples"])

        data_ = {"inputs": all_inputs, "data_samples": all_data_samples}

        # forward the model
        with torch.no_grad():
            result_list = model.test_step(data_)

        if not is_batch:
            return result_list[0]

        return result_list

    def iter_dataloader(self, loader: torch.utils.data.DataLoader):
        """Runs detection against all items in the provided loader

        :param loader: The dataloader object
        :type loader: torch.utils.data.DataLoader
        :returns: a generator of tuples (bboxes, frame, meta)
        :rtype: generator
        """
        logger.info("Running detector against dataloader object...")

        total_frame_count = 0
        for batch_idx, (frames, meta) in enumerate(loader):
            logger.info(
                f"Processing detector batch #{batch_idx} - Includes {len(frames)} frames"
            )
            meta_as_list_of_dicts = []
            for idx in range(len(frames)):
                meta_list_item = {}
                for key in meta.keys():
                    meta_list_item[key] = meta[key][idx]

                meta_as_list_of_dicts.append(meta_list_item)

            total_frame_count += len(frames)

            list_np_imgs = list(
                frames.cpu().detach().numpy()
            )  # Annoying we have to move frames/images to the CPU to run detector
            det_results = self.inference_detector(
                model=self.detector, imgs=list_np_imgs
            )
            for idx, det_result in enumerate(det_results):
                frame = frames[idx]
                meta = meta_as_list_of_dicts[idx]

                bboxes_and_scores = torch.concatenate(
                    (
                        det_result.pred_instances.bboxes,
                        det_result.pred_instances.labels[:, None],
                        det_result.pred_instances.scores[:, None],
                    ),
                    axis=1,
                )

                # Filter by label (person == 0) and bbox_threshold
                bboxes_and_scores = bboxes_and_scores[
                    torch.logical_and(
                        bboxes_and_scores[:, 4] == 0,
                        bboxes_and_scores[:, 5] > self.bbox_threshold,
                    )
                ]

                # Filter by NMS iou_threshold
                nms_filter_idxs = torchvision.ops.nms(
                    boxes=bboxes_and_scores[:, :4],
                    scores=bboxes_and_scores[:, 5],
                    iou_threshold=self.nms_iou_threshold,
                )
                retained_bboxes = bboxes_and_scores[nms_filter_idxs]

                # Remove label id but retain score
                retained_bboxes = torch.concatenate(
                    (
                        retained_bboxes[:, :4],
                        retained_bboxes[:, 5, None],
                    ),
                    axis=1,
                )  # [x, y, [x|w], [y|h], score]

                # if meta["camera_device_id"] == "c9f013f9-3100-4c2f-9762-c1fb35b445a0":
                #     timestamp = datetime.utcfromtimestamp(float(meta["frame_timestamp"]))
                #     logger.info(f"Found {len(retained_bboxes)} boxes at {timestamp}")

                yield retained_bboxes, frame, meta

    def __del__(self):
        del self.detector
