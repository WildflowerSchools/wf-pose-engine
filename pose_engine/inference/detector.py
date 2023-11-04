import pathlib

from mmdet.apis import inference_detector, init_detector
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
import numpy as np
import torch
import torch.utils.data
import torchvision.ops

from pose_engine.log import logger


class Detector:
    def __init__(
        self,
        config: str,
        checkpoint: str,
        device="cpu",
        nms_iou_threshold=0.3,
        bbox_threshold=0.3,
    ):
        logger.info("Initializing object detector...")

        detector_config = config
        detector_checkpoint = checkpoint
        detector = init_detector(
            config=detector_config, checkpoint=detector_checkpoint, device=device
        )
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        detector.share_memory()

        self.detector = detector
        self.nms_iou_threshold = nms_iou_threshold
        self.bbox_threshold = bbox_threshold

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
            det_results = inference_detector(model=self.detector, imgs=list_np_imgs)
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
                retained_bboxes = bboxes_and_scores[nms_filter_idxs, :4]

                yield retained_bboxes, frame, meta
