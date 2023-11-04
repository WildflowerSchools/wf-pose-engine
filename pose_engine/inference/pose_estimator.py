from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
import numpy as np
import torch.utils.data

from pose_engine.log import logger


class PoseEstimator:
    def __init__(self, config: str, checkpoint: str, device: str = "cuda:1"):
        logger.info("Initializing pose estimator...")

        pose_config = config
        pose_checkpoint = checkpoint
        pose_estimator = init_pose_estimator(
            config=pose_config, checkpoint=pose_checkpoint, device=device
        )

        pose_estimator.share_memory()
        self.pose_estimator = pose_estimator

    def iter_dataloader(self, loader: torch.utils.data.DataLoader):
        """Runs pose estimation against all items in the provided loader

        :param loader: The dataloader object
        :type loader: torch.utils.data.DataLoader
        :returns: a generator of tuples (poses, meta)
        :rtype: generator
        """
        logger.info("Running pose estimation against dataloader object...")

        total_frame_count = 0
        for batch_idx, (bboxes, frames, meta) in enumerate(loader):
            total_frame_count += len(frames)

            logger.info(
                f"Processing pose estimation batch #{batch_idx} - Includes {len(frames)} frames"
            )
            for idx, img in enumerate(frames):
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()

                if isinstance(bboxes[idx], torch.Tensor):
                    bboxes[idx] = bboxes[idx].detach().cpu().numpy()

                # TODO: Update inference_topdown to work with Tensors
                pose_results = inference_topdown(self.pose_estimator, img, bboxes[idx])
                # data_samples = merge_data_samples(pose_results)

                yield pose_results
