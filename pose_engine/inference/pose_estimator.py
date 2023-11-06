from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
import numpy as np
import torch.utils.data

from pose_engine.log import logger


class PoseEstimator:
    def __init__(
        self,
        preset_model: str = None,
        config: str = None,
        checkpoint: str = None,
        device: str = "cuda:1",
    ):
        logger.info("Initializing pose estimator...")

        if preset_model is None and (config is None or checkpoint is None):
            raise ValueError(
                "Pose estimator must be initialized with a default_model setting or by providing a config + checkpoint pair"
            )

        if preset_model == "small":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-s_8xb256-420e_body8-256x192.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth"
        elif preset_model == "medium_256":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
        elif preset_model == "medium_384":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-384x288.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth"
        elif preset_model == "large_256":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"
        elif preset_model == "large_384":
            config = "./configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-384x288.py"
            checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth"

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
