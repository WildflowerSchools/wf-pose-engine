from mmpose.apis import init_model as init_pose_estimator

from pose_engine.log import logger


class PoseEstimator:
    def __init__(
        self,
        config: str,
        checkpoint: str
    ):
        logger.info("Initializing pose estimator...")

        pose_config = "../configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
        pose_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
        pose_estimator = init_pose_estimator(
            config=pose_config,
            checkpoint=pose_checkpoint,
            device="cuda:1"
        )

        self.pose_estimator = pose_estimator