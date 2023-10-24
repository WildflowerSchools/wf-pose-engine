from mmdet.apis import inference_detector, init_detector
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline

from . import inference
from .log import logger


def run():
    

    detector = inference.Detector(
        config="../configs/mmdet/rtmdet_m_640-8xb32_coco-person.py",
        checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    )

    pose_estimator = inference.PoseEstimator(
        config="../configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py",
        checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
    )

    video_path = "../input/test_video/dahlia_example.mp4"


    # TODO: Stream video to dataloader

    # TODO: Process video for bboxes
    # TODO: Append bboxes to dataloader

    # TODO: Process bboxes for poses
    # TODO: Append poses to dataloader

    # TODO: Output poses to XXXX
