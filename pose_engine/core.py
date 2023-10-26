import numpy as np

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline

from . import inference
from .log import logger
from .pipeline.video_frames_dataloader import VideoFramesDataLoader
from .pipeline.video_frames_dataset import VideoFramesDataset


def run():
    detector = inference.Detector(
        config="./configs/mmdet/rtmdet_m_640-8xb32_coco-person.py",
        checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
    )

    dataset = VideoFramesDataset(
        video_paths=[
            "./input/test_video/output000.mp4",
        ],
        wait_for_video_files=False,
    )

    loader = VideoFramesDataLoader(
        dataset, device="cuda:0", shuffle=False, num_workers=0, batch_size=4
    )

    logger.info("Running detector")
    for batch_idx, (frames, meta) in enumerate(loader):
        logger.info(f"Processing batch #{batch_idx}")
        np_imgs = frames.cpu().detach().numpy()
        list_np_imgs = []
        for np_img in np_imgs:
            list_np_imgs.append(np_img)

        pred_instances = []
        det_results = inference_detector(model=detector.detector, imgs=list_np_imgs)
        for det_result in det_results:
            pred_instances.append(det_result.pred_instances.cpu().numpy())

        all_bboxes = []
        for pred_instance in pred_instances:
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
            )
            bboxes = bboxes[
                np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)
            ]
            bboxes = bboxes[nms(bboxes, 0.3), :4]
            all_bboxes.append(bboxes)

    logger.info("Finished running detector")
    # pose_estimator = inference.PoseEstimator(
    #     config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py",
    #     checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
    # )

    # video_path = "../input/test_video/dahlia_example.mp4"

    # TODO: Stream video to dataloader

    # TODO: Process video for bboxes
    # TODO: Append bboxes to dataloader

    # TODO: Process bboxes for poses
    # TODO: Append poses to dataloader

    # TODO: Output poses to XXXX
