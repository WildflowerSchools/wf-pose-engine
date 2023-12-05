import logging
import time

import cv2
import torch

import mmcv
from mmengine.structures import InstanceData, PixelData
from mmpose.structures import PoseDataSample
from mmpose.structures import merge_data_samples, split_instances
from mmpose.registry import VISUALIZERS




VIDEO_OUTPUT_FILE = "./overlay.mp4"

pose_meta = dict(img_shape=(800, 1216),
                    crop_size=(256, 192),
                    heatmap_size=(64, 48))

gt_instances = InstanceData()
gt_instances.bboxes = torch.rand((1, 4))
gt_instances.keypoints = torch.rand((1, 17, 2))
gt_instances.keypoints_visible = torch.rand((1, 17, 1))

gt_fields = PixelData()
gt_fields.heatmaps = torch.rand((17, 64, 48))

data_sample = PoseDataSample(gt_instances=gt_instances,
                                gt_fields=gt_fields,
                                metainfo=pose_meta)
assert 'img_shape' in data_sample
len(data_sample.gt_intances)


pose_results = ["<<POSEDATA>>"]
data_samples = merge_data_samples(pose_results)


video_writer = None
pred_instances_list = []
frame_idx = 0

visualizer = VISUALIZERS.build({
    "type": 'PoseLocalVisualizer',
    "vis_backends": {
        "type": 'LocalVisBackend'
    },
    "name": 'visualizer'
})
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(
    {},
    skeleton_style="mmpose")



# output videos
frame_vis = visualizer.get_image()

if video_writer is None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # the size of the image with visualization may vary
    # depending on the presence of heatmaps
    video_writer = cv2.VideoWriter(
        VIDEO_OUTPUT_FILE,
        fourcc,
        25,  # saved fps
        (frame_vis.shape[1], frame_vis.shape[0]))

    video_writer.write(mmcv.rgb2bgr(frame_vis))

    
if video_writer:
    video_writer.release()

cap.release()
