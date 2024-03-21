from typing import List, Optional

import numba
import numpy as np
import torch
from torch import Tensor
import torchvision

from mmpose.structures.bbox import bbox_overlaps


def nms_torch(
    bboxes, scores, threshold=0.65, iou_calculator=bbox_overlaps, return_group=False
):
    return torchvision.ops.nms(boxes=bboxes, scores=scores, iou_threshold=threshold)


def nms_torch_optimized(
    bboxes, scores, threshold=0.65, iou_calculator=bbox_overlaps, return_group=False
):

    # torch_start = torch.cuda.Event(enable_timing=True)
    # torch_end = torch.cuda.Event(enable_timing=True)

    # import time
    # s = time.time()
    # torch.cuda.synchronize()
    # print(f"nms_torch: synchronize timing: {round(time.time() - s, 3)} seconds")

    # torch_start.record()
    _, indices = scores.sort(descending=True)
    groups = []
    bboxes = bboxes[indices]

    all_ious = iou_calculator(bboxes, bboxes, is_aligned=False).triu(diagonal=1)

    while len(indices):
        idx, indices = indices[0], indices[1:]
        bbox_ious = all_ious[idx][indices]

        close_indices = torch.where(bbox_ious > threshold)
        if len(indices) == 0:
            groups.append(idx[None])
        else:
            keep_indices = torch.ones_like(indices, dtype=torch.bool)
            keep_indices[close_indices] = 0
            groups.append(torch.cat((idx[None], indices[close_indices])))
            indices = indices[keep_indices]

        # torch.cuda.synchronize()

    # torch_end.record()
    # torch.cuda.synchronize()
    # print(f"nms_torch: nms loop timing: {round(torch_start.elapsed_time(torch_end) / 1000, 3)} seconds")

    if return_group:
        return groups
    else:
        # torch_start.record()
        cat = torch.cat([g[:1] for g in groups])
        # torch_end.record()
        # torch.cuda.synchronize()
        # print(f"nms_torch: nms cat/prep timing: {round(torch_start.elapsed_time(torch_end) / 1000, 3)} seconds")
        return cat


def nms_torch_deprecated(
    bboxes: Tensor,
    scores: Tensor,
    threshold: float = 0.65,
    iou_calculator=bbox_overlaps,
    return_group: bool = False,
):
    """Perform Non-Maximum Suppression (NMS) on a set of bounding boxes using
    their corresponding scores.

    Args:

        bboxes (Tensor): list of bounding boxes (each containing 4 elements
            for x1, y1, x2, y2).
        scores (Tensor): scores associated with each bounding box.
        threshold (float): IoU threshold to determine overlap.
        iou_calculator (function): method to calculate IoU.
        return_group (bool): if True, returns groups of overlapping bounding
            boxes, otherwise returns the main bounding boxes.
    """

    # torch_start = torch.cuda.Event(enable_timing=True)
    # torch_end = torch.cuda.Event(enable_timing=True)

    # import time
    # s = time.time()
    # torch.cuda.synchronize()
    # print(f"nms_torch: synchronize timing: {round(time.time() - s, 3)} seconds")

    # print(f"nms_torch: bboxes size: {bboxes.shape}")

    _, indices = scores.sort(descending=True)
    groups = []

    # torch_start.record()
    while len(indices):
        idx, indices = indices[0], indices[1:]
        bbox = bboxes[idx]
        ious = iou_calculator(bbox, bboxes[indices])

        if ious.ndim == 1:
            ious = torch.unsqueeze(ious, 0)

        close_indices = torch.where(ious > threshold)[1]
        if len(indices) == 0:
            groups.append(idx[None])
        else:
            keep_indices = torch.ones_like(indices, dtype=torch.bool)
            keep_indices[close_indices] = 0
            groups.append(torch.cat((idx[None], indices[close_indices])))
            indices = indices[keep_indices]

    # torch_end.record()
    # torch.cuda.synchronize()
    # print(f"nms_torch: nms loop timing: {round(torch_start.elapsed_time(torch_end) / 1000, 3)} seconds")

    if return_group:
        return groups
    else:
        # torch_start.record()
        cat = torch.cat([g[:1] for g in groups])
        # torch_end.record()
        # torch.cuda.synchronize()
        # print(f"nms_torch: nms cat/prep timing: {round(torch_start.elapsed_time(torch_end) / 1000, 3)} seconds")
        return cat


@numba.njit
def nb_pose_area_step(_pose_area, _num_people):
    _pose_area = np.sqrt(np.power(_pose_area, 2).sum(axis=1))
    _pose_area = _pose_area.reshape(_num_people, 1, 1)
    return _pose_area


@numba.njit
def nb_close_instance(_pose_area, _dist_thr, _kpts, _num_nearby_joints_thr):
    close_dist_thr = _pose_area * _dist_thr

    # count nearby joints between instances
    instance_dist = _kpts[:, None] - _kpts
    instance_dist = np.sqrt(np.power(instance_dist, 2).sum(axis=3))
    close_instance_num = (instance_dist < close_dist_thr).sum(2)
    close_instance = close_instance_num > _num_nearby_joints_thr

    return close_instance


def nearby_joints_nms(
    kpts_db: List[dict],
    dist_thr: float = 0.05,
    num_nearby_joints_thr: Optional[int] = None,
    score_per_joint: bool = False,
    max_dets: int = 30,
):
    """Nearby joints NMS implementations. Instances with non-maximum scores
    will be suppressed if they have too much closed joints with other
    instances. This function is modified from project
    `DEKR<https://github.com/HRNet/DEKR/blob/main/lib/core/nms.py>`.

    Args:
        kpts_db (list[dict]): keypoints and scores.
        dist_thr (float): threshold for judging whether two joints are close.
            Defaults to 0.05.
        num_nearby_joints_thr (int): threshold for judging whether two
            instances are close.
        max_dets (int): max number of detections to keep. Defaults to 30.
        score_per_joint (bool): the input scores (in kpts_db) are per joint
            scores.

    Returns:
        np.ndarray: indexes to keep.
    """

    assert dist_thr > 0, "`dist_thr` must be greater than 0."
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k["score"].mean() for k in kpts_db])
    else:
        scores = np.array([k["score"] for k in kpts_db])

    kpts = np.array([k["keypoints"] for k in kpts_db])

    num_people, num_joints, _ = kpts.shape
    if num_nearby_joints_thr is None:
        num_nearby_joints_thr = num_joints // 2
    assert num_nearby_joints_thr < num_joints, (
        "`num_nearby_joints_thr` must " "be less than the number of joints."
    )

    # compute distance threshold
    pose_area = kpts.max(axis=1) - kpts.min(axis=1)

    pose_area = nb_pose_area_step(pose_area, num_people)
    pose_area = np.tile(pose_area, (num_people, num_joints))

    close_instance = nb_close_instance(pose_area, dist_thr, kpts, num_nearby_joints_thr)

    # apply nms
    ignored_pose_inds, keep_pose_inds = set(), list()
    indexes = np.argsort(scores)[::-1]
    for i in indexes:
        if i in ignored_pose_inds:
            continue
        keep_inds = close_instance[i].nonzero()[0]
        keep_ind = keep_inds[np.argmax(scores[keep_inds])]
        if keep_ind not in ignored_pose_inds:
            keep_pose_inds.append(keep_ind)
            ignored_pose_inds = ignored_pose_inds.union(set(keep_inds))

    # limit the number of output instances
    if max_dets > 0 and len(keep_pose_inds) > max_dets:
        sub_inds = np.argsort(scores[keep_pose_inds])[-1 : -max_dets - 1 : -1]
        keep_pose_inds = [keep_pose_inds[i] for i in sub_inds]

    return keep_pose_inds
