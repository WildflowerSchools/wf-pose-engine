import torch
from torch import Tensor
import torchvision

from mmpose.structures.bbox import bbox_overlaps

from mmpose.utils.logger import logging

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
