import locale
import sys

# import torch
from torch import Tensor
import torch.multiprocessing as mp

import mmpose.evaluation.functional
import mmpose.evaluation.functional.nms
from mmpose.structures.bbox import bbox_overlaps


def nms_torch(
    bboxes: Tensor,
    scores: Tensor,
    threshold: float = 0.65,
    iou_calculator=bbox_overlaps,
    return_group: bool = False,
):
    print("MADE IT")
    exit(1)


from .cli import cli


if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    mmpose.evaluation.functional.nms_torch = nms_torch
    mmpose.evaluation.functional.nms.nms_torch = nms_torch
    # module = sys.modules['mmpose.evaluation.functional']
    # module.nms_torch = nms_torch_redux
    # sys.modules['mmpose.evaluation.functional'] = module

    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    args = []
    for idx, val in enumerate(sys.argv):
        if ".py" in val:
            args = sys.argv[slice(idx + 1, len(sys.argv))]

    cli(args)
