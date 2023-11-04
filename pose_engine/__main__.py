import os
import sys

import torch.multiprocessing as mp

from .cli import cli


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    args = []
    for idx, val in enumerate(sys.argv):
        if ".py" in val:
            args = sys.argv[slice(idx + 1, len(sys.argv))]

    cli(args)
