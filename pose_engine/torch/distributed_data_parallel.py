from datetime import timedelta

import torch.distributed

from pose_engine.log import logger


def ddp_setup(rank, world_size):
    """
    Set up the distributed environment.

    Args:
        rank: The rank of the current process. Unique identifier for each process in the distributed training.
        world_size: Total number of processes participating in the distributed training.
    """
    # Set the current CUDA device to the specified device (identified by rank).
    # This ensures that each process uses a different GPU in a multi-GPU setup.
    torch.cuda.set_device(rank)

    # Initialize the process group.
    # 'backend' specifies the communication backend to be used, "nccl" is optimized for GPU training.
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )

    logger.info(
        f"DDP setup finished for rank '{rank}' with world size of '{world_size}'"
    )
