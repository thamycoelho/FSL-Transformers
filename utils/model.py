import torch
import os
from torch.distributed import init_process_group

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10012"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
