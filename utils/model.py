import torch
import torch.distributed as dist
import os

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    
def init_distributed_mode(args):
    if args.device != 'cuda':
        print('Not using distributed mode')
        args.distributed = False
        return

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("here RANK")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        print("here SLURM_PROCID")

        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    args.distributed = True

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)

    #torch.distributed.barrier()
    torch.distributed.barrier(device_ids=[args.gpu])
    torch.cuda.set_device(args.gpu)
    setup_for_distributed(args.rank == 0)
