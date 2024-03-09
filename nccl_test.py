import os
import torch
import torch.distributed as dist

def simple_nccl_test(rank, world_size):
    # Set the environment variables for initialization
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # Choose an open port.

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create a tensor and perform an all-reduce operation
    tensor = torch.ones(4).cuda(rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}, After all_reduce: {tensor}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    # Set these to the appropriate values for your setup
    rank = 0  # For single GPU, rank is 0
    world_size = 1  # World size is 1 since we're using only one GPU

    simple_nccl_test(rank, world_size)
