import torch
import torch.distributed as dist


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)  # add
    return reduced_inp / world_size
