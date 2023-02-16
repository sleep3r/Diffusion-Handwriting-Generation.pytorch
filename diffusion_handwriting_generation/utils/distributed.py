from torch import distributed as dist


def get_dist_info() -> (int, int):
    """Gets the rank and world size of the current process group."""
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
