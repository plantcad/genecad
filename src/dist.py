import os


def process_group() -> tuple[int, int, int]:
    """Determine the rank, local rank, and world size from environment variables."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size
