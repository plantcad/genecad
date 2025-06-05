import os

def process_group() -> tuple[int, int]:
    """Determine the rank and world size from environment variables."""
    if os.environ.get("RANK") and os.environ.get("WORLD_SIZE"):
        rank = int(os.environ.get("RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        return rank, world_size
    return 0, 1 

