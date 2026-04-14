import os

import torch
import torch.distributed as dist


def init_process_group() -> None:
    """Initialise the distributed process group.

    Uses NCCL on CUDA-enabled machines, Gloo otherwise.
    This is a no-op when:
    - ``torch.distributed`` is not available, or
    - the group has already been initialised, or
    - the ``RANK`` environment variable is not set (i.e., single-process mode).
    """
    if not dist.is_available():
        return
    if dist.is_initialized():
        return
    # torchrun sets RANK; when absent we are not inside a distributed launch
    if "RANK" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def process_group() -> tuple[int, int]:
    """Return ``(rank, world_size)`` for the current process.

    Returns ``(0, 1)`` when running outside a distributed context.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    # Legacy fallback: honour env vars set by custom launchers
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def local_rank() -> int:
    """Return the LOCAL_RANK of this process (GPU index on the current node).

    Falls back to ``RANK`` then 0 when the variable is absent.
    """
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))


def is_main_process() -> bool:
    """Return ``True`` on rank 0 only."""
    rank, _ = process_group()
    return rank == 0


def barrier() -> None:
    """Block until all ranks reach this call (no-op in single-process mode)."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def destroy_process_group() -> None:
    """Tear down the distributed process group (no-op in single-process mode)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
