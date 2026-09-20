from contextlib import contextmanager
import os
from typing import Iterator

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
    # Use gloo for synchronisation: this script only needs barrier() between
    # ranks and performs no GPU collective ops (all_reduce, broadcast, etc.).
    # NCCL triggers P2P GPU memory probes during init which cause
    # "illegal memory access" errors on servers where GPUs lack P2P support.
    dist.init_process_group(backend="gloo")


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


def broadcast_from_main(value: object) -> object:
    """Broadcast a picklable value from rank zero to every rank.

    No-op outside a distributed context: returns `value` unchanged.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return value
    box = [value]
    dist.broadcast_object_list(box, src=0)
    return box[0]


@contextmanager
def guarded_on_main() -> Iterator[None]:
    """Wrap a block that gates its work with ``if is_main_process(): ...``;
    every rank learns whether rank zero's part of it raised.

    Wrap any block shaped like ``if is_main_process(): risky()`` that is
    normally followed by ``barrier()`` -- the ``is_main_process()`` check
    still belongs inside the block, same as before. Without this, an
    exception raised only on rank zero skips that barrier, leaving every
    other rank blocked on it indefinitely -- or, once rank zero tears down
    its own process group in a `finally`, facing a confusing low-level
    connection error instead of the real failure.

    Every rank exits this block together: rank zero re-raises its own
    exception with its original type and traceback, and every other rank
    raises a `RuntimeError` describing what rank zero hit. The broadcast
    this performs also acts as the barrier, so no separate `barrier()`
    call is needed after it.

    A non-main rank's own exception (which shouldn't occur, since its body
    should be a no-op guarded by ``is_main_process()``) still lets every
    rank reach the broadcast -- skipping straight to `raise` here would
    leave rank zero and every other rank blocked on that same collective
    call, waiting for a participant that already left. It's re-raised only
    after that, taking priority over rank zero's status.
    """
    error = None
    own_exc = None
    local_exc = None
    try:
        yield
    except Exception as exc:
        if is_main_process():
            error = f"{type(exc).__name__}: {exc}"
            own_exc = exc
        else:
            local_exc = exc
    error = broadcast_from_main(error)
    if local_exc is not None:
        raise local_exc
    if own_exc is not None:
        raise own_exc
    if error is not None:
        raise RuntimeError(f"Rank zero failed: {error}")
