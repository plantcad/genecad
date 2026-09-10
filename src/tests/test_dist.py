"""Multi-process coverage for guarded_on_main: a rank-zero-only failure must
reach every rank instead of leaving them blocked on the barrier that would
normally follow (see scripts/predict.py's prepare_run/finish_run calls)."""

import multiprocessing
import socket


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, queue, fail_on_rank):
    import torch.distributed as torch_dist

    from src import dist as dist_module

    torch_dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        with dist_module.guarded_on_main():
            if rank == fail_on_rank:
                raise ValueError(f"boom from rank {rank}")
        queue.put((rank, "ok", None))
    except Exception as exc:
        queue.put((rank, "raised", f"{type(exc).__name__}: {exc}"))
    finally:
        torch_dist.destroy_process_group()


def _run_ranks(
    world_size: int, fail_on_rank: int | None, timeout: int = 20
) -> dict[int, tuple[str, str | None]]:
    port = _free_port()
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = [
        context.Process(
            target=_worker, args=(rank, world_size, port, queue, fail_on_rank)
        )
        for rank in range(world_size)
    ]
    try:
        for worker in workers:
            worker.start()
        results: dict[int, tuple[str, str | None]] = {}
        for _ in range(world_size):
            rank, status, message = queue.get(timeout=timeout)
            results[rank] = (status, message)
        for worker in workers:
            worker.join(timeout=timeout)
            assert worker.exitcode == 0
        return results
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()


def test_guarded_on_main_failure_reaches_every_rank_without_hanging():
    """If this regresses to a bare `if is_main_process(): risky()`, rank 1/2
    block forever on the barrier rank zero never reaches, and queue.get
    above times out instead of returning -- a hang, not a silent pass."""
    results = _run_ranks(world_size=3, fail_on_rank=0)

    assert results[0] == ("raised", "ValueError: boom from rank 0")
    for rank in (1, 2):
        status, message = results[rank]
        assert status == "raised"
        assert message is not None and "boom from rank 0" in message


def test_guarded_on_main_success_path_all_ranks_proceed():
    results = _run_ranks(world_size=3, fail_on_rank=None)
    assert results == {r: ("ok", None) for r in range(3)}


def test_guarded_on_main_non_main_failure_does_not_block_the_others():
    """A non-main rank's own exception must not leave rank zero (or any
    other rank) blocked on the broadcast collective this raise skips past
    for its own process -- only the raising rank should fail."""
    results = _run_ranks(world_size=3, fail_on_rank=1)

    assert results[0] == ("ok", None)
    assert results[1] == ("raised", "ValueError: boom from rank 1")
    assert results[2] == ("ok", None)
