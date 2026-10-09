"""Prediction segments and recovery metadata.

Workers write independent segments and commit each with an atomic JSON receipt.
Restart discards segments with missing or invalid receipts. Distributed ranks
share one output directory under a rank-zero lock.
"""

import hashlib
import fcntl
import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import TypeVar
from uuid import uuid4

from src.atomic_io import atomic_output_path

logger = logging.getLogger(__name__)
Item = TypeVar("Item")
Result = TypeVar("Result")
RUN = "run.json"
SUCCESS = "_SUCCESS.json"
VERSION = 1
HASH_WORKERS_ENV = "GENECAD_HASH_WORKERS"
MAX_HASH_WORKERS = 64
ZARR_METADATA = {".zmetadata", ".zattrs", ".zarray", ".zgroup", "zarr.json"}


class PredictionResumeError(ValueError):
    """Retrying with a smaller GPU batch cannot resolve this output conflict."""


@contextmanager
def prediction_lock(output_dir: str):
    """One writer job per chromosome; all its ranks run under rank zero's lock."""
    root = Path(output_dir)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + ".lock")).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PredictionResumeError(
                f"Another prediction job owns {output_dir}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_workers() -> int:
    """Threads used to read and hash prediction files.

    Hashing is bound by per-file latency on network file systems (Lustre, NFS),
    so files are read concurrently; hashlib and file reads release the GIL.
    """
    value = os.environ.get(HASH_WORKERS_ENV)
    if value is not None:
        try:
            workers = int(value)
        except ValueError:
            workers = 0
        if workers < 1:
            raise ValueError(
                f"{HASH_WORKERS_ENV} must be a positive integer: {value!r}"
            )
        return workers
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS
        cpus = os.cpu_count() or 1
    return min(MAX_HASH_WORKERS, 4 * cpus)


def _parallel(
    function: Callable[[Item], Result], items: list[Item], workers: int
) -> list[Result | Exception]:
    """The result of ``function`` for each item, or the exception it raised, in order."""

    def capture(item: Item) -> Result | Exception:
        try:
            return function(item)
        except Exception as error:
            return error

    workers = min(workers, len(items))
    if workers <= 1:
        return [capture(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(capture, items))


def _digests(paths: list[Path], workers: int) -> list[str]:
    """Digests in input order; raises the error of the first file that failed."""
    digests = []
    for value in _parallel(digest_file, paths, workers):
        if isinstance(value, Exception):
            raise value
        digests.append(value)
    return digests


def _list_files(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file()]


def file_hashes(path: str | Path) -> dict[str, str]:
    root = Path(path)
    files = _list_files(root)
    return {
        str(p.relative_to(root)): digest
        for p, digest in zip(files, _digests(files, hash_workers()))
    }


def receipt_digests(root: Path) -> dict[str, str]:
    receipts = sorted(root.glob("segment.*.json"))
    return {
        p.name: digest
        for p, digest in zip(receipts, _digests(receipts, hash_workers()))
    }


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _metadata_digest(path: Path, digest: str) -> str:
    """Digest of a Zarr JSON metadata file's content, or `digest` if it is not JSON."""
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (UnicodeDecodeError, ValueError):
        return digest
    return fingerprint(value)


def input_fingerprints(path: str | Path) -> tuple[str, str]:
    """Fingerprints of an input Zarr store: (content, bytes).

    The content fingerprint hashes Zarr's JSON metadata files by their parsed
    content. Writing the same sequences again lists the metadata keys in another
    order, which changes the bytes of `.zmetadata` but not its content. The bytes
    fingerprint hashes every file as is; runs started before the content
    fingerprint existed recorded that one.
    """
    root = Path(path)
    hashes = file_hashes(root)
    content = {
        name: _metadata_digest(root / name, digest)
        if Path(name).name in ZARR_METADATA
        else digest
        for name, digest in hashes.items()
    }
    return fingerprint(content), fingerprint(hashes)


def write_json(path: Path, value: object) -> None:
    with atomic_output_path(path) as temporary:
        with open(temporary, "w") as handle:
            json.dump(value, handle, sort_keys=True)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _segment_output(path: Path) -> bool:
    return path.name.startswith("segment.") or path.name in (SUCCESS, SUCCESS + ".tmp")


def prepare_run(
    output_dir: str, identity: dict, length: int, accepted: tuple[dict, ...] = ()
) -> None:
    """Check run identity and remove invalid segments before starting workers.

    A run recorded with `identity` or one of the `accepted` identities is resumed.
    Called by rank zero. Incompatible runs raise without deleting outputs.
    Segments left without a run.json cannot be checked against any identity and
    are removed.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = {"version": VERSION, "identity": identity, "length": length}
    if length <= 0:
        raise ValueError("Cannot predict an empty chromosome")
    if (root / RUN).exists():
        if read_json(root / RUN) not in [
            expected,
            *({**expected, "identity": other} for other in accepted),
        ]:
            raise PredictionResumeError(
                f"Prediction inputs/model/settings changed in {root}; delete it to "
                "predict this sequence again, or use a new output directory"
            )
    else:
        leftovers = [p for p in root.iterdir() if p.name != RUN + ".tmp"]
        if not all(_segment_output(p) for p in leftovers):
            raise PredictionResumeError(
                f"Unverified legacy predictions in {root}; use a new output directory"
            )
        if leftovers:
            logger.warning(
                "Removing %d prediction files without %s in %s; "
                "their windows are predicted again",
                len(leftovers),
                RUN,
                root,
            )
        for path in leftovers:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        write_json(root / RUN, expected)
    (root / SUCCESS).unlink(missing_ok=True)
    records = segments(root, verify=True, repair=True)
    retained = {record["store"] for record in records}
    for path in root.glob("segment.*.zarr"):
        if path.name not in retained:
            shutil.rmtree(path)


def _check_receipt(
    receipt: Path, run_id: str, verify: bool
) -> tuple[dict, Path, list[Path]]:
    record = read_json(receipt)
    receipt_hash = record.pop("receipt_hash")
    store = receipt.with_suffix(".zarr")
    if (
        receipt_hash != fingerprint(record)
        or record["run_id"] != run_id
        or record["store"] != store.name
        or record["strand"] not in ("positive", "negative")
        or not 0 <= record["start"] < record["stop"]
        or not 0 <= record["window_start"] < record["window_stop"]
        or not record["files"]
        or not store.is_dir()
    ):
        raise ValueError(f"Invalid receipt: {receipt}")
    return record, store, _list_files(store) if verify else []


def segments(root: str | Path, *, verify: bool, repair: bool = False) -> list[dict]:
    """Valid segments in receipt order.

    Receipts are read, and all files of all stores hashed, concurrently; the
    outcome (records, warnings, deletions and which error is raised) is the same
    as checking the receipts one at a time in sorted order.
    """
    root = Path(root)
    run_id = fingerprint(read_json(root / RUN))
    workers = hash_workers()
    receipts = sorted(root.glob("segment.*.json"))
    checked = _parallel(
        lambda receipt: _check_receipt(receipt, run_id, verify), receipts, workers
    )
    files = [
        path
        for value in checked
        if not isinstance(value, Exception)
        for path in value[2]
    ]
    hashed = iter(_parallel(digest_file, files, workers))

    records = []
    for receipt, value in zip(receipts, checked):
        error: Exception | None = None
        if isinstance(value, Exception):
            error = value
        else:
            record, store, store_files = value
            digests = [next(hashed) for _ in store_files]
            failed = [d for d in digests if isinstance(d, Exception)]
            if failed:
                error = failed[0]
            elif (
                verify
                and {
                    str(path.relative_to(store)): digest
                    for path, digest in zip(store_files, digests)
                }
                != record["files"]
            ):
                error = ValueError(f"Prediction checksum mismatch: {store}")
            else:
                records.append(record)
                continue
        if not isinstance(error, (OSError, ValueError, KeyError, TypeError)):
            raise error
        if not repair:
            raise error
        logger.warning("Discarding incomplete/corrupt prediction segment %s", receipt)
        receipt.unlink(missing_ok=True)
    return records


def commit_segment(output_dir: str, result, strand: str, window_ids: list[int]) -> None:
    """Write one contiguous batch, then commit its content hashes (bounded I/O RAM)."""
    import numpy as np

    if not window_ids or window_ids != list(range(window_ids[0], window_ids[-1] + 1)):
        raise ValueError("A segment must contain contiguous windows")
    required = {
        "token_logits",
        "feature_logits",
        "token_predictions",
        "feature_predictions",
    }
    if not required <= set(result.data_vars):
        raise ValueError("Prediction segment is missing required arrays")
    coordinates = result.sequence.values
    if len(coordinates) == 0 or not np.all(np.diff(coordinates) == 1):
        raise ValueError(
            "Prediction segment coordinates must be contiguous and ascending"
        )
    root = Path(output_dir)
    name = f"segment.{strand}.{window_ids[0]}.{uuid4().hex}"
    store = root / (name + ".zarr")
    result.to_zarr(str(store), group=strand, zarr_format=2, consolidated=True, mode="w")
    record = {
        "run_id": fingerprint(read_json(root / RUN)),
        "store": store.name,
        "strand": strand,
        "start": int(coordinates[0]),
        "stop": int(coordinates[-1]) + 1,
        "window_start": window_ids[0],
        "window_stop": window_ids[-1] + 1,
        "files": file_hashes(store),
    }
    write_json(root / (name + ".json"), {**record, "receipt_hash": fingerprint(record)})


def validate_coverage(records: list[dict], length: int) -> None:
    for strand in ("positive", "negative"):
        selected = [r for r in records if r["strand"] == strand]
        end = 0
        for record in sorted(selected, key=lambda r: r["start"]):
            if record["start"] != end:
                raise ValueError(f"Gap or overlap in {strand} predictions at {end}")
            end = record["stop"]
        if end != length:
            raise ValueError(f"Incomplete {strand} predictions: {end} of {length}")
        window_end = 0
        for record in sorted(selected, key=lambda r: r["window_start"]):
            if record["window_start"] != window_end:
                raise ValueError(f"Gap or overlap in {strand} prediction windows")
            window_end = record["window_stop"]


def finish_run(output_dir: str) -> None:
    root = Path(output_dir)
    run = read_json(root / RUN)
    records = segments(root, verify=True)
    validate_coverage(records, run["length"])
    write_json(
        root / SUCCESS,
        {
            "run_id": fingerprint(run),
            "receipts": receipt_digests(root),
        },
    )


def completed_segments(output_dir: str) -> list[dict]:
    """Fail closed before downstream stages consume any new-format predictions."""
    root = Path(output_dir)
    run = read_json(root / RUN)
    complete = read_json(root / SUCCESS)
    receipts = receipt_digests(root)
    if complete != {"run_id": fingerprint(run), "receipts": receipts}:
        raise ValueError(f"Prediction completion manifest mismatch in {root}")
    records = segments(root, verify=True)
    validate_coverage(records, run["length"])
    return records


def pending_batches(
    windows,
    records: list[dict],
    strand: str,
    rank: int,
    world_size: int,
    batch_size: int,
    total_windows: int,
):
    """Yield bounded batches of missing windows; IDs survive batch/GPU changes."""
    if batch_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid prediction batch size or rank")
    ranges = sorted(
        (r["window_start"], r["window_stop"]) for r in records if r["strand"] == strand
    )
    cursor = 0
    batch, ids = [], []
    width, extra = divmod(total_windows, world_size)
    first = rank * width + min(rank, extra)
    stop = first + width + (rank < extra)
    # Ownership is recomputed after restart; receipts remain rank-independent.
    for index, window in enumerate(windows):
        while cursor < len(ranges) and ranges[cursor][1] <= index:
            cursor += 1
        done = cursor < len(ranges) and ranges[cursor][0] <= index < ranges[cursor][1]
        owned = first <= index < stop
        if done or not owned:
            if batch:
                yield ids, batch
                batch, ids = [], []
            continue
        batch.append(window)
        ids.append(index)
        if len(batch) == batch_size:
            yield ids, batch
            batch, ids = [], []
    if batch:
        yield ids, batch
