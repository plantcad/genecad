"""Prediction segments and recovery metadata.

Workers write independent segments and commit each with an atomic JSON receipt.
Restart discards segments with missing or invalid receipts. Distributed ranks
share one output directory under a rank-zero lock.
"""

import hashlib
import fcntl
import json
import logging
from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4

from src.atomic_io import atomic_output_path

logger = logging.getLogger(__name__)
RUN = "run.json"
SUCCESS = "_SUCCESS.json"
VERSION = 1


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


def file_hashes(path: str | Path) -> dict[str, str]:
    root = Path(path)
    return {
        str(p.relative_to(root)): digest_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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


def prepare_run(output_dir: str, identity: dict, length: int) -> None:
    """Check run identity and remove invalid segments before starting workers.

    Called by rank zero. Incompatible runs raise without deleting outputs.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = {"version": VERSION, "identity": identity, "length": length}
    if length <= 0:
        raise ValueError("Cannot predict an empty chromosome")
    if (root / RUN).exists():
        if read_json(root / RUN) != expected:
            raise PredictionResumeError(
                f"Prediction inputs/model/settings changed in {root}; "
                "use a new output directory to avoid mixing results"
            )
    else:
        if any(p.name != RUN + ".tmp" for p in root.iterdir()):
            raise PredictionResumeError(
                f"Unverified legacy predictions in {root}; use a new output directory"
            )
        write_json(root / RUN, expected)
    (root / SUCCESS).unlink(missing_ok=True)
    records = segments(root, verify=True, repair=True)
    retained = {record["store"] for record in records}
    for path in root.glob("segment.*.zarr"):
        if path.name not in retained:
            shutil.rmtree(path)


def segments(root: str | Path, *, verify: bool, repair: bool = False) -> list[dict]:
    root = Path(root)
    run_id = fingerprint(read_json(root / RUN))
    records = []
    for receipt in sorted(root.glob("segment.*.json")):
        try:
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
            if verify and file_hashes(store) != record["files"]:
                raise ValueError(f"Prediction checksum mismatch: {store}")
            records.append(record)
        except (OSError, ValueError, KeyError, TypeError):
            if not repair:
                raise
            logger.warning(
                "Discarding incomplete/corrupt prediction segment %s", receipt
            )
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
            "receipts": {
                p.name: digest_file(p) for p in sorted(root.glob("segment.*.json"))
            },
        },
    )


def completed_segments(output_dir: str) -> list[dict]:
    """Fail closed before downstream stages consume any new-format predictions."""
    root = Path(output_dir)
    run = read_json(root / RUN)
    complete = read_json(root / SUCCESS)
    receipts = {p.name: digest_file(p) for p in sorted(root.glob("segment.*.json"))}
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
