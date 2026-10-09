"""Parallel hashing of prediction segments gives the same outcome as one at a time."""

import json
import logging
import threading
import time
from pathlib import Path

import pytest

from src import prediction_checkpoint as checkpoint


def serial_file_hashes(path):
    root = Path(path)
    return {
        str(p.relative_to(root)): checkpoint.digest_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def serial_segments(root, *, verify, repair=False):
    """The one-receipt-at-a-time implementation the parallel version replaced."""
    root = Path(root)
    run_id = checkpoint.fingerprint(checkpoint.read_json(root / checkpoint.RUN))
    records = []
    for receipt in sorted(root.glob("segment.*.json")):
        try:
            record = checkpoint.read_json(receipt)
            receipt_hash = record.pop("receipt_hash")
            store = receipt.with_suffix(".zarr")
            if (
                receipt_hash != checkpoint.fingerprint(record)
                or record["run_id"] != run_id
                or record["store"] != store.name
                or record["strand"] not in ("positive", "negative")
                or not 0 <= record["start"] < record["stop"]
                or not 0 <= record["window_start"] < record["window_stop"]
                or not record["files"]
                or not store.is_dir()
            ):
                raise ValueError(f"Invalid receipt: {receipt}")
            if verify and serial_file_hashes(store) != record["files"]:
                raise ValueError(f"Prediction checksum mismatch: {store}")
            records.append(record)
        except (OSError, ValueError, KeyError, TypeError):
            if not repair:
                raise
            logging.getLogger(checkpoint.__name__).warning(
                "Discarding incomplete/corrupt prediction segment %s", receipt
            )
            receipt.unlink(missing_ok=True)
    return records


def make_run(root, n_segments=12, files_per_segment=5):
    """Segments in the committed layout, with nested files like a Zarr store."""
    root.mkdir()
    checkpoint.write_json(
        root / checkpoint.RUN,
        {"version": checkpoint.VERSION, "identity": {"m": 1}, "length": 10_000},
    )
    run_id = checkpoint.fingerprint(checkpoint.read_json(root / checkpoint.RUN))
    for i in range(n_segments):
        strand = ("positive", "negative")[i % 2]
        name = f"segment.{strand}.{i // 2}.{i:032x}"
        store = root / (name + ".zarr")
        for j in range(files_per_segment):
            path = store / strand / f"var{j}" / "0"
            path.parent.mkdir(parents=True)
            path.write_bytes(bytes([i, j]) * (1 + 3000 * j))
        (store / ".zmetadata").write_text(json.dumps({"i": i}))
        record = {
            "run_id": run_id,
            "store": store.name,
            "strand": strand,
            "start": 100 * (i // 2),
            "stop": 100 * (i // 2 + 1),
            "window_start": i // 2,
            "window_stop": i // 2 + 1,
            "files": serial_file_hashes(store),
        }
        checkpoint.write_json(
            root / (name + ".json"),
            {**record, "receipt_hash": checkpoint.fingerprint(record)},
        )
    return root


def receipts(root):
    return sorted(p.name for p in root.glob("segment.*.json"))


def stores(root):
    return sorted(p for p in root.glob("segment.*.zarr"))


# Each damage function breaks one segment in a way the checks must catch.
def corrupt_content(root):
    path = next(stores(root)[3].rglob("0"))
    path.write_bytes(path.read_bytes()[:-1] + b"\xff")


def remove_file(root):
    (stores(root)[5] / ".zmetadata").unlink()


def add_file(root):
    (stores(root)[1] / "extra").write_text("x")


def bad_json(root):
    (root / receipts(root)[7]).write_text("{not json")


def tamper_receipt(root):
    path = root / receipts(root)[2]
    value = json.loads(path.read_text())
    value["stop"] += 1
    path.write_text(json.dumps(value))


def missing_store(root):
    import shutil

    shutil.rmtree(stores(root)[9])


def several(root):
    corrupt_content(root)
    bad_json(root)
    missing_store(root)
    remove_file(root)


DAMAGE = [
    None,
    corrupt_content,
    remove_file,
    add_file,
    bad_json,
    tamper_receipt,
    missing_store,
    several,
]


def outcome(function, root, caplog, **kwargs):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=checkpoint.__name__):
        try:
            result = ("ok", function(root, **kwargs))
        except Exception as error:
            result = ("error", type(error), str(error))
    warnings = [r.getMessage() for r in caplog.records]
    return result, warnings, receipts(root)


@pytest.mark.parametrize(
    "damage", DAMAGE, ids=lambda d: getattr(d, "__name__", "intact")
)
@pytest.mark.parametrize("verify,repair", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("workers", ["1", "3", "64"])
def test_segments_match_serial(
    tmp_path, caplog, monkeypatch, damage, verify, repair, workers
):
    monkeypatch.setenv(checkpoint.HASH_WORKERS_ENV, workers)
    expected_root = make_run(tmp_path / "serial")
    actual_root = make_run(tmp_path / "parallel")
    if damage:
        damage(expected_root)
        damage(actual_root)
    kwargs = {"verify": verify, "repair": repair}
    expected = outcome(serial_segments, expected_root, caplog, **kwargs)
    actual = outcome(checkpoint.segments, actual_root, caplog, **kwargs)
    # Paths differ only by the run directory name.
    assert repr(actual).replace("parallel", "serial") == repr(expected)


@pytest.mark.parametrize("workers", ["1", "2", "64"])
def test_file_hashes_match_serial(tmp_path, monkeypatch, workers):
    monkeypatch.setenv(checkpoint.HASH_WORKERS_ENV, workers)
    root = make_run(tmp_path / "run")
    for store in stores(root):
        assert checkpoint.file_hashes(store) == serial_file_hashes(store)
        assert list(checkpoint.file_hashes(store)) == list(serial_file_hashes(store))
    assert checkpoint.file_hashes(tmp_path / "empty") == {}


def test_first_failing_file_in_sorted_order_is_raised(tmp_path, monkeypatch):
    root = make_run(tmp_path / "run", n_segments=1, files_per_segment=6)
    store = stores(root)[0]
    files = [p for p in sorted(store.rglob("*")) if p.is_file()]
    broken = {files[4], files[1]}
    real = checkpoint.digest_file

    def digest(path):
        if Path(path) in broken:
            raise OSError(f"cannot read {Path(path).name}:{files.index(Path(path))}")
        return real(path)

    monkeypatch.setattr(checkpoint, "digest_file", digest)
    with pytest.raises(OSError, match=":1$"):
        checkpoint.file_hashes(store)
    with pytest.raises(OSError, match=":1$"):
        checkpoint.segments(root, verify=True)


def test_unexpected_errors_are_not_swallowed_in_repair_mode(tmp_path, monkeypatch):
    root = make_run(tmp_path / "run")
    target = stores(root)[4]

    def digest(path):
        if target in Path(path).parents:
            raise RuntimeError("boom")
        return "x"

    monkeypatch.setattr(checkpoint, "digest_file", digest)
    with pytest.raises(RuntimeError, match="boom"):
        checkpoint.segments(root, verify=True, repair=True)
    # Earlier segments were discarded (their digests do not match); later ones kept.
    assert len(receipts(root)) == 12 - 4


def test_verify_false_reads_no_store_files(tmp_path, monkeypatch):
    root = make_run(tmp_path / "run")

    def digest(path):
        raise AssertionError(f"read {path}")

    monkeypatch.setattr(checkpoint, "digest_file", digest)
    assert len(checkpoint.segments(root, verify=False)) == 12


def test_files_are_hashed_concurrently(tmp_path, monkeypatch):
    """With per-file latency, as on a network file system, reads overlap."""
    monkeypatch.setenv(checkpoint.HASH_WORKERS_ENV, "8")
    root = make_run(tmp_path / "run", n_segments=8, files_per_segment=4)
    real = checkpoint.digest_file
    lock = threading.Lock()
    active = peak = 0

    def slow(path):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return real(path)

    monkeypatch.setattr(checkpoint, "digest_file", slow)
    assert len(checkpoint.segments(root, verify=True)) == 8
    assert peak > 1


def test_receipt_digests_match_serial(tmp_path, monkeypatch):
    monkeypatch.setenv(checkpoint.HASH_WORKERS_ENV, "5")
    root = make_run(tmp_path / "run")
    assert checkpoint.receipt_digests(root) == {
        p.name: checkpoint.digest_file(p) for p in sorted(root.glob("segment.*.json"))
    }


@pytest.mark.parametrize("value", ["0", "-2", "many", ""])
def test_invalid_worker_count_is_rejected(monkeypatch, value):
    monkeypatch.setenv(checkpoint.HASH_WORKERS_ENV, value)
    with pytest.raises(ValueError, match=checkpoint.HASH_WORKERS_ENV):
        checkpoint.hash_workers()


def test_default_worker_count(monkeypatch):
    monkeypatch.delenv(checkpoint.HASH_WORKERS_ENV, raising=False)
    assert 1 <= checkpoint.hash_workers() <= checkpoint.MAX_HASH_WORKERS


def test_interrupt_cancels_queued_files(tmp_path, monkeypatch):
    """Executor.map cancels the files still queued when a result raises."""
    monkeypatch.setenv(checkpoint.HASH_WORKERS_ENV, "2")
    root = make_run(tmp_path / "run", n_segments=20, files_per_segment=10)
    calls = 0
    lock = threading.Lock()

    def digest(path):
        nonlocal calls
        with lock:
            calls += 1
            first = calls == 1
        if first:
            raise KeyboardInterrupt
        time.sleep(0.01)
        return "x"

    monkeypatch.setattr(checkpoint, "digest_file", digest)
    with pytest.raises(KeyboardInterrupt):
        checkpoint.segments(root, verify=True)
    assert calls < 20 * 11
