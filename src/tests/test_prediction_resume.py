"""Exercise real inference batching and Zarr I/O with deterministic CPU logits."""

import importlib.util
import dataclasses
import json
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from src import prediction_checkpoint as checkpoint
from src.prediction import merge_prediction_datasets

spec = importlib.util.spec_from_file_location(
    "predict_resume_tests", Path(__file__).resolve().parents[2] / "scripts/predict.py"
)
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)


class Classifier:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.windows = 0
        self.fail_after = fail_after
        self.config = SimpleNamespace(
            token_class_names=["a", "b"],
            token_entity_names_with_background=lambda: ["background", "gene"],
            use_precomputed_base_encodings=False,
        )

    def __call__(self, input_ids, inputs_embeds=None):
        if self.calls == self.fail_after:
            raise RuntimeError("simulated OOM")
        self.calls += 1
        self.windows += len(input_ids)
        # Include window context: resuming from arbitrary bases would differ.
        values = input_ids.float() + input_ids.sum(dim=1, keepdim=True)
        return torch.stack((values, -values), dim=-1)

    def aggregate_logits(self, logits):
        return logits * 2


def infer(
    root,
    monkeypatch,
    *,
    batch=2,
    rank=0,
    world=1,
    fail_after=None,
    length=47,
    sync=None,
):
    monkeypatch.setattr(predict, "process_group", lambda: (rank, world))
    monkeypatch.setattr(predict, "barrier", sync or (lambda: None))
    values = np.arange(length) % 5
    ds = xr.Dataset(
        {
            "sequence_input_ids": (
                ["strand", "sequence"],
                np.stack([values, values + 1]),
            )
        },
        coords={"strand": ["positive", "negative"], "sequence": np.arange(length)},
    )
    classifier = Classifier(fail_after)
    predict._create_predictions(
        ds=ds,
        base_model=None,
        classifier=classifier,
        tokenizer=SimpleNamespace(unk_token_id=0),
        species_id="sp",
        chromosome_id="chr",
        model_checkpoint="test",
        model_path="test",
        output_dir=str(root),
        batch_size=batch,
        window_size=8,
        stride=4,
        device="cpu",
        tqdm_position=0,
    )
    return classifier.windows


def prepare(root, length=47, **identity):
    checkpoint.prepare_run(str(root), {"model": "test", **identity}, length)


def test_interrupted_resume_changed_batch_and_world_size(tmp_path, monkeypatch):
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    prepare(baseline)
    total = infer(baseline, monkeypatch, batch=3)
    checkpoint.finish_run(str(baseline))
    expected = merge_prediction_datasets(str(baseline)).load()

    prepare(resumed)
    with pytest.raises(RuntimeError, match="simulated OOM"):
        infer(resumed, monkeypatch, batch=2, fail_after=2)
    with pytest.raises(FileNotFoundError):
        merge_prediction_datasets(str(resumed))
    saved = checkpoint.segments(resumed, verify=True)
    assert sum(r["window_stop"] - r["window_start"] for r in saved) == 4
    saved_hashes = {r["store"]: r["files"] for r in saved}

    prepare(resumed)
    remaining = sum(
        infer(resumed, monkeypatch, batch=1, rank=r, world=3) for r in range(3)
    )
    assert remaining == total - 4
    checkpoint.finish_run(str(resumed))
    actual = merge_prediction_datasets(str(resumed)).load()
    xr.testing.assert_equal(actual, expected)
    for store, hashes in saved_hashes.items():
        assert checkpoint.file_hashes(resumed / store) == hashes
    prepare(resumed)
    assert infer(resumed, monkeypatch, batch=5) == 0
    checkpoint.finish_run(str(resumed))


@pytest.mark.parametrize(
    "damage", ["truncated", "missing", "uncommitted", "receipt", "metadata"]
)
def test_repairs_only_damaged_segment(tmp_path, monkeypatch, damage):
    prepare(tmp_path)
    infer(tmp_path, monkeypatch)
    checkpoint.finish_run(str(tmp_path))
    expected = merge_prediction_datasets(str(tmp_path)).load()
    record = checkpoint.segments(tmp_path, verify=True)[0]
    store = tmp_path / record["store"]
    receipt = store.with_suffix(".json")
    if damage == "uncommitted":
        receipt.unlink()
    elif damage == "receipt":
        receipt.write_text("{broken")
    elif damage == "metadata":
        value = json.loads(receipt.read_text())
        value["window_stop"] += 1
        receipt.write_text(json.dumps(value))
    else:
        chunk = next(
            p for p in store.rglob("*") if p.is_file() and not p.name.startswith(".")
        )
        if damage == "missing":
            chunk.unlink()
        else:
            chunk.write_bytes(b"broken")
    with pytest.raises((ValueError, OSError)):  # pyrefly: ignore[no-matching-overload]
        merge_prediction_datasets(str(tmp_path))
    prepare(tmp_path)
    assert (
        infer(tmp_path, monkeypatch, batch=1)
        == record["window_stop"] - record["window_start"]
    )
    checkpoint.finish_run(str(tmp_path))
    xr.testing.assert_equal(merge_prediction_datasets(str(tmp_path)).load(), expected)


def test_crash_during_segment_write(tmp_path, monkeypatch):
    prepare(tmp_path)
    original = xr.Dataset.to_zarr
    calls = 0

    def write_then_crash(self, *args, **kwargs):
        nonlocal calls
        result = original(self, *args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("killed before receipt")
        return result

    with monkeypatch.context() as context:
        context.setattr(xr.Dataset, "to_zarr", write_then_crash)
        with pytest.raises(RuntimeError, match="before receipt"):
            infer(tmp_path, monkeypatch)
    assert len(checkpoint.segments(tmp_path, verify=True)) == 1
    assert len(list(tmp_path.glob("segment.*.zarr"))) == 2
    prepare(tmp_path)
    assert len(list(tmp_path.glob("segment.*.zarr"))) == 1
    infer(tmp_path, monkeypatch, batch=3)
    checkpoint.finish_run(str(tmp_path))
    assert merge_prediction_datasets(str(tmp_path)).sizes["sequence"] == 47


def test_changed_identity_rejected_without_destroying_progress(tmp_path, monkeypatch):
    prepare(tmp_path)
    infer(tmp_path, monkeypatch)
    before = checkpoint.file_hashes(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        prepare(tmp_path, model="different")
    assert checkpoint.file_hashes(tmp_path) == before


def test_more_ranks_than_windows_and_short_sequence(tmp_path, monkeypatch):
    prepare(tmp_path, length=3)
    counts = [infer(tmp_path, monkeypatch, rank=r, world=4, length=3) for r in range(4)]
    assert counts == [2, 0, 0, 0]
    checkpoint.finish_run(str(tmp_path))
    assert merge_prediction_datasets(str(tmp_path)).sizes["sequence"] == 3


def test_coverage_rejects_gaps_overlaps_and_missing_strand():
    positive = {
        "strand": "positive",
        "start": 0,
        "stop": 10,
        "window_start": 0,
        "window_stop": 2,
    }
    negative = {**positive, "strand": "negative"}
    checkpoint.validate_coverage([positive, negative], 10)
    for records in (
        [positive],
        [positive, positive, negative],
        [{**positive, "start": 1}, negative],
    ):
        with pytest.raises(ValueError):
            checkpoint.validate_coverage(records, 10)


def test_legacy_rank_stores_still_readable(tmp_path, monkeypatch):
    fresh = tmp_path / "fresh"
    prepare(fresh)
    infer(fresh, monkeypatch)
    checkpoint.finish_run(str(fresh))
    expected = merge_prediction_datasets(str(fresh)).load()
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for strand in ("positive", "negative"):
        ds = expected.sel(strand=strand, drop=True)
        # Remove inherited per-segment chunk encodings before a fresh write.
        ds = ds.drop_encoding()
        ds.to_zarr(str(legacy / "predictions.0.zarr"), group=strand, zarr_format=2)
    xr.testing.assert_equal(merge_prediction_datasets(str(legacy)).load(), expected)


def test_writer_lock_excludes_second_job_and_releases(tmp_path):
    output = str(tmp_path / "prediction")
    with checkpoint.prediction_lock(output):
        with pytest.raises(
            checkpoint.PredictionResumeError, match="Another prediction job"
        ):
            with checkpoint.prediction_lock(output):
                pass
    with checkpoint.prediction_lock(output):
        pass


def test_model_identity_detects_weights_and_tokenizer_changes():
    @dataclasses.dataclass
    class Config:
        hidden_size: int = 2

    model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
    model.config = Config()
    tokenizer = SimpleNamespace(get_vocab=lambda: {"A": 0}, unk_token_id=0)
    initial = predict.prediction_model_digest(None, model, tokenizer)
    assert predict.prediction_model_digest(None, model, tokenizer) == initial
    with torch.no_grad():
        model.weight.add_(1)
    changed = predict.prediction_model_digest(None, model, tokenizer)
    assert changed != initial
    tokenizer.get_vocab = lambda: {"C": 0}
    assert predict.prediction_model_digest(None, model, tokenizer) != changed


def _rank_worker(root, rank, sync):
    with pytest.MonkeyPatch.context() as patch:
        infer(root, patch, rank=rank, world=2, sync=sync.wait)


def test_concurrent_workers_commit_disjoint_segments(tmp_path):
    prepare(tmp_path)
    context = multiprocessing.get_context("spawn")
    sync = context.Barrier(2, timeout=60)
    workers = [
        context.Process(target=_rank_worker, args=(tmp_path, r, sync)) for r in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=90)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
    checkpoint.finish_run(str(tmp_path))
    assert merge_prediction_datasets(str(tmp_path)).sizes["sequence"] == 47


def test_visualization_commands_match_current_parsers(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]

    def load(name):
        spec = importlib.util.spec_from_file_location(
            name, repo / "scripts" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    visualization = load("predict_and_visualize")
    extraction = load("extract_fasta")
    detection = load("detect_intervals")
    commands = []
    monkeypatch.setattr(visualization, "_run", commands.append)
    scripts = str(repo / "scripts")
    visualization.step_extract(
        "input.fa", "sp", "chr", "model", str(tmp_path / "seq"), scripts
    )
    visualization.step_predict(
        "seq", "sp", "chr", "model", "head", "pred", 2, "cuda", "float32", 8, 4, scripts
    )
    visualization.step_detect_intervals(
        "pred", str(tmp_path / "intervals"), "direct", "plant", scripts, "input.fa"
    )
    calls = []
    monkeypatch.setattr(
        extraction,
        "extract_fasta_file",
        lambda *args, **kwargs: calls.append("extract"),
    )
    monkeypatch.setattr(
        predict, "create_predictions", lambda **kwargs: calls.append("predict")
    )
    monkeypatch.setattr(
        detection, "detect_intervals", lambda **kwargs: calls.append("detect")
    )
    for command, module in zip(commands, (extraction, predict, detection)):
        monkeypatch.setattr(sys, "argv", command[1:])
        module.main()
    assert calls == ["extract", "predict", "detect"]
