"""Persistent model lifetime, real Zarr inference, and shell dispatch integration."""

import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest
import torch
import xarray as xr

from src.prediction import merge_prediction_datasets
from src import prediction_checkpoint as checkpoint
from src.tests.test_prediction_resume import Classifier, predict


class LimitedClassifier(Classifier):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.attempts = []

    def __call__(self, input_ids, inputs_embeds=None):
        self.attempts.append(len(input_ids))
        if len(input_ids) > self.limit:
            raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")
        return super().__call__(input_ids, inputs_embeds)


def make_manifest(root, count=3):
    root.mkdir(parents=True)
    entries = []
    for i in range(count):
        chrom = f"scaffold{i}"
        length = 37 + 5 * i
        values = (np.arange(length) + i) % 5
        ds = xr.Dataset(
            {
                "sequence_input_ids": (
                    ["strand", "sequence"],
                    np.stack([values, values + 1]),
                )
            },
            coords={"strand": ["positive", "negative"], "sequence": np.arange(length)},
        )
        input_path = str(root / f"{chrom}.zarr")
        ds.to_zarr(input_path, group=f"sp/{chrom}", zarr_format=2)
        entries.append(
            {
                "chromosome_id": chrom,
                "sequence_zarr": input_path,
                "predictions_dir": str(root / chrom),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(entries))
    return manifest, entries


def run_worker(manifest, monkeypatch, classifier, *, batch=4, cache=None):
    loaded, fingerprinted, sizes, models = [], [], [], []
    monkeypatch.setattr(predict.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(predict.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(predict, "init_process_group", lambda: None)
    monkeypatch.setattr(predict, "destroy_process_group", lambda: None)
    monkeypatch.setattr(predict, "process_group", lambda: (0, 1))
    monkeypatch.setattr(predict, "is_main_process", lambda: True)
    monkeypatch.setattr(predict, "barrier", lambda: None)

    def load(**kwargs):
        from types import SimpleNamespace

        loaded.append(True)
        return None, classifier, SimpleNamespace(unk_token_id=0)

    def digest(*args):
        fingerprinted.append(True)
        return "test-model"

    inference = predict._create_predictions

    def infer(**kwargs):
        sizes.append(kwargs["batch_size"])
        models.append(kwargs["classifier"])
        kwargs["device"] = "cpu"
        return inference(**kwargs)

    monkeypatch.setattr(predict, "load_models", load)
    monkeypatch.setattr(predict, "prediction_model_digest", digest)
    monkeypatch.setattr(predict, "_create_predictions", infer)
    predict.create_predictions(
        manifest=str(manifest),
        species_id="sp",
        chromosome_id=None,
        input_zarr=None,
        output_dir=None,
        model_checkpoint="test",
        model_path="test",
        window_size=8,
        stride=4,
        batch_size=batch,
        dtype="float32",
        tqdm_position=0,
        show_dynamo_errors=False,
        batch_size_cache=str(cache) if cache else None,
    )
    return loaded, fingerprinted, sizes, models


def test_model_loaded_once_oom_cap_reused_and_outputs_match(tmp_path, monkeypatch):
    baseline, baseline_entries = make_manifest(tmp_path / "baseline")
    resumed, entries = make_manifest(tmp_path / "worker")
    with monkeypatch.context() as patch:
        run_worker(baseline, patch, Classifier(), batch=2)
    classifier = LimitedClassifier(2)
    cache = tmp_path / "batch_size.txt"
    with monkeypatch.context() as patch:
        loaded, hashed, sizes, models = run_worker(
            resumed, patch, classifier, cache=cache
        )
    assert len(loaded) == len(hashed) == 1
    assert sizes == [4, 2, 2]
    assert all(model is classifier for model in models)
    assert classifier.attempts[:3] == [4, 3, 2]
    assert all(size <= 2 for size in classifier.attempts[2:])
    assert cache.read_text().strip() == "2"
    for expected, actual in zip(baseline_entries, entries):
        xr.testing.assert_equal(
            merge_prediction_datasets(expected["predictions_dir"]).load(),
            merge_prediction_datasets(actual["predictions_dir"]).load(),
        )
    # A completed manifest verifies the outputs but does not invoke the model.
    complete = Classifier()
    with monkeypatch.context() as patch:
        _, _, sizes, _ = run_worker(resumed, patch, complete, cache=cache)
    assert complete.calls == 0
    assert sizes == [2, 2, 2]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("model bug"),
        OSError("disk full"),
        torch.cuda.OutOfMemoryError("one window cannot fit"),
    ],
)
def test_worker_fails_without_reloading_or_retrying_unrecoverable_errors(
    tmp_path, monkeypatch, error
):
    manifest, entries = make_manifest(tmp_path / "worker", count=1)

    class Broken(Classifier):
        def __call__(self, *args, **kwargs):
            self.calls += 1
            raise error

    classifier = Broken()
    with pytest.raises(type(error), match=str(error)):
        run_worker(manifest, monkeypatch, classifier, batch=1)
    assert classifier.calls == 1
    assert not (Path(entries[0]["predictions_dir"]) / "_SUCCESS.json").exists()


def test_empty_manifest_needs_no_cuda_or_model(tmp_path, monkeypatch):
    manifest = tmp_path / "empty.json"
    manifest.write_text("[]")
    monkeypatch.setattr(
        predict.torch.cuda,
        "is_available",
        lambda: pytest.fail("CUDA probed for empty work"),
    )
    monkeypatch.setattr(
        predict, "load_models", lambda **kwargs: pytest.fail("model loaded")
    )
    predict.create_predictions(
        manifest=str(manifest),
        species_id="sp",
        chromosome_id=None,
        input_zarr=None,
        output_dir=None,
        model_checkpoint="test",
        model_path="test",
        window_size=8,
        stride=4,
        batch_size=2,
        dtype="float32",
        tqdm_position=0,
        show_dynamo_errors=False,
    )


def test_sequence_resources_released_between_scaffolds(tmp_path, monkeypatch):
    manifest, _ = make_manifest(tmp_path / "worker")
    original = predict.load_seq_data
    refs, closed = [], []

    def load(**kwargs):
        assert all(ref() is None for ref in refs)
        ds = original(**kwargs)
        close_tree = ds._close
        chrom = kwargs["chromosome_id"]

        def close():
            close_tree()
            closed.append(chrom)

        ds.set_close(close)
        refs.append(weakref.ref(ds))
        return ds

    monkeypatch.setattr(predict, "load_seq_data", load)
    run_worker(manifest, monkeypatch, Classifier(), batch=2)
    assert closed == ["scaffold0", "scaffold1", "scaffold2"]
    assert all(ref() is None for ref in refs)


def test_oom_after_committed_batch_preserves_that_batch(tmp_path, monkeypatch):
    baseline, expected = make_manifest(tmp_path / "baseline", count=1)
    resumed, actual = make_manifest(tmp_path / "worker", count=1)
    baseline_model = Classifier()
    with monkeypatch.context() as patch:
        run_worker(baseline, patch, baseline_model, batch=4)

    class LaterOOM(LimitedClassifier):
        def __call__(self, *args, **kwargs):
            if self.calls:
                self.limit = 2
            return super().__call__(*args, **kwargs)

    model = LaterOOM(4)
    retained = []

    def after_oom():
        retained.extend(checkpoint.segments(actual[0]["predictions_dir"], verify=True))

    with monkeypatch.context() as patch:
        patch.setattr(predict.torch.cuda, "empty_cache", after_oom)
        run_worker(resumed, patch, model, batch=4)
    assert retained
    assert model.windows == baseline_model.windows
    for record in retained:
        store = Path(actual[0]["predictions_dir"]) / record["store"]
        assert checkpoint.file_hashes(store) == record["files"]
    xr.testing.assert_equal(
        merge_prediction_datasets(expected[0]["predictions_dir"]).load(),
        merge_prediction_datasets(actual[0]["predictions_dir"]).load(),
    )


def shell_functions():
    text = (Path(__file__).resolve().parents[2] / "predict.sh").read_text()
    return text[
        text.index("process_chromosome() {") : text.index('\nif [[ "$FRAME_AWARE"')
    ]


@pytest.mark.parametrize("mode", ["single", "ddp", "ddp_slurm"])
@pytest.mark.parametrize("failure", [False, True])
def test_shell_starts_one_worker_per_gpu_and_preserves_progress(
    tmp_path, mode, failure
):
    output = tmp_path / "output with spaces"
    state = output / ".state"
    state.mkdir(parents=True)
    done = output / "done"
    done.mkdir()
    (done / "predictions_filtered_done.gff").touch()
    fake = tmp_path / "fake.py"
    fake.write_text("""import json, os, pathlib, sys
args = sys.argv[1:]
assert "--chromosome-id" not in args
path = pathlib.Path(args[args.index("--manifest") + 1])
entries = json.loads(path.read_text())
with path.with_suffix(".launches").open("a") as handle:
    handle.write("loaded once\\n")
for entry in entries:
    root = pathlib.Path(entry["predictions_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "saved_segment").write_text("keep")
    if os.environ["FAIL_WORKER"] == "1":
        sys.exit(1)
    (root / "_SUCCESS.json").write_text("{}")
""")
    launcher = tmp_path / "launch"
    launcher.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{fake}" "$@"\n')
    launcher.chmod(0o755)
    env = {
        **os.environ,
        "OUTPUT_DIR": str(output),
        "BATCH_SIZE_STATE_DIR": str(state),
        "CHROM_IDS": "done\nchr0\nchr1\nchr2\nchr3\nchr4",
        "GPU_LIST_STR": "0,1",
        "PREDICT_MODE": mode,
        "PYTHON": sys.executable,
        "PY_LAUNCHER": str(launcher),
        "SCRIPT_DIR": "/unused",
        "BASE_MODEL": "base",
        "HEAD_MODEL": "head",
        "SPECIES_ID": "sp",
        "DTYPE": "float32",
        "FAIL_WORKER": str(int(failure)),
    }
    script = (
        shell_functions()
        + "\ndeclare -A GPU_BATCH_SIZES=([0]=4 [1]=4)\nDDP_BATCH=4\nrun_prediction_workers"
    )
    result = subprocess.run(
        ["bash", "-c", script], env=env, text=True, capture_output=True
    )
    assert result.returncode == int(failure), result.stdout + result.stderr
    launches = list(state.glob("*.launches"))
    assert len(launches) == (2 if mode == "single" else 1)
    assert all(path.read_text() == "loaded once\n" for path in launches)
    manifests = [
        json.loads(path.read_text()) for path in state.glob("predict_manifest_*.json")
    ]
    assigned = [entry["chromosome_id"] for manifest in manifests for entry in manifest]
    assert sorted(assigned) == [f"chr{i}" for i in range(5)]
    assert list(output.glob("*/*/saved_segment"))  # failures never delete progress
    if not failure:
        assert len(list(output.glob("*/*/_SUCCESS.json"))) == 5
    # Re-running an entirely finished pipeline starts no model workers.
    for i in range(5):
        directory = output / f"chr{i}"
        directory.mkdir(exist_ok=True)
        (directory / f"predictions_filtered_chr{i}.gff").touch()
    again = subprocess.run(
        ["bash", "-c", script], env=env, text=True, capture_output=True
    )
    assert again.returncode == 0
    assert all(path.read_text() == "loaded once\n" for path in launches)


def test_cpu_stages_reject_incomplete_prediction(tmp_path):
    chrom = tmp_path / "chr"
    chrom.mkdir()
    (chrom / "sequences_chr.zarr").mkdir()
    env = {**os.environ, "OUTPUT_DIR": str(tmp_path)}
    result = subprocess.run(
        ["bash", "-c", shell_functions() + "\nprocess_chromosome chr 4 0"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "refusing downstream" in result.stdout


def test_large_scaffold_list_uses_files_instead_of_environment(tmp_path):
    names = [f"scaffold_{i:06d}" for i in range(30000)]
    source = tmp_path / "ids.txt"
    source.write_text("\n".join(names))
    state = tmp_path / ".state"
    state.mkdir()
    env = {
        **os.environ,
        "SOURCE_IDS": str(source),
        "OUTPUT_DIR": str(tmp_path),
        "BATCH_SIZE_STATE_DIR": str(state),
        "GPU_LIST_STR": "0,1",
        "PREDICT_MODE": "single",
        "PYTHON": sys.executable,
    }
    script = (
        shell_functions()
        + """
CHROM_IDS=$(cat "$SOURCE_IDS")
declare -A GPU_BATCH_SIZES=([0]=4 [1]=4)
run_prediction_manifest() { return 0; }
run_prediction_workers
"""
    )
    result = subprocess.run(
        ["bash", "-c", script], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    entries = [
        entry
        for path in state.glob("predict_manifest_*.json")
        for entry in json.loads(path.read_text())
    ]
    assert sorted(entry["chromosome_id"] for entry in entries) == names

    # The FASTA extraction manifest must use the same file-based transport.
    text = (Path(__file__).resolve().parents[2] / "predict.sh").read_text()
    extraction = text[
        text.index('EXTRACT_MANIFEST="$BATCH_SIZE_STATE_DIR/') : text.index(
            '\nif [[ "$EXTRACT_MANIFEST_COUNT"'
        )
    ]
    result = subprocess.run(
        ["bash", "-c", 'CHROM_IDS=$(cat "$SOURCE_IDS")\n' + extraction],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    entries = json.loads((state / "extract_manifest.json").read_text())
    assert [entry["chromosome_id"] for entry in entries] == names
