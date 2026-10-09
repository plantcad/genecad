"""Persistent model lifetime, real Zarr inference, and shell dispatch integration."""

import gzip
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
    start = text.index("process_chromosome() {")
    return text[start : text.index('\nif [[ "$FRAME_AWARE"', start)]


def chromosome_discovery():
    text = (Path(__file__).resolve().parents[2] / "predict.sh").read_text()
    section = text.index("# Step 1: Discover chromosomes from FASTA headers")
    start = text.index("# Sequences without bases have no genes", section)
    end = text.index("\nCHROM_COUNT=", start)
    return text[start:end]


@pytest.mark.parametrize("compressed", [False, True])
def test_top_n_contigs_selects_longest_in_fasta_order(tmp_path, compressed):
    contents = """>middle description
AAAA
AAAA
>short
AAA
>long
CCCCCC
CCCCCC
>tiny
TT
"""
    fasta = tmp_path / ("input.fa.gz" if compressed else "input.fa")
    if compressed:
        with gzip.open(fasta, "wt") as handle:
            handle.write(contents)
    else:
        fasta.write_text(contents)

    result = subprocess.run(
        ["bash", "-c", chromosome_discovery() + '\nprintf "%s\\n" "$CHROM_IDS"'],
        env={
            **os.environ,
            "INPUT_FILE": str(fasta),
            "TOP_N_CONTIGS": "2",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["middle", "long"]


def discovered(tmp_path, contents, top, compressed=False):
    fasta = tmp_path / ("input.fa.gz" if compressed else "input.fa")
    if compressed:
        with gzip.open(fasta, "wt") as handle:
            handle.write(contents)
    else:
        fasta.write_text(contents)
    result = subprocess.run(
        ["bash", "-c", chromosome_discovery() + '\nprintf "%s\\n" "$CHROM_IDS"'],
        env={**os.environ, "INPUT_FILE": str(fasta), "TOP_N_CONTIGS": top},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.splitlines()


WITH_EMPTY = (
    ">chr1 desc\nACGT\nAC\n>empty1\n>chr2\nGGG\n\n>empty2\n\n>chr3\nA\n>empty3\n"
)


@pytest.mark.parametrize("compressed", [False, True])
def test_sequences_without_bases_are_skipped(tmp_path, compressed):
    assert discovered(tmp_path, WITH_EMPTY, "all", compressed) == [
        "WARNING: Skipping 3 sequence(s) without bases: empty1 empty2 empty3",
        "chr1",
        "chr2",
        "chr3",
    ]
    assert discovered(tmp_path, WITH_EMPTY, "5", compressed) == ["chr1", "chr2", "chr3"]
    assert discovered(tmp_path, WITH_EMPTY, "2", compressed) == ["chr1", "chr2"]


def test_all_sequences_are_listed_in_fasta_order(tmp_path):
    contents = "\n>b x\nAC\n>a\nG\n>c\tdesc\nT"
    assert discovered(tmp_path, contents, "all") == ["b", "a", "c"]


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
    extraction = (
        text[
            text.index('EXTRACT_MANIFEST="$BATCH_SIZE_STATE_DIR/') : text.index(
                "extract_needed_sequences() {"
            )
        ]
        + text[
            text.index("EXTRACT_MANIFEST_COUNT=$(") : text.index(
                '\nif [[ "$EXTRACT_MANIFEST_COUNT"'
            )
        ]
    )
    result = subprocess.run(
        ["bash", "-c", 'CHROM_IDS=$(cat "$SOURCE_IDS")\n' + extraction],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    entries = json.loads((state / "extract_manifest.json").read_text())
    assert [entry["chromosome_id"] for entry in entries] == names


# A Python that runs heredocs for real and fakes the three decoding steps: each step
# writes its output, and fails at the sequence named in FAIL_ID (after the earlier ones).
FAKE_STEPS = r"""import json, os, pathlib, sys
args = sys.argv[1:]
script = pathlib.Path(args[0]).name
if "--manifest" in args:
    entries = json.loads(pathlib.Path(args[args.index("--manifest") + 1]).read_text())
else:
    flag = {"detect_intervals.py": "--output-zarr", "export_gff.py": "--output-gff",
            "filter_raw_gff.py": "--output-gff"}[script]
    output = args[args.index(flag) + 1]
    entries = [{"chromosome_id": pathlib.Path(output).parent.name,
                "intervals_zarr": output, "raw_gff": output, "filtered_gff": output}]
with open(os.environ["CALLS"], "a") as handle:
    ids = [entry["chromosome_id"] for entry in entries]
    handle.write(json.dumps({"args": args, "ids": ids}) + "\n")
for entry in entries:
    if entry["chromosome_id"] == os.environ.get("FAIL_ID"):
        sys.exit(7)
    if script == "detect_intervals.py":
        pathlib.Path(entry["intervals_zarr"]).mkdir()
    elif script == "export_gff.py":
        pathlib.Path(entry["raw_gff"]).write_text("raw")
    else:
        pathlib.Path(entry["filtered_gff"]).write_text("filtered")
"""


def decoding_env(tmp_path, ids, **extra):
    output = tmp_path / "out"
    state = output / ".state"
    state.mkdir(parents=True)
    for chrom in ids:
        predictions = output / chrom / f"predictions_{chrom}"
        predictions.mkdir(parents=True)
        (predictions / "_SUCCESS.json").write_text("{}")
        (output / chrom / f"sequences_{chrom}.zarr").mkdir()
    steps = tmp_path / "steps.py"
    steps.write_text(FAKE_STEPS)
    fake = tmp_path / "python"
    fake.write_text(
        f'#!/bin/bash\n[[ "$1" == - ]] && exec "{sys.executable}" "$@"\n'
        f'exec "{sys.executable}" "{steps}" "$@"\n'
    )
    fake.chmod(0o755)
    return output, {
        **os.environ,
        "OUTPUT_DIR": str(output),
        "BATCH_SIZE_STATE_DIR": str(state),
        "DECODE_FAILURES": str(state / "decode_failures.txt"),
        "PYTHON": str(fake),
        "SCRIPT_DIR": "/genecad",
        "MODE": "plant",
        "MIN_TRANSCRIPT_LENGTH": "3",
        "CPU_WORKERS": "1",
        "CLEAN_INTERMEDIATES": "0",
        "DECODE_SETTINGS": "mode=plant frame_aware=0",
        "CALLS": str(tmp_path / "calls.jsonl"),
        **extra,
    }


def run_group(tmp_path, ids, env):
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -e\nFRAME_AWARE_ARGS=()\n"
            + shell_functions()
            + "\nprocess_chromosome_group 1 "
            + " ".join(ids),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    log = (tmp_path / "calls.jsonl").read_text().splitlines()
    return result, [json.loads(line) for line in log]


def steps(calls):
    """(script, grouped, sequences) for each call."""
    return [
        (Path(c["args"][0]).name, "--manifest" in c["args"], c["ids"]) for c in calls
    ]


def test_short_sequences_are_decoded_together(tmp_path):
    ids = ["s1", "s2", "s3"]
    output, env = decoding_env(tmp_path, ids)
    # s2 was decoded up to its intervals before.
    (output / "s2" / "intervals_s2.zarr").mkdir()
    result, calls = run_group(tmp_path, ids, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert steps(calls) == [
        ("detect_intervals.py", True, ["s1", "s3"]),
        ("export_gff.py", True, ids),
        ("filter_raw_gff.py", True, ids),
    ]
    detect, export = calls[0]["args"], calls[1]["args"]
    assert detect[detect.index("--domain") + 1] == "plant"
    assert export[export.index("--tqdm-position") + 1] == "1"
    assert export[export.index("--min-transcript-length") + 1] == "3"
    for chrom in ids:
        directory = output / chrom
        assert (
            directory / f"predictions_filtered_{chrom}.gff"
        ).read_text() == "filtered"
        assert (
            directory / "decoding_settings.txt"
        ).read_text() == "mode=plant frame_aware=0\n"
    assert "[s1 and 2 more@GPU1] Done!" in result.stdout
    # Manifests are removed afterwards.
    assert not list((output / ".state").glob("decode_group.*"))


def test_a_group_only_runs_the_steps_its_sequences_need(tmp_path):
    ids = ["s1", "s2"]
    output, env = decoding_env(tmp_path, ids)
    for chrom in ids:
        (output / chrom / f"intervals_{chrom}.zarr").mkdir()
    (output / "s1" / "predictions_raw_s1.gff").write_text("raw")
    result, calls = run_group(tmp_path, ids, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert steps(calls) == [
        ("export_gff.py", True, ["s2"]),
        ("filter_raw_gff.py", True, ids),
    ]


def test_a_failed_step_decodes_the_rest_one_at_a_time(tmp_path):
    ids = ["s1", "s2", "s3"]
    output, env = decoding_env(tmp_path, ids, FAIL_ID="s2")
    result, calls = run_group(tmp_path, ids, env)
    assert result.returncode == 1
    assert "decoding the unfinished sequences one at a time" in result.stdout
    for chrom in ("s1", "s3"):
        directory = output / chrom
        assert (directory / f"predictions_filtered_{chrom}.gff").is_file()
        assert (directory / "decoding_settings.txt").is_file()
    assert not (output / "s2" / "predictions_filtered_s2.gff").exists()
    assert not (output / "s2" / "decoding_settings.txt").exists()
    assert (output / ".state" / "decode_failures.txt").read_text() == "s2\n"
    # One grouped detect_intervals call, then s1 (its intervals already made by the
    # group), s2 and s3 one at a time.
    assert steps(calls) == [
        ("detect_intervals.py", True, ids),
        ("export_gff.py", False, ["s1"]),
        ("filter_raw_gff.py", False, ["s1"]),
        ("detect_intervals.py", False, ["s2"]),
        ("detect_intervals.py", False, ["s3"]),
        ("export_gff.py", False, ["s3"]),
        ("filter_raw_gff.py", False, ["s3"]),
    ]


def plan_jobs(tmp_path, chromosomes, parallel, short_bp=1000, group_bp=2000):
    """Run the planner on an output folder; chromosomes maps ID to (length, state)."""
    start = PREDICT_SH_TEXT.index("SHORT_SEQUENCE_BP=1000000")
    end = PREDICT_SH_TEXT.index("# Sequences that failed, one per line")
    output = tmp_path / "out"
    state = output / ".state"
    state.mkdir(parents=True)
    for chrom, (length, status) in chromosomes.items():
        directory = output / chrom
        predictions = directory / f"predictions_{chrom}"
        predictions.mkdir(parents=True)
        if status != "unpredicted":
            (predictions / "_SUCCESS.json").write_text("{}")
        if length is not None:
            (predictions / "run.json").write_text(json.dumps({"length": length}))
        if status != "no sequences":
            (directory / f"sequences_{chrom}.zarr").mkdir()
        if status == "done":
            (directory / f"predictions_filtered_{chrom}.gff").touch()
    ids = state / "chromosome_ids.txt"
    ids.write_text("".join(f"{c}\n" for c in chromosomes))
    section = (
        PREDICT_SH_TEXT[start:end]
        .replace("SHORT_SEQUENCE_BP=1000000", f"SHORT_SEQUENCE_BP={short_bp}")
        .replace("GROUP_SEQUENCE_BP=20000000", f"GROUP_SEQUENCE_BP={group_bp}")
    )
    result = subprocess.run(
        ["bash", "-c", section + '\nprintf "%s\\n" "${DECODE_JOBS[@]}"'],
        env={
            **os.environ,
            "OUTPUT_DIR": str(output),
            "BATCH_SIZE_STATE_DIR": str(state),
            "CHROM_IDS_FILE": str(ids),
            "PARALLEL_CHROMOSOMES": str(parallel),
            "PYTHON": sys.executable,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.splitlines()


PREDICT_SH_TEXT = (Path(__file__).resolve().parents[2] / "predict.sh").read_text()


def test_short_sequences_are_planned_in_balanced_groups(tmp_path):
    chromosomes = {
        "big": (5000, "ready"),
        "a": (900, "ready"),
        "b": (100, "ready"),
        "done": (10, "done"),
        "c": (800, "ready"),
        "d": (300, "ready"),
        "unpredicted": (10, "unpredicted"),
        "legacy": (None, "ready"),
        "e": (200, "ready"),
        "lost": (10, "no sequences"),
    }
    jobs = plan_jobs(tmp_path, chromosomes, parallel=2)
    # Long, finished and unusual sequences stay on their own, in FASTA order, so that
    # process_chromosome reports or handles them as before. 2300 short bases make two
    # groups (at least PARALLEL_CHROMOSOMES), filled longest first.
    assert jobs == ["big", "done", "unpredicted", "legacy", "lost", "a b e", "c d"]


def test_groups_hold_about_group_sequence_bp_bases(tmp_path):
    chromosomes = {f"s{i}": (100, "ready") for i in range(12)}
    jobs = plan_jobs(tmp_path, chromosomes, parallel=1, group_bp=400)
    assert jobs == ["s0 s3 s6 s9", "s1 s4 s7 s10", "s2 s5 s8 s11"]
    # Fewer short sequences than jobs at once: each is a job of its own.
    assert plan_jobs(
        tmp_path / "few", {"x": (5, "ready"), "y": (6, "ready")}, parallel=4
    ) == [
        "x",
        "y",
    ]
