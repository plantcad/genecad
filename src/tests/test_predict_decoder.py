"""Tests for predict.sh's --decoder / --keep-partial handling (steps 7-8)."""

import subprocess
from pathlib import Path

import pytest

PREDICT_SH = (Path(__file__).resolve().parents[2] / "predict.sh").read_text()


def option_parsing() -> str:
    start = PREDICT_SH.index('INPUT_FILE="data/example/')
    end = PREDICT_SH.index('echo "GeneCAD Prediction Pipeline"')
    return PREDICT_SH[start:end]


def parse(*args: str) -> subprocess.CompletedProcess:
    script = (
        "usage() { exit 2; }\n"
        + option_parsing()
        + '\necho "$DECODER $FRAME_AWARE $KEEP_PARTIAL $MERGE_MAX_GAP"'
    )
    return subprocess.run(
        ["bash", "-c", script, "predict.sh", *args], text=True, capture_output=True
    )


@pytest.mark.parametrize(
    "args,expected",
    [
        ((), "hybrid 0 0 20000"),
        (("--decoder", "frame-aware"), "frame-aware 1 0 20000"),
        (("--decoder", "plain"), "plain 0 0 20000"),
        (("--no-frame-aware",), "plain 0 0 20000"),
        (("--keep-partial",), "hybrid 0 1 20000"),
        (("--merge-max-gap", "10000"), "hybrid 0 0 10000"),
    ],
)
def test_decoder_options(args, expected):
    result = parse(*args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == expected


@pytest.mark.parametrize("args", [("--decoder", "viterbi"), ("--merge-max-gap", "far")])
def test_invalid_decoder_options_are_rejected(args):
    assert parse(*args).returncode != 0


def post_processing() -> str:
    start = PREDICT_SH.index('echo "[7/8] Repairing CDS boundaries')
    end = PREDICT_SH.index('\necho "All done!')
    return PREDICT_SH[start:end]


def run_post_processing(tmp_path, decoder: str, keep_partial: str) -> list[list[str]]:
    """Run steps 7-8 with a fake Python that logs each command line."""
    log = tmp_path / "calls.log"
    fake = tmp_path / "python"
    fake.write_text(f'#!/bin/bash\necho "$*" >> {log}\n')
    fake.chmod(0o755)
    out = tmp_path / "out"
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHON": str(fake),
        "SCRIPT_DIR": "/genecad",
        "INPUT_FILE": "genome.fa",
        "OUTPUT_DIR": str(out),
        "SPECIES_ID": "Sp",
        "MODE": "plant",
        "RAW_GFF": f"{out}/Sp_GeneCAD_raw.gff",
        "ORF_GFF": f"{out}/Sp_GeneCAD_orf.gff",
        "FINAL_GFF": f"{out}/Sp_GeneCAD_final.gff",
        "ORF_MAX_SHIFT": "300",
        "GPU_LIST_STR": "0",
        "CPU_WORKERS": "4",
        "DECODER": decoder,
        "KEEP_PARTIAL": keep_partial,
        "MERGE_MAX_GAP": "20000",
        "MIN_INTRON_LENGTH": "20",
        "MIN_CODING_RUN_LENGTH": "9",
        "EXON_LENGTH_STRICTNESS": "16",
        "ALLOW_U12_INTRONS": "0",
    }
    result = subprocess.run(
        ["bash", "-c", post_processing()], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [line.split() for line in log.read_text().splitlines()]


def call(calls, script):
    return next(c for c in calls if c[0].endswith(script))


def value(args, flag):
    return args[args.index(flag) + 1]


def test_hybrid_keeps_partials_for_rescue_then_refines_the_hybrid_gff(tmp_path):
    calls = run_post_processing(tmp_path, "hybrid", "0")
    assert [c[0].rsplit("/", 1)[-1] for c in calls] == [
        "fix_orf.py",
        "hybrid_decode.py",
        "refine.py",
    ]

    fix_orf, hybrid, refine = calls
    assert "--drop-partial" not in fix_orf
    assert value(hybrid, "--input-gff") == value(fix_orf, "--output-gff")
    assert value(hybrid, "--predictions-root") == str(tmp_path / "out")
    assert value(hybrid, "--max-gap") == "20000"
    assert value(hybrid, "--workers") == "4"
    assert value(hybrid, "--min-intron-length") == "20"
    assert "--keep-partial" not in hybrid
    assert value(refine, "--input-gff") == value(hybrid, "--output-gff")


def test_hybrid_keep_partial_is_passed_to_the_hybrid_step(tmp_path):
    calls = run_post_processing(tmp_path, "hybrid", "1")
    assert "--keep-partial" in call(calls, "hybrid_decode.py")


@pytest.mark.parametrize("decoder", ["frame-aware", "plain"])
def test_other_decoders_drop_partials_in_fix_orf_and_skip_hybrid(tmp_path, decoder):
    calls = run_post_processing(tmp_path, decoder, "0")
    assert [c[0].rsplit("/", 1)[-1] for c in calls] == ["fix_orf.py", "refine.py"]
    fix_orf, refine = calls
    assert "--drop-partial" in fix_orf
    assert value(refine, "--input-gff") == value(fix_orf, "--output-gff")


def test_keep_partial_disables_dropping(tmp_path):
    fix_orf = call(run_post_processing(tmp_path, "frame-aware", "1"), "fix_orf.py")
    assert "--drop-partial" not in fix_orf
