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
        "INTERMEDIATE_DIR": f"{out}/intermediate",
        "RAW_GFF": f"{out}/intermediate/Sp_GeneCAD_raw.gff",
        "ORF_GFF": f"{out}/intermediate/Sp_GeneCAD_orf.gff",
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


# -------------------------------------------------------------------------------------------------
# --cpu-stage-parallel: how many chromosomes run their CPU stages at once
# -------------------------------------------------------------------------------------------------


def shell_function(name: str) -> str:
    start = PREDICT_SH.index(f"{name}() {{")
    return PREDICT_SH[start : PREDICT_SH.index("\n}\n", start) + 3]


def resolve(requested: str, gpus: int, largest_mb: int, available_gb: int) -> str:
    script = (
        shell_function("resolve_cpu_stage_parallel")
        + f"resolve_cpu_stage_parallel {requested} {gpus} "
        + f"{largest_mb * 1_000_000} {available_gb * 1024 * 1024}"
    )
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize(
    "gpus,largest_mb,available_gb,expected",
    [
        (4, 300, 256, "2"),  # 300 Mb needs ~105 GB each; 0.9 * 256 GB fits two
        (4, 50, 256, "4"),  # small chromosomes: never more than one per GPU
        (1, 300, 256, "1"),
        (4, 800, 256, "1"),  # does not fit even alone: still run one at a time
    ],
)
def test_auto_cpu_stage_parallel_fits_the_largest_chromosome_in_memory(
    gpus, largest_mb, available_gb, expected
):
    assert resolve("auto", gpus, largest_mb, available_gb) == expected


def test_explicit_cpu_stage_parallel_is_used_as_given():
    assert resolve("3", 4, 800, 16) == "3"


def test_cpu_stage_parallel_option():
    script = "usage() { exit 2; }\n" + option_parsing() + '\necho "$CPU_STAGE_PARALLEL"'

    def run(*args):
        return subprocess.run(
            ["bash", "-c", script, "predict.sh", *args], text=True, capture_output=True
        )

    assert run().stdout.strip().splitlines()[-1] == "auto"
    assert run("--cpu-stage-parallel", "2").stdout.strip().splitlines()[-1] == "2"
    assert run("--cpu-stage-parallel", "0").returncode != 0
    assert run("--cpu-stage-parallel", "many").returncode != 0


def test_chromosome_cpu_stages_never_exceed_the_limit(tmp_path):
    """Four GPUs but a limit of two: at most two process_chromosome calls overlap."""
    start = PREDICT_SH.index("    declare -a PIDS=()")
    end = PREDICT_SH.index("\nif [[ $FAILED -gt 0 ]]", start)
    loop = PREDICT_SH[start:end].rsplit("\nfi", 1)[0]
    log = tmp_path / "log"
    script = (
        f'process_chromosome() {{ echo "+ $1" >> {log}; sleep 0.3; echo "- $1" >> {log}; }}\n'
        "GPU_ARRAY=(0 1 2 3); NUM_GPUS=4; CPU_STAGE_JOBS=2; FAILED=0\n"
        "declare -A GPU_BATCH_SIZES=([0]=8 [1]=8 [2]=8 [3]=8)\n"
        "CHR_ARRAY=(c1 c2 c3 c4 c5 c6)\n" + loop
    )
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

    running = peak = 0
    for line in log.read_text().splitlines():
        running += 1 if line.startswith("+") else -1
        peak = max(peak, running)
    assert peak == 2
    assert sum(line.startswith("-") for line in log.read_text().splitlines()) == 6


# -------------------------------------------------------------------------------------------------
# Output layout: one final GFF at the top, intermediates in intermediate/
# -------------------------------------------------------------------------------------------------


def test_intermediates_go_to_their_own_folder_and_the_final_gff_is_named_at_the_end(
    tmp_path,
):
    start = PREDICT_SH.index('echo "[6/8] Merging per-chromosome GFFs')
    log = tmp_path / "calls.log"
    fake = tmp_path / "python"
    fake.write_text(f'#!/bin/bash\necho "$*" >> {log}\n')
    fake.chmod(0o755)
    out = tmp_path / "out"
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHON": str(fake),
        "SCRIPT_DIR": "/genecad",
        "MERGE_SCRIPT": "/genecad/scripts/merge_gff.py",
        "INPUT_FILE": "genome.fa",
        "OUTPUT_DIR": str(out),
        "SPECIES_ID": "Sp",
        "MODE": "plant",
        "ORF_MAX_SHIFT": "300",
        "GPU_LIST_STR": "0",
        "CPU_WORKERS": "1",
        "DECODER": "hybrid",
        "KEEP_PARTIAL": "0",
        "MERGE_MAX_GAP": "20000",
        "MIN_INTRON_LENGTH": "20",
        "MIN_CODING_RUN_LENGTH": "9",
        "EXON_LENGTH_STRICTNESS": "16",
        "ALLOW_U12_INTRONS": "0",
    }
    result = subprocess.run(
        ["bash", "-c", "RECALL_GFFS=()\n" + PREDICT_SH[start:]],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    calls = {
        c.split()[0].rsplit("/", 1)[-1]: c.split() for c in log.read_text().splitlines()
    }
    intermediate = str(out / "intermediate")
    assert value(calls["merge_gff.py"], "--output-gff").startswith(intermediate + "/")
    assert value(calls["fix_orf.py"], "--output-gff").startswith(intermediate + "/")
    assert value(calls["hybrid_decode.py"], "--output-gff").startswith(
        intermediate + "/"
    )
    final = str(out / "Sp_GeneCAD_final.gff")
    assert value(calls["refine.py"], "--output-gff") == final

    summary = result.stdout[result.stdout.index("All done!") :]
    assert f"Final annotation (use this file):\n  {final}" in summary
