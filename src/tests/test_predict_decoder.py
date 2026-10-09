"""Tests for predict.sh's --decoder / --keep-partial handling (steps 7-8)."""

import os
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


def test_no_frame_aware_still_works_but_warns_that_it_is_deprecated():
    result = parse("--no-frame-aware")
    assert result.stdout.strip().splitlines()[-1] == "plain 0 0 20000"
    assert "deprecated" in result.stderr
    assert "--decoder plain" in result.stderr
    assert "deprecated" not in parse("--decoder", "plain").stderr


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
    # Steps that are not under test here: skip-checks and predicting deleted files again.
    stubs = (
        "refresh_stage() { :; }\nrecord_stage() { :; }\n"
        "extract_needed_sequences() { :; }\nrun_prediction_workers() { :; }\n"
    )
    env["ALLOW_MISSING_PREDICTIONS"] = "0"
    result = subprocess.run(
        ["bash", "-c", stubs + post_processing()],
        env=env,
        text=True,
        capture_output=True,
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
    assert "--keep-partial" in fix_orf
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
def test_other_decoders_let_fix_orf_drop_partials_and_skip_hybrid(tmp_path, decoder):
    calls = run_post_processing(tmp_path, decoder, "0")
    assert [c[0].rsplit("/", 1)[-1] for c in calls] == ["fix_orf.py", "refine.py"]
    fix_orf, refine = calls
    assert "--keep-partial" not in fix_orf
    assert value(refine, "--input-gff") == value(fix_orf, "--output-gff")


def test_keep_partial_is_passed_on_to_fix_orf(tmp_path):
    fix_orf = call(run_post_processing(tmp_path, "frame-aware", "1"), "fix_orf.py")
    assert "--keep-partial" in fix_orf


# -------------------------------------------------------------------------------------------------
# --max-parallel-chromosomes: how many chromosomes are decoded and exported at once
# -------------------------------------------------------------------------------------------------


def shell_function(name: str) -> str:
    start = PREDICT_SH.index(f"{name}() {{")
    return PREDICT_SH[start : PREDICT_SH.index("\n}\n", start) + 3]


def resolve(
    requested: str,
    max_auto: int,
    largest_mb: int,
    available_gb: int,
    gb_per_mb: str = "",
) -> str:
    script = (
        shell_function("resolve_parallel_chromosomes")
        + f"resolve_parallel_chromosomes {requested} {max_auto} "
        + f"{largest_mb * 1_000_000} {available_gb * 1024 * 1024} {gb_per_mb}"
    )
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize(
    "max_auto,largest_mb,available_gb,expected",
    [
        (4, 300, 256, "2"),  # 300 Mb needs ~105 GB each; 0.9 * 256 GB fits two
        (4, 50, 256, "4"),  # small chromosomes: capped at max_auto
        (16, 5, 256, "16"),  # thousands of small scaffolds: many at once
        (1, 300, 256, "1"),
        (4, 800, 256, "1"),  # does not fit even alone: still run one at a time
    ],
)
def test_auto_parallel_chromosomes_fit_the_largest_chromosome_in_memory(
    max_auto, largest_mb, available_gb, expected
):
    assert resolve("auto", max_auto, largest_mb, available_gb) == expected


@pytest.mark.parametrize(
    "max_auto,largest_mb,available_gb,expected",
    [
        (4, 300, 256, "4"),  # 15 GB each: capped at max_auto
        (4, 1800, 256, "2"),  # 90 GB each: 0.9 * 256 GB fits two
        (4, 1800, 64, "1"),  # does not fit even alone: still run one at a time
    ],
)
def test_plain_and_hybrid_decoding_need_far_less_memory_per_mb(
    max_auto, largest_mb, available_gb, expected
):
    assert resolve("auto", max_auto, largest_mb, available_gb, "0.05") == expected


@pytest.mark.parametrize("decoder,per_mb", [("frame-aware", "0.35"), ("plain", "0.05")])
def test_memory_per_mb_follows_the_decoder(decoder, per_mb):
    section = PREDICT_SH[PREDICT_SH.index('if [[ "$DECODER" == "frame-aware" ]]') :]
    script = f'DECODER={decoder}\n{section[: section.index("fi") + 2]}\necho "$DECODE_GB_PER_MB"'
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    assert result.stdout.strip() == per_mb


def test_explicit_parallel_chromosomes_are_used_as_given():
    assert resolve("3", 4, 800, 16) == "3"


def test_max_parallel_chromosomes_option():
    script = (
        "usage() { exit 2; }\n"
        + option_parsing()
        + '\necho "$MAX_PARALLEL_CHROMOSOMES"'
    )

    def run(*args):
        return subprocess.run(
            ["bash", "-c", script, "predict.sh", *args], text=True, capture_output=True
        )

    assert run().stdout.strip().splitlines()[-1] == "auto"
    assert run("--max-parallel-chromosomes", "2").stdout.strip().splitlines()[-1] == "2"
    assert run("--max-parallel-chromosomes", "0").returncode != 0
    assert run("--max-parallel-chromosomes", "many").returncode != 0
    # The name used before this option was renamed is not accepted any more
    assert run("--cpu-stage-parallel", "2").returncode != 0


def decode_loop(tmp_path, jobs, mode="single", fail=()):
    """Run the decoding loop with process_chromosome(_group) replaced by functions that
    log when they start and end, and fail for the IDs in fail."""
    start = PREDICT_SH.index("# Sequences that failed, one per line")
    end = PREDICT_SH.index("\nRECALL_GFFS=()", start)
    log = tmp_path / "log"
    script = (
        f'job() {{ echo "+ $1" >> {log}; sleep 0.3; echo "- $1" >> {log}; }}\n'
        f'fails() {{ [[ " {" ".join(fail)} " == *" $1 "* ]]; }}\n'
        'process_chromosome() { job "$1"; ! fails "$1"; }\n'
        "process_chromosome_group() {\n"
        '    shift; job "$*"; local id status=0\n'
        '    for id in "$@"; do\n'
        '        if fails "$id"; then echo "$id" >> "$DECODE_FAILURES"; status=1; fi\n'
        "    done\n"
        '    return "$status"\n'
        "}\n"
        f"BATCH_SIZE_STATE_DIR={tmp_path}; PREDICT_MODE={mode}; DDP_BATCH=8\n"
        "GPU_ARRAY=(0 1 2 3); NUM_GPUS=4; PARALLEL_CHROMOSOMES=2\n"
        "declare -A GPU_BATCH_SIZES=([0]=8 [1]=8 [2]=8 [3]=8)\n"
        "DECODE_JOBS=(" + " ".join(f"'{j}'" for j in jobs) + ")\n"
        "set -e\n" + PREDICT_SH[start:end]
    )
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    return result, log


def test_chromosomes_decoded_at_once_never_exceed_the_limit(tmp_path):
    """Four GPUs but a limit of two: at most two process_chromosome calls overlap."""
    result, log = decode_loop(tmp_path, ["c1", "c2", "c3", "c4", "c5", "c6"])
    assert result.returncode == 0, result.stdout + result.stderr

    running = peak = 0
    for line in log.read_text().splitlines():
        running += 1 if line.startswith("+") else -1
        peak = max(peak, running)
    assert peak == 2
    assert sum(line.startswith("-") for line in log.read_text().splitlines()) == 6


@pytest.mark.parametrize("mode", ["single", "ddp"])
def test_failed_sequences_are_counted_one_by_one_also_in_groups(tmp_path, mode):
    jobs = ["big1", "a b c", "big2", "d e"]
    result, log = decode_loop(tmp_path, jobs, mode, fail=("big2", "a", "c", "e"))
    assert result.returncode == 1
    assert "ERROR: 4 chromosome(s) failed." in result.stdout
    started = [line[2:] for line in log.read_text().splitlines() if line[0] == "+"]
    assert sorted(started) == sorted(jobs)
    assert sorted((tmp_path / "decode_failures.txt").read_text().split()) == [
        "a",
        "big2",
        "c",
        "e",
    ]


def test_no_failure_message_when_every_job_succeeds(tmp_path):
    result, _ = decode_loop(tmp_path, ["big1", "a b c"], "ddp")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ERROR" not in result.stdout


# -------------------------------------------------------------------------------------------------
# Output layout: one final GFF at the top, intermediates in intermediate/
# -------------------------------------------------------------------------------------------------


def test_intermediates_go_to_their_own_folder_and_the_final_gff_is_named_at_the_end(
    tmp_path,
):
    start = PREDICT_SH.index('echo "[6/8] Merging per-chromosome GFFs')
    log = tmp_path / "calls.log"
    fake = tmp_path / "python"
    fake.write_text(
        f'#!/bin/bash\necho "$*" >> {log}\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  if [[ "$1" == --output-gff ]]; then mkdir -p "$(dirname "$2")"; : > "$2"; fi\n'
        "  shift\n"
        "done\n"
    )
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
    stubs = (
        "refresh_stage() { :; }\nrecord_stage() { :; }\n"
        "extract_needed_sequences() { :; }\nrun_prediction_workers() { :; }\n"
    )
    env["ALLOW_MISSING_PREDICTIONS"] = "0"
    result = subprocess.run(
        ["bash", "-c", stubs + "RECALL_GFFS=()\n" + PREDICT_SH[start:]],
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


def test_available_cores_ignores_omp_num_threads():
    script = shell_function("available_cores") + "available_cores"
    expected = subprocess.run(
        ["bash", "-c", "unset OMP_NUM_THREADS; nproc"], text=True, capture_output=True
    ).stdout.strip()
    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.stdout.strip() == expected


def test_available_cores_does_not_need_env():
    script = (
        "env() { echo 'broken env' >&2; return 126; }\n"
        + shell_function("available_cores")
        + "available_cores"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert int(result.stdout.strip()) > 1 or os.cpu_count() == 1


def test_available_cores_falls_back_to_one_without_nproc_and_getconf():
    script = (
        "nproc() { return 127; }\ngetconf() { return 127; }\n"
        + shell_function("available_cores")
        + "available_cores"
    )
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    assert result.stdout.strip() == "1"


def run_functions(call: str, names: list[str], env: dict[str, str]) -> str:
    script = "".join(shell_function(name) for name in names) + call
    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        env={**os.environ, **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def cgroup_env(tmp_path: Path, proc_cgroup: str) -> dict[str, str]:
    write(tmp_path / "proc_cgroup", proc_cgroup)
    return {
        "CGROUP_ROOT": str(tmp_path / "cg"),
        "CGROUP_FILE": str(tmp_path / "proc_cgroup"),
    }


def test_cgroup_v2_limit_is_found_in_a_parent_cgroup(tmp_path):
    root = tmp_path / "cg"
    write(root / "cgroup.controllers", "cpu memory")
    write(root / "job.slice" / "memory.max", str(8 * 1024**3))
    write(root / "job.slice" / "step" / "memory.max", "max")
    write(root / "job.slice" / "cpu.max", "400000 100000")
    env = cgroup_env(tmp_path, "0::/job.slice/step\n")
    assert run_functions("cgroup_limit memory", ["cgroup_limit"], env) == str(
        8 * 1024 * 1024
    )
    assert run_functions("cgroup_limit cpu", ["cgroup_limit"], env) == "4"


def test_cgroup_v2_without_a_limit_prints_nothing(tmp_path):
    root = tmp_path / "cg"
    write(root / "cgroup.controllers", "cpu memory")
    write(root / "memory.max", "max")
    write(root / "cpu.max", "max 100000")
    env = cgroup_env(tmp_path, "0::/\n")
    assert run_functions("cgroup_limit memory", ["cgroup_limit"], env) == ""
    assert run_functions("cgroup_limit cpu", ["cgroup_limit"], env) == ""


def test_cgroup_v1_limits_are_read(tmp_path):
    root = tmp_path / "cg"
    write(
        root / "memory" / "docker" / "abc" / "memory.limit_in_bytes", str(2 * 1024**3)
    )
    write(root / "cpu" / "docker" / "abc" / "cpu.cfs_quota_us", "150000")
    write(root / "cpu" / "docker" / "abc" / "cpu.cfs_period_us", "100000")
    env = cgroup_env(
        tmp_path, "4:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n1:name=systemd:/\n"
    )
    assert run_functions("cgroup_limit memory", ["cgroup_limit"], env) == str(
        2 * 1024 * 1024
    )
    assert run_functions("cgroup_limit cpu", ["cgroup_limit"], env) == "2"


def test_cgroup_v1_unlimited_memory_is_a_huge_number_that_never_wins(tmp_path):
    root = tmp_path / "cg"
    write(root / "memory" / "memory.limit_in_bytes", "9223372036854771712")
    write(tmp_path / "meminfo", "MemAvailable:   64000000 kB\n")
    env = cgroup_env(tmp_path, "4:memory:/\n")
    env["MEMINFO_FILE"] = str(tmp_path / "meminfo")
    names = ["cgroup_limit", "available_memory_kb"]
    assert run_functions("available_memory_kb", names, env) == "64000000"


def test_available_memory_uses_the_smallest_limit(tmp_path):
    root = tmp_path / "cg"
    write(root / "cgroup.controllers", "memory")
    write(root / "memory.max", str(16 * 1024**3))
    write(tmp_path / "meminfo", "MemAvailable:   64000000 kB\n")
    env = cgroup_env(tmp_path, "0::/\n")
    env["MEMINFO_FILE"] = str(tmp_path / "meminfo")
    names = ["cgroup_limit", "available_memory_kb"]
    assert run_functions("available_memory_kb", names, env) == str(16 * 1024 * 1024)
    env["SLURM_MEM_PER_NODE"] = "4096"
    assert run_functions("available_memory_kb", names, env) == str(4096 * 1024)
    env = {k: v for k, v in env.items() if k != "SLURM_MEM_PER_NODE"}
    env.update(SLURM_MEM_PER_CPU="1000", SLURM_CPUS_ON_NODE="2")
    assert run_functions("available_memory_kb", names, env) == str(2000 * 1024)


def test_available_memory_is_zero_when_it_cannot_be_read(tmp_path):
    env = cgroup_env(tmp_path, "")
    env["MEMINFO_FILE"] = str(tmp_path / "does_not_exist")
    names = ["cgroup_limit", "available_memory_kb"]
    assert run_functions("available_memory_kb", names, env) == "0"


def test_available_cores_applies_a_cpu_quota(tmp_path):
    root = tmp_path / "cg"
    write(root / "cgroup.controllers", "cpu")
    write(root / "cpu.max", "200000 100000")
    env = cgroup_env(tmp_path, "0::/\n")
    env["OMP_NUM_THREADS"] = "1"
    names = ["cgroup_limit", "available_cores"]
    assert run_functions("available_cores", names, env) == "2"


def test_allow_missing_predictions_option_is_passed_to_hybrid_decoding():
    assert '--allow-missing-predictions) ALLOW_MISSING_PREDICTIONS="1"' in PREDICT_SH
    assert "HYBRID_ARGS+=(--allow-missing-predictions)" in PREDICT_SH


def test_deleted_prediction_files_are_predicted_again_just_before_hybrid_decoding():
    """Hybrid decoding reads the logits, so deleting them must not leave the run broken."""
    assert PREDICT_SH.count('os.environ.get("NEEDS_LOGITS") != "1"') == 2
    assert "extract_needed_sequences() {" in PREDICT_SH
    hybrid = PREDICT_SH[PREDICT_SH.index('refresh_stage "$HYBRID_GFF"') :]
    hybrid = hybrid[: hybrid.index("HYBRID_ARGS=()")]
    assert '"$ALLOW_MISSING_PREDICTIONS" != "1"' in hybrid
    assert "NEEDS_LOGITS=1" in hybrid
    assert hybrid.index("extract_needed_sequences") < hybrid.index(
        "run_prediction_workers"
    )
    assert (
        "NEEDS_LOGITS=0" in PREDICT_SH[: PREDICT_SH.index("extract_needed_sequences()")]
    )


def make_chromosome_files(root: Path, chrom: str, filtered: bool = True) -> None:
    folder = root / chrom
    write(folder / f"sequences_{chrom}.zarr" / ".zgroup", "{}")
    write(folder / f"intervals_{chrom}.zarr" / ".zgroup", "{}")
    write(folder / f"predictions_{chrom}" / "_SUCCESS.json", "{}")
    write(folder / f"predictions_{chrom}.lock", "")
    write(folder / f"predictions_raw_{chrom}.gff", "raw")
    if filtered:
        write(folder / f"predictions_filtered_{chrom}.gff", "filtered")


def test_cleaning_a_decoded_chromosome_keeps_its_gff_and_predictions(tmp_path):
    make_chromosome_files(tmp_path, "chr1")
    run_functions(
        "clean_decoded_chromosome chr1",
        ["clean_decoded_chromosome"],
        {"OUTPUT_DIR": str(tmp_path)},
    )
    folder = tmp_path / "chr1"
    assert not (folder / "sequences_chr1.zarr").exists()
    assert not (folder / "intervals_chr1.zarr").exists()
    for kept in (
        "predictions_chr1",
        "predictions_chr1.lock",
        "predictions_raw_chr1.gff",
        "predictions_filtered_chr1.gff",
    ):
        assert (folder / kept).exists(), kept


def test_a_chromosome_without_its_filtered_gff_is_not_cleaned(tmp_path):
    make_chromosome_files(tmp_path, "chr1", filtered=False)
    run_functions(
        "clean_decoded_chromosome chr1",
        ["clean_decoded_chromosome"],
        {"OUTPUT_DIR": str(tmp_path)},
    )
    assert (tmp_path / "chr1" / "sequences_chr1.zarr").exists()
    assert (tmp_path / "chr1" / "intervals_chr1.zarr").exists()


def test_predictions_are_removed_only_after_the_final_gff_exists(tmp_path):
    make_chromosome_files(tmp_path, "chr1")
    make_chromosome_files(tmp_path, "chr2")
    env = {"OUTPUT_DIR": str(tmp_path), "FINAL_GFF": str(tmp_path / "final.gff")}
    run_functions("clean_predictions chr1 chr2", ["clean_predictions"], env)
    assert (tmp_path / "chr1" / "predictions_chr1").exists()

    write(tmp_path / "final.gff", "")  # an empty final GFF does not count
    run_functions("clean_predictions chr1 chr2", ["clean_predictions"], env)
    assert (tmp_path / "chr2" / "predictions_chr2").exists()

    write(tmp_path / "final.gff", "##gff-version 3\n")
    run_functions("clean_predictions chr1", ["clean_predictions"], env)
    assert not (tmp_path / "chr1" / "predictions_chr1").exists()
    assert not (tmp_path / "chr1" / "predictions_chr1.lock").exists()
    assert (tmp_path / "chr1" / "predictions_filtered_chr1.gff").exists()
    assert (tmp_path / "chr1" / "predictions_raw_chr1.gff").exists()
    assert (tmp_path / "chr2" / "predictions_chr2").exists()  # not listed
    assert (tmp_path / "final.gff").exists()


def test_cleaning_does_nothing_without_an_output_directory(tmp_path):
    write(tmp_path / "chr1" / "predictions_filtered_chr1.gff", "x")
    env = {"OUTPUT_DIR": "", "FINAL_GFF": str(tmp_path / "final.gff")}
    write(tmp_path / "final.gff", "x")
    run_functions("clean_predictions chr1", ["clean_predictions"], env)
    run_functions("clean_decoded_chromosome chr1", ["clean_decoded_chromosome"], env)
    assert (tmp_path / "chr1" / "predictions_filtered_chr1.gff").exists()


def test_clean_intermediates_is_off_by_default_and_exported():
    assert 'CLEAN_INTERMEDIATES="0"' in PREDICT_SH
    assert '--clean-intermediates) CLEAN_INTERMEDIATES="1"' in PREDICT_SH
    assert "export -f process_chromosome clean_decoded_chromosome" in PREDICT_SH
    assert "export CLEAN_INTERMEDIATES OUTPUT_DIR" in PREDICT_SH


def decode_settings_check() -> str:
    start = PREDICT_SH.index('DECODE_SETTINGS="mode=')
    end = PREDICT_SH.index("extract_needed_sequences\necho")
    return PREDICT_SH[start:end]


def check_decode_settings(tmp_path, chroms, **options):
    import sys

    ids = tmp_path / "ids.txt"
    ids.write_text("\n".join(chroms) + "\n")
    env = {
        "PATH": os.environ["PATH"],
        "PYTHON": sys.executable,
        "OUTPUT_DIR": str(tmp_path / "out"),
        "CHROM_IDS_FILE": str(ids),
        "MODE": "plant",
        "FRAME_AWARE": "0",
        "MIN_TRANSCRIPT_LENGTH": "3",
        "MIN_INTRON_LENGTH": "20",
        "MIN_CODING_RUN_LENGTH": "9",
        "EXON_LENGTH_STRICTNESS": "16",
        "ALLOW_U12_INTRONS": "0",
        **options,
    }
    result = subprocess.run(
        ["bash", "-c", decode_settings_check()], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def decoded_files(root, chrom):
    folder = root / chrom
    return sorted(
        p.name
        for p in folder.iterdir()
        if p.name.startswith(("predictions_raw", "predictions_filtered", "intervals"))
    )


def test_sequences_decoded_with_other_options_are_decoded_again(tmp_path):
    out = tmp_path / "out"
    make_chromosome_files(out, "chr1")
    make_chromosome_files(out, "chr2")
    make_chromosome_files(out, "chr3", filtered=False)
    everything = decoded_files(out, "chr1")

    # Decoded by an earlier version: the current options are recorded, nothing removed.
    message = check_decode_settings(tmp_path, ["chr1", "chr2", "chr3"])
    assert "2 sequence(s), e.g. chr1, were decoded by an earlier version" in message
    assert "delete their predictions_filtered_<ID>.gff" in message
    record = (out / "chr1" / "decoding_settings.txt").read_text()
    assert record == "mode=plant frame_aware=0 min_transcript_length=3\n"
    assert not (out / "chr3" / "decoding_settings.txt").exists()
    assert check_decode_settings(tmp_path, ["chr1", "chr2"]) == ""
    assert decoded_files(out, "chr1") == everything

    message = check_decode_settings(tmp_path, ["chr1"], FRAME_AWARE="1")
    assert "1 sequence(s), e.g. chr1 (frame_aware 0 -> 1)" in message
    assert decoded_files(out, "chr1") == []
    assert not (out / "chr1" / "decoding_settings.txt").exists()
    # Predictions and sequences are kept; other sequences are untouched.
    assert (out / "chr1" / "predictions_chr1" / "_SUCCESS.json").exists()
    assert (out / "chr1" / "sequences_chr1.zarr").exists()
    assert decoded_files(out, "chr2") == [n.replace("chr1", "chr2") for n in everything]

    message = check_decode_settings(tmp_path, ["chr2"], MIN_TRANSCRIPT_LENGTH="30")
    assert "chr2 (min_transcript_length 3 -> 30)" in message


def test_frame_aware_options_matter_only_to_frame_aware_decoding(tmp_path):
    out = tmp_path / "out"
    make_chromosome_files(out, "chr1")
    check_decode_settings(tmp_path, ["chr1"])
    assert check_decode_settings(tmp_path, ["chr1"], MIN_INTRON_LENGTH="60") == ""
    check_decode_settings(tmp_path, ["chr1"], FRAME_AWARE="1")
    make_chromosome_files(out, "chr1")
    (out / "chr1" / "decoding_settings.txt").write_text(
        "mode=plant frame_aware=1 min_transcript_length=3 min_intron_length=20 "
        "min_coding_run_length=9 exon_length_strictness=16 allow_u12_introns=0\n"
    )
    message = check_decode_settings(
        tmp_path, ["chr1"], FRAME_AWARE="1", MIN_INTRON_LENGTH="60"
    )
    assert "chr1 (min_intron_length 20 -> 60)" in message


def test_a_decoded_sequence_records_its_options():
    body = shell_function("process_chromosome")
    record = body.index('"$CHR_OUTPUT_DIR/decoding_settings.txt"')
    assert (
        body.index("filter_raw_gff.py")
        < record
        < body.index('echo "${LOG_PREFIX} Done!"')
    )
    assert "export DECODE_SETTINGS" in PREDICT_SH


def run_post_processing_logged(tmp_path, decoder, keep_partial, hybrid_exits=()):
    """Steps 7-8 with a fake Python whose hybrid_decode.py exits with the given
    statuses in turn; stage helpers and re-prediction are logged."""
    log = tmp_path / "calls.log"
    count = tmp_path / "hybrid_calls"
    count.write_text("0")
    statuses = " ".join(str(s) for s in hybrid_exits)
    fake = tmp_path / "python"
    fake.write_text(
        "#!/bin/bash\n"
        f'echo "$*" >> {log}\n'
        'if [[ "$1" == */hybrid_decode.py ]]; then\n'
        f"  n=$(cat {count}); echo $((n + 1)) > {count}\n"
        f"  statuses=({statuses})\n"
        '  exit "${statuses[$n]:-0}"\n'
        "fi\n"
    )
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
        "ALLOW_MISSING_PREDICTIONS": "0",
    }
    stubs = (
        f'refresh_stage() {{ echo "refresh $*" >> {log}; }}\n'
        f'record_stage() {{ echo "record $*" >> {log}; }}\n'
        f'extract_needed_sequences() {{ echo "extract" >> {log}; }}\n'
        f'run_prediction_workers() {{ echo "predict" >> {log}; }}\n'
    )
    result = subprocess.run(
        ["bash", "-c", stubs + post_processing()],
        env=env,
        text=True,
        capture_output=True,
    )
    lines = log.read_text().splitlines() if log.exists() else []
    return result, [line.split() for line in lines]


def names(calls):
    return [c[0].rsplit("/", 1)[-1] for c in calls]


def test_damaged_predictions_are_predicted_again_and_hybrid_decoding_rerun(tmp_path):
    result, calls = run_post_processing_logged(tmp_path, "hybrid", "0", [3])
    assert result.returncode == 0, result.stdout + result.stderr
    steps = [n for n in names(calls) if n not in ("refresh", "record")]
    assert steps == [
        "fix_orf.py",
        "extract",
        "predict",
        "hybrid_decode.py",
        "extract",
        "predict",
        "hybrid_decode.py",
        "refine.py",
    ]
    hybrid_records = [
        c for c in calls if c[0] == "record" and "Sp_GeneCAD_hybrid.gff" in c[1]
    ]
    assert len(hybrid_records) == 1


@pytest.mark.parametrize("statuses", [[3, 3], [1], [2]])
def test_hybrid_decoding_that_still_fails_stops_the_run(tmp_path, statuses):
    result, calls = run_post_processing_logged(tmp_path, "hybrid", "0", statuses)
    assert result.returncode == statuses[-1]
    assert "refine.py" not in names(calls)
    assert not any(c[0] == "record" and "Sp_GeneCAD_hybrid.gff" in c[1] for c in calls)
    assert names(calls).count("hybrid_decode.py") == len(statuses)


def stage_settings(calls, stage, output):
    (call,) = [c for c in calls if c[0] == stage and c[1].endswith(output)]
    return [call[i + 1] for i, a in enumerate(call) if a == "--setting"]


@pytest.mark.parametrize(
    "decoder,keep_partial,orf_keep",
    [("hybrid", "0", "1"), ("plain", "0", "0"), ("plain", "1", "1")],
)
def test_orf_and_hybrid_stages_are_redone_when_their_options_change(
    tmp_path, decoder, keep_partial, orf_keep
):
    result, calls = run_post_processing_logged(tmp_path, decoder, keep_partial)
    assert result.returncode == 0, result.stderr
    orf = ["max_shift=300", f"keep_partial={orf_keep}"]
    assert stage_settings(calls, "refresh", "Sp_GeneCAD_orf.gff") == orf
    assert stage_settings(calls, "record", "Sp_GeneCAD_orf.gff") == orf
    if decoder == "hybrid":
        hybrid = stage_settings(calls, "refresh", "Sp_GeneCAD_hybrid.gff")
        assert hybrid == stage_settings(calls, "record", "Sp_GeneCAD_hybrid.gff")
        assert "max_gap=20000" in hybrid and "mode=plant" in hybrid
        assert f"keep_partial={keep_partial}" in hybrid


def check_input_fasta(tmp_path, fasta):
    """Run predict.sh's check that the output directory belongs to this FASTA file."""
    import sys

    start = PREDICT_SH.index(
        "# An output directory holds the results of one FASTA file"
    )
    end = PREDICT_SH.index("# Step 1: Discover chromosomes from FASTA headers", start)
    output = tmp_path / "out"
    state = output / ".state"
    state.mkdir(parents=True, exist_ok=True)
    script = PREDICT_SH[start:end].rsplit("\n# ===", 1)[0]
    return subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "INPUT_FILE": str(fasta),
            "OUTPUT_DIR": str(output),
            "BATCH_SIZE_STATE_DIR": str(state),
            "PYTHON": sys.executable,
        },
        text=True,
        capture_output=True,
    )


def test_an_output_directory_belongs_to_one_fasta_file(tmp_path):
    import json

    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nACGT\n")
    assert check_input_fasta(tmp_path, fasta).returncode == 0
    record = tmp_path / "out" / ".state" / "input_fasta.json"
    first = json.loads(record.read_text())
    (tmp_path / "out" / "chr1").mkdir()

    # The same bases at another path, or touched: accepted.
    moved = tmp_path / "moved.fa"
    moved.write_text(">chr1\nACGT\n")
    result = check_input_fasta(tmp_path, moved)
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())["sha256"] == first["sha256"]
    assert json.loads(record.read_text())["path"] == str(moved.resolve())

    # The same file compressed: accepted.
    import gzip

    packed = tmp_path / "genome.fa.gz"
    packed.write_bytes(gzip.compress(b">chr1\nACGT\n"))
    result = check_input_fasta(tmp_path, packed)
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())["sha256"] == first["sha256"]
    moved.touch()
    assert check_input_fasta(tmp_path, moved).returncode == 0

    # Another genome with the same chromosome names: stopped, nothing changed.
    other = tmp_path / "other.fa"
    other.write_text(">chr1\nTTTT\n")
    result = check_input_fasta(tmp_path, other)
    assert result.returncode == 1
    assert "has the results of another FASTA file" in result.stderr
    assert str(moved.resolve()) in result.stderr
    assert json.loads(record.read_text())["path"] == str(moved.resolve())


def test_another_fasta_is_accepted_while_there_are_no_results(tmp_path):
    import json

    first, second = tmp_path / "a.fa", tmp_path / "b.fa"
    first.write_text(">chr1\nACGT\n")
    second.write_text(">chr1\nTTTT\n")
    assert check_input_fasta(tmp_path, first).returncode == 0
    result = check_input_fasta(tmp_path, second)
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "out" / ".state" / "input_fasta.json").read_text())
    assert record["path"] == str(second.resolve())
