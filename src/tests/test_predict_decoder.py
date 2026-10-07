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


def test_chromosomes_decoded_at_once_never_exceed_the_limit(tmp_path):
    """Four GPUs but a limit of two: at most two process_chromosome calls overlap."""
    start = PREDICT_SH.index("    declare -a PIDS=()")
    end = PREDICT_SH.index("\nif [[ $FAILED -gt 0 ]]", start)
    loop = PREDICT_SH[start:end].rsplit("\nfi", 1)[0]
    log = tmp_path / "log"
    script = (
        f'process_chromosome() {{ echo "+ $1" >> {log}; sleep 0.3; echo "- $1" >> {log}; }}\n'
        "GPU_ARRAY=(0 1 2 3); NUM_GPUS=4; PARALLEL_CHROMOSOMES=2; FAILED=0\n"
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


def test_finished_sequences_without_prediction_files_are_predicted_again():
    """Hybrid decoding reads the logits, so deleting them must not leave the run broken."""
    assert PREDICT_SH.count('os.environ.get("NEEDS_LOGITS") != "1"') == 2
    start = PREDICT_SH.index("NEEDS_LOGITS=0")
    block = PREDICT_SH[start : PREDICT_SH.index("export NEEDS_LOGITS")]
    assert '"$DECODER" == "hybrid"' in block
    assert '"$ALLOW_MISSING_PREDICTIONS" != "1"' in block
    assert "_GeneCAD_final.gff" in block and "_GeneCAD_hybrid.gff" in block


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
