"""Tests for predict.sh's automatic per-GPU batch size (-b auto)."""

import subprocess
from pathlib import Path

import pytest

PREDICT_SH = (Path(__file__).resolve().parents[2] / "predict.sh").read_text()


def auto_batch_size(tmp_path, free_mib: int, requested: str = "auto") -> str:
    """Run resolve_batch_size_for_gpu with a fake nvidia-smi reporting free_mib."""
    fake = tmp_path / "nvidia-smi"
    fake.write_text(f"#!/bin/bash\necho {free_mib}\n")
    fake.chmod(0o755)
    start = PREDICT_SH.index("resolve_batch_size_for_gpu() {")
    function = PREDICT_SH[start : PREDICT_SH.index("\n}\n", start) + 3]
    result = subprocess.run(
        ["bash", "-c", function + "resolve_batch_size_for_gpu 0"],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin", "BATCH_SIZE_ARG": requested},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize(
    "free_mib,expected",
    [
        (20_000, "17"),  # 19 GB free x 0.9
        (40_000, "35"),  # a 40 GB A100
        (81_000, "35"),  # an 80 GB H100 would be 71 without the cap
        (4_000, "8"),  # never below 8
    ],
)
def test_auto_batch_size_scales_with_free_memory_up_to_35(tmp_path, free_mib, expected):
    assert auto_batch_size(tmp_path, free_mib) == expected


def test_an_explicit_batch_size_is_not_capped(tmp_path):
    assert auto_batch_size(tmp_path, 81_000, requested="64") == "64"
