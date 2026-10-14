import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "pyproject.toml", "rb") as f:
    VERSION = tomllib.load(f)["project"]["version"]


@pytest.mark.parametrize("script", ["predict.sh", "train.sh"])
@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_pipeline_scripts_print_the_version(script, flag):
    result = subprocess.run(
        ["bash", str(ROOT / script), flag], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(f"GeneCAD {VERSION}")


def test_train_help_shows_every_option_and_stops_before_requirements():
    result = subprocess.run(
        ["bash", str(ROOT / "train.sh"), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "-V, --version" in result.stdout
    assert result.stdout.rstrip().endswith("-h, --help              Show this message")
    assert "Requirements" not in result.stdout


def test_cli_prints_the_version():
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--version"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"GeneCAD {VERSION}"


def run_genecad_version(root, python):
    helper = ROOT / "scripts" / "genecad_version.sh"
    result = subprocess.run(
        ["bash", "-c", f'source "{helper}"; genecad_version "{root}"'],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "GENECAD_PYTHON": python},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_installed_wheel_reads_the_package_metadata(tmp_path):
    # A wheel ships predict.sh without pyproject.toml or a git checkout
    shutil.copy(ROOT / "predict.sh", tmp_path)
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\necho "1.2.3"\n')
    fake_python.chmod(0o755)
    assert run_genecad_version(tmp_path, str(fake_python)) == "1.2.3"


def test_unknown_version_when_nothing_can_be_found(tmp_path):
    assert run_genecad_version(tmp_path, "/nonexistent/python") == "unknown"
