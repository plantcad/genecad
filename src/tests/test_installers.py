"""Check installer routing without downloading or changing an environment."""

import os
import subprocess
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def installer_commands(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    uname = commands / "uname"
    uname.write_text(
        '#!/bin/bash\ncase "$1" in -s) echo Linux ;; -m) echo "$TEST_ARCH" ;; esac\n'
    )
    uname.chmod(0o755)
    uv = commands / "uv"
    uv.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_UV_LOG"\n')
    uv.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "TEST_UV_LOG": str(tmp_path / "uv.log"),
    }
    return tmp_path, environment


def test_arm_release_installer_routes_help_without_installing(installer_commands):
    root, environment = installer_commands
    environment["TEST_ARCH"] = "aarch64"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "install_release.sh"), "--help"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "install_arm64.sh" in result.stdout
    assert "--wheel-dir" in result.stdout
    assert not (root / "uv.log").exists()


def test_x86_release_installer_keeps_release_and_mamba_installation(installer_commands):
    root, environment = installer_commands
    environment.update(TEST_ARCH="x86_64", GENECAD_VERSION="0.6.1")
    subprocess.run(
        ["bash", str(SCRIPTS / "install_release.sh")],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    commands = (root / "uv.log").read_text().splitlines()
    assert len(commands) == 2
    assert (
        "genecad[torch] @ https://github.com/plantcad/genecad/releases/download/v0.6.1/"
        in commands[0]
    )
    assert "transformers<5" in commands[0]
    assert "--no-build-isolation" in commands[1]
    assert "mamba@v2.2.4" in commands[1]
    assert "causal-conv1d@v1.5.0.post8" in commands[1]


def test_arm_installer_rejects_x86_before_installing(installer_commands):
    root, environment = installer_commands
    environment["TEST_ARCH"] = "x86_64"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "install_arm64.sh")],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "requires Linux aarch64" in result.stderr
    assert not (root / "uv.log").exists()


def test_arm_installer_reports_missing_option_value(installer_commands):
    root, environment = installer_commands
    environment["TEST_ARCH"] = "aarch64"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "install_arm64.sh"), "--venv"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Missing value for --venv" in result.stderr
    assert not (root / "uv.log").exists()


def test_cpu_install_check_defers_native_imports_that_initialize_cuda(monkeypatch):
    from scripts import check_install

    monkeypatch.setattr(check_install.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(check_install, "version", lambda name: "installed")

    def fail_native_import(name):
        pytest.fail(f"CPU check attempted CUDA-dependent import: {name}")

    monkeypatch.setattr(check_install, "import_module", fail_native_import)
    report = check_install.check_install(imports_only=True)
    assert report["status"] == "imports_only"
    assert report["native_cuda_imports"] == "deferred_no_gpu"
    with pytest.raises(RuntimeError, match="CUDA GPU unavailable"):
        check_install.check_install()
