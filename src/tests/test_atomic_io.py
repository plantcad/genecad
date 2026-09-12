"""Cover atomic_output_path's overwrite behavior for files and directories."""

import os

import pytest

from src.atomic_io import atomic_output_path


def test_file_write_creates_final_path(tmp_path):
    target = tmp_path / "out.txt"
    with atomic_output_path(str(target)) as tmp:
        with open(tmp, "w") as fh:
            fh.write("v1")
    assert target.read_text() == "v1"


def test_file_overwrite_replaces_content(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("v1")
    with atomic_output_path(str(target)) as tmp:
        with open(tmp, "w") as fh:
            fh.write("v2")
    assert target.read_text() == "v2"


def test_file_overwrite_survives_replace_failure(tmp_path, monkeypatch):
    """If the final rename fails, the previously complete file must survive.

    Deleting final_path before attempting the rename (the old implementation)
    loses the old file the moment os.replace raises, since the file has
    already been removed by then.
    """
    target = tmp_path / "out.txt"
    target.write_text("v1")

    real_replace = os.replace

    def failing_replace(src, dst):
        if dst == str(target):
            raise OSError("simulated rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError):
        with atomic_output_path(str(target)) as tmp:
            with open(tmp, "w") as fh:
                fh.write("v2")

    assert target.read_text() == "v1"


def test_directory_write_creates_final_path(tmp_path):
    target = tmp_path / "store"
    with atomic_output_path(str(target)) as tmp:
        os.makedirs(tmp)
        (open(os.path.join(tmp, "a.txt"), "w")).write("v1")
    assert (target / "a.txt").read_text() == "v1"


def test_directory_overwrite_replaces_contents(tmp_path):
    target = tmp_path / "store"
    os.makedirs(target)
    (target / "old.txt").write_text("stale")

    with atomic_output_path(str(target)) as tmp:
        os.makedirs(tmp)
        (open(os.path.join(tmp, "a.txt"), "w")).write("v2")

    assert (target / "a.txt").read_text() == "v2"
    assert not (target / "old.txt").exists()
    assert not (tmp_path / "store.old").exists()


def test_directory_overwrite_survives_replace_failure(tmp_path, monkeypatch):
    """A failed final rename must not have already destroyed the old store.

    The old delete-then-replace implementation ran shutil.rmtree on
    final_path before attempting the rename, so any failure -- or a crash
    mid-rmtree -- left final_path missing or half-deleted, and a caller's
    existence check could mistake that half-deleted directory for a
    complete one.
    """
    target = tmp_path / "store"
    os.makedirs(target)
    (target / "old.txt").write_text("stale")

    real_replace = os.replace

    def failing_replace(src, dst):
        if dst == str(target):
            raise OSError("simulated rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError):
        with atomic_output_path(str(target)) as tmp:
            os.makedirs(tmp)
            (open(os.path.join(tmp, "a.txt"), "w")).write("v2")

    # The old, fully-written directory must still be intact and recognizable
    # as complete -- not deleted, and not left half-written.
    assert (target / "old.txt").read_text() == "stale"


def test_directory_overwrite_recovers_stale_old_path(tmp_path):
    """A leftover .old from an interrupted prior cleanup must not linger.

    If a crash happens after a successful swap but before the trailing
    _remove_path(old_path) cleanup runs, the next call must clear that
    leftover itself rather than erroring or leaking it forever.
    """
    target = tmp_path / "store"
    os.makedirs(target)
    (target / "current.txt").write_text("v1")

    stale_old = tmp_path / "store.old"
    os.makedirs(stale_old)
    (stale_old / "leftover.txt").write_text("from a prior crash")

    with atomic_output_path(str(target)) as tmp:
        os.makedirs(tmp)
        (open(os.path.join(tmp, "a.txt"), "w")).write("v2")

    assert (target / "a.txt").read_text() == "v2"
    assert not stale_old.exists()
