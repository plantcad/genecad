import importlib.util
import os
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stage_inputs.py"
spec = importlib.util.spec_from_file_location("stage_inputs", SCRIPT)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def make(tmp_path, text="a"):
    source, output = tmp_path / "in.gff", tmp_path / "out.gff"
    source.write_text(text)
    output.write_text("built")
    return source, output


def age(path, seconds_from_now):
    stamp = os.path.getmtime(path) + seconds_from_now
    os.utime(path, (stamp, stamp))


def test_recorded_output_is_up_to_date_until_an_input_changes(tmp_path):
    source, output = make(tmp_path)
    stage.record(str(output), [str(source)])
    assert stage.check(str(output), [str(source)]) is None
    source.write_text("b")
    assert "changed" in stage.check(str(output), [str(source)])


def test_record_ignores_modification_times(tmp_path):
    """Copying a directory scrambles the times; the content decides."""
    source, output = make(tmp_path)
    stage.record(str(output), [str(source)])
    age(source, 10_000)
    assert stage.check(str(output), [str(source)]) is None


def test_a_new_or_removed_input_is_a_change(tmp_path):
    first, output = make(tmp_path)
    second = tmp_path / "second.gff"
    second.write_text("c")
    stage.record(str(output), [str(first)])
    assert stage.check(str(output), [str(first), str(second)]) is not None
    stage.record(str(output), [str(first), str(second)])
    assert "missing" in stage.check(
        str(output), [str(first), str(tmp_path / "gone.gff")]
    )


def test_without_a_record_only_a_clearly_newer_input_triggers_a_rebuild(tmp_path):
    source, output = make(tmp_path)
    age(source, 60)  # within the tolerance, as after copying a directory
    assert stage.check(str(output), [str(source)]) is None
    age(source, 3600)
    assert "newer" in stage.check(str(output), [str(source)])


def test_missing_output_needs_a_build(tmp_path):
    source, output = make(tmp_path)
    output.unlink()
    assert stage.check(str(output), [str(source)]) is not None


def test_record_is_written_next_to_the_output(tmp_path):
    source, output = make(tmp_path)
    stage.record(str(output), [str(source)])
    assert (tmp_path / "out.gff.inputs.json").is_file()


def test_a_copied_directory_is_still_up_to_date(tmp_path):
    """The record must not depend on where the files are."""
    import shutil

    source, output = make(tmp_path)
    stage.record(str(output), [str(source)])
    moved = tmp_path / "elsewhere"
    shutil.copytree(tmp_path, moved, ignore=shutil.ignore_patterns("elsewhere"))
    assert stage.check(str(moved / "out.gff"), [str(moved / "in.gff")]) is None
    (moved / "in.gff").write_text("different")
    assert stage.check(str(moved / "out.gff"), [str(moved / "in.gff")]) is not None
