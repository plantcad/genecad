"""Crash-safe output writing.

Every stage of the prediction pipeline is resumable: predict.sh decides
whether to skip a step by checking if its output file/directory already
exists. If a run is killed mid-write (OOM killer, SLURM time limit, power
loss), a half-written file can be left at that exact path, and resume logic
has no way to tell it apart from a completed one — it gets skipped, and
downstream steps silently consume truncated data.

``atomic_output_path`` closes that gap: callers write to a ``.tmp`` sibling
path and it is renamed onto the real path only after the ``with`` block
exits successfully. ``os.replace`` is atomic on the same filesystem for both
files and directories (e.g. Zarr stores), so the final path only ever exists
in a fully-written state. A crash leaves only the orphaned ``.tmp`` path,
which the next attempt clears before retrying.
"""

import os
import shutil
from contextlib import contextmanager
from typing import Iterator


def _remove_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path) or os.path.islink(path):
        os.remove(path)


@contextmanager
def atomic_output_path(final_path: "str | os.PathLike[str]") -> Iterator[str]:
    """Yield a temp path to write to; rename it onto `final_path` on success.

    Leaves `final_path` untouched until the write fully succeeds. Any
    leftover `.tmp` path from a previous crashed attempt is cleared first.
    """
    final_path = os.fspath(final_path)
    tmp_path = final_path.rstrip("/") + ".tmp"
    _remove_path(tmp_path)
    try:
        yield tmp_path
    except BaseException:
        _remove_path(tmp_path)
        raise
    else:
        _remove_path(final_path)
        os.replace(tmp_path, final_path)
