"""Crash-safe output writing.

Every stage of the prediction pipeline is resumable: predict.sh decides
whether to skip a step by checking if its output file/directory already
exists. If a run is killed mid-write (OOM killer, SLURM time limit, power
loss), a half-written file can be left at that exact path, and resume logic
has no way to tell it apart from a completed one — it gets skipped, and
downstream steps silently consume truncated data.

``atomic_output_path`` closes that gap: callers write to a ``.tmp`` sibling
path and it is renamed onto the real path only after the ``with`` block
exits successfully. ``os.replace`` is atomic on the same filesystem, so a
plain file's final path only ever exists in a fully-written state. A
non-empty directory (e.g. a Zarr store) can't be replaced by a single
``os.replace`` if the final path already exists, so that case is instead
swapped in via two atomic renames with the old directory held aside, rather
than deleted first -- the final path is never missing for longer than the
gap between those two renames. A crash leaves only the orphaned ``.tmp``
(and, for directories, possibly ``.old``) path, which the next attempt
clears before retrying.
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
    old_path = final_path.rstrip("/") + ".old"
    _remove_path(tmp_path)
    try:
        yield tmp_path
    except BaseException:
        _remove_path(tmp_path)
        raise
    else:
        if os.path.isdir(tmp_path) and not os.path.islink(tmp_path):
            # os.replace can't rename a directory onto a non-empty one, so a
            # plain "delete final_path, then rename" would leave final_path
            # missing (or, if killed mid-rmtree, half-deleted and mistaken
            # for complete by a caller's existence check) for as long as the
            # delete takes. Swap the old directory aside first instead: the
            # two renames below are each atomic, so final_path is only ever
            # missing for the gap between them, not for the duration of a
            # multi-file delete. If the second rename itself fails, put the
            # old directory straight back rather than leaving final_path
            # missing.
            _remove_path(old_path)
            swapped_old_aside = os.path.exists(final_path) or os.path.islink(final_path)
            if swapped_old_aside:
                os.rename(final_path, old_path)
            try:
                os.replace(tmp_path, final_path)
            except BaseException:
                if swapped_old_aside:
                    os.rename(old_path, final_path)
                _remove_path(tmp_path)
                raise
            _remove_path(old_path)
        else:
            os.replace(tmp_path, final_path)
