#!/usr/bin/env python3
"""Tell whether a pipeline stage has to be redone because its input or settings changed.

    stage_inputs.py check OUTPUT INPUT... [--setting KEY=VALUE]...
        exit 10 if OUTPUT must be rebuilt (the reason is printed), 0 if it is up to date
    stage_inputs.py record OUTPUT INPUT... [--setting KEY=VALUE]...
        remember which inputs and settings OUTPUT was built from

A record (OUTPUT.inputs.json) holds the SHA-256 of every input and the settings. Without a
record, which is the case for results made by earlier versions, OUTPUT is rebuilt only when
an input was modified more than TOLERANCE seconds after it, so that copying a directory does
not trigger a rebuild. Records written by earlier versions hold no settings; their outputs
are not rebuilt for a setting.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REBUILD = 10
TOLERANCE = 120
# A file name cannot contain "/", so this key never collides with an input.
SETTINGS = "/settings"


def sidecar(output: str) -> Path:
    return Path(output + ".inputs.json")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(inputs: list[str]) -> dict[str, str]:
    """SHA-256 by file name, so that moving or copying the directory changes nothing."""
    names = [os.path.basename(p) for p in inputs]
    if len(set(names)) != len(names):
        raise ValueError("input files must have different names")
    return {name: sha256(path) for name, path in zip(names, inputs)}


def check(
    output: str, inputs: list[str], settings: dict[str, str] | None = None
) -> str | None:
    """Return the reason OUTPUT is out of date, or None when it is up to date."""
    if not os.path.isfile(output):
        return "the output does not exist"
    missing = [p for p in inputs if not os.path.isfile(p)]
    record = sidecar(output)
    if record.is_file():
        old = json.loads(record.read_text())
        old_settings: dict[str, str] | None = old.pop(SETTINGS, None)
        if missing:
            return f"input missing: {missing[0]}"
        new = fingerprint(inputs)
        if new != old:
            changed = sorted(
                {n for n in set(new) | set(old) if new.get(n) != old.get(n)}
            )
            return f"{len(changed)} input file(s) changed, e.g. {changed[0]}"
        if settings is not None and old_settings is not None:
            keys: set[str] = set(settings) | set(old_settings)
            for key in sorted(keys):
                if settings.get(key) != old_settings.get(key):
                    return (
                        f"setting {key} changed from {old_settings.get(key)} "
                        f"to {settings.get(key)}"
                    )
        return None
    built = os.path.getmtime(output)
    for path in inputs:
        if os.path.isfile(path) and os.path.getmtime(path) > built + TOLERANCE:
            return f"{os.path.basename(path)} is newer"
    return None


def record(
    output: str, inputs: list[str], settings: dict[str, str] | None = None
) -> None:
    value: dict = fingerprint(inputs)
    if settings is not None:
        value[SETTINGS] = settings
    target = sidecar(output)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=0, sort_keys=True))
    temporary.replace(target)


def setting(text: str) -> tuple[str, str]:
    key, separator, value = text.partition("=")
    if not key or not separator:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["check", "record"])
    parser.add_argument("output")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--setting",
        type=setting,
        action="append",
        help="KEY=VALUE of an option the output depends on (repeatable)",
    )
    args = parser.parse_args()
    settings = dict(args.setting) if args.setting else None
    if args.action == "record":
        record(args.output, args.inputs, settings)
        return 0
    reason = check(args.output, args.inputs, settings)
    if reason:
        print(reason)
        return REBUILD
    return 0


if __name__ == "__main__":
    sys.exit(main())
