#!/usr/bin/env python3
"""Refuse a commit stop marker left in the library source.

A value or code path that must not ship (a measurement build, a temporary switch) is sometimes marked with a
comment meant to be removed before commit. A comment stops nothing: this check fails when such a marker is still in
src/. The markers are matched case-insensitively:

  MEASUREMENT BUILD, replaced before commit, DO NOT COMMIT, TEMP:

Reading no file fails (an empty scan is not a pass). Usage: commit_marker_guard.py [repo root]
"""
import pathlib
import re
import sys

MARKERS = re.compile(r"measurement build|replaced before commit|do not commit|\btemp:", re.IGNORECASE)
SUFFIXES = {".rs", ".wgsl", ".glsl", ".hlsl", ".metal", ".py", ".toml"}


def scan(root: pathlib.Path) -> tuple[list[str], int]:
    found, files = [], 0
    for p in sorted((root / "src").rglob("*")):
        if not p.is_file() or p.suffix not in SUFFIXES:
            continue
        files += 1
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if MARKERS.search(line):
                found.append(f"{p.relative_to(root).as_posix()}:{n}: {line.strip()}")
    return found, files


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    found, files = scan(root)
    if files == 0:
        print("commit-marker-guard: 0 source files read under src/ (nothing compared)")
        return 1
    for f in found:
        print(f"commit-marker-guard: {f}")
    print(f"commit-marker-guard: {files} files read, {len(found)} marker(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
