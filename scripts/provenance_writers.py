#!/usr/bin/env python3
"""Every public writer of a durable format stores the semantics identifier (ALICE-Law G8).

A writer is a `pub fn` named `export*`, `save*`, `write*` or `to_bytes` in the modules that write
files or byte streams (WRITER_FILES), outside the `#[cfg(test)]` module; the `*_with_provenance`
variants are counted with the writer they extend. Each writer must be either

  - covered: named as `"<file>::<fn>"` in WRITERS of tests/test_provenance.rs, which writes a file,
    reads the identifier back and checks a mismatch, or
  - pending: listed in scripts/provenance-pending.txt (writers that do not store it yet).

The pending file also holds `expected-writers <N>`, the number of writers found: a new writer, or
one removed, changes the count and fails until the number is updated on purpose.

Fails when: no writer is found; the count differs from expected-writers; a writer is neither
covered nor pending; a pending line is covered already, or names no writer; a covered name is not
a writer.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WRITER_DIRS = ["src/io"]
WRITER_FILES = [
    "src/volume/export.rs",
    "src/svo/streaming.rs",
    "src/mesh/lod_persist.rs",
    "src/validity.rs",
]
TEST_FILE = "tests/test_provenance.rs"
PENDING_FILE = "scripts/provenance-pending.txt"

WRITER = re.compile(r"^\s*pub fn ((?:export|save|write)\w*|to_bytes)\b", re.M)
TEST_MODULE = re.compile(r"^#\[cfg\(test\)\]", re.M)
COVERED = re.compile(r'\(\s*"(src/[\w/]+\.rs::\w+)"\s*,')


def writer_files(root: str) -> list[str]:
    out = []
    for d in WRITER_DIRS:
        full = os.path.join(root, d)
        if os.path.isdir(full):
            out += sorted(os.path.join(d, f) for f in os.listdir(full) if f.endswith(".rs"))
    out += [f for f in WRITER_FILES if os.path.isfile(os.path.join(root, f))]
    return out


def writers(root: str) -> list[str]:
    found = []
    for rel in writer_files(root):
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            text = fh.read()
        m = TEST_MODULE.search(text)
        if m:
            text = text[: m.start()]
        for name in WRITER.findall(text):
            if not name.endswith("_with_provenance"):
                found.append(f"{rel}::{name}")
    return found


def covered(root: str) -> set[str]:
    path = os.path.join(root, TEST_FILE)
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return set(COVERED.findall(fh.read()))


def pending(root: str) -> tuple[int | None, list[str]]:
    expected, lines = None, []
    with open(os.path.join(root, PENDING_FILE), encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("expected-writers "):
                expected = int(line.split()[1])
            else:
                lines.append(line)
    return expected, lines


def check(root: str) -> tuple[list[str], int]:
    found = writers(root)
    errors = []
    if not found:
        return ["no writer found (compared nothing)"], 0
    expected, todo = pending(root)
    cov = covered(root)
    have = set(found)
    if expected is None:
        errors.append(f"{PENDING_FILE}: no `expected-writers <N>` line")
    elif expected != len(found):
        errors.append(
            f"{len(found)} writers found, {PENDING_FILE} expects {expected}: a writer was added or "
            "removed; give a new one a round-trip entry in tests/test_provenance.rs, then update the number"
        )
    dup = sorted({t for t in todo if todo.count(t) > 1})
    errors += [f"{PENDING_FILE}: listed twice: {t}" for t in dup]
    for w in sorted(have):
        if w not in cov and w not in todo:
            errors.append(f"{w}: neither covered by {TEST_FILE} nor pending")
    for t in sorted(set(todo)):
        if t in cov:
            errors.append(f"{PENDING_FILE}: {t} is covered by {TEST_FILE}; remove the line")
        if t not in have:
            errors.append(f"{PENDING_FILE}: {t} is not a writer")
    for c in sorted(cov):
        if c not in have:
            errors.append(f"{TEST_FILE}: {c} is not a writer")
    return errors, len(found)


def main(argv: list[str]) -> int:
    root = argv[1] if len(argv) > 1 else ROOT
    errors, n = check(root)
    if errors:
        for e in errors:
            print(f"provenance-writers: {e}", file=sys.stderr)
        return 1
    _, todo = pending(root)
    print(f"provenance-writers: ok ({n} writers: {n - len(todo)} covered, {len(todo)} pending)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
