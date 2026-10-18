#!/usr/bin/env python3
"""Every public writer of a durable format stores the semantics identifier (ALICE-Law G8).

A writer is a `pub fn` named `export*`, `save*`, `write*` or `to_bytes` in the modules that write
files or byte streams (WRITER_FILES), outside the `#[cfg(test)]` module; the `*_with_provenance`
and `*_with_stamp` variants are counted with the writer they extend. Each variant must extend a
writer of the same file (the name without the suffix) and must be called in tests/test_provenance.rs;
a variant with no base writer, or one the test never calls, fails (a suffix alone does not take a
new writer out of the count). Each writer must be either

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
    "src/neural.rs",
]
TEST_FILE = "tests/test_provenance.rs"
PENDING_FILE = "scripts/provenance-pending.txt"

WRITER = re.compile(
    r"^\s*pub\s+(?:(?:const|async|unsafe|extern\s*(?:\"[^\"]*\")?)\s+)*fn\s+((?:export|save|write)\w*|to_bytes)\b",
    re.M,
)
TEST_MODULE = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*(?:#\s*\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{")
COVERED = re.compile(r'\(\s*"(src/[\w/]+\.rs::\w+)"\s*,')


def blank_comments_and_strings(text: str) -> str:
    """The text with comments, string and char literals replaced by spaces (newlines kept)"""
    out = []
    i, n = 0, len(text)

    def blank(a: int, b: int) -> None:
        out.append("".join(c if c == "\n" else " " for c in text[a:b]))

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif (m := re.match(r'b?r(#*)"', text[i:])) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            close = '"' + m.group(1)
            j = text.find(close, i + m.end())
            j = n if j < 0 else j + len(close)
            blank(i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        elif c == "'" and (m := re.match(r"'(?:\\.[^']*|[^'\\\n])'", text[i:])):
            blank(i, i + m.end())
            i += m.end()
        else:
            out.append(c)
            i += 1
    return "".join(out)


def outside_test_modules(text: str) -> str:
    """The source with every `#[cfg(test)] mod name { ... }` block removed (comments and strings blanked)"""
    clean = blank_comments_and_strings(text)
    while (m := TEST_MODULE.search(clean)) is not None:
        depth, j = 1, m.end()
        while j < len(clean) and depth:
            depth += {"{": 1, "}": -1}.get(clean[j], 0)
            j += 1
        clean = clean[: m.start()] + clean[j:]
    return clean


def writer_files(root: str) -> list[str]:
    out = []
    for d in WRITER_DIRS:
        full = os.path.join(root, d)
        if os.path.isdir(full):
            # keys use "/" on every OS: they are compared with the names in the test and the list
            out += sorted(f"{d}/{f}" for f in os.listdir(full) if f.endswith(".rs"))
    out += [f for f in WRITER_FILES if os.path.isfile(os.path.join(root, f))]
    return out


VARIANT = re.compile(r"_with_(?:provenance|stamp)$")


def variants(root: str) -> list[tuple[str, str]]:
    """(file, name) of every `*_with_provenance` / `*_with_stamp` writer variant"""
    out = []
    for rel in writer_files(root):
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            text = fh.read()
        text = outside_test_modules(text)
        out += [(rel, name) for name in WRITER.findall(text) if VARIANT.search(name)]
    return out


def writers(root: str) -> list[str]:
    found = []
    for rel in writer_files(root):
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            text = fh.read()
        text = outside_test_modules(text)
        for name in WRITER.findall(text):
            if not name.endswith(("_with_provenance", "_with_stamp")):
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
    test_path = os.path.join(root, TEST_FILE)
    test_text = ""
    if os.path.isfile(test_path):
        with open(test_path, encoding="utf-8") as fh:
            test_text = blank_comments_and_strings(fh.read())
    for rel, name in variants(root):
        base = f"{rel}::{VARIANT.sub('', name)}"
        if base not in have:
            errors.append(f"{rel}::{name}: extends no writer ({base} is not one)")
        if not re.search(r"\b" + re.escape(name) + r"\s*\(", test_text):
            errors.append(f"{rel}::{name}: not called in {TEST_FILE}")
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
