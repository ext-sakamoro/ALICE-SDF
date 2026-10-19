#!/usr/bin/env python3
"""Every public writer of a durable format stores the semantics identifier (ALICE-Law G8).

A candidate is a `pub fn` anywhere under src/, outside `#[cfg(test)]` modules, that either has a
writer's name (`export*`, `save*`, `write*`, `bake_to*`, `encode*`, `serialize*`, `persist*`,
`dump*`, `to_bytes`, `to_vec`) or reaches a write sink (its signature takes a `Write`, or its body
calls `File::create`, `fs::write` or `OpenOptions`). The `*_with_provenance` and `*_with_stamp`
variants are counted with the writer they extend. Each variant must extend a writer of the same
file (the name without the suffix) and must be called in tests/test_provenance.rs; a variant with
no base writer, or one the test never calls, fails (a suffix alone does not take a new writer out
of the count). Each candidate must be exactly one of

  - covered: named as `"<file>::<fn>"` in a table of tests/test_provenance.rs, which writes it,
    reads the identifier back and checks a mismatch,
  - pending: listed in scripts/provenance-pending.txt (writers that do not store it yet), or
  - not a writer: listed in scripts/provenance-not-writers.txt with its reason. A reason
    `delegates to <fn>` must name a covered or pending writer and the candidate's body must call
    it; any other reason has at least five words.

The pending file also holds `expected-writers <N>`, the number of writers found (candidates that
are not "not a writer"): a new writer, or one removed, changes the count and fails until the
number is updated on purpose.

Fails when: no writer is found; the count differs from expected-writers; a candidate is in none of
the three lists or in two; a pending, covered or not-a-writer line names no candidate; a
`delegates to` reason names no writer or the body does not call it; any other reason is short.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIR = "src"
TEST_FILE = "tests/test_provenance.rs"
PENDING_FILE = "scripts/provenance-pending.txt"
NOT_WRITERS_FILE = "scripts/provenance-not-writers.txt"

PUB_FN = re.compile(r"^\s*pub\s+(?:(?:const|async|unsafe|extern\s*(?:\"[^\"]*\")?)\s+)*fn\s+(\w+)", re.M)
WRITER_NAME = re.compile(r"^(?:(?:export|save|write|bake_to|encode|serialize|persist|dump)\w*|to_bytes|to_vec)$")
# a write sink in the signature (a `Write` parameter) or the body (a file created or written)
SINK_SIG = re.compile(r"\b(?:impl|dyn)\s+(?:std::)?(?:io::)?Write\b|\w+\s*:\s*(?:std::)?(?:io::)?Write\b")
SINK_BODY = re.compile(r"\bFile::create\b|\bfs::write\b|\bOpenOptions\b")
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
    for dirpath, _, files in os.walk(os.path.join(root, SRC_DIR)):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        # keys use "/" on every OS: they are compared with the names in the test and the lists
        out += [f"{rel_dir}/{f}" for f in files if f.endswith(".rs")]
    return sorted(out)


VARIANT = re.compile(r"_with_(?:provenance|stamp)$")


def signature_end(text: str, i: int) -> int | None:
    """The index of the `{` that opens the body of the fn whose signature starts at `i`, or None
    when a `;` ends it first; a `;` inside brackets (an array type `[u8; 32]`) does not"""
    depth = 0
    while i < len(text):
        c = text[i]
        if c in "([<":
            depth += 1
        elif c in ")]>" and depth and not (c == ">" and text[i - 1] == "-"):
            depth -= 1
        elif c == "{" and depth == 0:
            return i
        elif c == ";" and depth == 0:
            return None
        i += 1
    return None


def candidates(root: str) -> dict[str, str]:
    """{"<file>::<fn>": body} of every public fn that has a writer's name or reaches a write sink"""
    out: dict[str, str] = {}
    for rel in writer_files(root):
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            text = outside_test_modules(fh.read())
        for m in PUB_FN.finditer(text):
            name = m.group(1)
            brace = signature_end(text, m.end())
            if brace is None:
                continue  # a declaration without a body (trait or extern item)
            depth, end = 1, brace + 1
            while end < len(text) and depth:
                depth += {"{": 1, "}": -1}.get(text[end], 0)
                end += 1
            sig, body = text[m.end():brace], text[brace:end]
            if WRITER_NAME.match(name) or SINK_SIG.search(sig) or SINK_BODY.search(body):
                out[f"{rel}::{name}"] = body
    return out


def variants(root: str) -> list[tuple[str, str]]:
    """(file, name) of every `*_with_provenance` / `*_with_stamp` writer variant"""
    return [tuple(k.split("::", 1)) for k in candidates(root) if VARIANT.search(k)]


def not_writers(root: str) -> list[tuple[str, str]]:
    """(key, reason) of scripts/provenance-not-writers.txt"""
    path = os.path.join(root, NOT_WRITERS_FILE)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, reason = line.partition(" ")
            out.append((key, reason.strip()))
    return out


def writers(root: str) -> list[str]:
    skip = {k for k, _ in not_writers(root)}
    return [k for k in candidates(root) if not VARIANT.search(k) and k not in skip]


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


def calls(body: str, target: str) -> bool:
    """Whether `body` calls `target`, directly or through a `use ... target as alias` in the body"""
    names = [target] + re.findall(r"\b" + re.escape(target) + r"\s+as\s+(\w+)", body)
    return any(re.search(r"\b" + re.escape(n) + r"\s*\(", body) for n in names)


def check(root: str) -> tuple[list[str], int]:
    cand = candidates(root)
    found = writers(root)
    errors = []
    if not found:
        return ["no writer found (compared nothing)"], 0
    expected, todo = pending(root)
    cov = covered(root)
    nots = not_writers(root)
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
    keys = [k for k, _ in nots]
    errors += [f"{NOT_WRITERS_FILE}: listed twice: {k}" for k in sorted({k for k in keys if keys.count(k) > 1})]
    for w in sorted(have):
        if w not in cov and w not in todo:
            errors.append(f"{w}: neither covered by {TEST_FILE} nor pending nor in {NOT_WRITERS_FILE}")
    for t in sorted(set(todo)):
        if t in cov:
            errors.append(f"{PENDING_FILE}: {t} is covered by {TEST_FILE}; remove the line")
        if t not in have:
            errors.append(f"{PENDING_FILE}: {t} is not a writer")
    for c in sorted(cov):
        if c not in have:
            errors.append(f"{TEST_FILE}: {c} is not a writer")
    writer_names = {w.split("::", 1)[1] for w in have}
    for key, reason in nots:
        if key not in cand or VARIANT.search(key):
            errors.append(f"{NOT_WRITERS_FILE}: {key} is not a candidate (remove the line)")
            continue
        if key in cov or key in todo:
            errors.append(f"{NOT_WRITERS_FILE}: {key} is also covered or pending")
        if reason.startswith("delegates to "):
            target = reason[len("delegates to "):].split()[0]
            if target not in writer_names:
                errors.append(f"{NOT_WRITERS_FILE}: {key} delegates to `{target}`, which is no covered or pending writer")
            elif not calls(cand[key], target):
                errors.append(f"{NOT_WRITERS_FILE}: {key} delegates to `{target}`, but its body does not call it")
        elif len(reason.split()) < 5:
            errors.append(f"{NOT_WRITERS_FILE}: {key}: the reason needs at least five words")
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
    print(f"provenance-writers: ok ({n} writers: {n - len(todo)} covered, {len(todo)} pending; "
          f"{len(not_writers(root))} candidates not writers)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
