#!/usr/bin/env python3
"""ALICE-Law G3 for the shader text this crate emits: no fused multiply-add intrinsic.

The Rust side is held by clippy (f32 / f64 mul_add in clippy.toml; wide's fused methods and
cranelift's InstBuilder::fma in .github/clippy-lib/clippy.toml, each with a canary form). The
WGSL / GLSL / HLSL / MSL / BlinkScript transpilers and shader templates write shader source as
Rust string literals, which clippy does not read: an `fma(` written into one of them would put
a one-rounding multiply-add (and so a negated one) into the GPU path while the CPU paths round
twice. This script reads every string literal of the library source (src/, outside
`#[cfg(test)]` modules) and fails on a contraction-capable intrinsic in one:
`fma(`, `fmaf(`, `mad(` (HLSL / BlinkScript), also after a namespace (`metal::precise::fma(`),
with any whitespace before the parenthesis.

Shader text that is composed from pieces at run time (`"fm"` + `"a("`) is not seen here;
tests/test_shader_no_fused_multiply_add.rs transpiles the node corpus through every backend
and mode and scans the emitted text for the same intrinsics.

Not covered (a stated residual, not a pass): a GPU driver may contract `a * b + c` into an fma
by itself; no source text shows that.

Fails on a hit, on 0 files read, and on 0 string literals read (compared nothing).

usage: scripts/shader_fma_guard.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

INTRINSIC = re.compile(r"(?<![A-Za-z0-9_])(fma|fmaf|mad)\s*\(")
TEST_MOD = re.compile(r"#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{")


def strip_test_modules(src: str) -> str:
    """Blank out `#[cfg(test)] mod x { ... }` bodies (brace-matched on code, not strings)."""
    out = list(src)
    for m in TEST_MOD.finditer(src):
        depth, i = 1, m.end()
        while i < len(src) and depth:
            j, _ = skip_non_code(src, i)
            if j != i:
                i = j
                continue
            c = src[i]
            depth += c == "{"
            depth -= c == "}"
            i += 1
        for k in range(m.start(), i):
            if out[k] != "\n":
                out[k] = " "
    return "".join(out)


RAW_START = re.compile(r"b?r(#*)\"")


def skip_non_code(src: str, i: int) -> tuple[int, str | None]:
    """At a comment, string or char literal starting at i: (index after it, literal text or None)."""
    if src.startswith("//", i):
        j = src.find("\n", i)
        return (len(src) if j < 0 else j), None
    if src.startswith("/*", i):
        depth, j = 1, i + 2
        while j < len(src) and depth:
            if src.startswith("/*", j):
                depth, j = depth + 1, j + 2
            elif src.startswith("*/", j):
                depth, j = depth - 1, j + 2
            else:
                j += 1
        return j, None
    prev = src[i - 1] if i else " "
    if not (prev.isalnum() or prev == "_"):
        m = RAW_START.match(src, i)
        if m:
            end = '"' + m.group(1)
            j = src.find(end, m.end())
            j = len(src) if j < 0 else j
            return j + len(end), src[m.end():j]
    if src[i] == '"' or (src.startswith('b"', i) and not (prev.isalnum() or prev == "_")):
        j = i + (2 if src[i] == "b" else 1)
        start = j
        while j < len(src) and src[j] != '"':
            j += 2 if src[j] == "\\" else 1
        return j + 1, src[start:j]
    if src[i] == "'":
        # a char literal ('a', '\n', '\u{..}') and not a lifetime ('a followed by no quote)
        m = re.match(r"'(?:\\(?:u\{[0-9A-Fa-f]+\}|x[0-9A-Fa-f]{2}|.)|[^\\'])'", src[i:])
        if m:
            return i + m.end(), None
    return i, None


def literals(src: str) -> list[tuple[int, str]]:
    """(line, text) of every string literal of the code, comments skipped."""
    found, i = [], 0
    while i < len(src):
        j, lit = skip_non_code(src, i)
        if j == i:
            i += 1
            continue
        if lit is not None:
            found.append((src.count("\n", 0, i) + 1, lit))
        i = j
    return found


def scan(root: Path, files: list[str]) -> tuple[list[str], int, int]:
    hits, n_files, n_lits = [], 0, 0
    for rel in files:
        n_files += 1
        src = strip_test_modules((root / rel).read_text(encoding="utf-8"))
        for line, text in literals(src):
            n_lits += 1
            for m in INTRINSIC.finditer(text):
                hits.append(f"{rel}:{line}: `{m.group(0)}` in a string literal")
    return hits, n_files, n_lits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    root = Path(args.root)
    files = subprocess.run(["git", "-C", str(root), "ls-files", "src/*.rs", "src/**/*.rs"],
                           capture_output=True, text=True, encoding="utf-8", check=True).stdout.split()
    hits, n_files, n_lits = scan(root, sorted(set(files)))
    if n_files == 0 or n_lits == 0:
        print(f"shader_fma_guard: read {n_files} files, {n_lits} string literals (compared nothing)")
        return 2
    for h in hits:
        print(f"shader_fma_guard: {h}")
    print(f"shader_fma_guard: {n_files} files, {n_lits} string literals, {len(hits)} fused multiply-add intrinsic(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
