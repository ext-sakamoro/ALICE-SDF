#!/usr/bin/env python3
"""ALICE-Law G3 for the shader text this crate emits: no fused multiply-add intrinsic, and no
negation of a whole multiply-add.

The Rust side is held by clippy (f32 / f64 mul_add in clippy.toml; wide's fused methods and
cranelift's InstBuilder::fma in .github/clippy-lib/clippy.toml, each with a canary form). The
WGSL / GLSL / HLSL / MSL / BlinkScript transpilers and shader templates write shader source as
Rust string literals, which clippy does not read: an `fma(` written into one of them would put
a one-rounding multiply-add (and so a negated one) into the GPU path while the CPU paths round
twice. This script reads every string literal of the library source (src/, outside
`#[cfg(test)]` modules) and fails on a contraction-capable intrinsic in one:
`fma(`, `fmaf(`, `mad(` (HLSL / BlinkScript), also after a namespace (`metal::precise::fma(`),
with any whitespace before the parenthesis. It also reads, whole, every tracked shader file the
repository ships (SHADER_EXTENSIONS: the Unity / VRChat .shader / .cginc / .compute, the Unreal
.ush / .usf, and any .wgsl / .glsl / .hlsl / .metal), for the same intrinsics and negations.

Shader text that is composed from pieces at run time (`"fm"` + `"a("`) is not seen here;
tests/test_shader_no_fused_multiply_add.rs transpiles the node corpus through every backend
and mode and scans the emitted text for the same intrinsics.

A GPU driver may contract `a * b + c` into an fma by itself, so a literal must also not negate a
whole multiply-add: after a contraction that is the outer fneg of an fma G3 forbids. Flagged: a
unary minus over a parenthesised expression with a top-level product and a top-level binary
`+` / `-` (`-(a * b + c)`, `0.0 - (..)`, `-1.0 * (..)`), and a unary minus on a name assigned such
an expression earlier in the same literal (`p = ky - kx * kx; .. sqrt(-p)`). The operand form
(`kx * kx - ky`) is what the transpilers write; a contraction turns it into the allowed
`fma(kx, kx, -ky)`. This sees every helper and template, including ones no corpus node reaches.

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
UNARY_BEFORE = set("(,=+-*/?:<>!&|[{;")
ASSIGN = re.compile(r"\b([A-Za-z_]\w*)\s*(?::\s*[\w<>]+)?\s*=(?!=)\s*([^;{}]+);")


def group_end(text: str, open_at: int) -> int:
    """index of the ')' closing the '(' at open_at, or -1"""
    depth = 0
    for j in range(open_at, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


def is_multiply_add(expr: str) -> bool:
    """a top-level '*' and a top-level binary '+' / '-' (not an exponent, not a sign after an operator)"""
    depth, star, add, prev, prev2 = 0, False, False, "", ""
    for c in expr:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif depth == 0 and c == "*":
            star = True
        elif depth == 0 and c in "+-":
            exponent = prev in "eE" and (prev2.isdigit() or prev2 == ".")
            if prev and prev not in "(,=+-*/?:<>!&|[" and not exponent:
                add = True
        if not c.isspace():
            prev2, prev = prev, c
    return star and add


def unary_at(text: str, at: int) -> bool:
    before = text[:at].rstrip()
    return not before or before.endswith("return") or before[-1] in UNARY_BEFORE


def negated_multiply_adds(text: str) -> list[str]:
    """the negations of a whole multiply-add in one literal"""
    found = []
    for m in re.finditer(r"-\s*\(", text):
        if not unary_at(text, m.start()):
            continue
        open_at = m.end() - 1
        end = group_end(text, open_at)
        if end > 0 and is_multiply_add(text[open_at + 1:end]):
            found.append(text[m.start():end + 1])
    for m in re.finditer(r"(?<![A-Za-z0-9_.])(?:0\.0f?\s*-|-1\.0f?\s*\*)\s*\(", text):
        open_at = m.end() - 1
        end = group_end(text, open_at)
        if end > 0 and is_multiply_add(text[open_at + 1:end]):
            found.append(text[m.start():end + 1])
    # every assignment of a name, in order: the last one before a use decides
    # (`h = q * q + p; h = sqrt(h); .. -h` negates a square root, not a multiply-add)
    assigned: dict[str, list[tuple[int, bool]]] = {}
    for m in ASSIGN.finditer(text):
        assigned.setdefault(m.group(1), []).append((m.start(), is_multiply_add(m.group(2))))
    for m in re.finditer(r"-\s*([A-Za-z_]\w*)", text):
        name = m.group(1)
        before = [ma for at, ma in assigned.get(name, []) if at < m.start()]
        if not before or not before[-1] or not unary_at(text, m.start()):
            continue
        if text[m.end():].lstrip()[:1] in ("(", "*", "/", "."):
            continue  # a call, an operand of a product, a field
        found.append(f"-{name} (assigned a multiply-add earlier in the literal)")
    return found
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


SHADER_EXTENSIONS = (".wgsl", ".glsl", ".hlsl", ".hlsli", ".cginc", ".compute", ".shader", ".ush", ".usf",
                     ".metal", ".gdshader", ".gdshaderinc", ".frag", ".vert", ".comp", ".fx")


def scan_shader_files(root: Path, files: list[str]) -> tuple[list[str], int]:
    """the intrinsics and negations in whole shader files (not Rust: no literal to extract)"""
    hits = []
    for rel in files:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        for m in INTRINSIC.finditer(text):
            hits.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: `{m.group(0)}`")
        for neg in negated_multiply_adds(text):
            hits.append(f"{rel}: negated multiply-add `{neg[:80]}`")
    return hits, len(files)


def scan(root: Path, files: list[str]) -> tuple[list[str], int, int]:
    hits, n_files, n_lits = [], 0, 0
    for rel in files:
        n_files += 1
        src = strip_test_modules((root / rel).read_text(encoding="utf-8"))
        for line, text in literals(src):
            n_lits += 1
            for m in INTRINSIC.finditer(text):
                hits.append(f"{rel}:{line}: `{m.group(0)}` in a string literal")
            for neg in negated_multiply_adds(text):
                hits.append(f"{rel}:{line}: negated multiply-add `{neg[:80]}` in a string literal")
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
    # NUL-separated: shipped sample paths contain spaces ("Samples~/SDF Gallery/...")
    tracked = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True,
                             encoding="utf-8", check=True).stdout.split("\0")
    shaders = sorted(f for f in tracked if f.lower().endswith(SHADER_EXTENSIONS))
    shader_hits, n_shaders = scan_shader_files(root, shaders)
    hits += shader_hits
    for h in hits:
        print(f"shader_fma_guard: {h}")
    print(f"shader_fma_guard: {n_files} files, {n_lits} string literals, {n_shaders} shader files, "
          f"{len(hits)} fused or negated multiply-add(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
