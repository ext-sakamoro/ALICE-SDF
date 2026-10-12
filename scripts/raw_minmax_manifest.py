#!/usr/bin/env python3
"""Which library modules may call the platform max / min, and how often.

The library may not call `f32::max` / `f32::min` (and the `f64` / `f32x8` forms)
directly (`.github/clippy-lib/clippy.toml`). The one exception is the `raw_*`
helpers in `src/raw_minmax.rs`, for values that never reach a distance, bound,
mesh or persisted byte. This script keeps those calls visible:

  * every file under `src/` that names a `raw_*` helper is listed in
    `scripts/raw_minmax_manifest.toml` with exactly that many uses
  * a listed file with no use, or a different count, fails (lower the number when
    a use goes away; raising it needs a `why` that says what the new use is)
  * default-deny: only files under a prefix in `[scope] raw-allowed` may use one;
    a use anywhere else fails, and the manifest may not list a file outside them
  * a helper imported under another name (`use ... raw_max_f32 as m`) fails,
    since the count could not see the calls
  * no file scanned, or no `raw_*` definition found, fails

  python3 scripts/raw_minmax_manifest.py            check
  python3 scripts/raw_minmax_manifest.py --base REF also: a count above the one at
                                                    REF needs a `why` that differs
                                                    from the one at REF
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DEFINITIONS = "src/raw_minmax.rs"
MANIFEST = Path(__file__).resolve().parent / "raw_minmax_manifest.toml"
HELPERS = ("raw_max_f32", "raw_min_f32", "raw_max_f64", "raw_min_f64", "raw_max_f32x8", "raw_min_f32x8")
USE_RE = re.compile(r"\braw_(?:max|min)(?:_f32x8|_f32|_f64)?\b")
ALIAS_RE = re.compile(r"\b(?:" + "|".join(HELPERS) + r"|RawMinMax)\s+as\s+\w+")
DEF_RE = re.compile(r"\bfn\s+(" + "|".join(HELPERS) + r")\s*\(")


def strip_comments(text: str) -> str:
    """Rust source without comments (block and line); string literals are kept,
    which is fine here: a helper name inside a string still counts as a use"""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def scan(files: dict[str, str]) -> tuple[dict[str, int], list[str], int]:
    """(uses per file, problems, number of definitions) for {path: source}"""
    uses: dict[str, int] = {}
    problems: list[str] = []
    defs = 0
    for path, text in sorted(files.items()):
        code = strip_comments(text)
        if path == DEFINITIONS:
            defs = len(DEF_RE.findall(code))
            continue
        for m in ALIAS_RE.finditer(code):
            problems.append(f"{path}: {m.group(0)!r} imports a raw helper under another name")
        n = len(USE_RE.findall(code))
        if n:
            uses[path] = n
    return uses, problems, defs


def load(text: str) -> tuple[dict[str, dict], list[str]]:
    data = tomllib.loads(text)
    scope = data.get("scope", {}).get("raw-allowed", [])
    if not isinstance(scope, list) or not scope or not all(isinstance(p, str) and p.startswith("src/") for p in scope):
        raise ValueError("[scope] raw-allowed must be a non-empty list of src/ prefixes")
    allowed = {}
    for path, e in data.get("allowed", {}).items():
        if set(e) - {"uses", "why"}:
            raise ValueError(f"allowed.{path!r}: unknown keys {sorted(set(e) - {'uses', 'why'})}")
        if not isinstance(e.get("uses"), int) or e["uses"] <= 0:
            raise ValueError(f"allowed.{path!r}: `uses` must be a positive count")
        if not isinstance(e.get("why"), str) or not e["why"].strip():
            raise ValueError(f"allowed.{path!r}: `why` must say why the zero sign cannot matter there")
        allowed[path] = e
    return allowed, scope


def check(files: dict[str, str], manifest_text: str, base_text: str | None = None) -> list[str]:
    try:
        allowed, scope = load(manifest_text)
    except (ValueError, tomllib.TOMLDecodeError) as e:
        return [f"{MANIFEST.name}: {e}"]
    if not files:
        return ["no source file scanned"]
    uses, problems, defs = scan(files)
    if defs != len(HELPERS):
        problems.append(f"{DEFINITIONS}: found {defs} of the {len(HELPERS)} raw helper definitions")
    def in_scope(path: str) -> bool:
        return any(path.startswith(p) for p in scope)

    for path in allowed:
        if not in_scope(path):
            problems.append(f"{path}: listed as allowed but outside [scope] raw-allowed")
    for path, n in uses.items():
        if not in_scope(path):
            problems.append(f"{path}: uses a raw helper {n} time(s) outside [scope] raw-allowed")
        elif path not in allowed:
            problems.append(f"{path}: uses a raw helper {n} time(s) but is not in the manifest")
        elif allowed[path]["uses"] != n:
            problems.append(f"{path}: {n} use(s), the manifest says {allowed[path]['uses']}")
    for path in allowed:
        if path not in uses:
            problems.append(f"{path}: in the manifest but uses no raw helper (remove the entry)")
    if base_text is not None:
        try:
            base, _ = load(base_text)
        except (ValueError, tomllib.TOMLDecodeError):
            base = {}
        for path, e in allowed.items():
            old = base.get(path)
            if old is not None and e["uses"] > old["uses"] and e["why"].strip() == old["why"].strip():
                problems.append(f"{path}: uses went up from {old['uses']} to {e['uses']} without a new `why`")
    return problems


def read_tree() -> dict[str, str]:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): p.read_text(encoding="utf-8")
            for p in sorted(SRC.rglob("*.rs"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", help="a git ref whose manifest the counts are compared against")
    a = ap.parse_args(argv)
    base_text = None
    if a.base:
        r = subprocess.run(["git", "-C", str(ROOT), "show", f"{a.base}:scripts/{MANIFEST.name}"],
                           capture_output=True, text=True)
        base_text = r.stdout if r.returncode == 0 else ""
    files = read_tree()
    problems = check(files, MANIFEST.read_text(encoding="utf-8"), base_text)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if problems:
        return 1
    allowed, scope = load(MANIFEST.read_text(encoding="utf-8"))
    total = sum(e["uses"] for e in allowed.values())
    print(f"raw-minmax: {len(files)} files scanned, {total} raw use(s) in {len(allowed)} listed file(s), "
          f"raw helpers allowed only under {', '.join(scope)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
