#!/usr/bin/env python3
"""The default build of alice-sdf links permissively licensed crates only.

alice-sdf is Apache-2.0. Some optional features (`physics`, `codec`, `sdf-cache`)
link crates under the AGPL or a commercial licence; they are not default and not part
of the Law path. This reads `cargo metadata` with the default features, walks the
resolved graph from alice-sdf along normal and build dependencies (not dev), and
fails when a reached package's licence expression has no alternative made only of
permissive licences. A package without a licence expression fails too. Reaching no
package other than the crate itself compares nothing and fails.

usage: python3 scripts/default_dependency_licenses.py [--metadata FILE]
  --metadata FILE   read this `cargo metadata --format-version 1` output instead of running cargo
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERMISSIVE = {
    "MIT", "MIT-0", "Apache-2.0", "Apache-2.0 WITH LLVM-exception", "BSD-2-Clause", "BSD-3-Clause",
    "ISC", "Zlib", "Unicode-3.0", "Unicode-DFS-2016", "BSL-1.0", "CC0-1.0", "Unlicense", "0BSD",
}


def alternatives(expr: str) -> list[list[str]]:
    """The expression as OR of AND terms (parentheses flattened; `/` as OR, the old cargo form)"""
    e = expr.replace("/", " OR ").replace("(", " ").replace(")", " ")
    out = []
    for alt in re.split(r"\s+OR\s+", e.strip()):
        terms = [t.strip() for t in re.split(r"\s+AND\s+", alt.strip()) if t.strip()]
        out.append(terms)
    return out


def permissive(expr: str | None) -> bool:
    if not expr:
        return False
    return any(terms and all(t in PERMISSIVE for t in terms) for terms in alternatives(expr))


def reached(meta: dict) -> list[dict]:
    """Packages reached from the root along normal and build dependencies"""
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen = {root}
    todo = [root]
    while todo:
        n = nodes[todo.pop()]
        for d in n.get("deps", []):
            kinds = {k.get("kind") for k in d.get("dep_kinds", [])}
            if kinds & {None, "build"} and d["pkg"] not in seen:
                seen.add(d["pkg"])
                todo.append(d["pkg"])
    seen.discard(root)
    return [pkgs[i] for i in sorted(seen)]


def check(meta: dict) -> tuple[list[str], int]:
    problems = []
    pkgs = reached(meta)
    for p in pkgs:
        if not permissive(p.get("license")):
            problems.append(f"{p['name']} {p['version']}: licence {p.get('license')!r} is not permissive")
    return problems, len(pkgs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata")
    a = ap.parse_args(argv)
    if a.metadata:
        meta = json.loads(Path(a.metadata).read_text(encoding="utf-8"))
    else:
        r = subprocess.run(["cargo", "metadata", "--format-version", "1", "--locked"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"default-dependency-licenses: cargo metadata failed: {r.stderr.strip()[:300]}", file=sys.stderr)
            return 2
        meta = json.loads(r.stdout)
    problems, n = check(meta)
    if n == 0:
        print("default-dependency-licenses: reached no dependency (compared nothing)", file=sys.stderr)
        return 2
    for p in problems:
        print(p)
    print(f"default-dependency-licenses: {n} packages in the default build, {len(problems)} not permissive")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
