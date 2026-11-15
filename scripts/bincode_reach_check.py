#!/usr/bin/env python3
"""bincode (RUSTSEC-2025-0141, unmaintained, no fixed release) is resolved only through the
`asdf` feature, and only two files use it.

A consumer that builds this crate with `default-features = false` and without a feature that
turns `asdf` on (ALICE-LOL: `glsl` only) must not resolve bincode at all, so it carries no
advisory hold for it. This reads `cargo metadata --locked` for three feature sets and walks
the resolve graph from this package along normal and build edges (dev edges are not part of
what a consumer builds):

  - `--no-default-features --features glsl` (the ALICE-LOL set): bincode is not reached
  - `--no-default-features`: bincode is not reached
  - default features: bincode IS reached (the walk and the package name are right; a check
    that never finds bincode anywhere would pass the first two vacuously)

and scans src/ for `bincode::`: the files that name it must be exactly src/asp_bridge.rs and
src/io/asdf.rs (both compiled only with `asdf`; the `--no-default-features` build lanes are
the compile-side half: a third user without the gate fails to resolve the crate there).

Fails when a walk reached no package or no src file was read (compared nothing).

usage: scripts/bincode_reach_check.py [--root DIR] [--metadata SET=FILE ...]
  --metadata SET=FILE  read cargo metadata JSON for SET (lol, none, default) from FILE
                       instead of running cargo (tests)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

CRATE = "bincode"
USERS = {"src/asp_bridge.rs", "src/io/asdf.rs"}
SETS = {
    "lol": ["--no-default-features", "--features", "glsl"],
    "none": ["--no-default-features"],
    "default": [],
}
EXPECT_REACHED = {"lol": False, "none": False, "default": True}
USE_RE = re.compile(r"\bbincode::")


def reached(meta: dict) -> set[str]:
    """names of the packages the root package builds with (normal and build edges, not dev)"""
    names = {p["id"]: p["name"] for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    if root is None or root not in nodes:
        return set()
    seen, todo = {root}, [root]
    while todo:
        for dep in nodes[todo.pop()].get("deps", []):
            kinds = {k.get("kind") for k in dep.get("dep_kinds", [])}
            if kinds and kinds <= {"dev"}:
                continue
            if dep["pkg"] not in seen and dep["pkg"] in nodes:
                seen.add(dep["pkg"])
                todo.append(dep["pkg"])
    return {names[i] for i in seen if i in names}


def users(root: Path) -> tuple[set[str], int]:
    """src files naming `bincode::` outside comments, and how many files were read"""
    found, n = set(), 0
    for f in sorted((root / "src").rglob("*.rs")):
        n += 1
        for line in f.read_text(encoding="utf-8").splitlines():
            code = line.split("//", 1)[0]
            if USE_RE.search(code):
                found.add(f.relative_to(root).as_posix())
                break
    return found, n


def check(metas: dict[str, dict], found: set[str], files: int) -> list[str]:
    problems = []
    for name, meta in metas.items():
        pkgs = reached(meta)
        if not pkgs:
            problems.append(f"{name}: the walk reached no package (compared nothing)")
            continue
        has = CRATE in pkgs
        if has != EXPECT_REACHED[name]:
            want = "reached" if EXPECT_REACHED[name] else "not reached"
            problems.append(f"{name} ({' '.join(SETS[name]) or 'default features'}): {CRATE} is "
                            f"{'reached' if has else 'not reached'}, expected {want} ({len(pkgs)} packages)")
    if files == 0:
        problems.append("src: no .rs file read (compared nothing)")
    elif found != USERS:
        problems.append(f"src: `{CRATE}::` is used in {sorted(found)}, expected exactly {sorted(USERS)}"
                        " (a new user goes behind the `asdf` feature and is added here)")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--metadata", action="append", default=[])
    args = ap.parse_args(argv)
    root = Path(args.root)
    given = dict(m.split("=", 1) for m in args.metadata)
    metas = {}
    for name, flags in SETS.items():
        if name in given:
            metas[name] = json.loads(Path(given[name]).read_text(encoding="utf-8"))
        else:
            out = subprocess.run(["cargo", "metadata", "--format-version", "1", "--locked", *flags],
                                 cwd=root, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                 check=True).stdout
            metas[name] = json.loads(out)
    found, files = users(root)
    problems = check(metas, found, files)
    for p in problems:
        print(f"bincode_reach_check: {p}")
    print(f"bincode_reach_check: {len(metas)} feature set(s), {files} src file(s), "
          f"users {sorted(found)}, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
