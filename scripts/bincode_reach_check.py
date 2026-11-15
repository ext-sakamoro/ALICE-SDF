#!/usr/bin/env python3
"""bincode (RUSTSEC-2025-0141, unmaintained, no fixed release) is resolved in no feature set; the
binary `.asdf` body and the ASP bridge use alice-serial's bincode1 (the bincode 1 layout), and that
is resolved only through the `asdf` feature, used by two files.

A consumer that builds this crate with `default-features = false` and without a feature that
turns `asdf` on (ALICE-LOL: `glsl` only) must not resolve alice-serial either. This reads
`cargo metadata --locked` for four feature sets and walks the resolve graph from this package
along normal and build edges (dev edges are not part of what a consumer builds):

  - `--no-default-features --features glsl` (the ALICE-LOL set): neither is reached
  - `--no-default-features`: neither is reached
  - default features: alice-serial IS reached, bincode is not (the walk and the package names
    are right; a check that never finds alice-serial anywhere would pass the others vacuously)
  - `--all-features`: alice-serial is reached, bincode is not

and scans src/ outside comments: no file names `bincode::`, and the files that name
`bincode1::` are exactly src/asp_bridge.rs and src/io/asdf.rs (both compiled only with `asdf`;
the `--no-default-features` build lanes are the compile-side half).

Fails when a walk reached no package or no src file was read (compared nothing).

usage: scripts/bincode_reach_check.py [--root DIR] [--metadata SET=FILE ...]
  --metadata SET=FILE  read cargo metadata JSON for SET (lol, none, default, all) from FILE
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
CODEC = "alice-serial"
USERS = {"src/asp_bridge.rs", "src/io/asdf.rs"}
SETS = {
    "lol": ["--no-default-features", "--features", "glsl"],
    "none": ["--no-default-features"],
    "default": [],
    "all": ["--all-features"],
}
# alice-serial is reached exactly where `asdf` is on; bincode nowhere
EXPECT_CODEC = {"lol": False, "none": False, "default": True, "all": True}
USE_RE = re.compile(r"\bbincode::")
CODEC_RE = re.compile(r"\bbincode1::")


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


def users(root: Path) -> tuple[set[str], set[str], int]:
    """src files naming `bincode::` and `bincode1::` outside comments, and how many files were read"""
    old, codec, n = set(), set(), 0
    for f in sorted((root / "src").rglob("*.rs")):
        n += 1
        rel = f.relative_to(root).as_posix()
        for line in f.read_text(encoding="utf-8").splitlines():
            code = line.split("//", 1)[0]
            if USE_RE.search(code):
                old.add(rel)
            if CODEC_RE.search(code):
                codec.add(rel)
    return old, codec, n


def check(metas: dict[str, dict], old: set[str], codec: set[str], files: int) -> list[str]:
    problems = []
    for name, meta in metas.items():
        pkgs = reached(meta)
        flags = " ".join(SETS[name]) or "default features"
        if not pkgs:
            problems.append(f"{name}: the walk reached no package (compared nothing)")
            continue
        if CRATE in pkgs:
            problems.append(f"{name} ({flags}): {CRATE} is reached ({len(pkgs)} packages); "
                            f"the stored forms use {CODEC}'s bincode1")
        has = CODEC in pkgs
        if has != EXPECT_CODEC[name]:
            want = "reached" if EXPECT_CODEC[name] else "not reached"
            problems.append(f"{name} ({flags}): {CODEC} is {'reached' if has else 'not reached'}, "
                            f"expected {want} ({len(pkgs)} packages)")
    if files == 0:
        problems.append("src: no .rs file read (compared nothing)")
    else:
        if old:
            problems.append(f"src: `{CRATE}::` is used in {sorted(old)}; use alice_serial::bincode1")
        if codec != USERS:
            problems.append(f"src: `bincode1::` is used in {sorted(codec)}, expected exactly "
                            f"{sorted(USERS)} (a new user goes behind the `asdf` feature and is added here)")
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
    old, codec, files = users(root)
    problems = check(metas, old, codec, files)
    for p in problems:
        print(f"bincode_reach_check: {p}")
    print(f"bincode_reach_check: {len(metas)} feature set(s), {files} src file(s), "
          f"bincode1 users {sorted(codec)}, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
