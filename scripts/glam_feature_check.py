#!/usr/bin/env python3
"""ALICE-Law G3: no glam in the resolved build has its `fast-math` feature on.

glam's SSE2 `m128_mul_add` (Quat / Mat paths) becomes a fused multiply-add under the glam
feature `fast-math` together with the `fma` target feature, while every other path rounds
twice; clippy cannot see a feature, and any crate in the graph can turn it on for the one
glam build every crate shares. This script reads `cargo metadata` with every feature of this
crate on (all targets, so platform-specific dependencies count) and fails when a resolved
glam package has `fast-math` in its enabled features.

Fails on a glam with `fast-math`, and when it saw no glam package at all (compared nothing).

usage: scripts/glam_feature_check.py [--root DIR] [--metadata FILE]
  --metadata FILE  read cargo metadata JSON from a file instead of running cargo (tests)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BANNED = {"fast-math"}


def glam_features(meta: dict) -> list[tuple[str, list[str]]]:
    """(package id, enabled features) of every resolved glam package."""
    names = {p["id"]: p["name"] for p in meta["packages"]}
    return [(n["id"], sorted(n.get("features", [])))
            for n in meta["resolve"]["nodes"] if names.get(n["id"]) == "glam"]


def check(meta: dict) -> tuple[list[str], int]:
    seen = glam_features(meta)
    problems = [f"{pid}: feature {f} is on" for pid, feats in seen for f in feats if f in BANNED]
    return problems, len(seen)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--metadata")
    args = ap.parse_args(argv)
    if args.metadata:
        meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    else:
        out = subprocess.run(["cargo", "metadata", "--format-version", "1", "--all-features", "--locked"],
                             cwd=args.root, stdout=subprocess.PIPE, text=True, encoding="utf-8", check=True).stdout
        meta = json.loads(out)
    problems, n = check(meta)
    if n == 0:
        print("glam_feature_check: no glam package in the resolved graph (compared nothing)")
        return 2
    for p in problems:
        print(f"glam_feature_check: {p} (fused multiply-add on SSE2 with fma; ALICE-Law G3)")
    print(f"glam_feature_check: {n} glam package(s), {len(problems)} with a banned feature")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
