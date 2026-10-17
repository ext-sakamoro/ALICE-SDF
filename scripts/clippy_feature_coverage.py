#!/usr/bin/env python3
"""Every feature of Cargo.toml is compiled by a clippy run of CI (ALICE-Law G1).

clippy.toml's bans (libm, platform max / min, clocks, ...) only look at code that is
compiled: a module behind a feature no clippy run turns on is not checked at all. This
reads the `cargo clippy` commands of the clippy job in .github/workflows/ci.yml
(`--features` lists, with `$NAME` taken from the job's `env`), and fails when a
feature of Cargo.toml (other than `default`) is on in none of them, when a listed name
is not a feature, when scripts/preflight.sh's LINT_FEATURES differs from the job's,
and when it read no clippy command (compared nothing).

usage: python3 scripts/clippy_feature_coverage.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

JOB = "clippy"


def job_text(workflow: str, job: str) -> str:
    m = re.search(rf"^  {re.escape(job)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)", workflow, re.M | re.S)
    return m.group(1) if m else ""


def job_env(text: str) -> dict[str, str]:
    """The job-level `env:` block (4-space key, 6-space entries)"""
    m = re.search(r"^    env:\n((?:      .*\n|\s*#.*\n)+)", text, re.M)
    out = {}
    if m:
        for k, v in re.findall(r'^      ([A-Z_][A-Z0-9_]*):\s*"?([^"\n]*)"?\s*$', m.group(1), re.M):
            out[k] = v.strip()
    return out


def clippy_feature_sets(text: str, env: dict[str, str]) -> list[set[str]]:
    sets = []
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        if "cargo clippy" not in code:
            continue
        feats: set[str] = set()
        for m in re.finditer(r'--features[ =](?:"([^"]*)"|(\S+))', code):
            raw = m.group(1) if m.group(1) is not None else m.group(2)
            raw = re.sub(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", lambda v: env.get(v.group(1), "\0" + v.group(1)), raw)
            feats |= {f.strip() for f in raw.split(",") if f.strip()}
        sets.append(feats)
    return sets


def preflight_lint_features(text: str) -> str | None:
    m = re.search(r"^LINT_FEATURES='([^']*)'", text, re.M)
    return m.group(1) if m else None


def check(root: Path) -> tuple[list[str], int]:
    features = set(tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8")).get("features", {}))
    features.discard("default")
    text = job_text((root / ".github/workflows/ci.yml").read_text(encoding="utf-8"), JOB)
    env = job_env(text)
    sets = clippy_feature_sets(text, env)
    problems = []
    if not sets:
        return [f"ci.yml job `{JOB}`: no `cargo clippy` command read (compared nothing)"], 0
    covered = set().union(*sets)
    for f in sorted(covered):
        if f.startswith("\0"):
            problems.append(f"ci.yml job `{JOB}`: ${f[1:]} is not set in the job's env")
        elif f not in features:
            problems.append(f"ci.yml job `{JOB}`: `{f}` is not a feature of Cargo.toml")
    for f in sorted(features - covered):
        problems.append(f"feature `{f}` is on in no clippy run of the `{JOB}` job, so clippy.toml's bans never see its code")
    pre = (root / "scripts/preflight.sh")
    if pre.exists():
        p = preflight_lint_features(pre.read_text(encoding="utf-8"))
        if p is None:
            problems.append("scripts/preflight.sh: no LINT_FEATURES='...' line")
        elif "LINT_FEATURES" in env and p != env["LINT_FEATURES"]:
            problems.append("scripts/preflight.sh LINT_FEATURES differs from the clippy job's env LINT_FEATURES")
    return problems, len(features)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args(argv)
    problems, n = check(Path(a.root))
    for p in problems:
        print(f"clippy-feature-coverage: {p}")
    if n == 0 and not problems:
        print("clippy-feature-coverage: Cargo.toml has no feature (compared nothing)")
        return 1
    print(f"clippy-feature-coverage: {n} features, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
