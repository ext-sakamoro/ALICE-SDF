#!/usr/bin/env python3
"""Fail when a tracked Cargo.lock takes an alice-* crate (or libasp) by path from outside this repository or
from a registry, or links more than one alice-det-math

Every alice-* dependency comes by git (branch main + version): the lock records the commit. This
repository's own packages (the `[package]` names of its tracked Cargo.toml files) are by path. A
lock that takes a published alice-* on purpose is left out with --skip PREFIX (a path prefix).

    python3 scripts/lock_sources_check.py [--skip PREFIX ...]   (the locks `git ls-files` lists)

Exit 0 when every lock passes, 1 on a finding, 2 when no lock was compared.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib

def tracked(pattern: str) -> list[str]:
    return subprocess.run(["git", "ls-files", pattern], capture_output=True, text=True,
                          check=True).stdout.split()


def own_packages() -> set[str]:
    """The package names of this repository's tracked manifests."""
    names = set()
    for manifest in tracked("*Cargo.toml"):
        with open(manifest, "rb") as f:
            package = tomllib.load(f).get("package", {})
        if "name" in package:
            names.add(package["name"])
    return names


# ALICE crates whose package name does not start with `alice`
OTHER_NAMES = {"libasp"}


def findings(lock: str, text: str, own: set[str]) -> list[str]:
    pk = [p for p in tomllib.loads(text).get("package", [])
          if p["name"].startswith("alice") or p["name"] in OTHER_NAMES]
    out = []
    for p in pk:
        src = p.get("source", "")
        if (not src and p["name"] not in own) or (src and not src.startswith("git+")):
            out.append(f"{lock}: {p['name']} {p['version']} from {src or 'a path'}")
    if sum(p["name"] == "alice-det-math" for p in pk) > 1:
        out.append(f"{lock}: more than one alice-det-math")
    return out


def main(argv: list[str]) -> int:
    skip = tuple(argv[i + 1] for i, a in enumerate(argv) if a == "--skip")
    locks = [lock for lock in tracked("*Cargo.lock") if not lock.startswith(skip)] if skip or not argv else []
    own = own_packages()
    bad = [f for lock in locks for f in findings(lock, open(lock, encoding="utf-8").read(), own)]
    print(f"lock sources: {len(locks)} lock(s) compared, {len(bad)} finding(s)")
    for f in bad:
        print(f"  {f}")
    if not locks:
        return 2
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
