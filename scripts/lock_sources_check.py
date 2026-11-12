#!/usr/bin/env python3
"""Fail when a tracked Cargo.lock takes an alice-* crate (or libasp) by path from outside this repository or
from a registry, links more than one alice-det-math, or resolves an alice-* crate to another source than the
other tracked locks do

Every alice-* dependency comes by git (branch main + version): the lock records the commit. This
repository's own packages (the `[package]` names of its tracked Cargo.toml files) are by path. The fuzz
targets, tools and probes are their own workspaces with their own Cargo.lock, and a change that moves an
alice-* crate in one lock and not in the others builds one workspace on the new commit and leaves the
others on the old one, so every alice-* crate a lock takes by git must have the same source in every
tracked lock that has it.

A lock that has to keep another source of a crate on purpose (a reader of an old format kept to reproduce
data written by an old release, for example) is listed in `scripts/lock_sources.toml`:

    [[exempt]]
    lock = "compat/v0_10_reader/Cargo.lock"
    crate = "alice-det-math"
    reason = "why this lock needs the other source"
    test = "path/of/the/test/that/needs/it.rs"

That crate in that lock is left out of both checks. An entry fails when its lock is not tracked, the crate
is not in it, its test file is not in the repository, its reason is empty, or nothing about that crate in
that lock would fail without it (an exemption nothing needs).

    python3 scripts/lock_sources_check.py [--repo DIR]   (the locks `git ls-files` lists in DIR)

Exit 0 when every lock passes, 1 on a finding, 2 when no lock was read. Locks that share no alice-* crate
taken by git pass and say that the agreement compared nothing.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

# ALICE crates whose package name does not start with `alice`
OTHER_NAMES = {"libasp"}

EXEMPT_FILE = "scripts/lock_sources.toml"


def tracked(repo: Path, pattern: str) -> list[str]:
    return subprocess.run(["git", "-C", str(repo), "ls-files", pattern], capture_output=True, text=True,
                          check=True).stdout.split()


def own_packages(repo: Path) -> set[str]:
    """The package names of this repository's tracked manifests."""
    names = set()
    for manifest in tracked(repo, "*Cargo.toml"):
        with open(repo / manifest, "rb") as f:
            package = tomllib.load(f).get("package", {})
        if "name" in package:
            names.add(package["name"])
    return names


def alice_packages(text: str) -> list[dict]:
    return [p for p in tomllib.loads(text).get("package", [])
            if p["name"].startswith("alice") or p["name"] in OTHER_NAMES]


def findings(lock: str, text: str, own: set[str], exempt: frozenset[str] = frozenset()) -> list[str]:
    """The source findings of one lock, leaving out the crates in `exempt`."""
    pk = [p for p in alice_packages(text) if p["name"] not in exempt]
    out = []
    for p in pk:
        src = p.get("source", "")
        if (not src and p["name"] not in own) or (src and not src.startswith("git+")):
            out.append(f"{lock}: {p['name']} {p['version']} from {src or 'a path'}")
    if sum(p["name"] == "alice-det-math" for p in pk) > 1:
        out.append(f"{lock}: more than one alice-det-math")
    return out


def git_sources(text: str) -> dict[str, str]:
    """crate -> the git source(s) this lock takes it from"""
    out: dict[str, set[str]] = {}
    for p in alice_packages(text):
        src = p.get("source", "")
        if src.startswith("git+"):
            out.setdefault(p["name"], set()).add(src)
    return {name: " | ".join(sorted(s)) for name, s in out.items()}


def read_exemptions(repo: Path) -> tuple[list[dict[str, str]], list[str]]:
    path = repo / EXEMPT_FILE
    if not path.exists():
        return [], []
    entries, problems = [], []
    for i, e in enumerate(tomllib.loads(path.read_text(encoding="utf-8")).get("exempt", [])):
        missing = [k for k in ("lock", "crate", "reason", "test") if not str(e.get(k, "")).strip()]
        if missing:
            problems.append(f"{EXEMPT_FILE} exempt #{i + 1}: missing {', '.join(missing)}")
            continue
        if not (repo / e["test"]).is_file():
            problems.append(f"{EXEMPT_FILE} exempt #{i + 1}: test {e['test']} is not a file here")
        entries.append({k: str(e[k]) for k in ("lock", "crate", "reason", "test")})
    return entries, problems


def disagreements(texts: dict[str, str], exempt: set[tuple[str, str]]) -> tuple[list[str], int]:
    """(findings, crates compared) of the git sources across locks"""
    seen: dict[str, dict[str, str]] = {}
    for lock, text in texts.items():
        for name, src in git_sources(text).items():
            if (lock, name) not in exempt:
                seen.setdefault(name, {})[lock] = src
    out, compared = [], 0
    for name, by_lock in sorted(seen.items()):
        if len(by_lock) < 2:
            continue
        compared += 1
        if len(set(by_lock.values())) > 1:
            out.append(f"{name} resolves differently: "
                       + "; ".join(f"{lock}: {src}" for lock, src in sorted(by_lock.items())))
    return out, compared


def check(repo: Path) -> tuple[list[str], int, int]:
    """(findings, locks read, crates compared across locks)"""
    locks = tracked(repo, "*Cargo.lock")
    texts = {lock: (repo / lock).read_text(encoding="utf-8") for lock in locks}
    own = own_packages(repo)
    entries, bad = read_exemptions(repo)
    exempt = {(e["lock"], e["crate"]) for e in entries}
    by_lock: dict[str, set[str]] = {}
    for lock, crate in exempt:
        by_lock.setdefault(lock, set()).add(crate)
    for lock, text in texts.items():
        bad += findings(lock, text, own, frozenset(by_lock.get(lock, set())))
    agree, compared = disagreements(texts, exempt)
    bad += agree
    for e in entries:
        lock, crate = e["lock"], e["crate"]
        if lock not in texts or crate not in {p["name"] for p in alice_packages(texts[lock])}:
            bad.append(f"exempt {crate} in {lock}: no such crate in a tracked lock of that path")
            continue
        others = exempt - {(lock, crate)}
        own_findings = [f for f in findings(lock, texts[lock], own,
                                            frozenset(c for lk, c in others if lk == lock))
                        if f.startswith(f"{lock}: {crate} ") or (crate == "alice-det-math" and "more than one" in f)]
        cross, _ = disagreements(texts, others)
        if not own_findings and not any(f.startswith(f"{crate} resolves") for f in cross):
            bad.append(f"exempt {crate} in {lock}: nothing fails without it (remove the entry)")
    return bad, len(locks), compared


def main(argv: list[str]) -> int:
    repo = Path(".")
    args = list(argv)
    if args[:1] == ["--repo"] and len(args) >= 2:
        repo, args = Path(args[1]), args[2:]
    if args:
        print(f"lock sources: unknown arguments {args} (exemptions go in {EXEMPT_FILE})", file=sys.stderr)
        return 2
    bad, locks, compared = check(repo)
    agreement = (f"{compared} alice-* crate(s) compared across locks" if compared
                 else "no alice-* crate taken by git in two locks (the agreement compared nothing)")
    print(f"lock sources: {locks} lock(s) read, {len(bad)} finding(s); {agreement}")
    for f in bad:
        print(f"  {f}")
    if not locks:
        return 2
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
