#!/usr/bin/env python3
"""The local ../ALICE-* checkouts are the commits CI builds against.

Until the versions a manifest needs are on crates.io, it takes them from sibling
checkouts ("../ALICE-Zip", or a package inside one, "../ALICE-Physics/crates/alice-physics");
CI clones each at the full SHA in .github/alice-siblings
("<Repo> <40-hex SHA>" per line, "#" comments) through .github/actions/alice-siblings.
This check makes the local build use the same commits. It fails when a line is
malformed or repeated, when a manifest names a sibling path that is not listed (by
file and line) or a listed repository is named by no manifest, when a manifest
path does not resolve to the sibling directory, or when a sibling is missing, at
another commit or has uncommitted changes.

An empty or missing list is right only for a repository whose manifests name no
sibling path (its siblings come by git and rev, or not at all): the manifests are
still read (none read is a failure), and the pass says "N manifests scanned, 0 path
deps, nothing to pin". A sibling path beside an empty list is a failure naming it.
Manifests are read as TOML: the dependencies are the entries of the dependency tables
(`[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`, their `[target.<cfg>.*]` forms,
`[workspace.dependencies]` and `[patch.<source>]`) that have a `path`, however the TOML spells them
(quoted or dotted keys, any string form); comments and other strings are not dependencies. A manifest
that does not parse is a failure.

    python3 scripts/alice_siblings_check.py [repo root]
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

# a POSIX string, not a Path: it appears in messages that must read the same on every OS
LIST_FILE = ".github/alice-siblings"
# read in the repository root and in every [workspace] member (members may be globs,
# [workspace].exclude removes matches): the package manifest and its fuzz manifest
MANIFESTS = ("Cargo.toml", "fuzz/Cargo.toml")
LINE_RE = re.compile(r"(ALICE-[A-Za-z0-9-]+) ([0-9a-f]{40})")
# a sibling, or a package inside one: after the repository only plain components (no "."
# or "..", no empty one), so the repository the path names is the one it reaches
# a sibling, or a package inside one: after the repository only plain components (no "." or "..", no empty one), so
# the repository the path names is the one it reaches
SIBLING_RE = re.compile(r"((?:\.\./)+)(ALICE-[A-Za-z0-9-]+)((?:/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*)/?")
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def dependency_paths(doc: dict) -> list[str]:
    """the `path` of every dependency entry of a parsed manifest (see the module doc for the tables read)"""
    out: list[str] = []

    def table(t: object) -> None:
        if isinstance(t, dict):
            for spec in t.values():
                if isinstance(spec, dict) and isinstance(spec.get("path"), str):
                    out.append(spec["path"])

    for k in DEP_TABLES:
        table(doc.get(k))
    for tgt in (doc.get("target") or {}).values():
        if isinstance(tgt, dict):
            for k in DEP_TABLES:
                table(tgt.get(k))
    table((doc.get("workspace") or {}).get("dependencies"))
    for src in (doc.get("patch") or {}).values():
        table(src)
    return out


def line_of(text: str, value: str) -> str:
    """the first line naming `value` (for the message only; the dependency itself is read as TOML)"""
    for no, line in enumerate(text.splitlines(), 1):
        if value in line:
            return str(no)
    return "?"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def read_list(root: Path) -> tuple[dict[str, str], list[str]]:
    """(repository -> pinned commit, problems with the lines); a missing or empty list is an empty mapping (whether
    that is right depends on the manifests: see `check`)"""
    listed: dict[str, str] = {}
    problems: list[str] = []
    path = root / LIST_FILE
    if not path.is_file():
        return listed, []
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = " ".join(raw.split("#", 1)[0].split())
        if not line:
            continue
        m = LINE_RE.fullmatch(line)
        if not m:
            problems.append(f"{LIST_FILE}:{no}: not '<ALICE-Repo> <40-hex SHA>': {raw!r}")
        elif m.group(1) in listed:
            problems.append(f"{LIST_FILE}:{no}: {m.group(1)} is listed twice")
        else:
            listed[m.group(1)] = m.group(2)
    return listed, problems


def manifests(root: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """([(POSIX path relative to root, file)], problems): MANIFESTS in the root and in each
    [workspace] member; a member pattern matching no package and reading no manifest fail"""
    problems: list[str] = []
    dirs = [root]
    top = root / "Cargo.toml"
    if top.is_file():
        try:
            workspace = tomllib.loads(top.read_text(encoding="utf-8")).get("workspace", {})
        except tomllib.TOMLDecodeError:
            workspace = {}  # reported as "is not TOML" when the manifest itself is read
        excluded: set[Path] = set()
        for pattern in workspace.get("exclude", []):
            excluded |= {d.resolve() for d in root.glob(pattern)}
        for pattern in workspace.get("members", []):
            matched = sorted(d for d in root.glob(pattern) if (d / "Cargo.toml").is_file())
            matched = [d for d in matched if d.resolve() not in excluded]
            if not matched and not any(d.resolve() in excluded for d in root.glob(pattern)):
                problems.append(f"Cargo.toml: workspace member {pattern!r} matches no package")
            dirs += [d for d in matched if d.resolve() != root.resolve() and d not in dirs]
    found: list[tuple[str, Path]] = []
    for d in dirs:
        for rel in MANIFESTS:
            f = d / rel
            if f.is_file():
                found.append((f.relative_to(root).as_posix(), f))
    if not found and not problems:
        problems.append("no Cargo.toml was read (compared nothing)")
    return found, problems


def check(root: Path) -> list[str]:
    listed, problems = read_list(root)
    if problems:
        return problems

    found, problems = manifests(root)
    if problems:
        return problems

    named: set[str] = set()
    for rel, manifest in found:
        text = manifest.read_text(encoding="utf-8")
        try:
            doc = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            problems.append(f"{rel} is not TOML: {e}")
            continue
        for path in dependency_paths(doc):
            m = SIBLING_RE.fullmatch(path)
            if m is None:
                if "ALICE-" in path:
                    problems.append(f'{rel} names path = "{path}", not ../ALICE-<Repo> or a plain path inside it')
                continue
            repo = m.group(2)
            named.add(repo)
            if (manifest.parent / f"{m.group(1)}{repo}").resolve() != (root / ".." / repo).resolve():
                problems.append(f'{rel} names path = "{path}", not the sibling ../{repo}')
            if repo not in listed:
                problems.append(f"{rel}:{line_of(text, path)} names ../{repo}, which {LIST_FILE} does not list")
    for repo in sorted(set(listed) - named):
        problems.append(f"{LIST_FILE} lists {repo}, which no manifest names as a path")

    for repo, ref in listed.items():
        sibling = (root / ".." / repo).resolve()
        if not (sibling / ".git").exists():
            problems.append(f"../{repo} is not a git checkout; clone {repo} there at {ref}")
            continue
        head = git(sibling, "rev-parse", "HEAD")
        if head != ref:
            problems.append(f"../{repo} is at {head}, {LIST_FILE} says {ref}")
        if git(sibling, "status", "--porcelain", "--untracked-files=no"):
            problems.append(f"../{repo} has uncommitted changes; CI builds the clean commit")
    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    problems = check(root)
    for p in problems:
        print(f"alice_siblings_check: {p}")
    if problems:
        return 1
    listed, _ = read_list(root)
    if not listed:
        # check() passed with nothing listed: the manifests were read and name no sibling path
        found, _ = manifests(root)
        print(f"alice_siblings_check: {len(found)} manifests scanned, 0 path deps, nothing to pin")
        return 0
    for repo, ref in listed.items():
        print(f"alice_siblings_check: ../{repo} at {ref}")
    print(f"alice_siblings_check: {len(listed)} sibling(s), each at the commit CI builds against")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
