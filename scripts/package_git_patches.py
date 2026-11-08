"""Patches for packaging a crate whose dependencies are taken from git (a rev, a branch or a tag).

While a sibling's version is not on crates.io, Cargo.toml takes it as
`{ git = URL, rev = <40 hex>, version = "X" }`. `cargo package` drops `git` and
`rev`, so the packaged crate would resolve the version from crates.io and fail.
The Package integrity job packages and builds the crate with a
`[patch.crates-io]` entry for each such dependency, at the rev Cargo.toml names,
so the .crate is checked against what it is consumed with.

  args          prints the `--config` arguments (one per line) for those patches
  verify LOCK   reads the packaged Cargo.lock and fails when
                  - a pinned dependency is not in it, or is resolved from anywhere
                    but git at exactly the pinned rev (the patch was not applied),
                  - it holds a git source that Cargo.toml does not pin
                    (a patch that does not come from Cargo.toml); a package of the
                    same repository at the pinned rev is pinned by it (a facade
                    and the crates it reaches by path in that repository)

A dependency that names `git` and `version` without a `rev` (`branch`, `tag` or
the default branch) is patched with the same reference (`branch = ..`, `tag = ..`
or none), not with a `rev`: the crates it depends on through git take their
siblings by that reference too, and a patch by `rev` would give cargo a second
source for the same package (two copies linked, 2026-11-08: alice-det-math by
`?rev=` from the patch and by `?branch=main` from alice-analytics). `verify`
expects the source cargo writes for that reference at the commit Cargo.lock
resolved it to, read by `locked_git_commit` of scripts/id_move_check.py (the one
place that rule lives): no Cargo.lock, the package not in it from that url, a
source without a 40-hex commit, or two commits fail both modes, as does a `rev`
that is not 40 hex, or a dependency naming more than one of rev / branch / tag.

Before alice-physics is published, the pinned sibling is published and the
dependency goes back to `version` only; then there is nothing to patch.

run: python3 scripts/package_git_patches.py args
     python3 scripts/package_git_patches.py verify <unpacked crate>/Cargo.lock
"""
from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from id_move_check import CheckError, locked_git_commit  # noqa: E402
REV = re.compile(r"[0-9a-f]{40}")
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def _dep_tables(manifest: dict):
    for t in DEP_TABLES:
        yield t, manifest.get(t, {})
    for target, body in manifest.get("target", {}).items():
        for t in DEP_TABLES:
            yield f"target.{target}.{t}", body.get(t, {})


REFS = ("rev", "branch", "tag")


def pins(manifest_text: str, root: str = ROOT) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    """(name, git, commit, reference) of each dependency taken from git with a version: the
    reference is how Cargo.toml names it (`rev=<40 hex>`, `branch=<name>`, `tag=<name>`, or ""
    for the default branch), the commit is the rev, or for the others the commit `root`'s
    Cargo.lock resolved it to; problems."""
    manifest = tomllib.loads(manifest_text)
    found: dict[str, tuple[str, str, str, str]] = {}
    problems: list[str] = []
    for table, deps in _dep_tables(manifest):
        for key, spec in deps.items():
            if not isinstance(spec, dict) or "git" not in spec or "version" not in spec:
                continue
            name = spec.get("package", key)
            named = [k for k in REFS if k in spec]
            if len(named) > 1:
                problems.append(f"{table}.{key}: names more than one of {', '.join(named)}")
                continue
            rev = spec.get("rev")
            if rev is not None:
                if not REV.fullmatch(rev):
                    problems.append(f"{table}.{key}: git with a version but rev is not 40 hex ({rev!r})")
                    continue
                commit, ref = rev, f"rev={rev}"
            else:
                try:
                    commit = locked_git_commit(Path(root), Path(root), name, spec["git"])
                except CheckError as e:
                    problems.append(f"{table}.{key}: {e}")
                    continue
                ref = f"{named[0]}={spec[named[0]]}" if named else ""
            pin = (name, spec["git"], commit, ref)
            if name in found and found[name] != pin:
                problems.append(f"{name}: pinned twice with different sources {found[name]} / {pin}")
                continue
            found[name] = pin
    return sorted(found.values()), problems


def config_args(pinned: list[tuple[str, str, str, str]]) -> list[str]:
    out = []
    for name, git, _commit, ref in pinned:
        out += ["--config", f'patch.crates-io.{name}.git="{git}"']
        if ref:
            kind, value = ref.split("=", 1)
            out += ["--config", f'patch.crates-io.{name}.{kind}="{value}"']
    return out


def source_of(git: str, commit: str, ref: str) -> str:
    """the source cargo writes in Cargo.lock for a git dependency by that reference at that commit"""
    return f"git+{git}?{ref}#{commit}" if ref else f"git+{git}#{commit}"


def verify(pinned: list[tuple[str, str, str, str]], lock_text: str) -> list[str]:
    lock = tomllib.loads(lock_text)
    packages = lock.get("package", [])
    problems = []
    want = {name: source_of(git, commit, ref) for name, git, commit, ref in pinned}
    for name, source in want.items():
        got = [p.get("source", "(path)") for p in packages if p.get("name") == name]
        if not got:
            problems.append(f"{name}: not in the packaged Cargo.lock")
        elif got != [source]:
            problems.append(f"{name}: resolved from {got}, not the pinned {source} (patch not applied)")
    pinned_sources = set(want.values())
    for p in packages:
        source = p.get("source", "")
        if source.startswith("git+") and source not in pinned_sources:
            problems.append(f"{p.get('name')}: git source {source} that Cargo.toml does not pin")
    return problems


def main(argv: list[str]) -> int:
    with open(os.path.join(ROOT, "Cargo.toml"), encoding="utf-8") as f:
        pinned, problems = pins(f.read())
    if problems:
        for p in problems:
            print(f"package_git_patches: {p}", file=sys.stderr)
        return 1
    if argv[:1] == ["args"] and len(argv) == 1:
        print("\n".join(config_args(pinned)))
        return 0
    if argv[:1] == ["verify"] and len(argv) == 2:
        with open(argv[1], encoding="utf-8") as f:
            problems = verify(pinned, f.read())
        for p in problems:
            print(f"package_git_patches: {p}", file=sys.stderr)
        print(f"package_git_patches: {len(pinned)} pinned git dependency(ies) checked against {argv[1]}, "
              f"{len(problems)} problem(s)")
        return 1 if problems else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
