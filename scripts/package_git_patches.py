"""Patches for packaging a crate whose dependencies are taken from git at a pinned rev.

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
                    (a patch that does not come from Cargo.toml)

Cargo.toml fails both modes when a dependency names `git` and `version` without
a 40-hex `rev` (not pinned).

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = re.compile(r"[0-9a-f]{40}")
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def _dep_tables(manifest: dict):
    for t in DEP_TABLES:
        yield t, manifest.get(t, {})
    for target, body in manifest.get("target", {}).items():
        for t in DEP_TABLES:
            yield f"target.{target}.{t}", body.get(t, {})


def pins(manifest_text: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """(name, git, rev) of each dependency taken from git with a version; problems."""
    manifest = tomllib.loads(manifest_text)
    found: dict[str, tuple[str, str, str]] = {}
    problems: list[str] = []
    for table, deps in _dep_tables(manifest):
        for key, spec in deps.items():
            if not isinstance(spec, dict) or "git" not in spec or "version" not in spec:
                continue
            name = spec.get("package", key)
            rev = spec.get("rev", "")
            if not REV.fullmatch(rev):
                problems.append(f"{table}.{key}: git with a version but rev is not 40 hex ({rev!r})")
                continue
            pin = (name, spec["git"], rev)
            if name in found and found[name] != pin:
                problems.append(f"{name}: pinned twice with different sources {found[name]} / {pin}")
                continue
            found[name] = pin
    return sorted(found.values()), problems


def config_args(pinned: list[tuple[str, str, str]]) -> list[str]:
    out = []
    for name, git, rev in pinned:
        out += ["--config", f'patch.crates-io.{name}.git="{git}"']
        out += ["--config", f'patch.crates-io.{name}.rev="{rev}"']
    return out


def verify(pinned: list[tuple[str, str, str]], lock_text: str) -> list[str]:
    lock = tomllib.loads(lock_text)
    packages = lock.get("package", [])
    problems = []
    want = {name: f"git+{git}?rev={rev}#{rev}" for name, git, rev in pinned}
    for name, source in want.items():
        got = [p.get("source", "(path)") for p in packages if p.get("name") == name]
        if not got:
            problems.append(f"{name}: not in the packaged Cargo.lock")
        elif got != [source]:
            problems.append(f"{name}: resolved from {got}, not the pinned {source} (patch not applied)")
    for p in packages:
        source = p.get("source", "")
        if source.startswith("git+") and want.get(p.get("name")) != source:
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
