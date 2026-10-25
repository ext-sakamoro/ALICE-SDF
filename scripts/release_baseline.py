#!/usr/bin/env python3
"""The comparison base of a crate whose releases are not published: the commit before its newest
version bump on main.

While publication is frozen, a crate's version still moves inside its repository, and a gate that
compares against the latest published release (crates.io, a release tag) would compare against
something ever older, or nothing. The base here is read from the history alone: the newest commit
on main's first-parent line that changed `[package].version` of the crate is the bump, and its
PARENT is the base (the tree and the version before the bump), so everything since then, the bump
commit included, is what HEAD's version is judged against.

- the bump is looked for from the merge base of HEAD and main: HEAD on a branch is compared with
  what main had, and HEAD being the bump commit itself needs no special case;
- only first parents are followed: a version that moved and moved back on a merged branch is not a
  bump, a merge whose first parent had another version is one;
- `version.workspace = true` is read from the workspace root at the same commit;
- a renamed crate is followed through `base_package` (its name before the rename);
- the crate absent at the parent (or a root commit, which has none) is "introduced", there is no
  older version to compare with, only when it is new: the manifest at the parent does not hold a
  package under another name (a rename without `base_package`), and no Cargo.toml on the first-parent
  line up to the parent ever named the package (a moved manifest). Either of those is an error;
- a manifest that is there but does not parse is an error, never "absent";
- the crate absent at HEAD, or main / HEAD unreadable, is an error: never "no base, so nothing to
  compare".

git only, no network (fetch main first): the same repository gives the same answer.

usage: scripts/release_baseline.py --manifest Cargo.toml --package <name> [--base-package <old name>]
                                   [--main origin/main] [--head HEAD] [--root DIR] --print
  prints `<40-hex sha> <version>`, or `introduced`
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath
from typing import NamedTuple


class BaselineError(Exception):
    """The base cannot be determined."""


class Baseline(NamedTuple):
    sha: str | None        # the parent of the bump commit (None when the crate came with a root commit)
    version: str | None    # the crate's version at `sha` (None when introduced)
    introduced: bool       # the crate is absent at the parent: nothing older to compare with


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    # a hook exports GIT_DIR and friends for the repository being sent: read `root`, not that
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, env=env)


def _show(root: Path, commit: str, path: str) -> str | None:
    r = _git(root, "show", f"{commit}:{path}")
    return r.stdout if r.returncode == 0 else None


def _toml(text: str | None, where: str = "a manifest") -> dict | None:
    """the parsed manifest, None when the file is not there; a file that is there and does not parse is an error
    (reading it as "absent" would make the crate look introduced, and the gate would compare nothing)"""
    if text is None:
        return None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise BaselineError(f"{where} is not TOML: {e}") from None


def version_at(root: Path, commit: str, manifest: str, names: tuple[str, ...]) -> str | None:
    """the version of the crate named one of `names` in `manifest` at `commit`; None when the manifest or
    the crate is not there"""
    doc = _toml(_show(root, commit, manifest), f"{manifest} at {commit[:12]}")
    pkg = (doc or {}).get("package") or {}
    if pkg.get("name") not in names:
        return None
    v = pkg.get("version")
    if isinstance(v, str):
        return v
    if isinstance(v, dict) and v.get("workspace") is True:
        d = PurePosixPath(manifest).parent
        while True:
            ws_path = str(d / "Cargo.toml") if str(d) != "." else "Cargo.toml"
            ws = _toml(_show(root, commit, ws_path), f"{ws_path} at {commit[:12]}")
            wv = (((ws or {}).get("workspace") or {}).get("package") or {}).get("version")
            if isinstance(wv, str):
                return wv
            if str(d) in (".", ""):
                return None
            d = d.parent
    return None


def _rev(root: Path, ref: str, what: str) -> str:
    r = _git(root, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}")
    if r.returncode != 0 or not r.stdout.strip():
        raise BaselineError(f"{what} {ref!r} is not a commit in {root}")
    return r.stdout.strip()


def _version_paths(manifest: str) -> list[str]:
    """the files a commit must touch to change the crate's version: its manifest and every Cargo.toml above it
    (a workspace root holding `[workspace.package].version`)"""
    paths = [manifest]
    d = PurePosixPath(manifest).parent
    while str(d) not in (".", ""):
        d = d.parent
        paths.append("Cargo.toml" if str(d) in (".", "") else str(d / "Cargo.toml"))
    return list(dict.fromkeys(paths))


def _introduced(root: Path, commit: str, manifest: str, names: tuple[str, ...]) -> None:
    """raise unless the crate, absent at `commit`, is new there: the manifest at `commit` holds no package under
    another name, and no Cargo.toml on the first-parent line up to `commit` ever named one of `names`"""
    doc = _toml(_show(root, commit, manifest), f"{manifest} at {commit[:12]}")
    other = ((doc or {}).get("package") or {}).get("name")
    if isinstance(other, str):
        raise BaselineError(f"{manifest} at {commit[:12]} holds the package {other!r}, not {names[0]!r}: a rename "
                            f"needs base_package (the name before it), or it is another crate")
    pattern = "^[[:space:]]*name[[:space:]]*=[[:space:]]*[\"\'](" + "|".join(n.replace(".", "\\.") for n in names) + ")[\"\']"
    hits = _git(root, "log", "--first-parent", "--format=%H", "-G", pattern, commit, "--", "*Cargo.toml")
    if hits.returncode != 0:
        raise BaselineError(f"the history before {commit[:12]} cannot be read: {hits.stderr.strip()}")
    first = hits.stdout.split()
    if first:
        raise BaselineError(f"{names[0]} is absent from {manifest} at {commit[:12]} but an earlier commit "
                            f"({first[0][:12]}) named it in a Cargo.toml: a moved manifest (give the manifest path "
                            f"at that commit) or a rename (base_package), not a new crate")


def bump_baseline(root: Path | str, manifest: str, package: str, *, base_package: str | None = None,
                  main_ref: str = "origin/main", head: str = "HEAD", prefilter: bool = True) -> Baseline:
    """`prefilter` reads only the first-parent commits whose diff against their first parent touches the version
    files (`_version_paths`): only those can change the version; False reads every first-parent commit (the
    definition, kept for the test that the two agree)"""
    root = Path(root)
    head_sha = _rev(root, head, "HEAD")
    if version_at(root, head_sha, manifest, (package,)) is None:
        raise BaselineError(f"{package} is not in {manifest} at {head} ({head_sha[:12]}), so no version bump "
                            f"on {main_ref} can be its base")
    _rev(root, main_ref, "main")
    mb = _git(root, "merge-base", head_sha, main_ref)
    if mb.returncode != 0 or not mb.stdout.strip():
        raise BaselineError(f"{head} and {main_ref} have no common commit in {root}")
    names = (package,) + ((base_package,) if base_package else ())
    start = mb.stdout.strip()
    if version_at(root, start, manifest, names) is None:
        # the crate is not on main at the merge base (it came on HEAD's branch)
        _introduced(root, start, manifest, names)
        return Baseline(start, None, True)
    if prefilter:
        commits = _git(root, "rev-list", "--first-parent", start, "--", *_version_paths(manifest)).stdout.split()
    else:
        commits = _git(root, "rev-list", "--first-parent", start).stdout.split()
    for c in commits:
        v = version_at(root, c, manifest, names)
        if v is None:
            # the crate is not on main at this point of the line (it came on HEAD's branch)
            _introduced(root, c, manifest, names)
            return Baseline(c, None, True)
        parents = _git(root, "rev-list", "--parents", "-n", "1", c).stdout.split()[1:]
        if not parents:
            return Baseline(None, None, True)
        pv = version_at(root, parents[0], manifest, names)
        if pv is None:
            _introduced(root, parents[0], manifest, names)
            return Baseline(parents[0], None, True)
        if pv != v:
            return Baseline(parents[0], pv, False)
    raise BaselineError(f"no commit on {main_ref}'s first-parent line from {head} introduces or bumps {package} "
                        f"in {manifest}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--base-package")
    ap.add_argument("--main", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--print", action="store_true", required=True)
    a = ap.parse_args(argv)
    try:
        b = bump_baseline(a.root, a.manifest, a.package, base_package=a.base_package, main_ref=a.main, head=a.head)
    except BaselineError as e:
        print(f"release_baseline: {e}", file=sys.stderr)
        return 1
    print("introduced" if b.introduced else f"{b.sha} {b.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
