#!/usr/bin/env python3
"""Local-only publish gate: which sibling repos does publishing this crate
break, and whose `cargo publish` does it unblock?

Why this exists
---------------
`cargo semver-checks` (wired into `scripts/preflight.sh` and
`.github/workflows/security-audit.yml`) answers "is this API change breaking?".
It never answers "who breaks", because it only looks at this crate's own rustdoc
JSON. Nothing in the repo looked at the consumers at all.

That gap produced two measured failures:

1. **2026-09-30** — `alice-sdf` was bumped to 4.0.0 and never published, while
   five sibling crates declare `{ version = "4.0", path = "../ALICE-SDF" }`.
   Local builds stayed green forever, because a `path` wins over `version` for
   path resolution. `cargo publish` strips the path and asks the registry, so
   `alice-lol` (0.3.0 already on crates.io) could not publish at all:

       error: failed to select a version for the requirement `alice-sdf = "^4.0"`
       candidate versions found which didn't match: 3.1.1, 3.1.0, 3.0.0, ...

   A `version + path` declaration is therefore *not* a local concern. It is a
   latent publish blocker that no local build, and no CI job that builds from
   the working tree, can see.

2. **2026-09-28** — `alice-sdf` 4.0.0 landed, `ALICE-LOL` was updated to follow
   it, and that follow-up stayed a *local commit*. `text-to-print` depends on
   both through `path`, so the developer's tree was green while CI, which
   clones from `origin`, went red (trap `sibling-bump-local-green-ci-red`).
   A `path` dependency makes the local checkout the source of truth, so the
   push state of every intermediate repo is part of the build contract — and
   `preflight.sh` cannot observe it by construction.

So this gate reads the consumers, not this crate.

Not wired into CI, on purpose
-----------------------------
`preflight.sh` and `.github/workflows/*` must NOT call this script. The sibling
repos do not exist on a GitHub-hosted runner, so every classification would come
back empty and the gate would pass vacuously — exactly the failure mode
`abi_decl_check.py` guards against with its `compared == 0` check. This is a
**local-only gate: run it by hand before `cargo publish`**, from a checkout that
sits next to the consumers.

What it reports
---------------
Each declaration of the target crate found in a sibling manifest is classified
by *how* it is declared:

* `path only`     — no `version` key, so the registry is never consulted and
                    publishing cannot affect it.
* `version only`  — resolved from the registry; publishing reaches it.
* `version+path`  — local builds use the path, `cargo publish` uses the version.
                    The dangerous shape: green locally, blocked on publish.
* `ws:<shape>`    — `workspace = true`, resolved through the workspace root's
                    `[workspace.dependencies]` and then judged on the shape it
                    inherited (`ws:path only`, `ws:version+path`, ...).
* `unresolved`    — inheritance that could not be resolved, or a version
                    requirement this script does not implement. Never silently
                    dropped, because a requirement we cannot read is exactly
                    where a blocker hides.
* `intra-repo`    — a manifest inside the target crate's own checkout
                    (`fuzz/`, `server/`); listed as excluded, not as downstream.

and, for the ones that carry a version requirement, by what publishing the
candidate version does to them:

* `BLOCKED`              — the requirement matches no published version, and the
                           candidate does not fix it. That crate cannot
                           `cargo publish` today and this publish will not help.
                           **This is the only verdict that fails the gate.**
* `UNBLOCKED-BY-PUBLISH`  — matches nothing published, but the candidate matches.
                           Publishing unblocks that crate's own publish.
* `WILL-RECEIVE`          — already satisfiable, and the candidate also matches,
                           so the new version is what a fresh resolve picks.
* `UNAFFECTED`            — satisfiable today, candidate does not match (an old
                           major pin such as `"1.9"`); publishing cannot reach it.

`BLOCKED` and `UNBLOCKED-BY-PUBLISH` are the same situation today (neither can
publish); they are kept apart, with `UNBLOCKED-BY-PUBLISH` taking priority,
because only one of them is a reason not to publish.

`publish = false` is counted apart, never folded in
---------------------------------------------------
A crate whose manifest sets `package.publish = false` never reaches the registry,
so it can be neither blocked from publishing nor unblocked by one. Those
declarations get the suffixed verdicts `UNBLOCKED-BY-PUBLISH (publish = false)`
and `UNSATISFIABLE (publish = false)`, and their own counters; in particular they
**do not fail the gate**, because an opt-out is not a red. The rows stay in the
table rather than being dropped, so the totals remain explainable.

Measured 2026-09-30: `alice-lol-font` and `alice-lol-vision` both set it, so of
the seven declarations that the 4.0.0 candidate satisfies, only five belong to
crates that can actually publish. Folding all seven together overstates the
benefit of publishing by two.

Counting: declarations, crate directories, repositories
------------------------------------------------------
These are three different denominators and the output names all three. One
repository can hold several crate directories (`ALICE-LOL/alice-lol` and
`ALICE-LOL/alice-lol-ui`), and one manifest can declare the crate more than once
(`ALICE-Bamboo` has it in both `[dependencies]` and `[dev-dependencies]`, and
`cargo publish` resolves dev-dependencies too, so both count).

⚠️ Measured 2026-09-30: an independent sweep of the same tree reported 30
repositories where this script reported 38. The entire discrepancy was this
script printing its **crate-directory** count under the label "dependent
repo(s)" — the two sweeps never actually disagreed about the tree. A count whose
unit is not stated is not a count.

Depth, and why the swept depth is in the summary
-----------------------------------------------
The default is 3, not 2. Measured 2026-09-30: `ALICE-Bio-Platform` and
`ALICE-Registry` declare the crate only in `services/core-engine/Cargo.toml`
(depth 3), the standard ALICE-* SaaS template layout, so a depth-2 sweep drops
that whole layer. Raising the default to 3 does not end the problem, it moves it
to depth 4 — so the summary line reports `depth=N`, because what the gate looked
at is part of its answer.

Separately, every repo that declares a `path` dependency is checked against its
remote: `git ls-remote origin refs/heads/main`, **not** the `origin/main`
remote-tracking ref, which is only a snapshot from the last `git fetch` and
happily reports "in sync" for a branch that moved hours ago.

Vacuous-pass guards
-------------------
Absence is never read as success. The run fails when no dependent repo is found,
when the registry lists no published version, when no declaration could be
classified, or when the registry could not be reached at all. Repos that are not
git checkouts, or have no `origin`, are reported as "unverified" rather than
counted as zero unpushed commits.

Portability note
----------------
Every read passes `encoding="utf-8"` explicitly, and manifests are opened in
binary for `tomllib`. `Path.read_text()` without an encoding uses
`locale.getpreferredencoding()` (cp1252 / cp932 on Windows) and these manifests
carry Japanese comments; the gate would then die with `UnicodeDecodeError` on
windows-latest while passing on macOS and Linux (measured 2026-09-30 in
`abi_decl_check.py`, run for 71dc2ee).

Usage
-----
    python3 scripts/downstream_check.py                     # ~ , alice-sdf, depth 3
    python3 scripts/downstream_check.py --root ~ --crate alice-sdf
    python3 scripts/downstream_check.py --candidate 4.0.0
    python3 scripts/downstream_check.py --depth 5           # wider sweep
    python3 scripts/downstream_check.py --self-test         # version logic only
    python3 scripts/downstream_check.py --no-remote         # skip ls-remote

Exit status 0 when nothing is `BLOCKED` and nothing was vacuous, 1 otherwise.
A failure prints the marker DOWNSTREAM-CHECK-FAIL so it is greppable, and the
run always ends with a countable summary line beginning DOWNSTREAM-CHECK:.

Author: Moroya Sakamoto
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "DOWNSTREAM-CHECK-FAIL"
SUMMARY = "DOWNSTREAM-CHECK"
INDEX_BASE = "https://index.crates.io"
HTTP_TIMEOUT = 20
GIT_TIMEOUT = 60
DEFAULT_DEPTH = 3

DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")

# Directories never descended into. Pruning happens *during* descent, not as an
# output filter — see `walk_manifests`.
PRUNE_DIRS = frozenset(
    {
        "target", "node_modules", "vendor", "Library", "__pycache__",
        "DerivedData", "Pods", ".build", ".stack-work", ".gradle",
    }
)

# ---------------------------------------------------------------------------
# semver
#
# Hand-rolled on purpose: the gate must run with the standard library only, and
# `--self-test` pins every rule below against a table so a silent inversion in
# the caret bound (the one mistake that would flip a classification) shows up as
# a named failing case instead of a plausible-looking table.
# ---------------------------------------------------------------------------

Version = tuple[tuple[int, int, int], tuple]


def _pre_ids(pre: str) -> tuple:
    """Prerelease identifiers for semver precedence: numeric before alphanumeric."""
    out = []
    for part in pre.split("."):
        if part.isdigit():
            out.append((0, int(part), ""))
        else:
            out.append((1, 0, part))
    return tuple(out)


def parse_version(text: str) -> Version | None:
    """A full `major.minor.patch[-pre][+build]`. Returns None if unparseable."""
    s = text.strip()
    if not s:
        return None
    s = s.split("+", 1)[0]
    pre = ""
    if "-" in s:
        s, pre = s.split("-", 1)
    parts = s.split(".")
    if len(parts) != 3:
        return None
    try:
        nums = tuple(int(p) for p in parts)
    except ValueError:
        return None
    if any(n < 0 for n in nums):
        return None
    return (nums[0], nums[1], nums[2]), _pre_ids(pre) if pre else ()


def version_key(v: Version):
    """Sort key honouring `1.0.0-alpha < 1.0.0`."""
    nums, pre = v
    return (nums, 0 if pre else 1, pre)


def _partial(text: str) -> tuple[tuple[int, int | None, int | None], tuple] | None:
    """A possibly-partial version as written in a requirement: `4`, `4.0`, `4.0.0`."""
    s = text.strip()
    if not s:
        return None
    s = s.split("+", 1)[0]
    pre = ""
    if "-" in s:
        s, pre = s.split("-", 1)
    parts = s.split(".")
    if not 1 <= len(parts) <= 3:
        return None
    nums: list[int | None] = []
    for p in parts:
        p = p.strip()
        if p in ("*", "x", "X"):
            break
        if not p.isdigit():
            return None
        nums.append(int(p))
    if not nums:
        return None
    while len(nums) < 3:
        nums.append(None)
    return (nums[0], nums[1], nums[2]), _pre_ids(pre) if pre else ()


@dataclass(frozen=True)
class Bound:
    """One comparator as an interval. `None` on either side means unbounded.

    Both ends carry their own inclusivity, because `=1.2.3` (a point) and
    `<=1.2.3` (inclusive above) cannot be written as a half-open interval, and
    quietly widening them by one patch is exactly the kind of off-by-one that
    turns a BLOCKED verdict into a green one.
    """

    lo: Version | None
    lo_inclusive: bool
    hi: Version | None
    hi_inclusive: bool
    # the comparator mentioned a prerelease; only then may a prerelease match,
    # and only on the same major.minor.patch
    pre_base: tuple[int, int, int] | None


def _caret_upper(maj: int, minr: int | None, pat: int | None) -> Version:
    """Increment the leftmost non-zero specified component.

    `^4.0` → <5.0.0, `^0.3` → <0.4.0, `^0.0.3` → <0.0.4, `^0` → <1.0.0,
    `^0.0` → <0.1.0. Getting this backwards is the single mistake that would
    silently reclassify every declaration, which is why `--self-test` pins all
    five shapes.
    """
    if maj > 0 or minr is None:
        return (maj + 1, 0, 0), ()
    if minr > 0 or pat is None:
        return (0, minr + 1, 0), ()
    return (0, 0, pat + 1), ()


def parse_req(req: str) -> list[Bound] | None:
    """A cargo version requirement as a list of comparators (AND).

    Returns None for a requirement this script does not implement, so the caller
    reports it as unresolved instead of guessing.
    """
    text = req.strip()
    if not text:
        return None
    bounds: list[Bound] = []
    for raw in text.split(","):
        c = raw.strip()
        if not c:
            return None
        if c in ("*", "x", "X"):
            bounds.append(Bound(None, True, None, True, None))
            continue
        op = "^"
        for cand in (">=", "<=", "^", "~", "=", ">", "<"):
            if c.startswith(cand):
                op, c = cand, c[len(cand) :].strip()
                break
        part = _partial(c)
        if part is None:
            return None
        (maj, minr, pat), pre = part
        exact: Version = ((maj, minr or 0, pat or 0), pre)
        base = (maj, minr or 0, pat or 0) if pre else None
        if op == "^":
            bounds.append(Bound(exact, True, _caret_upper(maj, minr, pat), False, base))
        elif op == "~":
            hi: Version = ((maj + 1, 0, 0), ()) if minr is None else ((maj, minr + 1, 0), ())
            bounds.append(Bound(exact, True, hi, False, base))
        elif op == "=":
            if minr is None:
                bounds.append(Bound(exact, True, ((maj + 1, 0, 0), ()), False, base))
            elif pat is None:
                bounds.append(Bound(exact, True, ((maj, minr + 1, 0), ()), False, base))
            else:
                bounds.append(Bound(exact, True, exact, True, base))
        elif op == ">=":
            bounds.append(Bound(exact, True, None, True, base))
        elif op == ">":
            bounds.append(Bound(exact, False, None, True, base))
        elif op == "<":
            bounds.append(Bound(None, True, exact, False, base))
        elif op == "<=":
            bounds.append(Bound(None, True, exact, True, base))
        else:  # pragma: no cover - the loop above cannot produce another op
            return None
    return bounds


def _bound_contains(b: Bound, v: Version) -> bool:
    key = version_key(v)
    if b.lo is not None:
        lo = version_key(b.lo)
        if key < lo or (key == lo and not b.lo_inclusive):
            return False
    if b.hi is not None:
        hi = version_key(b.hi)
        if key > hi or (key == hi and not b.hi_inclusive):
            return False
    if v[1]:
        # a prerelease matches only a comparator that mentions one, on the same
        # major.minor.patch (`4.0` must not resolve to `4.1.0-alpha.1`)
        if b.pre_base is None or v[0] != b.pre_base:
            return False
    return True


def req_matches(bounds: list[Bound], v: Version) -> bool:
    return all(_bound_contains(b, v) for b in bounds)


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------


@dataclass
class Decl:
    manifest: Path
    repo: Path
    table: str  # e.g. `dependencies`, `target.cfg(unix).dependencies`
    shape: str  # path only / version only / version+path / workspace / unresolved
    req: str | None
    has_path: bool
    note: str = ""
    verdict: str = ""
    never_publishes: bool = False


@dataclass
class RepoState:
    root: Path
    head: str = ""
    remote: str = ""
    status: str = ""  # in-sync / unpushed / behind / diverged / unverified
    dirty: int = 0
    detail: str = ""
    decls: list[Decl] = field(default_factory=list)


def load_manifest(path: Path) -> dict | None:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def iter_dep_tables(doc: dict):
    """Yield `(label, table)` for every dependency table in a manifest.

    Covers `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`,
    `[target.*.<kind>-dependencies]` and `[workspace.dependencies]`. A regex over
    the raw text would miss the `[dependencies.alice-sdf]` sub-table form that
    `fuzz/Cargo.toml` uses, and would match the commented-out declaration in
    `ALICE-VCS/Cargo.toml`; `tomllib` gets both right.
    """
    for name in DEP_TABLES:
        t = doc.get(name)
        if isinstance(t, dict):
            yield name, t
    targets = doc.get("target")
    if isinstance(targets, dict):
        for cfg, spec in targets.items():
            if not isinstance(spec, dict):
                continue
            for name in DEP_TABLES:
                t = spec.get(name)
                if isinstance(t, dict):
                    yield f"target.{cfg}.{name}", t
    ws = doc.get("workspace")
    if isinstance(ws, dict):
        t = ws.get("dependencies")
        if isinstance(t, dict):
            yield "workspace.dependencies", t


def find_workspace_root(manifest: Path, doc: dict, ceiling: Path) -> Path | None:
    """The manifest that owns `[workspace.dependencies]` for this member."""
    pkg = doc.get("package")
    if isinstance(pkg, dict) and isinstance(pkg.get("workspace"), str):
        cand = (manifest.parent / pkg["workspace"]).resolve() / "Cargo.toml"
        return cand if cand.is_file() else None
    if isinstance(doc.get("workspace"), dict):
        return manifest
    here = manifest.parent.resolve()
    ceiling = ceiling.resolve()
    for _ in range(8):
        parent = here.parent
        if parent == here:
            return None
        cand = parent / "Cargo.toml"
        if cand.is_file():
            sub = load_manifest(cand)
            if sub is not None and isinstance(sub.get("workspace"), dict):
                return cand
        if parent == ceiling:
            return None
        here = parent
    return None


def classify(spec, manifest: Path, doc: dict, crate: str, ceiling: Path) -> tuple[str, str | None, bool, str]:
    """`(shape, requirement, has_path, note)` for one dependency declaration.

    A `workspace = true` entry is resolved to the shape of the inherited entry
    and prefixed `ws:`, so it is judged on the requirement it actually carries.
    Reporting inheritance itself as a shape would put every `ws:path only` entry
    in the unresolved bucket — six of them here — and "unresolved" has to mean
    "this gate could not read the requirement", not "the requirement arrived by
    a different route".
    """
    if isinstance(spec, str):
        return "version only", spec, False, ""
    if not isinstance(spec, dict):
        return "unresolved", None, False, f"declaration is a {type(spec).__name__}, not a string or table"
    if spec.get("workspace") is True:
        root = find_workspace_root(manifest, doc, ceiling)
        if root is None:
            return "unresolved", None, False, "workspace = true but no workspace root with [workspace.dependencies] was found"
        rdoc = load_manifest(root)
        if rdoc is None:
            return "unresolved", None, False, f"workspace root {root} could not be parsed"
        inherited = (rdoc.get("workspace") or {}).get("dependencies", {}).get(crate)
        if inherited is None:
            return "unresolved", None, False, f"workspace = true but {root} declares no {crate} in [workspace.dependencies]"
        shape, req, has_path, note = classify(inherited, root, rdoc, crate, ceiling)
        if shape == "unresolved":
            return shape, req, has_path, f"via {root}: {note}"
        return f"ws:{shape}", req, has_path, f"inherited from {root}"
    req = spec.get("version")
    has_path = "path" in spec
    if req is None:
        if has_path:
            return "path only", None, True, ""
        if spec.get("git") or spec.get("registry") or spec.get("registry-index"):
            return "unresolved", None, False, "git / alternate-registry dependency, outside this gate"
        return "unresolved", None, False, "neither version nor path"
    if not isinstance(req, str):
        return "unresolved", None, has_path, f"version is a {type(req).__name__}"
    return ("version+path" if has_path else "version only"), req, has_path, ""


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def index_path(name: str) -> str:
    """crates.io index prefix. The rule changes with the name's length, so a
    4+-character name like `alice-sdf` lives at `al/ic/alice-sdf` and a
    three-character one at `3/f/foo`."""
    n = name.lower()
    if len(n) == 1:
        return f"1/{n}"
    if len(n) == 2:
        return f"2/{n}"
    if len(n) == 3:
        return f"3/{n[0]}/{n}"
    return f"{n[0:2]}/{n[2:4]}/{n}"


def fetch_published(name: str, base: str) -> tuple[list[Version], list[str], str]:
    """`(versions, raw_strings, error)`. One request, no retries."""
    url = f"{base.rstrip('/')}/{index_path(name)}"
    req = urllib.request.Request(url, headers={"User-Agent": "alice-downstream-check"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        return [], [], f"{url}: {exc}"
    versions: list[Version] = []
    raw: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            return [], [], f"{url}: index line is not JSON ({exc})"
        if rec.get("yanked"):
            continue
        v = parse_version(str(rec.get("vers", "")))
        if v is None:
            continue
        versions.append(v)
        raw.append(str(rec["vers"]))
    return versions, raw, ""


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def _git(args: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=GIT_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return p.returncode, (p.stdout or p.stderr).strip()


def probe_repo(root: Path, check_remote: bool, scan_root: Path) -> RepoState:
    """HEAD vs the remote `main` **as the remote reports it now**.

    `origin/main` is a remote-tracking ref written by the last `git fetch`, so it
    describes the past; `git ls-remote` asks the server.

    `rev-parse --show-toplevel` succeeding does **not** mean this directory is a
    checkout: it means *some* ancestor is one. Measured 2026-09-30: `ALICE-Kintsugi`
    and `Project-ALICE` have no repository of their own, so the command walked up
    to the scan root (`~`, itself a git repo with 984 dirty files) and an earlier
    draft of this gate collapsed both into one row and called them `in-sync` —
    a false green for two repos whose sources no repository tracks. A toplevel at
    or above the scan root is therefore reported as unverified.
    """
    st = RepoState(root=root)
    rc, top = _git(["git", "-C", str(root), "rev-parse", "--show-toplevel"])
    if rc != 0:
        st.status, st.detail = "unverified", "not a git checkout"
        return st
    top_path = Path(top).resolve()
    # `top_path in scan_root.parents` (not the reverse): the toplevel must be a
    # descendant of the scan root. A toplevel *at* the root, or above it, means
    # this directory only inherits an outer repository.
    if top_path == scan_root or top_path in scan_root.parents:
        st.status = "unverified"
        st.detail = f"no checkout of its own (nearest is {top_path}, at or above the scan root)"
        return st
    st.root = top_path
    rc, head = _git(["git", "-C", str(root), "rev-parse", "HEAD"])
    if rc != 0:
        st.status, st.detail = "unverified", "no HEAD (empty repository?)"
        return st
    st.head = head
    rc, porcelain = _git(["git", "-C", str(root), "status", "--porcelain"])
    if rc == 0:
        st.dirty = len([ln for ln in porcelain.splitlines() if ln.strip()])
    if not check_remote:
        st.status, st.detail = "unverified", "--no-remote"
        return st
    rc, remotes = _git(["git", "-C", str(root), "remote"])
    if rc != 0 or "origin" not in remotes.split():
        st.status, st.detail = "unverified", "no origin remote"
        return st
    rc, out = _git(["git", "-C", str(root), "ls-remote", "origin", "refs/heads/main"])
    if rc != 0:
        st.status, st.detail = "unverified", f"ls-remote failed: {out.splitlines()[:1]}"
        return st
    if not out:
        st.status, st.detail = "unverified", "origin has no refs/heads/main"
        return st
    st.remote = out.split()[0]
    if st.remote == st.head:
        st.status = "in-sync"
        return st
    have_remote = _git(["git", "-C", str(root), "cat-file", "-e", st.remote + "^{commit}"])[0] == 0
    if not have_remote:
        st.status = "unpushed"
        st.detail = "remote commit is not in the local object store (stale fetch)"
        return st
    head_in_remote = _git(["git", "-C", str(root), "merge-base", "--is-ancestor", st.head, st.remote])[0] == 0
    remote_in_head = _git(["git", "-C", str(root), "merge-base", "--is-ancestor", st.remote, st.head])[0] == 0
    if head_in_remote:
        rc, n = _git(["git", "-C", str(root), "rev-list", "--count", f"{st.head}..{st.remote}"])
        st.status, st.detail = "behind", f"{n} commit behind origin/main"
    elif remote_in_head:
        rc, n = _git(["git", "-C", str(root), "rev-list", "--count", f"{st.remote}..{st.head}"])
        st.status, st.detail = "unpushed", f"{n} commit not on origin/main"
    else:
        st.status, st.detail = "diverged", "HEAD and origin/main have no ancestry relation"
    return st


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

SELF_TEST = [
    # (requirement, version, expected)
    ("4.0", "4.0.0", True),
    ("4.0", "4.1.7", True),
    ("4.0", "3.1.1", False),
    ("4.0", "5.0.0", False),
    ("4.0.0", "4.0.0", True),
    ("4.0.0", "4.2.0", True),
    ("4.0.0", "3.1.1", False),
    ("^4.0", "4.0.0", True),
    ("1.9", "1.9.2", True),
    ("1.9", "1.10.3", True),
    ("1.9", "4.0.0", False),
    ("1.9.2", "1.9.1", False),
    ("1.9.2", "1.9.2", True),
    ("2", "2.0.0", True),
    ("2", "2.9.9", True),
    ("2", "4.0.0", False),
    ("2", "1.9.9", False),
    ("3.1", "3.1.1", True),
    ("3.1", "4.0.0", False),
    ("0.3", "0.3.9", True),
    ("0.3", "0.4.0", False),
    ("0.0.3", "0.0.3", True),
    ("0.0.3", "0.0.4", False),
    ("0", "0.9.9", True),
    ("0", "1.0.0", False),
    ("0.0", "0.0.9", True),
    ("0.0", "0.1.0", False),
    ("~1.2.3", "1.2.9", True),
    ("~1.2.3", "1.3.0", False),
    ("~1.2", "1.2.9", True),
    ("~1.2", "1.3.0", False),
    ("~1", "1.9.9", True),
    ("~1", "2.0.0", False),
    ("=1.2.3", "1.2.3", True),
    ("=1.2.3", "1.2.4", False),
    (">=1.2.3", "9.9.9", True),
    (">=1.2.3", "1.2.2", False),
    ("<2.0.0", "1.9.9", True),
    ("<2.0.0", "2.0.0", False),
    ("<=2.0.0", "2.0.0", True),
    ("<=2.0.0", "2.0.1", False),
    (">=1.0, <1.5", "1.4.0", True),
    (">=1.0, <1.5", "1.5.0", False),
    ("*", "0.0.1", True),
    ("1.*", "1.7.0", True),
    ("1.*", "2.0.0", False),
    # a prerelease needs a requirement that mentions one
    ("4.0", "4.1.0-alpha.1", False),
    ("4.0.0-alpha", "4.0.0-alpha.1", True),
]


def self_test() -> int:
    bad = []
    for req, ver, want in SELF_TEST:
        bounds = parse_req(req)
        v = parse_version(ver)
        if bounds is None or v is None:
            bad.append(f"  {req!r} vs {ver!r}: unparseable (bounds={bounds}, version={v})")
            continue
        got = req_matches(bounds, v)
        if got != want:
            bad.append(f"  {req!r} vs {ver!r}: got {got}, want {want}")
    print(f"downstream_check --self-test: {len(SELF_TEST)} cases, {len(bad)} failed")
    if bad:
        print(f"{MARKER}: the version-requirement logic is wrong.")
        print("\n".join(bad))
        return 1
    if not SELF_TEST:
        print(f"{MARKER}: the self-test table is empty — nothing was verified")
        return 1
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def own_version(crate: str) -> tuple[str | None, str]:
    manifest = Path(__file__).resolve().parent.parent / "Cargo.toml"
    doc = load_manifest(manifest)
    if doc is None:
        return None, f"{manifest} could not be parsed"
    pkg = doc.get("package") or {}
    if pkg.get("name") != crate:
        return None, f"{manifest} declares package.name = {pkg.get('name')!r}, not {crate!r}"
    v = pkg.get("version")
    if not isinstance(v, str):
        return None, f"{manifest} has no string package.version"
    return v, ""


def walk_manifests(root: Path, depth: int) -> tuple[list[Path], int]:
    """Every `Cargo.toml` whose directory is 1..`depth` components below `root`.

    `(manifests, pruned_directory_count)`.

    ⚠️ Pruning must happen **during descent**. Both `Path.glob("*/*/…/Cargo.toml")`
    and `find … -not -path '*/target/*'` filter the *output* while still walking
    into every directory, so on this machine — where `target/` holds 335 GB — a
    depth-7 sweep does not finish. Measured 2026-09-30: an earlier glob-based
    version of this function was killed at 120 s on `--depth 7`, while this walk
    returns in about a second. The pruned set is reported in the output, because
    a directory this gate never looked into is not a directory it cleared.
    """
    manifests: list[Path] = []
    pruned = 0
    frontier: list[tuple[Path, int]] = [(root, 0)]
    while frontier:
        here, level = frontier.pop()
        try:
            entries = list(os.scandir(here))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_file(follow_symlinks=False):
                    if e.name == "Cargo.toml" and 1 <= level <= depth:
                        manifests.append(Path(e.path))
                elif e.is_dir(follow_symlinks=False) and level + 1 <= depth:
                    if e.name in PRUNE_DIRS or e.name.startswith("."):
                        pruned += 1
                        continue
                    frontier.append((Path(e.path), level + 1))
            except OSError:
                continue
    return sorted(set(manifests)), pruned


def scan(root: Path, crate: str, depth: int) -> tuple[list[Decl], list[Path], list[str], int]:
    """`(declarations, target_checkouts, skipped, pruned)` down to `depth`.

    ⚠️ `depth = 2` does **not** see every consumer, which is why the default is 3.
    Measured 2026-09-30: `ALICE-Bio-Platform` and `ALICE-Registry` declare the
    crate in `services/core-engine/Cargo.toml` — depth 3 — because
    `services/<name>/` is the standard layout of the ALICE-* SaaS template. Both
    are `path only` today, so the publish verdict did not change; but a
    `version+path` placed there would have been invisible, which is the exact
    shape this gate exists to catch. Raising the default from 2 to 3 does not end
    the problem, it moves it to depth 4, so **the summary line reports the depth
    that was actually swept**: what the gate looked at is part of its answer.
    """
    manifests, pruned = walk_manifests(root, depth)
    decls: list[Decl] = []
    targets: list[Path] = []
    skipped: list[str] = []
    docs: dict[Path, dict] = {}
    for m in manifests:
        if "target" in m.parts[len(root.parts) :][:-1]:
            continue  # a build directory, not a crate
        doc = load_manifest(m)
        if doc is None:
            skipped.append(f"{m}: manifest could not be parsed")
            continue
        docs[m] = doc
        if (doc.get("package") or {}).get("name") == crate:
            targets.append(m.parent.resolve())
    for m, doc in docs.items():
        for label, table in iter_dep_tables(doc):
            spec = table.get(crate)
            if spec is None:
                continue
            shape, req, has_path, note = classify(spec, m, doc, crate, root)
            # `publish = false` means this crate never goes to the registry, so a
            # version requirement it carries can never block a `cargo publish`.
            # Measured 2026-09-30: `alice-lol-font` and `alice-lol-vision` both set
            # it, so counting them as "unblocked by this publish" overstates the
            # benefit by two of seven declarations.
            never = (doc.get("package") or {}).get("publish") is False
            if never and req is not None:
                # only worth saying where a version requirement exists: on a
                # `path only` row the registry is irrelevant either way
                note = (note + "; " if note else "") + "package.publish = false (never reaches the registry)"
            repo_root = m.parent
            for t in targets:
                try:
                    m.resolve().relative_to(t)
                except ValueError:
                    continue
                note = (note + "; " if note else "") + f"inside the {crate} checkout at {t}"
                shape = "intra-repo"
                break
            decls.append(
                Decl(
                    manifest=m, repo=repo_root, table=label, shape=shape, req=req,
                    has_path=has_path, note=note, never_publishes=never,
                )
            )
    return decls, targets, skipped, pruned


def main() -> int:
    ap = argparse.ArgumentParser(description="local publish gate: downstream impact of publishing this crate")
    ap.add_argument("--root", default="~", help="scan root (default: ~)")
    ap.add_argument(
        "--depth", type=int, default=DEFAULT_DEPTH,
        help=f"how many directory levels below the root to sweep (default: {DEFAULT_DEPTH}). "
             "⚠️ depth 2 misses the ALICE-* SaaS layout `services/<name>/Cargo.toml`; "
             "the depth actually swept is reported in the summary line",
    )
    ap.add_argument("--crate", default="alice-sdf", help="crate whose consumers are swept (default: alice-sdf)")
    ap.add_argument("--candidate", default=None, help="version about to be published (default: this repo's package.version)")
    ap.add_argument("--index-url", default=INDEX_BASE, help=f"crates.io index base (default: {INDEX_BASE})")
    ap.add_argument("--no-remote", action="store_true", help="skip git ls-remote (offline); push state is then reported as unverified")
    ap.add_argument("--self-test", action="store_true", help="verify the version-requirement logic and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rc_self = self_test()
    if rc_self != 0:
        return rc_self

    root = Path(args.root).expanduser().resolve()
    crate = args.crate
    if args.depth < 1:
        print(f"{MARKER}: --depth must be at least 1 (got {args.depth})")
        return 1
    if not root.is_dir():
        # same field set as the normal summary, so one grep covers every exit path
        print(
            f"{SUMMARY}: blocked=0 blocked_publish_false=0 unblocked=0 "
            f"unblocked_publish_false=0 will_receive=0 unaffected=0 unresolved=0 "
            f"unpushed=0 depth={args.depth}"
        )
        print(f"{MARKER}: scan root {root} is not a directory")
        return 1

    candidate_str = args.candidate
    if candidate_str is None:
        candidate_str, err = own_version(crate)
        if candidate_str is None:
            print(f"{MARKER}: no candidate version — {err} (pass --candidate)")
            return 1
    candidate = parse_version(candidate_str)
    if candidate is None:
        print(f"{MARKER}: --candidate {candidate_str!r} is not a semver version")
        return 1

    decls, targets, skipped, pruned = scan(root, crate, args.depth)
    downstream = [d for d in decls if d.shape != "intra-repo"]
    intra = [d for d in decls if d.shape == "intra-repo"]
    # Two different denominators, kept apart on purpose. An earlier version
    # printed the crate-directory count under the label "dependent repo(s)",
    # which read as 38 repos where a per-repository count gives 30 — the whole
    # 8-repo discrepancy against the independent sweep was this one label.
    # `ALICE-LOL/alice-lol` and `ALICE-LOL/alice-lol-ui` are two crate
    # directories inside one repository.
    crate_dirs = sorted({d.repo for d in downstream}, key=str)
    repos = sorted({d.manifest.relative_to(root).parts[0] for d in downstream})

    published, published_raw, net_err = fetch_published(crate, args.index_url)
    published.sort(key=version_key)

    # ---- push state of every repo that declares a `path` dependency ----
    path_repos = sorted({d.repo for d in downstream if d.has_path}, key=str)
    states: dict[Path, RepoState] = {}
    if path_repos:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futs = {pool.submit(probe_repo, p, not args.no_remote, root): p for p in path_repos}
            for fut in concurrent.futures.as_completed(futs):
                states[futs[fut]] = fut.result()
    # several manifests can share one git checkout; collapse by toplevel
    by_top: dict[Path, RepoState] = {}
    for p in path_repos:
        st = states[p]
        key = st.root if st.status != "unverified" or st.head else p
        prev = by_top.get(key)
        if prev is None:
            by_top[key] = st
        elif prev.status == "unverified" and st.status != "unverified":
            by_top[key] = st

    # ---- verdicts ----
    counts = {
        "blocked": 0, "blocked_publish_false": 0,
        "unblocked": 0, "unblocked_publish_false": 0,
        "will_receive": 0, "unaffected": 0, "unresolved": 0,
    }
    for d in downstream:
        if d.shape.endswith("path only"):
            d.verdict = "PATH-ONLY"
            continue
        if d.shape == "unresolved" or d.req is None:
            d.verdict = "UNRESOLVED"
            counts["unresolved"] += 1
            continue
        bounds = parse_req(d.req)
        if bounds is None:
            d.verdict = "UNRESOLVED"
            d.note = (d.note + "; " if d.note else "") + f"requirement {d.req!r} is not implemented by this gate"
            counts["unresolved"] += 1
            continue
        now = any(req_matches(bounds, v) for v in published)
        fut_ = req_matches(bounds, candidate)
        if not now and fut_:
            # A crate that never reaches the registry cannot be unblocked by a
            # publish, so it is counted apart rather than folded in: this gate
            # answers "who cannot publish", and a crate that opted out of
            # publishing is not an answer to that question. The row stays
            # visible, because dropping it silently would leave the totals
            # unexplainable.
            if d.never_publishes:
                d.verdict = "UNBLOCKED-BY-PUBLISH (publish = false)"
                counts["unblocked_publish_false"] += 1
            else:
                d.verdict = "UNBLOCKED-BY-PUBLISH"
                counts["unblocked"] += 1
        elif not now and not fut_:
            # Same category split as above, and for the same reason: a
            # `publish = false` crate is not blocked *from publishing*, so it
            # must not fail the gate. Folding it into BLOCKED would turn an
            # opt-out into a red.
            if d.never_publishes:
                d.verdict = "UNSATISFIABLE (publish = false)"
                counts["blocked_publish_false"] += 1
            else:
                d.verdict = "BLOCKED"
                counts["blocked"] += 1
        elif now and fut_:
            d.verdict = "WILL-RECEIVE"
            counts["will_receive"] += 1
        else:
            d.verdict = "UNAFFECTED"
            counts["unaffected"] += 1

    classified = sum(1 for d in downstream if d.verdict and d.verdict != "UNRESOLVED")
    unpushed = [s for s in by_top.values() if s.status in ("unpushed", "diverged")]
    unverified = [s for s in by_top.values() if s.status == "unverified"]

    # ---- report ----
    print(f"downstream_check: crate={crate} candidate={candidate_str} root={root}")
    if published_raw:
        newest = sorted(published, key=version_key)[-1]
        print(
            f"  registry: {len(published_raw)} published version(s), newest "
            f"{'.'.join(str(x) for x in newest[0])}{'-' + '.'.join(i[2] or str(i[1]) for i in newest[1]) if newest[1] else ''}"
            f"{'  (candidate is NOT yet published)' if not any(v == candidate for v in published) else '  (candidate already published)'}"
        )
    print(
        f"  swept: depth <= {args.depth} below the root, {pruned} directory subtree(s) pruned "
        f"({', '.join(sorted(PRUNE_DIRS))}, and dotted directories)"
    )
    print(
        f"  found: {len(decls)} declaration(s) in {len(crate_dirs)} crate directory(ies), "
        f"belonging to {len(repos)} repository(ies)"
        f"{f'; {len(intra)} declaration(s) inside the {crate} checkout itself (excluded)' if intra else ''}"
    )
    never = [d for d in downstream if d.never_publishes and d.req is not None]
    if never:
        print(
            f"  of the version-bearing declarations, {len(never)} belong to a crate with "
            f"`publish = false`, counted separately: "
            + ", ".join(sorted({d.manifest.parent.name for d in never}))
        )
    print()

    order = [
        "BLOCKED",
        "UNBLOCKED-BY-PUBLISH",
        "UNBLOCKED-BY-PUBLISH (publish = false)",
        "UNSATISFIABLE (publish = false)",
        "WILL-RECEIVE",
        "UNAFFECTED",
        "UNRESOLVED",
        "PATH-ONLY",
    ]
    width = max((len(str(d.manifest.relative_to(root))) for d in downstream), default=10)
    for verdict in order:
        rows = [d for d in downstream if d.verdict == verdict]
        if not rows:
            continue
        print(f"{verdict}  ({len(rows)})")
        for d in sorted(rows, key=lambda x: str(x.manifest)):
            rel = str(d.manifest.relative_to(root))
            req = d.req if d.req is not None else "-"
            print(f"  {rel:<{width}}  {d.table:<24} {d.shape:<16} req={req:<10}")
            if d.note:
                print(f"  {'':<{width}}  └ {d.note}")
        print()

    print(f"path-dependency repos, HEAD vs `git ls-remote origin refs/heads/main`  ({len(by_top)})")
    for st in sorted(by_top.values(), key=lambda s: str(s.root)):
        rel = str(st.root) if not str(st.root).startswith(str(root)) else str(st.root.relative_to(root))
        extra = f"  {st.detail}" if st.detail else ""
        dirty = f"  dirty={st.dirty}" if st.dirty else ""
        print(f"  {rel:<{width}}  {st.status:<10} HEAD={st.head[:8] or '-':<8} remote={st.remote[:8] or '-':<8}{dirty}{extra}")
    print()

    problems: list[str] = []
    if net_err:
        problems.append(f"the registry could not be read, so no requirement was checked: {net_err}")
    if not published:
        problems.append(f"no published version of {crate} was found — the registry lookup produced nothing to compare against")
    if not repos:
        problems.append(
            f"no dependent repo found under {root} at depth <= {args.depth} — "
            f"the scan is broken or the root is wrong"
        )
    if classified == 0:
        problems.append("no declaration could be classified — nothing was compared")
    if counts["blocked"]:
        problems.append(f"{counts['blocked']} declaration(s) match no published version and the candidate does not fix it")
    if skipped:
        for s in skipped:
            print(f"  note: skipped {s}")

    if unverified:
        print(f"unverified push state ({len(unverified)} repo(s)) — NOT counted as zero:")
        for st in unverified:
            print(f"  {st.root}: {st.detail}")
        print()
    if unpushed:
        print(f"⚠️  {len(unpushed)} repo(s) carry commits that `origin/main` does not have. A `path`")
        print("   dependency makes the local tree the source of truth, so CI (which clones")
        print("   from origin) sees a different crate than this working copy does.")
        print()

    # `depth` is part of the summary because the answer is only as wide as the
    # sweep: a reader who cannot see the depth cannot tell a clean result from an
    # unlooked-at one.
    print(
        f"{SUMMARY}: blocked={counts['blocked']} "
        f"blocked_publish_false={counts['blocked_publish_false']} "
        f"unblocked={counts['unblocked']} "
        f"unblocked_publish_false={counts['unblocked_publish_false']} "
        f"will_receive={counts['will_receive']} unaffected={counts['unaffected']} "
        f"unresolved={counts['unresolved']} unpushed={len(unpushed)} depth={args.depth}"
    )
    if problems:
        print(f"{MARKER}:")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
