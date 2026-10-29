#!/usr/bin/env python3
"""An identifier moves only with a breaking version and naming both sides (ALICE-Law G9).

The identifiers of scripts/id_move.toml (`SEMANTICS_ID`, `*_SEMANTICS_ID`: constants a
stored value is stamped with) are read as values, not as text: most are folded by a
`const fn`, so the source can stay the same while the value moves (a resolved upstream
crate changed) and change while the value stays. For the base ref and for the working
tree, a probe binary that depends on the crate by path prints each identifier and each
previous-identifier list; the two are then compared.

The base is the newest release tag reachable from HEAD that matches a pattern of the
config (`v{version}`, `<crate>-v{version}`, ...), so an identifier moved twice between two
releases counts once, against what was published. `--base <ref>` overrides it.

For each identifier that exists at the base and has another value now:

- when the base's value is in the identifier's previous list, the move claims the old bits
  are reproduced (not a change of meaning) and needs no version bump, but the claim is tested
  by behaviour: the [[id]] names `previous-tests` (`<path>.rs::<fn>`, each a `#[test]` fn, not
  `#[ignore]`d), and the tool runs each of them twice in a copy of the working tree next to it:
  as it is (it must pass, one test run) and with the old value taken out of the previous list's
  definition (located by its rustdoc span; the value written as `hex32("…")` / a hex string, as
  32 byte literals, or as `[0xNN; 32]`) (it must fail). A test that passes without the old value
  does not depend on it, and naming none fails, so the list cannot be filled to dodge a bump.
  A previous list this crate re-exports from another crate (`pub use upstream::PREVIOUS_…`) is
  followed to its definition when that crate is a sibling repository pinned in
  .github/alice-siblings (`<Repo> <40-hex SHA>`): the sibling must be at the pinned commit and
  clean, the list is located by the sibling's rustdoc span, and the value is taken out of a
  copy of the sibling laid out next to a copy of this crate (the real sibling is never
  changed). A re-export that does not resolve that way fails, with the reason;
- otherwise (not in the list, or no list) the move is a change of meaning and it must be
  announced to the consumers: the version is a breaking step from the base's (a major for 1.x
  and later, a minor or a major for 0.x), or, while publication is frozen and the version does
  not move, the `[Unreleased]` section of CHANGELOG.md has a `**Breaking:**` entry (a list item
  that starts with it) naming both values (their first 8 hex digits) in that one entry;
- in both cases CHANGELOG.md must name both values (their first 8 hex digits).

An identifier missing at the base is new and passes (the probe's path does not resolve there, or, for
an associated const `Type::NAME`, rustc reports exactly that item missing on exactly that type and
nothing else). Comparing no identifier fails, and
so does a config naming a constant the working tree does not have.

Completeness: every `const` / `static` item of the crate whose type resolves to `[u8; 32]`
(the form every identifier of these crates takes; any visibility, through `&`, a type alias,
a length written as `0x20`, `32usize` or a constant) must be an [[id]] of the config or be
listed under [exempt] with a reason, so a second identifier (a format id next to
SEMANTICS_ID) cannot be left out. The items come from rustdoc's JSON output (RUSTC_BOOTSTRAP=1,
--document-private-items, format_version 60, as scripts/api_boundary.py reads it), where the
compiler has resolved the types: a text scan misses a type on the next line or behind an
alias. A public re-export of another crate's constant or static counts too (rustdoc does not
give its type), so a crate that stamps an upstream identifier lists it; a public glob re-export of
another crate (`pub use other::*`) is listed by its source (`other::*`) under [exempt] with a reason. Associated consts count too (an inherent impl's, a trait's and a trait impl's), named
`Type::NAME` / `Trait::NAME`. rustdoc runs with every feature (`--all-features`, or the
config's `doc-features` when a feature cannot build here), so an identifier behind a feature
is seen; the values are still read with the config's `features`. An [exempt] entry that names
no such item fails, and finding none fails as having compared nothing. A `const` inside a
function body is not an item rustdoc lists, and an item behind a target cfg (`target_os`) is
seen only on that target.

The base's identifiers are read from the version as published (crates.io, `={version}`),
which is what a stored value was stamped by; a version never published is read from the tag's
tree instead (`--base-from tag` forces that, `--base-from registry` refuses the fallback).

While publication is frozen, the version moves on main without a tag or a release: the base is
then the commit before the newest version bump on main's first-parent line (scripts/
release_baseline.py, a file shared with the other ALICE crates), read from its tree, and its
version is the one HEAD's is judged against (`--base-from bump`). `auto` judges against both
when the tag is an ancestor of the commit before the bump (the tag is what stored data in the
wild was stamped by, the bump base the version line main moved on since), and a problem against
either fails, named with the base it came from; on a tie, or when the tag is newer, the tag
alone. Against a bump base every move must be named in CHANGELOG.md's `[Unreleased]`
section or the section of HEAD's version (the release the move goes out in), not anywhere in
the file; a crate absent at the commit before its first version on main has no base, and every
identifier is new. `--main <ref>` names main (default `origin/main`; fetch it first).

A registry that cannot be read fails the run on the default branch, on tags and locally, and
is a warning on ci/ and wip/ branches and pull requests (an outage does not turn every branch
red; the default branch still does not pass a check it did not run).

`--print-siblings` prints, one "<Repo> <40-hex SHA>" line each, every sibling commit a base tree
the check may build can need: every line of each base's .github/alice-siblings (the check refuses a
sibling a base reaches by path without a pin there, so what it reaches, at any depth, is among
them), and nothing for a base that reaches no sibling. It reads only this repository's history, so
CI runs it before any sibling is cloned and gives the file to the alice-siblings action as `list:`,
which clones the repositories with their full history and makes each commit present. It fails
on a base whose tree path-depends on a sibling it does not pin and on a broken base (the pin file is
in its first-parent history but not in its tree). A base from before the pin file takes each sibling
it reaches, and the ones those reach in turn, at the newest commit of the sibling's main (first
parent) whose committer date is not later than the base commit's, as the check does: the printer
fetches each such sibling's main read-only over git into a temporary directory (from
$ALICE_SIBLINGS_GIT_BASE/<Repo>, default https://github.com/ext-sakamoro, the action's source; no
REST call, every fetch bounded by a timeout) and prints the commit it resolved. It fails, naming
the repository, on one that .github/alice-siblings-private lists (it needs a key the printer does
not have), one that cannot be fetched, and one with no commit of its main by that date.

usage: scripts/id_move_check.py [--root DIR] [--base REF] [--base-from auto|registry|tag|bump] [--main REF]
                                 [--print-base | --print-siblings]
config: scripts/id_move.toml
    package = "alice-zip"                 # the cargo package
    manifest = "Cargo.toml"               # its manifest, relative to the root
    features = ["std"]                    # features the probe builds it with (optional)
    default-features = true               # (optional)
    tags = ["v{version}"]                 # release tag patterns
    doc-features = ["std", "lzma"]        # the features rustdoc (completeness) and the previous-tests
                                          # are built with (default: all; for crates whose features
                                          # cannot all be on at once)
    probe-dependencies = ["alice-det-math"]   # crates a path below names besides the package (the
                                          # probe takes the version the package resolves)
    base-package = "alice-zip"            # (optional) the package name the release tags and the
                                          # registry carry, when the package was renamed since; the
                                          # probe names both under the current lib name, so a path
                                          # below resolves in the base and in the working tree
    [exempt]                              # [u8; 32] constants that are not identifiers
    SOME_KEY = "why it is not an identifier"
    [[id]]
    path = "alice_zip::law::SEMANTICS_ID"
    previous = "alice_zip::semantics::PREVIOUS_SEMANTICS_IDS"   # optional
    previous-tests = ["tests/golden.rs::previous_ids_reproduce"]  # optional, see above
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

# the base of a crate whose version moves on main without a release (a file shared with the other
# ALICE crates, next to this one)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_baseline  # noqa: E402

CONFIG = "scripts/id_move.toml"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")
MISSING_RE = re.compile(r"error\[E0(?:425|432|433)\]|cannot find|unresolved import|could not find")
# an associated const the base's type does not have: rustc names the item and the type
ASSOC_MISSING_RE = re.compile(
    r"^error\[E0599\]: no (?:associated item|associated function or constant|function or associated item|"
    r"variant or associated item|variant, associated function, or constant) named `([^`]+)` found for "
    r"(?:struct|enum|union|type|trait) `([^`]+)`")
ERROR_RE = re.compile(r"^error\[E\d+\]")


def probe_says_absent(path: str, stderr: str) -> bool:
    """True when a failed probe of `path` means the base does not have the identifier (it is new):
    the path does not resolve (a segment of it named in a resolution error), or, for an associated
    const `…::Type::NAME`, every compile error is rustc's E0599 for exactly that item of exactly
    that type (both parsed from the message, the type without its path and generics). Any other
    failure is an error, so a broken base cannot pass as "the identifier is new"."""
    segs = path.split("::")[1:]
    errs = [ln for ln in stderr.splitlines() if MISSING_RE.search(ln)]
    if errs and "id-move-probe" in stderr and any(re.search(rf"`{re.escape(s)}`", ln) for ln in errs for s in segs):
        return True
    if len(segs) < 2:
        return False
    item, ty = segs[-1], segs[-2]
    errors = [ln for ln in stderr.splitlines() if ERROR_RE.match(ln)]
    if not errors:
        return False
    for ln in errors:
        m = ASSOC_MISSING_RE.match(ln)
        if not m:
            return False
        found_ty = re.sub(r"<.*$", "", m.group(2)).rsplit("::", 1)[-1]
        if m.group(1) != item or found_ty != ty:
            return False
    return "id-move-probe" in stderr


class CheckError(Exception):
    pass


@dataclass(frozen=True)
class IdSpec:
    path: str
    previous: str | None
    previous_tests: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    package: str
    manifest: str
    features: tuple[str, ...]
    default_features: bool
    tags: tuple[str, ...]
    ids: tuple[IdSpec, ...]
    exempt: tuple[tuple[str, str], ...] = ()
    doc_features: tuple[str, ...] | None = None
    probe_deps: tuple[str, ...] = ()
    base_package: str | None = None

    @property
    def base_pkg(self) -> str:
        """the package the release tags and the registry name (the package itself unless renamed)"""
        return self.base_package or self.package


MIN_REASON = 12
SUPPORTED_FORMATS = {60}


def rustdoc_json(crate_dir: Path, cfg: Config) -> dict:
    env = dict(os.environ, RUSTC_BOOTSTRAP="1", RUSTDOCFLAGS="")
    target = Path(os.environ.get("ID_MOVE_TARGET_DIR") or tempfile.mkdtemp(prefix="id-move-doc-")) / "rustdoc"
    try:
        cmd = ["cargo", "rustdoc", "-q", "--lib", "--manifest-path", str(crate_dir / "Cargo.toml"),
               "--target-dir", str(target)]
        # every feature: an identifier behind one must be listed too
        if cfg.doc_features is None:
            cmd.append("--all-features")
        elif cfg.doc_features:
            cmd += ["--features", ",".join(cfg.doc_features)]
        cmd += ["--", "-Z", "unstable-options", "--output-format", "json",
                "--document-private-items", "--document-hidden-items"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise CheckError(f"rustdoc failed in {crate_dir}: {r.stderr.strip()[-800:]}")
        lib = cfg.package.replace("-", "_")
        doc = target / "doc" / f"{lib}.json"
        if not doc.exists():
            docs = list((target / "doc").glob("*.json"))
            if len(docs) != 1:
                raise CheckError(f"rustdoc wrote no JSON for {cfg.package} ({[d.name for d in docs]})")
            doc = docs[0]
        data = json.loads(doc.read_text(encoding="utf-8"))
        doc.unlink()
    finally:
        if not os.environ.get("ID_MOVE_TARGET_DIR"):
            shutil.rmtree(target.parent, ignore_errors=True)
    if data.get("format_version") not in SUPPORTED_FORMATS:
        raise CheckError(f"rustdoc JSON format_version {data.get('format_version')} is not one of {sorted(SUPPORTED_FORMATS)}")
    return data


def is_id_type(ty, index: dict, depth: int = 0) -> bool:
    """the type resolves to [u8; 32] (through &, type aliases); a length that is not a number
    counts as a hit, so an identifier written with a computed length is not skipped"""
    if not isinstance(ty, dict) or depth > 16:
        return False
    if "borrowed_ref" in ty:
        return is_id_type(ty["borrowed_ref"].get("type"), index, depth + 1)
    if "array" in ty:
        arr = ty["array"]
        if arr.get("type") != {"primitive": "u8"}:
            return False
        length = str(arr.get("len", "")).strip()
        try:
            return int(re.sub(r"(?:usize|u\d+)$", "", length).replace("_", ""), 0) == 32
        except ValueError:
            return True
    if "resolved_path" in ty:
        target = index.get(str(ty["resolved_path"].get("id")))
        inner = (target or {}).get("inner", {})
        if "type_alias" in inner:
            return is_id_type(inner["type_alias"].get("type"), index, depth + 1)
    return False


def id_constants_from_doc(doc: dict) -> dict[str, list[str]]:
    """{name: [file:line, ...]} of every const / static item of the documented crate whose type
    resolves to [u8; 32]"""
    index = doc.get("index", {})
    root_crate = index.get(str(doc.get("root")), {}).get("crate_id", 0)
    # the owner of an associated const: the impl's self type, or the trait
    owner: dict[str, str] = {}
    for item in index.values():
        inner = item.get("inner", {})
        if "impl" in inner:
            for_ty = inner["impl"].get("for") or {}
            name = (for_ty.get("resolved_path") or {}).get("path", "").rsplit("::", 1)[-1] or "?"
            for child in inner["impl"].get("items", []):
                owner[str(child)] = name
        elif "trait" in inner:
            for child in inner["trait"].get("items", []):
                owner[str(child)] = item.get("name") or "?"
    out: dict[str, list[str]] = {}
    for key, item in index.items():
        if item.get("crate_id") != root_crate or not item.get("name"):
            continue
        inner = item.get("inner", {})
        kind = next((k for k in ("constant", "static", "assoc_const") if k in inner), None)
        if kind is None or not is_id_type(inner[kind].get("type"), index):
            continue
        name = item["name"] if kind != "assoc_const" else f"{owner.get(str(key), '?')}::{item['name']}"
        span = item.get("span") or {}
        where = f"{span.get('filename', '?')}:{(span.get('begin') or ['?'])[0]}"
        out.setdefault(name, []).append(where)
    # a public re-export of another crate's constant or static (`pub use alice_core::law::SEMANTICS_ID`):
    # the JSON carries the target's kind but not its type, so every such re-export counts
    paths = doc.get("paths", {})
    for item in index.values():
        use = item.get("inner", {}).get("use")
        if item.get("crate_id") != root_crate or not use or item.get("visibility") != "public":
            continue
        target = paths.get(str(use.get("id")), {})
        if use.get("is_glob"):
            # `pub use other_crate::*` (or a module of it): whatever [u8; 32] it holds becomes public
            # here unseen, so the glob itself must be listed under [exempt] with a reason
            if target.get("crate_id") != root_crate:
                span = item.get("span") or {}
                where = f"{span.get('filename', '?')}:{(span.get('begin') or ['?'])[0]} (glob re-export)"
                out.setdefault(f"{use.get('source') or '?'}::*", []).append(where)
            continue
        if target.get("crate_id") == root_crate or target.get("kind") not in ("constant", "static"):
            continue
        span = item.get("span") or {}
        where = f"{span.get('filename', '?')}:{(span.get('begin') or ['?'])[0]} (re-export of {'::'.join(target.get('path', []))})"
        out.setdefault(use.get("name") or "?", []).append(where)
    return {k: sorted(v) for k, v in out.items()}


_DOC: dict[str, dict] = {}


def id_constants(crate_dir: Path, cfg: Config, root: Path | None = None) -> dict[str, list[str]]:
    doc = rustdoc_json(crate_dir, cfg)
    if root is not None:
        _DOC[str(root)] = doc
    return id_constants_from_doc(doc)


def completeness(cfg: Config, found: dict[str, list[str]]) -> list[str]:
    problems = []
    if not found:
        return ["no const / static item of type [u8; 32] found in the crate (compared nothing)"]
    # a constant by its name, an associated const by Type::NAME (the last two segments)
    listed = {spec.path.rsplit("::", 1)[1] for spec in cfg.ids} | {"::".join(spec.path.split("::")[-2:]) for spec in cfg.ids}
    exempt = dict(cfg.exempt)
    for name, where in sorted(found.items()):
        if name not in listed and name not in exempt:
            problems.append(f"{name} ({', '.join(where)}): a [u8; 32] item that is neither an [[id]] of "
                            f"{CONFIG} nor under [exempt] with a reason")
    for name, reason in sorted(exempt.items()):
        if name not in found:
            problems.append(f"[exempt] {name}: names no const / static item of type [u8; 32] (stale)")
        if len(reason.strip()) < MIN_REASON:
            problems.append(f"[exempt] {name}: reason shorter than {MIN_REASON} characters")
        if name in listed:
            problems.append(f"[exempt] {name}: also an [[id]]")
    return problems


def load_config(root: Path) -> Config:
    p = root / CONFIG
    if not p.exists():
        raise CheckError(f"{CONFIG} is missing")
    doc = tomllib.loads(p.read_text(encoding="utf-8"))
    ids = []
    for e in doc.get("id", []):
        path = e.get("path")
        if not isinstance(path, str) or not re.fullmatch(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)+", path):
            raise CheckError(f"{CONFIG}: [[id]] path {path!r} is not a Rust path")
        prev = e.get("previous")
        if prev is not None and (not isinstance(prev, str) or not re.fullmatch(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)+", prev)):
            raise CheckError(f"{CONFIG}: [[id]] previous {prev!r} is not a Rust path")
        ptests = e.get("previous-tests", [])
        if not (isinstance(ptests, list) and all(isinstance(t, str) and re.fullmatch(r"[\w./-]+\.rs::\w+", t) for t in ptests)):
            raise CheckError(f"{CONFIG}: [[id]] previous-tests must be a list of `<path>.rs::<fn>`")
        if ptests and prev is None:
            raise CheckError(f"{CONFIG}: [[id]] {path}: previous-tests without a previous list")
        ids.append(IdSpec(path, prev, tuple(ptests)))
    tags = doc.get("tags", [])
    if not ids:
        raise CheckError(f"{CONFIG}: no [[id]] (compared nothing)")
    if not tags or not all(isinstance(t, str) and "{version}" in t for t in tags):
        raise CheckError(f"{CONFIG}: tags must be patterns containing {{version}}")
    for key in ("package", "manifest"):
        if not isinstance(doc.get(key), str) or not doc[key]:
            raise CheckError(f"{CONFIG}: {key} is missing")
    exempt = doc.get("exempt", {})
    if not isinstance(exempt, dict) or not all(isinstance(v, str) for v in exempt.values()):
        raise CheckError(f"{CONFIG}: [exempt] must map names to reasons")
    doc_features = doc.get("doc-features")
    if doc_features is not None and not (isinstance(doc_features, list) and all(isinstance(f, str) for f in doc_features)):
        raise CheckError(f"{CONFIG}: doc-features must be a list of feature names")
    probe_deps = doc.get("probe-dependencies", [])
    if not (isinstance(probe_deps, list) and all(isinstance(d, str) and re.fullmatch(r"[A-Za-z0-9_-]+", d) for d in probe_deps)):
        raise CheckError(f"{CONFIG}: probe-dependencies must be a list of crate names")
    base_package = doc.get("base-package")
    if base_package is not None and not (isinstance(base_package, str) and re.fullmatch(r"[A-Za-z0-9_-]+", base_package)):
        raise CheckError(f"{CONFIG}: base-package must be a package name")
    return Config(doc["package"], doc["manifest"], tuple(doc.get("features", [])),
                  bool(doc.get("default-features", True)), tuple(tags), tuple(ids), tuple(sorted(exempt.items())),
                  None if doc_features is None else tuple(doc_features), tuple(probe_deps), base_package)


def parse_version(v: str) -> tuple[int, int, int]:
    m = VERSION_RE.match(v)
    if not m:
        raise CheckError(f"version {v!r} is not semver")
    return int(m[1]), int(m[2]), int(m[3])


def is_breaking(old: str, new: str) -> bool:
    """`new` is a semver-breaking step from `old`: a major, or a minor while the major is 0."""
    a, b = parse_version(old), parse_version(new)
    if b[0] != a[0]:
        return b[0] > a[0]
    if a[0] == 0:
        return b[1] > a[1]
    return False


def next_breaking(v: str) -> str:
    """the smallest breaking step from `v`: the next minor in 0.x, the next major from 1.0"""
    a = parse_version(v)
    return f"0.{a[1] + 1}.0" if a[0] == 0 else f"{a[0] + 1}.0.0"


def package_version(manifest_text: str, package: str) -> str:
    doc = tomllib.loads(manifest_text)
    pkg = doc.get("package", {})
    if pkg.get("name") != package:
        raise CheckError(f"manifest names package {pkg.get('name')!r}, the config {package!r}")
    v = pkg.get("version")
    if isinstance(v, dict):
        raise CheckError("version.workspace = true is not read; give the member's version")
    if not isinstance(v, str):
        raise CheckError("the manifest has no version")
    return v


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise CheckError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def release_base(root: Path, cfg: Config) -> tuple[str, str]:
    """(tag, version) of the newest release tag reachable from HEAD matching a pattern."""
    candidates = []
    for tag in git(root, "tag", "--merged", "HEAD").split():
        for pat in cfg.tags:
            pre, post = pat.split("{version}", 1)
            if tag.startswith(pre) and tag.endswith(post):
                v = tag[len(pre):len(tag) - len(post) if post else None]
                if VERSION_RE.match(v):
                    candidates.append((parse_version(v), tag, v))
    # newest first; a tag counts only when the manifest at it is this package at that
    # version (a repository can carry tags of another package or of an earlier language)
    for _, tag, v in sorted(candidates, reverse=True):
        r = subprocess.run(["git", "-C", str(root), "show", f"{tag}:{cfg.manifest}"], capture_output=True, text=True)
        if r.returncode != 0:
            continue
        try:
            if package_version(r.stdout, cfg.base_pkg) == v:
                return tag, v
        except (CheckError, tomllib.TOMLDecodeError):
            continue
    raise CheckError(f"no release tag reachable from HEAD matches {list(cfg.tags)} with {cfg.base_pkg} "
                     f"at that version in {cfg.manifest} (nothing published to compare with; pass --base to name one)")


MAIN_REF = "origin/main"


def is_ancestor(root: Path, a: str, b: str) -> bool:
    return subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", a, b],
                          capture_output=True).returncode == 0


def choose_bases(root: Path, cfg: Config, base_from: str, main_ref: str = MAIN_REF) -> list[tuple[str | None, str]]:
    """the bases to judge against, each (ref, kind): kind "tag" (a release tag), "bump" (the commit
    before the newest version bump on main) or "introduced" (no base: the crate came after the last
    commit before its first version, ref None). `auto` judges against both the tag and the bump base
    when the bump base is newer: the tag is what stored data in the wild was stamped by, the bump
    base the version line main moved on since"""
    if base_from in ("registry", "tag"):
        return [(release_base(root, cfg)[0], "tag")]
    tag, why_tag = None, ""
    if base_from == "auto":
        try:
            tag = release_base(root, cfg)[0]
        except CheckError as e:
            why_tag = str(e)
    try:
        bump = release_baseline.bump_baseline(root, cfg.manifest, cfg.package, base_package=cfg.base_package,
                                              main_ref=main_ref)
    except release_baseline.BaselineError as e:
        if tag is None:
            raise CheckError(f"no base: {why_tag + '; ' if why_tag else ''}{e}") from None
        return [(tag, "tag")]
    if bump.introduced:
        return [(tag, "tag")] if tag is not None else [(None, "introduced")]
    if tag is None:
        return [(bump.sha, "bump")]
    tag_sha = git(root, "rev-parse", f"{tag}^{{commit}}").strip()
    if tag_sha != bump.sha and is_ancestor(root, tag_sha, bump.sha):
        return [(tag, "tag"), (bump.sha, "bump")]
    return [(tag, "tag")]


def choose_base(root: Path, cfg: Config, base_from: str, main_ref: str = MAIN_REF) -> tuple[str | None, str]:
    """the newest of the bases (the one --print-base names)"""
    return choose_bases(root, cfg, base_from, main_ref)[-1]


CARGO_OUTPUT_PARSER = True

# terminal escape sequences (CSI, character-set selectors, OSC): with CARGO_TERM_COLOR=always cargo
# and libtest colour their words, so the output is read only after these are removed
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[()*+][0-9A-Za-z]|\][^\x07\x1b]*(?:\x07|\x1b\\))")


def tests_ran(stdout: str) -> int:
    """the tests a `cargo test` output reports as run (passed plus failed), read without colour"""
    plain = ANSI.sub("", stdout)
    return sum(int(x) for x in re.findall(r"test result: \w+\. (\d+) passed", plain)) + \
        sum(int(x) for x in re.findall(r"(\d+) failed", plain))


SECTION = re.compile(r"^##\s+\[?([^\]\s]+)\]?")
ITEM = re.compile(r"^\s*[-*]\s+")


def breaking_entries(text: str) -> list[str]:
    """the list items of CHANGELOG.md's `[Unreleased]` section that start with `**Breaking:**`, each
    with its continuation lines (up to the next item or heading)"""
    items: list[str] = []
    current: list[str] | None = None
    unreleased = False
    for line in text.split("\n"):
        m = SECTION.match(line)
        if m:
            unreleased = m.group(1).lower() == "unreleased"
            current = None
            continue
        if not unreleased:
            continue
        if line.startswith("#"):
            current = None
            continue
        if ITEM.match(line):
            body = ITEM.sub("", line, count=1)
            current = [body] if body.startswith("**Breaking:**") else None
            if current is not None:
                items.append("")
        elif current is not None and line.strip():
            current.append(line)
        if current is not None:
            items[-1] = " ".join(current)
    return items


def changelog_sections(text: str, head_version: str) -> str:
    """the `[Unreleased]` section and the section of `head_version` (each from its `## ` heading to
    the next `## ` heading), where a move against a bump base goes out"""
    out, keep = [], False
    for line in text.split("\n"):
        m = SECTION.match(line)
        if m:
            keep = m.group(1).lower() == "unreleased" or m.group(1).lstrip("v") == head_version
        if keep:
            out.append(line)
    return "\n".join(out)


def probe_source(cfg: Config, spec: IdSpec, with_previous: bool = True) -> str:
    lines = ["fn hex(b: &[u8; 32]) -> String { b.iter().map(|x| format!(\"{:02x}\", x)).collect() }",
             "fn main() {",
             f"    let id: [u8; 32] = {spec.path};",
             "    print!(\"{{\\\"id\\\": \\\"{}\\\", \\\"previous\\\": \", hex(&id));"]
    if spec.previous and with_previous:
        lines += [f"    let prev: &[[u8; 32]] = {spec.previous};",
                  "    let p: Vec<String> = prev.iter().map(hex).collect();",
                  "    println!(\"[{}]}}\", p.iter().map(|h| format!(\"\\\"{}\\\"\", h)).collect::<Vec<_>>().join(\", \"));"]
    else:
        lines.append("    println!(\"null}}\");")
    lines.append("}")
    return "\n".join(lines) + "\n"


class NotPublished(CheckError):
    pass


class RegistryUnreachable(CheckError):
    pass


UNREACHABLE_RE = re.compile(r"failed to (?:download|get|query|update)|spurious network error|Couldn't resolve host|"
                            r"Could not resolve host|timed out|SSL|failed to fetch|network failure|HTTP status (?:5\d\d|429)")


def unreadable_base_policy(env: dict[str, str]) -> str:
    """"fail" or "warn" when the base cannot be read (the registry is unreachable): a registry
    outage must not turn every branch red, but the default branch and release tags must not
    pass a check they did not run. Local runs (no GITHUB_REF) fail."""
    ref = env.get("GITHUB_REF", "")
    if env.get("GITHUB_EVENT_NAME") in ("pull_request", "pull_request_target"):
        return "warn"
    if ref.startswith("refs/heads/ci/") or ref.startswith("refs/heads/wip/"):
        return "warn"
    return "fail"


def read_ids(source: Path | str, cfg: Config, target_dir: Path, with_previous: bool = True,
             is_base: bool = False) -> dict[str, dict | None]:
    """{path: {"id": hex, "previous": [hex] | None} or None when the crate does not have it}, of the
    crate in the tree `source` (a Path) or of the published version `source` (a str) from the
    registry; the previous lists are read only where they are judged (the working tree): the base
    may predate them; `is_base`: `source` is the base (its package is `base-package`)"""
    package = cfg.base_pkg if is_base else cfg.package
    out: dict[str, dict | None] = {}
    for i, spec in enumerate(cfg.ids):
        probe = Path(tempfile.mkdtemp(prefix="id-move-probe-"))
        try:
            (probe / "src").mkdir()
            if isinstance(source, Path):
                dep = {"path": str((source / cfg.manifest).resolve().parent)}
            else:
                dep = {"version": f"={source}"}
            if cfg.base_package is not None:
                # one dependency key (the current lib name) for both: a path resolves in either
                dep["package"] = package
            dep["default-features"] = cfg.default_features
            if cfg.features:
                dep["features"] = list(cfg.features)
            dep_toml = "{ " + ", ".join(f"{k} = {json.dumps(v)}" for k, v in dep.items()) + " }"
            (probe / "Cargo.toml").write_text(
                f"[package]\nname = \"id-move-probe-{i}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n"
                f"[dependencies]\n{json.dumps(cfg.package if cfg.base_package is None else cfg.package.replace('-', '_'))} = {dep_toml}\n"
                # "*" unifies with the package's own requirement, so the probe reads the same version
                + "".join(f"{json.dumps(d)} = {{ version = \"*\", default-features = false }}\n" for d in cfg.probe_deps)
                + "\n[workspace]\n", encoding="utf-8")
            (probe / "src" / "main.rs").write_text(probe_source(cfg, spec, with_previous), encoding="utf-8")
            env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir), RUSTFLAGS="")
            env.pop("RUSTC_WRAPPER", None)
            r = subprocess.run(["cargo", "run", "--quiet", "--manifest-path", str(probe / "Cargo.toml")],
                               capture_output=True, text=True, env=env)
            if r.returncode != 0 and isinstance(source, str) and re.search(
                    r"failed to select a version|no matching package named|could not find .* in registry", r.stderr):
                raise NotPublished(f"{package} {source} is not in the registry")
            if r.returncode != 0 and isinstance(source, str) and UNREACHABLE_RE.search(r.stderr):
                raise RegistryUnreachable(f"cannot read the base {package} {source} from the registry: "
                                          f"{r.stderr.strip()[-400:]}")
            if r.returncode != 0:
                if probe_says_absent(spec.path, r.stderr):
                    out[spec.path] = None
                    continue
                raise CheckError(f"the probe for {spec.path} did not run against {source}: {r.stderr.strip()[-800:]}")
            line = r.stdout.strip().splitlines()[-1]
            out[spec.path] = json.loads(line)
        finally:
            shutil.rmtree(probe, ignore_errors=True)
    return out


def _test_fn_problem(crate_dir: Path, ref: str, list_name: str) -> str | None:
    rel, fn = ref.split("::")
    f = crate_dir / rel
    if not f.exists():
        return f"{rel} does not exist"
    text = f.read_text(encoding="utf-8")
    m = re.search(r"((?:[ \t]*#\[[^\n]*\]\s*\n)*)[ \t]*(?:pub\s+)?fn\s+" + re.escape(fn) + r"\s*\(", text)
    if not m:
        return f"no fn {fn} in {rel}"
    if "#[test]" not in m.group(1):
        return f"fn {fn} has no #[test]"
    if re.search(r"#\[\s*ignore", m.group(1)):
        return f"fn {fn} is #[ignore]d"
    return None


def remove_value(source: str, old_hex: str) -> str | None:
    """`source` (a previous list's definition) with the value `old_hex` changed so it no longer
    matches, or None when the value is not written there in a form this tool reads"""
    if old_hex in source.lower():
        i = source.lower().index(old_hex)
        last = source[i + 63]
        flipped = "0" if last.lower() != "0" else "1"
        return source[:i + 63] + flipped + source[i + 64:]
    byte_list = [old_hex[k:k + 2] for k in range(0, 64, 2)]
    rx = r"\s*,\s*".join(rf"0[xX]{b}(?:_?u8)?" for b in byte_list)
    m = re.search(rx, source, re.I)
    if m:
        tail = re.search(r"0[xX][0-9a-fA-F]{2}(?:_?u8)?$", m.group(0))
        flipped = "0x00" if byte_list[-1] != "00" else "0x01"
        return source[:m.start() + tail.start()] + flipped + source[m.end():]
    if len(set(byte_list)) == 1:
        m = re.search(rf"\[\s*0[xX]{byte_list[0]}(?:_?u8)?\s*;\s*32\s*\]", source, re.I)
        if m:
            flipped = "0x00" if byte_list[0] != "00" else "0x01"
            return source[:m.start()] + f"[{flipped}; 32]" + source[m.end():]
    return None


def run_named_test(crate_dir: Path, ref: str, target_dir: Path, features: tuple[str, ...] | None = None) -> tuple[bool, int, str]:
    """(passed, tests run, output tail) of one named test in the crate at `crate_dir`, built with
    `features` (the config's doc-features; every feature when None)"""
    rel, fn = ref.split("::")
    sel = ["--test", Path(rel).stem] if rel.startswith("tests/") else ["--lib"]
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    feat = ["--all-features"] if features is None else (["--features", ",".join(features)] if features else [])
    r = subprocess.run(["cargo", "test", "--quiet", "--manifest-path", str(crate_dir / "Cargo.toml"), *feat,
                        *sel, fn, "--", "--exact"], capture_output=True, text=True, env=env)
    ran = tests_ran(r.stdout)
    return r.returncode == 0, ran, (r.stdout + r.stderr).strip()[-400:]


# a POSIX string, not a Path: it appears in messages that must read the same on every OS
SIBLINGS_FILE = ".github/alice-siblings"
SIBLING_LINE = re.compile(r"(ALICE-[A-Za-z0-9-]+) ([0-9a-f]{40})")


def listed_siblings(root: Path) -> tuple[dict[str, str], list[str]]:
    """the repositories in .github/alice-siblings ("<Repo> <40-hex SHA>" per line) as {repo: sha}"""
    path = root / SIBLINGS_FILE
    if not path.is_file():
        return {}, [f"{SIBLINGS_FILE} is missing"]
    return parse_siblings(path.read_text(encoding="utf-8"))


def parse_siblings(text: str) -> tuple[dict[str, str], list[str]]:
    """the lines of a .github/alice-siblings as {repo: sha}, and the lines that are not one pin"""
    listed, problems = {}, []
    for no, raw in enumerate(text.splitlines(), 1):
        line = " ".join(raw.split("#", 1)[0].split())
        if not line:
            continue
        m = SIBLING_LINE.fullmatch(line)
        if not m or m.group(1) in listed:
            problems.append(f"{SIBLINGS_FILE}:{no}: not one '<ALICE-Repo> <40-hex SHA>' line: {raw!r}")
        else:
            listed[m.group(1)] = m.group(2)
    return listed, problems


def reexport_target(doc: dict, list_name: str) -> tuple[str, list[str]] | None:
    """(defining crate, item path) of a `pub use other::LIST` re-export named `list_name`"""
    for item in doc.get("index", {}).values():
        use = item.get("inner", {}).get("use")
        if not use or use.get("is_glob") or use.get("name") != list_name or use.get("id") is None:
            continue
        target = doc.get("paths", {}).get(str(use["id"]))
        if not target:
            continue
        crate = doc.get("external_crates", {}).get(str(target.get("crate_id")), {}).get("name")
        if crate and target.get("crate_id") != 0:
            return crate, list(target.get("path", []))
    return None


def constant_span(doc: dict, item_path: list[str]) -> tuple[str, int, int] | None:
    """source span of the constant whose rustdoc path is `item_path` (the defining crate's doc)"""
    for iid, item in doc.get("index", {}).items():
        if item.get("name") != item_path[-1] or "constant" not in item.get("inner", {}) or not item.get("span"):
            continue
        known = doc.get("paths", {}).get(str(iid), {}).get("path")
        if known is not None and known != item_path:
            continue
        sp = item["span"]
        return sp["filename"], sp["begin"][0], sp["end"][0]
    return None


def copy_tracked(src: Path, dst: Path) -> None:
    """the tracked and untracked-but-not-ignored files of the repository at `src`, with new mtimes
    (a build left in the shared target dir from an earlier, changed copy must not look newer)"""
    files = subprocess.run(["git", "-C", str(src), "ls-files", "-co", "--exclude-standard"],
                           capture_output=True, text=True, check=True).stdout.split("\n")
    for f in files:
        if not f or f.startswith("target/"):
            continue
        s = src / f
        if s.is_file():
            d = dst / f
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(s, d)


def _dep_paths(node, deps: bool = False):
    """the `path` of every dependency entry of a parsed manifest: under any table whose name ends in
    `dependencies` (also `target.<cfg>.*` and `workspace.dependencies`) and under `[patch.*]`"""
    if isinstance(node, dict):
        if deps and isinstance(node.get("path"), str):
            yield node["path"]
            return
        for k, v in node.items():
            yield from _dep_paths(v, deps or k.endswith("dependencies") or k == "patch")
    elif isinstance(node, list):
        for v in node:
            yield from _dep_paths(v, deps)


def _manifests(repo: Path) -> list[Path]:
    if (repo / ".git").exists():
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "-co", "--exclude-standard", "--", "Cargo.toml",
                              "*/Cargo.toml"], capture_output=True, text=True, check=True).stdout.split("\n")
        return [repo / f for f in out if f and (repo / f).is_file()]
    return [m for m in repo.rglob("Cargo.toml") if "target" not in m.relative_to(repo).parts]


def path_dep_siblings(copy_root: Path, real_parent: Path, copied: frozenset[str] = frozenset()) -> list[str]:
    """the sibling repositories (`../<Repo>`) that the manifests of `copy_root` name as path
    dependencies, and the ones their manifests name in turn: a sibling may point back at this
    repository (`../<this repo>/<crate>`), so a copy is built under a fresh parent that holds the
    copy under the repository's own name and a link to each of these, and every `../` resolves
    inside it; a repository in `copied` is laid out next to the copy already and is read there"""
    own, seen, queue = copy_root.name, set(), [copy_root]
    while queue:
        repo = queue.pop()
        for m in _manifests(repo):
            try:
                doc = tomllib.loads(m.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as e:
                raise CheckError(f"{m}: {e}") from e
            for dep in _dep_paths(doc):
                t = Path(os.path.normpath(m.parent / dep))
                if t == repo or repo in t.parents:
                    continue
                try:
                    name = t.relative_to(repo.parent).parts[0]
                except ValueError:
                    raise CheckError(f"{m}: path dependency {dep!r} leaves the directory of the sibling "
                                     f"repositories ({repo.parent})") from None
                if name == own or name in seen:
                    continue
                if name in copied:
                    seen.add(name)
                    queue.append(copy_root.parent / name)
                    continue
                if not (real_parent / name).is_dir():
                    raise CheckError(f"{m}: path dependency {dep!r} names ../{name}, which is not checked out "
                                     f"next to the repository ({real_parent / name})")
                seen.add(name)
                queue.append(real_parent / name)
    return sorted(seen)


def link_siblings(copy_root: Path, real_parent: Path, copied: set[str]) -> None:
    """links in the parent of `copy_root` to the siblings its manifests reach (except `copied`)"""
    for name in path_dep_siblings(copy_root, real_parent, frozenset(copied)):
        if name not in copied:
            (copy_root.parent / name).symlink_to(real_parent / name, target_is_directory=True)


def _sibling_path_deps(repo: Path) -> list[str]:
    """the sibling repositories (`../<Repo>`) the manifests of the checkout `repo` name by path"""
    names = []
    for m in _manifests(repo):
        try:
            doc = tomllib.loads(m.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise CheckError(f"{m}: {e}") from e
        for dep in _dep_paths(doc):
            t = Path(os.path.normpath(m.parent / dep))
            if t == repo or repo in t.parents:
                continue
            try:
                name = t.relative_to(repo.parent).parts[0]
            except ValueError:
                raise CheckError(f"{m}: path dependency {dep!r} leaves the directory of the sibling "
                                 f"repositories ({repo.parent})") from None
            if name not in names:
                names.append(name)
    return names


def _sibling_at_date(name: str, real: Path, date: str) -> str:
    """the newest commit of `real`'s origin main (first parent) whose committer date is not later
    than `date`: what CI cloned before the repository pinned its siblings"""
    if git(real, "rev-parse", "--is-shallow-repository").strip() == "true":
        raise CheckError(f"../{name} is a shallow clone: its main as of {date} cannot be read")
    subprocess.run(["git", "-C", str(real), "fetch", "-q", "origin"], capture_output=True)
    r = subprocess.run(["git", "-C", str(real), "rev-list", "-1", "--first-parent", f"--before={date}",
                        "origin/main"], capture_output=True, text=True)
    sha = r.stdout.strip()
    if r.returncode != 0 or not sha:
        raise CheckError(f"../{name} has no commit of its origin main committed at or before {date} "
                         f"(the base predates the sibling)")
    return sha


def materialize_pinned_siblings(tree: Path, real_parent: Path, work: Path,
                                root: Path | None = None, base_ref: str | None = None) -> list[str]:
    """write next to `tree` (a base tree under a fresh parent) every sibling its manifests reach by
    path dependency, at the commit `tree`'s .github/alice-siblings pins, and the siblings those
    reach in turn: the base is built against the siblings it was built against, not today's
    checkouts (a sibling may since have taken another major version)

    A base from before the repository had the pin file (no commit of `base_ref`'s first-parent
    history has it) takes each sibling at its origin main as of the base commit's committer date;
    a base that misses the file although its first-parent history has it is broken. Returns what was resolved, as
    report lines"""
    pins, problems = listed_siblings(tree)
    as_of = None
    if not (tree / SIBLINGS_FILE).is_file() and root is not None and base_ref is not None:
        # the fallback is for a base from before the pins only: if any commit of the base's
        # first-parent history has the file, the base lacks it because it was removed, and is broken
        touched = git(root, "log", "--first-parent", "--format=%H", base_ref, "--", SIBLINGS_FILE).split()
        if touched:
            raise CheckError(f"broken base {base_ref}: {SIBLINGS_FILE} is in the base's first-parent history "
                             f"(last changed in {touched[0][:12]}), but the base tree has none")
        as_of = git(root, "show", "-s", "--format=%cI", base_ref).strip()
    resolved = []
    own, done, queue = tree.name, set(), [tree]
    while queue:
        repo = queue.pop()
        for name in _sibling_path_deps(repo):
            if name == own or name in done:
                continue
            real = real_parent / name
            if as_of is not None:
                if not (real / ".git").exists():
                    raise CheckError(f"the base tree path-depends on ../{name}, which is not checked out next "
                                     f"to the repository ({real})")
                sha = _sibling_at_date(name, real, as_of)
                resolved.append(f"../{name} at {sha[:12]} (sibling main as of {as_of}, base predates pins)")
            elif name not in pins:
                why = "; ".join(problems) if problems else f"{SIBLINGS_FILE} lists no {name}"
                raise CheckError(f"the base tree path-depends on ../{name} but has no pin for it ({why})")
            else:
                sha = pins[name]
                resolved.append(f"../{name} at {sha[:12]} (pinned by the base tree)")
            if not (real / ".git").exists():
                raise CheckError(f"the base tree pins ../{name} at {sha[:12]}, which is not checked out next "
                                 f"to the repository ({real})")
            if subprocess.run(["git", "-C", str(real), "cat-file", "-e", f"{sha}^{{commit}}"],
                              capture_output=True).returncode != 0:
                subprocess.run(["git", "-C", str(real), "fetch", "-q", "origin", sha], capture_output=True)
                if subprocess.run(["git", "-C", str(real), "cat-file", "-e", f"{sha}^{{commit}}"],
                                  capture_output=True).returncode != 0:
                    raise CheckError(f"the base tree pins ../{name} at {sha[:12]}, which is not a commit of "
                                     f"{real} (nor fetchable from its origin)")
            dst = tree.parent / name
            env = dict(os.environ, GIT_INDEX_FILE=str(work / f"sibling-{name}.index"))
            for args in (["read-tree", sha], ["checkout-index", "--all", "--force", f"--prefix={dst}/"]):
                r = subprocess.run(["git", "-C", str(real), *args], capture_output=True, text=True, env=env)
                if r.returncode != 0:
                    raise CheckError(f"git {args[0]} {sha[:12]} in ../{name}: {r.stderr.strip()}")
            done.add(name)
            queue.append(dst)
    return resolved


FETCH_TIMEOUT = 300  # seconds for one git command against a sibling's remote (a fetch, or a lazy blob fetch)
GIT_BASE_DEFAULT = "https://github.com/ext-sakamoro"
PRIVATE_FILE = ".github/alice-siblings-private"


def _git_timed(repo: Path, *args: str) -> str:
    """git in `repo`, failing (CheckError) instead of waiting longer than FETCH_TIMEOUT or prompting"""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        r = subprocess.run(["git", "-C", str(repo), "-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=60",
                            *args], capture_output=True, text=True, env=env, timeout=FETCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise CheckError(f"git {' '.join(args[:2])} in {repo.name}: no answer within {FETCH_TIMEOUT} s") from None
    if r.returncode != 0:
        raise CheckError(f"git {' '.join(args[:2])} in {repo.name}: {r.stderr.strip()}")
    return r.stdout


def _tree_sibling_path_deps(root: Path, ref: str) -> list[str]:
    """the sibling repositories (`../<Repo>`) the manifests of the tree `ref` name by path, read out
    of the repository's history (nothing is checked out)"""
    repo = Path("/") / "tree" / root.resolve().name
    names = []
    for rel in _git_timed(root, "ls-tree", "-r", "--name-only", ref).splitlines():
        if rel != "Cargo.toml" and not rel.endswith("/Cargo.toml"):
            continue
        if "target" in Path(rel).parts:
            continue
        try:
            doc = tomllib.loads(_git_timed(root, "show", f"{ref}:{rel}"))
        except tomllib.TOMLDecodeError as e:
            raise CheckError(f"{ref}:{rel}: {e}") from e
        m = repo / rel
        for dep in _dep_paths(doc):
            t = Path(os.path.normpath(m.parent / dep))
            if t == repo or repo in t.parents:
                continue
            try:
                name = t.relative_to(repo.parent).parts[0]
            except ValueError:
                raise CheckError(f"{ref}:{rel}: path dependency {dep!r} leaves the directory of the sibling "
                                 f"repositories") from None
            if name not in names:
                names.append(name)
    return names


def _private_siblings(root: Path) -> set[str]:
    path = root / PRIVATE_FILE
    if not path.is_file():
        return set()
    return {line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()} - {""}


def _siblings_as_of(root: Path, ref: str, reached: list[str], as_of: str) -> list[tuple[str, str]]:
    """(repository, commit) of every sibling a base from before the pin file reaches, at any depth:
    each at the newest commit of its main (first parent) committed at or before `as_of`, the rule
    materialize_pinned_siblings applies; each sibling's main is fetched read-only into a temporary
    directory (from $ALICE_SIBLINGS_GIT_BASE, the alice-siblings action's source)"""
    base_url = os.environ.get("ALICE_SIBLINGS_GIT_BASE") or GIT_BASE_DEFAULT
    private = _private_siblings(root)
    own = root.resolve().name
    work = Path(tempfile.mkdtemp(prefix="id-move-siblings-"))
    found: dict[str, str] = {}
    try:
        queue = list(reached)
        while queue:
            name = queue.pop(0)
            if name == own or name in found:
                continue
            if name in private:
                raise CheckError(f"base {ref} predates {SIBLINGS_FILE} and reaches ../{name}, which "
                                 f"{PRIVATE_FILE} lists: its main as of {as_of} cannot be read without its key")
            repo = work / name
            repo.mkdir()
            _git_timed(repo, "init", "-q", "--bare")
            _git_timed(repo, "remote", "add", "origin", f"{base_url}/{name}")
            try:
                _git_timed(repo, "fetch", "-q", "--filter=blob:none", "origin", "+refs/heads/main:refs/remotes/origin/main")
            except CheckError as e:
                raise CheckError(f"base {ref} predates {SIBLINGS_FILE} and reaches ../{name}, whose main cannot be "
                                 f"fetched from {base_url}/{name} ({e})") from None
            sha = _git_timed(repo, "rev-list", "-1", "--first-parent", f"--before={as_of}",
                             "refs/remotes/origin/main").strip()
            if not sha:
                raise CheckError(f"../{name} has no commit of its origin main committed at or before {as_of} "
                                 f"(the base predates the sibling)")
            found[name] = sha
            queue.extend(_tree_sibling_path_deps(repo, sha))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return sorted(found.items())


def base_sibling_pins(root: Path, refs: list[str]) -> list[tuple[str, str]]:
    """every (repository, commit) the base trees `refs` may build against: each base's
    .github/alice-siblings in full (materialize_pinned_siblings refuses a sibling a base reaches
    without a pin, so the siblings it reaches at any depth are among these pins), and for a base
    from before the pin file the commits `_siblings_as_of` resolves; a base that reaches no sibling
    contributes nothing"""
    out: list[tuple[str, str]] = []
    for ref in refs:
        reached = _tree_sibling_path_deps(root, ref)
        r = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{SIBLINGS_FILE}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            touched = git(root, "log", "--first-parent", "--format=%H", ref, "--", SIBLINGS_FILE).split()
            if touched:
                raise CheckError(f"broken base {ref}: {SIBLINGS_FILE} is in the base's first-parent history "
                                 f"(last changed in {touched[0][:12]}), but the base tree has none")
            if reached:
                as_of = git(root, "show", "-s", "--format=%cI", ref).strip()
                for pair in _siblings_as_of(root, ref, reached, as_of):
                    if pair not in out:
                        out.append(pair)
            continue
        if not reached:
            continue
        pins, problems = parse_siblings(r.stdout)
        if problems:
            raise CheckError(f"base {ref}: " + "; ".join(problems))
        unpinned = [n for n in reached if n != root.resolve().name and n not in pins]
        if unpinned:
            raise CheckError(f"base {ref} path-depends on {', '.join('../' + n for n in unpinned)} but its "
                             f"{SIBLINGS_FILE} has no pin for it")
        for pair in pins.items():
            if pair not in out:
                out.append(pair)
    return sorted(out)


def git_rev_dependency(root: Path, crate_dir: Path, crate: str) -> dict | str | None:
    """the `git` + `rev` source the manifests of the crate (and its workspace) give the dependency
    `crate` (lib name), as {url, rev, package}; None when no manifest takes it from git"""
    for m in dict.fromkeys([crate_dir / "Cargo.toml", root / "Cargo.toml"]):
        if not m.is_file():
            continue
        doc = tomllib.loads(m.read_text(encoding="utf-8"))
        tables = [doc.get("dependencies", {}), doc.get("workspace", {}).get("dependencies", {})]
        tables += [t.get("dependencies", {}) for t in doc.get("target", {}).values() if isinstance(t, dict)]
        for table in tables:
            for key, spec in table.items():
                if not isinstance(spec, dict) or "git" not in spec:
                    continue
                package = str(spec.get("package", key))
                if package.replace("-", "_") != crate:
                    continue
                rev = spec.get("rev")
                if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
                    return f"{m}: {key} is taken from git without a 40-hex `rev`"
                return {"url": str(spec["git"]), "rev": rev, "package": package}
    return None


def checked_rev_clone(root: Path, source: dict, cache: Path | None = None) -> tuple[str, Path]:
    """(repository name, local clone) for a git + rev source: the clone next to the repository, or
    in the home directory, or (with neither, as on a CI runner) a partial bare repository under
    `cache` whose origin is the source's url; the rev is fetched from its origin and must be on an
    origin branch; a clone's HEAD and working tree are neither read nor moved"""
    name = Path(urlparse(source["url"]).path).name.removesuffix(".git")
    for clone in (root.resolve().parent / name, Path.home() / name):
        if (clone / ".git").exists():
            break
    else:
        if cache is None:
            raise CheckError(f"{source['url']} rev {source['rev'][:12]}: no clone of {name} next to the "
                             f"repository or in the home directory to read it from")
        clone = cache / f"{name}.git"
        if not (clone / "HEAD").is_file():
            clone.mkdir(parents=True, exist_ok=True)
            for args in (["init", "-q", "--bare"], ["remote", "add", "origin", source["url"]],
                         ["config", "remote.origin.promisor", "true"],
                         ["config", "remote.origin.partialclonefilter", "blob:none"]):
                r = subprocess.run(["git", "-C", str(clone), *args], capture_output=True, text=True)
                if r.returncode != 0:
                    raise CheckError(f"{source['url']}: preparing a fetch of {name} failed: {r.stderr.strip()[-300:]}")
        # no credential prompt: a url that needs credentials fails here instead of waiting
        r = subprocess.run(["git", "-C", str(clone), "fetch", "-q", "--filter=blob:none", "origin",
                            "+refs/heads/*:refs/remotes/origin/*"], capture_output=True, text=True,
                           env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        if r.returncode != 0:
            raise CheckError(f"{source['url']}: fetching {name} failed: {r.stderr.strip()[-300:]}")
    subprocess.run(["git", "-C", str(clone), "fetch", "-q", "origin"], capture_output=True,
                   env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    rev = source["rev"]
    if subprocess.run(["git", "-C", str(clone), "cat-file", "-e", f"{rev}^{{commit}}"],
                      capture_output=True).returncode != 0:
        raise CheckError(f"{source['url']} rev {rev[:12]} is not a commit of {clone}")
    on = subprocess.run(["git", "-C", str(clone), "branch", "-r", "--contains", rev, "--list", "origin/*"],
                        capture_output=True, text=True).stdout.strip()
    if not on:
        raise CheckError(f"{source['url']} rev {rev[:12]} is not on an origin branch of {clone}")
    return name, clone


def archive_rev(clone: Path, rev: str, dst: Path) -> None:
    """the files of `rev` in `clone`, written to `dst` (git archive; the clone is not touched)"""
    r = subprocess.run(["git", "-C", str(clone), "archive", "--format=tar", rev], capture_output=True)
    if r.returncode != 0:
        raise CheckError(f"git archive {rev[:12]} in {clone}: {r.stderr.decode(errors='replace').strip()}")
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:
        tar.extractall(dst, filter="data")


class PreviousVerifier:
    """runs each [[id]]'s previous-tests against a copy of the working tree with the old value
    taken out of the previous list; a list this crate re-exports from a sibling repository listed
    in .github/alice-siblings is taken out of a copy of that sibling laid out next to the copy"""

    def __init__(self, root: Path, cfg: Config, doc: dict | None, target_dir: Path):
        self.root, self.cfg, self.doc, self.target = root, cfg, doc, target_dir

    def _list_span(self, spec: IdSpec) -> tuple[str, int, int] | None:
        name = (spec.previous or "").rsplit("::", 1)[-1]
        for item in (self.doc or {}).get("index", {}).values():
            if item.get("name") == name and "constant" in item.get("inner", {}) and item.get("span"):
                sp = item["span"]
                return sp["filename"], sp["begin"][0], sp["end"][0]
        return None

    def _sibling_span(self, spec: IdSpec) -> tuple[str, tuple[str, int, int]] | str:
        """(sibling repository, span in it) of a re-exported previous list, or why it cannot be read"""
        list_name = spec.previous.rsplit("::", 1)[-1]
        target = reexport_target(self.doc or {}, list_name)
        if target is None:
            return (f"{spec.previous} has no source span in the rustdoc JSON and is not a re-export of another "
                    f"crate's constant, so the old value cannot be taken out of it")
        crate, item_path = target
        top = self.root.resolve()
        crate_dir = (self.root / self.cfg.manifest).resolve().parent
        source = git_rev_dependency(top, crate_dir, crate)
        if isinstance(source, str):
            return f"{spec.previous} re-exports {'::'.join(item_path)}: {source}"
        if source is not None:
            try:
                repo, clone = checked_rev_clone(top, source, self.target / "id-move-git-sources")
                scratch = top.parent / f".id-move-rev-{os.getpid()}"
                try:
                    archive_rev(clone, source["rev"], scratch / repo)
                    sib_doc = rustdoc_json(scratch / repo,
                                           replace(self.cfg, package=source["package"], doc_features=None))
                finally:
                    shutil.rmtree(scratch, ignore_errors=True)
            except CheckError as e:
                return f"{spec.previous} re-exports {'::'.join(item_path)}: {e}"
            span = constant_span(sib_doc, item_path)
            if span is None:
                return (f"{spec.previous} re-exports {'::'.join(item_path)} from {source['url']} rev "
                        f"{source['rev'][:12]}, but its rustdoc JSON has no constant there with a source span")
            return repo, span, dict(source, clone=clone)
        listed, problems = listed_siblings(self.root)
        if problems:
            return f"{spec.previous} re-exports {'::'.join(item_path)}: " + "; ".join(problems)
        for repo, sha in listed.items():
            manifest = top.parent / repo / "Cargo.toml"
            if not manifest.is_file():
                continue
            pkg = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {})
            if str(pkg.get("name", "")).replace("-", "_") != crate:
                continue
            sib = top.parent / repo
            head = git(sib, "rev-parse", "HEAD").strip()
            if head != sha:
                return (f"{spec.previous} re-exports {'::'.join(item_path)} from ../{repo}, which is at {head[:12]} "
                        f"but {SIBLINGS_FILE} pins {sha[:12]}")
            if git(sib, "status", "--porcelain", "--untracked-files=no").strip():
                return f"{spec.previous} re-exports {'::'.join(item_path)} from ../{repo}, which has uncommitted changes"
            sib_doc = rustdoc_json(sib, replace(self.cfg, package=str(pkg["name"]), doc_features=None))
            span = constant_span(sib_doc, item_path)
            if span is None:
                return (f"{spec.previous} re-exports {'::'.join(item_path)} from ../{repo}, but its rustdoc JSON has "
                        f"no constant there with a source span")
            return repo, span, None
        return (f"{spec.previous} re-exports {'::'.join(item_path)} from {crate}, which is not a sibling repository "
                f"listed in {SIBLINGS_FILE} (checked out next to this one)")

    def __call__(self, spec: IdSpec, old: str) -> tuple[list[str], list[str]]:
        problems, passed_refs = [], []
        if not spec.previous_tests:
            return [], [f"{spec.path}: the move keeps {old[:8]} in {spec.previous} but the [[id]] names no "
                        f"previous-tests: name the tests that reproduce the old generation, or treat the move as a "
                        f"change of meaning (breaking version, value out of the list)"]
        crate_dir = (self.root / self.cfg.manifest).resolve().parent
        list_name = spec.previous.rsplit("::", 1)[-1]
        for ref in spec.previous_tests:
            why = _test_fn_problem(crate_dir, ref, list_name)
            if why:
                problems.append(f"{spec.path}: previous-tests {ref}: {why}")
        if problems:
            return [], problems
        top = self.root.resolve()
        own = self._list_span(spec)
        if own is not None:
            edit_repo, (rel, begin, end), source = None, own, None
        else:
            found = self._sibling_span(spec)
            if isinstance(found, str):
                return [], [f"{spec.path}: {found}"]
            edit_repo, (rel, begin, end), source = found
        # a fresh parent holds the copy under the repository's own name, the copy of the sibling
        # to change (if the list is a sibling's), and links to the other siblings the manifests
        # reach, so every `../<Repo>` resolves inside it: a sibling that points back at this
        # repository then reaches the copy, not the working tree (one package, not two)
        cleanup = top.parent / f".id-move-previous-{os.getpid()}"
        base = cleanup
        copy = base / top.name
        try:
            copy_tracked(top, copy)
            if source is not None:
                # the rev's files next to the copy, and the copy's git source of that package patched
                # to them; the build must resolve the package there, not fetch the rev again
                archive_rev(source["clone"], source["rev"], base / edit_repo)
                with open(copy / "Cargo.toml", "a", encoding="utf-8") as f:
                    f.write(f'\n[patch."{source["url"]}"]\n{source["package"]} = {{ path = "../{edit_repo}" }}\n')
            elif edit_repo is not None:
                copy_tracked(top.parent / edit_repo, base / edit_repo)
            try:
                link_siblings(copy, top.parent, {edit_repo} if edit_repo else set())
            except CheckError as e:
                return [], [f"{spec.path}: {e}"]
            if source is not None:
                meta = subprocess.run(["cargo", "metadata", "--format-version", "1", "--manifest-path",
                                       str(copy / "Cargo.toml")], capture_output=True, text=True,
                                      env=dict(os.environ, CARGO_TARGET_DIR=str(self.target)))
                if meta.returncode != 0:
                    return [], [f"{spec.path}: the copy with {edit_repo} patched in does not resolve: "
                                f"{meta.stderr.strip()[-400:]}"]
                used = [Path(pk["manifest_path"]).resolve().parent for pk in json.loads(meta.stdout)["packages"]
                        if pk["name"] == source["package"]]
                if used != [(base / edit_repo).resolve()]:
                    return [], [f"{spec.path}: the copy resolves {source['package']} at {used}, not at the patched "
                                f"../{edit_repo} (the old value would not be taken out of what is built)"]
            if edit_repo is not None:
                where = f"../{edit_repo}/{Path(rel).as_posix()}"
                path = base / edit_repo / rel
            else:
                # rustdoc writes a span relative to the directory rustc ran in: the crate itself, or
                # the workspace root for some members (a wrong pick fails closed: the old value is
                # then not found in the block)
                for under in (crate_dir, top):
                    if (under / rel).is_file():
                        break
                else:
                    return [], [f"{spec.path}: the span {rel} of {spec.previous} is neither under the repository "
                                f"root nor under the crate"]
                where = (under / rel).relative_to(top).as_posix()
                path = copy / where
            copy_crate = copy / crate_dir.relative_to(top)
            for ref in spec.previous_tests:
                ok, ran, out = run_named_test(copy_crate, ref, self.target, self.cfg.doc_features)
                if not ok or ran < 1:
                    problems.append(f"{spec.path}: previous-tests {ref} does not pass as it is ({ran} run): {out}")
            if problems:
                return [], problems
            lines = path.read_text(encoding="utf-8").split("\n")
            block = "\n".join(lines[begin - 1:end])
            changed = remove_value(block, old)
            if changed is None:
                return [], [f"{spec.path}: {old[:8]} is not written in {where}:{begin}-{end} ({list_name}) as a hex string, "
                            f"32 byte literals or [0xNN; 32], so the tool cannot take it out to test the claim"]
            lines[begin - 1:end] = changed.split("\n")
            path.write_text("\n".join(lines), encoding="utf-8")
            for ref in spec.previous_tests:
                ok, ran, out = run_named_test(copy_crate, ref, self.target, self.cfg.doc_features)
                if ok:
                    problems.append(f"{spec.path}: previous-tests {ref} still passes with {old[:8]} taken out of "
                                    f"{list_name} ({where}): it does not depend on the old value")
                elif ran < 1:
                    # a build that breaks is not a test that tells the old value apart
                    problems.append(f"{spec.path}: previous-tests {ref} did not run with {old[:8]} taken out of "
                                    f"{list_name} (the copy does not build): {out}")
                else:
                    passed_refs.append(ref)
        finally:
            shutil.rmtree(cleanup, ignore_errors=True)
        return passed_refs, problems


def judge(base_version: str, head_version: str, base: dict[str, dict | None],
          head: dict[str, dict | None], changelog: str, cfg: Config,
          verify_previous=None) -> tuple[list[str], list[str]]:
    """(problems, lines of what was compared); `verify_previous(spec, old)` returns (tests that
    passed with the old value and failed without it, problems)"""
    problems, report = [], []
    for spec in cfg.ids:
        now, then = head.get(spec.path), base.get(spec.path)
        if now is None:
            problems.append(f"{spec.path}: not found in the working tree ({CONFIG} names it)")
            continue
        if then is None:
            report.append(f"{spec.path}: new since the base ({now['id'][:8]})")
            continue
        if now["id"] == then["id"]:
            report.append(f"{spec.path}: unchanged ({now['id'][:8]})")
            continue
        old, new = then["id"], now["id"]
        in_previous = bool(spec.previous) and old in (now.get("previous") or [])
        if in_previous:
            # same bits claimed: no bump, but the old generation must still be tested
            if verify_previous is None:
                refs, why = [], [f"{spec.path}: the move keeps {old[:8]} in {spec.previous}, and nothing tested the claim"]
            else:
                refs, why = verify_previous(spec, old)
            problems.extend(why)
            report.append(f"{spec.path}: moved {old[:8]} -> {new[:8]}, the old value kept in {spec.previous}"
                          + (f" and exercised by {', '.join(refs)}" if refs else ""))
            if not refs and not why:
                problems.append(f"{spec.path}: moved {old[:8]} -> {new[:8]} keeping {old[:8]} in {spec.previous}, "
                                f"but no previous-tests entry showed it depends on the old value")
        else:
            report.append(f"{spec.path}: moved {old[:8]} -> {new[:8]}")
        announced = any(old[:8] in e and new[:8] in e for e in breaking_entries(changelog))
        if not in_previous and not is_breaking(base_version, head_version) and announced:
            report.append(f"{spec.path}: announced by a **Breaking:** entry of [Unreleased] naming "
                          f"{old[:8]} and {new[:8]} (the version does not move while publication is frozen)")
        elif not in_previous and not is_breaking(base_version, head_version):
            fix = next_breaking(base_version)
            if head_version == base_version:
                problems.append(f"{spec.path}: moved {old[:8]} -> {new[:8]} at {head_version}, the version of the "
                                f"release it is compared with: bump the version (set version = \"{fix}\" in "
                                f"{cfg.manifest}), or name {old[:8]} and {new[:8]} in a **Breaking:** entry of "
                                f"CHANGELOG.md's [Unreleased]")
            else:
                problems.append(f"{spec.path}: moved {old[:8]} -> {new[:8]} but the version {base_version} -> "
                                f"{head_version} is not a breaking step (a major, or a minor in 0.x): set "
                                f"version = \"{fix}\" in {cfg.manifest}, or name {old[:8]} and {new[:8]} in a "
                                f"**Breaking:** entry of CHANGELOG.md's [Unreleased]")
        missing = [h[:8] for h in (old, new) if h[:8] not in changelog]
        if missing:
            problems.append(f"{spec.path}: moved, and CHANGELOG.md does not name {', '.join(missing)}")
    return problems, report


def check(root: Path, base_ref: str | None, base_from: str = "auto",
          main_ref: str = MAIN_REF) -> tuple[list[str], list[str]]:
    cfg = load_config(root)
    missing = completeness(cfg, id_constants((root / cfg.manifest).resolve().parent, cfg, root))
    problems, report = check_values(root, base_ref, base_from, main_ref)
    return missing + problems, report


def check_values(root: Path, base_ref: str | None, base_from: str = "auto",
                 main_ref: str = MAIN_REF) -> tuple[list[str], list[str]]:
    """judged against every base (`choose_bases`; one when `base_ref` names it): a problem against
    any of them fails, and with two bases each problem names the one it came from"""
    cfg = load_config(root)
    bases = [(base_ref, "tag")] if base_ref is not None else choose_bases(root, cfg, base_from, main_ref)
    problems, report = [], []
    for ref, kind in bases:
        p, r = check_against(root, ref, kind, base_from, main_ref)
        if len(bases) > 1:
            label = f"[{kind} base {ref if kind == 'tag' else (ref or '')[:12]}] "
            p = [label + x for x in p]
        problems += p
        report += r
    return problems, report


def check_against(root: Path, base_ref: str | None, kind: str, base_from: str = "auto",
                  main_ref: str = MAIN_REF) -> tuple[list[str], list[str]]:
    """`base_from`: "registry" reads the base version as published (what G9 is about: a path
    dependency of the tag's tree on a sibling repository would resolve against today's sibling,
    not the released one), "tag" builds the tag's tree, "auto" the registry and the tag when the
    version was never published"""
    cfg = load_config(root)
    head_version = package_version((root / cfg.manifest).read_text(encoding="utf-8"), cfg.package)
    changelog_path = (root / cfg.manifest).parent / "CHANGELOG.md"
    if not changelog_path.exists():
        changelog_path = root / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    work = Path(tempfile.mkdtemp(prefix="id-move-"))
    target = Path(os.environ.get("ID_MOVE_TARGET_DIR") or (work / "target"))
    if kind == "introduced":
        try:
            head = read_ids(root, cfg, target)
            problems, report = judge(head_version, head_version, {}, head, changelog, cfg)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        report.insert(0, f"no base: {cfg.package} has no version on {main_ref} before its first, working tree "
                         f"{head_version}")
        return problems, report
    base_version = package_version(git(root, "show", f"{base_ref}:{cfg.manifest}"), cfg.base_pkg)
    if kind == "bump":
        # the move goes out in HEAD's version: it is named where that release is described
        changelog = changelog_sections(changelog, head_version)
    source = base_ref
    if kind == "tag" and base_from in ("registry", "auto"):
        try:
            base = read_ids(base_version, cfg, target, with_previous=False, is_base=True)
            head = read_ids(root, cfg, target)
            source = f"{cfg.base_pkg} {base_version} from the registry"
        except NotPublished:
            if base_from == "registry":
                shutil.rmtree(work, ignore_errors=True)
                raise
            base = None
    else:
        base = None
    if base is not None:
        try:
            problems, report = judge(base_version, head_version, base, head, changelog, cfg,
                                    PreviousVerifier(root, cfg, _DOC.get(str(root)), target))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        report.insert(0, f"base {base_ref}: {source}, working tree {head_version}")
        return problems, report
    # the base tree sits under a fresh parent with the repository's own name, next to each sibling
    # it reaches by path dependency at the commit the base tree pins, so `../ALICE-SDF` resolves
    # to what the base was built against
    fresh = root.resolve().parent / f".id-move-base-{os.getpid()}"
    wt = fresh / root.resolve().name
    siblings: list[str] = []
    try:
        # every file of the tag, written through a throwaway index: the repository's own index and
        # checkout are not touched, nothing is registered, and unlike `git archive` no file marked
        # export-ignore in .gitattributes is left out of the build
        wt.mkdir(parents=True)
        env = dict(os.environ, GIT_INDEX_FILE=str(work / "base.index"))
        for args in (["read-tree", base_ref], ["checkout-index", "--all", "--force", f"--prefix={wt}/"]):
            r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, env=env)
            if r.returncode != 0:
                raise CheckError(f"git {args[0]} {base_ref}: {r.stderr.strip()}")
        siblings = materialize_pinned_siblings(wt, root.resolve().parent, work, root, base_ref)
        base = read_ids(wt, cfg, target, with_previous=False, is_base=True)
        shutil.rmtree(fresh, ignore_errors=True)
        head = read_ids(root, cfg, target)
        problems, report = judge(base_version, head_version, base, head, changelog, cfg,
                                PreviousVerifier(root, cfg, _DOC.get(str(root)), target))
    finally:
        shutil.rmtree(fresh, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    what = ("the commit before the newest version bump on " + main_ref) if kind == "bump" else "the tag's tree"
    report.insert(0, f"base {base_ref}: {what} ({base_version}), working tree {head_version}")
    report.extend(f"base {base_ref} sibling {line}" for line in siblings)
    return problems, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--base")
    ap.add_argument("--base-from", choices=["auto", "registry", "tag", "bump"], default="auto")
    ap.add_argument("--main", default=MAIN_REF, help="the ref of main (a bump base is looked for on it)")
    ap.add_argument("--print-base", action="store_true",
                    help="print <package>-<version> of the base release (a cache key for its .crate) and exit")
    ap.add_argument("--print-siblings", action="store_true",
                    help="print '<Repo> <SHA>' for every sibling commit a base tree may build against and exit")
    a = ap.parse_args(argv)
    try:
        if a.print_siblings:
            cfg = load_config(Path(a.root))
            bases = [(a.base, "tag")] if a.base is not None else choose_bases(Path(a.root), cfg, a.base_from, a.main)
            for repo, sha in base_sibling_pins(Path(a.root), [ref for ref, kind in bases if kind != "introduced"]):
                print(f"{repo} {sha}")
            return 0
        if a.print_base:
            cfg = load_config(Path(a.root))
            ref, kind = (a.base, "tag") if a.base is not None else choose_base(Path(a.root), cfg, a.base_from, a.main)
            if kind == "introduced":
                print(f"{cfg.base_pkg}-none")
            else:
                print(f"{cfg.base_pkg}-{package_version(git(Path(a.root), 'show', f'{ref}:{cfg.manifest}'), cfg.base_pkg)}")
            return 0
        problems, report = check(Path(a.root), a.base, a.base_from, a.main)
    except RegistryUnreachable as e:
        if unreadable_base_policy(dict(os.environ)) == "warn":
            print(f"id_move_check: WARNING (not checked on this ref): {e}", file=sys.stderr)
            return 0
        print(f"id_move_check: {e}", file=sys.stderr)
        return 1
    except CheckError as e:
        print(f"id_move_check: {e}", file=sys.stderr)
        return 1
    for line in report:
        print(f"id_move_check: {line}")
    for p in problems:
        print(f"id_move_check: {p}", file=sys.stderr)
    if problems:
        return 1
    compared = [r for r in report if not r.startswith(("base ", "no base"))]
    if not compared:
        print("id_move_check: compared no identifier", file=sys.stderr)
        return 1
    # one header line per base ("base <ref>: ..." / "no base ..."); a sibling line names its base too
    bases = sum(1 for r in report if r.startswith("no base") or re.match(r"base \S+: ", r))
    print(f"id_move_check: {len(compared)} identifier comparison(s) against {bases} base(s), 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
