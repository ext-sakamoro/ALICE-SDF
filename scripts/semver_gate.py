#!/usr/bin/env python3
"""cargo semver-checks against the latest crates.io release, once per consumer feature set.

A crate's public API may change only as far as the version in its Cargo.toml allows,
measured from the latest release on crates.io:

  - the step from that release to the Cargo.toml version is the one Cargo's compatibility
    rules name (the leftmost non-zero part is "major": 0.4 -> 0.5 is major, 0.4.1 -> 0.4.2
    is minor, 1.2.3 -> 1.2.4 is patch), see `version_step`
  - `cargo semver-checks check-release --baseline-version <release>` runs once per feature
    set of the consumer build (default features, none, and the declared consumer sets, or
    every feature at once when none are declared), see `feature_specs`
  - "every feature" is the feature universe of the package as Cargo defines it, read from
    `cargo metadata --no-deps`: the [features] table and the implicit feature of each
    optional dependency that no feature names as `dep:<name>` (an item behind
    `#[cfg(feature = "<optional dependency>")]` that no named feature enables is compared
    too), `default` left out; a metadata failure is an error, never a fallback to the
    [features] keys, see `feature_universe`. The same universe, with `feature_closure`
    (what a set of features enables), gives a coverage check the features no declared set
    reaches, see `uncovered_features` and `unknown_features`
  - a run is refused when the tool did not run, names a different step than the two
    versions make, prints no count of checks, compared 0 checks, or found a change the
    step does not allow (or exited non-zero), see `semver_verdict`
  - a major step is not compared (any change is allowed); a crate with no release has
    nothing to compare; a Cargo.toml version that is not newer than the release (SemVer
    2.0 precedence, so a pre-release below its release is refused) is refused, see
    `semver_problems`

Shared module. The functions `feature_specs`, `version_step`, `version_precedence`,
`semver_feature_args`, `run_semver_checks`, `semver_verdict`, `previous_release` and
`semver_problems` are those of ALICE-Law scripts/release_train.py (main eb0779d), with the
same names and results (except as listed below), so that file can import them from here. This file and
scripts/test_semver_gate.py hold no repository-specific name or path; the data of a
repository (its consumer feature sets, the equal-version mode) is a small TOML file
passed with --config. Keep every copy byte-identical and change them together.

Differences from release_train.py eb0779d (behaviour, not code layout):
  - `fetch` is typed `Fetch` from this module (release_train.py names it `dc.Fetch`, the
    same Callable[[str], dict]); `latest_release` and `RegistryError` live here, and
    `previous_release` raises this module's RegistryError on a body without a version,
    where release_train.py's `latest` raises its own RegistryError
  - `semver_problems` takes one more keyword argument, `equal_version`. Its default,
    "refuse", is release_train.py's behaviour unchanged: a Cargo.toml version equal to the
    latest release is refused. The release path (the publish gate) keeps that default and
    stays strict: the version being published must be newer than the latest release.
    "next-smallest-step" is an opt-in for a repository that keeps the released version in
    Cargo.toml while changes accumulate before the next release: an equal version is
    checked as the smallest next release (1.2.3 -> 1.2.4 patch, 0.5.1 -> 0.5.2 minor,
    0.0.3 -> 0.0.4 major, which is not compared), with that step passed to the tool as
    `--release-type` so it cannot infer "no change"; a change that needs a larger step
    fails until Cargo.toml carries that bump. A pre-release that equals the latest release
    has no defined smallest next release and is refused in this mode too
  - `http_fetch` does not retry an HTTP 4xx answer (a 404 is the answer "no such crate")
  - `feature_specs` and `semver_problems` take one more keyword argument, `universe` (the
    list `feature_universe` returns). With no declared consumer set the all-features set
    is that universe; without it they raise ValueError (eb0779d used the [features] keys,
    which miss the implicit feature of an optional dependency no feature names). With a
    declared set the universe is not used

The feature universe (shared with a coverage check, so "covered" and "checked" are the
same set of names):

  meta = cargo_metadata(manifest_path)               # fail-closed, MetadataError
  feats = features_from_metadata(meta, package, manifest_path)   # the package's map
  universe = universe_from_features(feats)           # sorted, `default` left out
  feature_universe(manifest_path, package=None)      # the three steps above
  feature_closure(feats, names)                      # names plus all they enable
  uncovered_features(universe, feats, sets, excluded)   # universe - closure - excluded
  unknown_features(feats, names)                     # names the package does not define
  coverage_refusals(universe, feats, entry)          # the CLI's coverage check (below)

`package` selects the package by name in a workspace (needed when the manifest is a
virtual workspace); without it the package whose manifest is `manifest_path` is taken.

CLI (exit 0 pass, 1 refused, 2 the gate could not decide; every failure path is closed):

  python3 scripts/semver_gate.py [--manifest Cargo.toml] [--config FILE] [--target-dir DIR]

  --config FILE   TOML with the keys of a release-train roster entry that this gate reads:
                  `consumer-features` (cargo build sets), `consumer-check-features`
                  (cargo check sets), `consumer-features-why` (required with either),
                  `exclude-features` (a table `feature = "why it is not built"`, a
                  non-empty reason each, only with declared sets; the shape of the
                  ALICE-Law roster key of the same name) and `equal-version` ("refuse",
                  the default, or "next-smallest-step"); any other key is an error; a set
                  may name only features of the universe (implicit ones included) or
                  `default`; `exclude-features` may name only features of the universe
                  (not `default`) and none that a declared set names
  Feature coverage: a declared set does not grow with the manifest, so when the config
  declares sets, every feature of the universe must be in the closure of one of them
  (`feature_closure`) or in `exclude-features`; each one that is not is refused with
  "feature <name> is in no consumer-features / consumer-check-features set and not in
  exclude-features" and the gate exits 1 before any run. With no declared set the
  all-features set is the universe, nothing is left out, and the check prints that it is
  not applicable.
  In the "next-smallest-step" mode an equal version is announced with the version that is
  checked ("x 0.5.1 equals the latest release 0.5.1: checking it as the next release 0.5.2
  (minor step)"): cargo-semver-checks 0.50 has no option to name the current version, so
  its own line still reads "v0.5.1 -> v0.5.1 (assume minor change)".
  The crate name and version are read from [package] of the manifest, the feature universe
  from `cargo metadata --no-deps` (a failure exits 2); the latest release
  from the crates.io API (`max_stable_version`, else `max_version`). A network error, an
  HTTP answer other than 200 / 404, or a body without a version exits 2.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

API = "https://crates.io/api/v1/crates"
USER_AGENT = "semver-gate (https://github.com/ext-sakamoro)"

Fetch = Callable[[str], dict]


class RegistryError(RuntimeError):
    """crates.io answered, but not with the version asked for"""


class ConfigError(ValueError):
    """the --config file or the manifest cannot be used"""


def http_fetch(path: str) -> dict:
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            time.sleep(1)  # crates.io asks for at most one request per second
            return data
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def latest_release(fetch: Fetch, crate: str) -> str:
    c = fetch(crate)["crate"]
    return c.get("max_stable_version") or c["max_version"]


class MetadataError(ConfigError):
    """`cargo metadata` failed, printed no JSON, or has no such package"""


def cargo_metadata(manifest_path: str | Path) -> dict:
    """`cargo metadata --no-deps --format-version 1 --manifest-path <manifest>`, parsed;
    MetadataError on any failure (there is no fallback to the manifest's [features])"""
    cmd = ["cargo", "metadata", "--no-deps", "--format-version", "1", "--manifest-path", str(manifest_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CARGO_TERM_COLOR": "never"})
    except OSError as e:
        raise MetadataError(f"cargo metadata could not run: {e}") from None
    if r.returncode != 0:
        tail = " / ".join(line.strip() for line in r.stderr.strip().splitlines()[-3:])
        raise MetadataError(f"cargo metadata failed for {manifest_path} (exit {r.returncode}): {tail}")
    try:
        meta = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise MetadataError(f"cargo metadata printed no JSON for {manifest_path}: {e}") from None
    if not isinstance(meta, dict) or not isinstance(meta.get("packages"), list):
        raise MetadataError(f"cargo metadata for {manifest_path} has no package list")
    return meta


def features_from_metadata(metadata: dict, package: str | None, manifest_path: str | Path | None = None) -> dict:
    """the `features` map of one package of `cargo metadata` output: the package named
    `package`, or (with no name) the one whose manifest is `manifest_path`; MetadataError when
    there is not exactly one"""
    pkgs = metadata.get("packages") or []
    if package is not None:
        found = [p for p in pkgs if p.get("name") == package]
        what = f"named {package}"
    elif manifest_path is not None:
        want = Path(manifest_path).resolve()
        found = [p for p in pkgs if p.get("manifest_path") and Path(p["manifest_path"]).resolve() == want]
        what = f"with the manifest {want}"
    else:
        raise MetadataError("features_from_metadata needs a package name or a manifest path")
    if len(found) != 1:
        raise MetadataError(f"cargo metadata has {len(found)} package(s) {what} (expected 1)")
    feats = found[0].get("features")
    if not isinstance(feats, dict) or not all(isinstance(k, str) and isinstance(v, list) for k, v in feats.items()):
        raise MetadataError(f"cargo metadata has no features map for the package {what}")
    return feats


def universe_from_features(features: dict) -> list[str]:
    """every feature of the map but `default`, sorted"""
    return sorted(f for f in features if f != "default")


def feature_universe(manifest_path: str | Path, package: str | None = None) -> list[str]:
    """every feature Cargo defines for the package (the [features] table and the implicit
    feature of each optional dependency no feature names as `dep:`), `default` left out,
    from `cargo metadata --no-deps`; MetadataError on any failure"""
    return universe_from_features(features_from_metadata(cargo_metadata(manifest_path), package, manifest_path))


def feature_closure(features: dict, names) -> set[str]:
    """`names` and every feature of this package they enable, through the `features` map of
    `cargo metadata`: `x` enables the feature x; `dep:x` enables the dependency only (no
    feature x exists when `dep:` names it); `x/y` enables the feature x when x is a feature of
    this package (an optional dependency's implicit feature); `x?/y` does not enable x"""
    seen, todo = set(), list(names)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for item in features.get(f, []):
            if item.startswith("dep:"):
                continue
            if "/" in item:
                x = item.split("/", 1)[0]
                if x.endswith("?") or x not in features:
                    continue
                todo.append(x)
            else:
                todo.append(item)
    return seen


def uncovered_features(universe: list[str], features: dict, sets, excluded) -> list[str]:
    """the features of the universe that no declared set enables (through `feature_closure`)
    and that are not excluded, sorted"""
    declared = set().union(*(set(s) for s in sets)) if sets else set()
    return sorted(set(universe) - feature_closure(features, declared) - set(excluded))


def unknown_features(features: dict, names) -> list[str]:
    """the names that are not features of the package (implicit ones and `default`, when it
    is defined, are features), sorted"""
    return sorted(set(names) - set(features))


def feature_specs(manifest: dict, entry: dict | None,
                  universe: list[str] | None = None) -> list[tuple[str, str, list[str] | None, str]]:
    """(label, dependency spec, features, cargo command) of each build: the default features,
    none, and either the roster's `consumer-features` sets (`cargo build`) and
    `consumer-check-features` sets (`cargo check`: compiled but not linked, for a feature whose
    library another tool links, such as a Python extension) or every feature of `universe`
    (from `feature_universe`; required when no set is declared) at once"""
    entry = entry or {}
    specs = [("default features", "", None, "build"), ("no default features", ", default-features = false", [], "build")]
    declared = [(fs, "build") for fs in entry.get("consumer-features", [])] + \
               [(fs, "check") for fs in entry.get("consumer-check-features", [])]
    if declared:
        for fs, cmd in declared:
            specs.append((f"features {fs}" + (" (cargo check)" if cmd == "check" else ""),
                          ", default-features = false, features = " + json.dumps(fs), fs, cmd))
    else:
        if universe is None:
            raise ValueError("feature_specs needs the feature universe (feature_universe) when no consumer set "
                             "is declared: the [features] keys miss implicit optional-dependency features")
        every = sorted(f for f in universe if f != "default")
        if every:
            specs.append(("all features", ", features = " + json.dumps(every), every, "build"))
    return specs


# This module parses cargo-semver-checks output (the `Checking ... change` claim, the
# `N checks:` count, the `--- failure` lints). Every run forces CARGO_TERM_COLOR=never
# (run_semver_checks) and the output is read through ANSI.sub (semver_verdict), and
# test_semver_gate.py checks a coloured sample; the declaration below is what
# scripts/cargo_output_parser_check.py reads where a repository has that checker
CARGO_OUTPUT_PARSER = True

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def version_step(previous: str, version: str) -> str:
    """the step from one release to the next as cargo-semver-checks names it: a change of
    the leftmost non-zero part is "major", of the next one "minor", of the last "patch"
    (Cargo's compatibility: 0.4 -> 0.5 is major, 0.4.1 -> 0.4.2 is minor)"""
    old = [int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", previous).groups()]
    new = [int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", version).groups()]
    first = next((i for i, x in enumerate(old) if x), 2)
    for i in range(3):
        if old[i] != new[i]:
            return "major" if i <= first else ("minor" if i == first + 1 else "patch")
    return "patch"


def version_precedence(version: str) -> tuple:
    """a sort key of SemVer 2.0 precedence (build metadata ignored): 1.0.0-rc.1 < 1.0.0,
    1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-beta, numeric identifiers below alphanumeric ones"""
    core, _, pre = version.split("+")[0].partition("-")
    nums = tuple(int(x) for x in core.split("."))
    if not pre:
        return nums, 1, ()
    ids = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return nums, 0, ids


def semver_feature_args(features: list[str] | None) -> list[str]:
    """None: the default features; a list (possibly empty): exactly those"""
    if features is None:
        return ["--default-features"]
    return ["--only-explicit-features"] + (["--features", ",".join(features)] if features else [])


def run_semver_checks(repo: Path, manifest_path: str, baseline: list[str], features: list[str] | None,
                      target_dir: Path) -> tuple[int, str]:
    """(exit code, stdout + stderr) of `cargo semver-checks check-release` for one feature set"""
    cmd = ["cargo", "semver-checks", "check-release", "--manifest-path", str(repo / manifest_path),
           *baseline, *semver_feature_args(features)]
    env = {**os.environ, "CARGO_TERM_COLOR": "never", "CARGO_TARGET_DIR": str(target_dir)}
    try:
        r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    except FileNotFoundError as e:
        return 127, str(e)
    return r.returncode, r.stdout + r.stderr


def semver_verdict(code: int, output: str, step: str, label: str) -> tuple[str | None, int]:
    """(problem or None, checks counted) of one cargo-semver-checks run whose two versions
    make `step`"""
    text = ANSI.sub("", output)
    claim = re.search(r"Checking \S+ v\S+ -> v\S+ \((?:assume )?(major|minor|patch) change\)", text)
    if claim is None:
        tail = " / ".join(line.strip() for line in text.strip().splitlines()[-3:])
        return f"{label}: cargo semver-checks did not run (exit {code}): {tail}", 0
    if claim.group(1) != step:
        return f"{label}: cargo semver-checks compared a {claim.group(1)} step, the versions make a {step} step", 0
    if step == "major":
        return None, 0
    # "223 checks: 223 pass, 31 skip" when every check passed, "196 checks: 195 pass, 1 fail, ..." otherwise
    checked = re.search(r"Checked \[[^\]]*\]\s+(\d+) checks?: (\d+) pass(?:, (\d+) fail)?", text)
    if checked is None:
        return f"{label}: cargo semver-checks printed no count of checks (exit {code})", 0
    total, failed = int(checked.group(1)), int(checked.group(3) or 0)
    if total == 0:
        return f"{label}: cargo semver-checks compared 0 checks for a {step} step", 0
    lints = re.findall(r"--- failure ([A-Za-z0-9_]+):", text)
    summary = re.search(r"Summary (.+)", text)
    if failed or lints or code != 0:
        return (f"{label}: a {step} version does not allow {', '.join(lints) or f'{failed} failed check(s)'}"
                f" ({summary.group(1).strip() if summary else f'exit {code}'})"), 0
    return None, 1


def previous_release(fetch: Fetch, crate: str) -> str | None:
    """the latest release on crates.io, None when the crate has none"""
    try:
        return latest_release(fetch, crate)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except (KeyError, TypeError) as e:
        raise RegistryError(f"{crate}: crates.io answered without a latest release ({e!r})") from None


EQUAL_VERSION_MODES = ("refuse", "next-smallest-step")


def next_patch(version: str) -> str:
    """the smallest release after `version` (a release, not a pre-release): x.y.z -> x.y.(z+1)"""
    major, minor, patch = (int(x) for x in re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version).groups())
    return f"{major}.{minor}.{patch + 1}"


def smallest_next_release(previous: str, version: str) -> tuple[str, str] | None:
    """(version checked, step) when `version` equals the release `previous` (SemVer
    precedence) and is not a pre-release: the smallest release after it and the step
    Cargo's compatibility names for it (0.5.1 -> 0.5.2 minor, 1.2.3 -> 1.2.4 patch,
    0.0.3 -> 0.0.4 major); None otherwise"""
    if version_precedence(previous) != version_precedence(version) or "-" in version.split("+")[0]:
        return None
    upcoming = next_patch(version.split("+")[0])
    return upcoming, version_step(previous, upcoming)


def semver_problems(repo: Path, manifest_path: str, crate: str, version: str, manifest: dict, entry: dict | None,
                    fetch: Fetch, run=run_semver_checks, target_dir: Path | None = None,
                    equal_version: str = "refuse", universe: list[str] | None = None) -> tuple[list[str], int, str]:
    """(problems, runs compared, what was compared) of cargo semver-checks against the latest
    release, once per feature set of the consumer build"""
    if equal_version not in EQUAL_VERSION_MODES:
        raise ValueError(f"equal_version must be one of {EQUAL_VERSION_MODES}, not {equal_version!r}")
    previous = previous_release(fetch, crate)
    if previous is None:
        return [], 0, f"{crate} has no release on crates.io: nothing to compare"
    baseline = ["--baseline-version", previous]
    note = ""
    if equal_version == "next-smallest-step" and version_precedence(previous) == version_precedence(version):
        if "-" in version.split("+")[0]:
            return [f"{crate}: the version {version} equals the latest release, a pre-release: "
                    f"no smallest next release is defined, set the next version in Cargo.toml"], 0, ""
        upcoming, step = smallest_next_release(previous, version)
        note = (f"version equals the published {previous}: checking the next release as the smallest step, "
                f"{upcoming} ({step}); ")
        if step == "major":
            return [], 0, note + "a major step: any change is allowed, not compared"
        baseline += ["--release-type", step]
    else:
        step = version_step(previous, version)
        if version_precedence(previous) >= version_precedence(version):
            return [f"{crate}: the latest release {previous} is not older than {version}"], 0, ""
        if step == "major":
            return [], 0, f"{previous} -> {version} is a major step: any change is allowed, not compared"
    problems, compared = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, _, features, _ in feature_specs(manifest, entry, universe):
            code, out = run(repo, manifest_path, baseline, features, target_dir or Path(tmp))
            p, n = semver_verdict(code, out, step, f"{crate} {version} ({label})")
            if p:
                problems.append(p)
            compared += n
    # every run either counts or is a problem (a run that compared 0 checks is refused above)
    return problems, compared, note + f"{previous} -> {version} ({step}), {compared} feature set(s) compared"


# ---------------------------------------------------------------- the CLI

CONFIG_KEYS = {"consumer-features", "consumer-check-features", "consumer-features-why", "exclude-features",
               "equal-version"}


def load_config(text: str) -> dict:
    """the --config file: the consumer feature sets, the features excluded from them and the
    equal-version mode"""
    c = tomllib.loads(text)
    unknown = sorted(set(c) - CONFIG_KEYS)
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} (known: {sorted(CONFIG_KEYS)})")
    for key in ("consumer-features", "consumer-check-features"):
        sets = c.get(key, [])
        if not isinstance(sets, list) or not all(isinstance(s, list) and all(isinstance(f, str) for f in s)
                                                 for s in sets):
            raise ConfigError(f"`{key}` must be a list of lists of feature names")
    declared = "consumer-features" in c or "consumer-check-features" in c
    if declared and not str(c.get("consumer-features-why", "")).strip():
        raise ConfigError("`consumer-features` needs `consumer-features-why` (how the sets were chosen)")
    if not declared and "consumer-features-why" in c:
        raise ConfigError("`consumer-features-why` without `consumer-features`")
    if "exclude-features" in c:
        ex = c["exclude-features"]
        if not declared:
            raise ConfigError("`exclude-features` without `consumer-features` (with no declared set every feature "
                              "is built at once, so there is nothing to exclude from)")
        if not isinstance(ex, dict) or not ex or not all(
                isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in ex.items()):
            raise ConfigError("`exclude-features` must be a table of feature = \"why it is not built\"")
        if "default" in ex:
            raise ConfigError("`exclude-features` may not name `default` (it is not in the feature universe)")
        in_sets = set().union(*(set(s) for s in c.get("consumer-features", []) + c.get("consumer-check-features", [])))
        both = sorted(set(ex) & in_sets)
        if both:
            raise ConfigError(f"{both} are both in `exclude-features` and in a declared set")
    if c.get("equal-version", "refuse") not in EQUAL_VERSION_MODES:
        raise ConfigError(f"`equal-version` must be one of {EQUAL_VERSION_MODES}")
    return c


def check_features_exist(features: dict, entry: dict) -> None:
    """every name of a declared set is a feature of the package (`features`, the map of
    `features_from_metadata`, so an implicit optional-dependency feature counts), and every
    name of `exclude-features` is in the feature universe"""
    for key in ("consumer-features", "consumer-check-features"):
        for s in entry.get(key, []):
            missing = unknown_features(features, s)
            if missing:
                raise ConfigError(f"`{key}` names feature(s) {missing} that the manifest does not define")
    missing = unknown_features({f: [] for f in universe_from_features(features)}, entry.get("exclude-features", {}))
    if missing:
        raise ConfigError(f"`exclude-features` names feature(s) {missing} that are not in the feature universe")


def coverage_refusals(universe: list[str], features: dict, entry: dict) -> list[str] | None:
    """None when the entry declares no consumer set (the all-features set is the universe, so
    there is nothing to check); otherwise one refusal per feature of the universe that no
    declared set reaches (`uncovered_features`) and that is not in `exclude-features`"""
    sets = entry.get("consumer-features", []) + entry.get("consumer-check-features", [])
    if not sets:
        return None
    return [f"feature {f} is in no consumer-features / consumer-check-features set and not in exclude-features"
            for f in uncovered_features(universe, features, sets, entry.get("exclude-features", {}))]


def echo_run(run):
    """a `run` that also prints the tool's output, for the job log"""
    def wrapped(repo, manifest_path, baseline, features, target_dir):
        code, out = run(repo, manifest_path, baseline, features, target_dir)
        print(f"::group::cargo semver-checks {' '.join(baseline)} {' '.join(semver_feature_args(features))} "
              f"(exit {code})")
        print(ANSI.sub("", out).rstrip())
        print("::endgroup::")
        return code, out
    return wrapped


def main(argv: list[str] | None = None, fetch: Fetch = http_fetch, run=run_semver_checks) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="Cargo.toml")
    ap.add_argument("--config", help="TOML with the consumer feature sets and the equal-version mode")
    ap.add_argument("--target-dir", help="CARGO_TARGET_DIR of the runs (default: a temporary directory)")
    a = ap.parse_args(argv)
    manifest_path = Path(a.manifest).resolve()
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        pkg = manifest.get("package", {})
        crate, version = pkg.get("name"), pkg.get("version")
        if not isinstance(crate, str) or not isinstance(version, str):
            raise ConfigError(f"{manifest_path}: [package] needs a literal name and version")
        entry = load_config(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
        features = features_from_metadata(cargo_metadata(manifest_path), crate, manifest_path)
        universe = universe_from_features(features)
        check_features_exist(features, entry)
    except (OSError, tomllib.TOMLDecodeError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    mode = entry.get("equal-version", "refuse")
    print(f"semver gate: {crate} {version}, equal-version mode: {mode}"
          + (" (a version equal to the latest release is refused)" if mode == "refuse"
             else " (a version equal to the latest release is checked as the smallest next release)"))
    print(f"  feature universe (cargo metadata, without default): {', '.join(universe) or '(none)'}")
    for label, _, fs, _ in feature_specs(manifest, entry, universe):
        print(f"  feature set: {label}" + (f" {fs}" if label == "all features" else ""))
    for f, why in sorted(entry.get("exclude-features", {}).items()):
        print(f"  excluded feature: {f} ({why})")
    uncovered = coverage_refusals(universe, features, entry)
    if uncovered is None:
        print("  feature coverage: not applicable (no consumer set is declared: the all-features set is the universe)")
    elif uncovered:
        for p in uncovered:
            print(f"refused: {p}")
        return 1
    else:
        print(f"  feature coverage: every feature of the universe is reached by a declared set or excluded "
              f"({len(universe)} compared)")
    answers: dict[str, dict] = {}

    def fetch_once(path: str) -> dict:
        # the registry is asked once: the target printed here is the one semver_problems checks
        if path not in answers:
            answers[path] = fetch(path)
        return answers[path]
    try:
        previous = previous_release(fetch_once, crate)
        target = smallest_next_release(previous, version) if previous and mode == "next-smallest-step" else None
        if target:
            print(f"semver gate: {crate} {version} equals the latest release {previous}: "
                  f"checking it as the next release {target[0]} ({target[1]} step)")
        problems, compared, note = semver_problems(
            manifest_path.parent, manifest_path.name, crate, version, manifest, entry, fetch_once,
            run=echo_run(run), target_dir=Path(a.target_dir).resolve() if a.target_dir else None,
            equal_version=mode, universe=universe)
    except (urllib.error.URLError, TimeoutError, RegistryError, json.JSONDecodeError) as e:
        print(f"error: the latest release of {crate} could not be read from crates.io: {e}", file=sys.stderr)
        return 2
    for p in problems:
        print(f"refused: {p}")
    if problems:
        return 1
    print(f"semver gate: pass: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
