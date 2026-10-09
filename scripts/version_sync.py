#!/usr/bin/env python3
"""Check that every version this repository states follows the crate version.

`Cargo.toml` `[package] version` is the one source. Two kinds of statement are
checked against it, so a release that bumps the crate and forgets one of them
fails CI instead of leaving an old number behind:

  * metadata  every tracked file that declares a version of its own
              (`package.json`, `pyproject.toml`, `*.uplugin`, a non-root
              `Cargo.toml`, a Python `__version__`, `*.podspec`,
              `build.gradle(.kts)`, `*.csproj` / `*.nuspec`) must be listed in
              scripts/version-sync.toml, either under `follow` (its version must
              equal the crate version) or under `independent` with a reason
              (a separately versioned package). A version-bearing file that is
              in neither list is an error, so a new one cannot drift unseen.
  * install   in every tracked Markdown file except changelogs, a dependency
              line naming the crate (`<name> = "X"`, `<name> = { version = "X" }`,
              `pip install <name>==X`, `npm install <name>@X`) must accept the
              crate version (caret rule for Cargo, equality for pip / npm).

Historical statements ("added in v1.7.3", a changelog entry) are not checked:
they record a past version on purpose. A document that is a record of past
releases (a roadmap log, a migration guide from an old major) is listed under
`historical` in the config with a reason; its install lines are counted as
skipped, not checked.

The check fails when it compared nothing, so a broken discovery cannot pass.

Usage: `python3 scripts/version_sync.py --check` (exit 1 on any mismatch);
`--root DIR` runs against another tree (used by scripts/test_version_sync.py).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join("scripts", "version-sync.toml")

# file name patterns that can carry a version of their own
BEARERS = [
    "package.json", "pyproject.toml", "*.uplugin", "Cargo.toml", "__init__.py",
    "*.podspec", "build.gradle", "build.gradle.kts", "*.csproj", "*.nuspec",
]


def tracked(root: str) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return [l for l in out.stdout.splitlines() if l]


def read(root: str, rel: str) -> str:
    with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
        return f.read()


def toml_version(text: str, section: str) -> str | None:
    """`version = "X"` directly under `[section]` (stdlib-free, enough for these files)."""
    cur = None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"^\[([^\]]+)\]$", s)
        if m:
            cur = m.group(1).strip()
            continue
        if cur == section:
            m = re.match(r'^version\s*=\s*"([^"]+)"', s)
            if m:
                return m.group(1)
    return None


def declared_version(rel: str, text: str) -> str | None:
    """The version a file declares for itself, or None when it declares none."""
    base = os.path.basename(rel)
    if base == "package.json" or base.endswith(".uplugin"):
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        v = data.get("VersionName", data.get("version"))
        return v if isinstance(v, str) else None
    if base == "pyproject.toml":
        return toml_version(text, "project") or toml_version(text, "tool.poetry")
    if base == "Cargo.toml":
        return toml_version(text, "package")
    if base == "__init__.py":
        m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
        return m.group(1) if m else None
    if base.endswith(".podspec"):
        m = re.search(r'\.version\s*=\s*["\']([^"\']+)["\']', text)
        return m.group(1) if m else None
    if base.startswith("build.gradle"):
        m = re.search(r'versionName\s*[=(]?\s*["\']([^"\']+)["\']', text)
        return m.group(1) if m else None
    if base.endswith((".csproj", ".nuspec")):
        m = re.search(r"<(?:Version|version)>([^<]+)</", text)
        return m.group(1).strip() if m else None
    return None


def caret_ok(req: str, version: str) -> bool:
    """Whether Cargo's default (caret) requirement `req` accepts `version`."""
    def parts(v: str) -> list[int]:
        core = v.split("-", 1)[0].split("+", 1)[0]
        return [int(x) for x in core.split(".") if x.isdigit()]
    r, v = parts(req.lstrip("^")), parts(version)
    if not r or not v:
        return False
    r += [0] * (3 - len(r))
    v += [0] * (3 - len(v))
    if tuple(v) < tuple(r):
        return False
    # the first non-zero component of the requirement must match
    for i in range(3):
        if r[i] != 0 or i == 2:
            return v[: i + 1] == r[: i + 1]
    return True


def load_config(root: str) -> dict:
    text = read(root, CONFIG)
    if tomllib is not None:
        return tomllib.loads(text)
    raise SystemExit("version_sync needs Python 3.11+ (tomllib)")


def check(root: str) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts = {"metadata": 0, "install": 0, "historical": 0}
    cargo = read(root, "Cargo.toml")
    version = toml_version(cargo, "package")
    name = None
    m = re.search(r'^\[package\][^\[]*?^name\s*=\s*"([^"]+)"', cargo, re.M | re.S)
    if m:
        name = m.group(1)
    if not version or not name:
        return ["Cargo.toml has no [package] name / version"], counts
    cfg = load_config(root)
    follow = {e["file"] for e in cfg.get("follow", [])}
    independent = {e["file"]: e.get("reason", "") for e in cfg.get("independent", [])}
    historical = {e["file"]: e.get("reason", "") for e in cfg.get("historical", [])}
    for kind, entries in (("independent", independent), ("historical", historical)):
        for f, reason in entries.items():
            if not reason.strip():
                errors.append(f"{CONFIG}: {kind} entry {f} has no reason")
    files = tracked(root)
    present = set(files)
    for f in sorted(follow | set(independent) | set(historical)):
        if f not in present:
            errors.append(f"{CONFIG}: {f} is listed but not tracked (remove the entry)")
    for rel in files:
        base = os.path.basename(rel)
        if rel == "Cargo.toml" or not any(fnmatch.fnmatch(base, p) for p in BEARERS):
            continue
        declared = declared_version(rel, read(root, rel))
        if declared is None:
            continue
        if rel in follow:
            counts["metadata"] += 1
            if declared != version:
                errors.append(f"{rel}: version {declared} does not follow Cargo.toml {version}")
        elif rel not in independent:
            errors.append(
                f"{rel}: declares version {declared} but is not in {CONFIG} "
                f"(add it under `follow` or under `independent` with a reason)"
            )
    n = re.escape(name)
    cargo_pat = re.compile(n + r'\s*=\s*(?:"([^"]+)"|\{[^}\n]*?version\s*=\s*"([^"]+)")')
    pip_pat = re.compile(r"pip3?\s+install\s+[^\n]*?\b" + n.replace("\\-", "[-_]") + r"==([0-9][^\s\"'`]*)")
    npm_pat = re.compile(r"npm\s+(?:install|i)\s+[^\n]*?\b" + n + r"@([0-9][^\s\"'`]*)")
    for rel in files:
        if not rel.endswith(".md") or os.path.basename(rel).upper().startswith("CHANGELOG"):
            continue
        text = read(root, rel)
        if rel in historical:
            counts["historical"] += sum(1 for _ in cargo_pat.finditer(text))
            continue
        for m in cargo_pat.finditer(text):
            req = m.group(1) or m.group(2)
            counts["install"] += 1
            if not caret_ok(req, version):
                errors.append(f"{rel}: `{name} = \"{req}\"` does not accept {version}")
        for pat, kind in ((pip_pat, "pip"), (npm_pat, "npm")):
            for m in pat.finditer(text):
                counts["install"] += 1
                if m.group(1) != version:
                    errors.append(f"{rel}: {kind} pins {name} {m.group(1)}, the crate is {version}")
    if counts["metadata"] == 0:
        errors.append(f"compared no metadata file: {CONFIG} lists none under `follow`")
    return errors, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on any mismatch")
    ap.add_argument("--root", default=os.path.dirname(HERE))
    args = ap.parse_args(argv)
    errors, counts = check(args.root)
    for e in errors:
        print(f"error: {e}")
    print(
        f"compared: metadata {counts['metadata']}, install lines {counts['install']} "
        f"(skipped in historical documents: {counts['historical']})"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
