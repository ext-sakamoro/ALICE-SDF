#!/usr/bin/env python3
"""The configured workspace members depend on no source of entropy or wall-clock time (ALICE-Law G10).

A Law-bearing value must not depend on an unseeded RNG, a freshly generated UUID or an ambient
wall-clock read. Rust code can only call a crate its manifest depends on directly, so this checks
the manifests instead of scanning text: no configured member may list a denied crate as a normal
or build dependency, under any target. Dev-dependencies are the tests' and do not reach the
member's library.

What is checked comes from a per-repository file, `.github/entropy-deps.toml` next to the
manifest (or `--config`), so this script holds no crate name of its own and is the same file in
every repository that uses it:

    members = ["<package>", ...]   # the members whose output bears the Law
    deny = ["<crate or glob>", ...]  # sources of entropy or wall-clock time, a reason each
    features = "all"               # "all": an optional dependency counts under any feature
                                   # "default": only when the member's default features enable it

A `deny` entry is a crate name or a glob (`rand_*`) matched against the whole name, so a longer
name that only contains a denied word (`timely`, `operand`) is another crate. A config that is
missing, has no member or no denied crate, or a `features` value other than the two, is exit 2; a
configured member that is not a workspace member is a finding (it would otherwise pass
unchecked).

It reads `cargo metadata --no-deps` (no resolution, no network). It fails when a configured member
lists a denied crate, and when no configured member is read (compared nothing).

It also reports, without failing, the denied crates reachable from the configured members through
normal and build dependencies with the default features (`cargo metadata --locked --offline`, the
resolution of the lock): a library the crate uses may draw entropy or read the clock inside, which
the direct-dependency rule cannot see. That line is a count to read, not a verdict: a crate in the
graph is not necessarily called on a path that feeds a Law-bearing value. When the resolution cannot
be read (no lock, a cache miss offline) the line says so.

    python3 scripts/entropy_deps_guard.py [--manifest-path Cargo.toml] [--config .github/entropy-deps.toml]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

FEATURES = ("all", "default")


class ConfigError(Exception):
    """a config this guard cannot use"""


@dataclass(frozen=True)
class Config:
    members: tuple[str, ...]
    deny: tuple[str, ...]
    features: str = "all"

    def denies(self, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, pat) for pat in self.deny)


def read_config(path: Path) -> Config:
    if not path.is_file():
        raise ConfigError(f"{path}: no config (the members and the denied crates are read from it)")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    members, deny = data.get("members"), data.get("deny")
    features = data.get("features", "all")
    if not isinstance(members, list) or not members or not all(isinstance(m, str) and m for m in members):
        raise ConfigError(f"{path}: `members` must be a non-empty list of package names")
    if not isinstance(deny, list) or not deny or not all(isinstance(d, str) and d for d in deny):
        raise ConfigError(f"{path}: `deny` must be a non-empty list of crate names or globs")
    if features not in FEATURES:
        raise ConfigError(f"{path}: `features` must be one of {', '.join(FEATURES)}, not {features!r}")
    unknown = sorted(set(data) - {"members", "deny", "features"})
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {', '.join(unknown)}")
    return Config(tuple(members), tuple(deny), features)


def default_enabled(pkg: dict) -> set[str]:
    """the optional dependencies (by the name the manifest gives them) the default features enable"""
    feats = pkg.get("features", {})
    optional = {d.get("rename") or d["name"] for d in pkg.get("dependencies", []) if d.get("optional")}
    seen, todo, out = set(), ["default"], set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        if f not in feats:
            if f in optional:
                out.add(f)  # (an optional dependency's implicit feature)
            continue
        for item in feats[f]:
            if item.startswith("dep:"):
                out.add(item[4:])
            elif "/" in item:
                dep, _ = item.split("/", 1)
                if not dep.endswith("?") and dep in optional:
                    out.add(dep)
            else:
                todo.append(item)
    return out


def counts(dep: dict, pkg: dict, config: Config, enabled: set[str]) -> bool:
    """whether this dependency is part of the build the config checks"""
    if dep.get("kind") == "dev":
        return False
    if not dep.get("optional") or config.features == "all":
        return True
    return (dep.get("rename") or dep["name"]) in enabled


def denied_deps(metadata: dict, config: Config) -> tuple[list[str], list[str]]:
    """(the configured members read, one problem per denied dependency)"""
    members = set(metadata.get("workspace_members", []))
    read, problems = [], []
    for pkg in metadata.get("packages", []):
        if pkg.get("id") not in members or pkg.get("name") not in config.members:
            continue
        read.append(pkg["name"])
        enabled = default_enabled(pkg)
        for dep in pkg.get("dependencies", []):
            if not counts(dep, pkg, config, enabled) or not config.denies(dep["name"]):
                continue
            name = dep.get("rename") or dep["name"]
            where = f" (target {dep['target']})" if dep.get("target") else ""
            kind = dep.get("kind") or "normal"
            how = " (optional, enabled by the default features)" if dep.get("optional") and config.features == "default" else ""
            problems.append(f"{pkg['name']}: {kind} dependency `{name}` ({dep['name']}){where}{how} is a source of "
                            "entropy or wall-clock time (ALICE-Law G10): take the randomness / time as an input")
    return read, problems


def members_differ(read: list[str], config: Config) -> list[str]:
    """one problem per configured member that was not read as a workspace member"""
    return [f"{n} is configured but was not read as a workspace member" for n in sorted(set(config.members) - set(read))]


def transitive(metadata: dict, config: Config) -> list[str]:
    """the denied crates reachable from the configured members through normal and build dependencies"""
    members = set(metadata.get("workspace_members", []))
    pkgs = {p["id"]: p for p in metadata.get("packages", [])}
    nodes = {n["id"]: n for n in (metadata.get("resolve") or {}).get("nodes", [])}
    todo = [i for i in members if pkgs.get(i, {}).get("name") in config.members]
    seen, found = set(todo), set()
    while todo:
        node = nodes.get(todo.pop())
        if node is None:
            continue
        for d in node.get("deps", []):
            if all(k.get("kind") == "dev" for k in d.get("dep_kinds", [])):
                continue
            if d["pkg"] in seen:
                continue
            seen.add(d["pkg"])
            name = pkgs.get(d["pkg"], {}).get("name", "")
            if config.denies(name):
                found.add(f"{name} {pkgs[d['pkg']].get('version', '')}")
            todo.append(d["pkg"])
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-path", default="Cargo.toml")
    ap.add_argument("--config", help="default: .github/entropy-deps.toml next to the manifest")
    a = ap.parse_args(argv)
    config_path = Path(a.config) if a.config else Path(a.manifest_path).resolve().parent / ".github" / "entropy-deps.toml"
    try:
        config = read_config(config_path)
    except ConfigError as e:
        print(f"entropy-deps-guard: {e}", file=sys.stderr)
        return 2
    p = subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1", "--manifest-path", a.manifest_path],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"entropy-deps-guard: cargo metadata failed:\n{p.stderr}", file=sys.stderr)
        return 2
    read, problems = denied_deps(json.loads(p.stdout), config)
    problems += members_differ(read, config)
    for e in problems:
        print(f"error: {e}", file=sys.stderr)
    if not read:
        print("entropy-deps-guard: no configured workspace member read (compared nothing)", file=sys.stderr)
        return 2
    print(f"entropy-deps-guard: {len(read)} configured member(s) ({', '.join(sorted(read))}), {len(config.deny)} "
          f"denied pattern(s), features {config.features}, "
          f"{len(problems)} problem{'' if len(problems) == 1 else 's'}")
    full = subprocess.run(["cargo", "metadata", "--format-version", "1", "--locked", "--offline",
                           "--manifest-path", a.manifest_path], capture_output=True, text=True)
    if full.returncode != 0:
        print("entropy-deps-guard: transitive (informational, not a check): the resolution could not be read offline "
              "from the lock here (no registry cache), so no count is reported")
    else:
        reach = transitive(json.loads(full.stdout), config)
        print(f"entropy-deps-guard: transitive (report only): {len(reach)} denied crate(s) reachable through "
              f"normal / build dependencies with the default features{': ' + ', '.join(reach) if reach else ''}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
