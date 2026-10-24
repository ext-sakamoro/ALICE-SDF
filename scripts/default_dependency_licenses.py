#!/usr/bin/env python3
"""Every dependency of the default build has a licence the crate's own declared licence can take.

The crate's licence is the `license` of the root package as `cargo metadata --locked` reads it
from the manifest of the checked-out commit (the lock fixes every dependency's version, and with
it the licence expression each one declares). The policy follows that declaration, so changing
the crate's `license` changes what is required without an edit here:

  * a permissive declaration (an expression with an alternative made only of PERMISSIVE
    licences, e.g. `Apache-2.0` or `MIT`): every reached dependency must be permissive too;
  * `LicenseRef-Proprietary`: every reached dependency must be allowed by the deny policy of
    deny.toml (`[licenses] allow`, the `[[licenses.exceptions]]` of that crate, the
    `[[licenses.clarify]]` expression that replaces a crate's own), or be
    `LicenseRef-Proprietary` itself;
  * any other declaration: no policy, which fails.

The graph is walked from the root along normal and build dependencies (not dev) with the
default features. A package without a licence expression fails, and so does an expression with
no alternative whose every term is accepted. Reaching no package other than the crate itself
compares nothing and fails (exit 2), as does a root package without a `license`.

usage: python3 scripts/default_dependency_licenses.py [--metadata FILE] [--deny FILE]
  --metadata FILE   read this `cargo metadata --format-version 1` output instead of running cargo
  --deny FILE       read this deny policy instead of deny.toml at the repository root
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERMISSIVE = {
    "MIT", "MIT-0", "Apache-2.0", "Apache-2.0 WITH LLVM-exception", "BSD-2-Clause", "BSD-3-Clause",
    "ISC", "Zlib", "Unicode-3.0", "Unicode-DFS-2016", "BSL-1.0", "CC0-1.0", "Unlicense", "0BSD",
}
PROPRIETARY = "LicenseRef-Proprietary"


@dataclass
class DenyPolicy:
    """the licence part of deny.toml"""
    allow: set[str] = field(default_factory=set)
    exceptions: dict[str, set[str]] = field(default_factory=dict)
    clarify: dict[str, str] = field(default_factory=dict)


def read_deny(path: Path) -> DenyPolicy:
    lic = tomllib.loads(path.read_text(encoding="utf-8")).get("licenses", {})
    exceptions: dict[str, set[str]] = {}
    for e in lic.get("exceptions", []):
        exceptions.setdefault(e["name"], set()).update(e.get("allow", []))
    return DenyPolicy(set(lic.get("allow", [])), exceptions,
                      {c["name"]: c["expression"] for c in lic.get("clarify", []) if c.get("expression")})


def alternatives(expr: str) -> list[list[str]]:
    """The expression as OR of AND terms (parentheses flattened; `/` as OR, the old cargo form)"""
    e = expr.replace("/", " OR ").replace("(", " ").replace(")", " ")
    out = []
    for alt in re.split(r"\s+OR\s+", e.strip()):
        terms = [t.strip() for t in re.split(r"\s+AND\s+", alt.strip()) if t.strip()]
        out.append(terms)
    return out


def accepted(expr: str | None, ok: set[str]) -> bool:
    """some alternative of the expression has every term in `ok`"""
    if not expr:
        return False
    return any(terms and all(t in ok for t in terms) for terms in alternatives(expr))


def permissive(expr: str | None) -> bool:
    return accepted(expr, PERMISSIVE)


def reached(meta: dict) -> list[dict]:
    """Packages reached from the root along normal and build dependencies"""
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen = {root}
    todo = [root]
    while todo:
        n = nodes[todo.pop()]
        for d in n.get("deps", []):
            kinds = {k.get("kind") for k in d.get("dep_kinds", [])}
            if kinds & {None, "build"} and d["pkg"] not in seen:
                seen.add(d["pkg"])
                todo.append(d["pkg"])
    seen.discard(root)
    return [pkgs[i] for i in sorted(seen)]


def root_licence(meta: dict) -> str | None:
    root = meta["resolve"]["root"]
    return next((p.get("license") for p in meta["packages"] if p["id"] == root), None)


def check(meta: dict, deny: DenyPolicy) -> tuple[list[str], int, str]:
    """(problems, dependencies compared, the policy applied)"""
    own = root_licence(meta)
    if not own:
        return ["the root package declares no licence (nothing to compare with)"], 0, "none"
    pkgs = reached(meta)
    if permissive(own):
        policy = f"{own} (permissive): permissive dependencies only"
        ok_for = lambda name: PERMISSIVE  # noqa: E731
    elif own == PROPRIETARY:
        policy = f"{own}: what deny.toml allows, or {PROPRIETARY}"
        ok_for = lambda name: deny.allow | deny.exceptions.get(name, set()) | {PROPRIETARY}  # noqa: E731
    else:
        return [f"the root package declares {own!r}, for which this check has no policy"], len(pkgs), "none"
    problems = []
    for p in pkgs:
        expr = deny.clarify.get(p["name"], p.get("license")) if own == PROPRIETARY else p.get("license")
        if not accepted(expr, ok_for(p["name"])):
            problems.append(f"{p['name']} {p['version']}: licence {p.get('license')!r} is not accepted for a "
                            f"crate declared {own!r}")
    return problems, len(pkgs), policy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata")
    ap.add_argument("--deny", default=str(ROOT / "deny.toml"))
    a = ap.parse_args(argv)
    if a.metadata:
        meta = json.loads(Path(a.metadata).read_text(encoding="utf-8"))
    else:
        r = subprocess.run(["cargo", "metadata", "--format-version", "1", "--locked"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"default-dependency-licenses: cargo metadata failed: {r.stderr.strip()[:300]}", file=sys.stderr)
            return 2
        meta = json.loads(r.stdout)
    try:
        deny = read_deny(Path(a.deny))
    except (OSError, tomllib.TOMLDecodeError, KeyError) as e:
        print(f"default-dependency-licenses: cannot read the deny policy {a.deny}: {e}", file=sys.stderr)
        return 2
    problems, n, policy = check(meta, deny)
    if n == 0:
        for p in problems:
            print(p, file=sys.stderr)
        print("default-dependency-licenses: reached no dependency (compared nothing)", file=sys.stderr)
        return 2
    for p in problems:
        print(p)
    print(f"default-dependency-licenses: {n} packages in the default build, policy {policy}, "
          f"{len(problems)} not accepted")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
