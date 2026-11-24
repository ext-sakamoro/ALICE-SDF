#!/usr/bin/env python3
"""Every tool a workflow installs is pinned to an exact version, built or fetched without the repository's toolchain.

A tool installed without a version takes whatever was released last: the day a new release needs a newer rustc
than the repository pins, the install fails (cargo-deny 0.22.0 needs rustc 1.95, a repository pinning 1.92 cannot
build it), and an old tool kept by accident can stop reading the current advisory data and check less. So, in every
workflow file:

- a `cargo install` or `cargo binstall` names an exact version (`--version X.Y.Z`, `--version =X.Y.Z` or
  `crate@X.Y.Z`) for each crate it installs; a `cargo install` also names a pinned toolchain (`cargo +X.Y.Z
  install`): without one the tool is compiled by the repository's toolchain (rust-toolchain.toml), and a floating
  `+stable` / `+beta` / `+nightly` moves on its own. `--path` (a crate of the repository) is exempt; `--git` needs
  `--rev <40 hex>`.
- a step that uses taiki-e/install-action is pinned to a commit (40 hex), each entry of its `tool:` input is
  `name@X.Y.Z`, and its `fallback:` input is `none` (the action's default falls back to cargo-binstall, which
  installs neither the pinned version's checked binary nor checks it against the action's manifest).
  One form of version is taken from the lock instead of written out: `name@${{ steps.<id>.outputs.<key> }}`, where
  the step <id> of the same job reads Cargo.lock, selects the package by its exact name (`/^name = "<name>"$/` or
  `$0=="name = \"<name>\""`, not a substring that also matches `<name>-macro`), stops on an empty value
  (`set -e` with `test -n` / `[ -n ]`, or `[ -n ... ] || ... exit N`) and writes <key>= to $GITHUB_OUTPUT: a CLI
  that must equal the crate the lock resolves (wasm-bindgen) is pinned by the lock.

Workflows are read with the strict reader of scripts/changed_paths.py. Fails on any finding, and when no workflow
was read.

usage: workflow_tool_pins.py [workflows dir]

Shared byte-identical by the ALICE crates that carry it (this file and its test, ALICE-Law's shared table); change
every copy together.
"""
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import changed_paths as cp
except ImportError:  # the reader is required: no fallback to reading lines
    cp = None

EXACT = re.compile(r"=?\d+\.\d+\.\d+")
TOOLCHAIN = re.compile(r"\+\d+\.\d+\.\d+")
SHA = re.compile(r"[0-9a-f]{40}")
INSTALL_ACTION = "taiki-e/install-action"
# options of cargo install / binstall that take a value (the value is not a crate name)
VALUED = {"--version", "--vers", "--git", "--branch", "--tag", "--rev", "--path", "--root", "--index", "--registry",
          "--features", "-F", "--target", "--target-dir", "--profile", "-j", "--jobs", "--config", "-Z", "--bin",
          "--example", "--color", "--install-path", "--min-tls-version", "--pkg-url", "--pkg-fmt", "--bin-dir",
          "--strategies", "--disable-strategies", "--manifest-path", "--maximum-resolution-timeout"}


def commands(run: str) -> list[list[str]]:
    """the cargo install / binstall commands of a run script, one token list each (continued lines joined)"""
    text = re.sub(r"\\\n", " ", run)
    out = []
    for line in text.splitlines():
        for part in re.split(r"&&|\|\||;|\|", line):
            try:
                toks = shlex.split(part, comments=True)
            except ValueError:
                toks = part.split()
            while toks and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", toks[0]):
                toks = toks[1:]
            if len(toks) >= 2 and toks[0] == "cargo":
                rest = toks[1:]
                chain = rest[0] if rest[0].startswith("+") else None
                if chain:
                    rest = rest[1:]
                if rest and rest[0] in ("install", "binstall"):
                    out.append(["cargo"] + ([chain] if chain else []) + rest)
    return out


def check_command(toks: list[str]) -> list[str]:
    chain = toks[1] if toks[1].startswith("+") else None
    verb_at = 2 if chain else 1
    verb, args = toks[verb_at], toks[verb_at + 1:]
    shown = " ".join(toks)
    crates, version, path, git, rev = [], None, False, False, None
    i = 0
    while i < len(args):
        a = args[i]
        key, _, inline = a.partition("=") if a.startswith("--") else (a, "", "")
        if key in VALUED:
            value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
            i += 1 if inline else 2
            if key in ("--version", "--vers"):
                version = value
            elif key == "--path":
                path = True
            elif key == "--git":
                git = True
            elif key == "--rev":
                rev = value
            continue
        if not a.startswith("-"):
            crates.append(a)
        i += 1
    if path:
        return []
    found = []
    if verb == "install" and not (chain and TOOLCHAIN.fullmatch(chain)):
        found.append(f"`{shown}`: builds with " + (f"the floating toolchain {chain}" if chain else
                     "the repository's toolchain") + " (name one: cargo +X.Y.Z install)")
    if git:
        if not (rev and SHA.fullmatch(rev)):
            found.append(f"`{shown}`: --git without --rev <40 hex>")
        return found
    if not crates:
        found.append(f"`{shown}`: no crate named")
    for c in crates:
        name, at, v = c.partition("@")
        if at:
            if not EXACT.fullmatch(v):
                found.append(f"`{shown}`: {name}@{v} is not an exact version")
        elif not (version and EXACT.fullmatch(version)):
            found.append(f"`{shown}`: {name} has no exact version (--version X.Y.Z or {name}@X.Y.Z)")
    if version and len(crates) > 1:
        found.append(f"`{shown}`: one --version for several crates")
    return found


FROM_STEP = re.compile(r"\$\{\{\s*steps\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)\s*\}\}")


def lock_pinned(name: str, ref: str, steps: list) -> str | None:
    """None when `name@<ref>` takes its version from a step that reads it from Cargo.lock as required, else why not"""
    m = FROM_STEP.fullmatch(ref)
    if not m:
        return "is not name@X.Y.Z"
    sid, key = m.groups()
    src = [s for s in steps if isinstance(s, dict) and s.get("id") == sid]
    if len(src) != 1 or not isinstance(src[0].get("run"), str):
        return f"names step {sid!r}, which is not one run step of this job"
    run = src[0]["run"]
    q = re.escape(name)
    exact = (re.search(r"/\^name = \"" + q + r"\"\$/", run)
             or re.search(r'==\s*"name = \\"' + q + r'\\""', run))
    empty_stops = (re.search(r"\[\s*-n\s+\"?\$\{?\w+\}?\"?\s*\]\s*\|\|[^\n]*exit\s+[1-9]", run)
                   or (re.search(r"(^|\n)\s*set\s+-[a-z]*e", run)
                       and re.search(r"(^|\n)\s*(test\s+-n|\[\s*-n)\s+\"?\$\{?\w+\}?\"?\s*\]?\s*($|\n)", run)))
    if "Cargo.lock" not in run:
        return f"step {sid!r} does not read Cargo.lock"
    if not exact:
        return f"step {sid!r} does not select {name} by its exact name in Cargo.lock"
    if not empty_stops:
        return f"step {sid!r} does not stop on an empty version"
    if not re.search(re.escape(key) + r"=[^\n]*GITHUB_OUTPUT", run):
        return f"step {sid!r} does not write {key}= to GITHUB_OUTPUT"
    return None


def check_action(step: dict, steps: list | None = None) -> list[str]:
    uses = step.get("uses", "")
    ref = uses.split("@", 1)[1] if "@" in uses else ""
    found = []
    if not SHA.fullmatch(ref):
        found.append(f"{uses}: not pinned to a commit (40 hex)")
    with_ = step.get("with") if isinstance(step.get("with"), dict) else {}
    tools = [t.strip() for t in re.split(r"[,\n]", str(with_.get("tool", ""))) if t.strip()]
    if not tools:
        found.append(f"{uses}: no tool input")
    for t in tools:
        name, at, v = t.partition("@")
        if at and EXACT.fullmatch(v):
            continue
        why = lock_pinned(name, v, steps or []) if at else "is not name@X.Y.Z"
        if why:
            found.append(f"{uses}: tool {t!r} {why}")
    if str(with_.get("fallback", "")).strip() != "none":
        found.append(f"{uses}: fallback is not none (the default installs through cargo-binstall, unchecked)")
    return found


def check_workflow(data) -> list[str]:
    found = []
    jobs = data.get("jobs") if isinstance(data, dict) else None
    for jid, job in (jobs or {}).items():
        steps = job.get("steps") if isinstance(job, dict) else None
        for n, step in enumerate(steps or []):
            if not isinstance(step, dict):
                continue
            where = f"jobs.{jid}.steps[{n}]"
            if str(step.get("uses", "")).split("@", 1)[0] == INSTALL_ACTION:
                found += [f"{where}: {m}" for m in check_action(step, steps)]
            run = step.get("run")
            if isinstance(run, str):
                for toks in commands(run):
                    found += [f"{where}: {m}" for m in check_command(toks)]
    return found


def main(argv: list[str]) -> int:
    if cp is None:
        print("error: scripts/changed_paths.py (the workflow reader) is missing", file=sys.stderr)
        return 1
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / ".github" / "workflows"
    files = sorted(list(root.glob("*.yml")) + list(root.glob("*.yaml")))
    findings = []
    for f in files:
        try:
            data = cp.load_yaml(f.read_text(encoding="utf-8"))
        except cp.FormError as e:
            findings.append(f"{f.name}: {e} (read with the strict reader of scripts/changed_paths.py)")
            continue
        findings += [f"{f.name}: {m}" for m in check_workflow(data)]
    for m in findings:
        print(m)
    if not files:
        print("error: no workflow read (compared nothing)", file=sys.stderr)
        return 1
    if findings:
        return 1
    print(f"ok: {len(files)} workflows read, every installed tool pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
