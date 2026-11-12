#!/usr/bin/env python3
"""Which installed Unreal engines the Unreal workflow builds against for one run.

usage: unreal_engines.py --roots JSON --event NAME --ref REF --specific BOOL

The workflow's one self-hosted Windows runner builds the plugin against every engine in
UE5_ENGINE_ROOTS in series (about 50 minutes each). Building every engine for every ci/ push
held the landing queue for hours, so:

  push to main (post-land), schedule (weekly)   every engine
  push to ci/**, the change is engine-specific  every engine
  push to ci/**, otherwise                      the newest engine only

"Engine-specific" is the `run` output of scripts/changed_paths.py decide over
.github/unreal-engines-both.txt (the plugin, the C header, the HLSL emitters, the Unreal
corpus example and this workflow); a failed decision there is run=true (fail-closed), so
every engine. A change that is not engine-specific but still breaks only an older engine is
caught by the post-land run on main and by the weekly run, which build every engine.

The newest engine is the root with the highest `major.minor` in its name (5.8 over 5.7); a
root without one counts as older than any that has one, and ties keep the later entry.

Output: `engines` (a JSON list, a subset of the roots in their order) in $GITHUB_OUTPUT, or
on stdout without it. The log names engine versions only, never an install path (the job log
is public). Exit 1 when the roots are not a non-empty JSON list of strings (nothing would be
built), or when an argument is missing or unknown.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

EVERY = "every"
NEWEST = "newest"


def parse_roots(text: str) -> list[str]:
    try:
        roots = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"UE5_ENGINE_ROOTS is not JSON: {e}") from e
    if not isinstance(roots, list) or not roots or not all(isinstance(r, str) and r for r in roots):
        raise ValueError("UE5_ENGINE_ROOTS must be a non-empty JSON list of non-empty strings")
    return roots


def version(root: str) -> tuple[int, int] | None:
    found = re.findall(r"(\d+)\.(\d+)", root)
    if not found:
        return None
    major, minor = found[-1]
    return int(major), int(minor)


def label(root: str) -> str:
    v = version(root)
    return f"UE {v[0]}.{v[1]}" if v else "engine (no version in its name)"


def newest(roots: list[str]) -> str:
    best, best_key = roots[0], (0, (0, 0))
    for r in roots:
        v = version(r)
        key = (1, v) if v else (0, (0, 0))
        if key >= best_key:
            best, best_key = r, key
    return best


def truthy(v: str) -> bool:
    if v not in ("true", "false"):
        raise ValueError(f"--specific must be true or false, got {v!r}")
    return v == "true"


def choose(event: str, ref: str, specific: bool) -> tuple[str, str]:
    """(EVERY or NEWEST, the reason)"""
    if event == "schedule":
        return EVERY, "scheduled run"
    if event != "push":
        raise ValueError(f"unexpected event {event!r} (the workflow runs on push and schedule)")
    if ref == "refs/heads/main":
        return EVERY, "post-land run on main"
    if specific:
        return EVERY, "the change touches an engine-specific path"
    return NEWEST, "the change is not engine-specific (every engine runs after landing and weekly)"


def select(roots: list[str], event: str, ref: str, specific: bool) -> tuple[list[str], str]:
    which, why = choose(event, ref, specific)
    if which == EVERY:
        return list(roots), why
    return [newest(roots)], why


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--roots", required=True)
    p.add_argument("--event", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--specific", required=True)
    a = p.parse_args(argv)
    try:
        roots = parse_roots(a.roots)
        chosen, why = select(roots, a.event, a.ref, truthy(a.specific))
    except ValueError as e:
        print(f"unreal_engines: {e}", file=sys.stderr)
        return 1
    print(f"unreal_engines: {len(chosen)} of {len(roots)}: "
          f"{', '.join(label(r) for r in chosen)} ({why})")
    line = f"engines={json.dumps(chosen)}"
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    else:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
