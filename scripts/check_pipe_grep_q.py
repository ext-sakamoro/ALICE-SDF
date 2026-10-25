#!/usr/bin/env python3
"""No shell pipeline ends in `grep -q` in the scripts and workflows

Under `set -o pipefail`, `cmd | grep -q pat` fails when grep exits on its first match while `cmd`
is still writing: `cmd` dies of SIGPIPE (exit 141) and the pipeline reports failure although the
line was found. It shows only under load (measured: 1 in 2000 runs with the CPU busy), so a gate
written that way fails at random, or, behind `||`, takes the wrong branch at random. Read the
output in full first: `grep -q pat <<<"$(cmd)"`.

    python3 scripts/check_pipe_grep_q.py          # check the tracked .sh files and workflows

Fails when it reads no file, so it never passes vacuously.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
# a pipe into grep whose options include q (-q, -qx, -xq, --quiet), on one logical line
PATTERN = re.compile(r"\|\s*grep\b[^|;&#]*?(?:\s-[A-Za-z]*q[A-Za-z]*\b|\s--quiet\b|\s--silent\b)")


def offending(text: str) -> list[tuple[int, str]]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        code = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
        if PATTERN.search(code):
            out.append((n, line.strip()))
    return out


def tracked() -> list[pathlib.Path]:
    names = subprocess.run(
        ["git", "ls-files", "--", "*.sh", ".github/workflows/*.yml", ".github/workflows/*.yaml"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return [ROOT / n for n in names]


def main() -> int:
    files = tracked()
    if not files:
        print("check_pipe_grep_q: read no file", file=sys.stderr)
        return 1
    bad = []
    for f in files:
        for n, line in offending(f.read_text(encoding="utf-8")):
            bad.append(f"{f.relative_to(ROOT).as_posix()}:{n}: {line}")
    for b in bad:
        print(f"check_pipe_grep_q: {b}", file=sys.stderr)
    print(f"check_pipe_grep_q: {len(files)} files read, {len(bad)} pipelines end in grep -q")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
