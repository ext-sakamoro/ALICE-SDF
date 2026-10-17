#!/usr/bin/env python3
"""Count the GPU unit tests that ran and that skipped, from a `cargo test -- --nocapture` log.

A GPU unit test that finds no usable device returns early and libtest reports it
`ok`, the same as a test that ran (`gpu_test_support` in src/lib.rs). Each such test prints
one `gpu-test: ran <name>` or `gpu-test: skipped <name>: <reason>` line instead; this
reads them back. With `--nocapture` libtest writes `test <name> ... ` before the
test's own output, so a marker can sit in the middle of a line, and escape codes are
removed first (scripts/ansi.py).

It fails when it finds no marker at all (the filter matched nothing, or the log was
captured: compared nothing), when fewer than `--ran` tests ran, when more than
`--max-skipped` were skipped, and when one name is marked twice.

usage: gpu_test_count.py <log> --ran N [--max-skipped M]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import ansi  # scripts/ansi.py

MARK = re.compile(r"gpu-test: (ran|skipped) ([A-Za-z0-9_]+)(?::\s*(.*))?")


def count(text: str) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """(names that ran, (name, reason) skipped, names marked more than once)"""
    ran: list[str] = []
    skipped: list[tuple[str, str]] = []
    for line in ansi.strip(text).splitlines():
        for m in MARK.finditer(line):
            if m.group(1) == "ran":
                ran.append(m.group(2))
            else:
                skipped.append((m.group(2), (m.group(3) or "").strip()))
    names = ran + [n for n, _ in skipped]
    twice = sorted({n for n in names if names.count(n) > 1})
    return ran, skipped, twice


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--ran", type=int, required=True)
    ap.add_argument("--max-skipped", type=int, default=None)
    a = ap.parse_args(argv)
    ran, skipped, twice = count(Path(a.log).read_text(encoding="utf-8", errors="replace"))
    print(f"gpu-test-count: {len(ran)} ran, {len(skipped)} skipped")
    for name, reason in skipped:
        print(f"gpu-test-count: skipped {name}: {reason}")
    problems = []
    if not ran and not skipped:
        problems.append("no gpu-test marker in the log (filter matched nothing, or output was captured): compared nothing")
    if len(ran) < a.ran:
        problems.append(f"{len(ran)} GPU tests ran, expected at least {a.ran}")
    if a.max_skipped is not None and len(skipped) > a.max_skipped:
        problems.append(f"{len(skipped)} GPU tests skipped, at most {a.max_skipped} allowed")
    if twice:
        problems.append(f"marked more than once: {twice}")
    for p in problems:
        print(f"::error::gpu-test-count: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
