#!/usr/bin/env python3
"""Read a libFuzzer log: how many inputs ran, and whether it crashed.

The fuzz job is green only when the target ran at least one input and did not crash.
Before this check the time-boxed run step had `continue-on-error: true`, so a crash
showed only as a warning on a green job, and a run that never started was green too.

usage: fuzz_runs.py <target> <log>
Prints one Markdown table row (target, runs, seconds, result) for the job summary.
Exit 1 on a crash or when the log reports no run (0 runs, or no "Done N runs" line).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DONE = re.compile(r"^Done (\d+) runs in (\d+) second", re.M)
# libFuzzer's own crash lines, and Rust's panic message
CRASH = re.compile(r"^==\d+== ERROR: libFuzzer|^SUMMARY: libFuzzer|panicked at ", re.M)


def read(log: str) -> tuple[int, int, bool]:
    """(runs, seconds, crashed); runs is 0 when the log has no "Done" line"""
    done = DONE.findall(log)
    runs, secs = (int(done[-1][0]), int(done[-1][1])) if done else (0, 0)
    return runs, secs, bool(CRASH.search(log))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-3], file=sys.stderr)
        return 2
    target, path = argv
    log = Path(path).read_text(encoding="utf-8", errors="replace") if Path(path).is_file() else ""
    runs, secs, crashed = read(log)
    if crashed:
        result = "crash"
    elif runs == 0:
        result = "no run"
    else:
        result = "ok"
    print(f"| {target} | {runs} | {secs} | {result} |")
    if result != "ok":
        print(f"error: fuzz target {target}: {result} ({runs} runs)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
