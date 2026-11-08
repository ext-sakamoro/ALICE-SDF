#!/usr/bin/env python3
"""Run a command and fail when its peak resident memory passes a budget.

    python3 scripts/peak_rss.py --max-mib N --what TEXT -- CMD [ARG...]

The peak is the largest resident set of the command's children (getrusage RUSAGE_CHILDREN:
KiB on Linux, bytes on macOS). Prints one line `peak-rss: TEXT N MiB in S s (budget M MiB)` and
exits with the command's code, 3 when the peak passes the budget, 4 when nothing was measured (a
peak of 0 means the command did not run, which must not pass as within budget).

Why: a GPU test whose shader takes more memory to compile than the runner holds does not fail as a
test; the hosted runner is shut down ("The runner has received a shutdown signal", exit 143) and the
cause is not in the log. A budget well under the runner's memory makes the regression a red line.
"""

from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import time

OVER_BUDGET = 3
NOTHING_MEASURED = 4


def peak_mib() -> float:
    """the largest resident set of the finished children, in MiB"""
    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    per_unit = 1.0 if sys.platform == "darwin" else 1024.0
    return raw * per_unit / (1024.0 * 1024.0)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-mib", type=float, required=True)
    ap.add_argument("--what", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    t = time.time()
    rc = subprocess.call(cmd)
    peak = peak_mib()
    print(f"peak-rss: {a.what} {peak:.0f} MiB in {time.time() - t:.1f} s (budget {a.max_mib:.0f} MiB)", flush=True)
    if peak <= 0.0:
        print(f"peak-rss: {a.what}: nothing measured", flush=True)
        return NOTHING_MEASURED
    if peak > a.max_mib:
        print(f"peak-rss: {a.what}: {peak:.0f} MiB is over the budget of {a.max_mib:.0f} MiB", flush=True)
        return OVER_BUDGET
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
