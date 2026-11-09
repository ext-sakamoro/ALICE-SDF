#!/usr/bin/env python3
"""Run a command and fail when its peak resident memory passes a budget.

    python3 scripts/peak_rss.py --max-mib N --what TEXT -- CMD [ARG...]

While the command runs, the resident sets of its whole process tree are summed every 0.2 s (from
/proc on Linux, from `ps` elsewhere). When the sum passes the budget the tree is killed at once and
the line `peak-rss: TEXT: N MiB is over the budget of M MiB at S s` is printed, exit 3: the budget
fires before the machine runs out of memory, so a runner that would be shut down reports the
number instead. After the command ends, the peak is the larger of the sampled sum and the largest
resident set of a finished child (getrusage RUSAGE_CHILDREN: KiB on Linux, bytes on macOS), which
catches a child shorter than one sample. Prints one line `peak-rss: TEXT N MiB in S s (budget M
MiB)` and exits with the command's code, 3 when the peak passes the budget, 4 when nothing was
measured (a peak of 0 means the command did not run, which must not pass as within budget).

Why: a GPU test whose shader takes more memory to compile than the runner holds does not fail as a
test; the hosted runner is shut down ("The runner has received a shutdown signal", exit 143) and the
cause is not in the log. A budget well under the runner's memory makes the regression a red line.
"""

from __future__ import annotations

import argparse
import os
import resource
import signal
import subprocess
import sys
import time

OVER_BUDGET = 3
NOTHING_MEASURED = 4
SAMPLE_S = 0.2


def peak_mib() -> float:
    """the largest resident set of the finished children, in MiB"""
    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    per_unit = 1.0 if sys.platform == "darwin" else 1024.0
    return raw * per_unit / (1024.0 * 1024.0)


def processes() -> dict[int, tuple[int, float]]:
    """pid -> (parent pid, resident MiB) of every process that can be read"""
    out: dict[int, tuple[int, float]] = {}
    if os.path.isdir("/proc/self"):
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/status", encoding="ascii", errors="replace") as f:
                    ppid, rss = 0, 0.0
                    for line in f:
                        if line.startswith("PPid:"):
                            ppid = int(line.split()[1])
                        elif line.startswith("VmRSS:"):
                            rss = int(line.split()[1]) / 1024.0
                out[int(d)] = (ppid, rss)
            except (OSError, ValueError, IndexError):
                continue
        return out
    r = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        f = line.split()
        if len(f) == 3 and all(x.isdigit() for x in f):
            out[int(f[0])] = (int(f[1]), int(f[2]) / 1024.0)
    return out


def tree_mib(root: int) -> float:
    """the summed resident sets of `root` and all its descendants"""
    procs = processes()
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _) in procs.items():
        kids.setdefault(ppid, []).append(pid)
    total, todo, seen = 0.0, [root], set()
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += procs.get(pid, (0, 0.0))[1]
        todo.extend(kids.get(pid, []))
    return total


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
    # its own session, so the whole tree is killed at the budget
    child = subprocess.Popen(cmd, start_new_session=True)
    sampled = 0.0
    while True:
        try:
            rc = child.wait(timeout=SAMPLE_S)
            break
        except subprocess.TimeoutExpired:
            pass
        now = tree_mib(child.pid)
        sampled = max(sampled, now)
        if now > a.max_mib:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            print(
                f"peak-rss: {a.what}: {now:.0f} MiB is over the budget of {a.max_mib:.0f} MiB "
                f"at {time.time() - t:.1f} s (the command was killed)",
                flush=True,
            )
            return OVER_BUDGET
    peak = max(sampled, peak_mib())
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
