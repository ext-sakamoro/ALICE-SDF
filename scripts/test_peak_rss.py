#!/usr/bin/env python3
"""peak_rss.py measures the child, passes it within the budget, fails it over the budget (and kills
a tree that passes it while running, summed over grandchildren), and passes the command's own
failure through

    python3 scripts/test_peak_rss.py
"""

import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "peak_rss.py")
# a child that holds about 120 MiB
ALLOC = "b = bytearray(120 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096])"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


class PeakRss(unittest.TestCase):
    def test_within_the_budget_passes_and_reports_the_peak(self):
        r = run("--max-mib", "1000", "--what", "alloc", "--", sys.executable, "-c", ALLOC)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        mib = float(r.stdout.split("peak-rss: alloc ")[1].split(" MiB")[0])
        self.assertGreater(mib, 100, r.stdout)

    def test_over_the_budget_fails(self):
        r = run("--max-mib", "50", "--what", "alloc", "--", sys.executable, "-c", ALLOC)
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("over the budget", r.stdout)

    def test_a_child_over_the_budget_is_killed_while_it_runs(self):
        # holds the memory for 60 s: only the sampling can stop it in time
        hold = ALLOC + "; import time; time.sleep(60)"
        r = run("--max-mib", "50", "--what", "hold", "--", sys.executable, "-c", hold)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("over the budget of 50 MiB at", r.stdout)
        self.assertIn("(the command was killed)", r.stdout)
        at = float(r.stdout.split(" MiB at ")[1].split(" s")[0])
        self.assertLess(at, 30.0, r.stdout)

    def test_the_tree_sum_counts_grandchildren(self):
        # two grandchildren of 120 MiB each under a shell: the sum passes 200, neither alone does
        hold = ALLOC + "; import time; time.sleep(60)"
        sh = f"{sys.executable} -c \"{hold}\" & {sys.executable} -c \"{hold}\" & wait"
        r = run("--max-mib", "200", "--what", "tree", "--", "/bin/sh", "-c", sh)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("(the command was killed)", r.stdout)

    def test_a_child_within_the_budget_passes_with_a_nonzero_sampled_peak(self):
        hold = ALLOC + "; import time; time.sleep(1.5)"
        r = run("--max-mib", "1000", "--what", "hold", "--", sys.executable, "-c", hold)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        mib = float(r.stdout.split("peak-rss: hold ")[1].split(" MiB")[0])
        self.assertGreater(mib, 100, r.stdout)

    def test_the_command_failure_passes_through(self):
        r = run("--max-mib", "1000", "--what", "fail", "--", sys.executable, "-c", "import sys; sys.exit(7)")
        self.assertEqual(r.returncode, 7, r.stdout)


if __name__ == "__main__":
    unittest.main()
