#!/usr/bin/env python3
"""peak_rss.py measures the child, passes it within the budget, fails it over the budget, and
passes the command's own failure through

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

    def test_the_command_failure_passes_through(self):
        r = run("--max-mib", "1000", "--what", "fail", "--", sys.executable, "-c", "import sys; sys.exit(7)")
        self.assertEqual(r.returncode, 7, r.stdout)


if __name__ == "__main__":
    unittest.main()
