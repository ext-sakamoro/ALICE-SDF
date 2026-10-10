#!/usr/bin/env python3
"""scripts/fuzz_runs.py の試験 (libFuzzer の log の形は実際の CI run の出力から)"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuzz_runs as fr  # noqa: E402

CLEAN = """INFO: Running with entropic power schedule (0xFF, 100).
#1682447\tDONE   cov: 1495 ft: 2772 corp: 653/84Kb lim: 4096 exec/s: 27581 rss: 636Mb
Done 1682447 runs in 61 second(s)
"""
CRASH = """INFO: A corpus is not provided, starting from an empty corpus
thread '<unnamed>' (3084) panicked at library/core/src/num/f32.rs:1665:9:
min > max, or either was NaN. min = 0.0, max = -65.0
==3084== ERROR: libFuzzer: deadly signal
SUMMARY: libFuzzer: deadly signal
"""


class FuzzRuns(unittest.TestCase):
    def run_on(self, log: str) -> int:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "fuzz.log")
            p.write_text(log, encoding="utf-8")
            return fr.main(["fuzz_lz77_decode", str(p)])

    def test_a_clean_run_passes(self):
        self.assertEqual(fr.read(CLEAN), (1682447, 61, False))
        self.assertEqual(self.run_on(CLEAN), 0)

    def test_a_crash_fails(self):
        self.assertTrue(fr.read(CRASH)[2])
        self.assertEqual(self.run_on(CRASH), 1)

    def test_a_crash_after_runs_fails(self):
        self.assertEqual(self.run_on(CLEAN + CRASH), 1)

    def test_each_crash_line_alone_fails(self):
        for line in ["==1== ERROR: libFuzzer: deadly signal", "SUMMARY: libFuzzer: timeout",
                     "thread 'x' panicked at src/a.rs:1:1:"]:
            with self.subTest(line=line):
                self.assertEqual(self.run_on(CLEAN + line + "\n"), 1)

    def test_zero_runs_fails(self):
        self.assertEqual(self.run_on("Done 0 runs in 0 second(s)\n"), 1)

    def test_no_done_line_fails(self):
        self.assertEqual(self.run_on("INFO: Seed: 1\n"), 1)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(fr.main(["t", str(Path(d, "missing.log"))]), 1)


if __name__ == "__main__":
    unittest.main()
