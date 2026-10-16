#!/usr/bin/env python3
"""Tests for scripts/concurrent_tests.py (the reading of libtest's result line;
the concurrent run itself needs cargo and runs in CI and preflight)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concurrent_tests as c  # noqa: E402


class Parse(unittest.TestCase):
    def test_a_passing_run(self):
        self.assertEqual(c.parse("running 124 tests\ntest result: ok. 124 passed; 0 failed; 0 ignored"), (124, 0))

    def test_a_failing_run(self):
        self.assertEqual(c.parse("test result: FAILED. 120 passed; 4 failed; 0 ignored"), (120, 4))

    def test_no_result_line_is_not_a_pass(self):
        self.assertEqual(c.parse("error: could not compile"), (-1, -1))

    def test_a_coloured_summary_is_read(self):
        # libtest with --color always (measured form, see scripts/libtest_count.py)
        self.assertEqual(c.parse("test result: \x1b[32mok\x1b(B\x1b[m. 124 passed; 0 failed; 0 ignored"), (124, 0))
        self.assertEqual(c.parse("test result: \x1b[31mFAILED\x1b(B\x1b[m. 120 passed; 4 failed"), (120, 4))

    def test_zero_tests_is_reported_as_zero(self):
        self.assertEqual(c.parse("test result: ok. 0 passed; 0 failed; 0 ignored"), (0, 0))


if __name__ == "__main__":
    unittest.main()
