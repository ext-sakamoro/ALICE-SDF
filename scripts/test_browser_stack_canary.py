#!/usr/bin/env python3
"""Tests of the result check of scripts/browser_stack_canary.py (no browser)

Author: Moroya Sakamoto
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import browser_stack_canary as b  # noqa: E402


def data(eval_depth, compile_depth):
    return {
        "results": [
            {"name": "canary_eval", "deepest": eval_depth, "stopped": "RangeError: x"},
            {"name": "canary_compile", "deepest": compile_depth, "stopped": "RangeError: x"},
        ]
    }


class Check(unittest.TestCase):
    def test_both_deep_enough_pass(self):
        _, fails = b.check(data(2000, 5000), 1792)
        self.assertEqual(fails, [])

    def test_shallow_export_fails(self):
        _, fails = b.check(data(1096, 4552), 1792)
        self.assertEqual(len(fails), 1)
        self.assertIn("canary_eval: 1096 levels < 1792", fails[0])

    def test_known_defect_reported_not_failed(self):
        rep, fails = b.check(data(1096, 4552), 1792, frozenset({"canary_eval"}))
        self.assertEqual(fails, [])
        self.assertTrue(any("known defect" in r for r in rep))

    def test_fixed_known_defect_fails(self):
        _, fails = b.check(data(1792, 4552), 1792, frozenset({"canary_eval"}))
        self.assertEqual(len(fails), 1)
        self.assertIn("no longer reproduces", fails[0])

    def test_known_defect_does_not_cover_the_other_export(self):
        _, fails = b.check(data(1096, 1000), 1792, frozenset({"canary_eval"}))
        self.assertEqual(len(fails), 1)
        self.assertIn("canary_compile", fails[0])

    def test_only_checks_one_export(self):
        _, fails = b.check(data(1096, 4552), 1792, only="canary_compile")
        self.assertEqual(fails, [])
        _, fails = b.check({"results": [data(1096, 4552)["results"][1]]}, 1792, only="canary_eval")
        self.assertIn("canary_eval: not measured", fails[0])

    def test_nothing_measured_and_errors_fail(self):
        self.assertEqual(b.check(None, 1792)[1], ["no result from the background thread (timeout)"])
        self.assertIn("thread error", b.check({"error": "boom"}, 1792)[1][0])
        _, fails = b.check({"results": []}, 1792)
        self.assertEqual(len(fails), 2)


if __name__ == "__main__":
    unittest.main()
