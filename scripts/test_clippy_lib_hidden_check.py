#!/usr/bin/env python3
"""Tests for scripts/clippy_lib_hidden_check.py (the rewrite and the reading of
clippy's messages; the scratch-copy clippy run itself needs cargo and runs in CI
and preflight)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clippy_lib_hidden_check as h  # noqa: E402


def msg(path: str, file: str, code: str = "clippy::disallowed_methods") -> dict:
    return {"code": {"code": code}, "message": f"use of a disallowed method `{path}`",
            "spans": [{"is_primary": True, "file_name": file, "line_start": 7}]}


class Rewrite(unittest.TestCase):
    def test_every_allow_of_the_lint_is_lifted_and_counted(self):
        src = "#[allow(clippy::disallowed_methods, reason = \"x\")]\nfn a() {}\n#![allow(clippy::disallowed_methods)]\n"
        out, n = h.neutralise(src)
        self.assertEqual(n, 2)
        self.assertNotIn(h.LINT, out)
        self.assertIn(h.STAND_IN, out)


class Judge(unittest.TestCase):
    def test_a_max_outside_the_raw_file_is_reported(self):
        hidden, seen = h.judge([msg("f32::max", "src/npr/sky.rs")])
        self.assertEqual((len(hidden), seen), (1, 1))

    def test_each_forbidden_path_is_recognised(self):
        for p in ("f32::max", "f32::min", "f64::max", "f64::min", "wide::f32x8::max", "wide::f32x8::min"):
            self.assertEqual(len(h.judge([msg(p, "src/a.rs")])[0]), 1, p)

    def test_the_raw_file_itself_is_exempt(self):
        hidden, seen = h.judge([msg("f32::max", "src/raw_minmax.rs")])
        self.assertEqual((hidden, seen), ([], 1))

    def test_a_libm_diagnostic_is_seen_but_not_hidden_max(self):
        hidden, seen = h.judge([msg("f32::sin", "src/npr/sky.rs")])
        self.assertEqual((hidden, seen), ([], 1))

    def test_other_lints_are_not_counted(self):
        self.assertEqual(h.judge([msg("f32::max", "src/a.rs", code="clippy::use_self")]), ([], 0))


if __name__ == "__main__":
    unittest.main()
