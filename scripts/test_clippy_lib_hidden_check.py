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



def at(path: str, file: str, line: int) -> dict:
    m = msg(path, file)
    m["spans"][0]["line_start"] = line
    return m


class StoredForm(unittest.TestCase):
    SITES = {"src/io/json.rs": [(10, 14), (30, 33)]}

    def test_reads_inside_the_reviewed_sites_pass(self):
        errors, n = h.judge_stored([at("serde_json::de::from_str", "src/io/json.rs", 12),
                                    at("serde_json::value::from_value", "src/io/json.rs", 30),
                                    at("f32::sin", "src/io/json.rs", 50)], self.SITES)
        self.assertEqual((errors, n), ([], 2))

    def test_a_read_outside_a_site_is_hidden(self):
        for path in ("serde_json::de::from_reader", "bincode::serde::decode_from_slice"):
            with self.subTest(path=path):
                errors, _ = h.judge_stored([at("serde_json::de::from_str", "src/io/json.rs", 12),
                                            at("serde_json::de::from_str", "src/io/json.rs", 31),
                                            at(path, "src/interval.rs", 12)], self.SITES)
                self.assertEqual(len(errors), 1, errors)
                self.assertIn("src/interval.rs:12", errors[0])

    def test_a_site_with_no_read_left_fails(self):
        errors, _ = h.judge_stored([at("serde_json::de::from_str", "src/io/json.rs", 12)], self.SITES)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("src/io/json.rs:30", errors[0])

    def test_the_real_tree_has_sites(self):
        import bare_serde_reads
        files, found = bare_serde_reads.scan(Path(__file__).resolve().parent.parent)
        self.assertGreater(files, 0)
        self.assertGreater(len(found), 0)


if __name__ == "__main__":
    unittest.main()
