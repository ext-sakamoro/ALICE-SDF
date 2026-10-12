#!/usr/bin/env python3
"""Tests for scripts/raw_minmax_manifest.py (on in-memory trees and manifests)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raw_minmax_manifest as r  # noqa: E402

DEFS = "\n".join(f"pub(crate) fn {h}(a: f32, b: f32) -> f32 {{ a }}" for h in r.HELPERS)

FORBIDDEN = '[forbidden]\nprefixes = ["src/compiled/", "src/interval.rs"]\n'


def manifest(**uses: int) -> str:
    out = FORBIDDEN
    for path, n in uses.items():
        out += f'\n[allowed."{path}"]\nuses = {n}\nwhy = "colour"\n'
    return out


def tree(**files: str) -> dict[str, str]:
    t = {r.DEFINITIONS: DEFS}
    t.update({k.replace("__", "/").replace("_rs", ".rs"): v for k, v in files.items()})
    return t


NPR = "src__npr__colour_rs"
NPR_PATH = "src/npr/colour.rs"


class Check(unittest.TestCase):
    def test_listed_with_the_right_count_passes(self):
        files = tree(**{NPR: "let a = raw_max_f32(x, y); let b = raw_min_f32(x, y);"})
        self.assertEqual(r.check(files, manifest(**{NPR_PATH: 2})), [])

    def test_a_different_count_fails(self):
        files = tree(**{NPR: "let a = raw_max_f32(x, y);"})
        self.assertTrue(r.check(files, manifest(**{NPR_PATH: 2})))

    def test_a_use_outside_the_manifest_fails(self):
        files = tree(**{NPR: "let a = raw_max_f32(x, y);"})
        self.assertTrue(any("not in the manifest" in p for p in r.check(files, manifest())))

    def test_an_entry_without_a_use_fails(self):
        files = tree(**{NPR: "let a = x;"})
        self.assertTrue(any("uses no raw helper" in p for p in r.check(files, manifest(**{NPR_PATH: 1}))))

    def test_a_use_under_a_forbidden_prefix_fails(self):
        files = tree(**{"src__compiled__refit_rs": "let a = raw_max_f32(x, y);"})
        self.assertTrue(any("forbidden prefix" in p for p in r.check(files, manifest())))

    def test_a_forbidden_file_in_the_manifest_fails_even_when_counted(self):
        files = tree(**{"src__interval_rs": "let a = raw_min_f32(x, y);"})
        problems = r.check(files, manifest(**{"src/interval.rs": 1}))
        self.assertTrue(any("listed as allowed but under a forbidden prefix" in p for p in problems))

    def test_a_renamed_import_fails(self):
        files = tree(**{NPR: "use crate::raw_minmax::raw_max_f32 as m; let a = m(x, y);"})
        self.assertTrue(any("another name" in p for p in r.check(files, manifest(**{NPR_PATH: 1}))))

    def test_a_helper_passed_as_a_value_counts(self):
        files = tree(**{NPR: "v.iter().copied().fold(0.0, raw_max_f32)"})
        self.assertEqual(r.check(files, manifest(**{NPR_PATH: 1})), [])

    def test_the_method_form_counts(self):
        files = tree(**{NPR: "let a = x.raw_max(y); let b = v.raw_min(w);"})
        self.assertEqual(r.check(files, manifest(**{NPR_PATH: 2})), [])

    def test_the_method_form_in_a_forbidden_file_fails(self):
        files = tree(**{"src__compiled__aabb_rs": "let a = x.raw_max(y);"})
        self.assertTrue(any("forbidden prefix" in p for p in r.check(files, manifest())))

    def test_a_law_call_is_not_a_raw_use(self):
        files = tree(**{NPR: "let a = x.law_max(y); let b = law_min_f32(x, y);"})
        self.assertEqual(r.check(files, manifest()), [])

    def test_a_comment_does_not_count(self):
        files = tree(**{NPR: "// raw_max_f32 is not used here\n/* raw_min_f32 */ let a = x;"})
        self.assertEqual(r.check(files, manifest()), [])

    def test_no_file_fails(self):
        self.assertTrue(r.check({}, manifest()))

    def test_missing_definitions_fail(self):
        files = tree()
        files[r.DEFINITIONS] = "pub(crate) fn raw_max_f32(a: f32, b: f32) -> f32 { a }"
        self.assertTrue(any("definitions" in p for p in r.check(files, manifest())))

    def test_an_empty_forbidden_list_is_refused(self):
        self.assertTrue(r.check(tree(), "[forbidden]\nprefixes = []\n"))

    def test_an_entry_without_why_is_refused(self):
        text = FORBIDDEN + f'\n[allowed."{NPR_PATH}"]\nuses = 1\n'
        self.assertTrue(r.check(tree(**{NPR: "raw_max_f32(a, b)"}), text))


class Ratchet(unittest.TestCase):
    def test_a_rise_without_a_new_why_fails(self):
        files = tree(**{NPR: "raw_max_f32(a, b); raw_min_f32(a, b);"})
        self.assertTrue(any("went up" in p for p in r.check(files, manifest(**{NPR_PATH: 2}),
                                                            base_text=manifest(**{NPR_PATH: 1}))))

    def test_a_rise_with_a_new_why_passes(self):
        files = tree(**{NPR: "raw_max_f32(a, b); raw_min_f32(a, b);"})
        new = manifest(**{NPR_PATH: 2}).replace('why = "colour"', 'why = "colour and the new tone curve"')
        self.assertEqual(r.check(files, new, base_text=manifest(**{NPR_PATH: 1})), [])

    def test_a_fall_passes(self):
        files = tree(**{NPR: "raw_max_f32(a, b);"})
        self.assertEqual(r.check(files, manifest(**{NPR_PATH: 1}), base_text=manifest(**{NPR_PATH: 2})), [])


class RealTree(unittest.TestCase):
    def test_the_repository_passes(self):
        self.assertEqual(r.main([]), 0)


if __name__ == "__main__":
    unittest.main()
