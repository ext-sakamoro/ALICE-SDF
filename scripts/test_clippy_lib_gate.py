#!/usr/bin/env python3
"""Tests for scripts/clippy_lib_gate.py: the generated library configuration and
the reading of the canary's clippy output (the canary run itself needs cargo and
runs in CI and preflight through `python3 scripts/clippy_lib_gate.py`)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clippy_lib_gate as g  # noqa: E402

ROOT = """\
# libm
disallowed-methods = [
    { path = "f32::sin", reason = "libm" },
    { path = "f64::cos", reason = "libm" },
]

disallowed-types = [
    { path = "std::collections::HashMap", reason = "seed" },
]
"""

CANARY = """\
//! doc
pub fn a(a: f32, b: f32) -> f32 {
    a.max(b)
}
pub fn b(a: f32, b: f32) -> f32 {
    f32::min(a, b)
}
pub fn c(a: f64, b: f64) -> f64 {
    a.min(b)
}
pub fn d(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
pub fn e(x: f32) -> f32 {
    x.sin()
}
pub fn f(v: wide::f32x8, w: wide::f32x8) -> wide::f32x8 {
    v.max(w)
}
pub fn g(v: wide::f32x8, w: wide::f32x8) -> wide::f32x8 {
    v.min(w)
}
pub fn h() -> std::time::Instant {
    std::time::Instant::now()
}
pub fn i() -> std::time::SystemTime {
    std::time::SystemTime::now()
}
pub fn j(i: std::time::Instant) -> std::time::Duration {
    i.elapsed()
}
pub fn k(s: std::time::SystemTime) -> Result<std::time::Duration, std::time::SystemTimeError> {
    s.elapsed()
}
pub fn l() -> std::collections::hash_map::RandomState {
    std::collections::hash_map::RandomState::new()
}
"""


def diag(line: int, path: str) -> str:
    return f"src/lib.rs:{line}:5: error: use of a disallowed method `{path}`"


class Render(unittest.TestCase):
    def test_the_extra_entries_come_first_and_the_root_list_follows_unchanged(self):
        out = g.render(ROOT)
        self.assertTrue(out.startswith(g.HEADER))
        body = out[len(g.HEADER) + 1:]
        self.assertEqual(body.replace("".join(
            f'    {{ path = "{p}", reason = "{g.REASON}" }},\n' for p in g.EXTRA), ""), ROOT)
        i = body.index("disallowed-methods = [\n")
        self.assertIn('path = "f32::max"', body[i:body.index('path = "f32::sin"')])
        self.assertEqual(g.count_entries(out), 2 + len(g.EXTRA))

    def test_the_extra_paths_are_exactly_f32_f64_and_f32x8_max_and_min(self):
        self.assertEqual(g.EXTRA, ["f32::max", "f32::min", "f64::max", "f64::min",
                                   "wide::f32x8::max", "wide::f32x8::min"])

    def test_a_root_list_that_already_forbids_max_is_refused(self):
        with self.assertRaises(ValueError):
            g.render(ROOT.replace('"f64::cos"', '"f32::max"'))

    def test_a_root_without_the_list_is_refused(self):
        with self.assertRaises(ValueError):
            g.render(ROOT.replace("disallowed-methods", "disallowed-macros"))


class Check(unittest.TestCase):
    def test_the_generated_file_passes(self):
        self.assertEqual(g.check(ROOT, g.render(ROOT)), [])

    def test_a_missing_file_fails(self):
        self.assertTrue(g.check(ROOT, None))

    def test_a_dropped_extra_entry_fails(self):
        lib = g.render(ROOT).replace(f'    {{ path = "f64::min", reason = "{g.REASON}" }},\n', "")
        self.assertTrue(g.check(ROOT, lib))

    def test_a_libm_entry_added_to_the_root_only_fails(self):
        lib = g.render(ROOT)
        root = ROOT.replace('    { path = "f64::cos", reason = "libm" },\n',
                            '    { path = "f64::cos", reason = "libm" },\n    { path = "f32::exp", reason = "libm" },\n')
        self.assertTrue(g.check(root, lib))

    def test_an_empty_root_list_fails(self):
        root = 'disallowed-methods = [\n]\n'
        self.assertTrue(g.check(root, g.render(root)))


class Canary(unittest.TestCase):
    def setUp(self):
        self.want = g.canary_lines(CANARY)

    def test_the_twelve_forms_are_found_on_their_lines(self):
        self.assertEqual(sorted(self.want), [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36])

    def test_a_diagnostic_on_every_line_and_a_non_zero_exit_passes(self):
        out = "\n".join(diag(n, "x") for n in self.want)
        errors, n = g.judge_canary(101, out, self.want)
        self.assertEqual((errors, n), ([], 12))

    def test_a_missing_line_fails(self):
        out = "\n".join(diag(n, "x") for n in self.want if n != 9)
        errors, _ = g.judge_canary(101, out, self.want)
        self.assertEqual(len(errors), 1)
        self.assertIn("line 9", errors[0])

    def test_exit_zero_fails_even_with_diagnostics(self):
        out = "\n".join(diag(n, "x") for n in self.want)
        errors, _ = g.judge_canary(0, out, self.want)
        self.assertTrue(errors)

    def test_no_diagnostic_at_all_fails(self):
        errors, n = g.judge_canary(101, "error: could not compile", self.want)
        self.assertEqual(n, 0)
        self.assertTrue(any("compared nothing" in e for e in errors))

    def test_a_diagnostic_from_another_file_does_not_count(self):
        out = "\n".join(f"src/other.rs:{n}:5: error: use of a disallowed method `x`" for n in self.want)
        errors, n = g.judge_canary(101, out, self.want)
        self.assertEqual(n, 0)
        self.assertTrue(errors)

    def test_a_form_written_twice_in_the_canary_is_refused(self):
        with self.assertRaises(ValueError):
            g.canary_lines(CANARY + "pub fn f(a: f32, b: f32) -> f32 { a.max(b) }\n")

    def test_the_real_canary_has_each_form_once(self):
        src = (Path(g.CANARY) / "src" / "lib.rs").read_text(encoding="utf-8")
        self.assertEqual(len(g.canary_lines(src)), len(g.CANARY_FORMS))


class RealFiles(unittest.TestCase):
    def test_clippy_toml_bans_elapsed(self):
        # `.elapsed()` reads the wall clock exactly like `::now()`, but a
        # lexical pattern for it would be too broad (any value named
        # `elapsed` anywhere); clippy's disallowed-methods is type-resolved
        # and carries this one instead -- this only confirms the ban is
        # actually present, not absent-by-typo
        root = Path(g.ROOT_CONFIG).read_text(encoding="utf-8")
        for path in ("std::time::Instant::now", "std::time::Instant::elapsed",
                     "std::time::SystemTime::now", "std::time::SystemTime::elapsed",
                     "std::collections::hash_map::RandomState::new"):
            self.assertIn(f'path = "{path}"', root, f"clippy.toml must disallow {path}")
        types = root[root.index("disallowed-types = ["):]
        self.assertIn('path = "std::collections::hash_map::RandomState"', types)

    def test_the_committed_library_configuration_is_up_to_date(self):
        root = Path(g.ROOT_CONFIG).read_text(encoding="utf-8")
        lib = Path(g.LIB_CONFIG).read_text(encoding="utf-8")
        self.assertEqual(g.check(root, lib), [])


if __name__ == "__main__":
    unittest.main()
