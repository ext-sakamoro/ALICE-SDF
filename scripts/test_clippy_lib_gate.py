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
    { path = "std::collections::hash_map::RandomState", reason = "seed" },
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
pub fn g_v2_min() {
    v2.min(o)
}
pub fn g_v2_max() {
    v2.max(o)
}
pub fn g_v2_clamp() {
    v2.clamp(lo, hi)
}
pub fn g_v2_min_element() {
    v2.min_element()
}
pub fn g_v2_max_element() {
    v2.max_element()
}
pub fn g_v3_min() {
    v3.min(o)
}
pub fn g_v3_max() {
    v3.max(o)
}
pub fn g_v3_clamp() {
    v3.clamp(lo, hi)
}
pub fn g_v3_min_element() {
    v3.min_element()
}
pub fn g_v3_max_element() {
    v3.max_element()
}
pub fn g_v3a_min() {
    v3a.min(o)
}
pub fn g_v3a_max() {
    v3a.max(o)
}
pub fn g_v3a_clamp() {
    v3a.clamp(lo, hi)
}
pub fn g_v3a_min_element() {
    v3a.min_element()
}
pub fn g_v3a_max_element() {
    v3a.max_element()
}
pub fn g_v4_min() {
    v4.min(o)
}
pub fn g_v4_max() {
    v4.max(o)
}
pub fn g_v4_clamp() {
    v4.clamp(lo, hi)
}
pub fn g_v4_min_element() {
    v4.min_element()
}
pub fn g_v4_max_element() {
    v4.max_element()
}
pub fn g_d2_min() {
    d2.min(o)
}
pub fn g_d2_max() {
    d2.max(o)
}
pub fn g_d2_clamp() {
    d2.clamp(lo, hi)
}
pub fn g_d2_min_element() {
    d2.min_element()
}
pub fn g_d2_max_element() {
    d2.max_element()
}
pub fn g_d3_min() {
    d3.min(o)
}
pub fn g_d3_max() {
    d3.max(o)
}
pub fn g_d3_clamp() {
    d3.clamp(lo, hi)
}
pub fn g_d3_min_element() {
    d3.min_element()
}
pub fn g_d3_max_element() {
    d3.max_element()
}
pub fn g_d4_min() {
    d4.min(o)
}
pub fn g_d4_max() {
    d4.max(o)
}
pub fn g_d4_clamp() {
    d4.clamp(lo, hi)
}
pub fn g_d4_min_element() {
    d4.min_element()
}
pub fn g_d4_max_element() {
    d4.max_element()
}
pub fn h0() {
    x(&std::collections::HashMap<u32, u32>)
}
pub fn h1() {
    x(&std::collections::HashSet<u32>)
}
pub fn h2() {
    x(hash_map::Iter<'_, u32, u32>)
}
pub fn h3() {
    x(m.keys()
}
pub fn h4() {
    x(m.values()
}
pub fn h5() {
    x(m.iter_mut()
}
pub fn h6() {
    x(s.iter()
}
pub fn s0() {
    x(serde_json::from_str(js)
}
pub fn s1() {
    x(sj::from_str(js)
}
pub fn s2() {
    x(read_text::<Value>(js)
}
pub fn s3() {
    x(serde_json::value::from_value(jv)
}
pub fn s4() {
    x(serde_json::from_value(jv.clone())
}
pub fn s5() {
    x(serde_json::from_slice(jb)
}
pub fn s6() {
    x(serde_json::from_reader(jr)
}
pub fn s7() {
    x(serde_json::Deserializer::from_str(js)
}
pub fn s8() {
    x(bincode::serde::decode_from_slice(bb,
}
pub fn s9() {
    x(bs::decode_from_slice(bb,
}
pub fn s10() {
    x(bincode::serde::decode_from_reader(rd,
}
pub fn s11() {
    x(bincode::serde::decode_from_std_read(br,
}
pub fn s12() {
    x(bincode::decode_from_slice(bn,
}
pub fn s13() {
    x(u32::deserialize(tv)
}
pub fn s14() {
    x(serde_json::Deserializer::new(serde_json::de::StrRead::new(ts))
}
pub fn s15() {
    x(<u32 as Deserialize>::deserialize(&mut td)
}
pub fn s16() {
    x(u32::deserialize(&pv)
}
pub fn s17() {
    x(std::marker::PhantomData::<u32>.deserialize(sv)
}
"""


def diag(line: int, path: str) -> str:
    return f"src/lib.rs:{line}:5: error: use of a disallowed method `{path}`"


class Render(unittest.TestCase):
    def test_the_extra_entries_come_first_and_the_root_list_follows_unchanged(self):
        out = g.render(ROOT)
        self.assertTrue(out.startswith(g.HEADER))
        body = out[len(g.HEADER) + 1:]
        added = "".join(f'    {{ path = "{p}", reason = "{g.REASON}" }},\n' for p in g.EXTRA)
        added += "".join(f'    {{ path = "{p}", reason = "{g.ORDER_REASON}" }},\n' for p in g.EXTRA_ORDER_METHODS)
        added += "".join(f'    {{ path = "{p}", reason = "{g.STORED_FORM_REASON}" }},\n'
                         for p in g.EXTRA_STORED_FORM_METHODS)
        types = "".join(f'    {{ path = "{p}", reason = "{g.ORDER_REASON}" }},\n' for p in g.EXTRA_TYPES)
        self.assertEqual(body.replace(added, "").replace(types, ""), ROOT)
        i = body.index("disallowed-methods = [\n")
        self.assertIn('path = "f32::max"', body[i:body.index('path = "f32::sin"')])
        self.assertEqual(g.count_entries(out),
                         2 + len(g.EXTRA) + len(g.EXTRA_ORDER_METHODS) + len(g.EXTRA_STORED_FORM_METHODS))

    def test_the_extra_paths_are_exactly_the_float_and_glam_max_and_min(self):
        glam = [f"glam::{t}::{m}" for t in ("Vec2", "Vec3", "Vec3A", "Vec4", "DVec2", "DVec3", "DVec4")
                for m in ("min", "max", "clamp", "min_element", "max_element")]
        self.assertEqual(g.EXTRA, ["f32::max", "f32::min", "f64::max", "f64::min",
                                   "wide::f32x8::max", "wide::f32x8::min"] + glam)

    def test_the_stored_form_readers_are_forbidden(self):
        # ALICE-Law G8: every serde_json / bincode entry point that reads a stored form,
        # in the path clippy resolves to (a re-export is matched by its definition)
        for p in ("serde_json::de::from_str", "serde_json::de::from_slice", "serde_json::de::from_reader",
                  "serde_json::value::from_value", "serde_json::Deserializer::from_str",
                  "bincode::serde::decode_from_slice", "bincode::serde::decode_from_reader",
                  "bincode::serde::decode_from_std_read", "bincode::serde::borrow_decode_from_slice",
                  "bincode::decode_from_slice", "bincode::decode_from_reader", "bincode::decode_from_std_read",
                  "serde_json::Deserializer::new", "serde::de::Deserialize::deserialize",
                  "serde::de::DeserializeSeed::deserialize"):
            self.assertIn(p, g.EXTRA_STORED_FORM_METHODS)
            self.assertIn(f'path = "{p}"', g.render(ROOT))

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
        root = 'disallowed-methods = [\n]\n\ndisallowed-types = [\n]\n'
        self.assertTrue(g.check(root, g.render(root)))


class Canary(unittest.TestCase):
    def setUp(self):
        self.want = g.canary_lines(CANARY)

    def test_every_form_is_found_on_its_line(self):
        self.assertEqual(sorted(self.want), list(range(3, 3 + 3 * len(g.CANARY_FORMS), 3)))

    def test_a_diagnostic_on_every_line_and_a_non_zero_exit_passes(self):
        out = "\n".join(diag(n, "x") for n in self.want)
        errors, n = g.judge_canary(101, out, self.want)
        self.assertEqual((errors, n), ([], len(g.CANARY_FORMS)))

    def test_a_missing_line_fails(self):
        out = "\n".join(diag(n, "x") for n in self.want if n != 9)
        errors, _ = g.judge_canary(101, out, self.want)
        self.assertEqual(len(errors), 1)
        self.assertIn("line 9", errors[0])

    def test_exit_zero_fails_even_with_diagnostics(self):
        out = "\n".join(diag(n, "x") for n in self.want)
        errors, _ = g.judge_canary(0, out, self.want)
        self.assertTrue(errors)

    def test_an_entry_clippy_cannot_resolve_fails(self):
        out = "\n".join(diag(n, "x") for n in self.want)
        out += "\n.github/clippy-lib/clippy.toml:9:5: warning: `bincode::serde::gone` does not refer to a reachable function"
        errors, _ = g.judge_canary(101, out, self.want)
        self.assertEqual(len(errors), 1)
        self.assertIn("bincode::serde::gone", errors[0])

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

    def test_the_library_bans_the_hash_collections_and_their_iteration(self):
        out = g.render(ROOT)
        types = out[out.index("disallowed-types = ["):]
        for path in g.EXTRA_TYPES:
            self.assertIn(f'path = "{path}"', types)
        methods = out[out.index("disallowed-methods = ["):out.index("disallowed-types = [")]
        for path in g.EXTRA_ORDER_METHODS:
            self.assertIn(f'path = "{path}"', methods)
        self.assertEqual(g.check(ROOT, out), [])

    def test_a_dropped_type_entry_fails_the_count(self):
        out = g.render(ROOT)
        line = next(l for l in out.splitlines() if 'path = "std::collections::HashSet"' in l)
        self.assertTrue(g.check(ROOT, out.replace(line + "\n", "")))

    def test_the_root_may_not_list_a_library_only_entry(self):
        with self.assertRaises(ValueError):
            g.render(ROOT.replace("disallowed-types = [\n", 'disallowed-types = [\n    { path = "std::collections::HashSet", reason = "x" },\n'))

    def test_the_committed_library_configuration_is_up_to_date(self):
        root = Path(g.ROOT_CONFIG).read_text(encoding="utf-8")
        lib = Path(g.LIB_CONFIG).read_text(encoding="utf-8")
        self.assertEqual(g.check(root, lib), [])


if __name__ == "__main__":
    unittest.main()
