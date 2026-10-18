#!/usr/bin/env python3
"""Tests for scripts/clippy_lib_gate.py (run: python3 scripts/test_clippy_lib_gate.py).

The same file in every repository that carries the gate. The judge is fed clippy output
written here, so each case fixes exactly which diagnostics appear; the last class checks
this repository's own data file and canary source.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clippy_lib_gate as g  # noqa: E402

DATA = '''
header = """\\
# generated
"""
method-reason = "zeros"
type-reason = "order"

[[method]]
path = "f32::max"

[[method]]
path = "std::collections::hash_map::RandomState::new"
reason = "entropy"

[[type]]
path = "std::collections::HashMap"

[[form]]
text = "a.max(b)"
path = "f32::max"

[[form]]
text = "RandomState::new()"
path = "std::collections::hash_map::RandomState::new"
also = ["std::collections::hash_map::RandomState"]

[[form]]
text = "&std::collections::HashMap<u32, u32>"
path = "std::collections::HashMap"

[[clean]]
text = "later.duration_since(earlier)"
'''

ROOT = '''disallowed-methods = [
    { path = "f32::sin", reason = "libm" },
]
disallowed-types = [
    { path = "std::collections::hash_map::RandomState", reason = "entropy" },
]
'''

SOURCE = """pub fn f(a: f32, b: f32) -> f32 {
    a.max(b)
}
pub fn r() -> u64 {
    let s = RandomState::new();
    0
}
pub fn h(m: &std::collections::HashMap<u32, u32>) -> usize {
    m.len()
}
pub fn t() {
    let _ = later.duration_since(earlier);
}
"""


def diag(line: int, kind: str, path: str) -> str:
    return f"src/lib.rs:{line}:5: error: use of a disallowed {kind} `{path}`\n"


GOOD = (diag(2, "method", "f32::max")
        + diag(5, "method", "std::collections::hash_map::RandomState::new")
        + diag(5, "type", "std::collections::hash_map::RandomState")
        + diag(8, "type", "std::collections::HashMap"))


def data(text: str = DATA) -> g.Data:
    return g.load_data(text)


def judge(output: str, source: str = SOURCE, rc: int = 101, d: g.Data | None = None):
    forms, clean = g.canary_lines(source, d or data())
    return g.judge_canary(rc, output, forms, clean)


class Data(unittest.TestCase):
    def test_loads_entries_reasons_forms_and_clean(self) -> None:
        d = data()
        self.assertEqual(d.methods, [("f32::max", "zeros"), ("std::collections::hash_map::RandomState::new", "entropy")])
        self.assertEqual(d.types, [("std::collections::HashMap", "order")])
        self.assertEqual([f.path for f in d.forms][1], "std::collections::hash_map::RandomState::new")
        self.assertEqual(d.clean, ["later.duration_since(earlier)"])

    def test_an_entry_no_form_names_is_refused(self) -> None:
        # clippy is silent about a method path that names nothing, so an unexercised entry is not trusted
        text = DATA.replace('[[type]]\npath = "std::collections::HashMap"\n',
                            '[[type]]\npath = "std::collections::HashMap"\n\n[[method]]\npath = "f32::no_such_method"\n')
        with self.assertRaisesRegex(ValueError, "no \\[\\[form\\]\\] naming it: f32::no_such_method"):
            data(text)

    def test_duplicate_form_text_entry_and_missing_fields_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "declared twice"):
            data(DATA + '\n[[form]]\ntext = "a.max(b)"\npath = "f32::max"\n')
        with self.assertRaisesRegex(ValueError, "more than once"):
            data(DATA + '\n[[method]]\npath = "f32::max"\n')
        with self.assertRaisesRegex(ValueError, "needs a text and the path"):
            data(DATA + '\n[[form]]\ntext = "x.y()"\n')
        with self.assertRaisesRegex(ValueError, "no reason"):
            data(DATA.replace('type-reason = "order"\n', ""))
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            data('extra = 1\n' + DATA)


class Render(unittest.TestCase):
    def test_entries_go_to_the_top_of_each_list_after_the_header(self) -> None:
        out = g.render(ROOT, data())
        self.assertTrue(out.startswith("# generated\n\n"))
        self.assertIn('disallowed-methods = [\n    { path = "f32::max", reason = "zeros" },\n'
                      '    { path = "std::collections::hash_map::RandomState::new", reason = "entropy" },\n'
                      '    { path = "f32::sin"', out)
        self.assertIn('disallowed-types = [\n    { path = "std::collections::HashMap", reason = "order" },\n', out)

    def test_an_entry_already_in_the_root_file_is_refused(self) -> None:
        root = ROOT.replace('"f32::sin"', '"f32::max"')
        with self.assertRaisesRegex(ValueError, "already lists f32::max"):
            g.render(root, data())

    def test_check_compares_with_the_generated_file_and_counts(self) -> None:
        d = data()
        self.assertEqual(g.check(ROOT, g.render(ROOT, d), d), [])
        self.assertIn("is missing", g.check(ROOT, None, d)[0])
        self.assertIn("differs", g.check(ROOT, g.render(ROOT, d) + "\n", d)[0])
        crowded = ROOT.replace('    { path = "f32::sin", reason = "libm" },\n',
                               '    { path = "f32::sin", reason = "libm" }, { path = "f32::cos", reason = "libm" },\n')
        self.assertIn("more than one entry", g.check(crowded, g.render(crowded, d), d)[0])

    def test_an_empty_root_method_list_is_red(self) -> None:
        d = data()
        root = ROOT.replace('    { path = "f32::sin", reason = "libm" },\n', "")
        self.assertEqual(g.check(root, g.render(root, d), d), ["clippy.toml: disallowed-methods lists nothing (compared nothing)"])

    def test_a_list_the_extras_need_must_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "no `disallowed-types = \\[` list"):
            g.render(ROOT.split("disallowed-types")[0], data())


class Judge(unittest.TestCase):
    def test_every_form_flagged_by_its_own_path_passes(self) -> None:
        errors, flagged, n = judge(GOOD)
        self.assertEqual(errors, [])
        self.assertEqual((flagged, n), (3, 4))

    def test_a_removed_entry_with_another_diagnostic_on_the_line_is_red(self) -> None:
        # the hole of the line-only judge: RandomState::new() still carries the type diagnostic
        out = GOOD.replace(diag(5, "method", "std::collections::hash_map::RandomState::new"), "")
        errors, flagged, _ = judge(out)
        self.assertEqual(flagged, 2)
        self.assertIn("has no diagnostic naming `std::collections::hash_map::RandomState::new` "
                      "(only std::collections::hash_map::RandomState)", errors[0])

    def test_a_wrong_path_only_is_red_and_unaccounted(self) -> None:
        out = GOOD.replace(diag(2, "method", "f32::max"), diag(2, "method", "f32::min"))
        errors, flagged, _ = judge(out)
        self.assertEqual(flagged, 2)
        self.assertTrue(any("no diagnostic naming `f32::max` (only f32::min)" in e for e in errors), errors)
        self.assertTrue(any("unaccounted diagnostic: line 2 `f32::min`" in e for e in errors), errors)

    def test_a_missing_diagnostic_is_red(self) -> None:
        errors, flagged, _ = judge(GOOD.replace(diag(8, "type", "std::collections::HashMap"), ""))
        self.assertEqual(flagged, 2)
        self.assertEqual(errors, ["canary line 8 (`&std::collections::HashMap<u32, u32>`) has no diagnostic "
                                  "naming `std::collections::HashMap`"])

    def test_an_extra_diagnostic_alone_is_red(self) -> None:
        errors, flagged, _ = judge(GOOD + diag(9, "method", "std::collections::HashMap::len"))
        self.assertEqual(flagged, 3)
        self.assertEqual(errors, ["unaccounted diagnostic: line 9 `std::collections::HashMap::len` "
                                  "(no [[form]] on that line names it)"])

    def test_a_second_path_not_listed_in_also_is_unaccounted(self) -> None:
        d = data(DATA.replace('also = ["std::collections::hash_map::RandomState"]\n', ""))
        errors, flagged, _ = judge(GOOD, d=d)
        self.assertEqual(flagged, 3)
        self.assertEqual(len(errors), 1)
        self.assertIn("unaccounted diagnostic: line 5 `std::collections::hash_map::RandomState`", errors[0])

    def test_a_form_text_that_occurs_twice_or_never_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "form `a.max\\(b\\)` occurs 2 times"):
            judge(GOOD, source=SOURCE + "fn g(a: f32, b: f32) -> f32 { a.max(b) }\n")
        with self.assertRaisesRegex(ValueError, "occurs 0 times"):
            judge(GOOD, source=SOURCE.replace("a.max(b)", "a.min(b)"))

    def test_a_commented_line_does_not_count_as_the_form(self) -> None:
        errors, _, _ = judge(GOOD, source=SOURCE + "// a.max(b) in a comment\n")
        self.assertEqual(errors, [])

    def test_a_clean_line_with_a_diagnostic_is_red(self) -> None:
        errors, _, _ = judge(GOOD + diag(12, "method", "std::time::Instant::duration_since"))
        self.assertEqual(errors, ["canary line 12 (`later.duration_since(earlier)`) must stay clean but has "
                                  "std::time::Instant::duration_since"])

    def test_no_forms_and_a_clean_exit_are_red(self) -> None:
        empty = g.Data("", [], [], [], [])
        errors, flagged, _ = g.judge_canary(101, "", {}, {})
        self.assertIn("no canary form to judge (expected 0, compared nothing)", errors)
        self.assertEqual(flagged, 0)
        self.assertEqual(g.canary_lines(SOURCE, empty), ({}, {}))
        errors, _, _ = judge(GOOD, rc=0)
        self.assertIn("clippy exited 0 on the canary: the configuration did not forbid anything", errors)

    def test_a_configured_path_that_refers_to_nothing_is_red(self) -> None:
        warn = ("/r/.github/clippy-lib/clippy.toml:45:5: warning: `std::nonexistent::fn_xyz` "
                "does not refer to a reachable function\n")
        errors, _, _ = judge(GOOD + warn)
        self.assertEqual(errors, ["clippy.toml entry `std::nonexistent::fn_xyz` does not refer to anything: "
                                  "it forbids nothing"])

    def test_diagnostics_of_other_files_are_ignored(self) -> None:
        errors, _, n = judge(GOOD + "src/other.rs:3:1: error: use of a disallowed method `f32::min`\n")
        self.assertEqual((errors, n), ([], 4))


class Main(unittest.TestCase):
    def test_a_refused_data_file_is_one_error_line_and_red(self) -> None:
        import io
        import tempfile
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "clippy_lib_gate.toml")
            with open(bad, "w", encoding="utf-8") as f:
                f.write(DATA + '\n[[method]]\npath = "f32::no_such_method"\n')
            saved, g.DATA = g.DATA, bad
            err = io.StringIO()
            try:
                with redirect_stderr(err):
                    rc = g.main(["--check"])
            finally:
                g.DATA = saved
        self.assertEqual(rc, 1)
        self.assertEqual(err.getvalue().splitlines(),
                         ["error: clippy_lib_gate.toml: 1 library-only entry has no [[form]] naming it: f32::no_such_method"])


class ThisRepository(unittest.TestCase):
    def test_the_data_file_loads_and_each_canary_text_occurs_once(self) -> None:
        with open(g.DATA, encoding="utf-8") as f:
            d = g.load_data(f.read())
        with open(os.path.join(g.CANARY, "src", "lib.rs"), encoding="utf-8") as f:
            forms, clean = g.canary_lines(f.read(), d)
        self.assertGreater(sum(len(v) for v in forms.values()), 0)
        self.assertEqual(sum(len(v) for v in forms.values()), len(d.forms))
        self.assertEqual(len(clean), len(d.clean))

    def test_the_generated_file_is_current(self) -> None:
        with open(g.DATA, encoding="utf-8") as f:
            d = g.load_data(f.read())
        with open(g.ROOT_CONFIG, encoding="utf-8") as f:
            root = f.read()
        lib = open(g.LIB_CONFIG, encoding="utf-8").read() if os.path.exists(g.LIB_CONFIG) else None
        self.assertEqual(g.check(root, lib, d), [])


if __name__ == "__main__":
    unittest.main()
