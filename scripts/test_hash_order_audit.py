#!/usr/bin/env python3
"""Tests for scripts/hash_order_audit.py: every planted iteration of a hash
collection is found (each method, a for loop, through a field), a comment, a
lookup and the unit-test module are not, and the list is compared both ways."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hash_order_audit as h  # noqa: E402

DECLS = "use std::collections::{HashMap, HashSet};\nfn f(m: &HashMap<u8, u8>, s: HashSet<u8>) {\n"


class Scan(unittest.TestCase):
    def test_each_planted_iteration_is_found(self):
        forms = ["m.iter().count();", "m.iter_mut();", "m.keys();", "m.values();", "m.values_mut();",
                 "s.into_iter();", "m.into_keys();", "m.into_values();", "s.drain();", "s.retain(|_| true);",
                 "for x in &m {}", "for x in s {}", "for x in &mut m {}"]
        for form in forms:
            with self.subTest(form=form):
                found = h.scan_text("src/a.rs", DECLS + "    " + form + "\n}\n")
                self.assertEqual(found, [("src/a.rs", form)])
        self.assertEqual(len(forms), 13)

    def test_a_method_chain_over_several_lines_is_one_finding(self):
        text = "struct S { entries: HashMap<u8, u8> }\nfn k(c: &S) -> Vec<u8> {\n    c\n        .entries\n        .keys()\n        .copied()\n        .collect()\n}\n"
        self.assertEqual([c for _, c in h.scan_text("src/a.rs", text)],
                         ["c.entries.keys().copied().collect()"])

    def test_a_let_bound_map_and_a_field_are_scanned(self):
        text = ("fn g() {\n    let mut seen = HashSet::new();\n    for v in &seen {}\n}\n"
                "struct S { pub entries: HashMap<u8, u8> }\nfn k(c: &S) { for e in c.entries.values() {} }\n")
        found = [c for _, c in h.scan_text("src/a.rs", text)]
        self.assertEqual(found, ["for v in &seen {}", "fn k(c: &S) { for e in c.entries.values() {} }"])

    def test_a_comment_a_lookup_an_ordered_map_and_the_test_module_are_not_found(self):
        text = (DECLS + "    // m.iter()\n    let x = m.get(&1);\n}\n"
                "fn o(b: &std::collections::BTreeMap<u8, u8>) { for x in b {} }\n"
                "#[cfg(test)]\nmod tests {\n    fn t(m: &HashMap<u8, u8>) { m.iter(); }\n}\n")
        self.assertEqual(h.scan_text("src/a.rs", text), [])


class Check(unittest.TestCase):
    F = [("src/a.rs", "m.keys();")]

    def entry(self, **kw):
        e = {"file": "src/a.rs", "code": "m.keys();", "verdict": "not-observed", "why": "a count of the keys only"}
        e.update(kw)
        return e

    def test_a_listed_finding_passes(self):
        self.assertEqual(h.check(self.F, [self.entry()]), [])

    def test_a_finding_without_an_entry_fails(self):
        self.assertTrue(any("without a verdict" in p for p in h.check(self.F, [])))

    def test_an_entry_the_scan_no_longer_finds_fails(self):
        self.assertTrue(any("no longer" in p or "the scan finds 0" in p
                            for p in h.check([], [self.entry()])))

    def test_an_unknown_verdict_or_a_short_why_fails(self):
        self.assertTrue(h.check(self.F, [self.entry(verdict="fine")]))
        self.assertTrue(h.check(self.F, [self.entry(why="ok")]))

    def test_count_covers_identical_lines(self):
        two = self.F * 2
        self.assertTrue(h.check(two, [self.entry()]))
        self.assertEqual(h.check(two, [self.entry(count=2)]), [])


class Main(unittest.TestCase):
    def test_this_repository_passes_and_finds_something(self):
        found, nfiles = h.scan(Path(h.__file__).resolve().parents[1])
        self.assertGreater(nfiles, 100)
        self.assertGreater(len(found), 5)
        self.assertEqual(h.main([]), 0)

    def test_nothing_found_is_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "src").mkdir()
            (Path(d) / "src" / "lib.rs").write_text("pub fn f() {}\n")
            self.assertEqual(h.main(["--root", d]), 2)


if __name__ == "__main__":
    unittest.main()
