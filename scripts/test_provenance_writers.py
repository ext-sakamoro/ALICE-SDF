#!/usr/bin/env python3
"""Tests of scripts/provenance_writers.py: each failure it promises, on a small tree."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance_writers as pw  # noqa: E402

SRC = """pub fn export_a(p: &Path) {}
pub fn export_a_with_provenance(p: &Path, id: &Provenance) {}
pub fn import_a(p: &Path) {}
fn export_private() {}
pub fn save_b() {}

#[cfg(test)]
mod tests {
    pub fn export_in_tests() {}
}
"""
TEST = 'const WRITERS: &[(&str, &str, Write)] = &[\n    ("src/io/a.rs::export_a", "a.x", |m, p, id| {}),\n];\n'


def tree(src: str = SRC, test: str = TEST, pending: str = "expected-writers 2\nsrc/io/a.rs::save_b\n") -> str:
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "src/io"))
    os.makedirs(os.path.join(root, "tests"))
    os.makedirs(os.path.join(root, "scripts"))
    for rel, text in [("src/io/a.rs", src), (pw.TEST_FILE, test), (pw.PENDING_FILE, pending)]:
        with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
            fh.write(text)
    return root


def errors(root: str) -> list[str]:
    return pw.check(root)[0]


class Writers(unittest.TestCase):
    def test_the_small_tree_passes(self):
        self.assertEqual(errors(tree()), [])
        self.assertEqual(pw.check(tree())[1], 2)

    def test_variants_private_fns_importers_and_the_test_module_are_not_writers(self):
        self.assertEqual(sorted(pw.writers(tree())), ["src/io/a.rs::export_a", "src/io/a.rs::save_b"])

    def test_a_new_writer_fails_twice_count_and_coverage(self):
        e = errors(tree(src=SRC.replace("#[cfg(test)]", "pub fn write_c() {}\n\n#[cfg(test)]")))
        self.assertTrue(any("3 writers found" in x for x in e), e)
        self.assertTrue(any("write_c: neither covered" in x for x in e), e)

    def test_a_removed_writer_fails_on_the_count(self):
        e = errors(tree(src=SRC.replace("pub fn save_b() {}\n", ""), pending="expected-writers 2\n"))
        self.assertTrue(any("1 writers found" in x for x in e), e)

    def test_a_pending_line_already_covered_fails(self):
        e = errors(tree(pending="expected-writers 2\nsrc/io/a.rs::save_b\nsrc/io/a.rs::export_a\n"))
        self.assertTrue(any("export_a is covered" in x for x in e), e)

    def test_a_pending_line_that_is_no_writer_fails(self):
        e = errors(tree(pending="expected-writers 2\nsrc/io/a.rs::save_b\nsrc/io/a.rs::gone\n"))
        self.assertTrue(any("gone is not a writer" in x for x in e), e)

    def test_a_covered_name_that_is_no_writer_fails(self):
        e = errors(tree(test=TEST.replace("export_a", "export_z")))
        self.assertTrue(any("export_z is not a writer" in x for x in e), e)
        self.assertTrue(any("export_a: neither covered" in x for x in e), e)

    def test_a_missing_count_fails(self):
        self.assertTrue(any("expected-writers" in x for x in errors(tree(pending="src/io/a.rs::save_b\n"))))

    def test_a_duplicate_pending_line_fails(self):
        e = errors(tree(pending="expected-writers 2\nsrc/io/a.rs::save_b\nsrc/io/a.rs::save_b\n"))
        self.assertTrue(any("listed twice" in x for x in e), e)

    def test_no_writer_found_fails(self):
        self.assertEqual(errors(tree(src="pub fn import_only() {}\n")), ["no writer found (compared nothing)"])

    def test_main_returns_1_on_error_and_0_when_clean(self):
        self.assertEqual(pw.main(["x", tree()]), 0)
        self.assertEqual(pw.main(["x", tree(src="")]), 1)

    def test_this_repository_passes(self):
        self.assertEqual(errors(pw.ROOT), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
