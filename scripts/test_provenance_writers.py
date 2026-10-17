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
TEST = (
    'const WRITERS: &[(&str, &str, Write)] = &[\n    ("src/io/a.rs::export_a", "a.x", |m, p, id| {}),\n];\n'
    "fn round_trip() { io::export_a_with_provenance(p, id); }\n"
)


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

    def test_a_variant_with_no_base_writer_fails(self):
        src = SRC.replace("#[cfg(test)]", "pub fn export_orphan_with_stamp(p: &Path) {}\n\n#[cfg(test)]")
        test = TEST + "fn t() { io::export_orphan_with_stamp(p); }\n"
        e = errors(tree(src=src, test=test))
        self.assertTrue(any("export_orphan_with_stamp: extends no writer" in x for x in e), e)

    def test_a_variant_the_test_never_calls_fails(self):
        src = SRC.replace("#[cfg(test)]", "pub fn export_a_with_stamp(p: &Path) {}\n\n#[cfg(test)]")
        e = errors(tree(src=src))
        self.assertTrue(any("export_a_with_stamp: not called" in x for x in e), e)
        e = errors(tree(src=src, test=TEST + "fn t() { io::export_a_with_stamp(p); }\n"))
        self.assertEqual(e, [])

    def test_qualified_pub_fns_are_writers(self):
        for q in ["unsafe ", 'extern "C" ', "const ", "async ", "pub(crate) "[:0]]:
            src = SRC.replace("#[cfg(test)]", f"pub {q}fn export_q() {{}}\n\n#[cfg(test)]")
            e = errors(tree(src=src))
            self.assertTrue(any("export_q: neither covered" in x for x in e), (q, e))

    def test_a_stray_cfg_test_item_hides_nothing_after_it(self):
        src = SRC.replace("pub fn save_b() {}", "#[cfg(test)]\nconst P: u8 = 0;\npub fn save_b() {}\npub fn export_late() {}")
        e = errors(tree(src=src))
        self.assertTrue(any("export_late: neither covered" in x for x in e), e)

    def test_a_writer_inside_a_cfg_test_module_is_not_counted(self):
        self.assertNotIn("src/io/a.rs::export_in_tests", pw.writers(tree()))

    def test_a_variant_named_only_in_a_comment_or_string_is_not_called(self):
        src = SRC.replace("#[cfg(test)]", "pub fn export_a_with_stamp(p: &Path) {}\n\n#[cfg(test)]")
        for t in ["// io::export_a_with_stamp(p)\n", "/* export_a_with_stamp() */\n", 'const S: &str = "export_a_with_stamp(p)";\n']:
            e = errors(tree(src=src, test=TEST + t))
            self.assertTrue(any("export_a_with_stamp: not called" in x for x in e), (t, e))

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
