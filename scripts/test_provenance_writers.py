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


def tree(
    src: str = SRC,
    test: str = TEST,
    pending: str = "expected-writers 2\nsrc/io/a.rs::save_b\n",
    not_writers: str | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "src/io"))
    os.makedirs(os.path.join(root, "tests"))
    os.makedirs(os.path.join(root, "scripts"))
    files = [("src/io/a.rs", src), (pw.TEST_FILE, test), (pw.PENDING_FILE, pending)]
    if not_writers is not None:
        files.append((pw.NOT_WRITERS_FILE, not_writers))
    files += list((extra or {}).items())
    for rel, text in files:
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
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

    def test_a_writer_name_anywhere_under_src_is_a_candidate(self):
        # a new module outside src/io with an encoder: counted, so it must be classified
        e = errors(tree(extra={"src/mesh/b.rs": "pub fn encode_foo(m: &Mesh) -> Vec<u8> { Vec::new() }\n"}))
        self.assertTrue(any("src/mesh/b.rs::encode_foo: neither covered" in x for x in e), e)
        self.assertTrue(any("3 writers found" in x for x in e), e)
        for name in ["bake_to_grid", "serialize", "persist_all", "dump_state", "to_vec"]:
            e = errors(tree(extra={"src/b.rs": f"pub fn {name}() {{}}\n"}))
            self.assertTrue(any(f"src/b.rs::{name}: neither covered" in x for x in e), (name, e))

    def test_a_write_sink_makes_any_name_a_candidate(self):
        for body in [
            "pub fn store(w: &mut impl Write) {}\n",
            "pub fn store<W: std::io::Write>(w: W) {}\n",
            "pub fn store(w: &mut dyn Write) {}\n",
            "pub fn store(p: &Path) { std::fs::write(p, b\"x\").unwrap(); }\n",
            "pub fn store(p: &Path) { let f = File::create(p); }\n",
            "pub fn store(p: &Path) { let f = OpenOptions::new(); }\n",
        ]:
            e = errors(tree(extra={"src/b.rs": body}))
            self.assertTrue(any("src/b.rs::store: neither covered" in x for x in e), (body, e))
        # a fmt::Write into a String inside the body is no sink, and no writer name
        e = errors(tree(extra={"src/b.rs": "pub fn hex(&self) -> String { let mut s = String::new(); write!(s, \"x\"); s }\n"}))
        self.assertEqual(e, [])

    def test_an_array_return_type_does_not_end_the_signature(self):
        e = errors(tree(extra={"src/b.rs": "pub fn to_bytes(&self) -> [u8; 4] { [0; 4] }\n"}))
        self.assertTrue(any("src/b.rs::to_bytes: neither covered" in x for x in e), e)

    def test_a_delegating_fn_listed_as_not_a_writer_passes(self):
        b = "pub fn export_a(p: &str) { use crate::io::export_a as io_export_a; io_export_a(p) }\n"
        root = tree(extra={"src/py.rs": b}, not_writers="src/py.rs::export_a delegates to export_a\n")
        self.assertEqual(errors(root), [])
        self.assertEqual(pw.check(root)[1], 2)

    def test_a_delegating_fn_that_stops_calling_its_writer_fails(self):
        b = "pub fn export_a(p: &str) { other(p) }\n"
        e = errors(tree(extra={"src/py.rs": b}, not_writers="src/py.rs::export_a delegates to export_a\n"))
        self.assertTrue(any("does not call it" in x for x in e), e)

    def test_a_delegation_to_no_writer_fails(self):
        b = "pub fn export_x(p: &str) { import_a(p) }\n"
        e = errors(tree(extra={"src/py.rs": b}, not_writers="src/py.rs::export_x delegates to import_a\n"))
        self.assertTrue(any("which is no covered or pending writer" in x for x in e), e)

    def test_a_short_reason_fails_and_a_long_one_passes(self):
        x = {"src/b.rs": "pub fn serialize() {}\n"}
        e = errors(tree(extra=x, not_writers="src/b.rs::serialize in memory\n"))
        self.assertTrue(any("at least five words" in y for y in e), e)
        self.assertEqual(errors(tree(extra=x, not_writers="src/b.rs::serialize packs words for an in-process upload\n")), [])

    def test_a_not_writer_line_that_names_no_candidate_fails(self):
        e = errors(tree(not_writers="src/io/a.rs::import_a reads a file and writes nothing at all\n"))
        self.assertTrue(any("import_a is not a candidate" in x for x in e), e)

    def test_a_not_writer_also_pending_or_listed_twice_fails(self):
        line = "src/io/a.rs::save_b writes nothing that is stored anywhere\n"
        e = errors(tree(not_writers=line))
        self.assertTrue(any("save_b is also covered or pending" in x for x in e), e)
        e = errors(tree(not_writers=line * 2))
        self.assertTrue(any("listed twice: src/io/a.rs::save_b" in x for x in e), e)

    def test_this_repository_passes(self):
        self.assertEqual(errors(pw.ROOT), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
