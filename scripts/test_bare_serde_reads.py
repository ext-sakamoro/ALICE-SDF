#!/usr/bin/env python3
"""Tests for scripts/bare_serde_reads.py: a new bare deserialization of a tree in library
code is flagged until reviewed, in each spelling the check knows; test modules are not
counted; a stale entry, a tree with no source file and a bad table row fail; and the
repository passes."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "bare_serde_reads.py"
HEADER = "# file\tfn\tcall\tstatus\treason\n"


class BareSerdeReads(unittest.TestCase):
    def run_check(self, files, rows, make_src=True):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            if make_src:
                (root / "src").mkdir()
            for name, text in files.items():
                (root / "src" / name).write_text(text, encoding="utf-8")
            table = root / "t.tsv"
            table.write_text(HEADER + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
            p = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--table", str(table)],
                               capture_output=True, text=True)
        return p.returncode, p.stdout

    def test_a_new_bare_tree_read_is_flagged_until_reviewed(self):
        for call, src in [
            ("serde_json::from_str", "pub fn f(s: &str) -> SdfTree {\n    serde_json::from_str::<SdfTree>(s).unwrap()\n}\n"),
            ("serde_json::from_slice", "pub fn f(b: &[u8]) -> SdfTree {\n    serde_json::from_slice(b).unwrap()\n}\n"),
            ("serde_json::from_reader", "pub fn f(r: std::fs::File) -> SdfTree {\n    serde_json::from_reader(r).unwrap()\n}\n"),
            ("bincode::serde::decode_from_slice",
             "pub fn f(b: &[u8]) -> SdfTree {\n    bincode::serde::decode_from_slice(b, cfg()).unwrap().0\n}\n"),
        ]:
            with self.subTest(call=call):
                rc, out = self.run_check({"lib.rs": src}, [])
                self.assertEqual(rc, 1, out)
                self.assertIn(f"not reviewed: src/lib.rs\tf\t{call}", out)
                rc, out = self.run_check({"lib.rs": src}, [("src/lib.rs", "f", call, "pending", "a reader still to check")])
                self.assertEqual(rc, 0, out)

    def test_test_modules_are_not_counted(self):
        src = ("pub fn id() {}\n#[cfg(test)]\nmod tests {\n    fn f(s: &str) { let _t: SdfTree = serde_json::from_str(s).unwrap(); }\n}\n"
               "#[cfg(all(test, feature = \"std\"))]\nmod more {\n    fn g(s: &str) { let _t: SdfTree = serde_json::from_str(s).unwrap(); }\n}\n")
        rc, out = self.run_check({"lib.rs": src}, [])
        self.assertEqual(rc, 0, out)
        self.assertIn("0 calls", out)

    def test_a_stale_entry_fails(self):
        rc, out = self.run_check({"lib.rs": "pub fn f() {}\n"}, [("src/lib.rs", "g", "serde_json::from_str", "checked", "a loader that is gone")])
        self.assertEqual(rc, 1, out)
        self.assertIn("listed but not found", out)

    def test_no_source_file_fails(self):
        rc, out = self.run_check({}, [], make_src=False)
        self.assertEqual(rc, 1, out)
        self.assertIn("0 scanned", out)

    def test_a_bad_row_is_refused(self):
        for row in [("src/lib.rs", "f", "serde_json::from_str", "allowed", "an unknown status"),
                    ("src/lib.rs", "f", "serde_json::from_str", "checked", "short")]:
            with self.subTest(row=row):
                rc, _ = self.run_check({"lib.rs": "pub fn f(s: &str) { serde_json::from_str::<u8>(s).ok(); }\n"}, [row])
                self.assertNotEqual(rc, 0)

    def test_the_repository_passes(self):
        p = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout)


if __name__ == "__main__":
    unittest.main()
