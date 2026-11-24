#!/usr/bin/env python3
"""Tests of commit_marker_guard.py: a planted marker of each form is found, clean source passes, and an empty
src/ fails."""
import importlib.util
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "guard", pathlib.Path(__file__).with_name("commit_marker_guard.py"))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def tree(files: dict) -> tempfile.TemporaryDirectory:
    d = tempfile.TemporaryDirectory()
    for rel, body in files.items():
        p = pathlib.Path(d.name) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return d


class GuardTest(unittest.TestCase):
    def test_each_marker_form_is_found(self):
        for line in [
            "pub const B: u32 = u32::MAX; // MEASUREMENT BUILD: replaced before commit",
            "// do not commit",
            "let x = 1; // Temp: widen for the probe",
            "// the value is Replaced Before Commit",
        ]:
            with self.subTest(line=line):
                d = tree({"src/a.rs": "fn a() {}\n" + line + "\n"})
                self.addCleanup(d.cleanup)
                found, files = guard.scan(pathlib.Path(d.name))
                self.assertEqual((len(found), files), (1, 1), found)
                self.assertIn("src/a.rs:2", found[0])

    def test_shaders_are_scanned(self):
        d = tree({"src/k.wgsl": "// DO NOT COMMIT\n", "src/a.rs": "fn a() {}\n"})
        self.addCleanup(d.cleanup)
        found, files = guard.scan(pathlib.Path(d.name))
        self.assertEqual((len(found), files), (1, 2), found)

    def test_clean_source_passes_and_words_alone_do_not_match(self):
        d = tree({"src/a.rs": "// a temperature: 20 C; template: none; measurement of the build time\n"})
        self.addCleanup(d.cleanup)
        found, files = guard.scan(pathlib.Path(d.name))
        self.assertEqual((found, files), ([], 1))

    def test_an_empty_src_reads_nothing(self):
        d = tree({"README.md": "x\n"})
        self.addCleanup(d.cleanup)
        self.assertEqual(guard.scan(pathlib.Path(d.name)), ([], 0))

    def test_this_repository_is_scanned(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        _, files = guard.scan(root)
        self.assertGreater(files, 100)


if __name__ == "__main__":
    unittest.main()
