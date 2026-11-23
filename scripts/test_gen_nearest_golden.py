#!/usr/bin/env python3
"""Tests of gen_nearest_golden.py: closed-form cases of both distances, the repository's GOLDEN rows reproduce, a
row with one output bit flipped is reported, and a file without rows fails."""
import importlib.util
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gen", HERE / "gen_nearest_golden.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
TEST = HERE.parent / "tests" / "test_sheet_field_nearest.rs"


class GenTest(unittest.TestCase):
    def test_closed_forms(self):
        a, b, c = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        for f in (gen.plane_seg, gen.ericson):
            self.assertEqual(f((0.25, 0.25, 2.0), a, b, c), 2.0)  # above the face
            self.assertEqual(f((-3.0, -4.0, 0.0), a, b, c), 5.0)  # nearest the vertex a
            self.assertEqual(f((0.5, -2.0, 0.0), a, b, c), 2.0)  # nearest the edge ab

    def test_the_repository_rows_reproduce(self):
        self.assertEqual(gen.check(str(TEST)), 0)

    def test_a_flipped_bit_is_reported(self):
        text = TEST.read_text(encoding="utf-8")
        m = gen.ROW.search(text, text.index("const GOLDEN"))
        er = m.group(3)
        flipped = text[: m.start(3)] + hex(int(er, 16) ^ 1) + text[m.end(3):]
        with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False, encoding="utf-8") as f:
            f.write(flipped)
        self.addCleanup(pathlib.Path(f.name).unlink)
        self.assertEqual(gen.check(f.name), 1)

    def test_no_rows_fails(self):
        with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False, encoding="utf-8") as f:
            f.write("const GOLDEN: &[([u64; 12], u64, u64)] = &[\n];\n")
        self.addCleanup(pathlib.Path(f.name).unlink)
        self.assertEqual(gen.check(f.name), 1)


if __name__ == "__main__":
    unittest.main()
