#!/usr/bin/env python3
"""Tests (canary) for scripts/det_math_guard.py: each planted call is found, the allowed forms are not.

The guard is run as the CI step runs it (from the repository root, reading `git ls-files`), in a
scratch git repository holding one law file.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

GUARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "det_math_guard.py")


def run(files: dict) -> subprocess.CompletedProcess:
    d = tempfile.mkdtemp()
    try:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
        subprocess.run(["git", "init", "-q"], cwd=d, check=True)
        subprocess.run(["git", "add", "-A"], cwd=d, check=True)
        return subprocess.run([sys.executable, GUARD], cwd=d, capture_output=True, text=True)
    finally:
        shutil.rmtree(d)


def law(body: str) -> dict:
    return {"src/operations/law.rs": f"pub fn f(x: f32, y: f32) -> f32 {{\n    {body}\n}}\n"}


class Guard(unittest.TestCase):
    def test_a_clean_law_passes(self):
        r = run(law("alice_det_math::sin(x) * y + 1.0"))
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_each_platform_libm_call_is_found(self):
        for call in ["x.sin()", "x.cos()", "x.sin_cos().0", "x.exp()", "x.ln()", "x.powf(y)", "x.atan2(y)", "x.hypot(y)", "x.cbrt()"]:
            with self.subTest(call=call):
                r = run(law(call))
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("src/operations/law.rs:2", r.stdout)

    def test_a_fused_mul_add_is_found(self):
        r = run(law("x.mul_add(y, 1.0)"))
        self.assertEqual(r.returncode, 1, r.stdout)

    def test_test_modules_comments_and_generic_laws_are_allowed(self):
        files = {"src/operations/law.rs": (
            "pub fn f<R: Real>(x: R) -> R { x.sin() }\n"
            "// x.sin() in a comment\n"
            "#[cfg(test)]\nmod tests {\n    fn r(x: f32) -> f32 { x.sin() }\n}\n")}
        r = run(files)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_the_excluded_file_and_other_directories_are_not_read(self):
        files = {"src/compiled/real.rs": "fn f(x: f32) -> f32 { x.sin() }\n",
                 "src/mesh/other.rs": "fn f(x: f32) -> f32 { x.sin() }\n",
                 "src/operations/law.rs": "fn f(x: f32) -> f32 { x + 1.0 }\n"}
        r = run(files)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_scanning_no_file_fails(self):
        r = run({"src/mesh/other.rs": "fn f() {}\n"})
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("scanned 0 files", r.stdout)


if __name__ == "__main__":
    unittest.main()
