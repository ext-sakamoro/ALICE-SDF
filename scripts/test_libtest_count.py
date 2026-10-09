#!/usr/bin/env python3
"""Tests for scripts/libtest_count.py.

The samples are captured from `cargo test --doc --features gpu GpuEvalFuture`
run with CARGO_TERM_COLOR=always CI=true (the env of ci.yml), output piped to a
file the way the CI step does (2026-10-09). COLOURED is the same run with
`-- --color always`: libtest then writes `\\x1b[32mok\\x1b(B\\x1b[m`, where
`\\x1b(B` is a charset selection that an SGR-only strip leaves behind.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import libtest_count as lc  # noqa: E402

# CARGO_TERM_COLOR=always, piped: cargo's status words are coloured, libtest is plain.
CI_ENV = (
    "\x1b[1m\x1b[92m    Finished\x1b[0m `test` profile [unoptimized + debuginfo] target(s) in 1m 45s\n"
    "\x1b[1m\x1b[92m   Doc-tests\x1b[0m alice_sdf\n"
    "\n"
    "running 2 tests\n"
    "test src/compiled/wgsl/gpu_eval.rs - compiled::wgsl::gpu_eval::GpuEvalFuture (line 1230) - compile fail ... ok\n"
    "test src/compiled/wgsl/gpu_eval.rs - compiled::wgsl::gpu_eval::GpuEvalFuture (line 1241) - compile ... ok\n"
    "\n"
    "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 36 filtered out; finished in 0.12s\n"
)

# Same run with `-- --color always`.
COLOURED = (
    "\x1b[1m\x1b[92m    Finished\x1b[0m `test` profile [unoptimized + debuginfo] target(s) in 0.18s\n"
    "\x1b[1m\x1b[92m   Doc-tests\x1b[0m alice_sdf\n"
    "\n"
    "running 2 tests\n"
    "test src/compiled/wgsl/gpu_eval.rs - compiled::wgsl::gpu_eval::GpuEvalFuture (line 1241) - compile ... \x1b[32mok\x1b(B\x1b[m\n"
    "test src/compiled/wgsl/gpu_eval.rs - compiled::wgsl::gpu_eval::GpuEvalFuture (line 1230) - compile fail ... \x1b[32mok\x1b(B\x1b[m\n"
    "\n"
    "test result: \x1b[32mok\x1b(B\x1b[m. 2 passed; 0 failed; 0 ignored; 0 measured; 36 filtered out; finished in 0.06s\n"
)


def run(log: str, *args: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "doc.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(log)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = lc.main([path, *args])
        return rc, out.getvalue()


class StripAnsi(unittest.TestCase):
    def test_removes_sgr_and_charset_selection(self):
        self.assertEqual(lc.strip_ansi("test result: \x1b[32mok\x1b(B\x1b[m. 2 passed"), "test result: ok. 2 passed")
        self.assertEqual(lc.strip_ansi("\x1b[1m\x1b[92m   Doc-tests\x1b[0m alice_sdf"), "   Doc-tests alice_sdf")

    def test_leaves_plain_text(self):
        self.assertEqual(lc.strip_ansi(CI_ENV), CI_ENV.replace("\x1b[1m\x1b[92m", "").replace("\x1b[0m", ""))


class PassedCount(unittest.TestCase):
    def test_ci_env_sample(self):
        self.assertEqual(lc.passed_count(CI_ENV), 2)

    def test_coloured_libtest_sample(self):
        self.assertEqual(lc.passed_count(COLOURED), 2)

    def test_last_summary_wins(self):
        log = "test result: ok. 9 passed; 0 failed\n" + COLOURED
        self.assertEqual(lc.passed_count(log), 2)

    def test_no_summary(self):
        self.assertIsNone(lc.passed_count("\x1b[1m\x1b[91merror\x1b[0m: could not compile `alice-sdf`\n"))

    def test_failed_summary_is_not_ok(self):
        self.assertIsNone(lc.passed_count("test result: \x1b[31mFAILED\x1b(B\x1b[m. 1 passed; 1 failed\n"))


class Main(unittest.TestCase):
    def test_coloured_meets_minimum(self):
        rc, out = run(COLOURED, "--min", "2")
        self.assertEqual((rc, out.strip()), (0, "2"))

    def test_below_minimum_fails_with_reason(self):
        rc, out = run(COLOURED, "--min", "3", "--what", "GpuEvalFuture doctests")
        self.assertEqual(rc, 1)
        self.assertIn("::error::expected >= 3 GpuEvalFuture doctests, ran 2", out)

    def test_zero_tests_fails(self):
        log = COLOURED.replace("2 passed", "0 passed").replace("36 filtered", "38 filtered")
        rc, out = run(log, "--min", "2")
        self.assertEqual(rc, 1)
        self.assertIn("ran 0", out)

    def test_missing_summary_fails_with_reason(self):
        rc, out = run("\x1b[1m\x1b[91merror\x1b[0m: could not compile\n", "--min", "1")
        self.assertEqual(rc, 1)
        self.assertIn("no `test result: ok. N passed` line", out)


if __name__ == "__main__":
    unittest.main()
