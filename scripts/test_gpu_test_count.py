#!/usr/bin/env python3
"""Tests for scripts/gpu_test_count.py on small logs."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_test_count as g  # noqa: E402

RAN = (
    "running 2 tests\n"
    "test compiled::wgsl::gpu_eval::tests::test_gpu_eval_sphere ... gpu-test: ran test_gpu_eval_sphere\n"
    "ok\n"
    "test mesh::gpu_marching_cubes::tests::test_gpu_mc_empty ... gpu-test: ran test_gpu_mc_empty\n"
    "ok\n"
    "test result: ok. 2 passed; 0 failed\n"
)
SKIP = "test x ... gpu-test: skipped test_gpu_mc_empty: device creation failed: Parent device is lost\nok\n"


def run(text, *args):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(text)
    try:
        return g.main([f.name, *args])
    finally:
        os.unlink(f.name)


class Count(unittest.TestCase):
    def test_markers_in_the_middle_of_a_line_are_read(self):
        ran, skipped, twice = g.count(RAN)
        self.assertEqual(ran, ["test_gpu_eval_sphere", "test_gpu_mc_empty"])
        self.assertEqual((skipped, twice), ([], []))

    def test_a_coloured_log_is_read(self):
        coloured = RAN.replace("ok\n", "\x1b[32mok\x1b(B\x1b[m\n").replace("gpu-test", "\x1b[0mgpu-test")
        self.assertEqual(len(g.count(coloured)[0]), 2)

    def test_the_reason_of_a_skip_is_kept(self):
        _, skipped, _ = g.count(SKIP)
        self.assertEqual(skipped, [("test_gpu_mc_empty", "device creation failed: Parent device is lost")])

    def test_enough_ran_passes(self):
        self.assertEqual(run(RAN, "--ran", "2", "--max-skipped", "0"), 0)

    def test_too_few_ran_fails(self):
        self.assertEqual(run(RAN, "--ran", "3"), 1)

    def test_a_skip_where_none_is_allowed_fails(self):
        self.assertEqual(run(RAN + SKIP.replace("test_gpu_mc_empty", "test_gpu_mc_sphere"), "--ran", "2", "--max-skipped", "0"), 1)

    def test_skips_are_reported_where_allowed(self):
        self.assertEqual(run(SKIP, "--ran", "0"), 0)

    def test_a_name_marked_twice_fails(self):
        self.assertEqual(run(RAN + RAN, "--ran", "2"), 1)

    def test_a_captured_log_compares_nothing(self):
        self.assertEqual(run("running 6 tests\ntest result: ok. 6 passed\n", "--ran", "0"), 1)


if __name__ == "__main__":
    unittest.main()
