#!/usr/bin/env python3
"""Tests for scripts/scip_scope.sh in scratch repositories with an origin/main."""
import os
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scip_scope.sh")


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                   capture_output=True)


class Scope(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        git(self.d, "init", "-q", "-b", "main")
        for rel in ("src/lib.rs", "docs/a.md", "tests/t.rs"):
            os.makedirs(os.path.join(self.d, os.path.dirname(rel)), exist_ok=True)
            open(os.path.join(self.d, rel), "w").write("x\n")
        git(self.d, "add", "-A")
        git(self.d, "commit", "-qm", "base")
        git(self.d, "update-ref", "refs/remotes/origin/main", "HEAD")

    def tearDown(self):
        shutil.rmtree(self.d)

    def write(self, rel, text="y\n"):
        os.makedirs(os.path.join(self.d, os.path.dirname(rel)), exist_ok=True)
        open(os.path.join(self.d, rel), "w").write(text)

    def run_scope(self):
        r = subprocess.run(["bash", SCRIPT], cwd=self.d, capture_output=True, text=True)
        return r.returncode, r.stdout.strip()

    def test_a_committed_src_change_runs(self):
        self.write("src/lib.rs")
        git(self.d, "commit", "-qam", "src")
        rc, out = self.run_scope()
        self.assertEqual(rc, 0, out)
        self.assertTrue(out.startswith("run: 1 file(s) under src/"), out)

    def test_an_uncommitted_or_untracked_src_change_runs(self):
        self.write("src/lib.rs")
        self.assertEqual(self.run_scope()[0], 0)
        git(self.d, "checkout", "-q", "--", "src/lib.rs")
        self.write("src/new.rs")
        self.assertEqual(self.run_scope()[0], 0)

    def test_a_docs_and_tests_only_change_skips_and_says_why(self):
        self.write("docs/a.md")
        self.write("tests/t.rs")
        git(self.d, "commit", "-qam", "docs")
        rc, out = self.run_scope()
        self.assertEqual(rc, 1, out)
        self.assertIn("skip: none of the 2 file(s)", out)

    def test_no_origin_main_runs(self):
        git(self.d, "update-ref", "-d", "refs/remotes/origin/main")
        rc, out = self.run_scope()
        self.assertEqual((rc, out), (0, "run: no merge-base with origin/main to compare with"))

    def test_no_difference_at_all_skips(self):
        rc, out = self.run_scope()
        self.assertEqual(rc, 1, out)
        self.assertIn("none of the 0 file(s)", out)


if __name__ == "__main__":
    unittest.main()
