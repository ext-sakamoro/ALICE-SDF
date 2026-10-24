#!/usr/bin/env python3
"""Tests of scripts/self_hosted_guard.py (fixture workflows written by hand)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import self_hosted_guard as g  # noqa: E402

GUARD = " && ".join(g.REQUIRED)
OK = f"""name: Unity
on:
  push:
    branches: [main, 'ci/**']
concurrency:
  group: unity-${{{{ github.ref }}}}
  cancel-in-progress: false
jobs:
  changes:
    runs-on: ubuntu-latest
    steps:
      - run: echo
  unity:
    needs: changes
    if: needs.changes.outputs.run == 'true' && {GUARD}
    runs-on: [self-hosted, windows, unity]
    steps:
      - run: echo
"""


class Guard(unittest.TestCase):
    def check(self, *texts):
        with tempfile.TemporaryDirectory() as d:
            for i, t in enumerate(texts):
                (Path(d) / f"w{i}.yml").write_text(t, encoding="utf-8")
            return g.check(Path(d))

    def test_a_guarded_job_passes(self):
        self.assertEqual(self.check(OK), ([], 1))

    def test_each_missing_condition_fails(self):
        for r in g.REQUIRED:
            with self.subTest(r=r):
                errs, _ = self.check(OK.replace(" && " + r, "").replace(r + " && ", ""))
                self.assertTrue(any("does not require" in e and r in e for e in errs), errs)

    def test_a_job_without_if_fails(self):
        t = OK.replace(f"    if: needs.changes.outputs.run == 'true' && {GUARD}\n", "")
        errs, n = self.check(t)
        self.assertEqual(n, 1)
        self.assertTrue(any("does not require" in e for e in errs), errs)

    def test_workflow_dispatch_fails(self):
        errs, _ = self.check(OK.replace("    branches: [main, 'ci/**']\n", "    branches: [main, 'ci/**']\n  workflow_dispatch:\n"))
        self.assertTrue(any("workflow_dispatch" in e for e in errs), errs)

    def test_a_cancelling_group_fails(self):
        for c in ("true", "${{ github.ref != 'refs/heads/main' }}"):
            with self.subTest(c=c):
                errs, _ = self.check(OK.replace("cancel-in-progress: false", f"cancel-in-progress: {c}"))
                self.assertTrue(any("cancel-in-progress: false" in e for e in errs), errs)
        errs, _ = self.check(OK.replace("concurrency:\n  group: unity-${{ github.ref }}\n  cancel-in-progress: false\n", ""))
        self.assertTrue(any("cancel-in-progress: false" in e for e in errs), errs)

    def test_runs_on_forms(self):
        for ro in ("self-hosted", "[self-hosted]", "{group: g, labels: [self-hosted, ue5]}"):
            with self.subTest(ro=ro):
                errs, n = self.check(OK.replace("runs-on: [self-hosted, windows, unity]", f"runs-on: {ro}"))
                self.assertEqual((errs, n), ([], 1))

    def test_only_hosted_runners_is_nothing_compared(self):
        t = OK.replace("runs-on: [self-hosted, windows, unity]", "runs-on: windows-latest")
        errs, n = self.check(t)
        self.assertEqual(n, 0)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "w.yml").write_text(t, encoding="utf-8")
            self.assertEqual(g.main([d]), 1)

    def test_a_file_outside_the_reader_subset_is_a_finding(self):
        errs, _ = self.check(OK, "on: &x\n  push: *x\n")
        self.assertTrue(any("w1.yml" in e for e in errs), errs)

    def test_this_repository_passes(self):
        errs, n = g.check(Path(__file__).resolve().parent.parent / ".github" / "workflows")
        self.assertEqual(errs, [])
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
