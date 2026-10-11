#!/usr/bin/env python3
"""Oracle for scripts/workflow_concurrency.py: a branch-push workflow without the key fails,
with it passes, tag-only and push-less workflows are exempt, an empty tree fails.

Kept byte-identical in ALICE-LOL, ALICE-SDF and ALICE-DetMath (this file and
scripts/workflow_concurrency.py); change every copy together."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workflow_concurrency as wc  # noqa: E402

JOBS = "jobs:\n  a:\n    runs-on: x\n    timeout-minutes: 5\n"
PER_SHA = "github.ref == 'refs/heads/main' && github.sha"
GROUP = "x-${{ " + PER_SHA + " || github.ref }}"
CANCEL = "${{ github.ref != 'refs/heads/main' }}"
GOOD = "concurrency:\n  group: " + GROUP + "\n  cancel-in-progress: " + CANCEL + "\n"


def run(*texts: str) -> int:
    with tempfile.TemporaryDirectory() as d:
        for i, t in enumerate(texts):
            Path(d, f"w{i}.yml").write_text(t, encoding="utf-8")
        return wc.main([d])


class Concurrency(unittest.TestCase):
    def test_a_branch_push_without_a_group_fails(self):
        self.assertEqual(run("name: x\non:\n  push:\n    branches: [main]\n" + JOBS), 1)

    def test_a_push_without_filters_needs_a_group_too(self):
        self.assertEqual(run("name: x\non:\n  push:\n  pull_request:\n" + JOBS), 1)

    def test_a_branch_push_with_the_rule_passes(self):
        self.assertEqual(run("name: x\non:\n  push:\n    branches: [main]\n" + GOOD + JOBS), 0)

    def test_cancelling_on_main_fails(self):
        self.assertEqual(run("name: x\non:\n  push:\n" + GOOD.replace(CANCEL, "true") + JOBS), 1)

    def test_a_ref_only_group_fails(self):
        self.assertEqual(run("name: x\non:\n  push:\n" + GOOD.replace(GROUP, "${{ github.ref }}") + JOBS), 1)

    def test_no_cancel_key_with_a_per_commit_group_passes(self):
        no_cancel = "concurrency:\n  group: x-${{ " + PER_SHA + " || github.ref }}\n"
        self.assertEqual(run("name: x\non:\n  push:\n" + no_cancel + JOBS), 0)

    def test_a_tag_only_push_is_exempt(self):
        self.assertEqual(run("name: x\non:\n  push:\n    tags: ['v*']\n  workflow_dispatch:\n" + JOBS), 0)

    def test_a_workflow_without_push_is_exempt(self):
        self.assertEqual(run("name: x\non:\n  schedule:\n    - cron: '0 0 * * *'\n" + JOBS), 0)

    def test_one_bad_workflow_among_good_ones_fails(self):
        good = "name: x\non:\n  push:\n" + GOOD + JOBS
        bad = "name: y\non:\n  push:\n    branches: ['ci/**']\n" + JOBS
        self.assertEqual(run(good, bad), 1)

    # single self-hosted runner exemption
    SH_JOBS = "jobs:\n  a:\n    runs-on: [self-hosted, windows]\n  b:\n    runs-on: [self-hosted, windows]\n"
    REF = "concurrency:\n  group: x-${{ github.ref }}\n  cancel-in-progress: " + CANCEL + "\n"
    REASON = "# concurrency: single self-hosted runner: a run takes 100 min on one machine\n"

    def test_self_hosted_with_a_reason_and_a_ref_group_passes(self):
        self.assertEqual(run("name: x\non:\n  push:\n" + self.REASON + self.REF + self.SH_JOBS), 0)

    def test_self_hosted_without_the_reason_fails(self):
        self.assertEqual(run("name: x\non:\n  push:\n" + self.REF + self.SH_JOBS), 1)

    def test_self_hosted_with_a_too_short_reason_fails(self):
        short = "# concurrency: single self-hosted runner: one\n"
        self.assertEqual(run("name: x\non:\n  push:\n" + short + self.REF + self.SH_JOBS), 1)

    def test_self_hosted_that_cancels_main_fails(self):
        cancels = self.REF.replace(CANCEL, "true")
        self.assertEqual(run("name: x\non:\n  push:\n" + self.REASON + cancels + self.SH_JOBS), 1)

    def test_self_hosted_with_a_group_without_the_ref_fails(self):
        no_ref = self.REF.replace("x-${{ github.ref }}", "${{ github.workflow }}")
        self.assertEqual(run("name: x\non:\n  push:\n" + self.REASON + no_ref + self.SH_JOBS), 1)

    def test_mixed_hosted_and_self_hosted_jobs_are_not_exempt(self):
        mixed = "jobs:\n  a:\n    runs-on: [self-hosted, windows]\n  b:\n    runs-on: ubuntu-latest\n"
        self.assertEqual(run("name: x\non:\n  push:\n" + self.REASON + self.REF + mixed), 1)

    def test_a_hosted_workflow_with_the_reason_comment_is_not_exempt(self):
        self.assertEqual(run("name: x\non:\n  push:\n" + self.REASON + self.REF + JOBS), 1)

    def test_no_workflow_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(wc.main([d]), 1)

    def test_the_repository_passes(self):
        self.assertEqual(wc.main([]), 0)


if __name__ == "__main__":
    unittest.main()
