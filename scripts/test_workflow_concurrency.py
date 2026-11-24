#!/usr/bin/env python3
"""Oracle for scripts/workflow_concurrency.py: a branch-push workflow without the key fails,
with it passes, tag-only and push-less workflows are exempt, an empty tree fails.

Shared byte-identical by the ALICE crates that carry it (this file and
scripts/workflow_concurrency.py, ALICE-Law's shared table); change every copy together."""
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

    # a self-hosted workflow that runs on ci/** through scripts/changed_paths.py: its `changes` job
    # (decide) and its gate job run on a hosted runner, every other job on the self-hosted one
    CHANGES = ("  changes:\n    runs-on: ubuntu-latest\n    steps:\n"
               "      - run: python3 scripts/changed_paths.py decide --patterns-file .github/paths/x.txt\n")
    HOSTED_GATE = ("  gate:\n    runs-on: ubuntu-latest\n    steps:\n"
                   "      - run: python3 scripts/changed_paths.py gate --needs-json \"$NEEDS\"\n")
    SH_JOB = "  a:\n    runs-on: [self-hosted, windows]\n"

    def wf(self, jobs: str) -> str:
        return "name: x\non:\n  push:\n" + self.REASON + self.REF + "jobs:\n" + jobs

    def test_self_hosted_with_the_changes_and_gate_jobs_passes(self):
        self.assertEqual(run(self.wf(self.CHANGES + self.SH_JOB + self.HOSTED_GATE)), 0)

    def test_a_hosted_job_besides_changes_and_the_gate_is_not_exempt(self):
        extra = "  b:\n    runs-on: ubuntu-latest\n"
        self.assertEqual(run(self.wf(self.CHANGES + self.SH_JOB + extra + self.HOSTED_GATE)), 1)

    def test_a_changes_job_that_does_not_decide_is_not_exempt(self):
        changes = self.CHANGES.replace("changed_paths.py decide", "something_else.py")
        self.assertEqual(run(self.wf(changes + self.SH_JOB + self.HOSTED_GATE)), 1)

    def test_a_gate_job_that_does_not_gate_is_not_exempt(self):
        gate = self.HOSTED_GATE.replace("changed_paths.py gate", "changed_paths.py sync")
        self.assertEqual(run(self.wf(self.CHANGES + self.SH_JOB + gate)), 1)

    POLICY = ("      - name: policy\n        run: |\n"
              "          if [ \"$DECIDED\" = true ] && [ \"$ALLOWED\" = true ]; then run=true; else run=false; fi\n"
              "          echo \"decided=$DECIDED allowed=$ALLOWED run=$run\"\n"
              "          echo \"run=$run\" >> \"$GITHUB_OUTPUT\"\n")

    def test_a_changes_job_with_the_policy_step_is_exempt(self):
        self.assertEqual(run(self.wf(self.CHANGES + self.POLICY + self.SH_JOB + self.HOSTED_GATE)), 0)

    def test_a_changes_job_with_another_step_is_not_exempt(self):
        for extra in (self.POLICY.replace("run=false", "run=true"), "      - run: cargo build\n"):
            with self.subTest(extra=extra):
                self.assertEqual(run(self.wf(self.CHANGES + extra + self.SH_JOB + self.HOSTED_GATE)), 1)

    # a decision-only step from DECISION_SCRIPTS: the exact command shape, reading the diff / the
    # environment and writing $GITHUB_OUTPUT only (scripts/unreal_engines.py picks the engines)
    ENGINES = ("      - name: engines\n"
               "        run: python3 scripts/unreal_engines.py --roots \"$ROOTS\" --event \"$EVENT\""
               " --ref \"$REF\" --specific \"$SPECIFIC\"\n")
    SECOND_DECIDE = ("      - name: engine-specific change\n        run: >-\n"
                     "          python3 scripts/changed_paths.py decide --event \"$EVENT\"\n"
                     "          --patterns-file .github/unreal-engines-both.txt\n")

    def test_two_decides_with_the_policy_are_exempt(self):
        jobs = self.CHANGES + self.POLICY + self.SECOND_DECIDE
        self.assertEqual(run(self.wf(jobs + self.SH_JOB + self.HOSTED_GATE)), 0)

    def test_a_listed_decision_script_is_exempt(self):
        jobs = self.CHANGES + self.POLICY + self.SECOND_DECIDE + self.ENGINES
        self.assertEqual(run(self.wf(jobs + self.SH_JOB + self.HOSTED_GATE)), 0)

    def test_a_decision_script_in_another_shape_is_not_exempt(self):
        for engines in (self.ENGINES.replace(' --specific "$SPECIFIC"', ""),
                        self.ENGINES.replace('"$SPECIFIC"\n', '"$SPECIFIC"; cargo build\n'),
                        self.ENGINES.replace('"$SPECIFIC"\n', '"$SPECIFIC" && curl -sO https://x\n'),
                        self.ENGINES.replace("--roots", "--roots $(id)"),
                        self.ENGINES.replace("unreal_engines.py", "unreal_engines.py --extra")):
            with self.subTest(engines=engines):
                jobs = self.CHANGES + self.POLICY + engines
                self.assertEqual(run(self.wf(jobs + self.SH_JOB + self.HOSTED_GATE)), 1)

    def test_an_unlisted_script_or_a_build_step_is_not_exempt(self):
        for extra in ("      - run: python3 scripts/other_choice.py --roots \"$ROOTS\"\n",
                      "      - run: cargo build --release\n",
                      "      - run: curl -sSfLO https://example.invalid/x.tar.gz\n",
                      "      - run: pip install pyyaml\n"):
            with self.subTest(extra=extra):
                jobs = self.CHANGES + self.POLICY + self.ENGINES + extra
                self.assertEqual(run(self.wf(jobs + self.SH_JOB + self.HOSTED_GATE)), 1)

    def test_a_decision_script_without_a_decide_is_not_exempt(self):
        changes = "  changes:\n    runs-on: ubuntu-latest\n    steps:\n" + self.ENGINES
        self.assertEqual(run(self.wf(changes + self.SH_JOB + self.HOSTED_GATE)), 1)

    def test_a_decide_with_a_command_substitution_is_not_exempt(self):
        changes = self.CHANGES.replace("--patterns-file", "--patterns-file $(cargo build)")
        self.assertEqual(run(self.wf(changes + self.SH_JOB + self.HOSTED_GATE)), 1)

    def test_a_repeated_policy_or_decision_step_is_not_exempt(self):
        for extra in (self.POLICY, self.ENGINES):
            with self.subTest(extra=extra):
                jobs = self.CHANGES + self.POLICY + self.ENGINES + extra
                self.assertEqual(run(self.wf(jobs + self.SH_JOB + self.HOSTED_GATE)), 1)

    def test_a_job_that_only_mentions_the_gate_is_not_exempt(self):
        gate = self.HOSTED_GATE.replace("      - run: python3", "      - run: echo changed_paths.py gate; python3")
        self.assertEqual(run(self.wf(self.CHANGES + self.SH_JOB + gate)), 1)
        two = self.HOSTED_GATE + "      - run: cargo build\n"
        self.assertEqual(run(self.wf(self.CHANGES + self.SH_JOB + two)), 1)

    def test_a_workflow_outside_the_reader_subset_fails(self):
        self.assertEqual(run("name: x\non: &t\n  push:\n" + GOOD + JOBS), 1)

    def test_changes_and_gate_without_a_self_hosted_job_are_not_exempt(self):
        self.assertEqual(run(self.wf(self.CHANGES + self.HOSTED_GATE)), 1)

    def test_a_hosted_workflow_with_the_reason_comment_is_not_exempt(self):
        self.assertEqual(run("name: x\non:\n  push:\n" + self.REASON + self.REF + JOBS), 1)

    def test_without_the_reader_the_check_fails(self):
        # scripts/changed_paths.py missing: no fallback to reading lines
        saved, wc.cp = wc.cp, None
        try:
            self.assertEqual(run("name: x\non:\n  pu" "sh:\n" + GOOD + JOBS), 1)
        finally:
            wc.cp = saved

    def test_no_workflow_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(wc.main([d]), 1)

    def test_the_repository_passes(self):
        self.assertEqual(wc.main([]), 0)



# a run of one event never shares a concurrency group with a run of another event when either may
# cancel: a manual run (workflow_dispatch) must not cancel the push run of the same commit
DISPATCH = "  workflow_dispatch:\n"
SCHEDULE = "  schedule:\n    - cron: '0 0 * * 1'\n"
KEYED = ("x-${{ github.event_name == 'workflow_dispatch' && format('dispatch-{0}', github.run_id) || ("
         + PER_SHA + " || github.ref) }}")
KEYED_GOOD = GOOD.replace(GROUP, KEYED)


def events_of(text: str) -> dict:
    """the group and cancel value of each event, for the assertions below"""
    data = wc.cp.load_yaml(text)
    return {e.name: wc.resolve(data["concurrency"], data, e) for e in wc.EVENTS}


class EventGroups(unittest.TestCase):
    def test_a_dispatch_that_shares_the_push_group_on_a_branch_fails(self):
        self.assertEqual(run("name: x\non:\n  push:\n" + DISPATCH + GOOD + JOBS), 1)

    def test_an_event_keyed_group_passes(self):
        self.assertEqual(run("name: x\non:\n  push:\n" + DISPATCH + SCHEDULE + KEYED_GOOD + JOBS), 0)

    def test_the_keyed_group_still_meets_the_per_commit_rule_on_main(self):
        r = events_of("name: x\non:\n  push:\n" + DISPATCH + KEYED_GOOD + JOBS)
        self.assertNotEqual(r["push ci"][0], r["workflow_dispatch ci"][0])
        self.assertNotEqual(r["push main"][0], r["workflow_dispatch main"][0])
        self.assertEqual(r["push ci"][1], True)
        self.assertEqual(r["push main"][1], False)
        self.assertTrue(r["push main"][0].endswith(wc.SHA_MAIN))

    def test_dropping_the_event_key_is_red(self):
        # the mutant of the keyed group: the same text without the event branch
        mutant = KEYED_GOOD.replace("github.event_name == 'workflow_dispatch' && format('dispatch-{0}', "
                                    "github.run_id) || ", "")
        self.assertEqual(run("name: x\non:\n  push:\n" + DISPATCH + mutant + JOBS), 1)

    def test_a_dispatch_and_schedule_workflow_without_push_is_checked_too(self):
        conc = "concurrency:\n  group: w-${{ github.ref }}\n  cancel-in-progress: true\n"
        self.assertEqual(run("name: x\non:\n" + DISPATCH + SCHEDULE + conc + JOBS), 1)

    def test_a_shared_group_that_never_cancels_passes(self):
        conc = "concurrency:\n  group: w-${{ github.ref }}\n  cancel-in-progress: false\n"
        self.assertEqual(run("name: x\non:\n" + DISPATCH + SCHEDULE + conc + JOBS), 0)

    def test_a_ci_push_that_can_cancel_a_main_run_through_a_shared_group_fails(self):
        conc = "concurrency:\n  group: w-${{ github.workflow }}\n  cancel-in-progress: true\n"
        self.assertEqual(run("name: x\non:\n" + DISPATCH + SCHEDULE + conc + JOBS), 1)
        # a manual run on a branch and one on main, same group, may cancel: the same rule
        conc = "concurrency:\n  group: w-${{ github.event_name }}\n  cancel-in-progress: true\n"
        self.assertEqual(run("name: x\non:\n" + DISPATCH + SCHEDULE + conc + JOBS), 1)

    def test_a_push_only_workflow_has_no_pair_to_compare(self):
        self.assertEqual(run("name: x\non:\n  push:\n" + GOOD + JOBS), 0)

    def test_a_push_guarded_group_and_cancel_pass(self):
        # the shape one repository already uses: only a push may cancel, a manual run has its run id
        conc = ("concurrency:\n  group: x-${{ github.event_name == 'push' && (" + PER_SHA
                + " || github.ref) || github.run_id }}\n  cancel-in-progress: ${{ github.event_name == 'push' "
                "&& github.ref != 'refs/heads/main' }}\n")
        self.assertEqual(run("name: x\non:\n  push:\n" + DISPATCH + conc + JOBS), 0)

    def test_a_group_naming_an_unknown_context_fails(self):
        conc = GOOD.replace(GROUP, "x-${{ github.actor }}")
        self.assertEqual(run("name: x\non:\n  push:\n" + DISPATCH + conc + JOBS), 1)

    def test_an_input_keyed_group_is_read_as_the_manual_run_s_value(self):
        conc = "concurrency:\n  group: verify-${{ inputs.sha }}\n  cancel-in-progress: false\n"
        text = "name: x\non:\n" + DISPATCH + SCHEDULE + conc + JOBS
        self.assertEqual(run(text), 0)
        r = events_of(text)
        # inputs.* is the manual run's value, empty for a schedule run
        self.assertEqual(r["schedule main"][0], "verify-")
        self.assertNotEqual(r["workflow_dispatch main"][0], "verify-")


class Expressions(unittest.TestCase):
    CTX = {"github.ref": "refs/heads/ci/x", "github.sha": "s", "github.event_name": "push"}

    def ev(self, expr):
        return wc.evaluate(expr, self.CTX)

    def test_and_or_return_an_operand(self):
        self.assertEqual(self.ev("github.ref == 'refs/heads/main' && github.sha || github.ref"), "refs/heads/ci/x")
        self.assertEqual(self.ev("github.ref != 'refs/heads/main' && 'a' || 'b'"), "a")

    def test_comparison_ignores_case_and_not_negates(self):
        self.assertIs(self.ev("github.event_name == 'PUSH'"), True)
        self.assertIs(self.ev("!(github.event_name == 'push')"), False)

    def test_format_fills_its_slots(self):
        self.assertEqual(self.ev("format('d-{0}-{1}', github.sha, 'z')"), "d-s-z")

    def test_a_malformed_expression_is_an_error(self):
        for bad in ("github.ref ==", "(github.ref", "format('{0}'", "github.ref & github.sha", "'open"):
            with self.subTest(bad=bad), self.assertRaises(wc.EvalError):
                self.ev(bad)

    def test_text_outside_the_markers_is_kept(self):
        self.assertEqual(wc.interpolate("a-${{ github.sha }}-b", self.CTX), "a-s-b")
        # two markers: the whole-value form does not stretch one marker over the text between them
        self.assertEqual(wc.interpolate("${{ github.sha }}-${{ github.event_name }}", self.CTX), "s-push")

if __name__ == "__main__":
    unittest.main()
