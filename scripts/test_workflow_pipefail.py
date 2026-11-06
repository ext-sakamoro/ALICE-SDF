"""Tests of scripts/workflow_pipefail.py: a `run:` step that pipes must run with pipefail.

GitHub runs a step with no `shell:` as `bash -e {0}`: without pipefail,
`script | tee log` takes the exit status of tee, so a failing script goes green.
`shell: bash` (on the step, the job's or the workflow's `defaults.run`) runs it
as `bash --noprofile --norc -eo pipefail {0}`.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "workflow_pipefail.py"
MADE: list[Path] = []

HEAD = "name: t\non: push\n"
JOB = "jobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n"


def repo(workflows: dict[str, str]) -> Path:
    d = Path(tempfile.mkdtemp())
    MADE.append(d)
    (d / ".github" / "workflows").mkdir(parents=True)
    for name, text in workflows.items():
        (d / ".github" / "workflows" / name).write_text(text, encoding="utf-8")
    return d


def run(d: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(d / ".github" / "workflows")],
                          capture_output=True, text=True, encoding="utf-8")


class Pipefail(unittest.TestCase):
    def tearDown(self):
        while MADE:
            shutil.rmtree(MADE.pop(), ignore_errors=True)

    def test_a_pipe_with_the_default_shell_fails(self):
        r = run(repo({"w.yml": HEAD + JOB + "      - run: bash x.sh 2>&1 | tee x.log\n"}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("w.yml:", r.stdout)
        self.assertIn("pipes without pipefail", r.stdout)

    def test_a_pipe_in_a_block_scalar_fails(self):
        r = run(repo({"w.yml": HEAD + JOB + "      - name: n\n        run: |\n          echo a\n          make | tee m.log\n"}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("w.yml:", r.stdout)

    def test_workflow_defaults_shell_bash_passes(self):
        text = HEAD + "defaults:\n  run:\n    shell: bash\n" + JOB + "      - run: bash x.sh | tee x.log\n"
        r = run(repo({"w.yml": text}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_job_defaults_or_step_shell_bash_passes(self):
        job = "jobs:\n  a:\n    runs-on: ubuntu-latest\n    defaults:\n      run:\n        shell: bash\n    steps:\n"
        step = HEAD + JOB + "      - shell: bash\n        run: bash x.sh | tee x.log\n"
        r = run(repo({"j.yml": HEAD + job + "      - run: a | tee b\n", "s.yml": step}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_set_o_pipefail_in_the_step_passes(self):
        r = run(repo({"w.yml": HEAD + JOB + "      - run: |\n          set -o pipefail\n          a | tee b\n"}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_or_operators_and_quoted_bars_are_not_pipes(self):
        text = HEAD + JOB + "      - run: rustup show || rustup toolchain install\n      - run: grep -E 'a|b' f\n"
        r = run(repo({"w.yml": text}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_step_shell_without_pipefail_fails(self):
        # W3: the step names a shell, and it is not one that sets pipefail
        r = run(repo({"w.yml": HEAD + JOB + "      - shell: bash -e {0}\n        run: a | tee b\n"}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("w.yml:", r.stdout)

    def test_a_step_shell_that_sets_pipefail_passes(self):
        r = run(repo({"w.yml": HEAD + JOB + "      - shell: bash -eo pipefail {0}\n        run: a | tee b\n"}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_the_job_setting_wins_over_the_workflow_setting(self):
        job = ("jobs:\n  a:\n    runs-on: ubuntu-latest\n    defaults:\n      run:\n        shell: sh\n"
               "    steps:\n      - run: a | tee b\n")
        r = run(repo({"w.yml": HEAD + "defaults:\n  run:\n    shell: bash\n" + job}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("w.yml:", r.stdout)

    def test_the_job_setting_applies_only_to_its_own_job(self):
        jobs = ("jobs:\n  a:\n    runs-on: ubuntu-latest\n    defaults:\n      run:\n        shell: bash\n"
                "    steps:\n      - run: echo a\n"
                "  b:\n    runs-on: ubuntu-latest\n    steps:\n      - run: a | tee b\n")
        r = run(repo({"w.yml": HEAD + jobs}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("w.yml:", r.stdout)

    def test_set_o_pipefail_after_the_first_pipe_does_not_count(self):
        r = run(repo({"w.yml": HEAD + JOB + "      - run: |\n          a | tee b\n          set -o pipefail\n          c | tee d\n"}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("w.yml:", r.stdout)

    def test_case_alternation_is_not_a_pipe(self):
        script = ("      - run: |\n          rc=$?\n          case $rc in\n            0 | 2 | 3) echo ok ;;\n"
                  "            *) exit 1 ;;\n          esac\n      - run: case \"$x\" in a|b) echo y ;; esac\n")
        r = run(repo({"w.yml": HEAD + JOB + script}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_real_pipe_inside_a_case_branch_still_fails(self):
        script = "      - run: |\n          case $os in\n            linux | mac) make | tee m.log ;;\n          esac\n"
        r = run(repo({"w.yml": HEAD + JOB + script}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("w.yml:", r.stdout)

    def test_a_step_in_a_shell_that_is_not_bash_is_not_checked(self):
        # pipefail is a bash setting; a pwsh / python / cmd step pipes objects or has no pipes at all
        for shell in ("pwsh", "powershell", "python", "cmd"):
            with self.subTest(shell=shell):
                r = run(repo({"w.yml": HEAD + JOB + f"      - shell: {shell}\n        run: a | ForEach-Object {{ $_ }}\n"}))
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_job_whose_defaults_shell_is_not_bash_is_not_checked(self):
        job = ("jobs:\n  w:\n    runs-on: windows-latest\n    defaults:\n      run:\n        shell: pwsh\n"
               "    steps:\n      - run: Get-Item x | Select-Object Name\n")
        r = run(repo({"w.yml": HEAD + job}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_pipe_whose_status_is_checked_with_pipestatus_passes(self):
        text = (HEAD + JOB + "      - run: bash x.sh 2>&1 | tee x.log; exit \"${PIPESTATUS[0]}\"\n"
                "      - run: |\n          make | tee m.log\n          rc=${PIPESTATUS[0]}\n          [ $rc -eq 0 ]\n")
        r = run(repo({"w.yml": text}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_pipestatus_before_the_pipe_does_not_count(self):
        r = run(repo({"w.yml": HEAD + JOB + "      - run: |\n          echo ${PIPESTATUS[0]}\n          make | tee m.log\n"}))
        self.assertNotEqual(r.returncode, 0)

    def test_no_workflow_is_a_failure(self):
        r = run(repo({}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no workflow", r.stdout)


if __name__ == "__main__":
    unittest.main()
