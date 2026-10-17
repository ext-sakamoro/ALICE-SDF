#!/usr/bin/env python3
"""tests of scripts/law_readme_block_check.py on fixture repositories and a fixture ALICE-Law clone
(python3 scripts/test_law_readme_block_check.py)

The fixture ALICE-Law carries a scripts/gen-law-readme-block.py that answers by a marker in the
crate's README.md, so each outcome of the real generator (up to date, stale, nothing compared, no
count line) can be produced without the real one."""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import law_readme_block_check as chk  # noqa: E402

FAKE_GEN = r'''
import sys
from pathlib import Path
args = sys.argv[1:]
repo = Path(args[args.index("--crate-repo") + 1])
crate = args[args.index("--crate") + 1]
text = (repo / "README.md").read_text(encoding="utf-8")
print("args " + " ".join(args[args.index("--crate"):]))
if "MARK:ZERO" in text:
    print("law-readme-block: 0 blocks compared, 0 problems"); sys.exit(0)
if "MARK:SILENT" in text:
    sys.exit(0)
if "MARK:STALE" in text:
    print("law-readme-block: README.md: the block differs")
    print("law-readme-block: 2 blocks compared, 1 problems"); sys.exit(1)
print("law-readme-block: 2 blocks compared, 0 problems")
'''

WORKFLOW = '''name: ALICE-Law README block
jobs:
  check:
    steps:
      - name: Check the block
        run: |
          python3 "$RUNNER_TEMP/gen-law-readme-block.py" check --law "$RUNNER_TEMP/law" --crate alice-fixture \\
            --crate-repo . --law-main origin/main --lint warn
'''


def run(*args: str, cwd: Path) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(list(args), cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


class Check(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        # an ALICE-Law origin with the fake generator, and a clone of it (what the script fetches)
        src = t / "law-src"
        src.mkdir()
        run("git", "init", "-q", "-b", "main", cwd=src)
        (src / "scripts").mkdir()
        (src / "scripts" / "gen-law-readme-block.py").write_text(FAKE_GEN, encoding="utf-8")
        run("git", "add", "-A", cwd=src)
        run("git", "commit", "-q", "-m", "gen", cwd=src)
        self.pin = run("git", "rev-parse", "HEAD", cwd=src)
        run("git", "clone", "-q", str(src), str(t / "law"), cwd=t)
        self.law = t / "law"
        self.src = src
        self.repo = t / "crate"
        self.repo.mkdir()
        self.saved = os.environ.get("ALICE_LAW_DIR")
        os.environ["ALICE_LAW_DIR"] = str(self.law)

    def tearDown(self) -> None:
        if self.saved is None:
            os.environ.pop("ALICE_LAW_DIR", None)
        else:
            os.environ["ALICE_LAW_DIR"] = self.saved
        self.tmp.cleanup()

    def crate(self, readme: str | None, workflow: str | None = WORKFLOW) -> None:
        if readme is not None:
            (self.repo / "README.md").write_text(readme, encoding="utf-8")
        if workflow is not None:
            wf = self.repo / ".github" / "workflows"
            wf.mkdir(parents=True, exist_ok=True)
            (wf / "law-readme-block.yml").write_text(workflow, encoding="utf-8")

    def block(self, mark: str = "", pin: str | None = None) -> str:
        return f"# x\n<!-- law-integration:begin law={pin or self.pin} -->\n{mark}\n<!-- law-integration:end -->\n"

    def check(self) -> tuple[int, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = chk.main(["--root", str(self.repo)])
        return rc, out.getvalue() + err.getvalue()

    def test_an_up_to_date_block_passes_with_the_job_arguments(self):
        self.crate(self.block())
        rc, out = self.check()
        self.assertEqual(rc, 0, out)
        self.assertIn("args --crate alice-fixture --crate-repo", out)
        self.assertIn("--law-main origin/main --lint warn", out)
        self.assertIn("2 block(s) compared, ok", out)

    def test_a_pass_names_uncommitted_inputs(self):
        self.crate(self.block())
        run("git", "init", "-q", cwd=self.repo)
        run("git", "add", "-A", cwd=self.repo)
        run("git", "commit", "-q", "-m", "c", cwd=self.repo)
        rc, out = self.check()
        self.assertEqual(rc, 0, out)
        self.assertNotIn("not covered", out)
        (self.repo / "README.md").write_text(self.block() + "edit\n", encoding="utf-8")
        rc, out = self.check()
        self.assertEqual(rc, 0, out)
        self.assertIn("uncommitted changes to README.md are not covered", out)

    def test_a_stale_block_fails(self):
        self.crate(self.block("MARK:STALE"))
        rc, out = self.check()
        self.assertEqual(rc, 1, out)
        self.assertIn("differs", out)

    def test_neither_workflow_nor_block_is_not_applicable(self):
        self.crate("# no block\n", workflow=None)
        rc, out = self.check()
        self.assertEqual(rc, 0, out)
        self.assertIn("n/a", out)

    def test_a_workflow_without_a_block_fails(self):
        self.crate("# no block\n")
        rc, out = self.check()
        self.assertEqual(rc, 1, out)
        self.assertIn("no law-integration block", out)

    def test_a_block_without_a_workflow_fails(self):
        self.crate(self.block(), workflow=None)
        rc, out = self.check()
        self.assertEqual(rc, 1, out)

    def test_zero_compared_is_not_a_pass(self):
        self.crate(self.block("MARK:ZERO"))
        rc, out = self.check()
        self.assertEqual(rc, 1, out)
        self.assertIn("0 blocks compared", out)

    def test_no_count_line_is_not_a_pass(self):
        self.crate(self.block("MARK:SILENT"))
        rc, out = self.check()
        self.assertEqual(rc, 1, out)
        self.assertIn("nothing is shown compared", out)

    def test_a_pin_the_clone_does_not_have_fails(self):
        self.crate(self.block(pin="0" * 40))
        rc, out = self.check()
        self.assertEqual(rc, 1, out)
        self.assertIn("has no commit", out)

    def test_a_pin_newer_than_the_clone_is_fetched(self):
        # a commit made on the origin after the clone: the fetch brings it in
        (self.src / "scripts" / "gen-law-readme-block.py").write_text(FAKE_GEN + "\n# v2\n", encoding="utf-8")
        run("git", "commit", "-q", "-am", "v2", cwd=self.src)
        newer = run("git", "rev-parse", "HEAD", cwd=self.src)
        self.crate(self.block(pin=newer))
        rc, out = self.check()
        self.assertEqual(rc, 0, out)

    def test_an_unreadable_workflow_is_an_error(self):
        self.crate(self.block(), workflow="name: ALICE-Law README block\njobs: {}\n")
        rc, out = self.check()
        self.assertEqual(rc, 2, out)

    def test_a_failing_fetch_fails(self):
        run("git", "remote", "set-url", "origin", str(Path(self.tmp.name) / "gone"), cwd=self.law)
        self.crate(self.block())
        rc, out = self.check()
        self.assertEqual(rc, 1, out)
        self.assertIn("cannot fetch", out)

    def test_the_workflow_pattern_reads_every_repository_s_job(self):
        # the job text of this repository is what the script mirrors
        wf = Path(__file__).resolve().parents[1] / chk.WORKFLOW
        if wf.exists():
            m = chk.CHECK_RE.search(wf.read_text(encoding="utf-8"))
            self.assertIsNotNone(m, f"{wf}: the check command is not readable")
            self.assertEqual(m["law_main"], "origin/main")
            self.assertIn(m["lint"], ("warn", "error"), "a repository picks its own lint level")


if __name__ == "__main__":
    unittest.main()
