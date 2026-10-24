#!/usr/bin/env python3
"""scripts/changed_paths.py の試験

The push cases run on a git repository made in a temp directory: `origin/main` is a
ref of that repository, and each case passes the SHAs GitHub would send (a push that
creates a branch sends created=true and before=0…0).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import changed_paths as cp  # noqa: E402

ZERO = "0" * 40
PATTERNS = ["**/src/**", "fuzz/**", ".github/workflows/fuzz.yml"]
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.invalid",
    "GIT_CONFIG_NOSYSTEM": "1",
}

FIXTURE_CONFIG = {"gc.auto": "0", "gc.autoDetach": "false", "maintenance.auto": "false"}


class Repo:
    def __init__(self, path: Path):
        self.path = path
        self.env = {**os.environ, **GIT_ENV, "HOME": str(path)}
        self.run("init", "-q", "-b", "main")
        self.run("config", "commit.gpgsign", "false")
        # git for Windows refuses a tab, quote or control character in an index path by
        # default (core.protectNTFS); the fixture records such names without writing them
        self.run("config", "core.protectNTFS", "false")
        # no automatic gc or maintenance: a detached one writes into .git while the fixture's
        # directory is being removed, and the cleanup fails with "Directory not empty"
        for key, value in FIXTURE_CONFIG.items():
            self.run("config", key, value)

    def run(self, *args: str) -> str:
        p = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, text=True,
                           encoding="utf-8", env=self.env, check=True)
        return p.stdout.strip()

    def commit(self, files: dict[str, str], msg: str = "c") -> str:
        for name, body in files.items():
            f = self.path / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(body, encoding="utf-8")
            self.run("add", name)
        self.run("commit", "-q", "-m", msg)
        return self.run("rev-parse", "HEAD")

    def commit_names(self, parent: str, names: list[str], msg: str = "names") -> str:
        """a commit on `parent` adding files by name, made with plumbing only: neither the
        working tree nor HEAD is touched (a tab, quote, newline or control character cannot
        be a file name on every file system, e.g. Windows)"""
        def git_in(args: list[str], data: bytes, env: dict[str, str]) -> str:
            return subprocess.run(["git", "-C", str(self.path), *args], input=data,
                                  capture_output=True, env=env, check=True).stdout.decode().strip()
        env = {**self.env, "GIT_INDEX_FILE": str(self.path / ".git" / "fixture-index")}
        blob = git_in(["hash-object", "-w", "--stdin"], b"x\n", env)
        git_in(["read-tree", parent], b"", env)
        info = b"".join(f"100644 {blob}\t".encode() + n.encode("utf-8") + b"\0" for n in names)
        git_in(["update-index", "-z", "--index-info"], info, env)
        tree = git_in(["write-tree"], b"", env)
        return git_in(["commit-tree", tree, "-p", parent, "-m", msg], b"", env)

    def checkout(self, *args: str) -> None:
        self.run("checkout", "-q", *args)

    def publish_main(self) -> str:
        """origin/main = the current main"""
        sha = self.run("rev-parse", "main")
        self.run("update-ref", "refs/remotes/origin/main", sha)
        return sha


def args(repo_: Repo | None, **kw) -> SimpleNamespace:
    base = dict(event="push", before="", after="", created="false", forced="false", base="",
                head="", patterns_file="", pattern=list(PATTERNS), main_ref="origin/main",
                repo=str(repo_.path) if repo_ else ".")
    base.update(kw)
    return SimpleNamespace(**base)


class FixtureCases(unittest.TestCase):
    def test_a_fixture_repository_runs_no_automatic_gc_or_maintenance(self):
        with tempfile.TemporaryDirectory() as d:
            r = Repo(Path(d))
            for key, value in FIXTURE_CONFIG.items():
                self.assertEqual(r.run("config", "--get", key), value, key)
            self.assertEqual(FIXTURE_CONFIG, {"gc.auto": "0", "gc.autoDetach": "false", "maintenance.auto": "false"})


class PushCases(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.r = Repo(Path(self._tmp.name))
        self.m0 = self.r.commit({"README.md": "x\n", "alice-lol/src/lib.rs": "a\n"}, "base")
        self.r.publish_main()

    def tearDown(self):
        self._tmp.cleanup()

    def branch_commit(self, files: dict[str, str], name: str = "work/x") -> str:
        self.r.checkout("-b", name, "main")
        return self.r.commit(files)

    # 1
    def test_new_branch_with_a_matching_commit_runs(self):
        c1 = self.branch_commit({"alice-lol/src/lib.rs": "b\n"})
        d = cp.decide(args(self.r, created="true", before=ZERO, after=c1))
        self.assertTrue(d.run, d.reason)
        self.assertEqual(d.files, ["alice-lol/src/lib.rs"])
        self.assertEqual(d.matched, ["alice-lol/src/lib.rs"])
        self.assertIn(f"{self.m0}..{c1}", d.reason)

    # 2: work/ was pushed first, then ci/ is created at the same SHA. GitHub's own filter
    # sees no commit new to the repository; the guard compares with merge-base(main)
    def test_twin_created_at_an_existing_sha_runs(self):
        c1 = self.branch_commit({"alice-lol/src/lib.rs": "b\n"}, "work/x")
        self.r.run("branch", "ci/x", c1)
        d = cp.decide(args(self.r, created="true", before=ZERO, after=c1))
        self.assertTrue(d.run, d.reason)
        self.assertEqual(d.matched, ["alice-lol/src/lib.rs"])
        self.assertIn("merge-base(origin/main)", d.reason)

    # 3: one push creating both twins: each push event is a created branch at the same SHA
    def test_one_push_creating_both_twins_runs_for_each(self):
        c1 = self.branch_commit({"fuzz/fuzz_targets/a.rs": "f\n", "docs/a.md": "d\n"})
        for _ref in ("refs/heads/work/x", "refs/heads/ci/x"):
            d = cp.decide(args(self.r, created="true", before=ZERO, after=c1))
            self.assertTrue(d.run, d.reason)
            self.assertEqual(d.files, ["docs/a.md", "fuzz/fuzz_targets/a.rs"])
            self.assertEqual(d.matched, ["fuzz/fuzz_targets/a.rs"])

    # 4
    def test_new_branch_touching_only_other_paths_skips(self):
        c1 = self.branch_commit({"README.md": "y\n", "docs/b.md": "z\n"})
        d = cp.decide(args(self.r, created="true", before=ZERO, after=c1))
        self.assertFalse(d.run, d.reason)
        self.assertEqual(d.files, ["README.md", "docs/b.md"])
        self.assertEqual(d.matched, [])

    # 5
    def test_follow_up_push_to_an_existing_branch(self):
        c1 = self.branch_commit({"alice-lol/src/lib.rs": "b\n"})
        c2 = self.r.commit({"alice-lol/src/new.rs": "n\n"})
        d = cp.decide(args(self.r, before=c1, after=c2))
        self.assertTrue(d.run, d.reason)
        self.assertEqual(d.files, ["alice-lol/src/new.rs"])
        c3 = self.r.commit({"README.md": "only docs\n"})
        d = cp.decide(args(self.r, before=c2, after=c3))
        self.assertFalse(d.run, d.reason)
        self.assertEqual(d.files, ["README.md"])
        self.assertIn("before..after", d.reason)

    def test_names_git_would_c_quote_still_match(self):
        # without -z git prints "alice-lol/src/a\tb.rs" with quotes and `**/src/**` misses
        odd = ["alice-lol/src/tab\there.rs", 'alice-lol/src/quo"te.rs', "alice-lol/src/back\\slash.rs",
               "alice-lol/src/new\nline.rs", "alice-lol/src/ctl\x01char.rs", "alice-lol/src/cr\rhere.rs",
               "alice-lol/src/\u00e9t\u00e9.rs"]
        for name in odd:
            with self.subTest(name=name):
                c1 = self.r.commit_names(self.m0, [name, "docs/plain.md"])
                d = cp.decide(args(self.r, created="true", before=ZERO, after=c1))
                self.assertTrue(d.run, d.reason)
                self.assertEqual(d.matched, [name])
                self.assertEqual(sorted(d.files), sorted([name, "docs/plain.md"]))

    def test_a_rename_out_of_a_matched_dir_runs(self):
        # with renames on, src/x.rs -> docs/x.rs lists docs/x.rs only and src/** misses it
        self.r.checkout("main")
        self.r.commit({"src/x.rs": "fn main() {}\n" * 20})
        self.r.publish_main()
        self.r.checkout("-b", "work/mv")
        (self.r.path / "docs").mkdir()
        self.r.run("mv", "src/x.rs", "docs/x.rs")
        self.r.run("commit", "-q", "-m", "mv")
        c1 = self.r.run("rev-parse", "HEAD")
        d = cp.decide(args(self.r, created="true", before=ZERO, after=c1, pattern=["src/**"]))
        self.assertTrue(d.run, d.reason)
        self.assertEqual(d.matched, ["src/x.rs"])

    def test_follow_up_push_with_no_file_change_skips(self):
        c1 = self.branch_commit({"alice-lol/src/lib.rs": "b\n"})
        self.r.run("commit", "-q", "--allow-empty", "-m", "empty")
        c2 = self.r.run("rev-parse", "HEAD")
        d = cp.decide(args(self.r, before=c1, after=c2))
        self.assertFalse(d.run, d.reason)
        self.assertEqual(d.files, [])
        self.assertIn("no file changed", d.reason)

    def test_force_push_compares_with_merge_base(self):
        # the branch had src + docs; amending the docs commit and force-pushing leaves a tip
        # whose only new commit (against the old tip) is docs, but the src change is still
        # in the branch against main
        self.branch_commit({"alice-lol/src/lib.rs": "b\n"})
        self.r.commit({"docs/a.md": "1\n"})
        old = self.r.run("rev-parse", "HEAD")
        self.r.run("reset", "-q", "--hard", "HEAD~1")
        new = self.r.commit({"docs/a.md": "2\n"})
        d = cp.decide(args(self.r, forced="true", before=old, after=new))
        self.assertTrue(d.run, d.reason)
        self.assertEqual(d.matched, ["alice-lol/src/lib.rs"])
        self.assertIn("forced", d.reason)

    def test_before_zero_without_created_compares_with_merge_base(self):
        c1 = self.branch_commit({"alice-lol/src/lib.rs": "b\n"})
        d = cp.decide(args(self.r, created="false", before=ZERO, after=c1))
        self.assertTrue(d.run, d.reason)
        self.assertIn("before is 0", d.reason)

    def test_new_branch_at_main_tip_runs_fail_closed(self):
        # an empty diff on a new branch says nothing about what the branch changes
        d = cp.decide(args(self.r, created="true", before=ZERO, after=self.m0))
        self.assertTrue(d.run, d.reason)
        self.assertIn("fail-closed", d.reason)

    def test_push_to_main_uses_before_after(self):
        self.r.checkout("main")
        c1 = self.r.commit({"README.md": "readme only\n"})
        d = cp.decide(args(self.r, before=self.m0, after=c1))
        self.assertFalse(d.run, d.reason)
        c2 = self.r.commit({"alice-lol/src/lib.rs": "c\n"})
        d = cp.decide(args(self.r, before=c1, after=c2))
        self.assertTrue(d.run, d.reason)

    def test_unknown_before_runs_fail_closed(self):
        c1 = self.branch_commit({"README.md": "y\n"})
        d = cp.decide(args(self.r, before="1234567890" * 4, after=c1))
        self.assertTrue(d.run, d.reason)
        self.assertIn("fail-closed", d.reason)

    def test_unknown_after_runs_fail_closed(self):
        d = cp.decide(args(self.r, created="true", before=ZERO, after="ab" * 20))
        self.assertTrue(d.run, d.reason)
        self.assertIn("fail-closed", d.reason)

    def test_missing_main_ref_runs_fail_closed(self):
        c1 = self.branch_commit({"README.md": "y\n"})
        d = cp.decide(args(self.r, created="true", before=ZERO, after=c1, main_ref="origin/nope"))
        self.assertTrue(d.run, d.reason)
        self.assertIn("fail-closed", d.reason)

    def test_unrelated_history_has_no_merge_base_and_runs(self):
        self.r.checkout("--orphan", "lonely")
        self.r.run("rm", "-rq", "--cached", ".")
        c1 = self.r.commit({"README.md": "other\n"})
        d = cp.decide(args(self.r, created="true", before=ZERO, after=c1))
        self.assertTrue(d.run, d.reason)
        self.assertIn("fail-closed", d.reason)

    def test_not_a_repository_runs_fail_closed(self):
        with tempfile.TemporaryDirectory() as d0:
            d = cp.decide(args(None, repo=d0, before="a" * 40, after="b" * 40))
        self.assertTrue(d.run, d.reason)

    def test_pull_request_uses_three_dot(self):
        self.branch_commit({"alice-lol/src/lib.rs": "b\n"})
        head = self.r.run("rev-parse", "HEAD")
        self.r.checkout("main")
        base = self.r.commit({"README.md": "main moved\n"})
        d = cp.decide(args(self.r, event="pull_request", base=base, head=head))
        self.assertTrue(d.run, d.reason)
        self.assertEqual(d.files, ["alice-lol/src/lib.rs"])  # not README.md (main's change)

    def test_main_writes_github_output_and_summary(self):
        c1 = self.branch_commit({"alice-lol/src/lib.rs": "b\n", "README.md": "z\n"})
        out, summ = Path(self._tmp.name, "out.txt"), Path(self._tmp.name, "summ.md")
        pf = Path(self._tmp.name, "pats.txt")
        pf.write_text("# comment\n\n**/src/**\n", encoding="utf-8")
        old = {k: os.environ.get(k) for k in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY")}
        os.environ["GITHUB_OUTPUT"], os.environ["GITHUB_STEP_SUMMARY"] = str(out), str(summ)
        try:
            rc = cp.main(["decide", "--event", "push", "--created", "true", "--before", ZERO,
                          "--after", c1, "--patterns-file", str(pf), "--repo", str(self.r.path)])
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual(rc, 0)
        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertIn("run=true", lines)
        self.assertIn("count=2", lines)
        self.assertIn("matched=1", lines)
        self.assertTrue(any(l.startswith("reason=run:") for l in lines))
        s = summ.read_text(encoding="utf-8")
        self.assertIn('`"alice-lol/src/lib.rs"`', s)
        self.assertIn('`"README.md"`', s)


class LogInjection(unittest.TestCase):
    """a file name must not be read as a workflow command (`::warning::` and the like)"""

    NAMES = ["::warning::x", "::add-mask::x", "::stop-commands::x", "::error::x",
             "   ::warning::lead", "\t::error::tab", "a\n::error::after-newline",
             "b\r\n::warning::crlf", "c\x1b[31m::error::ctl", "d\x00e", "src/ok.rs"]

    def output(self, d: cp.Decision) -> str:
        import contextlib
        import io
        buf = io.StringIO()
        old = {k: os.environ.pop(k, None) for k in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY")}
        try:
            with contextlib.redirect_stdout(buf):
                cp.report(d)
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v
        return buf.getvalue()

    def test_no_line_starts_a_command_except_the_stop_block(self):
        names = [n for n in self.NAMES if "\x00" not in n]
        d = cp.Decision(True, "run: x\n::error::in-reason", names, ["src/ok.rs"])
        out = self.output(d)
        lines = out.split("\n")
        opens = [l for l in lines if l.startswith("::stop-commands::")]
        self.assertEqual(len(opens), 1, out)
        token = opens[0][len("::stop-commands::"):]
        self.assertRegex(token, r"^[0-9a-f]{32}$")
        for line in lines:
            if line.lstrip().startswith("::"):
                self.assertIn(line, (f"::stop-commands::{token}", f"::{token}::"), out)
        # the list sits inside the block
        i, j = lines.index(f"::stop-commands::{token}"), lines.index(f"::{token}::")
        self.assertEqual(j - i - 1, len(names))
        # the token differs per call
        self.assertNotIn(token, self.output(d))

    def test_a_name_with_a_newline_is_one_escaped_line(self):
        out = self.output(cp.Decision(True, "r", ["a\n::error::after-newline", "c\x07bell"], []))
        self.assertIn('- "a\\n::error::after-newline"\n', out)
        self.assertIn('- "c\\u0007bell"\n', out)

    def test_github_output_has_one_line_per_key(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t, "out")
            os.environ["GITHUB_OUTPUT"] = str(f)
            try:
                cp.report(cp.Decision(True, "x\n::error::y\rz", ["a\nb"], []))
            finally:
                os.environ.pop("GITHUB_OUTPUT")
            lines = f.read_text(encoding="utf-8").split("\n")
        self.assertEqual([l.split("=")[0] for l in lines if l], ["run", "count", "matched", "reason"])


class Events(unittest.TestCase):
    def test_schedule_and_dispatch_always_run(self):
        for ev in ("schedule", "workflow_dispatch"):
            d = cp.decide(args(None, event=ev, repo="/nonexistent"))
            self.assertTrue(d.run, ev)
            self.assertIn("always runs", d.reason)

    def test_unknown_event_runs(self):
        d = cp.decide(args(None, event="merge_group"))
        self.assertTrue(d.run)
        self.assertIn("unknown event", d.reason)

    def test_bad_pattern_runs(self):
        d = cp.decide(args(None, pattern=["src/[ab].rs"], before="a" * 40, after="b" * 40))
        self.assertTrue(d.run)
        self.assertIn("patterns", d.reason)

    def test_no_pattern_runs(self):
        d = cp.decide(args(None, pattern=[], before="a" * 40, after="b" * 40))
        self.assertTrue(d.run)


class Patterns(unittest.TestCase):
    def sel(self, pats: list[str], path: str) -> bool:
        return cp.selected(path, cp.compile_patterns(pats))

    def test_double_star_spans_directories(self):
        for p in ("src/a.rs", "alice-lol/src/a.rs", "alice-lol/src/parser/mod.rs",
                  "a/b/c/src/d/e/f.rs"):
            self.assertTrue(self.sel(["**/src/**"], p), p)
        self.assertFalse(self.sel(["**/src/**"], "alice-lol/srcx/a.rs"))
        self.assertFalse(self.sel(["**/src/**"], "alice-lol/tests/src.rs"))
        self.assertTrue(self.sel(["fuzz/**"], "fuzz/fuzz_targets/deep/a.rs"))
        self.assertFalse(self.sel(["fuzz/**"], "fuzzy/a.rs"))

    def test_single_star_stays_in_one_segment(self):
        self.assertTrue(self.sel(["scripts/law_*.py"], "scripts/law_schema.py"))
        self.assertFalse(self.sel(["scripts/law_*.py"], "scripts/law_x/y.py"))
        self.assertFalse(self.sel(["scripts/law_*.py"], "scripts/test_law_schema.py"))
        self.assertTrue(self.sel(["*.md"], "README.md"))
        self.assertFalse(self.sel(["*.md"], "docs/a.md"))

    def test_question_mark_and_literal(self):
        self.assertTrue(self.sel(["a?.txt"], "ab.txt"))
        self.assertFalse(self.sel(["a?.txt"], "a/.txt"))
        self.assertTrue(self.sel([".github/workflows/fuzz.yml"], ".github/workflows/fuzz.yml"))
        self.assertFalse(self.sel([".github/workflows/fuzz.yml"], ".github/workflows/fuzzXyml"))

    def test_negation_in_order(self):
        pats = ["**/src/**", "!**/src/generated/**"]
        self.assertTrue(self.sel(pats, "a/src/x.rs"))
        self.assertFalse(self.sel(pats, "a/src/generated/x.rs"))
        # a later positive pattern selects again
        pats2 = pats + ["**/src/generated/keep.rs"]
        self.assertTrue(self.sel(pats2, "a/src/generated/keep.rs"))
        self.assertFalse(self.sel(pats2, "a/src/generated/other.rs"))

    def test_negation_skips_a_push(self):
        with tempfile.TemporaryDirectory() as t:
            r = Repo(Path(t))
            r.commit({"README.md": "x\n"})
            r.publish_main()
            r.checkout("-b", "ci/y")
            c1 = r.commit({"a/src/generated/x.rs": "g\n"})
            d = cp.decide(args(r, created="true", before=ZERO, after=c1,
                               pattern=["**/src/**", "!**/src/generated/**"]))
            self.assertFalse(d.run, d.reason)

    def test_unsupported_and_only_negated_patterns_are_rejected(self):
        for bad in (["src/[ab]"], ["a+b"], ["{a,b}"], [""], ["!src/**"], []):
            with self.assertRaises(cp.PatternError, msg=str(bad)):
                cp.compile_patterns(bad)


class Gate(unittest.TestCase):
    @staticmethod
    def needs(run: str | None, changes: str = "success", **heavy: str) -> str:
        n = {"changes": {"result": changes, "outputs": {} if run is None else {"run": run}}}
        n.update({k: {"result": v, "outputs": {}} for k, v in heavy.items()})
        return json.dumps(n)

    def test_run_and_success_passes(self):
        self.assertTrue(cp.gate(self.needs("true", fuzz="success"))[0])

    def test_run_and_not_success_fails(self):
        for r in ("skipped", "cancelled", "failure"):
            self.assertFalse(cp.gate(self.needs("true", fuzz=r))[0], r)

    def test_skip_and_skipped_passes(self):
        self.assertTrue(cp.gate(self.needs("false", fuzz="skipped"))[0])

    def test_skip_but_ran_fails(self):
        self.assertFalse(cp.gate(self.needs("false", fuzz="success"))[0])

    def test_changes_failed_or_without_output_fails(self):
        self.assertFalse(cp.gate(self.needs("true", "failure", fuzz="skipped"))[0])
        self.assertFalse(cp.gate(self.needs(None, fuzz="skipped"))[0])

    def test_no_heavy_job_fails(self):
        self.assertFalse(cp.gate(self.needs("true"))[0])

    def test_not_json_fails(self):
        self.assertFalse(cp.gate("{")[0])
        self.assertEqual(cp.main(["gate", "--needs-json", "{"]), 1)


WF_OK = """name: X
on:
  push:
    branches: [main, 'ci/**']
  pull_request:
    branches: [main]
    paths:
      - '**/src/**'
      - '.github/paths/x.txt'
jobs:
  changes:
    steps:
      - run: python3 scripts/changed_paths.py decide --patterns-file .github/paths/x.txt
  heavy:
    needs: changes
    steps:
      - run: true
  other:
    needs:
      - changes
      - heavy
    steps:
      - run: true
  gate:
    if: always()
    needs: [changes, heavy, other]
    steps:
      - run: python3 scripts/changed_paths.py gate --needs-json "$NEEDS"
"""


class Sync(unittest.TestCase):
    def tree(self, wf: str, pats: str | None = "**/src/**\n.github/paths/x.txt\n") -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / ".github" / "workflows").mkdir(parents=True)
        (root / ".github" / "workflows" / "x.yml").write_text(wf, encoding="utf-8")
        if pats is not None:
            (root / ".github" / "paths").mkdir()
            (root / ".github" / "paths" / "x.txt").write_text(pats, encoding="utf-8")
        return root

    def tearDown(self):
        if hasattr(self, "_tmp"):
            self._tmp.cleanup()

    def test_in_sync_passes(self):
        self.assertEqual(cp.sync(self.tree(WF_OK)), ([], 1))
        self.assertEqual(cp.main(["sync", str(self.tree(WF_OK))]), 0)

    def test_push_paths_fails(self):
        wf = WF_OK.replace("    branches: [main, 'ci/**']\n",
                           "    branches: [main, 'ci/**']\n    paths:\n      - 'src/**'\n")
        errs, _ = cp.sync(self.tree(wf))
        self.assertTrue(any("on.push has `paths`" in e for e in errs), errs)

    def test_push_paths_without_branches_fails(self):
        wf = WF_OK.replace("    branches: [main, 'ci/**']\n", "    paths:\n      - 'src/**'\n")
        errs, _ = cp.sync(self.tree(wf))
        self.assertTrue(any("on.push has `paths`" in e for e in errs), errs)

    def test_push_paths_on_main_only_pass(self):
        for br in ("    branches: [main]\n", "    branches:\n      - main\n"):
            wf = WF_OK.replace("    branches: [main, 'ci/**']\n", br + "    paths:\n      - 'src/**'\n")
            self.assertEqual(cp.sync(self.tree(wf)), ([], 1), br)
        wf = WF_OK.replace("    branches: [main, 'ci/**']\n",
                           "    branches:\n      - main\n      - 'ci/**'\n    paths-ignore:\n      - 'a'\n")
        errs, _ = cp.sync(self.tree(wf))
        self.assertTrue(any("on.push has `paths-ignore`" in e for e in errs), errs)

    def test_pull_request_paths_differ_fails(self):
        errs, _ = cp.sync(self.tree(WF_OK, "**/src/**\nfuzz/**\n.github/paths/x.txt\n"))
        self.assertTrue(any("differ" in e for e in errs), errs)

    def test_workflow_not_passing_the_file_fails(self):
        errs, _ = cp.sync(self.tree(WF_OK.replace("--patterns-file", "--pattern")))
        self.assertTrue(any("does not pass" in e for e in errs), errs)

    def test_pattern_file_must_list_itself(self):
        wf = WF_OK.replace("      - '.github/paths/x.txt'\n", "")
        errs, _ = cp.sync(self.tree(wf, "**/src/**\n"))
        self.assertTrue(any("does not list itself" in e for e in errs), errs)

    def test_gate_missing_a_job_fails(self):
        for need in ("[changes, heavy]", "[heavy, other]", "changes"):
            errs, _ = cp.sync(self.tree(WF_OK.replace("[changes, heavy, other]", need)))
            self.assertTrue(any("does not need" in e for e in errs), (need, errs))

    def test_gate_needs_as_a_list_passes(self):
        wf = WF_OK.replace("    needs: [changes, heavy, other]\n",
                           "    needs:\n      - changes\n      - 'heavy'\n      - other\n")
        self.assertEqual(cp.sync(self.tree(wf)), ([], 1))

    def test_no_gate_job_fails(self):
        wf = WF_OK.replace("changed_paths.py gate", "changed_paths.py gatex")
        errs, _ = cp.sync(self.tree(wf))
        self.assertTrue(any("0 jobs run" in e for e in errs), errs)

    def test_two_gate_jobs_fail(self):
        wf = WF_OK.replace("  heavy:\n    needs: changes\n    steps:\n      - run: true\n",
                           "  heavy:\n    needs: changes\n    steps:\n"
                           "      - run: python3 scripts/changed_paths.py gate --needs-json x\n")
        errs, _ = cp.sync(self.tree(wf))
        self.assertTrue(any("2 jobs run" in e for e in errs), errs)

    def test_gate_with_no_heavy_job_fails(self):
        wf = WF_OK.split("  heavy:\n")[0] + (
            "  gate:\n    needs: [changes]\n    steps:\n"
            "      - run: python3 scripts/changed_paths.py gate --needs-json x\n")
        errs, _ = cp.sync(self.tree(wf))
        self.assertTrue(any("no job besides" in e for e in errs), errs)

    def test_nothing_compared_fails(self):
        root = self.tree(WF_OK, None)
        self.assertEqual(cp.sync(root)[1], 0)
        self.assertEqual(cp.main(["sync", str(root)]), 1)

    # a YAML comment is not a key: at any indentation (column 0 included) it ends no block, so a
    # job, a push filter or a pull_request path written after it is still read

    UNGATED = "  late:\n    runs-on: x\n    steps:\n      - run: true\n"

    def test_a_job_after_a_comment_is_still_gate_covered(self):
        for comment in ("# a note\n", "  # a note\n", "      # a note\n"):
            with self.subTest(comment=comment):
                wf = WF_OK + comment + self.UNGATED
                errs, _ = cp.sync(self.tree(wf))
                self.assertTrue(any("does not need ['late']" in e for e in errs), errs)

    def test_a_comment_inside_a_job_hides_no_later_job(self):
        wf = WF_OK.replace("  heavy:\n    needs: changes\n",
                           "  heavy:\n# a note\n    needs: changes\n") + self.UNGATED
        errs, _ = cp.sync(self.tree(wf))
        self.assertTrue(any("does not need ['late']" in e for e in errs), errs)

    def test_push_paths_after_a_comment_fail(self):
        for comment in ("# a note\n", "  # a note\n", "    # a note\n"):
            with self.subTest(comment=comment):
                wf = WF_OK.replace("    branches: [main, 'ci/**']\n",
                                   f"    branches: [main, 'ci/**']\n{comment}    paths:\n      - 'src/**'\n")
                errs, _ = cp.sync(self.tree(wf))
                self.assertTrue(any("on.push has `paths`" in e for e in errs), errs)

    def test_a_push_block_after_a_comment_is_read(self):
        wf = WF_OK.replace("on:\n  push:\n", "on:\n# a note\n  push:\n").replace(
            "    branches: [main, 'ci/**']\n", "    branches: [main, 'ci/**']\n    paths:\n      - 'src/**'\n")
        errs, _ = cp.sync(self.tree(wf))
        self.assertTrue(any("on.push has `paths`" in e for e in errs), errs)

    def test_pull_request_paths_after_a_comment_are_compared(self):
        wf = WF_OK.replace("      - '**/src/**'\n", "      - '**/src/**'\n# a note\n      - 'fuzz/**'\n")
        # every item read: equal to a file that lists fuzz/** too, different from one that does not
        self.assertEqual(cp.sync(self.tree(wf, "**/src/**\nfuzz/**\n.github/paths/x.txt\n")), ([], 1))
        errs, _ = cp.sync(self.tree(wf))
        self.assertTrue(any("differ" in e for e in errs), errs)

    # a key is a key in either YAML style: a flow-style value (`paths: [...]`) is read like a block
    # list, and a form this lint cannot read fails instead of passing

    def test_flow_style_push_paths_fail(self):
        for key, val in (("paths", '["src/**"]'), ("paths", "['src/**', 'fuzz/**']"), ("paths-ignore", "[a]")):
            with self.subTest(key=key, val=val):
                wf = WF_OK.replace("    branches: [main, 'ci/**']\n",
                                   f"    branches: [main, 'ci/**']\n    {key}: {val}\n")
                errs, _ = cp.sync(self.tree(wf))
                self.assertTrue(any(f"on.push has `{key}`" in e for e in errs), errs)

    def test_flow_style_push_paths_on_main_only_pass(self):
        wf = WF_OK.replace("    branches: [main, 'ci/**']\n", "    branches: [main]\n    paths: ['src/**']\n")
        self.assertEqual(cp.sync(self.tree(wf)), ([], 1))

    def test_flow_style_pull_request_paths_are_compared(self):
        flow = "    paths: ['**/src/**', '.github/paths/x.txt']\n"
        block = "    paths:\n      - '**/src/**'\n      - '.github/paths/x.txt'\n"
        self.assertEqual(cp.sync(self.tree(WF_OK.replace(block, flow))), ([], 1))
        errs, _ = cp.sync(self.tree(WF_OK.replace(block, "    paths: ['**/src/**', 'fuzz/**', '.github/paths/x.txt']\n")))
        self.assertTrue(any("differ" in e for e in errs), errs)

    def test_an_unreadable_trigger_form_fails(self):
        for push in ("  push: {branches: [main, 'ci/**'], paths: [src]}\n",
                     "  push:\n    branches: [main, 'ci/**']\n    paths: src/**\n"):
            with self.subTest(push=push):
                wf = WF_OK.replace("  push:\n    branches: [main, 'ci/**']\n", push)
                errs, _ = cp.sync(self.tree(wf))
                self.assertTrue(errs, push)

    def test_a_push_trigger_that_is_not_a_mapping_fails(self):
        for push in ("  push: [main]\n", "  push: main\n"):
            with self.subTest(push=push):
                wf = WF_OK.replace("  push:\n    branches: [main, 'ci/**']\n", push)
                errs, _ = cp.sync(self.tree(wf))
                self.assertTrue(any("on.push is" in e for e in errs), errs)

    def test_comments_everywhere_in_a_synced_workflow_pass(self):
        wf = "\n".join(l + ("\n# c\n  # c" if l.strip() and not l.startswith("      - run") else "")
                       for l in WF_OK.split("\n"))
        self.assertEqual(cp.sync(self.tree(wf)), ([], 1), wf)


class Reader(unittest.TestCase):
    """the strict workflow reader: the tree yaml.safe_load builds (PyYAML 6.0.2 wrote each expected
    value below), FormError outside the subset, and FormError for text that is not YAML"""

    READ = [
        ('literal blank lines', 'run: |\n  a\n\n  b\n\n\nx: 1\n',
         {'run': 'a\n\nb\n', 'x': 1}),
        ('literal keep', 'run: |+\n  a\n\n\nx: 1\n',
         {'run': 'a\n\n\n', 'x': 1}),
        ('literal strip', 'run: |-\n  a\n  b\n',
         {'run': 'a\nb'}),
        ('literal more indented', 'run: |\n  a\n    b\n  c\n',
         {'run': 'a\n  b\nc\n'}),
        ('literal comment on the header', 'run: | # c\n  a\n',
         {'run': 'a\n'}),
        ('folded', 'run: >\n  a\n  b\n\n  c\n',
         {'run': 'a b\nc\n'}),
        ('folded more indented', 'run: >\n  a\n    b\n  c\n  d\n',
         {'run': 'a\n  b\nc d\n'}),
        ('folded strip', 'run: >-\n  a\n  b\n',
         {'run': 'a b'}),
        ('folded more indented then blank', 'run: >\n  a\n    b\n\n  c\n',
         {'run': 'a\n  b\n\nc\n'}),
        ('sequence of mappings', 'steps:\n  - name: a\n    run: b\n  - uses: c\n    with:\n      k: v\n',
         {'steps': [{'name': 'a', 'run': 'b'}, {'uses': 'c', 'with': {'k': 'v'}}]}),
        ("sequence at the key's indent", 'k:\n- a\n- b\nz: 1\n',
         {'k': ['a', 'b'], 'z': 1}),
        ('nested sequence', 'a:\n  - - x\n    - y\n  - z\n',
         {'a': [['x', 'y'], 'z']}),
        ('on is the key True', 'on:\n  push:\n    branches: [main]\n',
         {True: {'push': {'branches': ['main']}}}),
        ('quoted on stays a string', "'on':\n  push: null\n",
         {'on': {'push': None}}),
        ('YAML 1.1 bools and nulls', 'a: yes\nb: No\nc: off\nd: ~\ne:\n',
         {'a': True, 'b': False, 'c': False, 'd': None, 'e': None}),
        ('ints and floats', 'a: 0\nb: -12\nc: 1_000\nd: 3.12\ne: 3.10\n',
         {'a': 0, 'b': -12, 'c': 1000, 'd': 3.12, 'e': 3.1}),
        ('strings that only look typed', "a: 3.12.1\nb: v1\nc: '0123'\nd: 1e3x\n",
         {'a': '3.12.1', 'b': 'v1', 'c': '0123', 'd': '1e3x'}),
        ('colons inside plain scalars', "if: github.ref == 'refs/heads/main'\nrun: echo a:b\nurl: http://x.y/z\n",
         {'if': "github.ref == 'refs/heads/main'", 'run': 'echo a:b', 'url': 'http://x.y/z'}),
        ('hash inside and after plain scalars', "a: b#c\nb: c # d\nc: '# e'\n",
         {'a': 'b#c', 'b': 'c', 'c': '# e'}),
        ('double-quoted escapes', 'a: "x\\ty\\n\\"z\\u00e9"\n',
         {'a': 'x\ty\n"zé'}),
        ('single-quoted escape', "a: 'it''s'\n",
         {'a': "it's"}),
        ('flow collections', 'a: [x, \'y\', "z", 1, true]\nb: []\nc: {k: v, n: [1, 2]}\nd: {}\n',
         {'a': ['x', 'y', 'z', 1, True], 'b': [], 'c': {'k': 'v', 'n': [1, 2]}, 'd': {}}),
        ('flow mappings as sequence items', 'm:\n  - {os: a, t: b}\n  - { os: c }\n',
         {'m': [{'os': 'a', 't': 'b'}, {'os': 'c'}]}),
        ('colons inside flow scalars', "a: [ci/**, 'x:y', a:b]\n",
         {'a': ['ci/**', 'x:y', 'a:b']}),
        ('comments at every indentation', '# top\na:\n# c0\n  b: 1\n  # c2\n  c: 2\n',
         {'a': {'b': 1, 'c': 2}}),
        ('document start', '---\na: 1\n',
         {'a': 1}),
        ('expressions', "if: ${{ github.event_name == 'push' }}\nx: ${{ matrix.os }}\n",
         {'if': "${{ github.event_name == 'push' }}", 'x': '${{ matrix.os }}'}),
        ('a leading byte order mark is not content', '\ufeffon:\n  push:\n    paths: [src]\n',
         {True: {'push': {'paths': ['src']}}}),
        ('a block scalar that ends the file without a line break', 'n: |\n i',
         {'n': 'i'}),
        ('a kept block scalar that ends the file without a line break', 'n: |+\n i',
         {'n': 'i'}),
        ('a key that ends with a colon', 's:: 1\n',
         {'s:': 1}),
    ]

    REFUSE = [  # valid YAML outside the subset (PyYAML reads each one)
        ('an anchor and an alias', 'a: &x 1\nb: *x\n'),
        ('a merge key', 'a: {k: 1}\nb:\n  <<: {k: 2}\n'),
        ('a tag', 'a: !!str 1\n'),
        ('a multi-line plain scalar', 'a: b\n  c\n'),
        ('a multi-line quoted scalar', "a: 'b\n  c'\n"),
        ('a multi-line flow collection', 'a: [b,\n  c]\n'),
        ('an indentation indicator', 'a: |2\n   b\n'),
        ('a complex key', '? a\n: b\n'),
        ('a duplicate key', 'a: 1\na: 2\n'),
        ('an octal int', 'a: 0755\n'),
        ('a hex int', 'a: 0x1f\n'),
        ('a sexagesimal number', 'a: 1:30\n'),
        ('a float with an exponent', 'a: 1.0e+3\n'),
        ('an inf float', 'a: .inf\n'),
        ('a timestamp', 'a: 2026-10-24\n'),
        ('a tab in the indentation', 'a:\n\tb: 1\n'),
        ('a NEL line break', 'a: [x] # c\x85b: 1\n'),
        ('a line separator', 'on:\n  push:\n    branches: [x] # c\u2028    paths: [src]\n'),
        ('a paragraph separator', 'a: 1\u2029b: 2\n'),
    ]

    INVALID = [  # not YAML (PyYAML rejects each one)
        ('a colon-space inside a plain scalar', 'run: echo "a: b"\n'),
        ('a mapping value after a plain scalar', 'a: b: c\n'),
        ('a dedent that matches no level', 'a:\n  b: 1\n c: 2\n'),
        ('an unclosed flow sequence', 'a: [x\n'),
        ('an unclosed quote', "a: 'x\n"),
        ('a key after a top-level sequence', '- a\nb: 1\n'),
        ('two documents', 'a: 1\n---\nb: 2\n'),
        ('a column-0 comment inside a literal block', 'a: |\n  x\n# c\n  y\n'),
        ('a tab after the colon', 'a:\tb\n'),
        ('a tab after the colon on a later line', 'a: 1\nb:\tc\n'),
        ('a tab after a later sequence dash', 'a:\n  - 1\n  -\tx\n'),
        ('a block sequence entry inside a flow sequence', 'a: [x, - y]\n'),
        ('a leading blank line longer than the block scalar', 'a: |\n     \n  x\n'),
        ('a control character', 'a: x\x07y\n'),
        ('a NUL character', 'a: x\x00y\n'),
        ('a key that starts with a colon', 's:\n: :\n'),
        ('a value that starts with a closing brace', 'p: }\n'),
        ('a value that starts with a closing bracket', 's: ]\n'),
        ('a value that is a comma', 'l: ,\n'),
        ('a tab on a blank line of a block scalar', 'n: |\n\t\n'),
        ('a question mark inside a flow scalar', 's: [c?]\n'),
        ('a flow scalar that starts with a colon', 's: [n,:"]\n'),
    ]

    def test_reads_the_tree_yaml_builds(self):
        for name, src, want in self.READ:
            with self.subTest(name):
                self.assertEqual(cp.load_yaml(src), want)

    def test_refuses_what_is_outside_the_subset(self):
        for name, src in self.REFUSE:
            with self.subTest(name):
                with self.assertRaises(cp.FormError):
                    cp.load_yaml(src)

    def test_refuses_what_is_not_yaml(self):
        for name, src in self.INVALID:
            with self.subTest(name):
                with self.assertRaises(cp.FormError):
                    cp.load_yaml(src)

    def test_a_crlf_file_reads_like_its_lf_form(self):
        for name, src, want in self.READ:
            with self.subTest(name):
                self.assertEqual(cp.load_yaml(src.replace("\n", "\r\n")), want)
        with self.assertRaises(cp.FormError):
            cp.load_yaml("a: 1\rb: 2\n")

    def test_sync_reads_a_workflow_with_a_byte_order_mark(self):
        # the BOM is not part of the first key: `on` is still the trigger block
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / ".github" / "workflows" / "x.yml").write_text(
                "\ufeffon:\n  " + "pu" + "sh:\n    paths: [src]\n", encoding="utf-8")
            errs, _ = cp.sync(root)
            self.assertTrue(any("has `paths`" in e for e in errs), errs)

    def test_sync_reports_a_workflow_outside_the_subset(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / ".github" / "workflows" / "x.yml").write_text("on:\n  push: &p\n    branches: [a]\n",
                                                                  encoding="utf-8")
            errs, _ = cp.sync(root)
            self.assertTrue(any("strict subset" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main(verbosity=1)
