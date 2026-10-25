#!/usr/bin/env python3
"""Tests of scripts/check_pipe_grep_q.py: the forms it must catch and the ones it must leave"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "check_pipe_grep_q", pathlib.Path(__file__).resolve().parent / "check_pipe_grep_q.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class Caught(unittest.TestCase):
    def test_pipe_into_grep_q(self):
        for line in [
            "rustup target list --installed | grep -q '^wasm32$' || rustup target add wasm32",
            'printf \'%s\\n\' "$list" | grep -qx "$f" || exit 1',
            "cmd | grep -xq pat",
            "cmd |grep -Eq 'a|b'",
            "cmd | grep --quiet pat",
            "cmd | grep -E --silent pat",
            "has() { rustup toolchain list | grep -q \"^$1\"; }",
        ]:
            with self.subTest(line=line):
                self.assertEqual(len(check.offending(line)), 1)


class Left(unittest.TestCase):
    def test_other_forms(self):
        for line in [
            'grep -q "^$1" <<<"$(rustup toolchain list)"',
            'grep -qx "$f" <<<"$list"',
            "cmd | grep -c pat",
            "cmd | grep pat > out",
            "grep -q pat file",
            "# a comment: cmd | grep -q pat",
            "echo done  # cmd | grep -q pat",
        ]:
            with self.subTest(line=line):
                self.assertEqual(check.offending(line), [])

    def test_line_numbers(self):
        self.assertEqual(check.offending("ok\ncmd | grep -q x\n")[0][0], 2)


class Cli(unittest.TestCase):
    """The checker run as CI runs it, on its own copy in a throwaway git repository"""

    def run_in(self, files: dict[str, str]) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "scripts").mkdir()
            shutil.copy(pathlib.Path(check.__file__), root / "scripts" / "check_pipe_grep_q.py")
            for rel, text in files.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text, encoding="utf-8", newline="")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
            return subprocess.run([sys.executable, "scripts/check_pipe_grep_q.py"], cwd=root,
                                  capture_output=True, text=True)

    def test_a_pipe_into_grep_q_fails_and_is_named(self):
        r = self.run_in({"ci/gate.sh": "ok\nrustup target list | grep -q wasm32\n"})
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("ci/gate.sh:2:", r.stderr)
        self.assertIn("1 pipelines end in grep -q", r.stdout)
        if sys.platform == "win32":
            # paths are printed with '/', the same on every OS
            self.assertNotIn("\\", r.stderr + r.stdout)

    def test_a_clean_tree_passes(self):
        r = self.run_in({"ci/gate.sh": 'grep -q wasm32 <<<"$(rustup target list)"\n'})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1 files read, 0 pipelines", r.stdout)

    def test_no_file_read_fails(self):
        r = self.run_in({"README.md": "cmd | grep -q x\n"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("read no file", r.stderr)


if __name__ == "__main__":
    unittest.main()
