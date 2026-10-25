#!/usr/bin/env python3
"""Tests of scripts/check_eol.py (python3 scripts/test_check_eol.py)."""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_eol  # noqa: E402

# the fixtures' git and the checker's git read no system or global configuration (a host's init.defaultBranch
# or commit.gpgsign would otherwise decide the outcome)
os.environ.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

ENV_GIT = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false"]


def git(d: Path, *args: str) -> None:
    subprocess.run(["git", *ENV_GIT, *args], cwd=d, check=True, capture_output=True)


def run(root: Path) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = check_eol.main(["--root", str(root)])
    return code, out.getvalue()


class Eol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        git(self.d, "init", "-q", "-b", "main")

    def tearDown(self):
        self.tmp.cleanup()

    def commit(self, files: dict[str, bytes]) -> None:
        for rel, data in files.items():
            (self.d / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.d / rel).write_bytes(data)
        git(self.d, "add", "-A")
        git(self.d, "commit", "-q", "-m", "c")

    def test_lf_text_files_pass_and_are_counted(self):
        self.commit({".gitattributes": b"* text=auto eol=lf\n", "scripts/a.sh": b"echo a\n", "b.py": b"x = 1\n"})
        code, out = run(self.d)
        self.assertEqual(code, 0, out)
        self.assertIn("eol: compared 3 text files, 0 finding(s)", out)

    def test_a_crlf_checkout_of_a_text_file_fails(self):
        self.commit({"scripts/a.sh": b"echo a\n"})
        (self.d / "scripts/a.sh").write_bytes(b"echo a\r\n")
        code, out = run(self.d)
        self.assertEqual(code, 1, out)
        self.assertIn("scripts/a.sh: checked out with crlf line endings (w/crlf)", out)

    def test_a_committed_crlf_text_file_fails(self):
        self.commit({"b.py": b"x = 1\r\ny = 2\r\n"})
        code, out = run(self.d)
        self.assertEqual(code, 1, out)
        self.assertIn("b.py: committed with crlf line endings (i/crlf)", out)

    def test_mixed_line_endings_fail(self):
        self.commit({"c.toml": b"a = 1\r\nb = 2\n"})
        code, out = run(self.d)
        self.assertEqual(code, 1, out)
        self.assertIn("(i/mixed)", out)

    def test_a_minus_text_file_is_left_as_it_is(self):
        self.commit({".gitattributes": b"* text=auto eol=lf\n*.bat -text\n", "gradlew.bat": b"@echo off\r\n"})
        code, out = run(self.d)
        self.assertEqual(code, 0, out)
        self.assertIn("compared 1 text files", out)  # .gitattributes only

    def test_only_binary_files_compare_nothing(self):
        self.commit({"blob.bin": b"\x00\x01\x02\r\n"})
        code, out = run(self.d)
        self.assertEqual(code, 2, out)
        self.assertIn("compared nothing", out)

    def test_not_a_repository_is_unreadable(self):
        with tempfile.TemporaryDirectory() as t:
            code, out = run(Path(t))
        self.assertEqual(code, 2, out)
        self.assertIn("could not be read", out)


if __name__ == "__main__":
    unittest.main()
