#!/usr/bin/env python3
"""Tests of scripts/public_text.sh that need no network: what it cannot know to compare
exits 2 (a malformed pin, a base that is not a commit), before anything is fetched.

The check itself (the checker at the pin on the tree and the commits) runs in the
Public text workflow and in scripts/preflight.sh.

usage: python3 scripts/test_public_text.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "public_text.sh"
GOOD_PIN = "rev " + "a" * 40 + "\nsha256 " + "b" * 64 + "\n"


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


class PublicText(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="public-text-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.r = self.tmp / "r"
        (self.r / "scripts").mkdir(parents=True)
        (self.r / ".github").mkdir()
        shutil.copy(SCRIPT, self.r / "scripts" / "public_text.sh")
        git("init", "-q", "-b", "main", cwd=self.r)
        (self.r / ".github" / "public-text-pin").write_text(GOOD_PIN)
        git("add", "-A", cwd=self.r)
        git("commit", "-qm", "c", cwd=self.r)
        # (no network: a fetch would fail with a git error, never exit 2 with these messages)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env["GIT_CONFIG_NOSYSTEM"] = "1"
        self.env["HOME"] = str(self.tmp)

    def run_script(self, *args: str, base: str | None = None) -> subprocess.CompletedProcess:
        env = dict(self.env)
        if base is not None:
            env["PUBLIC_TEXT_BASE"] = base
        return subprocess.run(["bash", "scripts/public_text.sh", *args], cwd=self.r, capture_output=True, text=True,
                              env=env, timeout=60)

    def test_a_base_that_is_not_a_commit_exits_2(self) -> None:
        for args, base in ((("nosuchref",), None), (("deadbeef" * 5,), None), ((), "nosuchref"),
                           ((), "0" * 40)):
            with self.subTest(args=args, base=base):
                p = self.run_script(*args, base=base)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn("is not a commit in this repository", p.stderr)
                self.assertNotIn("no new commit", p.stdout)

    def test_a_malformed_pin_exits_2(self) -> None:
        for pin in ("", "rev abc\nsha256 " + "b" * 64 + "\n", "rev " + "a" * 40 + "\n", "sha256 " + "b" * 64 + "\n",
                    "rev " + "A" * 40 + "\nsha256 " + "b" * 64 + "\n"):
            with self.subTest(pin=pin):
                (self.r / ".github" / "public-text-pin").write_text(pin)
                p = self.run_script(base="HEAD")
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn("needs 'rev <40 hex>' and 'sha256 <64 hex>'", p.stderr)


if __name__ == "__main__":
    unittest.main()
