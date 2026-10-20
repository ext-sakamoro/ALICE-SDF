#!/usr/bin/env python3
"""Tests of scripts/public_text.sh.

`PublicText` needs no network: what it cannot know to compare exits 2 (a malformed
pin, a base that is not a commit), before anything is fetched. `Planted` fetches the
checker at this repository's pin and shows it has teeth: a planted note link is a
finding, and the same tree without it is clean with a non-zero file count.

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



# A note link, built at run time so this file itself carries none
LINK = "[" * 2 + "x_y" + "]" * 2

# An earlier ALICE-Law revision whose checker differs from later ones, with that file's
# SHA-256: the script must fetch the revision the pin names, not a branch
EARLIER_PIN = ("rev 779fa443f95bc0caca5ca2d5bdb9156051fd6143\n"
               "sha256 e439fdd7495e45486b67d57d227f828426b703c0224b8a00d2e8a1c2f1e66594\n")


class Planted(unittest.TestCase):
    """The checker at this repository's pin, on a repository with and without a planted
    note link, in the tree and in a commit message, and the integrity of what is fetched
    (fetches ALICE-Law)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="public-text-planted-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.r = self.tmp / "r"
        (self.r / "scripts").mkdir(parents=True)
        (self.r / ".github").mkdir()
        shutil.copy(SCRIPT, self.r / "scripts" / "public_text.sh")
        shutil.copy(ROOT / ".github" / "public-text-pin", self.r / ".github" / "public-text-pin")
        (self.r / "README.md").write_text("A clean file.\n")
        git("init", "-q", "-b", "main", cwd=self.r)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def commit(self, message: str = "c") -> None:
        git("add", "-A", cwd=self.r)
        git("commit", "-qm", message, cwd=self.r)

    def run_script(self, base: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "scripts/public_text.sh", base], cwd=self.r, capture_output=True,
                              text=True, env=self.env, timeout=300)

    def check(self) -> subprocess.CompletedProcess:
        self.commit()
        return self.run_script("HEAD")

    def test_a_planted_note_link_is_a_finding(self) -> None:
        (self.r / "notes.md").write_text(f"see {LINK} for the reason\n")
        p = self.check()
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("notes.md:1:", p.stdout)

    def test_the_same_tree_without_it_is_clean(self) -> None:
        p = self.check()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertRegex(p.stdout, r"public text: 0 findings in [1-9][0-9]* files at HEAD")

    def test_a_note_link_in_a_commit_message_is_a_finding(self) -> None:
        self.commit("base")
        (self.r / "README.md").write_text("A clean file, changed.\n")
        self.commit(f"see {LINK} for the reason")
        head = git("rev-parse", "HEAD", cwd=self.r)
        p = self.run_script("HEAD~1")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn(f"commit {head[:12]} message:1:", p.stdout)
        self.assertIn("1 findings in 1 commits", p.stdout)

    def test_a_clean_commit_range_is_clean(self) -> None:
        self.commit("base")
        (self.r / "README.md").write_text("A clean file, changed.\n")
        self.commit("a clean message")
        p = self.run_script("HEAD~1")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("public text: 0 findings in 1 commits", p.stdout)

    def test_no_new_commit_reads_no_range(self) -> None:
        p = self.check()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("no new commit (base HEAD is HEAD)", p.stdout)
        self.assertNotIn("commits", p.stdout)

    def test_a_wrong_checksum_exits_2(self) -> None:
        rev = (ROOT / ".github" / "public-text-pin").read_text().split()[1]
        (self.r / ".github" / "public-text-pin").write_text(f"rev {rev}\nsha256 {'0' * 64}\n")
        p = self.check()
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertRegex(p.stderr, r"public_text_check\.py at [0-9a-f]{40} has sha256 [0-9a-f]{64}, "
                                   r"\.github/public-text-pin says 0{64}")
        self.assertNotIn("findings", p.stdout)

    def test_the_named_revision_is_fetched(self) -> None:
        (self.r / ".github" / "public-text-pin").write_text(EARLIER_PIN)
        p = self.check()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertRegex(p.stdout, r"public text: 0 findings in [1-9][0-9]* files at HEAD")


if __name__ == "__main__":
    unittest.main()
