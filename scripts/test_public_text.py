#!/usr/bin/env python3
"""Tests of scripts/public_text.sh.

`PublicText` needs no network: what it cannot know to compare exits 2 (a malformed
pin, a base that is not a commit), before anything is fetched. `Planted` fetches the
checker at this repository's pin and shows it has teeth: a planted note link is a
finding, and the same tree without it is clean with a non-zero file count. It takes the
checker from this repository's pin (its `rev` and `sha256`) but not this repository's
settings: a pin may also approve settings (a third line `config-sha256 <sha256 of
.public-text.toml>`), and settings name this repository's own files (an allowance for a
line of its CHANGELOG is a finding where that line is absent). `SettingsApproval` plants a
pin with that line and settings of its own, so the approval is exercised in every
repository, whether or not it has settings.

The check itself (the checker at the pin on the tree and the commits) runs in the
Public text workflow and in scripts/preflight.sh.

usage: python3 scripts/test_public_text.py
"""
from __future__ import annotations

import hashlib
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


def plant_repo(source: Path, r: Path, settings: bool) -> None:
    """A repository at `r` with the script and the pin of `source`. With `settings`, the pin as it
    is and `source`'s `.public-text.toml` (which the pin's config-sha256 line approves); without,
    only the pin's checker (`rev` and `sha256`) and no settings."""
    (r / "scripts").mkdir(parents=True)
    (r / ".github").mkdir()
    shutil.copy(SCRIPT, r / "scripts" / "public_text.sh")
    pin = (source / ".github" / "public-text-pin").read_text()
    if settings:
        shutil.copy(source / ".public-text.toml", r / ".public-text.toml")
    else:
        pin = "".join(l for l in pin.splitlines(keepends=True) if l.split()[:1] in (["rev"], ["sha256"]))
    (r / ".github" / "public-text-pin").write_text(pin)
    (r / "README.md").write_text("A clean file.\n")
    git("init", "-q", "-b", "main", cwd=r)


class Planted(unittest.TestCase):
    """The checker at this repository's pin, on a repository with and without a planted
    note link, in the tree and in a commit message, and the integrity of what is fetched
    (fetches ALICE-Law)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="public-text-planted-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.r = self.tmp / "r"
        plant_repo(ROOT, self.r, settings=False)
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


class SettingsApproval(unittest.TestCase):
    """A pin with a third line `config-sha256 <sha256>` approves the settings file
    `.public-text.toml` with that hash. A source shaped that way (this repository's checker
    revision and SHA-256, settings of the test's own) is planted, so the copy of the settings and
    the checker's comparison are both exercised in every repository (fetches ALICE-Law)."""

    SETTINGS = "[public-text]\n"

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="public-text-settings-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.source = self.tmp / "source"
        (self.source / ".github").mkdir(parents=True)
        rev, checker = (ROOT / ".github" / "public-text-pin").read_text().split()[1:4:2]
        approved = hashlib.sha256(self.SETTINGS.encode()).hexdigest()
        (self.source / ".github" / "public-text-pin").write_text(
            f"rev {rev}\nsha256 {checker}\nconfig-sha256 {approved}\n")
        (self.source / ".public-text.toml").write_text(self.SETTINGS)
        self.r = self.tmp / "r"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def check(self) -> subprocess.CompletedProcess:
        git("add", "-A", cwd=self.r)
        git("commit", "-qm", "c", cwd=self.r)
        return subprocess.run(["bash", "scripts/public_text.sh", "HEAD"], cwd=self.r, capture_output=True,
                              text=True, env=self.env, timeout=300)

    def test_a_pin_that_approves_the_copied_settings_is_clean(self) -> None:
        plant_repo(self.source, self.r, settings=True)
        self.assertTrue((self.r / ".public-text.toml").is_file(), "the settings were not copied")
        p = self.check()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertRegex(p.stdout, r"public text: 0 findings in [1-9][0-9]* files at HEAD")

    def test_settings_the_pin_does_not_approve_are_a_finding(self) -> None:
        plant_repo(self.source, self.r, settings=True)
        (self.r / ".public-text.toml").write_text(self.SETTINGS + "\n")
        p = self.check()
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("config-sha256 does not match .public-text.toml", p.stdout)

    def test_the_planted_form_keeps_the_checker_and_drops_the_approval(self) -> None:
        # what Planted plants from a repository whose pin approves settings: the checker only
        plant_repo(self.source, self.r, settings=False)
        pin = (self.r / ".github" / "public-text-pin").read_text().splitlines()
        self.assertEqual([l.split()[0] for l in pin], ["rev", "sha256"])
        self.assertFalse((self.r / ".public-text.toml").exists())
        p = self.check()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertRegex(p.stdout, r"public text: 0 findings in [1-9][0-9]* files at HEAD")

    def test_a_pin_that_approves_settings_that_are_absent_is_a_finding(self) -> None:
        plant_repo(self.source, self.r, settings=True)
        (self.r / ".public-text.toml").unlink()
        p = self.check()
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("config-sha256 does not match .public-text.toml", p.stdout)


if __name__ == "__main__":
    unittest.main()
