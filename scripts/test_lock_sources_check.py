#!/usr/bin/env python3
"""Tests of scripts/lock_sources_check.py"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock_sources_check as c  # noqa: E402

OWN = {"alice-zip"}
GIT = 'source = "git+https://github.com/ext-sakamoro/ALICE-DetMath?branch=main#0123"'


def lock(*pkgs: str) -> str:
    return "version = 4\n" + "".join(f"\n[[package]]\n{p}\n" for p in pkgs)


class LockSources(unittest.TestCase):
    def test_git_and_own_path_pass(self):
        text = lock(f'name = "alice-det-math"\nversion = "0.6.1"\n{GIT}', 'name = "alice-zip"\nversion = "0.11.0"')
        self.assertEqual(c.findings("Cargo.lock", text, OWN), [])

    def test_registry_alice_fails(self):
        text = lock('name = "alice-codec"\nversion = "0.1.3"\nsource = "registry+https://github.com/rust-lang/crates.io-index"')
        self.assertEqual(len(c.findings("Cargo.lock", text, OWN)), 1)

    def test_foreign_path_alice_fails(self):
        text = lock('name = "alice-codec"\nversion = "0.2.0"')
        self.assertEqual(len(c.findings("Cargo.lock", text, OWN)), 1)

    def test_two_det_math_fail(self):
        text = lock(f'name = "alice-det-math"\nversion = "0.6.1"\n{GIT}',
                    f'name = "alice-det-math"\nversion = "0.6.0"\n{GIT.replace("0123", "4567")}')
        self.assertIn("more than one alice-det-math", " ".join(c.findings("Cargo.lock", text, OWN)))

    def test_registry_libasp_fails(self):
        text = lock('name = "libasp"\nversion = "1.0.0"\nsource = "registry+https://github.com/rust-lang/crates.io-index"')
        self.assertEqual(len(c.findings("Cargo.lock", text, OWN)), 1)

    def test_non_alice_ignored(self):
        text = lock('name = "serde"\nversion = "1.0.0"\nsource = "registry+https://github.com/rust-lang/crates.io-index"')
        self.assertEqual(c.findings("Cargo.lock", text, OWN), [])

    def test_two_det_math_in_one_lock_fail_even_when_one_is_exempt_for_another_crate(self):
        text = lock(f'name = "alice-det-math"\nversion = "0.6.1"\n{GIT}',
                    f'name = "alice-det-math"\nversion = "0.6.0"\n{GIT.replace("0123", "4567")}')
        self.assertTrue(c.findings("Cargo.lock", text, OWN, frozenset({"alice-zip"})))
        self.assertEqual(c.findings("Cargo.lock", text, OWN, frozenset({"alice-det-math"})), [])


ZIP_A = 'source = "git+https://example.invalid/ALICE-Zip?branch=main#aaaa"'
ZIP_B = 'source = "git+https://example.invalid/ALICE-Zip?branch=main#bbbb"'
REG = 'source = "registry+https://github.com/rust-lang/crates.io-index"'
EXEMPT = """[[exempt]]
lock = "compat/Cargo.lock"
crate = "alice-zip"
reason = "the old reader reads data written by the released crate"
test = "compat/tests/old_reader.rs"
"""


def pkg(name: str, version: str, source: str = "") -> str:
    return f'name = "{name}"\nversion = "{version}"' + (f"\n{source}" if source else "")


class Repository(unittest.TestCase):
    """The whole check on throwaway git repositories (tracked locks only)"""

    def repo(self, files: dict[str, str]) -> Path:
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(d)], check=False))
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        for rel, text in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(text, encoding="utf-8")
        subprocess.run(["git", "-C", str(d), "add", "."], check=True)
        return d

    def run_on(self, d: Path) -> int:
        return c.main(["--repo", str(d)])

    def test_no_lock_is_exit_2(self):
        self.assertEqual(self.run_on(self.repo({"README.md": "x\n"})), 2)

    def test_unknown_arguments_are_exit_2(self):
        self.assertEqual(c.main(["--skip", "compat/"]), 2)

    def test_the_same_source_in_every_lock_passes(self):
        d = self.repo({"Cargo.lock": lock(pkg("alice-zip", "0.11.0", ZIP_A)),
                       "fuzz/Cargo.lock": lock(pkg("alice-zip", "0.11.0", ZIP_A))})
        self.assertEqual(c.check(d), ([], 2, 1))
        self.assertEqual(self.run_on(d), 0)

    def test_a_lock_left_on_an_older_commit_fails_and_names_both(self):
        d = self.repo({"Cargo.lock": lock(pkg("alice-zip", "0.11.0", ZIP_B)),
                       "fuzz/Cargo.lock": lock(pkg("alice-zip", "0.11.0", ZIP_A))})
        bad, locks, compared = c.check(d)
        self.assertEqual((locks, compared, len(bad)), (2, 1, 1))
        self.assertIn("fuzz/Cargo.lock", bad[0])
        self.assertEqual(self.run_on(d), 1)

    def test_an_untracked_lock_is_not_read(self):
        d = self.repo({"Cargo.lock": lock(pkg("alice-zip", "0.11.0", ZIP_A))})
        (d / "tools").mkdir()
        (d / "tools/Cargo.lock").write_text(lock(pkg("alice-zip", "0.11.0", ZIP_B)), encoding="utf-8")
        self.assertEqual(self.run_on(d), 0)

    def test_locks_sharing_no_git_crate_pass_and_compare_nothing(self):
        d = self.repo({"Cargo.lock": lock(pkg("alice-zip", "0.11.0", ZIP_A)),
                       "fuzz/Cargo.lock": lock(pkg("serde", "1.0.0", REG))})
        self.assertEqual(c.check(d), ([], 2, 0))
        self.assertEqual(self.run_on(d), 0)

    def compat(self, exempt: str, compat_zip: str = REG, test_file: bool = True) -> dict[str, str]:
        files = {"Cargo.lock": lock(pkg("alice-zip", "0.11.0", ZIP_A)),
                 "compat/Cargo.lock": lock(pkg("alice-zip", "0.10.0", compat_zip)),
                 "scripts/lock_sources.toml": exempt}
        if test_file:
            files["compat/tests/old_reader.rs"] = "// reads the released format\n"
        return files

    def test_an_exempt_crate_is_left_out_of_both_checks(self):
        d = self.repo(self.compat(EXEMPT))
        self.assertEqual(c.check(d), ([], 2, 0))

    def test_without_the_exemption_the_registry_crate_fails(self):
        self.assertEqual(self.run_on(self.repo(self.compat(""))), 1)

    def test_an_exemption_nothing_needs_fails(self):
        d = self.repo(self.compat(EXEMPT, compat_zip=ZIP_A))
        bad = c.check(d)[0]
        self.assertEqual(len(bad), 1)
        self.assertIn("nothing fails without it", bad[0])

    def test_an_exemption_for_a_crate_not_in_the_lock_fails(self):
        d = self.repo(self.compat(EXEMPT.replace('crate = "alice-zip"', 'crate = "alice-db"')))
        self.assertTrue(any("no such crate" in f for f in c.check(d)[0]))

    def test_an_exemption_without_its_test_file_fails(self):
        d = self.repo(self.compat(EXEMPT, test_file=False))
        self.assertTrue(any("is not a file here" in f for f in c.check(d)[0]))

    def test_an_exemption_without_a_reason_fails(self):
        d = self.repo(self.compat(EXEMPT.replace("the old reader reads data written by the released crate", "")))
        self.assertTrue(any("missing reason" in f for f in c.check(d)[0]))

    def test_an_exempt_lock_that_disagrees_by_commit_is_left_out(self):
        d = self.repo(self.compat(EXEMPT, compat_zip=ZIP_B))
        self.assertEqual(c.check(d)[0], [])


if __name__ == "__main__":
    unittest.main()
