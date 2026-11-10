#!/usr/bin/env python3
"""Tests of scripts/lock_sources_check.py"""
from __future__ import annotations

import sys
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

    def test_no_lock_compared_is_exit_2(self):
        self.assertEqual(c.main(["--skip", ""]), 2)


if __name__ == "__main__":
    unittest.main()
