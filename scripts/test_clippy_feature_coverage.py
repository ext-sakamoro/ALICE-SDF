#!/usr/bin/env python3
"""Tests for scripts/clippy_feature_coverage.py on small repositories."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clippy_feature_coverage as c  # noqa: E402

CARGO = '[package]\nname = "x"\nversion = "1.0.0"\n\n[features]\ndefault = []\na = []\nb = []\nw = []\n'
CI = """jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy --features "a,b,w"
  clippy:
    runs-on: ubuntu-latest
    env:
      LINT_FEATURES: "a,b"
    steps:
      - name: strict
        run: cargo clippy --all-targets --features "$LINT_FEATURES"
      - name: wasm
        run: |
          cargo clippy --lib --target wasm32-unknown-unknown --features w
  fmt:
    runs-on: ubuntu-latest
"""
PRE = "LINT_FEATURES='a,b'\n"


def run(ci=CI, cargo=CARGO, pre=PRE):
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        (r / ".github/workflows").mkdir(parents=True)
        (r / "scripts").mkdir()
        (r / "Cargo.toml").write_text(cargo)
        (r / ".github/workflows/ci.yml").write_text(ci)
        if pre is not None:
            (r / "scripts/preflight.sh").write_text(pre)
        return c.check(r)


class Coverage(unittest.TestCase):
    def test_every_feature_linted_passes(self):
        self.assertEqual(run(), ([], 3))

    def test_a_feature_only_another_job_turns_on_fails(self):
        problems, _ = run(ci=CI.replace("--features w\n", "\n"))
        self.assertEqual(len(problems), 1)
        self.assertIn("feature `w`", problems[0])

    def test_a_new_feature_in_cargo_fails(self):
        problems, _ = run(cargo=CARGO + "c = []\n")
        self.assertTrue(any("feature `c`" in p for p in problems), problems)

    def test_an_unset_variable_fails(self):
        problems, _ = run(ci=CI.replace("LINT_FEATURES: \"a,b\"", "OTHER: \"a,b\""))
        self.assertTrue(any("$LINT_FEATURES is not set" in p for p in problems), problems)

    def test_a_name_that_is_not_a_feature_fails(self):
        problems, _ = run(ci=CI.replace('"a,b"', '"a,b,zz"'), pre="LINT_FEATURES='a,b,zz'\n")
        self.assertEqual(problems, ["ci.yml job `clippy`: `zz` is not a feature of Cargo.toml"])

    def test_a_commented_command_does_not_count(self):
        problems, _ = run(ci=CI.replace("          cargo clippy --lib --target", "          # cargo clippy --lib --target"))
        self.assertTrue(any("feature `w`" in p for p in problems), problems)

    def test_preflight_drift_fails(self):
        problems, _ = run(pre="LINT_FEATURES='a'\n")
        self.assertEqual(problems, ["scripts/preflight.sh LINT_FEATURES differs from the clippy job's env LINT_FEATURES"])

    def test_no_clippy_command_compares_nothing(self):
        problems, n = run(ci=CI.replace("cargo clippy", "cargo build"))
        self.assertEqual(n, 0)
        self.assertIn("compared nothing", problems[0])

    def test_this_repository_passes(self):
        self.assertEqual(c.main([]), 0)


if __name__ == "__main__":
    unittest.main()
