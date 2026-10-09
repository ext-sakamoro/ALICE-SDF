#!/usr/bin/env python3
"""Pins the `cargo semver-checks` check-count extraction against coloured output.

security-audit.yml (semver-checks job) and scripts/preflight.sh (`checks_ran`)
read the number of checks that ran with

    grep -oE '[0-9]+ checks: ' <log> | grep -oE '[0-9]+' | tail -1

and fail the gate when it is empty or 0. Under CARGO_TERM_COLOR=always
cargo-semver-checks colours its status word (`\\x1b[1m\\x1b[32m     Checked\\x1b[0m`),
not the count, so this pattern still matches (measured 2026-10-09: 223). A
pattern anchored on `Checked` followed by the timing would not. This test reads
the pattern out of both files and runs it over the captured coloured log, so a
change that makes it stop matching under colour fails here.
"""

from __future__ import annotations

import os
import re
import unittest

# Marks this file as a test of a cargo-output parser for
# scripts/cargo_output_parser_check.py (it must carry a coloured sample).
CARGO_OUTPUT_PARSER_TEST = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = (".github/workflows/security-audit.yml", "scripts/preflight.sh")
GREP = re.compile(r"grep -oE '([^']*checks[^']*)'")

# cargo semver-checks 0.50.0 check-release, CARGO_TERM_COLOR=always CI=true, piped (2026-10-09).
COLOURED = (
    "\x1b[1m\x1b[32m    Building\x1b[0m alice-sdf v5.0.0 (current)\n"
    "\x1b[1m\x1b[32m       Built\x1b[0m [  52.585s] (current)\n"
    "\x1b[1m\x1b[32m     Parsing\x1b[0m alice-sdf v5.0.0 (current)\n"
    "\x1b[1m\x1b[32m      Parsed\x1b[0m [   0.170s] (current)\n"
    "\x1b[1m\x1b[32m    Building\x1b[0m alice-sdf v5.0.0 (baseline)\n"
    "\x1b[1m\x1b[32m       Built\x1b[0m [  57.289s] (baseline)\n"
    "\x1b[1m\x1b[32m     Parsing\x1b[0m alice-sdf v5.0.0 (baseline)\n"
    "\x1b[1m\x1b[32m      Parsed\x1b[0m [   0.074s] (baseline)\n"
    "\x1b[1m\x1b[32m    Checking\x1b[0m alice-sdf v5.0.0 -> v5.0.0 (no change; assume patch)\n"
    "\x1b[1m\x1b[32m     Checked\x1b[0m [   0.239s] 223 checks: 223 pass, 31 skip\n"
    "\x1b[1m\x1b[32m     Summary\x1b[0m no semver update required\n"
    "\x1b[1m\x1b[32m    Finished\x1b[0m [ 116.053s] alice-sdf\n"
)


def patterns() -> dict[str, list[str]]:
    found = {}
    for rel in SOURCES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            found[rel] = GREP.findall(f.read())
    return found


def ran(pattern: str, log: str) -> str:
    """`grep -oE PATTERN | grep -oE '[0-9]+' | tail -1` (the ERE is also a valid Python regex)."""
    hits = re.findall(pattern, log)
    digits = [d for h in hits for d in re.findall(r"[0-9]+", h)]
    return digits[-1] if digits else ""


class SemverChecksCount(unittest.TestCase):
    def test_pattern_present_and_identical(self):
        found = patterns()
        for rel, pats in found.items():
            self.assertTrue(pats, f"no `grep -oE '...checks...'` in {rel}")
        self.assertEqual(len({p for pats in found.values() for p in pats}), 1, found)

    def test_coloured_log_matches(self):
        for pat in {p for pats in patterns().values() for p in pats}:
            self.assertEqual(ran(pat, COLOURED), "223", pat)

    def test_no_checked_line_is_empty(self):
        log = "".join(line + "\n" for line in COLOURED.splitlines() if "Checked" not in line)
        for pat in {p for pats in patterns().values() for p in pats}:
            self.assertEqual(ran(pat, log), "", pat)

    def test_checked_anchor_would_miss(self):
        # the form that broke elsewhere: `Checked` directly followed by the timing
        self.assertIsNone(re.search(r"Checked\s+\[[^\]]*\]\s+[0-9]+ checks", COLOURED))


if __name__ == "__main__":
    unittest.main()
