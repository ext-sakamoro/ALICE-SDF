#!/usr/bin/env python3
"""Tests for scripts/ansi.py.

The samples are real output captured on 2026-10-09 with CARGO_TERM_COLOR=always
CI=true and the output piped to a file, as the CI steps do: cargo's status
lines, cargo-semver-checks 0.50.0 and libtest run with `--color always`.
Each control case removes one rule from the pattern and shows the sample then
keeps an escape, so every rule is needed.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ansi  # noqa: E402

# Marks this file as a test of a cargo-output parser helper for
# scripts/cargo_output_parser_check.py (it must carry a coloured sample).
CARGO_OUTPUT_PARSER_TEST = True

CARGO = "\x1b[1m\x1b[92m   Doc-tests\x1b[0m alice_sdf"
SEMVER = "\x1b[1m\x1b[32m     Checked\x1b[0m [   0.239s] 223 checks: 223 pass, 31 skip"
LIBTEST = "test result: \x1b[32mok\x1b(B\x1b[m. 2 passed; 0 failed; 0 ignored; 0 measured; 36 filtered out; finished in 0.06s"
CRITERION = "                        Performance has \x1b[38;5;1mregressed\x1b[0m."
# OSC 8 hyperlink, terminated by ESC \ and by BEL
OSC = "see \x1b]8;;https://docs.rs/\x1b\\docs.rs\x1b]8;;\x1b\\ and \x1b]0;title\x07done"

PLAIN = {
    CARGO: "   Doc-tests alice_sdf",
    SEMVER: "     Checked [   0.239s] 223 checks: 223 pass, 31 skip",
    LIBTEST: "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 36 filtered out; finished in 0.06s",
    CRITERION: "                        Performance has regressed.",
    OSC: "see docs.rs and done",
}

CSI = r"\[[0-?]*[ -/]*[@-~]"
CHARSET = r"|[()*+][0-9A-Za-z]"
OSC_RULE = r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"


def without(rule: str) -> re.Pattern[str]:
    parts = [CSI, CHARSET, OSC_RULE]
    parts.remove(rule)
    body = "".join(parts)
    return re.compile(r"\x1b(?:" + (body[1:] if body.startswith("|") else body) + ")")


class Strip(unittest.TestCase):
    def test_pattern_is_csi_charset_osc(self):
        self.assertEqual(ansi.ANSI_RE.pattern, r"\x1b(?:" + CSI + CHARSET + OSC_RULE + ")")

    def test_samples(self):
        for coloured, plain in PLAIN.items():
            self.assertEqual(ansi.strip(coloured), plain, repr(coloured))

    def test_plain_text_unchanged(self):
        for plain in PLAIN.values():
            self.assertEqual(ansi.strip(plain), plain)


class Controls(unittest.TestCase):
    def test_without_charset_rule_libtest_keeps_escape(self):
        left = without(CHARSET).sub("", LIBTEST)
        self.assertIn("\x1b(B", left)
        self.assertIsNone(re.search(r"test result: ok\. ([0-9]+) passed", left))

    def test_without_csi_rule_cargo_keeps_escape(self):
        self.assertIn("\x1b[", without(CSI).sub("", CARGO))

    def test_without_osc_rule_hyperlink_keeps_escape(self):
        self.assertIn("\x1b]", without(OSC_RULE).sub("", OSC))


if __name__ == "__main__":
    unittest.main()
