#!/usr/bin/env python3
"""Tests for scripts/cargo_output_parser_check.py.

The first case runs the check on this repository. The others build a small
scripts/ directory, break one thing, and assert the check reports it.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cargo_output_parser_check as cc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESC = "\\x1b["  # the escape written as source text, as a test file would

PARSER = "import re\nCARGO_OUTPUT_" + "PARSER = True\nR = re.compile(r'test result: ok\\. ([0-9]+) passed')\n"
UNMARKED = "import re\nR = re.compile(r'test result: ok\\. ([0-9]+) passed')\n"
PLAIN_SCRIPT = "import re\nR = re.compile(r'^version = \"([^\"]+)\"')\n"
TEST_OK = "import p\nS = '" + ESC + "32mok'\n"
TEST_PLAIN = "import p\nS = 'test result: ok. 2 passed'\n"
INLINE_TEST_OK = "CARGO_OUTPUT_PARSER_" + "TEST = True\nS = '" + ESC + "1mChecked'\n"
INLINE_TEST_PLAIN = "CARGO_OUTPUT_PARSER_" + "TEST = True\nS = 'Checked 2 checks: '\n"


def scan(files: dict[str, str]) -> tuple[list[str], dict[str, int]]:
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(text)
        return cc.check(d)


class Repository(unittest.TestCase):
    def test_this_repository_passes_and_checks_something(self):
        errors, stats = cc.check(os.path.join(ROOT, "scripts"))
        self.assertEqual(errors, [])
        self.assertGreaterEqual(stats["parsers"], 1)
        self.assertGreaterEqual(stats["tests"], 2)


class Rules(unittest.TestCase):
    def test_marked_parser_with_coloured_test_passes(self):
        errors, stats = scan({"p.py": PARSER, "test_p.py": TEST_OK})
        self.assertEqual(errors, [])
        self.assertEqual(stats, {"parsers": 1, "tests": 1})

    def test_parser_test_without_escape_fails(self):
        errors, _ = scan({"p.py": PARSER, "test_p.py": TEST_PLAIN})
        self.assertTrue(any(e.startswith("no_colour_sample: test_p.py") for e in errors), errors)

    def test_parser_without_test_fails(self):
        errors, _ = scan({"p.py": PARSER, "test_other.py": TEST_OK.replace("import p", "import q")})
        self.assertTrue(any(e.startswith("untested: p.py") for e in errors), errors)

    def test_unmarked_parser_fails(self):
        errors, _ = scan({"p.py": PARSER, "test_p.py": TEST_OK, "u.py": UNMARKED})
        self.assertTrue(any(e.startswith("unmarked: u.py:2") for e in errors), errors)

    def test_unrelated_regex_is_not_a_parser(self):
        errors, _ = scan({"p.py": PARSER, "test_p.py": TEST_OK, "v.py": PLAIN_SCRIPT})
        self.assertEqual(errors, [])

    def test_inline_parser_test_needs_escape(self):
        self.assertEqual(scan({"test_i.py": INLINE_TEST_OK})[0], [])
        errors, _ = scan({"test_i.py": INLINE_TEST_PLAIN})
        self.assertTrue(any(e.startswith("no_colour_sample: test_i.py") for e in errors), errors)

    def test_empty_scan_fails(self):
        errors, _ = scan({"v.py": PLAIN_SCRIPT, "test_v.py": "import v\n"})
        self.assertTrue(any(e.startswith("empty_scan") for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
