#!/usr/bin/env python3
"""Every parser of cargo / libtest / cargo-semver-checks output must be tested on coloured output.

ci.yml and the other workflows set CARGO_TERM_COLOR=always, so cargo and the
cargo-* tools wrap their status words (and with `--color always`, libtest its
`ok`) in escape codes. A regex written against plain local output then matches
0 lines in CI only, which reads as "0 tests" or "compared nothing" or stops a
step under pipefail without a reason. This check makes the coloured case part
of the test of every such parser:

  1. a script in scripts/ whose regex mentions cargo / libtest output wording
     must declare `CARGO_OUTPUT_PARSER = True`;
  2. every declared parser must be imported by at least one scripts/test_*.py,
     and every such test file must contain an escape sample (`\\x1b[`);
  3. a test file of an inline (workflow / shell) parser declares
     `CARGO_OUTPUT_PARSER_TEST = True` and must contain an escape sample too;
  4. finding no parser and no such test at all is a failure, not a pass.

Usage: cargo_output_parser_check.py [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

PARSER_MARK = re.compile(r"^CARGO_OUTPUT_PARSER\s*=\s*True\b", re.M)
TEST_MARK = re.compile(r"^CARGO_OUTPUT_PARSER_TEST\s*=\s*True\b", re.M)
# a regex call whose literal mentions the wording cargo, libtest and the cargo-* tools print
REGEX_CALL = re.compile(r"\bre\.(?:compile|search|match|fullmatch|findall|finditer)\(")
TOOL_WORDING = re.compile(
    r"test result|Doc-tests|Compiling|Checked|Running|Finished|checks?:|MISSED|CAUGHT|TIMEOUT"
)


def has_escape_sample(src: str) -> bool:
    return "\\x1b[" in src or "\x1b[" in src


def imports(src: str, module: str) -> bool:
    return re.search(rf"^\s*(?:import\s+{module}\b|from\s+{module}\s+import\b)", src, re.M) is not None


def check(scripts_dir: str) -> tuple[list[str], dict[str, int]]:
    files = sorted(f for f in os.listdir(scripts_dir) if f.endswith(".py"))
    src = {}
    for f in files:
        with open(os.path.join(scripts_dir, f), encoding="utf-8") as fh:
            src[f] = fh.read()
    tests = [f for f in files if f.startswith("test_")]
    others = [f for f in files if not f.startswith("test_")]

    errors: list[str] = []
    parsers = [f for f in others if PARSER_MARK.search(src[f])]

    for f in others:
        if f in parsers or f == os.path.basename(__file__):
            continue
        for n, line in enumerate(src[f].splitlines(), 1):
            if REGEX_CALL.search(line) and TOOL_WORDING.search(line):
                errors.append(
                    f"unmarked: {f}:{n}: a regex on cargo / tool output wording, but no "
                    f"`CARGO_OUTPUT_PARSER = True` (its tests must then carry a coloured sample)"
                )
                break

    checked_tests: set[str] = set()
    for p in parsers:
        mod = p[:-3]
        users = [t for t in tests if imports(src[t], mod)]
        if not users:
            errors.append(f"untested: {p}: no scripts/test_*.py imports {mod}")
        for t in users:
            checked_tests.add(t)
            if not has_escape_sample(src[t]):
                errors.append(f"no_colour_sample: {t}: tests the cargo-output parser {p} without a `\\x1b[` sample")

    marked_tests = [t for t in tests if TEST_MARK.search(src[t])]
    for t in marked_tests:
        checked_tests.add(t)
        if not has_escape_sample(src[t]):
            errors.append(f"no_colour_sample: {t}: declares CARGO_OUTPUT_PARSER_TEST without a `\\x1b[` sample")

    if not parsers and not marked_tests:
        errors.append("empty_scan: no cargo-output parser and no cargo-output parser test found in scripts/")

    return errors, {"parsers": len(parsers), "tests": len(checked_tests)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv)
    errors, stats = check(os.path.join(args.root, "scripts"))
    for e in errors:
        print(e, file=sys.stderr)
    print(
        f"cargo-output-parser-check: {stats['parsers']} parser(s), {stats['tests']} test file(s) "
        f"checked, {len(errors)} violation(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
