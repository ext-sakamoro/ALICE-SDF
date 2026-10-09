#!/usr/bin/env python3
"""Read a `cargo test` log and check how many tests the libtest summary says passed.

The doctest step for `GpuEvalFuture` (ci.yml and scripts/preflight.sh) filters
by name, so a renamed doctest or a filter that stops matching makes it run 0
tests and still exit 0. This reads back the last `test result: ok. N passed`
line and fails when N is below the expected minimum.

Escape codes are removed before matching. With CARGO_TERM_COLOR=always (set at
the top of ci.yml) cargo colours its own status lines; libtest itself writes
plain text into a pipe, but with `--color always` it writes
`test result: \\x1b[32mok\\x1b(B\\x1b[m. 2 passed` (measured 2026-10-09). That
`\\x1b(B` is a charset selection, not an SGR sequence, so a pattern that only
removes `\\x1b[...m` still leaves the summary unmatched. The stripping is
scripts/ansi.py, the one place where tool output is stripped.

Usage: libtest_count.py <log> --min N [--what TEXT]
Prints N and exits 0, or prints a `::error::` line and exits 1.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import ansi  # scripts/ansi.py: CSI, character-set selectors and OSC

# Marks this script as a parser of cargo / libtest output for
# scripts/cargo_output_parser_check.py (its tests must carry a coloured sample).
CARGO_OUTPUT_PARSER = True

RESULT_OK = re.compile(r"test result: ok\. ([0-9]+) passed")


def strip_ansi(text: str) -> str:
    return ansi.strip(text)


def passed_count(log: str) -> int | None:
    """The count of the last `test result: ok. N passed` line, or None when there is none."""
    found = RESULT_OK.findall(strip_ansi(log))
    return int(found[-1]) if found else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--min", type=int, required=True, help="the fewest passed tests accepted")
    ap.add_argument("--what", default="tests", help="what the tests are, for the error message")
    args = ap.parse_args(argv)
    n = passed_count(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    if n is None or n < args.min:
        print(
            f"::error::expected >= {args.min} {args.what}, ran {n if n is not None else 0} "
            f"(no `test result: ok. N passed` line with N >= {args.min} in {args.log})"
        )
        return 1
    print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
