#!/usr/bin/env python3
"""Two copies of the library's io tests run at once with one TMPDIR do not collide.

A test that writes a fixed name under the system temporary directory passes
alone and fails when another run on the same machine (a second `cargo test`,
another CI job on a shared runner) writes the same name. This script runs the
library's unit-test binary twice concurrently, with the same TMPDIR, for a
number of rounds, and fails if either copy fails in any round. Before the
tests used src/test_dir.rs it failed in 8 of 8 rounds.

It fails when it compared nothing: no test binary was found, or a copy ran no
test.

  python3 scripts/concurrent_tests.py [--rounds N] [--filter io::] [-- <cargo test args>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import ansi  # scripts/ansi.py: CSI, character-set selectors and OSC

# Marks this script as a parser of cargo / libtest output for
# scripts/cargo_output_parser_check.py (its tests must carry a coloured sample).
CARGO_OUTPUT_PARSER = True

ROOT = Path(__file__).resolve().parent.parent
RESULT = re.compile(r"test result: (\w+)\. (\d+) passed; (\d+) failed")


def lib_test_binary(cargo_args: list[str]) -> str:
    out = subprocess.run(["cargo", "test", "--lib", "--no-run", "--message-format=json", *cargo_args],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = m.get("target", {})
        if m.get("reason") == "compiler-artifact" and m.get("profile", {}).get("test") \
                and m.get("executable") and Path(target.get("src_path", "")) == ROOT / "src" / "lib.rs":
            return m["executable"]
    raise SystemExit("error: no library test binary was built (compared nothing)")


def parse(text: str) -> tuple[int, int]:
    """(passed, failed) of a libtest run, or (-1, -1) when it printed no result."""
    m = RESULT.search(ansi.strip(text))
    return (int(m.group(2)), int(m.group(3))) if m else (-1, -1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--filter", default="io::")
    ap.add_argument("cargo_args", nargs="*")
    a = ap.parse_args(argv)
    binary = lib_test_binary(a.cargo_args)
    bad = 0
    with tempfile.TemporaryDirectory(prefix="alice-sdf-concurrent-") as tmp:
        env = dict(os.environ, TMPDIR=tmp + "/", TMP=tmp, TEMP=tmp)
        for i in range(1, a.rounds + 1):
            procs = [subprocess.Popen([binary, a.filter, "-q"], cwd=ROOT, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                     for _ in range(2)]
            outs = [p.communicate()[0] for p in procs]
            results = [parse(o) for o in outs]
            for p, o, (passed, failed) in zip(procs, outs, results):
                if passed <= 0:
                    print(f"error: round {i}: a copy ran no test (compared nothing)\n{o[-1500:]}", file=sys.stderr)
                    return 1
                if p.returncode != 0 or failed:
                    bad += 1
                    print(f"round {i}: a copy failed ({failed} failed)\n{o[-1500:]}", file=sys.stderr)
            print(f"round {i}: " + " / ".join(f"{p} passed, {f} failed" for p, f in results))
    if bad:
        print(f"error: {bad} failing copies in {a.rounds} rounds (tests collide on a shared TMPDIR)", file=sys.stderr)
        return 1
    print(f"concurrent tests: {a.rounds} rounds x 2 copies, filter {a.filter!r}, no collision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
