#!/usr/bin/env python3
"""ALICE-Law G13: every iteration of a HashMap / HashSet in the library has a verdict.

The iteration order of `std::collections::HashMap` / `HashSet` is unspecified
(it depends on the random hasher seed of the process and on the std version),
so an output built in that order differs between runs. This scans `src/`
(outside the unit-test modules) for the iterations of every name declared
with a hash collection type in the same file: the iteration methods
(`iter`, `iter_mut`, `keys`, `values`, `values_mut`, `into_iter`, `into_keys`,
`into_values`, `drain`, `retain`) and `for .. in` over the name. Each finding
must be listed in scripts/hash_order_audit.toml with a verdict saying why the
order cannot reach an output:

  not-observed   the loop sets a flag, counts, takes an integer sum / max, or
                 fills a set used only for membership
  sorted-after   the items are collected and sorted before use
  lookup         the line iterates a value found by `get`, which is a Vec in
                 insertion order, not the map

An iteration whose order would reach an output is not given a verdict: it is
fixed (an ordered collection, or a sort). The list is compared both ways: a
finding without an entry fails, and so does an entry the scan no longer finds.
Reading no source file, or finding nothing, fails.

usage: python3 scripts/hash_order_audit.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

VERDICTS = {"not-observed", "sorted-after", "lookup"}
DECL = re.compile(
    r"\b([a-z_][a-z0-9_]*)\s*:\s*(?:&(?:mut\s+)?)?(?:std::collections::)?Hash(?:Map|Set)\b")
NEW = re.compile(r"let\s+(?:mut\s+)?([a-z_][a-z0-9_]*)\s*(?::[^=]*)?=\s*(?:std::collections::)?Hash(?:Map|Set)::")
METHODS = r"\.(?:iter|iter_mut|keys|values|values_mut|into_iter|into_keys|into_values|drain|retain)\s*\("


def code_of(text: str) -> str:
    """The file without its unit-test module (by convention at the end)"""
    cut = text.find("#[cfg(test)]\nmod tests")
    return text if cut < 0 else text[:cut]


def scan_text(path: str, text: str) -> list[tuple[str, str]]:
    """(file, code line) of every iteration of a hash-typed name in `text`"""
    code = code_of(text)
    names = set(DECL.findall(code)) | set(NEW.findall(code))
    # a method chain written over several lines is one statement: join the lines
    # that start with `.` to the line before (comments dropped first)
    code = "\n".join(l.split("//")[0].rstrip() for l in code.split("\n"))
    code = re.sub(r"\n\s*\.", ".", code)
    found = []
    for line in code.split("\n"):
        s = line.split("//")[0].strip()
        if not s:
            continue
        for name in sorted(names):
            if (re.search(rf"\b{name}\s*{METHODS}", s)
                    or re.search(rf"\bfor\s+.*\bin\s+&?(?:mut\s+)?(?:\w+\.)?{name}\b", s)):
                found.append((path, s))
                break
    return found


def scan(root: Path) -> tuple[list[tuple[str, str]], int]:
    files = sorted(p for p in (root / "src").rglob("*.rs"))
    out = []
    for p in files:
        out.extend(scan_text(p.relative_to(root).as_posix(), p.read_text(encoding="utf-8")))
    return out, len(files)


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as fh:
        return tomllib.load(fh).get("site", [])


def check(found: list[tuple[str, str]], listed: list[dict]) -> list[str]:
    """Problems; an entry stands for `count` identical lines of its file (default 1)"""
    problems = []
    have: Counter = Counter()
    for e in listed:
        k = (e.get("file", ""), e.get("code", ""))
        if k in have:
            problems.append(f"{k[0]}: `{k[1]}`: listed twice (use count)")
        have[k] += int(e.get("count", 1))
        if e.get("verdict") not in VERDICTS:
            problems.append(f"{k[0]}: `{k[1]}`: verdict {e.get('verdict')!r} is not one of {sorted(VERDICTS)}")
        if len(str(e.get("why", ""))) < 12:
            problems.append(f"{k[0]}: `{k[1]}`: the why is missing or too short")
    want = Counter(found)
    for k in sorted(set(want) | set(have)):
        if want[k] > have[k]:
            problems.append(f"{k[0]}: `{k[1]}`: iterates a HashMap / HashSet without a verdict "
                            f"({want[k]} found, {have[k]} listed)")
        elif want[k] < have[k]:
            problems.append(f"{k[0]}: `{k[1]}`: listed {have[k]}, the scan finds {want[k]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args(argv)
    root = Path(a.root)
    found, nfiles = scan(root)
    if nfiles == 0 or not found:
        print(f"hash-order-audit: read {nfiles} files, found {len(found)} iterations (compared nothing)",
              file=sys.stderr)
        return 2
    problems = check(found, load(root / "scripts" / "hash_order_audit.toml"))
    for p in problems:
        print(p)
    print(f"hash-order-audit: {nfiles} files, {len(found)} iterations of a hash collection, "
          f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
