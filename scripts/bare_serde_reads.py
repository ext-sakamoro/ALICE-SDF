#!/usr/bin/env python3
"""ALICE-Law G8: every bare deserialization in library code is reviewed.

The SDF tree types keep `Deserialize` (a type's field layout, which every serde user of the
types relies on), so a bare `serde_json::from_*` / bincode decode of a tree would read a
stored form without checking its semantics identifier. This check lists every bare
deserialization call in `src/` (test modules excluded): `serde_json::from_str`, `from_slice`,
`from_reader`, `from_value`, `serde_json::Deserializer`, bincode's `decode_from_slice` /
`decode_from_std_read` / `deserialize`, and the rmp / yaml / toml / ron readers. Each must be
listed in `scripts/bare-serde-reads.tsv`, keyed by file, enclosing function and call, with:

* `checked`: the call is inside a loader that checks the stored identifier before the value
  is returned (the reason names the check)
* `other-type`: it reads a value that is not an SDF tree or a value computed under the SDF
  arithmetic (the reason names the type)
* `pending`: a reader that still has to check (the reason names the owner)

The run fails on a call that is not listed, on a listed entry that no longer matches, and
when no source file was scanned.

usage: bare_serde_reads.py [--root DIR] [--table TSV]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CALL = re.compile(
    r"\b(serde_json::from_(?:str|slice|reader|value)|serde_json::Deserializer"
    r"|bincode::(?:serde::)?(?:decode_from_slice|decode_from_std_read|borrow_decode_from_slice)"
    r"|bincode::deserialize(?:_from)?|rmp_serde::from_(?:slice|read)|serde_yaml::from_(?:str|slice|reader)"
    r"|toml::from_str|ron::(?:de::)?from_(?:str|bytes))\b")
FN = re.compile(r"\bfn\s+([A-Za-z_]\w*)")
CFG_TEST = re.compile(r"^\s*#\[cfg\((?:test|all\(test\b[^)]*\))\)\]")
STATUSES = ("checked", "other-type", "pending")


def code_of(line: str) -> str:
    return line.split("//", 1)[0]


def test_lines(lines: list[str]) -> set[int]:
    """line indexes inside a #[cfg(test)] item (braces counted from the item's first `{`)"""
    skip: set[int] = set()
    i = 0
    while i < len(lines):
        if CFG_TEST.match(lines[i]):
            depth, opened, j = 0, False, i
            while j < len(lines):
                c = code_of(lines[j])
                skip.add(j)
                depth += c.count("{") - c.count("}")
                opened = opened or "{" in c
                if (opened and depth <= 0) or (not opened and c.rstrip().endswith(";")):
                    break
                j += 1
            i = j + 1
            continue
        i += 1
    return skip


def scan(root: Path) -> tuple[int, dict[tuple[str, str, str], int]]:
    files = sorted((root / "src").rglob("*.rs"))
    found: dict[tuple[str, str, str], int] = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        lines = p.read_text(encoding="utf-8").split("\n")
        skip = test_lines(lines)
        fn = "<module>"
        for i, line in enumerate(lines):
            code = code_of(line)
            m = FN.search(code)
            if m:
                fn = m.group(1)
            if i in skip:
                continue
            for c in CALL.finditer(code):
                key = (rel, fn, c.group(1))
                found[key] = found.get(key, 0) + 1
    return len(files), found


def load_table(path: Path) -> dict[tuple[str, str, str], tuple[str, str]]:
    table = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 5:
            raise SystemExit(f"{path}:{n}: expected 5 tab-separated columns, got {len(cols)}")
        file, fn, call, status, reason = cols
        if status not in STATUSES:
            raise SystemExit(f"{path}:{n}: status must be one of {STATUSES}, got {status!r}")
        if len(reason.strip()) < 12:
            raise SystemExit(f"{path}:{n}: the reason must say why (12 characters or more)")
        if (file, fn, call) in table:
            raise SystemExit(f"{path}:{n}: duplicate entry {(file, fn, call)}")
        table[(file, fn, call)] = (status, reason)
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--table", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    files, found = scan(root)
    if files == 0:
        print(f"bare_serde_reads: FAIL: no source file under {root / 'src'} (0 scanned)")
        return 1
    table = load_table(Path(args.table) if args.table else root / "scripts" / "bare-serde-reads.tsv")
    problems = [f"not reviewed: {'	'.join(k)}" for k in sorted(found) if k not in table]
    problems += [f"listed but not found (remove it): {'	'.join(k)}" for k in sorted(table) if k not in found]
    by = {s: sum(found[k] for k in found if k in table and table[k][0] == s) for s in STATUSES}
    print(f"bare_serde_reads: {files} files scanned; {sum(found.values())} calls ("
          + ", ".join(f"{by[s]} {s}" for s in STATUSES) + f", {len(problems)} problems)")
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
