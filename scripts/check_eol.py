#!/usr/bin/env python3
"""Text files are LF in the index and in the working tree (shared by the ALICE crates).

Shared files are compared byte for byte with ALICE-Law's table (G11 (c)), and a checkout that turns a
text file into CRLF (Windows) would differ from it. `.gitattributes` sets `* text=auto eol=lf`; this
check reads `git ls-files --eol` and fails on a text file whose index or working-tree form has CRLF
or mixed line endings. A file marked `-text` or `binary` is left as it is.

  python3 scripts/check_eol.py [--root DIR]

Prints `eol: compared N text files, M finding(s)`. Exit 0 when nothing is CRLF, 1 on a finding, 2 when
no text file was compared or git could not be read.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TEXT_EOLS = {"lf", "crlf", "mixed"}
BAD_EOLS = {"crlf", "mixed"}


def entries(root: Path) -> list[tuple[str, str, str, str]]:
    """(index eol, working-tree eol, attributes, path) of every tracked file"""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "--eol", "-z"], capture_output=True, check=True).stdout
    rows = []
    for raw in out.decode("utf-8", "surrogateescape").split("\0"):
        if not raw:
            continue
        info, _, path = raw.partition("\t")
        fields = info.split()
        index = next((f[2:] for f in fields if f.startswith("i/")), "")
        tree = next((f[2:] for f in fields if f.startswith("w/")), "")
        attr = next((f[5:] for f in fields if f.startswith("attr/")), "")
        rows.append((index, tree, attr, path))
    return rows


def check(root: Path) -> tuple[list[str], int]:
    """(findings, text files compared)"""
    findings = []
    compared = 0
    for index, tree, attr, path in entries(root):
        if "-text" in attr.split() or "binary" in attr.split():
            continue
        if index not in TEXT_EOLS and tree not in TEXT_EOLS:
            continue  # git sees it as binary, or it is not in the working tree
        compared += 1
        if index in BAD_EOLS:
            findings.append(f"{path}: committed with {index} line endings (i/{index})")
        if tree in BAD_EOLS:
            findings.append(f"{path}: checked out with {tree} line endings (w/{tree})")
    return findings, compared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    a = ap.parse_args(argv)
    try:
        findings, compared = check(a.root)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"eol: git ls-files could not be read: {e}", file=sys.stderr)
        return 2
    for f in findings:
        print(f"eol: {f}", file=sys.stderr)
    print(f"eol: compared {compared} text files, {len(findings)} finding(s)")
    if compared == 0:
        print("eol: compared nothing", file=sys.stderr)
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
