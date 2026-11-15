#!/usr/bin/env python3
"""Re-stamp the semantics identifier in the headers of the binary `.asdf` fixtures

`tests/fixtures/bincode1/` pins the body bytes the bincode 2 legacy encoder wrote. Each `.asdf`
file's 48-byte header carries the semantics identifier of the build that wrote it (bytes 16..48),
which a reader of another identifier refuses. When `SDF_SEMANTICS_ID` moves, this script writes the
new identifier into every fixture header and touches nothing else:

  scripts/restamp_asdf_fixtures.py --old <64 hex> --new <64 hex>
      every header must hold exactly `--old` (a header with anything else is refused and nothing is
      written); `chain500.asdf` and each file in `corpus.bin` (u32 little-endian length, then the
      file) are re-stamped; SHA256SUMS is rewritten; 0 files re-stamped is a failure

  scripts/restamp_asdf_fixtures.py --verify-against <git revision> --new <64 hex>
      reads the fixtures at that revision and the working tree and checks, file by file, that the
      bytes after the header are equal, that the header up to byte 16 is equal and that every
      header now holds `--new`; it prints the number of files compared (0 is a failure)

Author: Moroya Sakamoto
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REL = "tests/fixtures/bincode1"
HEADER = 48
ID = slice(16, 48)


def split(corpus: bytes) -> list[bytes]:
    out, i = [], 0
    while i < len(corpus):
        (n,) = struct.unpack("<I", corpus[i : i + 4])
        out.append(corpus[i + 4 : i + 4 + n])
        i += 4 + n
    if i != len(corpus):
        raise SystemExit("restamp: corpus.bin does not split into length-prefixed files")
    return out


def files(read) -> list[tuple[str, bytes]]:
    """(name, file) for chain500.asdf and each corpus.bin file"""
    names = read("corpus.names").decode().split()
    corpus = split(read("corpus.bin"))
    if len(names) != len(corpus):
        raise SystemExit(f"restamp: {len(names)} names for {len(corpus)} corpus files")
    return [("chain500.asdf", read("chain500.asdf"))] + list(zip(names, corpus))


def hexid(s: str) -> bytes:
    b = bytes.fromhex(s)
    if len(b) != 32:
        raise SystemExit("restamp: an identifier is 64 hex digits")
    return b


def restamp(old: bytes, new: bytes) -> int:
    d = ROOT / REL
    def stamp(name: str, f: bytes) -> bytes:
        if f[:4] != b"ASDF" or len(f) < HEADER:
            raise SystemExit(f"restamp: {name} is not an .asdf file")
        if f[ID] != old:
            raise SystemExit(f"restamp: {name} holds {f[ID].hex()}, not the expected old identifier")
        return f[:16] + new + f[HEADER:]
    every = files(lambda n: (d / n).read_bytes())
    stamped = [(n, stamp(n, f)) for n, f in every]  # all checked before anything is written
    (d / "chain500.asdf").write_bytes(stamped[0][1])
    (d / "corpus.bin").write_bytes(b"".join(struct.pack("<I", len(f)) + f for _, f in stamped[1:]))
    sums = [f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}" for n in ("corpus.names", "corpus.bin", "chain500.asdf")]
    (d / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return len(stamped)


def verify(rev: str, new: bytes) -> int:
    def at_rev(n: str) -> bytes:
        return subprocess.run(["git", "show", f"{rev}:{REL}/{n}"], cwd=ROOT, capture_output=True, check=True).stdout
    before = files(at_rev)
    after = files(lambda n: (ROOT / REL / n).read_bytes())
    if [n for n, _ in before] != [n for n, _ in after]:
        raise SystemExit("restamp: the fixture list changed")
    for (n, a), (_, b) in zip(before, after):
        if a[HEADER:] != b[HEADER:]:
            raise SystemExit(f"restamp: {n}: the body changed")
        if a[:16] != b[:16]:
            raise SystemExit(f"restamp: {n}: the header before the identifier changed")
        if b[ID] != new:
            raise SystemExit(f"restamp: {n}: the header holds {b[ID].hex()}, not the new identifier")
    return len(after)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--old")
    p.add_argument("--new", required=True)
    p.add_argument("--verify-against")
    a = p.parse_args(argv[1:])
    if a.verify_against:
        n = verify(a.verify_against, hexid(a.new))
        print(f"restamp: {n} files: bodies and header prefixes equal to {a.verify_against}, identifier {a.new[:8]}")
    else:
        if not a.old:
            p.error("--old is required to re-stamp")
        n = restamp(hexid(a.old), hexid(a.new))
        print(f"restamp: {n} files re-stamped {a.old[:8]} -> {a.new[:8]}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
