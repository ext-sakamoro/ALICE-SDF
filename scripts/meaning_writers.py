#!/usr/bin/env python3
"""Every stored form is written with its meaning identifier, never the raw one (ALICE-Law G8).

A writer stores the identifier of a stored form through `Provenance::slot_for` (a binary slot),
`Provenance::tagged_for` / `Stamp::tagged_for` (a `key=<64 hex>` text) or `Provenance::named_for`
(a key and a value kept apart, a JSON field). Each of those writes the meaning identifier of the
form (`provenance::meaning_id`). The raw semantics identifier reaches a file only through the other
ways of naming it: the `semantics_id` field of a `Provenance` copied out (`&p.semantics_id`,
`p.semantics_id.chunks_exact`), its `hex()` or `tagged()` text, or the earlier key `KEY` written
next to a value.

This checker reads the library and example sources (`src/**/*.rs`, `examples/**/*.rs`: the examples
write stored files too), without `#[cfg(test)]` modules and without
the lines of `src/provenance.rs` that define those functions or read stored files) and:

- fails on any raw write it finds (a line it matches is named with the pattern);
- counts the meaning writer sites (calls of `slot_for(`, `tagged_for(`, `named_for(`) and compares
  the number with `scripts/meaning-writers.txt` (`expected-sites N`): a writer removed or renamed so
  that the pattern no longer finds it, or a new one, changes the number and has to be recorded;
- fails when it read no source file or found no meaning writer site, so it never passes vacuously.

    python3 scripts/meaning_writers.py           # check
    python3 scripts/meaning_writers.py ROOT      # check another tree (the oracle's fixtures)

Author: Moroya Sakamoto
"""

from __future__ import annotations

import pathlib
import re
import sys

COUNT_FILE = "scripts/meaning-writers.txt"

# a raw identifier leaving a Provenance: its bytes copied into an output, its hex text, its
# `alice:semantics_id=` text, or the earlier key written next to a value
RAW_WRITES = [
    ("raw bytes into an output", re.compile(
        r"(extend_from_slice|write_all|copy_from_slice)\(\s*&\s*[A-Za-z_][A-Za-z0-9_.()]*\.semantics_id\b")),
    ("raw bytes split into words", re.compile(r"\.semantics_id\s*\.\s*chunks(_exact)?\(")),
    ("raw bytes as a stored field", re.compile(r"semantics_id:\s*Some\(\s*[A-Za-z_][A-Za-z0-9_.()]*\.semantics_id\s*\)")),
    ("raw hex as a stored field", re.compile(r"semantics_id:\s*Some\(\s*[A-Za-z_][A-Za-z0-9_.()]*\.hex\(\)\s*\)")),
    ("raw tagged text", re.compile(r"\.tagged\(\)")),
    ("the earlier key written", re.compile(r"\{KEY\}\s*=|\{KEY\}\\\"|KEY\.to_string\(\)|provenance::KEY\b")),
]

# a semantics identifier constant (this crate's or another's) written to an output: the call may be
# split over lines by rustfmt, so it is matched over a line and the two after it
CONST_WRITE = re.compile(
    r"\b(extend_from_slice|write_all|copy_from_slice|extend)\(\s*&?\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*([A-Z][A-Z0-9_]*SEMANTICS_ID)\b"
)

# the constants another crate's code compares, written where that crate's own stored forms write
# them: (file, constant) -> reason. Each entry must still match, or the run fails (stale)
FOREIGN_IDS = {
    ("src/live_sdf.rs", "PHYSICS_SEMANTICS_ID"): "a foreign arithmetic id compared by the owner:"
    " alice-physics' identifier of the stepping the modifiers ran under, compared by Physics'"
    " constants, as in the header of its own world snapshot",
}

MEANING_SITE = re.compile(r"\.(slot_for|tagged_for|named_for)\(")

# lines of src/provenance.rs that name the raw identifier on purpose: `tagged` (the earlier text
# form, read back from earlier files), meaning_id hashing it, and the producer stamp of STL / FBX /
# 3MF, which stores another crate's own semantics identifier beside this crate's meaning
# identifier (it names that crate's arithmetic, not a stored form of this crate: the producer
# checks it against its own identifier, alice_sdf::provenance::check_producer)
PROVENANCE_DEFINITIONS = re.compile(
    r"pub fn tagged\(|format!\(\"\{KEY\}=\{\}\", self\.hex\(\)\)|LawIdHasher::new\(MEANING_KIND, &semantics\.semantics_id\)"
    r"|h\[48\.\.80\]\.copy_from_slice\(&producer\.semantics_id\)"
)


def library_lines(path: pathlib.Path) -> list[tuple[int, str]]:
    """The lines of a source file outside `#[cfg(test)]` modules and outside comments"""
    out: list[tuple[int, str]] = []
    depth_at_test: int | None = None
    depth = 0
    pending_test = False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if depth_at_test is None and stripped.startswith("#[cfg(test)]"):
            pending_test = True
        opens, closes = line.count("{"), line.count("}")
        if pending_test and opens:
            depth_at_test = depth
            pending_test = False
        in_test = depth_at_test is not None
        depth += opens - closes
        if in_test and depth <= depth_at_test:
            depth_at_test = None
            continue
        if in_test or stripped.startswith("//"):
            continue
        out.append((n, line))
    return out


def check(root: pathlib.Path) -> tuple[list[str], int, int]:
    """(problems, files read, meaning writer sites)"""
    problems: list[str] = []
    files = 0
    sites = 0
    paths = sorted((root / "src").rglob("*.rs")) + sorted((root / "examples").rglob("*.rs"))
    foreign_used: set[tuple[str, str]] = set()
    read: set[str] = set()
    for path in paths:
        files += 1
        rel = path.relative_to(root).as_posix()
        read.add(rel)
        lines = library_lines(path)
        for i, (n, line) in enumerate(lines):
            sites += len(MEANING_SITE.findall(line))
            window = " ".join(l for _, l in lines[i : i + 3])
            for m in CONST_WRITE.finditer(window):
                # report a call once: at the line where it starts
                if m.start() >= len(line):
                    continue
                key = (rel, m.group(2))
                if key in FOREIGN_IDS:
                    foreign_used.add(key)
                else:
                    problems.append(f"{rel}:{n}: a semantics identifier constant written raw: {line.strip()}")
            if rel == "src/provenance.rs" and PROVENANCE_DEFINITIONS.search(line):
                continue
            for what, pattern in RAW_WRITES:
                if pattern.search(line):
                    problems.append(f"{rel}:{n}: {what}: {line.strip()}")
    for key in sorted(k for k in set(FOREIGN_IDS) - foreign_used if k[0] in read):
        problems.append(f"{key[0]}: FOREIGN_IDS entry {key[1]} no longer written there (stale)")
    if files == 0:
        problems.append("no source file read under src/ (nothing compared)")
    if sites == 0:
        problems.append("no meaning writer site found (slot_for / tagged_for / named_for): nothing compared")
    count_path = root / COUNT_FILE
    if not count_path.is_file():
        problems.append(f"{COUNT_FILE} is missing")
    else:
        m = re.search(r"^expected-sites (\d+)$", count_path.read_text(encoding="utf-8"), re.M)
        if not m:
            problems.append(f"{COUNT_FILE} has no `expected-sites N` line")
        elif int(m.group(1)) != sites:
            problems.append(
                f"{sites} meaning writer sites found, {COUNT_FILE} records {m.group(1)}: a writer was "
                f"added, removed or no longer matched; record the new number only with a round trip "
                f"for it in tests/test_g8_meaning_id.rs")
    return problems, files, sites


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = pathlib.Path(args[0]) if args else pathlib.Path(__file__).resolve().parent.parent
    problems, files, sites = check(root)
    for p in problems:
        print(f"meaning-writers: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"meaning-writers: ok (files {files}, meaning writer sites {sites}, raw writes 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
