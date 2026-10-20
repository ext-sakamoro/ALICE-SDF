#!/usr/bin/env python3
"""Check the Alembic library a build of the oracle programs uses against tests/oracle/alembic-library.txt.

The writer in src/io/alembic.rs was checked against Alembic 1.8 (its archive layout, hashes and the
library version it stores), so the oracle programs must be built against that release. This reads
ALEMBIC_LIBRARY_VERSION from the library's installed Alembic/Util/Config.h (`1 * 10000 + 8 * 100 + 12`),
prints it beside the pinned version and fails unless major.minor agree. With --ogawa-only it also
fails when the library was built with HDF5 (ALEMBIC_WITH_HDF5 defined, or the AbcCoreHDF5 headers
installed): the CI job builds the Ogawa back end alone.

Usage: alembic_library_check.py <include dir> [--pin tests/oracle/alembic-library.txt] [--ogawa-only]
Prints both versions and exits 0, or prints a `::error::` line and exits 1.
"""
import argparse
import re
import sys
from pathlib import Path


def pinned(path: Path) -> dict:
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition(" ")
            fields[key] = value.strip()
    for key in ("version", "url", "sha256"):
        if not fields.get(key):
            raise ValueError(f"{path}: no `{key}` line")
    return fields


def library_version(config_h: Path) -> tuple:
    text = config_h.read_text(encoding="utf-8")
    m = re.search(r"#define\s+ALEMBIC_LIBRARY_VERSION\s+(\d+)\s*\*\s*10000\s*\+\s*(\d+)\s*\*\s*100\s*\+\s*(\d+)", text)
    if not m:
        raise ValueError(f"{config_h}: no ALEMBIC_LIBRARY_VERSION of the form a * 10000 + b * 100 + c")
    return tuple(int(x) for x in m.groups())


def with_hdf5(include: Path) -> bool:
    text = (include / "Alembic/Util/Config.h").read_text(encoding="utf-8")
    return bool(re.search(r"^\s*#define\s+ALEMBIC_WITH_HDF5\b", text, re.M)) or (include / "Alembic/AbcCoreHDF5").exists()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("include")
    ap.add_argument("--pin", default="tests/oracle/alembic-library.txt")
    ap.add_argument("--ogawa-only", action="store_true")
    a = ap.parse_args(argv)
    include = Path(a.include)
    try:
        pin = pinned(Path(a.pin))
        have = library_version(include / "Alembic/Util/Config.h")
    except (OSError, ValueError) as e:
        print(f"::error::Alembic library check: {e}")
        return 1
    want = tuple(int(x) for x in pin["version"].split("."))
    print(f"Alembic library: built against {'.'.join(map(str, have))}, pinned {pin['version']} ({include})")
    if have[:2] != want[:2]:
        print(f"::error::Alembic library {'.'.join(map(str, have))} is not the pinned {pin['version']} (major.minor differ)")
        return 1
    if a.ogawa_only and with_hdf5(include):
        print("::error::Alembic library was built with HDF5; the oracle job builds the Ogawa back end alone")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
