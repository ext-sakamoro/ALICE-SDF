#!/usr/bin/env python3
"""Tests for scripts/alembic_library_check.py.

The Config.h lines are the ones Alembic 1.8.12 installs (measured 2026-10-21: the Homebrew build,
which defines ALEMBIC_WITH_HDF5, and a build with -DUSE_HDF5=OFF, which writes it as a comment).
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alembic_library_check as alc  # noqa: E402

OGAWA = "#define ALEMBIC_LIBRARY_VERSION 1 * 10000 + 8 * 100 + 12\n\n/* #undef ALEMBIC_WITH_HDF5 */\n"
WITH_HDF5 = "#define ALEMBIC_LIBRARY_VERSION 1 * 10000 + 8 * 100 + 12\n\n#define ALEMBIC_WITH_HDF5\n"
PIN = "# comment\nversion 1.8.12\nurl https://example.invalid/a.tgz\nsha256 00\n"
REPO_PIN = Path(__file__).resolve().parent.parent / "tests/oracle/alembic-library.txt"


class Check(unittest.TestCase):
    def run_check(self, config: str, pin: str = PIN, ogawa_only: bool = True, hdf5_dir: bool = False):
        with tempfile.TemporaryDirectory() as d:
            inc = Path(d, "include")
            (inc / "Alembic/Util").mkdir(parents=True)
            (inc / "Alembic/Util/Config.h").write_text(config, encoding="utf-8")
            if hdf5_dir:
                (inc / "Alembic/AbcCoreHDF5").mkdir()
            Path(d, "pin.txt").write_text(pin, encoding="utf-8")
            args = [str(inc), "--pin", str(Path(d, "pin.txt"))] + (["--ogawa-only"] if ogawa_only else [])
            out = io.StringIO()
            with redirect_stdout(out):
                rc = alc.main(args)
            return rc, out.getvalue()

    def test_the_pinned_ogawa_build_passes_and_prints_both_versions(self):
        rc, out = self.run_check(OGAWA)
        self.assertEqual(rc, 0, out)
        self.assertIn("built against 1.8.12, pinned 1.8.12", out)

    def test_a_patch_release_passes_and_another_minor_fails(self):
        self.assertEqual(self.run_check(OGAWA.replace("+ 12", "+ 13"))[0], 0)
        rc, out = self.run_check(OGAWA.replace("8 * 100", "9 * 100"))
        self.assertEqual(rc, 1)
        self.assertIn("1.9.12 is not the pinned 1.8.12", out)

    def test_an_hdf5_build_fails_only_when_ogawa_only_is_asked(self):
        self.assertEqual(self.run_check(WITH_HDF5)[0], 1)
        self.assertEqual(self.run_check(OGAWA, hdf5_dir=True)[0], 1)
        self.assertEqual(self.run_check(WITH_HDF5, ogawa_only=False)[0], 0)

    def test_a_missing_version_or_pin_field_fails(self):
        self.assertEqual(self.run_check("/* nothing */\n")[0], 1)
        self.assertEqual(self.run_check(OGAWA, pin="version 1.8.12\n")[0], 1)

    def test_the_repository_pin_names_a_tag_tarball_and_a_sha256(self):
        pin = alc.pinned(REPO_PIN)
        self.assertRegex(pin["version"], r"^\d+\.\d+\.\d+$")
        self.assertEqual(pin["url"], f"https://github.com/alembic/alembic/archive/refs/tags/{pin['version']}.tar.gz")
        self.assertRegex(pin["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
