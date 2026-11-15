#!/usr/bin/env python3
"""Tests for scripts/bincode_reach_check.py (no cargo: the metadata is built here)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bincode_reach_check as brc  # noqa: E402


def meta(edges: dict[str, list[tuple[str, list]]], root: str | None = "alice-sdf") -> dict:
    """edges: package -> [(dependency, [dep kind, ...])]; kind None is a normal edge"""
    names = set(edges) | {d for deps in edges.values() for d, _ in deps}
    return {
        "packages": [{"id": f"{n}-id", "name": n} for n in sorted(names)],
        "resolve": {
            "root": f"{root}-id" if root else None,
            "nodes": [{"id": f"{n}-id", "deps": [{"pkg": f"{d}-id", "dep_kinds": [{"kind": k} for k in kinds]}
                                                 for d, kinds in edges.get(n, [])]}
                      for n in sorted(names)],
        },
    }


WITHOUT = meta({"alice-sdf": [("glam", [None]), ("serde", [None])]})
WITH = meta({"alice-sdf": [("glam", [None]), ("bincode", [None])], "bincode": [("serde", [None])]})


class Reached(unittest.TestCase):
    def test_normal_edge_is_followed(self):
        self.assertIn("bincode", brc.reached(WITH))

    def test_transitive_edge_is_followed(self):
        m = meta({"alice-sdf": [("lib", [None])], "lib": [("bincode", [None])]})
        self.assertIn("bincode", brc.reached(m))

    def test_build_edge_is_followed(self):
        self.assertIn("bincode", brc.reached(meta({"alice-sdf": [("bincode", ["build"])]})))

    def test_dev_only_edge_is_not_followed(self):
        self.assertNotIn("bincode", brc.reached(meta({"alice-sdf": [("glam", [None]), ("bincode", ["dev"])]})))

    def test_dev_and_normal_edge_is_followed(self):
        self.assertIn("bincode", brc.reached(meta({"alice-sdf": [("bincode", ["dev", None])]})))

    def test_no_root_reaches_nothing(self):
        self.assertEqual(brc.reached(meta({"alice-sdf": [("bincode", [None])]}, root=None)), set())


class Check(unittest.TestCase):
    good = {"lol": WITHOUT, "none": WITHOUT, "default": WITH}

    def test_expected_state_passes(self):
        self.assertEqual(brc.check(self.good, set(brc.USERS), 10), [])

    def test_lol_set_reaching_bincode_fails(self):
        p = brc.check({**self.good, "lol": WITH}, set(brc.USERS), 10)
        self.assertEqual(len(p), 1)
        self.assertIn("lol", p[0])

    def test_default_not_reaching_bincode_fails(self):
        # the positive control: a check that never sees bincode would pass the first two vacuously
        p = brc.check({**self.good, "default": WITHOUT}, set(brc.USERS), 10)
        self.assertEqual(len(p), 1)
        self.assertIn("expected reached", p[0])

    def test_empty_walk_fails(self):
        p = brc.check({**self.good, "none": meta({"alice-sdf": []}, root=None)}, set(brc.USERS), 10)
        self.assertTrue(any("compared nothing" in x for x in p))

    def test_a_third_user_fails(self):
        p = brc.check(self.good, set(brc.USERS) | {"src/other.rs"}, 10)
        self.assertTrue(any("src/other.rs" in x for x in p))

    def test_a_lost_user_fails(self):
        p = brc.check(self.good, {"src/io/asdf.rs"}, 10)
        self.assertTrue(any("expected exactly" in x for x in p))

    def test_no_files_read_fails(self):
        p = brc.check(self.good, set(), 0)
        self.assertTrue(any("compared nothing" in x for x in p))


class Users(unittest.TestCase):
    def test_code_counts_comment_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "src")
            (src / "io").mkdir(parents=True)
            (src / "io" / "asdf.rs").write_text("let x = bincode::serde::decode(1);\n", encoding="utf-8")
            (src / "doc.rs").write_text("// bincode::config is described here\n", encoding="utf-8")
            (src / "lib.rs").write_text("fn f() {}\n", encoding="utf-8")
            found, n = brc.users(Path(d))
        self.assertEqual(found, {"src/io/asdf.rs"})
        self.assertEqual(n, 3)


if __name__ == "__main__":
    unittest.main()
