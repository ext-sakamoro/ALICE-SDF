#!/usr/bin/env python3
"""Tests for scripts/default_dependency_licenses.py on small resolved graphs."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import default_dependency_licenses as d  # noqa: E402


def meta(packages, edges, root="root"):
    """packages: {id: licence}; edges: {id: [(dep, kind)]} with kind None / 'build' / 'dev'"""
    return {
        "packages": [{"id": i, "name": i, "version": "1.0.0", "license": lic} for i, lic in packages.items()],
        "resolve": {
            "root": root,
            "nodes": [{"id": i, "deps": [{"pkg": t, "dep_kinds": [{"kind": k}]} for t, k in edges.get(i, [])]}
                      for i in packages],
        },
    }


class Licences(unittest.TestCase):
    def test_expressions(self):
        for expr, ok in [("MIT", True), ("MIT OR Apache-2.0", True), ("MIT/Apache-2.0", True),
                         ("Apache-2.0 WITH LLVM-exception", True), ("(MIT OR Apache-2.0) AND Unicode-3.0", True),
                         ("AGPL-3.0-or-later", False), ("AGPL-3.0-or-later OR LicenseRef-Commercial", False),
                         ("MIT AND GPL-3.0", False), ("MPL-2.0", False), (None, False), ("", False)]:
            with self.subTest(expr=expr):
                self.assertEqual(d.permissive(expr), ok)


class Graph(unittest.TestCase):
    def test_a_permissive_tree_passes(self):
        m = meta({"root": "Apache-2.0", "a": "MIT", "b": "Zlib"},
                 {"root": [("a", None)], "a": [("b", "build")]})
        self.assertEqual(d.check(m), ([], 2))

    def test_an_agpl_crate_reached_fails(self):
        m = meta({"root": "Apache-2.0", "a": "MIT", "c": "AGPL-3.0-or-later"},
                 {"root": [("a", None)], "a": [("c", None)]})
        problems, n = d.check(m)
        self.assertEqual(n, 2)
        self.assertEqual(len(problems), 1)
        self.assertIn("c 1.0.0", problems[0])

    def test_dev_dependencies_and_unreached_packages_do_not_count(self):
        m = meta({"root": "Apache-2.0", "a": "MIT", "t": "AGPL-3.0-or-later", "x": "GPL-3.0"},
                 {"root": [("a", None), ("t", "dev")]})
        self.assertEqual(d.check(m), ([], 1))

    def test_nothing_reached_is_exit_2(self):
        m = meta({"root": "Apache-2.0"}, {})
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(m, f)
        try:
            self.assertEqual(d.main(["--metadata", f.name]), 2)
        finally:
            os.unlink(f.name)

    def test_this_repository_passes(self):
        self.assertEqual(d.main([]), 0)


if __name__ == "__main__":
    unittest.main()
