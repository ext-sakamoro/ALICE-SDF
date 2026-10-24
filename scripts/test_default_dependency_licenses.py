#!/usr/bin/env python3
"""Tests for scripts/default_dependency_licenses.py on small resolved graphs."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import default_dependency_licenses as d  # noqa: E402

# a deny policy shaped like deny.toml: permissive allow list, one AGPL exception, one clarify
DENY = d.DenyPolicy(allow={"MIT", "Apache-2.0", "Zlib", "ISC", "OpenSSL", "MPL-2.0"},
                    exceptions={"alice-physics": {"AGPL-3.0-or-later"}},
                    clarify={"ring": "MIT AND ISC AND OpenSSL"})


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
                         ("MIT AND GPL-3.0", False), ("MPL-2.0", False), (None, False), ("", False),
                         ("LicenseRef-Proprietary", False)]:
            with self.subTest(expr=expr):
                self.assertEqual(d.permissive(expr), ok)

    def test_the_deny_policy_is_read_from_the_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write('[licenses]\nallow = ["MIT"]\n\n[[licenses.exceptions]]\nname = "x"\nallow = ["AGPL-3.0-or-later"]\n'
                    '\n[[licenses.clarify]]\nname = "ring"\nversion = "*"\nexpression = "ISC"\n')
        try:
            p = d.read_deny(__import__("pathlib").Path(f.name))
        finally:
            os.unlink(f.name)
        self.assertEqual((p.allow, p.exceptions, p.clarify), ({"MIT"}, {"x": {"AGPL-3.0-or-later"}}, {"ring": "ISC"}))


class Apache(unittest.TestCase):
    """a crate declared Apache-2.0 (or MIT) takes permissive dependencies only"""

    def test_a_permissive_tree_passes(self):
        for own in ("Apache-2.0", "MIT", "MIT OR Apache-2.0"):
            with self.subTest(own=own):
                m = meta({"root": own, "a": "MIT", "b": "Zlib"}, {"root": [("a", None)], "a": [("b", "build")]})
                problems, n, _ = d.check(m, DENY)
                self.assertEqual((problems, n), ([], 2))

    def test_a_proprietary_dependency_fails(self):
        m = meta({"root": "Apache-2.0", "a": "MIT", "p": "LicenseRef-Proprietary"},
                 {"root": [("a", None)], "a": [("p", None)]})
        problems, n, _ = d.check(m, DENY)
        self.assertEqual(n, 2)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("p 1.0.0", problems[0])

    def test_an_agpl_dependency_fails_even_with_a_deny_exception(self):
        # deny.toml lets alice-physics be AGPL, but an Apache crate does not take it
        m = meta({"root": "Apache-2.0", "alice-physics": "AGPL-3.0-or-later"}, {"root": [("alice-physics", None)]})
        problems, n, _ = d.check(m, DENY)
        self.assertEqual((n, len(problems)), (1, 1), problems)
        self.assertIn("alice-physics", problems[0])


class Proprietary(unittest.TestCase):
    """a crate declared LicenseRef-Proprietary takes what deny.toml allows, and LicenseRef-Proprietary"""

    def test_a_proprietary_dependency_passes(self):
        m = meta({"root": "LicenseRef-Proprietary", "a": "MIT", "p": "LicenseRef-Proprietary"},
                 {"root": [("a", None), ("p", None)]})
        self.assertEqual(d.check(m, DENY)[:2], ([], 2))

    def test_a_deny_exception_and_a_deny_allow_entry_pass(self):
        m = meta({"root": "LicenseRef-Proprietary", "alice-physics": "AGPL-3.0-or-later", "m": "MPL-2.0"},
                 {"root": [("alice-physics", None), ("m", None)]})
        self.assertEqual(d.check(m, DENY)[:2], ([], 2))

    def test_agpl_without_an_exception_fails(self):
        m = meta({"root": "LicenseRef-Proprietary", "other": "AGPL-3.0-or-later"}, {"root": [("other", None)]})
        problems, n, _ = d.check(m, DENY)
        self.assertEqual((n, len(problems)), (1, 1), problems)
        self.assertIn("other", problems[0])

    def test_a_licence_outside_the_deny_policy_fails(self):
        m = meta({"root": "LicenseRef-Proprietary", "g": "GPL-3.0"}, {"root": [("g", None)]})
        self.assertEqual(len(d.check(m, DENY)[0]), 1)

    def test_the_clarify_expression_replaces_the_declared_one(self):
        m = meta({"root": "LicenseRef-Proprietary", "ring": None}, {"root": [("ring", None)]})
        self.assertEqual(d.check(m, DENY)[0], [])
        m = meta({"root": "LicenseRef-Proprietary", "other": None}, {"root": [("other", None)]})
        self.assertEqual(len(d.check(m, DENY)[0]), 1, "no expression and no clarify must fail")


class Graph(unittest.TestCase):
    def test_dev_dependencies_and_unreached_packages_do_not_count(self):
        m = meta({"root": "Apache-2.0", "a": "MIT", "t": "AGPL-3.0-or-later", "x": "GPL-3.0"},
                 {"root": [("a", None), ("t", "dev")]})
        self.assertEqual(d.check(m, DENY)[:2], ([], 1))

    def test_a_declaration_without_a_policy_fails(self):
        m = meta({"root": "GPL-3.0", "a": "MIT"}, {"root": [("a", None)]})
        problems, n, _ = d.check(m, DENY)
        self.assertEqual((n, len(problems)), (1, 1), problems)

    def run_main(self, m) -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(m, f)
        try:
            return d.main(["--metadata", f.name])
        finally:
            os.unlink(f.name)

    def test_nothing_reached_is_exit_2(self):
        self.assertEqual(self.run_main(meta({"root": "LicenseRef-Proprietary"}, {})), 2)

    def test_a_root_without_a_licence_is_exit_2(self):
        self.assertEqual(self.run_main(meta({"root": None, "a": "MIT"}, {"root": [("a", None)]})), 2)

    def test_a_failing_graph_is_exit_1(self):
        m = meta({"root": "Apache-2.0", "p": "LicenseRef-Proprietary"}, {"root": [("p", None)]})
        self.assertEqual(self.run_main(m), 1)

    def test_this_repository_passes(self):
        self.assertEqual(d.main([]), 0)


if __name__ == "__main__":
    unittest.main()
