"""Oracles of scripts/version_sync.py, on small temporary git trees.

run: python3 scripts/test_version_sync.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("version_sync", os.path.join(HERE, "version_sync.py"))
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)

CARGO = '[package]\nname = "alice-x"\nversion = "2.1.0"\n'
CONFIG = '[[follow]]\nfile = "web/package.json"\n'


def tree(files: dict[str, str]) -> str:
    root = tempfile.mkdtemp()
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def base(**extra: str) -> dict[str, str]:
    files = {
        "Cargo.toml": CARGO,
        "scripts/version-sync.toml": CONFIG,
        "web/package.json": json.dumps({"name": "w", "version": "2.1.0"}),
    }
    files.update(extra)
    return files


class VersionSync(unittest.TestCase):
    def run_check(self, files):
        return vs.check(tree(files))

    def test_everything_in_step_passes(self):
        errors, counts = self.run_check(base())
        self.assertEqual(errors, [])
        self.assertEqual(counts["metadata"], 1)

    def test_a_followed_file_behind_the_crate_fails(self):
        errors, _ = self.run_check(base(**{"web/package.json": json.dumps({"version": "0.3.0"})}))
        self.assertTrue(any("0.3.0 does not follow" in e for e in errors), errors)

    def test_an_unregistered_version_bearing_file_fails(self):
        errors, _ = self.run_check(base(**{"plugin/A.uplugin": json.dumps({"VersionName": "2.1.0"})}))
        self.assertTrue(any("plugin/A.uplugin" in e and "not in" in e for e in errors), errors)

    def test_python_dunder_version_is_a_version_bearer(self):
        errors, _ = self.run_check(base(**{"py/pkg/__init__.py": '__version__ = "1.6.0"\n'}))
        self.assertTrue(any("py/pkg/__init__.py" in e for e in errors), errors)

    def test_an_independent_file_needs_a_reason_and_is_then_accepted(self):
        cfg = CONFIG + '[[independent]]\nfile = "fuzz/Cargo.toml"\nreason = ""\n'
        files = base(**{"scripts/version-sync.toml": cfg,
                        "fuzz/Cargo.toml": '[package]\nname = "f"\nversion = "0.0.0"\n'})
        errors, _ = self.run_check(files)
        self.assertTrue(any("has no reason" in e for e in errors), errors)
        files["scripts/version-sync.toml"] = cfg.replace('reason = ""', 'reason = "not published"')
        self.assertEqual(self.run_check(files)[0], [])

    def test_a_stale_install_line_in_a_doc_fails_and_a_current_one_passes(self):
        stale = base(**{"docs/USAGE.md": '```toml\nalice-x = "0.14"\n```\n'})
        errors, counts = self.run_check(stale)
        self.assertTrue(any('alice-x = "0.14"' in e for e in errors), errors)
        ok = base(**{"docs/USAGE.md": 'alice-x = { version = "2", features = ["a"] }\n'})
        errors, counts = self.run_check(ok)
        self.assertEqual(errors, [])
        self.assertEqual(counts["install"], 1)

    def test_pip_and_npm_pins_must_equal_the_crate_version(self):
        doc = "pip install alice_x==2.0.0\nnpm install alice-x@2.1.0\n"
        errors, _ = self.run_check(base(**{"README.md": doc}))
        self.assertEqual(len([e for e in errors if "pins" in e]), 1, errors)

    def test_changelogs_and_historical_documents_are_not_checked(self):
        cfg = CONFIG + '[[historical]]\nfile = "docs/ROADMAP.md"\nreason = "release log"\n'
        files = base(**{"scripts/version-sync.toml": cfg,
                        "CHANGELOG.md": 'alice-x = "0.1"\n',
                        "docs/ROADMAP.md": 'alice-x = "0.14.0-preview.4"\n'})
        errors, counts = self.run_check(files)
        self.assertEqual(errors, [])
        self.assertEqual(counts["historical"], 1)

    def test_comparing_no_metadata_file_fails(self):
        files = base(**{"scripts/version-sync.toml": "# nothing\n"})
        files.pop("web/package.json")
        errors, _ = self.run_check(files)
        self.assertTrue(any("compared no metadata" in e for e in errors), errors)

    def test_a_listed_file_that_is_gone_fails(self):
        files = base()
        files.pop("web/package.json")
        errors, _ = self.run_check(files)
        self.assertTrue(any("listed but not tracked" in e for e in errors), errors)

    def test_caret(self):
        self.assertTrue(vs.caret_ok("2", "2.1.0"))
        self.assertTrue(vs.caret_ok("2.0", "2.1.0"))
        self.assertFalse(vs.caret_ok("1.1", "2.0.0"))
        self.assertFalse(vs.caret_ok("0.14", "2.0.0"))
        self.assertTrue(vs.caret_ok("0.14.1", "0.14.3"))
        self.assertFalse(vs.caret_ok("0.13", "0.14.0"))
        self.assertFalse(vs.caret_ok("2.2", "2.1.0"))

    def test_this_repository_is_in_step(self):
        errors, counts = vs.check(os.path.dirname(HERE))
        self.assertEqual(errors, [])
        self.assertGreater(counts["metadata"], 0)


if __name__ == "__main__":
    unittest.main()
