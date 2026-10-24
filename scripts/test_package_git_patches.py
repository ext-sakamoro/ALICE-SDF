"""Oracles of scripts/package_git_patches.py.

run: python3 scripts/test_package_git_patches.py
"""
from __future__ import annotations

import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "package_git_patches", os.path.join(HERE, "package_git_patches.py")
)
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

URL = "https://example.invalid/sibling"
REV = "5d45fb65de47a6a3b39a35f235a393ead8f2eb38"
OTHER = "0" * 40
MANIFEST = f"""
[package]
name = "probe"
version = "1.0.0"

[dependencies]
sib = {{ git = "{URL}", rev = "{REV}", version = "0.4", optional = true }}
plain = "1"
local = {{ path = "../local", version = "0.1" }}
"""


def lock(source: str | None, extra: str = "") -> str:
    entry = '[[package]]\nname = "sib"\nversion = "0.4.0"\n'
    if source is not None:
        entry += f'source = "{source}"\n'
    return 'version = 4\n\n[[package]]\nname = "probe"\nversion = "1.0.0"\n\n' + entry + extra


GOOD_SOURCE = f"git+{URL}?rev={REV}#{REV}"


class Pins(unittest.TestCase):
    def test_a_git_dependency_with_a_version_is_pinned(self):
        pinned, problems = pg.pins(MANIFEST)
        self.assertEqual(problems, [])
        self.assertEqual(pinned, [("sib", URL, REV)])

    def test_the_patch_arguments_name_the_pinned_rev(self):
        args = pg.config_args([("sib", URL, REV)])
        self.assertEqual(args, ["--config", f'patch.crates-io.sib.git="{URL}"',
                                "--config", f'patch.crates-io.sib.rev="{REV}"'])

    def test_a_short_rev_is_not_a_pin(self):
        _, problems = pg.pins(MANIFEST.replace(REV, REV[:12]))
        self.assertEqual(len(problems), 1)

    def test_a_branch_instead_of_a_rev_is_not_a_pin(self):
        _, problems = pg.pins(MANIFEST.replace(f'rev = "{REV}"', 'branch = "main"'))
        self.assertEqual(len(problems), 1)

    def test_a_target_table_is_read(self):
        text = MANIFEST + f'\n[target.\'cfg(unix)\'.dev-dependencies]\nt = {{ git = "{URL}", rev = "{OTHER}", version = "2" }}\n'
        pinned, problems = pg.pins(text)
        self.assertEqual(problems, [])
        self.assertIn(("t", URL, OTHER), pinned)


class Verify(unittest.TestCase):
    PINNED = [("sib", URL, REV)]

    def test_the_lock_at_the_pinned_rev_passes(self):
        self.assertEqual(pg.verify(self.PINNED, lock(GOOD_SOURCE)), [])

    def test_a_registry_resolution_fails_as_patch_not_applied(self):
        problems = pg.verify(self.PINNED, lock("registry+https://github.com/rust-lang/crates.io-index"))
        self.assertEqual(len(problems), 1)
        self.assertIn("patch not applied", problems[0])

    def test_another_rev_fails(self):
        problems = pg.verify(self.PINNED, lock(GOOD_SOURCE.replace(REV, OTHER)))
        self.assertTrue(problems)

    def test_a_missing_entry_fails(self):
        problems = pg.verify(self.PINNED, 'version = 4\n\n[[package]]\nname = "probe"\nversion = "1.0.0"\n')
        self.assertEqual(len(problems), 1)
        self.assertIn("not in the packaged", problems[0])

    def test_a_git_source_not_pinned_by_the_manifest_fails(self):
        extra = f'\n[[package]]\nname = "stray"\nversion = "1.0.0"\nsource = "git+{URL}?rev={OTHER}#{OTHER}"\n'
        problems = pg.verify(self.PINNED, lock(GOOD_SOURCE, extra))
        self.assertEqual(len(problems), 1)
        self.assertIn("does not pin", problems[0])

    def test_with_nothing_pinned_a_stray_git_source_still_fails(self):
        self.assertTrue(pg.verify([], lock(GOOD_SOURCE)))


class ThisRepository(unittest.TestCase):
    def test_the_manifest_has_no_unpinned_git_dependency(self):
        with open(os.path.join(pg.ROOT, "Cargo.toml"), encoding="utf-8") as f:
            _, problems = pg.pins(f.read())
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
