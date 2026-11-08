"""Oracles of scripts/package_git_patches.py.

run: python3 scripts/test_package_git_patches.py
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "package_git_patches", os.path.join(HERE, "package_git_patches.py")
)
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)
import id_move_check as imc  # noqa: E402  (the module package_git_patches reads branch commits with)

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
        self.assertEqual(pinned, [("sib", URL, REV, f"rev={REV}")])

    def test_the_patch_arguments_name_the_pinned_rev(self):
        args = pg.config_args([("sib", URL, REV, f"rev={REV}")])
        self.assertEqual(args, ["--config", f'patch.crates-io.sib.git="{URL}"',
                                "--config", f'patch.crates-io.sib.rev="{REV}"'])

    def test_a_short_rev_is_not_a_pin(self):
        _, problems = pg.pins(MANIFEST.replace(REV, REV[:12]))
        self.assertEqual(len(problems), 1)

    def test_a_branch_dependency_without_a_lock_is_not_a_pin(self):
        with tempfile.TemporaryDirectory() as d:
            _, problems = pg.pins(MANIFEST.replace(f'rev = "{REV}"', 'branch = "main"'), d)
        self.assertEqual(len(problems), 1)
        self.assertIn("no Cargo.lock", problems[0])

    def test_a_target_table_is_read(self):
        text = MANIFEST + f'\n[target.\'cfg(unix)\'.dev-dependencies]\nt = {{ git = "{URL}", rev = "{OTHER}", version = "2" }}\n'
        pinned, problems = pg.pins(text)
        self.assertEqual(problems, [])
        self.assertIn(("t", URL, OTHER, f"rev={OTHER}"), pinned)


class BranchDependency(unittest.TestCase):
    """A git dependency without a `rev` is pinned at the commit its Cargo.lock resolved it to,
    read by id_move_check's locked_git_commit (the one place that rule lives)"""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="pgp-branch-"))
        self.manifest = MANIFEST.replace(f'rev = "{REV}"', 'branch = "main"')

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def write_lock(self, source: str | None, name: str = "sib"):
        body = 'version = 4\n\n[[package]]\nname = "probe"\nversion = "1.0.0"\n'
        if source is not None:
            body += f'\n[[package]]\nname = "{name}"\nversion = "0.4.0"\nsource = "{source}"\n'
        (self.d / "Cargo.lock").write_text(body, encoding="utf-8")

    def test_a_branch_dependency_is_pinned_at_its_locked_commit(self):
        self.write_lock(f"git+{URL}?branch=main#{REV}")
        pinned, problems = pg.pins(self.manifest, str(self.d))
        self.assertEqual(problems, [])
        self.assertEqual(pinned, [("sib", URL, REV, "branch=main")])

    def test_both_scripts_read_the_same_commit_from_the_same_lock(self):
        self.write_lock(f"git+{URL}?branch=main#{REV}")
        pinned, _ = pg.pins(self.manifest, str(self.d))
        self.assertEqual(pinned[0][2], imc.locked_git_commit(self.d, self.d, "sib", URL))
        self.assertIs(pg.locked_git_commit, imc.locked_git_commit)

    def test_the_patch_keeps_the_branch_and_names_no_rev(self):
        # a rev would give cargo a second source for a package its git dependencies take by branch
        self.write_lock(f"git+{URL}?branch=main#{REV}")
        pinned, _ = pg.pins(self.manifest, str(self.d))
        args = pg.config_args(pinned)
        self.assertIn('patch.crates-io.sib.branch="main"', args)
        self.assertFalse(any(".rev=" in a for a in args), args)

    def test_a_tag_and_the_default_branch_keep_their_reference(self):
        for spec, ref, query in (('tag = "v1"', "tag=v1", "?tag=v1"), ("", "", "")):
            with self.subTest(spec=spec):
                manifest = MANIFEST.replace(f', rev = "{REV}"', f", {spec}" if spec else "")
                self.write_lock(f"git+{URL}{query}#{REV}")
                pinned, problems = pg.pins(manifest, str(self.d))
                self.assertEqual(problems, [])
                self.assertEqual(pinned, [("sib", URL, REV, ref)])
                self.assertEqual(pg.source_of(URL, REV, ref), f"git+{URL}{query}#{REV}")

    def test_two_references_are_refused(self):
        manifest = MANIFEST.replace(f'rev = "{REV}"', f'rev = "{REV}", branch = "main"')
        _, problems = pg.pins(manifest, str(self.d))
        self.assertEqual(len(problems), 1)
        self.assertIn("more than one", problems[0])

    def test_the_refusals_are_the_shared_rule_s(self):
        for source, name, word in ((None, "sib", "is not in it"),
                                   (f"git+{URL}?branch=main#{REV}", "other", "is not in it"),
                                   (f"git+{URL}?branch=main#{REV[:12]}", "sib", "not a 40-hex commit")):
            with self.subTest(source=source, name=name):
                self.write_lock(source, name)
                _, problems = pg.pins(self.manifest, str(self.d))
                self.assertEqual(len(problems), 1)
                self.assertIn(word, problems[0])


class Verify(unittest.TestCase):
    PINNED = [("sib", URL, REV, f"rev={REV}")]
    BRANCH = [("sib", URL, REV, "branch=main")]
    BRANCH_SOURCE = f"git+{URL}?branch=main#{REV}"

    def test_a_branch_pin_and_a_git_dependency_taking_it_by_the_same_branch_pass(self):
        # the packaged crate's other git dependency (here `user`, of the pinned repository) takes
        # `sib` by branch too: one source for `sib`, as the patch by branch gives
        extra = f'\n[[package]]\nname = "user"\nversion = "0.7.0"\nsource = "{self.BRANCH_SOURCE}"\n'
        self.assertEqual(pg.verify(self.BRANCH, lock(self.BRANCH_SOURCE, extra)), [])

    def test_the_same_package_from_a_rev_and_from_a_branch_fails(self):
        # 2026-11-08: patched by rev while alice-analytics took alice-det-math by branch = two copies
        extra = f'\n[[package]]\nname = "sib"\nversion = "0.4.0"\nsource = "{GOOD_SOURCE}"\n'
        problems = pg.verify(self.BRANCH, lock(self.BRANCH_SOURCE, extra))
        self.assertTrue(any("resolved from" in p for p in problems), problems)

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

    def test_another_crate_of_the_pinned_repository_at_the_pinned_rev_passes(self):
        # a facade pinned by git reaches the crates of its repository by path: the lock
        # gives them the facade's source
        extra = f'\n[[package]]\nname = "sib-core"\nversion = "0.4.0"\nsource = "{GOOD_SOURCE}"\n'
        self.assertEqual(pg.verify(self.PINNED, lock(GOOD_SOURCE, extra)), [])

    def test_another_repository_at_the_pinned_rev_fails(self):
        other_url = "https://example.invalid/other"
        extra = f'\n[[package]]\nname = "sib-core"\nversion = "0.4.0"\nsource = "git+{other_url}?rev={REV}#{REV}"\n'
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
