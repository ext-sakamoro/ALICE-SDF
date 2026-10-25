#!/usr/bin/env python3
"""Tests of scripts/release_baseline.py (the version-bump baseline of a repository under a
publication freeze).

Fixtures are throwaway git repositories built here; the expected baseline of each is known from
how the history was written (which commit changed `[package].version`), not from the code under
test. Nothing is fetched.

The baseline is the PARENT of the newest commit on main's first-parent line that changed
`[package].version`: the tree and the version before the bump, so everything since then
(the bump commit included) is what HEAD's version is judged against.

- linear history: the parent of the newest bump;
- HEAD being the bump commit needs no special case (its parent is the baseline);
- HEAD on a branch off main: the bump is looked for on main, from the merge base;
- a version that moved only on a merged branch's own commits (the merge leaves it unchanged) is
  not a bump; a merge whose first parent had another version is one;
- a pre-release-only step (7.0.0-beta.1 -> 7.0.0-beta.2) is a bump;
- bump then un-bump: the newest change wins;
- `version.workspace = true` is read from the workspace root at the same commit;
- a renamed package is followed through `base_package`; without it the rename is an error, and so is a
  moved manifest: "introduced" (nothing to compare) is only a package that no earlier first-parent commit
  had, or one that came with the root commit;
- a manifest that is there but does not parse is an error, never "absent";
- every case is answered alike by the walk over the commits that touch the version files and by the walk
  over every first-parent commit;
- a package that is not in the manifest at HEAD is an error naming the package and the ref;
- the same repository gives the same answer twice, and the CLI prints `<sha> <version>` or
  `introduced`.

usage: python3 scripts/test_release_baseline.py
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "release_baseline.py"
# the fixtures' git and the script's git read no system or global configuration (a host's
# commit.gpgsign would otherwise decide the outcome)
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
ENV.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)


def load():
    spec = importlib.util.spec_from_file_location("release_baseline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid",
                           "-c", "init.defaultBranch=main", *args], cwd=repo, check=True,
                          capture_output=True, text=True, env=ENV).stdout.strip()


def toml(name: str, version: str) -> str:
    return f'[package]\nname = "{name}"\nversion = "{version}"\nedition = "2021"\n'


class Baseline(unittest.TestCase):
    def setUp(self) -> None:
        self.rb = load()
        self.repo = Path(tempfile.mkdtemp(prefix="release-baseline-test-"))
        self.addCleanup(shutil.rmtree, self.repo, True)
        git(self.repo, "init", "-q")

    def commit(self, files: dict[str, str], msg: str = "c") -> str:
        for rel, text in files.items():
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        (self.repo / "touch").mkdir(exist_ok=True)
        (self.repo / "touch" / msg.replace(" ", "-")).write_text(msg)  # one file per commit: merges do not conflict
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", msg)
        return git(self.repo, "rev-parse", "HEAD")

    def base(self, package: str = "alice-x", manifest: str = "Cargo.toml", **kw):
        # both walks, every time: the commits that touch the version files, and every first-parent commit
        if "prefilter" in kw:
            return self.rb.bump_baseline(self.repo, manifest, package, main_ref="main", **kw)
        fast = self.rb.bump_baseline(self.repo, manifest, package, main_ref="main", prefilter=True, **kw)
        full = self.rb.bump_baseline(self.repo, manifest, package, main_ref="main", prefilter=False, **kw)
        self.assertEqual(fast, full, "the path-filtered walk and the full first-parent walk disagree")
        return fast

    def linear(self) -> list[str]:
        c1 = self.commit({"Cargo.toml": toml("alice-x", "0.1.0")}, "c1")
        c2 = self.commit({"src.rs": "a"}, "c2")
        c3 = self.commit({"Cargo.toml": toml("alice-x", "0.2.0")}, "c3 bump")
        c4 = self.commit({"src.rs": "b"}, "c4")
        return [c1, c2, c3, c4]

    def test_linear_history_takes_the_parent_of_the_newest_bump(self):
        c1, c2, c3, c4 = self.linear()
        b = self.base()
        self.assertEqual((b.sha, b.version, b.introduced), (c2, "0.1.0", False))

    def test_head_being_the_bump_takes_its_parent(self):
        c1, c2, c3, c4 = self.linear()
        git(self.repo, "checkout", "-q", "--detach", c3)
        git(self.repo, "branch", "-q", "-f", "main", c3)
        b = self.base()
        self.assertEqual((b.sha, b.version), (c2, "0.1.0"))

    def test_a_branch_off_main_looks_for_the_bump_on_main(self):
        c1, c2, c3, c4 = self.linear()
        git(self.repo, "checkout", "-q", "-b", "work")
        self.commit({"Cargo.toml": toml("alice-x", "0.3.0")}, "c5 bump on the branch")
        b = self.base()
        self.assertEqual((b.sha, b.version), (c2, "0.1.0"))

    def test_a_version_that_moved_only_on_a_merged_branch_is_not_a_bump(self):
        c1, c2, c3, c4 = self.linear()
        git(self.repo, "checkout", "-q", "-b", "side", c2)
        self.commit({"Cargo.toml": toml("alice-x", "0.9.0")}, "s1")
        self.commit({"Cargo.toml": toml("alice-x", "0.2.0")}, "s2")
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
        b = self.base()
        self.assertEqual((b.sha, b.version), (c2, "0.1.0"))

    def test_a_merge_whose_first_parent_had_another_version_is_a_bump(self):
        c1, c2, c3, c4 = self.linear()
        git(self.repo, "checkout", "-q", "-b", "side")
        self.commit({"Cargo.toml": toml("alice-x", "0.3.0")}, "s bump")
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
        b = self.base()
        self.assertEqual((b.sha, b.version), (c4, "0.2.0"))

    def test_a_pre_release_only_step_is_a_bump(self):
        c1 = self.commit({"Cargo.toml": toml("alice-x", "7.0.0-beta.1")}, "p1")
        self.commit({"Cargo.toml": toml("alice-x", "7.0.0-beta.2")}, "p2")
        self.commit({"src.rs": "x"}, "p3")
        b = self.base()
        self.assertEqual((b.sha, b.version), (c1, "7.0.0-beta.1"))

    def test_bump_then_un_bump_takes_the_newest_change(self):
        self.commit({"Cargo.toml": toml("alice-x", "0.1.0")}, "u1")
        up = self.commit({"Cargo.toml": toml("alice-x", "0.2.0")}, "u2 bump")
        self.commit({"Cargo.toml": toml("alice-x", "0.1.0")}, "u3 un-bump")
        b = self.base()
        self.assertEqual((b.sha, b.version), (up, "0.2.0"))

    def test_a_workspace_inherited_version_is_read_from_the_root(self):
        ws = '[workspace]\nmembers = ["crates/x"]\n\n[workspace.package]\nversion = "{v}"\n'
        member = '[package]\nname = "alice-x"\nversion.workspace = true\nedition = "2021"\n'
        w1 = self.commit({"Cargo.toml": ws.format(v="1.0.0"), "crates/x/Cargo.toml": member}, "w1")
        self.commit({"Cargo.toml": ws.format(v="1.1.0")}, "w2 bump")
        self.commit({"src.rs": "x"}, "w3")
        b = self.base(manifest="crates/x/Cargo.toml")
        self.assertEqual((b.sha, b.version), (w1, "1.0.0"))

    def test_a_renamed_package_is_followed_through_base_package(self):
        c1, c2, c3, c4 = self.linear()
        self.commit({"Cargo.toml": toml("alice-y", "0.2.0")}, "rename")
        self.commit({"src.rs": "c"}, "after")
        b = self.base(package="alice-y", base_package="alice-x")
        self.assertEqual((b.sha, b.version, b.introduced), (c2, "0.1.0", False))
        # without base_package the crate is absent at the rename's parent but was there earlier: an error
        with self.assertRaises(self.rb.BaselineError):
            self.base(package="alice-y")

    def test_a_package_introduced_at_the_root_commit_has_no_parent(self):
        self.commit({"Cargo.toml": toml("alice-x", "0.1.0")}, "r1")
        self.commit({"src.rs": "a"}, "r2")
        b = self.base()
        self.assertEqual((b.sha, b.version, b.introduced), (None, None, True))

    def test_a_package_missing_at_head_is_an_error_naming_it_and_the_ref(self):
        self.linear()
        with self.assertRaises(self.rb.BaselineError) as e:
            self.base(package="alice-nope")
        self.assertIn("alice-nope", str(e.exception))
        self.assertIn("main", str(e.exception))

    def test_the_path_prefilter_gives_the_full_walk_answer(self):
        # many commits that do not touch the manifest, merges in both directions, a bump in between:
        # reading only the commits that touch Cargo.toml (and the workspace roots above it) must give
        # what reading every first-parent commit gives
        self.commit({"Cargo.toml": toml("alice-x", "0.1.0")}, "f0")
        for i in range(20):
            self.commit({f"src/{i}.rs": str(i)}, f"n{i}")
        git(self.repo, "checkout", "-q", "-b", "side")
        for i in range(10):
            self.commit({f"side/{i}.rs": str(i)}, f"s{i}")
        self.commit({"Cargo.toml": toml("alice-x", "0.9.0")}, "s-bump")
        self.commit({"Cargo.toml": toml("alice-x", "0.1.0")}, "s-unbump")
        git(self.repo, "checkout", "-q", "main")
        for i in range(10):
            self.commit({f"src/m{i}.rs": str(i)}, f"m{i}")
        git(self.repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
        pre = git(self.repo, "rev-parse", "HEAD")
        self.commit({"Cargo.toml": toml("alice-x", "0.2.0")}, "bump")
        for i in range(10):
            self.commit({f"src/t{i}.rs": str(i)}, f"t{i}")
        git(self.repo, "checkout", "-q", "-b", "side2")
        self.commit({"Cargo.toml": toml("alice-x", "0.3.0")}, "s2-bump")
        git(self.repo, "checkout", "-q", "main")
        self.commit({"src/u.rs": "u"}, "u")
        git(self.repo, "merge", "-q", "--no-ff", "-m", "merge side2", "side2")
        merge2 = git(self.repo, "rev-parse", "HEAD^1")
        for i in range(5):
            self.commit({f"src/v{i}.rs": str(i)}, f"v{i}")
        fast, full = self.base(), self.base(prefilter=False)
        self.assertEqual(fast, full)
        self.assertEqual((fast.sha, fast.version), (merge2, "0.2.0"))
        # and before the second merge, the plain bump
        git(self.repo, "checkout", "-q", "--detach", "HEAD~6")
        git(self.repo, "branch", "-q", "-f", "main", "HEAD")
        fast, full = self.base(), self.base(prefilter=False)
        self.assertEqual(fast, full)
        self.assertEqual((fast.sha, fast.version), (pre, "0.1.0"))

    def test_a_manifest_that_does_not_parse_at_the_parent_is_an_error(self):
        self.commit({"Cargo.toml": toml("alice-x", "0.1.0")}, "c1")
        self.commit({"Cargo.toml": "[package\nname = broken\n"}, "c2 broken")
        self.commit({"Cargo.toml": toml("alice-x", "0.2.0")}, "c3 fixed")
        for prefilter in (True, False):
            with self.assertRaises(self.rb.BaselineError) as e:
                self.base(prefilter=prefilter)
            self.assertIn("not TOML", str(e.exception))

    def test_a_moved_manifest_is_an_error_not_introduced(self):
        self.commit({"Cargo.toml": toml("alice-x", "0.1.0")}, "c1")
        (self.repo / "Cargo.toml").unlink()
        self.commit({"crates/x/Cargo.toml": toml("alice-x", "0.2.0")}, "c2 moved")
        for prefilter in (True, False):
            with self.assertRaises(self.rb.BaselineError) as e:
                self.base(manifest="crates/x/Cargo.toml", prefilter=prefilter)
            self.assertIn("alice-x", str(e.exception))
            self.assertIn("earlier", str(e.exception))

    def test_a_rename_without_base_package_is_an_error_not_introduced(self):
        self.commit({"Cargo.toml": toml("alice-old", "0.1.0")}, "c1")
        self.commit({"Cargo.toml": toml("alice-x", "0.2.0")}, "c2 renamed")
        for prefilter in (True, False):
            with self.assertRaises(self.rb.BaselineError) as e:
                self.base(prefilter=prefilter)
            self.assertIn("base_package", str(e.exception))

    def test_a_package_added_after_the_root_commit_is_introduced(self):
        self.commit({"README": "r"}, "c1")
        self.commit({"Cargo.toml": toml("alice-x", "0.1.0")}, "c2 added")
        self.assertTrue(self.base().introduced)

    def test_the_answer_is_deterministic(self):
        self.linear()
        self.assertEqual(self.base(), self.base())

    def test_the_cli_prints_sha_and_version_or_introduced(self):
        c1, c2, c3, c4 = self.linear()
        args = [sys.executable, str(SCRIPT), "--root", str(self.repo), "--manifest", "Cargo.toml", "--main", "main",
                "--print", "--package"]
        r = subprocess.run(args + ["alice-x"], capture_output=True, text=True, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), f"{c2} 0.1.0")
        r = subprocess.run(args + ["alice-nope"], capture_output=True, text=True, env=ENV)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "")
        # a new crate at a new manifest (replacing the manifest of alice-x would be a rename, an error)
        self.commit({"crates/z/Cargo.toml": toml("alice-z", "0.1.0")}, "z")
        zargs = [a if a != "Cargo.toml" else "crates/z/Cargo.toml" for a in args]
        r = subprocess.run(zargs + ["alice-z"], capture_output=True, text=True, env=ENV)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "introduced"), r.stderr)

if __name__ == "__main__":
    unittest.main()
