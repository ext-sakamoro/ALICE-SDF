#!/usr/bin/env python3
"""Tests for scripts/alice_siblings_check.py (run: python3 scripts/test_alice_siblings_check.py)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alice_siblings_check as c  # noqa: E402

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def run(cwd: Path, *args: str) -> str:
    return subprocess.run(
        list(args), cwd=cwd, check=True, capture_output=True, text=True, env=ENV
    ).stdout.strip()


def repo_with_two_commits(path: Path) -> tuple[str, str]:
    path.mkdir()
    run(path, "git", "init", "-q")
    run(path, "git", "config", "maintenance.auto", "false")
    run(path, "git", "config", "gc.auto", "0")
    (path / "a.txt").write_text("a\n")
    run(path, "git", "add", "a.txt")
    run(path, "git", "commit", "-qm", "one")
    first = run(path, "git", "rev-parse", "HEAD")
    (path / "a.txt").write_text("b\n")
    run(path, "git", "commit", "-qam", "two")
    return first, run(path, "git", "rev-parse", "HEAD")


class SiblingsCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "ALICE-Crate"
        self.root.mkdir()
        (self.root / ".github").mkdir()
        self.zip_first, self.zip = repo_with_two_commits(base / "ALICE-Zip")
        _, self.db = repo_with_two_commits(base / "ALICE-DB")
        (self.root / "Cargo.toml").write_text(
            "[dependencies]\n"
            'alice-zip = { path = "../ALICE-Zip", version = ">=0.11, <0.12" }\n'
            'alice-db = { path = "../ALICE-DB", version = ">=0.6, <0.7" }\n'
        )
        self.write(f"# pinned\nALICE-Zip {self.zip}\nALICE-DB {self.db}  # db\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, text: str) -> None:
        (self.root / ".github" / "alice-siblings").write_text(text)

    def problems(self) -> list[str]:
        return c.check(self.root)

    def assertProblem(self, needle: str) -> None:
        got = self.problems()
        self.assertTrue(any(needle in p for p in got), got)

    def test_matching_heads_pass(self) -> None:
        self.assertEqual(self.problems(), [])
        self.assertEqual(c.main(["x", str(self.root)]), 0)

    def test_other_commit_fails(self) -> None:
        self.write(f"ALICE-Zip {self.zip_first}\nALICE-DB {self.db}\n")
        self.assertProblem("../ALICE-Zip is at")

    def test_short_or_branch_ref_fails(self) -> None:
        for bad in (self.zip[:12], "main"):
            self.write(f"ALICE-Zip {bad}\nALICE-DB {self.db}\n")
            self.assertProblem("not '<ALICE-Repo> <40-hex SHA>'")

    def test_repeated_repository_fails(self) -> None:
        self.write(f"ALICE-Zip {self.zip}\nALICE-Zip {self.zip}\nALICE-DB {self.db}\n")
        self.assertProblem("listed twice")

    def test_empty_list_beside_a_path_dependency_fails_naming_it(self) -> None:
        self.write("# nothing\n\n")
        self.assertProblem("Cargo.toml:2 names ../ALICE-Zip, which .github/alice-siblings does not list")
        self.assertProblem("Cargo.toml:3 names ../ALICE-DB, which .github/alice-siblings does not list")

    def test_unlisted_manifest_path_fails(self) -> None:
        self.write(f"ALICE-Zip {self.zip}\n")
        self.assertProblem("names ../ALICE-DB, which")

    def test_listed_but_unnamed_fails(self) -> None:
        (self.root / "Cargo.toml").write_text('[dependencies]\nalice-zip = { path = "../ALICE-Zip" }\n')
        self.assertProblem("lists ALICE-DB, which no manifest names")

    def test_fuzz_manifest_must_point_at_the_same_sibling(self) -> None:
        (self.root / "fuzz").mkdir()
        (self.root / "fuzz" / "Cargo.toml").write_text('[dependencies]\nalice-zip = { path = "../ALICE-Zip" }\n')
        self.assertProblem("not the sibling ../ALICE-Zip")
        (self.root / "fuzz" / "Cargo.toml").write_text('[dependencies]\nalice-zip = { path = "../../ALICE-Zip" }\n')
        self.assertEqual(self.problems(), [])

    def test_package_inside_a_sibling_names_that_sibling(self) -> None:
        (self.root / "Cargo.toml").write_text(
            "[dependencies]\n"
            'alice-zip = { path = "../ALICE-Zip/crates/alice-zip" }\n'
            'alice-db = { path = "../ALICE-DB" }\n'
        )
        self.assertEqual(self.problems(), [])
        (self.root / "Cargo.toml").write_text(
            '[dependencies]\nalice-zip = { path = "../../ALICE-Zip/crates/alice-zip" }\nalice-db = { path = "../ALICE-DB" }\n'
        )
        self.assertProblem("not the sibling ../ALICE-Zip")
        (self.root / "Cargo.toml").write_text(
            '[dependencies]\nalice-zip = { path = "../ALICE-Zip/" }\nalice-db = { path = "../ALICE-DB" }\n'
        )
        self.assertEqual(self.problems(), [])

    def test_a_path_that_leaves_the_sibling_is_refused(self) -> None:
        # the repository after ".." is the one the path reaches; it must not be read as ALICE-Zip
        for path in ("../ALICE-Zip/../ALICE-Other", "../ALICE-Zip/./crates", "../ALICE-Zip//crates"):
            (self.root / "Cargo.toml").write_text(
                f'[dependencies]\nalice-zip = {{ path = "{path}" }}\nalice-db = {{ path = "../ALICE-DB" }}\n'
            )
            self.assertProblem(f'names path = "{path}", not ../ALICE-<Repo> or a plain path inside it')

    def test_missing_sibling_fails(self) -> None:
        shutil.move(str(self.root.parent / "ALICE-DB"), str(self.root.parent / "elsewhere"))
        self.assertProblem("../ALICE-DB is not a git checkout")

    def test_dirty_sibling_fails(self) -> None:
        (self.root.parent / "ALICE-Zip" / "a.txt").write_text("dirty\n")
        self.assertProblem("uncommitted")

    def workspace(self, members: str, exclude: str = "") -> None:
        """the root becomes a virtual workspace; ALICE-Zip is named by crates/a, ALICE-DB by tools/deep/b"""
        ex = f"exclude = [{exclude}]\n" if exclude else ""
        (self.root / "Cargo.toml").write_text(f"[workspace]\nmembers = [{members}]\n{ex}")
        for rel, dep in (("crates/a", 'alice-zip = { path = "../../../ALICE-Zip" }'),
                         ("tools/deep/b", 'alice-db = { path = "../../../../ALICE-DB" }'),
                         ("crates/skip", 'alice-zip = { path = "../ALICE-Zip" }')):
            d = self.root / rel
            d.mkdir(parents=True, exist_ok=True)
            (d / "Cargo.toml").write_text(f'[package]\nname = "x"\n\n[dependencies]\n{dep}\n')

    def test_workspace_members_glob_nested_and_exclude(self) -> None:
        self.workspace('"crates/*", "tools/deep/b"', '"crates/skip"')
        self.assertEqual(self.problems(), [])
        # without the exclude, the excluded member's wrong path is read and fails
        self.workspace('"crates/*", "tools/deep/b"')
        self.assertProblem("crates/skip/Cargo.toml names")

    def test_workspace_member_not_read_leaves_a_listed_repo_unnamed(self) -> None:
        self.workspace('"crates/a"')
        self.assertProblem("lists ALICE-DB, which no manifest names")

    def test_workspace_member_fuzz_manifest_is_read(self) -> None:
        self.workspace('"crates/*", "tools/deep/b"', '"crates/skip"')
        (self.root / "crates" / "a" / "fuzz").mkdir()
        (self.root / "crates" / "a" / "fuzz" / "Cargo.toml").write_text('[dependencies]\nalice-zip = { path = "../ALICE-Zip" }\n')
        self.assertProblem("crates/a/fuzz/Cargo.toml names")

    def test_workspace_member_matching_nothing_fails(self) -> None:
        self.workspace('"crates/*", "tools/deep/b", "nowhere/*"', '"crates/skip"')
        self.assertProblem("workspace member 'nowhere/*' matches no package")

    def test_workspace_member_matching_only_dirs_without_a_manifest_fails(self) -> None:
        # the pattern matches an existing directory (docs, or a member whose Cargo.toml was
        # removed) but no package: that is not a member that was read
        self.workspace('"crates/*", "tools/deep/b", "docs/*"', '"crates/skip"')
        (self.root / "docs" / "guide").mkdir(parents=True)
        (self.root / "docs" / "guide" / "README.md").write_text("no manifest here\n")
        self.assertProblem("workspace member 'docs/*' matches no package")

    def test_no_manifest_read_fails(self) -> None:
        (self.root / "Cargo.toml").unlink()
        self.assertProblem("no Cargo.toml was read")

    def test_missing_list_beside_a_path_dependency_fails_naming_it(self) -> None:
        (self.root / ".github" / "alice-siblings").unlink()
        self.assertProblem("Cargo.toml:2 names ../ALICE-Zip, which .github/alice-siblings does not list")

    def no_path_dependency(self) -> None:
        (self.root / "Cargo.toml").write_text(
            "[dependencies]\n"
            'alice-zip = { git = "https://github.com/ext-sakamoro/ALICE-Zip", rev = "%s", version = "0.11" }\n'
            '# alice-db = { path = "../ALICE-DB", version = "0.6" }  (a comment names no dependency)\n'
            'serde = { version = "1", features = ["derive"] }  # path = "../ALICE-DB" in a comment\n' % ("0" * 40)
        )

    def main_output(self) -> tuple[int, str]:
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = c.main(["x", str(self.root)])
        return code, out.getvalue()

    def test_empty_list_with_no_path_dependency_passes_saying_so(self) -> None:
        self.no_path_dependency()
        self.write("# no sibling path dependency\n")
        self.assertEqual(self.problems(), [])
        code, out = self.main_output()
        self.assertEqual(code, 0, out)
        self.assertIn("alice_siblings_check: 1 manifests scanned, 0 path deps, nothing to pin\n", out)

    def test_missing_list_with_no_path_dependency_passes_saying_so(self) -> None:
        self.no_path_dependency()
        (self.root / ".github" / "alice-siblings").unlink()
        self.assertEqual(self.problems(), [])
        self.assertIn("1 manifests scanned, 0 path deps, nothing to pin", self.main_output()[1])

    def test_an_empty_list_with_no_manifest_read_fails(self) -> None:
        (self.root / "Cargo.toml").unlink()
        self.write("")
        self.assertProblem("no Cargo.toml was read (compared nothing)")

    def test_a_single_quoted_path_dependency_counts(self) -> None:
        (self.root / "Cargo.toml").write_text("[dependencies]\nalice-zip = { path = '../ALICE-Zip', version = '0.11' }\n")
        self.write("")
        self.assertProblem("Cargo.toml:2 names ../ALICE-Zip, which .github/alice-siblings does not list")
        self.write(f"ALICE-Zip {self.zip}\n")
        self.assertEqual(self.problems(), [])

    def test_a_path_dependency_toml_hides_from_a_line_scan_still_counts(self) -> None:
        # the manifest is read as TOML: a quoted key, a multi-line literal string, and an escaped quote followed
        # by a "#" earlier on the line all still name the sibling
        cases = {
            "quoted key": '[dependencies]\nalice-zip = { "path" = "../ALICE-Zip", version = "0.11" }\n',
            "multi-line literal": "[dependencies]\nalice-zip = { version = \"0.11\", path = \'\'\'../ALICE-Zip\'\'\' }\n",
            "escaped quote and #": '[dependencies]\nalice-zip = { version = "0.11", features = ["a\\" # x"], path = "../ALICE-Zip" }\n',
            "dotted key": "[dependencies]\nalice-zip.path = \"../ALICE-Zip\"\nalice-zip.version = \"0.11\"\n",
            "target table": "[target.'cfg(unix)'.dependencies]\nalice-zip = { path = \"../ALICE-Zip\" }\n",
            "dev-dependencies": "[dev-dependencies]\nalice-zip = { path = \"../ALICE-Zip\" }\n",
            "build-dependencies": "[build-dependencies]\nalice-zip = { path = \"../ALICE-Zip\" }\n",
            "workspace dependencies": "[workspace.dependencies]\nalice-zip = { path = \"../ALICE-Zip\" }\n",
            "patch": "[patch.crates-io]\nalice-zip = { path = \"../ALICE-Zip\" }\n",
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                (self.root / "Cargo.toml").write_text(text)
                self.write("")
                self.assertProblem("names ../ALICE-Zip, which .github/alice-siblings does not list")

    def test_a_manifest_that_does_not_parse_fails_closed(self) -> None:
        (self.root / "Cargo.toml").write_text("[dependencies\nalice-zip = { path = ")
        self.write("")
        self.assertProblem("Cargo.toml is not TOML")

    def test_a_path_that_is_not_a_dependency_is_not_a_sibling(self) -> None:
        # [lib] / [[bin]] paths, and a string that only mentions a sibling, name no dependency
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "x"\nversion = "0.1.0"\ndescription = "see path = \\"../ALICE-Zip\\""\n'
            '[lib]\npath = "src/lib.rs"\n[[bin]]\nname = "b"\npath = "src/bin/b.rs"\n')
        self.write("")
        self.assertEqual(self.problems(), [])

    def test_read_list_of_a_missing_or_empty_file_is_an_empty_mapping(self) -> None:
        self.write("# none\n")
        self.assertEqual(c.read_list(self.root), ({}, []))
        (self.root / ".github" / "alice-siblings").unlink()
        self.assertEqual(c.read_list(self.root), ({}, []))



ACTION = Path(__file__).resolve().parent.parent / ".github" / "actions" / "alice-siblings" / "action.yml"


def action_script() -> str:
    """the bash of .github/actions/alice-siblings/action.yml (the `run: |` block)"""
    lines = ACTION.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "run: |") + 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for l in lines[start:]:
        if l.strip() and len(l) - len(l.lstrip()) < indent:
            break
        body.append(l[indent:])
    return "\n".join(body) + "\n"


@unittest.skipIf(os.name == "nt" or not shutil.which("bash"), "bash with POSIX paths (the Windows CI jobs run the action itself)")
class SiblingsAction(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.remotes = base / "remotes"
        self.remotes.mkdir()
        self.first, self.second = repo_with_two_commits(self.remotes / "ALICE-Zip")
        self.work = base / "work"
        self.ws = self.work / "ws"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_action(self, repo_dir: str, listing: str) -> subprocess.CompletedProcess:
        list_dir = (self.ws / repo_dir) / ".github"
        list_dir.mkdir(parents=True, exist_ok=True)
        (list_dir / "alice-siblings").write_text(listing)
        env = {**ENV, "GITHUB_WORKSPACE": str(self.ws), "REPO_DIR": repo_dir,
               "ALICE_SIBLINGS_GIT_BASE": str(self.remotes)}
        return subprocess.run(["bash", "-c", action_script()], env=env, capture_output=True, text=True)

    def test_default_repo_dir_clones_next_to_the_workspace(self) -> None:
        r = self.run_action(".", f"ALICE-Zip {self.first}\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        dest = self.work / "ALICE-Zip"
        self.assertEqual(run(dest, "git", "rev-parse", "HEAD"), self.first)
        self.assertIn(f"ALICE-Zip sibling at {self.first}", r.stdout)

    def test_repo_dir_clones_next_to_that_directory(self) -> None:
        r = self.run_action("ALICE-LOL", f"ALICE-Zip {self.second}\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(run(self.ws / "ALICE-Zip", "git", "rev-parse", "HEAD"), self.second)
        self.assertFalse((self.work / "ALICE-Zip").exists())

    def test_empty_list_fails(self) -> None:
        r = self.run_action(".", "# nothing\n")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("lists no repository", r.stdout + r.stderr)

if __name__ == "__main__":
    unittest.main()
