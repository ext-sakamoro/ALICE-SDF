#!/usr/bin/env python3
"""Tests for scripts/package_integrity.py (run: python3 scripts/test_package_integrity.py).

The registry and cargo are injected: `versions_of` stands for the crates.io index and the
runner for cargo, so each case fixes exactly what is published and what cargo reports.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tarfile
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_integrity as pi  # noqa: E402

SHA = "72cd19faf37760e228ba6155fd3bdb5a0a85bcfa"


class Requirements(unittest.TestCase):
    def test_range_and_caret_forms(self) -> None:
        self.assertTrue(pi.satisfies((0, 11, 0), ">=0.11, <0.12"))
        self.assertFalse(pi.satisfies((0, 10, 0), ">=0.11, <0.12"))
        self.assertFalse(pi.satisfies((0, 12, 0), ">=0.11, <0.12"))
        self.assertTrue(pi.satisfies((0, 10, 3), "0.10"))
        self.assertFalse(pi.satisfies((0, 11, 0), "0.10"))
        self.assertTrue(pi.satisfies((7, 4, 0), "7"))
        self.assertFalse(pi.satisfies((8, 0, 0), ">=7, <8"))
        self.assertTrue(pi.satisfies((0, 0, 3), "^0.0.3"))
        self.assertFalse(pi.satisfies((0, 0, 4), "^0.0.3"))
        self.assertTrue(pi.satisfies((1, 2, 9), "~1.2"))
        self.assertFalse(pi.satisfies((1, 3, 0), "~1.2"))
        self.assertTrue(pi.satisfies((0, 5, 1), "=0.5.1"))
        self.assertFalse(pi.satisfies((0, 5, 2), "=0.5.1"))

    def test_unsupported_syntax_is_a_problem(self) -> None:
        for bad in ("0.11-beta", ">= ", "0.11,", "latest"):
            with self.assertRaises(pi.Problem, msg=bad):
                pi.comparators(bad)


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "ALICE-Crate"
        (self.root / ".github").mkdir(parents=True)
        (self.root / ".github" / "alice-siblings").write_text(f"ALICE-Zip {SHA}\n")
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "alice-crate"\nversion = "0.6.0"\n\n[dependencies]\n'
            'alice-zip = { path = "../ALICE-Zip", version = ">=0.11, <0.12", optional = true }\n'
        )
        (base / "ALICE-Zip").mkdir()
        (base / "ALICE-Zip" / "Cargo.toml").write_text('[package]\nname = "alice-zip"\nversion = "0.11.0"\n')
        self.target = base / "target"
        self.calls: list[list[str]] = []
        self.published = {"alice-zip": [(0, 10, 0), (0, 9, 0)]}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def versions_of(self, name: str) -> list:
        return self.published[name]

    def write_crate(self) -> None:
        out = self.target / "package"
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out / "alice-crate-0.6.0.crate", "w:gz") as tar:
            data = b'[package]\nname = "alice-crate"\nversion = "0.6.0"\n'
            info = tarfile.TarInfo("alice-crate-0.6.0/Cargo.toml")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    def runner(
        self,
        package_error: str | None = None,
        build_error: str | None = None,
        files: str = "Cargo.toml\nREADME.md\nsrc/lib.rs\n",
        doc_out: str = "test result: ok. 2 passed; 0 failed; 0 ignored",
        test_out: str = "test result: ok. 3 passed; 0 failed; 0 ignored",
        list_stderr: str = "",
    ):
        def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
            self.calls.append(cmd)
            if cmd[1] == "package" and "--list" in cmd:
                return 0, files, list_stderr
            if cmd[1] == "test":
                out = doc_out if "--doc" in cmd else test_out
                return (1 if " 0 failed" not in out else 0), out, ""
            if cmd[1] == "package":
                if package_error:
                    return 101, "", package_error
                self.write_crate()
                return 0, "", "Packaged"
            if build_error:
                return 101, "", build_error
            if cmd[1] == "generate-lockfile":
                (cwd / "Cargo.lock").write_text('[[package]]\nname = "alice-det-math"\n')
                return 0, "", "Locking"
            if cmd[1] == "update":
                return 0, "", "Updating"
            if "--locked" not in cmd:
                (cwd / "Cargo.lock").write_text('[[package]]\nname = "alice-det-math"\n')
            return 0, "", "Finished"

        return run

    def main(self, *argv: str, runner=None) -> tuple[int, str]:
        buf = io.StringIO()
        os.environ["CARGO_TARGET_DIR"] = str(self.target)
        with redirect_stdout(buf):
            rc = pi.main(["--root", str(self.root), *argv], versions_of=self.versions_of, runner=runner or self.runner())
        return rc, buf.getvalue()


class PackageIntegrity(Fixture):
    def test_unpublished_sibling_is_blocked_patched_and_green(self) -> None:
        rc, out = self.main("--expect-in-lock", "alice-det-math")
        self.assertEqual(rc, 0, out)
        self.assertIn("blocked: alice-zip >=0.11, <0.12 unpublished (crates.io max 0.10.0)", out)
        self.assertIn("blocked 1", out)
        self.assertIn("packages inspected 1", out)
        self.assertTrue(all(any("patch.crates-io.alice-zip.path=" in a for a in c) for c in self.calls), self.calls)

    def test_non_sibling_packaging_error_stays_red(self) -> None:
        rc, out = self.main(runner=self.runner(package_error="error: failed to read `src/missing.rs`"))
        self.assertEqual(rc, 1, out)
        self.assertIn("cargo package failed", out)
        rc, out = self.main(runner=self.runner(build_error="error[E0432]: unresolved import"))
        self.assertEqual(rc, 1, out)

    def test_sibling_on_the_registry_cannot_hide_a_failure(self) -> None:
        self.published["alice-zip"] = [(0, 11, 0), (0, 10, 0)]
        failure = "failed to select a version for the requirement `alice-zip`"
        rc, out = self.main(runner=self.runner(package_error=failure))
        self.assertEqual(rc, 1, out)
        self.assertIn("blocked 0", out)
        self.assertFalse(any("patch.crates-io" in a for c in self.calls for a in c), self.calls)

    def test_publish_gate_is_red_while_anything_is_blocked(self) -> None:
        rc, out = self.main("--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("publish gate: FAIL", out)
        self.assertEqual(self.calls, [])
        self.published["alice-zip"] = [(0, 11, 0)]
        rc, out = self.main("--publish-gate")
        self.assertEqual(rc, 0, out)
        self.assertIn("publish gate: OK", out)

    def test_no_crate_produced_fails(self) -> None:
        def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
            return 0, "", "Packaged"

        rc, out = self.main(runner=run)
        self.assertEqual(rc, 1, out)
        self.assertIn("inspected nothing", out)

    def test_sibling_version_must_satisfy_the_requirement(self) -> None:
        (self.root.parent / "ALICE-Zip" / "Cargo.toml").write_text('[package]\nname = "alice-zip"\nversion = "0.10.0"\n')
        rc, out = self.main()
        self.assertEqual(rc, 1, out)
        self.assertIn("does not satisfy", out)

    def test_unlisted_sibling_path_fails(self) -> None:
        (self.root / ".github" / "alice-siblings").write_text(f"ALICE-DB {SHA}\n")
        rc, out = self.main()
        self.assertEqual(rc, 1, out)
        self.assertIn("not in .github/alice-siblings", out)

    def test_expected_lock_entry_missing_fails(self) -> None:
        rc, out = self.main("--expect-in-lock", "alice-not-there")
        self.assertEqual(rc, 1, out)
        self.assertIn("did not resolve alice-not-there", out)

    def test_unreadable_index_fails(self) -> None:
        def down(name: str) -> list:
            raise OSError("network down")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pi.main(["--root", str(self.root)], versions_of=down, runner=self.runner())
        self.assertEqual(rc, 1, buf.getvalue())
        self.assertIn("index unreadable", buf.getvalue())



GIT = "https://github.com/ext-sakamoro/ALICE-DetMath"
REV = "2d01c63ff0849cf0ab82eafb89f87f1b729bd2a8"
PINNED = f"git+{GIT}?rev={REV}#{REV}"


class GitBase(Fixture):
    """the manifest, locks and runner of the git dependency tests"""

    def setUp(self) -> None:
        super().setUp()
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "alice-crate"\nversion = "0.6.0"\n\n[dependencies]\n'
            f'alice-det-math = {{ git = "{GIT}", rev = "{REV}", version = "0.5.2" }}\n'
        )
        (self.root / ".github" / "alice-siblings").write_text("")
        self.published = {"alice-det-math": [(0, 5, 1), (0, 4, 0)]}

    def lock(self, source: str | None) -> str:
        if source is None:
            return '[[package]]\nname = "alice-crate"\nversion = "0.6.0"\n'
        return f'[[package]]\nname = "alice-det-math"\nversion = "0.5.2"\nsource = "{source}"\n'

    def git_runner(self, fresh: str | None = PINNED, shipped: str | None = None, precise_error: str | None = None):
        """cargo as the git tests see it: the shipped-lock check writes `shipped`, a fresh resolution
        (`generate-lockfile`) writes `fresh` (the branch tip), and `update -p NAME --precise C` moves
        NAME's git source in the lock to commit C (or fails with `precise_error`)"""
        base = self.runner()
        state = {"fresh": False}

        def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
            if cmd[1] == "update" and precise_error is not None:
                self.calls.append(cmd)
                return 101, "", precise_error
            rc, out, err = base(cmd, cwd)
            if cmd[1] == "generate-lockfile":
                state["fresh"] = True
                (cwd / "Cargo.lock").write_text(self.lock(fresh))
            elif cmd[1] == "update" and "--precise" in cmd:
                commit = cmd[cmd.index("--precise") + 1]
                text = (cwd / "Cargo.lock").read_text(encoding="utf-8")
                (cwd / "Cargo.lock").write_text(re.sub(r"#[0-9a-f]{40}\"", f"#{commit}\"", text))
            elif cmd[1] == "check" and "--locked" in cmd and not state["fresh"] and shipped is not None:
                (cwd / "Cargo.lock").write_text(self.lock(shipped))
            elif cmd[1] == "check" and "--locked" not in cmd:
                (cwd / "Cargo.lock").write_text(self.lock(fresh))
            return rc, out, err

        return run

    def write_dep(self, spec: str) -> None:
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "alice-crate"\nversion = "0.6.0"\n\n[dependencies]\n'
            f'alice-det-math = {{ {spec} }}\n'
        )



class GitPinned(GitBase):
    """A dependency taken from git at a pinned rev while its version is not on crates.io"""

    def test_unpublished_git_dep_is_blocked_patched_and_its_lock_source_checked(self) -> None:
        rc, out = self.main(runner=self.git_runner(shipped=PINNED))
        self.assertEqual(rc, 0, out)
        self.assertIn(f"blocked: alice-det-math 0.5.2 unpublished (crates.io max 0.5.1), taken from git {GIT} at {REV[:12]}", out)
        self.assertIn("git pins: 1, compared 2 against the locks", out)
        cargo = [c for c in self.calls if c[0] == "cargo"]
        self.assertTrue(cargo)
        for c in cargo:
            self.assertIn(f'patch.crates-io.alice-det-math.git={json.dumps(GIT)}', c)
            self.assertIn(f'patch.crates-io.alice-det-math.rev={json.dumps(REV)}', c)

    def test_pinned_rev_not_used_fails(self) -> None:
        for fresh, why in [
            ("registry+https://github.com/rust-lang/crates.io-index", "not " + PINNED),
            (f"git+{GIT}?rev={'0' * 40}#{'0' * 40}", "not " + PINNED),
            (None, "has no alice-det-math (the git pin was not used)"),
        ]:
            rc, out = self.main(runner=self.git_runner(fresh=fresh))
            self.assertEqual(rc, 1, out)
            self.assertIn(why, out)

    def test_shipped_lock_is_checked_too(self) -> None:
        rc, out = self.main(runner=self.git_runner(shipped="registry+https://github.com/rust-lang/crates.io-index"))
        self.assertEqual(rc, 1, out)
        self.assertIn("the shipped Cargo.lock takes alice-det-math from", out)

    def test_a_rev_that_is_not_40_lowercase_hex_is_a_problem(self) -> None:
        for rev in (REV[:12], REV.upper()):
            self.write_dep(f'git = "{GIT}", rev = "{rev}", version = "0.5.2"')
            rc, out = self.main(runner=self.git_runner())
            self.assertEqual(rc, 1, out)
            self.assertIn("not 40 lowercase hex digits", out)

    def test_more_than_one_reference_is_a_problem(self) -> None:
        self.write_dep(f'git = "{GIT}", rev = "{REV}", branch = "main", version = "0.5.2"')
        rc, out = self.main(runner=self.git_runner())
        self.assertEqual(rc, 1, out)
        self.assertIn("names more than one of rev / branch / tag (branch, rev)", out)

    def test_a_branch_or_default_branch_dep_without_a_lock_entry_is_a_problem(self) -> None:
        for spec in (f'git = "{GIT}", branch = "main", version = "0.5.2"', f'git = "{GIT}", version = "0.5.2"'):
            self.write_dep(spec)
            rc, out = self.main(runner=self.git_runner())
            self.assertEqual(rc, 1, out)
            self.assertIn("there is no Cargo.lock to read the commit it resolved to", out)


TESTED = "1" * 40
MOVED = "2" * 40


class GitBranch(GitBase):
    """A dependency taken from git by branch / tag / the default branch while its version is not on
    crates.io: patched with the same reference, checked at the commit this repository's Cargo.lock
    resolved it to. oracle: the expected sources are written out here from the cargo source format
    (`git+URL?branch=B#C`), not built by package_integrity"""

    def use(self, ref: str) -> str:
        """write the manifest and this repository's Cargo.lock for reference `ref` ("branch=main",
        "tag=v1", or "" for the default branch); the source cargo writes for it at TESTED"""
        spec = f'git = "{GIT}", version = "0.5.2"' + (
            ", " + ref.replace("=", ' = "') + '"' if ref else "")
        self.write_dep(spec)
        source = f"git+{GIT}" + (f"?{ref}" if ref else "") + f"#{TESTED}"
        (self.root / "Cargo.lock").write_text(self.lock(source))
        return source

    def test_a_branch_tag_or_default_branch_dep_is_patched_with_its_own_reference(self) -> None:
        for ref, arg in (("branch=main", ("branch", "main")), ("tag=v1", ("tag", "v1")), ("", None)):
            source = self.use(ref)
            self.calls.clear()
            rc, out = self.main(runner=self.git_runner(fresh=source, shipped=source))
            self.assertEqual(rc, 0, f"{ref}: {out}")
            self.assertIn("git pins: 1, compared 2 against the locks", out)
            cargo = [c for c in self.calls if c[0] == "cargo"]
            self.assertTrue(cargo)
            for c in cargo:
                self.assertIn(f'patch.crates-io.alice-det-math.git={json.dumps(GIT)}', c)
                self.assertFalse(any("patch.crates-io.alice-det-math.rev" in a for a in c), c)
                if arg:
                    self.assertIn(f'patch.crates-io.alice-det-math.{arg[0]}={json.dumps(arg[1])}', c)
                else:
                    self.assertFalse(any(".branch=" in a or ".tag=" in a for a in c), c)

    def test_a_branch_whose_tip_moved_since_the_tested_lock_is_checked_at_the_locked_commit(self) -> None:
        """the fresh resolution takes the tip (MOVED); the check pins it back to the commit this
        repository's Cargo.lock resolved (TESTED) with `cargo update --precise` and passes: a push
        upstream does not turn this repository's check red (the "lock behind main" signal is the
        scheduled report's, not this gate's)"""
        source = self.use("branch=main")
        moved = source.replace(TESTED, MOVED)
        rc, out = self.main(runner=self.git_runner(fresh=moved, shipped=source))
        self.assertEqual(rc, 0, out)
        self.assertIn("git pins: 1, compared 2 against the locks", out)
        cargo = [c for c in self.calls if c[0] == "cargo"]
        verbs = [c[1] for c in cargo]
        gen = verbs.index("generate-lockfile")
        upd = [i for i, c in enumerate(cargo) if c[1] == "update"]
        self.assertEqual(len(upd), 1, cargo)
        self.assertEqual(cargo[upd[0]][2:6], ["-p", "alice-det-math", "--precise", TESTED])
        last_check = max(i for i, v in enumerate(verbs) if v == "check")
        self.assertLess(gen, upd[0])
        self.assertLess(upd[0], last_check)
        self.assertIn("--locked", cargo[last_check], "the fresh check must not move the pinned lock again")
        self.assertIn(f'patch.crates-io.alice-det-math.branch={json.dumps("main")}', cargo[upd[0]])

    def test_a_locked_commit_that_cannot_be_checked_out_fails(self) -> None:
        """the locked commit is gone from the remote (force-pushed away): red, not a fallback to the tip"""
        source = self.use("branch=main")
        rc, out = self.main(runner=self.git_runner(
            fresh=source.replace(TESTED, MOVED), shipped=source,
            precise_error="error: revspec '1111' not found"))
        self.assertEqual(rc, 1, out)
        self.assertIn(f"cannot take alice-det-math at the locked commit {TESTED}", out)
        self.assertIn("revspec", out)

    def test_a_rev_dep_needs_no_precise_and_a_crates_io_dep_stays_fresh(self) -> None:
        """only a git dependency by branch / tag / the default branch is pinned back; a rev is exact
        already, and a crates.io dependency keeps the fresh resolution (no `cargo update` names it)"""
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "alice-crate"\nversion = "0.6.0"\n\n[dependencies]\n'
            f'alice-det-math = {{ git = "{GIT}", rev = "{REV}", version = "0.5.2" }}\n'
            'serde = "1"\n'
        )
        rc, out = self.main(runner=self.git_runner(shipped=PINNED))
        self.assertEqual(rc, 0, out)
        self.assertEqual([c for c in self.calls if c[:2] == ["cargo", "update"]], [], self.calls)
        source = self.use("branch=main")
        with open(self.root / "Cargo.toml", "a", encoding="utf-8") as f:
            f.write('serde = "1"\n')
        self.calls.clear()
        rc, out = self.main(runner=self.git_runner(fresh=source.replace(TESTED, MOVED), shipped=source))
        self.assertEqual(rc, 0, out)
        names = [c[3] for c in self.calls if c[:2] == ["cargo", "update"]]
        self.assertEqual(names, ["alice-det-math"], self.calls)

    def test_a_second_source_of_the_package_fails(self) -> None:
        """a transitive git dependency taking the same package by `?rev` next to this crate's
        `?branch`: two copies linked"""
        source = self.use("branch=main")
        second = f"git+{GIT}?rev={TESTED}#{TESTED}"
        two = self.lock(source) + f'\n[[package]]\nname = "alice-det-math"\nversion = "0.5.2"\nsource = "{second}"\n'
        base = self.runner()

        def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
            rc, o, e = base(cmd, cwd)
            if cmd[1] == "check":
                (cwd / "Cargo.lock").write_text(two)
            return rc, o, e

        rc, out = self.main(runner=run)
        self.assertEqual(rc, 1, out)
        self.assertIn(second, out)

    def test_a_transitive_dependency_by_the_same_branch_passes(self) -> None:
        """the crates reached through git name the package by the same `?branch=main#C`: one source"""
        source = self.use("branch=main")
        same = self.lock(source) + f'\n[[package]]\nname = "alice-zip"\nversion = "0.11.0"\nsource = "git+https://github.com/ext-sakamoro/ALICE-Zip?branch=main#{MOVED}"\ndependencies = ["alice-det-math"]\n'
        base = self.runner()

        def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
            rc, o, e = base(cmd, cwd)
            if cmd[1] == "check":
                (cwd / "Cargo.lock").write_text(same)
            return rc, o, e

        rc, out = self.main(runner=run)
        self.assertEqual(rc, 0, out)
        self.assertIn("compared 2 against the locks", out)

    def test_the_branch_commit_is_reported(self) -> None:
        source = self.use("branch=main")
        rc, out = self.main(runner=self.git_runner(fresh=source, shipped=source))
        self.assertEqual(rc, 0, out)
        self.assertIn(f"taken from git {GIT} branch main at {TESTED[:12]}", out)


class GitPinnedMore(GitBase):
    """more rev cases"""

    def test_git_dep_without_a_version_is_a_problem_but_a_dev_dependency_is_left_out(self) -> None:
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "alice-crate"\nversion = "0.6.0"\n\n[dependencies]\n'
            f'alice-det-math = {{ git = "{GIT}", rev = "{REV}" }}\n'
        )
        rc, out = self.main(runner=self.git_runner())
        self.assertEqual(rc, 1, out)
        self.assertIn("git dependency without a version", out)
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "alice-crate"\nversion = "0.6.0"\n\n[dev-dependencies]\n'
            f'alice-det-math = {{ git = "{GIT}", rev = "{REV}" }}\n'
        )
        rc, out = self.main(runner=self.git_runner())
        self.assertEqual(rc, 0, out)
        self.assertIn("blocked 0", out)

    def test_a_published_version_that_satisfies_is_not_patched(self) -> None:
        self.published["alice-det-math"] = [(0, 5, 2)]
        rc, out = self.main(runner=self.git_runner(fresh="registry+https://github.com/rust-lang/crates.io-index"))
        self.assertEqual(rc, 0, out)
        self.assertIn("blocked 0", out)
        self.assertFalse(any("patch.crates-io" in a for c in self.calls for a in c), self.calls)

    def test_renamed_git_dependency_is_patched_under_its_package_name(self) -> None:
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "alice-crate"\nversion = "0.6.0"\n\n[dependencies]\n'
            f'dm = {{ package = "alice-det-math", git = "{GIT}", rev = "{REV}", version = "0.5.2" }}\n'
        )
        rc, out = self.main(runner=self.git_runner())
        self.assertEqual(rc, 0, out)
        self.assertTrue(any(f'patch.crates-io.alice-det-math.rev={json.dumps(REV)}' in c for c in self.calls))

    def test_publish_gate_is_red_for_a_git_pin(self) -> None:
        rc, out = self.main("--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("publish gate: FAIL", out)

    def test_patches_prints_the_git_and_rev_pair(self) -> None:
        buf, err = io.StringIO(), io.StringIO()
        from contextlib import redirect_stderr
        with redirect_stdout(buf), redirect_stderr(err):
            rc = pi.main(["--root", str(self.root), "--patches"], versions_of=self.versions_of, runner=self.git_runner())
        self.assertEqual(rc, 0, err.getvalue())
        self.assertEqual(buf.getvalue().splitlines(), [
            f'--config=patch.crates-io.alice-det-math.git={json.dumps(GIT)}',
            f'--config=patch.crates-io.alice-det-math.rev={json.dumps(REV)}',
        ])


class TestResultParser(unittest.TestCase):
    """`passed_tests` reads libtest's summary lines; CI colours them (`CARGO_TERM_COLOR=always`,
    `--color always`), so the coloured line must read the same as the plain one. oracle: the counts
    are written out from the sample lines"""

    def test_plain_and_coloured_summaries_read_the_same(self) -> None:
        plain = "test result: ok. 3 passed; 0 failed; 0 ignored\ntest result: FAILED. 1 passed; 2 failed; 0 ignored\n"
        coloured = ("test result: \x1b[32mok\x1b[0m. 3 passed; 0 failed; 0 ignored\n"
                    "test result: \x1b[31mFAILED\x1b[0m. 1 passed; 2 failed; 0 ignored\n")
        self.assertEqual(pi.passed_tests(plain), (4, 2))
        self.assertEqual(pi.passed_tests(coloured), (4, 2))

    def test_no_summary_line_is_zero_tests(self) -> None:
        self.assertEqual(pi.passed_tests("\x1b[1m\x1b[32m   Compiling\x1b[0m alice-crate v0.6.0\n"), (0, 0))


class RegistryIndex(unittest.TestCase):
    """The sparse index read: its path (a known-answer table) and how an answer is read"""

    def test_index_path_known_answers(self) -> None:
        # https://doc.rust-lang.org/cargo/reference/registry-index.html (index files)
        for name, path in [("a", "1/a"), ("ab", "2/ab"), ("abc", "3/a/abc"), ("abcd", "ab/cd/abcd"),
                           ("alice-sign", "al/ic/alice-sign"), ("Serde", "se/rd/serde"),
                           ("Abc", "3/a/abc")]:
            self.assertEqual(pi.index_path(name), path, name)

    @staticmethod
    def http_error(code: int):
        def fetch(url: str) -> bytes:
            raise urllib.error.HTTPError(url, code, "status", None, None)
        return fetch

    def test_a_crate_never_published_has_no_version(self) -> None:
        asked: list[str] = []

        def fetch(url: str) -> bytes:
            asked.append(url)
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

        self.assertEqual(pi.registry_versions("alice-sign", fetch=fetch), [])
        self.assertEqual(asked, ["https://index.crates.io/al/ic/alice-sign"])

    def test_every_other_failure_stays_unreadable(self) -> None:
        def timeout(url: str) -> bytes:
            raise TimeoutError("timed out")

        def dns(url: str) -> bytes:
            raise urllib.error.URLError("nodename nor servname provided")

        for fetch in [self.http_error(403), self.http_error(429), self.http_error(500),
                      self.http_error(503), timeout, dns]:
            with self.assertRaises(OSError):
                pi.registry_versions("alice-sign", fetch=fetch)

    def test_a_malformed_body_is_unreadable(self) -> None:
        for body in [b"not json\n", b'{"vers": "1.0.0"\n', b'{"name": "x"}\n', b"\xff\xfe\n"]:
            with self.assertRaises(OSError, msg=body):
                pi.registry_versions("alice-sign", fetch=lambda url, b=body: b)

    def test_published_versions_are_read_without_yanked_and_prereleases(self) -> None:
        body = b"\n".join([
            b'{"name": "alice-sign", "vers": "0.1.0", "yanked": false}',
            b'{"name": "alice-sign", "vers": "0.2.0", "yanked": true}',
            b'{"name": "alice-sign", "vers": "0.3.0-rc.1", "yanked": false}',
            b'{"name": "alice-sign", "vers": "0.1.1", "yanked": false}',
        ])
        self.assertEqual(pi.registry_versions("alice-sign", fetch=lambda url: body), [(0, 1, 0), (0, 1, 1)])


class NeverPublishedSibling(Fixture):
    """A sibling the index has never heard of (404) is blocked and patched, as one whose
    published versions are all too old; any other index failure is still red"""

    def run_with(self, fetch) -> tuple[int, str]:
        buf = io.StringIO()
        os.environ["CARGO_TARGET_DIR"] = str(self.target)
        with redirect_stdout(buf):
            rc = pi.main(["--root", str(self.root), "--expect-in-lock", "alice-det-math"],
                         versions_of=lambda n: pi.registry_versions(n, fetch=fetch), runner=self.runner())
        return rc, buf.getvalue()

    def test_404_is_blocked_patched_and_green(self) -> None:
        rc, out = self.run_with(RegistryIndex.http_error(404))
        self.assertEqual(rc, 0, out)
        self.assertIn("blocked: alice-zip >=0.11, <0.12 unpublished (crates.io max none)", out)
        self.assertTrue(all(any("patch.crates-io.alice-zip.path=" in a for a in c) for c in self.calls), self.calls)

    def test_500_is_unreadable_and_red(self) -> None:
        rc, out = self.run_with(RegistryIndex.http_error(500))
        self.assertEqual(rc, 1, out)
        self.assertIn("index unreadable", out)
        self.assertEqual(self.calls, [])

    def test_a_published_version_that_satisfies_resolves_without_a_patch(self) -> None:
        body = b'{"name": "alice-zip", "vers": "0.11.2", "yanked": false}'
        rc, out = self.run_with(lambda url: body)
        self.assertEqual(rc, 0, out)
        self.assertIn("blocked 0", out)
        self.assertFalse(any("patch.crates-io" in a for c in self.calls for a in c), self.calls)


class PackageContents(Fixture):
    def test_required_files_and_prefixes(self) -> None:
        rc, out = self.main("--require-file", "README.md", "--forbid-prefix", ".github/")
        self.assertEqual(rc, 0, out)
        self.assertIn("1 required present, 1 forbidden prefixes absent", out)
        rc, out = self.main("--require-file", "README_JP.md")
        self.assertEqual(rc, 1, out)
        self.assertIn("the package lacks README_JP.md", out)
        rc, out = self.main("--forbid-prefix", "scripts/", runner=self.runner(files="Cargo.toml\nscripts/x.py\n"))
        self.assertEqual(rc, 1, out)
        self.assertIn("forbidden prefix scripts/", out)

    def test_empty_package_list_fails(self) -> None:
        rc, out = self.main("--require-file", "README.md", runner=self.runner(files=""))
        self.assertEqual(rc, 1, out)
        self.assertIn("listed no file", out)

    def test_stderr_lines_are_not_packaged_files(self) -> None:
        # a cargo warning on stderr must not count as a file nor hide an empty list
        warn = "warning: manifest has no documentation\nnote: see more\n"
        rc, out = self.main("--forbid-prefix", ".github/", runner=self.runner(files="", list_stderr=warn))
        self.assertEqual(rc, 1, out)
        self.assertIn("listed no file", out)
        rc, out = self.main("--forbid-prefix", "warning", runner=self.runner(list_stderr=warn))
        self.assertEqual(rc, 0, out)
        self.assertIn("package lists 3 files", out)

    def test_doctests_minimum(self) -> None:
        rc, out = self.main("--doctests-min", "1")
        self.assertEqual(rc, 0, out)
        self.assertIn("unpacked doctests: 2 passed", out)
        rc, out = self.main("--doctests-min", "3")
        self.assertEqual(rc, 1, out)
        self.assertIn("ran 2 doctests, expected at least 3", out)
        rc, out = self.main("--doctests-min", "1", runner=self.runner(doc_out="test result: FAILED. 1 passed; 1 failed;"))
        self.assertEqual(rc, 1, out)

    def test_tests_in_package_must_run_and_pass(self) -> None:
        spec = "--features open-file-audit --test residual_arithmetic"
        rc, out = self.main(f"--test-in-package={spec}")
        self.assertEqual(rc, 0, out)
        self.assertIn("3 passed", out)
        self.assertTrue(any(c[:3] == ["cargo", "test", "--locked"] and "residual_arithmetic" in c for c in self.calls))
        rc, out = self.main(f"--test-in-package={spec}", runner=self.runner(test_out="test result: ok. 0 passed; 0 failed;"))
        self.assertEqual(rc, 1, out)
        self.assertIn("ran no test", out)
        rc, out = self.main(f"--test-in-package={spec}", runner=self.runner(test_out="test result: FAILED. 2 passed; 1 failed;"))
        self.assertEqual(rc, 1, out)

    def test_dev_dependency_on_the_crate_itself_is_not_a_sibling(self) -> None:
        # ALICE-DB: `alice-db = { path = "." }` in [dev-dependencies] (its tests enable a feature);
        # the root directory is named like a sibling, but it is the crate being packaged
        manifest = (self.root / "Cargo.toml").read_text()
        (self.root / "Cargo.toml").write_text(manifest + '\n[dev-dependencies]\nalice-crate = { path = ".", features = ["x"] }\n')
        rc, out = self.main("--publish-gate")
        self.assertIn("package_integrity: blocked 1", out)
        self.assertNotIn("alice-crate", out)
        # with a version too: still the crate itself, never a dependency to patch onto itself
        (self.root / "Cargo.toml").write_text(manifest + '\n[dev-dependencies]\nalice-crate = { path = ".", version = "0.6.0" }\n')
        self.published["alice-crate"] = []
        rc, out = self.main("--publish-gate")
        self.assertIn("package_integrity: blocked 1", out)
        self.assertNotIn("alice-crate", out)

    def test_every_cargo_call_carries_the_sibling_patch(self) -> None:
        rc, out = self.main("--require-file", "README.md", "--doctests-min", "1", "--test-in-package=--lib",
                            "--check-args=--lib --all-features", "--locked-check-args=--all-features --all-targets")
        self.assertEqual(rc, 0, out)
        self.assertGreaterEqual(len(self.calls), 6)
        self.assertTrue(all(any("patch.crates-io.alice-zip.path=" in a for a in c) for c in self.calls), self.calls)
        checks = [c for c in self.calls if c[1] == "check"]
        self.assertEqual(checks[-1][2:5], ["--locked", "--lib", "--all-features"], "the fresh check runs on the generated lock")
        self.assertIn("generate-lockfile", [c[1] for c in self.calls])
        self.assertIn(["--locked", "--all-features", "--all-targets"], [c[2:5] for c in self.calls if c[1] == "check" and "--locked" in c])


class Workspace(unittest.TestCase):
    """--member: a workspace whose packaged crate sits in a member directory (ALICE-LOL)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "ALICE-Ws"
        (self.root / ".github").mkdir(parents=True)
        (self.root / ".github" / "alice-siblings").write_text(f"ALICE-Zip {SHA}\n")
        (self.root / "Cargo.toml").write_text('[workspace]\nmembers = ["ws-macro", "ws-lib"]\n'
                                              '[workspace.dependencies]\nws-macro = { path = "ws-macro", version = "=0.4.0" }\n')
        (self.root / "ws-macro").mkdir()
        (self.root / "ws-macro" / "Cargo.toml").write_text('[package]\nname = "ws-macro"\nversion = "0.4.0"\n')
        self.lib = self.root / "ws-lib"
        self.lib.mkdir()
        self.write_lib('ws-macro = { path = "../ws-macro", version = "=0.4.0" }\n'
                       'alice-zip = { path = "../../ALICE-Zip", version = ">=0.11, <0.12" }\n')
        (self.base / "ALICE-Zip").mkdir()
        (self.base / "ALICE-Zip" / "Cargo.toml").write_text('[package]\nname = "alice-zip"\nversion = "0.11.0"\n')
        self.published = {"alice-zip": [(0, 10, 0)], "ws-macro": [(0, 3, 0)]}
        self.target = self.base / "target"
        self.calls: list[tuple[list[str], Path]] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_lib(self, deps: str, extra: str = "") -> None:
        (self.lib / "Cargo.toml").write_text(
            f'[package]\nname = "ws-lib"\nversion = "0.5.0"\n\n[dependencies]\n{deps}{extra}')

    def runner(self, cmd: list[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append((cmd, cwd))
        if cmd[1] == "package":
            out = self.target / "package"
            out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(out / "ws-lib-0.5.0.crate", "w:gz") as tar:
                data = b'[package]\nname = "ws-lib"\nversion = "0.5.0"\n'
                info = tarfile.TarInfo("ws-lib-0.5.0/Cargo.toml")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            return 0, "", "Packaged"
        return 0, "", "Finished"

    def main(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        os.environ["CARGO_TARGET_DIR"] = str(self.target)
        from contextlib import redirect_stderr
        with redirect_stdout(out), redirect_stderr(err):
            rc = pi.main(["--root", str(self.root), *argv], versions_of=lambda n: self.published[n], runner=self.runner)
        return rc, out.getvalue(), err.getvalue()

    def test_root_without_package_needs_member(self) -> None:
        rc, out, _ = self.main("--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("has no [package] (use --member for a workspace)", out)

    def test_member_sibling_two_levels_up_and_unpublished_member_dep_are_blocked(self) -> None:
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("blocked: ws-macro =0.4.0 unpublished (crates.io max 0.3.0), satisfied only by the "
                      "workspace member ws-macro (ws-macro 0.4.0)", out)
        self.assertIn("blocked: alice-zip >=0.11, <0.12 unpublished (crates.io max 0.10.0), satisfied only "
                      f"by .github/alice-siblings ALICE-Zip {SHA[:12]}", out)
        self.assertIn("package_integrity: blocked 2", out)

    def test_published_member_version_is_not_patched(self) -> None:
        self.published["ws-macro"] = [(0, 4, 0)]
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertIn("package_integrity: blocked 1", out)
        self.assertNotIn("blocked: ws-macro", out)

    def test_member_version_that_does_not_satisfy_is_a_problem(self) -> None:
        (self.root / "ws-macro" / "Cargo.toml").write_text('[package]\nname = "ws-macro"\nversion = "0.4.1"\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("ws-macro: the workspace member ws-macro is 0.4.1, which does not satisfy '=0.4.0'", out)

    def test_member_dep_without_version_is_a_problem_but_dev_dep_is_left_out(self) -> None:
        self.write_lib('ws-macro = { path = "../ws-macro" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("ws-macro: path dependency without a version cannot be published", out)
        self.write_lib("", '\n[dev-dependencies]\nws-macro = { path = "../ws-macro" }\nws-lib = { path = "." }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 0, out)
        self.assertIn("package_integrity: blocked 0", out)

    def test_unresolved_path_dependency_is_a_problem(self) -> None:
        for path in ("../../Other", "../../../ALICE-Zip", "../../ALICE-Zip/../ALICE-Other",
                     "../../ALICE-Zip/./sub", "../../ALICE-Zip//sub", "../../ALICE-Zip/sub/.."):
            self.write_lib(f'thing = {{ path = "{path}", version = "1" }}\n')
            rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
            self.assertEqual(rc, 1, (path, out))
            self.assertIn(f"thing: path '{path}' is neither a listed sibling (<root>/../<Repo>, or plain "
                          "components inside it) nor a package in this repository", out)

    def test_package_inside_a_sibling_is_that_sibling(self) -> None:
        # a repository whose root is one package and whose other package sits in crates/<name>
        (self.root / ".github" / "alice-siblings").write_text(f"ALICE-Zip {SHA}\nALICE-Physics {SHA}\n")
        pkg = self.base / "ALICE-Physics" / "crates" / "alice-physics"
        pkg.mkdir(parents=True)
        (pkg / "Cargo.toml").write_text('[package]\nname = "alice-physics"\nversion = "4.0.0"\n')
        for path in ("../../ALICE-Physics/crates/alice-physics", "../../ALICE-Physics/crates/alice-physics/"):
            self.write_lib(f'alice-physics = {{ path = "{path}", version = ">=3.0, <5" }}\n')
            self.published["alice-physics"] = [(3, 1, 0)]
            rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
            self.assertEqual(rc, 0, (path, out))
            self.assertIn("package_integrity: blocked 0", out)
            self.published["alice-physics"] = [(2, 0, 0)]
            rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
            self.assertEqual(rc, 1, out)
            self.assertIn("blocked: alice-physics >=3.0, <5 unpublished (crates.io max 2.0.0), satisfied only by "
                          f".github/alice-siblings ALICE-Physics {SHA[:12]}", out)
            rc, out, err = self.main("--member", "ws-lib", "--patches")
            self.assertEqual((rc, out.splitlines()), (0, [f'--config=patch.crates-io.alice-physics.path={json.dumps(str(pkg.resolve()))}']), err)
        # the package is read from that directory: no manifest there fails
        self.write_lib('alice-physics = { path = "../../ALICE-Physics/crates/missing", version = ">=3.0, <5" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("alice-physics: ../ALICE-Physics/crates/missing/Cargo.toml is missing", out)

    def physics_repo(self) -> Path:
        """ALICE-Physics: package alice-physics-core at the root, alice-physics in crates/."""
        (self.root / ".github" / "alice-siblings").write_text(f"ALICE-Zip {SHA}\nALICE-Physics {SHA}\n")
        repo = self.base / "ALICE-Physics"
        (repo / "crates" / "alice-physics").mkdir(parents=True)
        (repo / "Cargo.toml").write_text('[package]\nname = "alice-physics-core"\nversion = "4.0.0"\n')
        (repo / "crates" / "alice-physics" / "Cargo.toml").write_text('[package]\nname = "alice-physics"\nversion = "4.0.0"\n')
        self.published.update({"alice-physics": [(3, 1, 0)], "alice-physics-core": [(4, 0, 0)]})
        return repo

    def test_two_packages_of_one_sibling_are_both_checked(self) -> None:
        self.physics_repo()
        both = ('alice-physics-core = {{ path = "../../ALICE-Physics", version = "{}" }}\n'
                'alice-physics = {{ path = "../../ALICE-Physics/crates/alice-physics", version = ">=3.0, <5" }}\n')
        self.write_lib(both.format("4"))
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 0, out)
        self.assertIn("package_integrity: blocked 0", out)
        self.write_lib(both.format("5"))
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("alice-physics-core: the sibling ../ALICE-Physics is 4.0.0, which does not satisfy '5'", out)

    def test_one_name_in_two_places_of_one_sibling_is_checked_in_each(self) -> None:
        # the same dependency name in a target table, pointing at another directory of the same
        # repository: the second place is checked on its own, not taken as the first
        self.physics_repo()
        self.write_lib('alice-physics = { path = "../../ALICE-Physics/crates/alice-physics", version = ">=3.0, <5" }\n',
                       '\n[target.\'cfg(unix)\'.dependencies]\n'
                       'alice-physics = { path = "../../ALICE-Physics", version = ">=3.0, <5" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("alice-physics: ../ALICE-Physics is package 'alice-physics-core', not alice-physics", out)

    def test_two_names_for_one_directory_are_each_checked(self) -> None:
        # a second name for the same directory is its own claim: the package there must be it
        self.physics_repo()
        self.write_lib('alice-physics = { path = "../../ALICE-Physics/crates/alice-physics", version = ">=3.0, <5" }\n'
                       'alice-physics-core = { path = "../../ALICE-Physics/crates/alice-physics", version = "4" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("alice-physics-core: ../ALICE-Physics/crates/alice-physics is package 'alice-physics', "
                      "not alice-physics-core", out)

    def test_one_name_at_one_subdir_of_two_siblings_is_checked_in_each(self) -> None:
        # the same name and the same place inside two different repositories: the repository is
        # part of where the claim points, so the second is checked on its own
        self.physics_repo()
        self.write_lib('alice-physics-core = { path = "../../ALICE-Physics", version = "4" }\n',
                       '\n[target.\'cfg(unix)\'.dependencies]\n'
                       'alice-physics-core = { path = "../../ALICE-Zip", version = "4" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("alice-physics-core: ../ALICE-Zip is package 'alice-zip', not alice-physics-core", out)

    def other_member(self) -> None:
        (self.root / "ws-other").mkdir()
        (self.root / "ws-other" / "Cargo.toml").write_text('[package]\nname = "ws-other"\nversion = "0.4.0"\n')

    def test_one_name_at_two_member_dirs_is_checked_in_each(self) -> None:
        self.other_member()
        self.write_lib('ws-macro = { path = "../ws-macro", version = "=0.4.0" }\n',
                       '\n[target.\'cfg(unix)\'.dependencies]\n'
                       'ws-macro = { path = "../ws-other", version = "=0.4.0" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("ws-macro: ws-other is package 'ws-other', not ws-macro", out)

    def test_two_names_for_one_member_dir_are_each_checked(self) -> None:
        self.write_lib('ws-macro = { path = "../ws-macro", version = "=0.4.0" }\n'
                       'ws-other = { path = "../ws-macro", version = "=0.4.0" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("ws-other: ws-macro is package 'ws-macro', not ws-other", out)

    def test_same_crate_in_two_tables_is_checked_once(self) -> None:
        # one name at one place in [dependencies] and [build-dependencies]: one entry, one patch
        self.write_lib('alice-zip = { path = "../../ALICE-Zip", version = ">=0.11, <0.12" }\n',
                       '\n[build-dependencies]\nalice-zip = { path = "../../ALICE-Zip", version = ">=0.11, <0.12" }\n')
        rc, out, err = self.main("--member", "ws-lib", "--patches")
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.splitlines(),
                         [f'--config=patch.crates-io.alice-zip.path={json.dumps(str((self.base / "ALICE-Zip").resolve()))}'])
        self.assertIn("blocked 1", err)

    def test_dependency_on_the_workspace_root_package_is_a_member(self) -> None:
        (self.root / "Cargo.toml").write_text('[package]\nname = "ws-root"\nversion = "0.4.0"\n\n'
                                              '[workspace]\nmembers = ["ws-macro", "ws-lib"]\n')
        self.published["ws-root"] = []
        self.write_lib('ws-root = { path = "..", version = "=0.4.0" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertIn("satisfied only by the workspace member . (ws-root 0.4.0)", out)

    def test_renamed_dependency_is_checked_under_its_package_name(self) -> None:
        self.write_lib('zip = { package = "alice-zip", path = "../../ALICE-Zip", version = ">=0.11, <0.12" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertIn("blocked: alice-zip >=0.11, <0.12 unpublished", out)
        self.assertNotIn("not zip", out)

    def test_inherited_sibling_path_is_read_as_written(self) -> None:
        # [workspace.dependencies] paths are relative to the workspace root, joined before use
        for path, sibling in (("../ALICE-Zip", True), ("../ALICE-Zip/../ALICE-Other", False)):
            with self.subTest(path=path):
                (self.root / "Cargo.toml").write_text(
                    '[workspace]\nmembers = ["ws-macro", "ws-lib"]\n[workspace.dependencies]\n'
                    f'alice-zip = {{ path = "{path}", version = ">=0.11, <0.12" }}\n')
                self.write_lib("alice-zip = { workspace = true }\n")
                rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
                self.assertEqual(rc, 1, out)
                if sibling:
                    self.assertIn("blocked: alice-zip >=0.11, <0.12 unpublished", out)
                else:
                    self.assertIn("alice-zip: path ", out)
                    self.assertIn("is neither a listed sibling", out)

    def test_unresolved_paths_are_reported_per_name_and_per_path(self) -> None:
        for deps, extra, name, path in (
            # one name at two paths: the second path is its own problem
            ('alice-zip = { path = "../../Other", version = "1" }\n',
             '\n[target.\'cfg(unix)\'.dependencies]\nalice-zip = { path = "../../Else", version = "1" }\n',
             "alice-zip", "../../Else"),
            # two names for one path: the second name is its own problem
            ('alice-zip = { path = "../../Other", version = "1" }\n'
             'alice-other = { path = "../../Other", version = "1" }\n', "", "alice-other", "../../Other"),
        ):
            with self.subTest(name=name, path=path):
                self.write_lib(deps, extra)
                rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
                self.assertEqual(rc, 1, out)
                self.assertIn(f"{name}: path {path!r} is neither a listed sibling", out)

    def test_version_less_path_in_dependencies_is_a_problem_in_every_kind(self) -> None:
        # only a dev-dependency is left out of the package; in [dependencies] it is published
        for spec, expect in (
            ('other = { path = "../../Other" }', "other: path '../../Other' is neither a listed sibling"),
            ('ws-macro = { path = "../ws-macro" }', "ws-macro: path dependency without a version"),
            ('alice-zip = { path = "../../ALICE-Zip" }', "alice-zip: path dependency without a version"),
        ):
            self.write_lib(spec + "\n")
            rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
            self.assertEqual(rc, 1, (spec, out))
            self.assertIn(expect, out)

    def test_version_less_sibling_dev_dependency_is_left_out(self) -> None:
        self.write_lib("", '\n[dev-dependencies]\nalice-zip = { path = "../../ALICE-Zip", features = ["x"] }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 0, out)
        self.assertIn("package_integrity: blocked 0", out)

    def test_same_name_with_another_path_in_a_target_table_is_checked(self) -> None:
        self.write_lib('alice-zip = { path = "../../ALICE-Zip", version = ">=0.11, <0.12" }\n',
                       '\n[target.\'cfg(unix)\'.dependencies]\nalice-zip = { path = "../../Other", version = "1" }\n')
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("alice-zip: path '../../Other' is neither a listed sibling", out)

    def test_inherited_workspace_dependency_is_followed(self) -> None:
        self.write_lib("ws-macro = { workspace = true }\n")
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertIn("blocked: ws-macro =0.4.0 unpublished", out)
        self.write_lib("nothing = { workspace = true }\n")
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("[workspace.dependencies] has no nothing", out)

    def test_patches_prints_one_config_per_blocked_dep_and_nothing_else(self) -> None:
        rc, out, err = self.main("--member", "ws-lib", "--patches")
        self.assertEqual(rc, 0, err)
        lines = out.splitlines()
        self.assertEqual(lines, [
            f'--config=patch.crates-io.ws-macro.path={json.dumps(str((self.root / "ws-macro").resolve()))}',
            f'--config=patch.crates-io.alice-zip.path={json.dumps(str((self.base / "ALICE-Zip").resolve()))}',
        ])
        self.assertIn("blocked 2", err)
        self.published.update({"ws-macro": [(0, 4, 0)], "alice-zip": [(0, 11, 0)]})
        rc, out, err = self.main("--member", "ws-lib", "--patches")
        self.assertEqual((rc, out), (0, ""), err)
        self.write_lib('thing = { path = "../../Other", version = "1" }\n')
        rc, out, err = self.main("--member", "ws-lib", "--patches")
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("neither a listed sibling", err)

    def test_member_is_packaged_in_its_directory_into_the_workspace_target(self) -> None:
        os.environ.pop("CARGO_TARGET_DIR", None)
        self.target = self.root / "target"
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out):
            rc = pi.main(["--root", str(self.root), "--member", "ws-lib"],
                         versions_of=lambda n: self.published[n], runner=self.runner)
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("packaged ws-lib-0.5.0.crate", out.getvalue())
        package_calls = [c for c in self.calls if c[0][1] == "package"]
        self.assertTrue(package_calls)
        self.assertTrue(all(Path(cwd).resolve() == self.lib.resolve() for _, cwd in package_calls), package_calls)
        self.assertTrue(all(sum("patch.crates-io." in a for a in c[0]) == 2 for c in self.calls), self.calls)

    def test_sibling_reached_through_a_symlink_is_still_a_sibling(self) -> None:
        real = self.base / "elsewhere" / "zip"
        real.parent.mkdir()
        (self.base / "ALICE-Zip").rename(real)
        (self.base / "ALICE-Zip").symlink_to(real, target_is_directory=True)
        rc, out, _ = self.main("--member", "ws-lib", "--publish-gate")
        self.assertIn("blocked: alice-zip", out)


if __name__ == "__main__":
    unittest.main()
