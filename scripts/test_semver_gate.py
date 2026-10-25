#!/usr/bin/env python3
"""Tests of scripts/semver_gate.py (python3 scripts/test_semver_gate.py)

No network: crates.io is replaced by a fixture. The recorded outputs are the lines
cargo-semver-checks 0.50 prints; SemverEndToEnd runs the real tool on two small crates
(it fails when cargo-semver-checks is not installed, so the fixtures cannot drift from
the tool unseen). FeatureUniverse runs the real `cargo metadata` on small crates. The cases of class Semver are those of ALICE-Law
scripts/test_release_train.py (main eb0779d) for the same functions.

Shared file: holds no repository-specific name; keep every copy byte-identical.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the module next to this file, first on the path (also under python -I, which leaves
# the script directory out); a plain import is what scripts/cargo_output_parser_check.py
# looks for where a repository has that checker
sys.path.insert(0, str(HERE))
import semver_gate as sg  # noqa: E402


def semver_output(step: str, total: int, failed: int, lints=(), from_v="1.0.0", to_v="1.1.0",
                  assume=False) -> str:
    """what cargo semver-checks 0.50 prints for one run (the lines the gate reads); with no
    failure it prints no "fail" count at all"""
    counts = f"{total - failed} pass, {failed} fail, 0 warn, 3 skip" if failed else f"{total} pass, 3 skip"
    out = [f"     Building x v{to_v} (current)",
           f"    Checking x v{from_v} -> v{to_v} ({'assume ' if assume else ''}{step} change)",
           f"     Checked [   0.200s] {total} checks: {counts}"]
    out += [f"\n--- failure {lint}: some description ---\n" for lint in lints]
    out.append(f"     Summary {'semver requires new major version: 1 major and 0 minor checks failed' if failed else 'no semver update required'}")
    return "\n".join(out)


def latest_fixture(crate: str, version: str | None):
    def fetch(path: str) -> dict:
        if version is None:
            raise urllib.error.HTTPError(path, 404, "Not Found", None, None)
        assert path == crate, path
        return {"crate": {"max_stable_version": version, "max_version": version}}
    return fetch


def problems(version: str, latest_v: str | None, run, equal_version: str = "refuse", manifest=None, entry=None,
             universe=()):
    return sg.semver_problems(Path("."), "Cargo.toml", "x", version, manifest or {}, entry, latest_fixture("x", latest_v),
                              run=run, equal_version=equal_version, universe=list(universe))


class Semver(unittest.TestCase):
    def test_feature_specs(self):
        self.assertEqual([l for l, *_ in sg.feature_specs({}, None, [])], ["default features", "no default features"])
        self.assertEqual([l for l, *_ in sg.feature_specs({"features": {"default": [], "x": []}}, None, ["x"])][-1],
                         "all features")
        # the all-features set is the universe, not the [features] keys
        self.assertEqual(sg.feature_specs({"features": {"x": []}}, None, ["depx", "x"])[-1][2], ["depx", "x"])
        # with no declared set the universe is required: no fallback to the [features] keys
        with self.assertRaises(ValueError):
            sg.feature_specs({"features": {"x": []}}, None)
        specs = sg.feature_specs({"features": {"x": []}}, {"consumer-features": [["x"], []],
                                                           "consumer-check-features": [["py"]]})
        self.assertEqual([(fs, cmd) for _, _, fs, cmd in specs][2:], [(["x"], "build"), ([], "build"), (["py"], "check")])

    def test_the_step_follows_cargo_compatibility(self):
        for old, new, step in (("3.0.1", "3.1.0", "minor"), ("3.0.1", "3.0.2", "patch"), ("3.0.1", "4.0.0", "major"),
                               ("0.5.0", "0.5.1", "minor"), ("0.5.1", "0.6.0", "major"), ("0.0.1", "0.0.2", "major"),
                               ("1.9.9", "1.10.0", "minor")):
            self.assertEqual(sg.version_step(old, new), step, (old, new))

    def test_a_clean_minor_run_counts_and_a_breaking_one_is_refused(self):
        self.assertEqual(sg.semver_verdict(0, semver_output("minor", 196, 0), "minor", "l"), (None, 1))
        p, n = sg.semver_verdict(1, semver_output("minor", 196, 1, ["enum_discriminants_undefined_non_unit_variant"]),
                                 "minor", "alice-physics 3.1.0 (default features)")
        self.assertEqual(n, 0)
        self.assertIn("a minor version does not allow enum_discriminants_undefined_non_unit_variant", p)
        self.assertIn("semver requires new major version", p)

    def test_zero_checks_a_wrong_step_a_silent_failure_and_no_run_are_refused(self):
        self.assertIn("compared 0 checks", sg.semver_verdict(0, semver_output("minor", 0, 0), "minor", "l")[0])
        self.assertIn("compared a patch step, the versions make a minor step",
                      sg.semver_verdict(0, semver_output("patch", 10, 0), "minor", "l")[0])
        # a nonzero exit with no failed check counted is still refused
        self.assertIsNotNone(sg.semver_verdict(1, semver_output("minor", 10, 0), "minor", "l")[0])
        self.assertIn("did not run (exit 127)", sg.semver_verdict(127, "No such file: cargo", "minor", "l")[0])
        self.assertIn("printed no count", sg.semver_verdict(0, "    Checking x v1.0.0 -> v1.1.0 (minor change)", "minor", "l")[0])
        # colour codes do not hide the lines the gate reads
        coloured = semver_output("minor", 5, 0).replace("Checking", "\x1b[1m\x1b[32mChecking\x1b[0m")
        self.assertEqual(sg.semver_verdict(0, coloured, "minor", "l"), (None, 1))

    def test_the_real_all_pass_line_has_no_fail_field(self):
        # recorded from cargo-semver-checks 0.50.0 on a passing run
        out = ("    Checking alice-db v0.5.1 -> v0.5.1 (assume minor change)\n"
               "     Checked [   0.031s] 223 checks: 223 pass, 31 skip\n"
               "     Summary no semver update required\n")
        self.assertEqual(sg.semver_verdict(0, out, "minor", "l"), (None, 1))

    def test_a_major_step_is_not_compared(self):
        calls = []
        p, n, note = problems("2.0.0", "1.4.0", lambda *a: calls.append(a) or (0, ""))
        self.assertEqual((p, n, calls), ([], 0, []))
        self.assertIn("major step", note)

    def test_a_crate_with_no_release_has_nothing_to_compare(self):
        p, n, note = problems("0.1.0", None, lambda *a: self.fail("ran"))
        self.assertEqual((p, n), ([], 0))
        self.assertIn("no release", note)

    def test_a_version_not_newer_than_the_latest_is_refused(self):
        for latest_v, version in (("1.1.0", "1.1.0"), ("1.2.0", "1.1.0"), ("1.0.0", "1.0.0-rc.1"),
                                  ("1.0.0-rc.2", "1.0.0-rc.1"), ("1.0.0+build.2", "1.0.0+build.1")):
            p, _, _ = problems(version, latest_v, lambda *a: self.fail("ran"))
            self.assertEqual(len(p), 1, (latest_v, version))

    def test_a_pre_release_of_a_later_version_is_compared(self):
        p, n, note = problems("1.1.0-rc.1", "1.0.0", lambda *a: (0, semver_output("minor", 5, 0)))
        self.assertEqual((p, n), ([], 2))
        self.assertIn("(minor)", note)

    def test_precedence_follows_semver(self):
        order = ["0.9.9", "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
                 "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0", "1.0.1", "1.10.0"]
        keys = [sg.version_precedence(v) for v in order]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(sg.version_precedence("1.0.0+a"), sg.version_precedence("1.0.0"))

    def test_every_feature_set_of_the_consumer_build_is_compared(self):
        seen = []

        def run(repo, manifest_path, baseline, features, target_dir):
            seen.append((tuple(baseline), None if features is None else tuple(features)))
            return 0, semver_output("minor", 50, 0)
        entry = {"consumer-features": [["a", "std"]], "consumer-check-features": [["py"]]}
        p, n, note = problems("1.1.0", "1.0.0", run, manifest={"features": {"a": [], "py": []}}, entry=entry)
        self.assertEqual((p, n), ([], 4))
        base = ("--baseline-version", "1.0.0")
        self.assertEqual(seen, [(base, None), (base, ()), (base, ("a", "std")), (base, ("py",))])
        self.assertEqual(sg.semver_feature_args(None), ["--default-features"])
        self.assertEqual(sg.semver_feature_args([]), ["--only-explicit-features"])
        self.assertEqual(sg.semver_feature_args(["a", "b"]), ["--only-explicit-features", "--features", "a,b"])

    def test_one_refused_feature_set_refuses_the_release(self):
        runs = iter([(0, semver_output("minor", 50, 0)),
                     (1, semver_output("minor", 50, 1, ["enum_variant_added"]))])
        p, n, _ = problems("1.1.0", "1.0.0", lambda *a: next(runs))
        self.assertEqual((len(p), n), (1, 1), p)
        self.assertIn("(no default features)", p[0])

    def test_a_body_without_a_version_is_a_registry_error(self):
        with self.assertRaises(sg.RegistryError):
            sg.semver_problems(Path("."), "Cargo.toml", "x", "1.0.0", {}, None, lambda path: {"crate": {}},
                               run=lambda *a: self.fail("ran"))

    def test_an_http_error_other_than_404_is_raised(self):
        def fetch(path):
            raise urllib.error.HTTPError(path, 403, "Forbidden", None, None)
        with self.assertRaises(urllib.error.HTTPError):
            sg.semver_problems(Path("."), "Cargo.toml", "x", "1.0.0", {}, None, fetch, run=lambda *a: self.fail("ran"))


class EqualVersion(unittest.TestCase):
    """the opt-in "next-smallest-step" mode: the default stays the strict release-path rule"""

    MODE = "next-smallest-step"

    def test_the_default_refuses_an_equal_version(self):
        p, n, _ = problems("0.5.1", "0.5.1", lambda *a: self.fail("ran"))
        self.assertEqual((n, len(p)), (0, 1))
        self.assertIn("is not older than 0.5.1", p[0])

    def test_the_release_path_signature_keeps_the_strict_rule(self):
        # the call release_train.py makes (no equal_version argument) refuses an equal version
        p, n, _ = sg.semver_problems(Path("."), "Cargo.toml", "x", "0.5.1", {}, None, latest_fixture("x", "0.5.1"),
                                     run=lambda *a: self.fail("ran"))
        self.assertEqual((n, len(p)), (0, 1))

    def test_an_equal_version_is_checked_as_the_smallest_next_release(self):
        seen = []

        def run(repo, manifest_path, baseline, features, target_dir):
            seen.append(tuple(baseline))
            return 0, semver_output("minor", 223, 0, from_v="0.5.1", to_v="0.5.1", assume=True)
        p, n, note = problems("0.5.1", "0.5.1", run, self.MODE)
        self.assertEqual((p, n), ([], 2))
        # the step is passed to the tool, so it cannot infer "no change"
        self.assertEqual(set(seen), {("--baseline-version", "0.5.1", "--release-type", "minor")})
        self.assertIn("version equals the published 0.5.1: checking the next release as the smallest step, "
                      "0.5.2 (minor)", note)

    def test_the_smallest_step_follows_cargo_compatibility(self):
        for version, step in (("1.2.3", "patch"), ("0.5.1", "minor"), ("1.0.0", "patch")):
            seen = []

            def run(repo, manifest_path, baseline, features, target_dir, step=step):
                seen.append(baseline[-1])
                return 0, semver_output(step, 9, 0, assume=True)
            p, n, _ = problems(version, version, run, self.MODE)
            self.assertEqual((p, n, set(seen)), ([], 2, {step}), version)

    def test_an_equal_0_0_z_version_makes_a_major_step_and_is_not_compared(self):
        p, n, note = problems("0.0.3", "0.0.3", lambda *a: self.fail("ran"), self.MODE)
        self.assertEqual((p, n), ([], 0))
        self.assertIn("0.0.4 (major)", note)

    def test_an_equal_version_with_a_break_is_refused(self):
        out = semver_output("minor", 223, 1, ["function_missing"], "0.5.1", "0.5.1", assume=True)
        p, n, _ = problems("0.5.1", "0.5.1", lambda *a: (1, out), self.MODE)
        self.assertEqual((n, len(p)), (0, 2))
        self.assertIn("a minor version does not allow function_missing", p[0])

    def test_an_equal_version_whose_run_compared_nothing_is_refused(self):
        out = semver_output("minor", 0, 0, from_v="0.5.1", to_v="0.5.1", assume=True)
        p, n, _ = problems("0.5.1", "0.5.1", lambda *a: (0, out), self.MODE)
        self.assertEqual((n, len(p)), (0, 2))
        self.assertIn("compared 0 checks", p[0])

    def test_an_equal_version_where_the_tool_names_another_step_is_refused(self):
        out = semver_output("patch", 10, 0, from_v="0.5.1", to_v="0.5.1", assume=True)
        p, _, _ = problems("0.5.1", "0.5.1", lambda *a: (0, out), self.MODE)
        self.assertIn("compared a patch step, the versions make a minor step", p[0])

    def test_a_greater_version_takes_the_release_path(self):
        seen = []

        def run(repo, manifest_path, baseline, features, target_dir):
            seen.append(tuple(baseline))
            return 0, semver_output("minor", 50, 0, from_v="0.5.1", to_v="0.5.2")
        p, n, note = problems("0.5.2", "0.5.1", run, self.MODE)
        self.assertEqual((p, n), ([], 2))
        self.assertEqual(set(seen), {("--baseline-version", "0.5.1")})
        self.assertNotIn("equals", note)
        self.assertIn("0.5.1 -> 0.5.2 (minor)", note)

    def test_a_lower_version_is_refused(self):
        for latest_v, version in (("0.5.2", "0.5.1"), ("1.0.0", "1.0.0-rc.1"), ("1.0.0-rc.2", "1.0.0-rc.1")):
            p, n, _ = problems(version, latest_v, lambda *a: self.fail("ran"), self.MODE)
            self.assertEqual((n, len(p)), (0, 1), (latest_v, version))
            self.assertIn("is not older than", p[0])

    def test_an_equal_pre_release_is_refused(self):
        p, n, _ = problems("1.0.0-rc.1", "1.0.0-rc.1", lambda *a: self.fail("ran"), self.MODE)
        self.assertEqual((n, len(p)), (0, 1))
        self.assertIn("pre-release", p[0])

    def test_an_unknown_mode_is_an_error(self):
        with self.assertRaises(ValueError):
            problems("0.5.1", "0.5.1", lambda *a: self.fail("ran"), "skip")


class BumpBaseline(unittest.TestCase):
    """`baseline = "bump"`: under a publication freeze the base is the commit before main's newest version bump
    (scripts/release_baseline.py), compared with `cargo semver-checks --baseline-rev <that commit>`, the step judged
    from that commit's version; `auto` takes the registry release when it is not older than that version"""

    def setUp(self) -> None:
        self.repo = Path(tempfile.mkdtemp(prefix="semver-bump-test-"))
        self.addCleanup(shutil.rmtree, self.repo, True)
        self.env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("GIT_")}
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        import subprocess
        return subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", *args],
                              cwd=self.repo, check=True, capture_output=True, text=True, env=self.env).stdout.strip()

    def at(self, version: str) -> str:
        (self.repo / "Cargo.toml").write_text(f'[package]\nname = "x"\nversion = "{version}"\n', encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", version)
        return self.git("rev-parse", "HEAD")

    def check(self, version: str, run, latest_v: str | None = None, baseline: str = "bump", entry: dict | None = None):
        self.args = []

        def capture(repo, manifest_path, base, features, target_dir):
            self.args.append(list(base))
            return run(repo, manifest_path, base, features, target_dir)
        return sg.semver_problems(self.repo, "Cargo.toml", "x", version, {}, entry, latest_fixture("x", latest_v),
                                  run=capture, universe=[], baseline=baseline, main_ref="main")

    def named(self, name: str, version: str) -> str:
        (self.repo / "Cargo.toml").write_text(f'[package]\nname = "{name}"\nversion = "{version}"\n', encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", f"{name} {version}")
        return self.git("rev-parse", "HEAD")

    def test_a_head_version_not_newer_than_the_bump_base_is_refused(self):
        # SG6: the base is the parent of the newest bump; a HEAD version at or below that base's version is a
        # mistake (a version that went back), refused before anything runs
        self.at("1.0.0")
        self.at("1.1.0")
        for version in ("1.0.0", "0.9.0"):
            p, n, _ = self.check(version, lambda *a: self.fail("ran"))
            self.assertEqual(n, 0)
            self.assertEqual(len(p), 1, p)
            self.assertIn("is not older than", p[0])

    def test_auto_compares_with_the_bump_base_when_the_registry_is_older(self):
        # SG7: crates.io has 0.4.0 (frozen), main bumped 0.5.0 -> 0.5.1: the newer base, the 0.5.0 commit, is the
        # one compared, by rev, as a minor step (0.x: the second number is the compatibility line)
        base = self.at("0.5.0")
        self.at("0.5.1")
        p, n, note = self.check("0.5.1", lambda *a: (0, semver_output("minor", 9, 0)), latest_v="0.4.0", baseline="auto")
        self.assertEqual(p, [])
        self.assertTrue(n > 0)
        self.assertTrue(self.args and all(a == ["--baseline-rev", base, "--release-type", "minor"] for a in self.args),
                        self.args)
        self.assertIn("0.5.0 -> 0.5.1 (minor)", note)

    def test_a_renamed_crate_is_followed_through_base_package(self):
        # F3: the config's `base-package` (the crate's name before a rename) reaches the bump base
        before = self.named("old", "1.0.0")
        self.named("x", "1.1.0")
        p, n, note = self.check("1.1.0", lambda *a: (0, semver_output("minor", 9, 0)), entry={"base-package": "old"})
        self.assertEqual(p, [])
        self.assertTrue(self.args and all(a[:2] == ["--baseline-rev", before] for a in self.args), self.args)
        # without it the rename is an error, not "introduced"
        p, n, _ = self.check("1.1.0", lambda *a: self.fail("ran"))
        self.assertEqual(n, 0)
        self.assertEqual(len(p), 1, p)
        self.assertIn("base_package", p[0])

    def test_the_parent_of_the_bump_is_compared_by_rev(self):
        before = self.at("1.0.0")
        self.at("1.1.0")
        p, n, note = self.check("1.1.0", lambda *a: (0, semver_output("minor", 9, 0)))
        self.assertEqual((p, n), ([], len(sg.feature_specs({}, None, []))))
        self.assertTrue(self.args and all(a == ["--baseline-rev", before, "--release-type", "minor"] for a in self.args),
                        self.args)
        self.assertIn(f"bump base {before[:12]} (1.0.0)", note)
        self.assertIn("1.0.0 -> 1.1.0 (minor)", note)

    def test_a_break_under_a_minor_step_is_refused(self):
        self.at("1.0.0")
        self.at("1.1.0")
        p, n, _ = self.check("1.1.0", lambda *a: (1, semver_output("minor", 9, 1, lints=["enum_variant_added"])))
        self.assertEqual(len(p), len(sg.feature_specs({}, None, [])))
        self.assertIn("enum_variant_added", p[0])

    def test_a_major_step_from_the_base_is_not_compared(self):
        self.at("1.0.0")
        self.at("2.0.0")
        p, n, note = self.check("2.0.0", lambda *a: self.fail("ran"))
        self.assertEqual((p, n), ([], 0))
        self.assertIn("1.0.0 -> 2.0.0 is a major step", note)

    def test_a_run_that_checked_nothing_is_refused(self):
        self.at("1.0.0")
        self.at("1.1.0")
        p, _, _ = self.check("1.1.0", lambda *a: (0, semver_output("minor", 0, 0)))
        self.assertEqual(len(p), len(sg.feature_specs({}, None, [])))

    def test_an_introduced_crate_has_nothing_to_compare(self):
        self.at("0.1.0")
        p, n, note = self.check("0.1.0", lambda *a: self.fail("ran"))
        self.assertEqual((p, n), ([], 0))
        self.assertIn("introduced", note)

    def test_no_history_is_refused_not_passed(self):
        shutil.rmtree(self.repo / ".git")
        (self.repo / "Cargo.toml").write_text('[package]\nname = "x"\nversion = "1.0.0"\n', encoding="utf-8")
        p, n, _ = self.check("1.0.0", lambda *a: self.fail("ran"))
        self.assertEqual(n, 0)
        self.assertEqual(len(p), 1)
        self.assertIn("bump base", p[0])

    def test_auto_takes_the_registry_when_it_is_not_older_than_the_bump_base(self):
        # history 1.0.0 -> 1.1.0 -> 1.2.0: the bump base is the 1.1.0 commit; every case below is a minor step
        # whichever base is taken, so a run happens and shows which base was passed
        self.at("1.0.0")
        base = self.at("1.1.0")
        self.at("1.2.0")
        ok = lambda *a: (0, semver_output("minor", 9, 0, from_v="1.1.0", to_v="1.2.0"))
        self.check("1.2.0", ok, latest_v="1.1.0", baseline="auto")
        self.assertTrue(self.args and all(a[:2] == ["--baseline-version", "1.1.0"] for a in self.args), self.args)
        self.check("1.2.0", lambda *a: (0, semver_output("minor", 9, 0, from_v="1.0.0", to_v="1.2.0")),
                   latest_v="1.0.0", baseline="auto")
        self.assertTrue(self.args and all(a[:2] == ["--baseline-rev", base] for a in self.args), self.args)
        self.check("1.2.0", ok, latest_v=None, baseline="auto")
        self.assertTrue(self.args and all(a[:2] == ["--baseline-rev", base] for a in self.args), self.args)

    def test_the_cli_reads_the_baseline_and_asks_no_registry_for_bump(self):
        (self.repo / "src").mkdir()
        (self.repo / "src" / "lib.rs").write_text("", encoding="utf-8")
        self.at("1.0.0")
        self.at("1.1.0")
        (self.repo / "c.toml").write_text('baseline = "bump"\n', encoding="utf-8")

        def no_registry(path):
            raise AssertionError("the registry was asked")
        out = io.StringIO()
        import os
        cwd = os.getcwd()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                os.chdir(self.repo)
                code = sg.main(["--manifest", str(self.repo / "Cargo.toml"), "--config", str(self.repo / "c.toml")],
                               fetch=no_registry, run=lambda *a: (0, semver_output("minor", 9, 0)))
            finally:
                os.chdir(cwd)
        self.assertNotEqual(code, 0, out.getvalue())  # main_ref is origin/main, absent here: refused, not passed
        self.assertIn("baseline: bump", out.getvalue())
        self.assertIn("the bump base cannot be read", out.getvalue())

    def test_the_config_names_the_baseline(self):
        self.assertEqual(sg.load_config('baseline = "bump"\n').get("baseline"), "bump")
        with self.assertRaises(sg.ConfigError):
            sg.load_config('baseline = "tag"\n')

    def test_the_config_names_the_crate_before_a_rename(self):
        c = sg.load_config('baseline = "bump"\nbase-package = "alice-old"\n')
        self.assertEqual(c["base-package"], "alice-old")
        self.assertEqual(sg.load_config('baseline = "auto"\nbase-package = "o"\n')["base-package"], "o")
        for bad in ('baseline = "bump"\nbase-package = ""\n', 'base-package = "alice-old"\n'):
            with self.assertRaises(sg.ConfigError, msg=bad):
                sg.load_config(bad)


class Cli(unittest.TestCase):
    MANIFEST = '[package]\nname = "x"\nversion = "0.5.1"\n\n[features]\ndefault = ["a"]\na = []\npy = []\n'

    def cli(self, config: str | None, latest_v: str | None, run, manifest: str = MANIFEST) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d, "Cargo.toml")
            m.write_text(manifest, encoding="utf-8")
            Path(d, "src").mkdir()
            Path(d, "src", "lib.rs").write_text("", encoding="utf-8")
            argv = ["--manifest", str(m)]
            if config is not None:
                Path(d, "c.toml").write_text(config, encoding="utf-8")
                argv += ["--config", str(Path(d, "c.toml"))]
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = sg.main(argv, fetch=latest_fixture("x", latest_v), run=run)
        return code, out.getvalue()

    CONFIG = ('consumer-features = [["a"]]\nconsumer-features-why = "w"\nconsumer-check-features = [["py"]]\n'
              'equal-version = "next-smallest-step"\n')

    def test_the_mode_is_printed_and_an_additive_change_passes(self):
        code, out = self.cli(self.CONFIG, "0.5.1",
                             lambda *a: (0, semver_output("minor", 9, 0, from_v="0.5.1", to_v="0.5.1", assume=True)))
        self.assertEqual(code, 0, out)
        self.assertIn("equal-version mode: next-smallest-step", out)
        self.assertIn("version equals the published 0.5.1: checking the next release as the smallest step, 0.5.2", out)
        self.assertIn("4 feature set(s) compared", out)

    def test_an_equal_version_prints_the_version_checked(self):
        for version, upcoming, step in (("0.5.1", "0.5.2", "minor"), ("1.2.3", "1.2.4", "patch")):
            manifest = self.MANIFEST.replace('version = "0.5.1"', f'version = "{version}"')
            code, out = self.cli(self.CONFIG, version, lambda *a, step=step: (
                0, semver_output(step, 9, 0, from_v=version, to_v=version, assume=True)), manifest=manifest)
            self.assertEqual(code, 0, out)
            self.assertIn(f"semver gate: x {version} equals the latest release {version}: "
                          f"checking it as the next release {upcoming} ({step} step)\n", out)
            self.assertEqual(sg.smallest_next_release(version, version), (upcoming, step))
        # a newer version, a pre-release and the strict mode print no target
        self.assertIsNone(sg.smallest_next_release("0.5.1", "0.5.2"))
        self.assertIsNone(sg.smallest_next_release("1.0.0-rc.1", "1.0.0-rc.1"))
        code, out = self.cli(None, "0.5.1", lambda *a: self.fail("ran"))
        self.assertNotIn("checking it as the next release", out)

    def test_a_break_exits_1(self):
        code, out = self.cli(self.CONFIG, "0.5.1", lambda *a: (
            1, semver_output("minor", 9, 1, ["function_missing"], "0.5.1", "0.5.1", assume=True)))
        self.assertEqual(code, 1, out)
        self.assertIn("refused: x 0.5.1 (default features): a minor version does not allow function_missing", out)

    def test_the_default_mode_is_the_strict_rule(self):
        code, out = self.cli(None, "0.5.1", lambda *a: self.fail("ran"))
        self.assertEqual(code, 1, out)
        self.assertIn("equal-version mode: refuse", out)

    def test_no_release_passes_with_a_message(self):
        code, out = self.cli(self.CONFIG, None, lambda *a: self.fail("ran"))
        self.assertEqual(code, 0, out)
        self.assertIn("x has no release on crates.io: nothing to compare", out)

    def test_the_tool_missing_exits_1(self):
        code, out = self.cli(self.CONFIG, "0.5.1", lambda *a: (127, "[Errno 2] No such file or directory: 'cargo'"))
        self.assertEqual(code, 1, out)
        self.assertIn("did not run (exit 127)", out)

    def test_a_registry_failure_exits_2(self):
        def fetch(path):
            raise urllib.error.URLError("unreachable")
        with tempfile.TemporaryDirectory() as d:
            m = Path(d, "Cargo.toml")
            m.write_text(self.MANIFEST, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sg.main(["--manifest", str(m)], fetch=fetch, run=lambda *a: self.fail("ran")), 2)

    def test_a_bad_config_or_manifest_exits_2(self):
        never = lambda *a: self.fail("ran")  # noqa: E731
        for config in ('unknown = 1\n', 'consumer-features = [["a"]]\n', 'consumer-features-why = "w"\n',
                       'equal-version = "skip"\n', 'consumer-features = ["a"]\nconsumer-features-why = "w"\n',
                       'consumer-features = [["nope"]]\nconsumer-features-why = "w"\n'):
            self.assertEqual(self.cli(config, "0.5.1", never)[0], 2, config)
        self.assertEqual(self.cli(None, "0.5.1", never, manifest='[package]\nname = "x"\nversion.workspace = true\n')[0], 2)
        # a manifest cargo metadata cannot read exits 2 (no fallback to the [features] keys)
        self.assertEqual(self.cli(None, "0.5.1", never, manifest=self.MANIFEST + 'a = []\n')[0], 2)

    def test_the_universe_is_printed_and_a_set_may_name_an_implicit_feature(self):
        manifest = (self.MANIFEST.replace("[features]", '[dependencies]\nnum = { path = "num", optional = true }\n\n'
                                          "[features]"))
        with tempfile.TemporaryDirectory() as d:
            num = Path(d, "num")
            (num / "src").mkdir(parents=True)
            (num / "Cargo.toml").write_text('[package]\nname = "num"\nversion = "0.1.0"\nedition = "2021"\n',
                                            encoding="utf-8")
            (num / "src" / "lib.rs").write_text("", encoding="utf-8")
            config = 'consumer-features = [["num"], ["a", "py"]]\nconsumer-features-why = "w"\n'
            code, out = self.cli(config, "0.5.0",
                                 lambda *a: (0, semver_output("minor", 9, 0, from_v="0.5.0", to_v="0.5.1")),
                                 manifest=manifest.replace('path = "num"', f'path = "{num}"'))
        self.assertEqual(code, 0, out)
        self.assertIn("feature universe (cargo metadata, without default): a, num, py", out)


class Coverage(unittest.TestCase):
    """the CLI's feature coverage: with declared sets, a feature added to the manifest later
    (`zz`) is in no set; it must be refused, not listed and passed"""

    MANIFEST = Cli.MANIFEST + 'zz = []\n'
    CONFIG = 'consumer-features = [["a"]]\nconsumer-features-why = "w"\nconsumer-check-features = [["py"]]\n'
    PASS = staticmethod(lambda *a: (0, semver_output("minor", 9, 0, from_v="0.5.0", to_v="0.5.1")))

    def cli(self, config, manifest=MANIFEST, run=None):
        return Cli.cli(self, config, "0.5.0", run or self.PASS, manifest=manifest)

    # the zz case as a function, so the mutant below can be run through it
    def zz_case(self) -> None:
        code, out = self.cli(self.CONFIG, run=lambda *a: self.fail("ran"))
        self.assertEqual(code, 1, out)
        self.assertIn("refused: feature zz is in no consumer-features / consumer-check-features set "
                      "and not in exclude-features\n", out)
        self.assertIn("feature universe (cargo metadata, without default): a, py, zz", out)

    def test_an_uncovered_feature_of_a_declared_set_config_is_refused(self):
        self.zz_case()

    def test_each_uncovered_feature_is_named(self):
        code, out = self.cli('consumer-features = [["a"]]\nconsumer-features-why = "w"\n',
                             run=lambda *a: self.fail("ran"))
        self.assertEqual(code, 1, out)
        for f in ("py", "zz"):
            self.assertIn(f"refused: feature {f} is in no consumer-features", out)
        self.assertNotIn("refused: feature a ", out)

    def test_an_excluded_feature_with_a_reason_passes(self):
        code, out = self.cli(self.CONFIG + 'exclude-features = { zz = "needs a board" }\n')
        self.assertEqual(code, 0, out)
        self.assertIn("excluded feature: zz (needs a board)", out)
        self.assertIn("feature coverage: every feature of the universe is reached by a declared set or excluded "
                      "(3 compared)", out)
        self.assertIn("semver gate: pass", out)

    def test_a_feature_reached_through_the_closure_is_covered(self):
        code, out = self.cli(self.CONFIG.replace('[["a"]]', '[["b"]]'),
                             manifest=Cli.MANIFEST.replace("a = []", 'a = []\nb = ["a", "zz"]\nzz = []'))
        self.assertEqual(code, 0, out)
        self.assertIn("feature coverage: every feature", out)

    def test_bad_exclude_features_are_a_config_error(self):
        never = lambda *a: self.fail("ran")  # noqa: E731
        for ex in ('exclude-features = { zz = "" }\n', 'exclude-features = { zz = "  " }\n',
                   'exclude-features = ["zz"]\n', 'exclude-features = {}\n',
                   'exclude-features = { nope = "why" }\n', 'exclude-features = { default = "why" }\n',
                   'exclude-features = { a = "why" }\n'):
            code, out = self.cli(self.CONFIG + ex, run=never)
            self.assertEqual(code, 2, ex + out)
        # no declared set: nothing to exclude from
        code, out = self.cli('exclude-features = { zz = "why" }\n', run=never)
        self.assertEqual(code, 2, out)
        self.assertIn("`exclude-features` without `consumer-features`", out)
        code, out = self.cli(self.CONFIG + 'exclude-features = { nope = "why" }\n', run=never)
        self.assertIn("`exclude-features` names feature(s) ['nope'] that are not in the feature universe", out)

    def test_with_no_declared_set_the_check_is_not_applicable(self):
        code, out = self.cli(None)
        self.assertEqual(code, 0, out)
        self.assertIn("feature coverage: not applicable (no consumer set is declared: "
                      "the all-features set is the universe)", out)
        self.assertIn("feature set: all features ['a', 'py', 'zz']", out)

    def test_mutant_without_the_coverage_call_is_red(self):
        import unittest.mock as um
        with um.patch.object(sg, "coverage_refusals", lambda *a: []), self.assertRaises(AssertionError):
            self.zz_case()


class SemverEndToEnd(unittest.TestCase):
    """the real cargo semver-checks on two small crates: the baseline 1.0.0 has a
    #[non_exhaustive] unit-only enum and a public function"""

    BASE = ("#[derive(Clone, Copy, Debug)]\n#[non_exhaustive]\npub enum Check {\n    A,\n    B,\n}\n"
            "pub fn probe() -> u32 {\n    1\n}\n")

    def crate(self, root: Path, name: str, version: str, src: str) -> Path:
        d = root / name
        (d / "src").mkdir(parents=True)
        (d / "Cargo.toml").write_text(f'[package]\nname = "semverprobe"\nversion = "{version}"\nedition = "2021"\n'
                                      f'publish = false\n\n[workspace]\n', encoding="utf-8")
        (d / "src" / "lib.rs").write_text(src, encoding="utf-8")
        return d

    def gate(self, version: str, src: str, equal_version: str = "refuse") -> tuple[list[str], int, str]:
        if shutil.which("cargo-semver-checks") is None:
            self.fail("cargo-semver-checks is not installed (CI installs it; locally: cargo install cargo-semver-checks)")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = self.crate(root, "base", "1.0.0", self.BASE)
            new = self.crate(root, "new", version, src)
            return sg.semver_problems(
                new, "Cargo.toml", "semverprobe", version, {}, None, latest_fixture("semverprobe", "1.0.0"),
                run=lambda repo, mp, baseline, features, td: sg.run_semver_checks(
                    repo, mp, ["--baseline-root", str(base), *baseline[2:]], features, td),
                target_dir=root / "target", equal_version=equal_version,
                universe=sg.feature_universe(new / "Cargo.toml"))

    def test_a_field_variant_under_a_minor_version_is_refused(self):
        src = self.BASE.replace("    B,\n", "    B,\n    C { id: [u8; 32] },\n")
        p, n, _ = self.gate("1.1.0", src)
        self.assertEqual(n, 0)
        self.assertEqual(len(p), 2, p)  # default features and none
        self.assertIn("enum_discriminants_undefined_non_unit_variant", p[0])

    def test_a_unit_variant_under_a_minor_version_passes(self):
        p, n, note = self.gate("1.1.0", self.BASE.replace("    B,\n", "    B,\n    C,\n"))
        self.assertEqual((p, n), ([], 2), p)
        self.assertIn("1.0.0 -> 1.1.0 (minor)", note)

    def test_an_unchanged_api_under_a_patch_version_passes(self):
        self.assertEqual(self.gate("1.0.1", self.BASE)[:2], ([], 2))

    def test_the_same_change_under_a_major_version_is_not_compared(self):
        p, n, note = self.gate("2.0.0", self.BASE.replace("    B,\n", "    B,\n    C { id: [u8; 32] },\n"))
        self.assertEqual((p, n), ([], 0))
        self.assertIn("major step", note)

    def test_an_equal_version_with_an_addition_passes(self):
        p, n, note = self.gate("1.0.0", self.BASE + "pub fn added() -> u32 {\n    2\n}\n", "next-smallest-step")
        self.assertEqual((p, n), ([], 2), p)
        self.assertIn("smallest step, 1.0.1 (patch)", note)

    def test_an_equal_version_with_a_removed_function_is_refused(self):
        p, n, _ = self.gate("1.0.0", self.BASE.replace("pub fn probe", "fn probe"), "next-smallest-step")
        self.assertEqual(n, 0)
        self.assertEqual(len(p), 2, p)
        self.assertIn("a patch version does not allow function_missing", p[0])


def write_crate(root: Path, name: str, version: str = "1.0.0", deps: str = "", features: str = "",
                src: str = "") -> Path:
    d = root / name
    (d / "src").mkdir(parents=True)
    (d / "Cargo.toml").write_text(f'[package]\nname = "{name}"\nversion = "{version}"\nedition = "2021"\n'
                                  f'publish = false\n\n[workspace]\n\n[dependencies]\n{deps}\n[features]\n{features}',
                                  encoding="utf-8")
    (d / "src" / "lib.rs").write_text(src, encoding="utf-8")
    return d


class FeatureUniverse(unittest.TestCase):
    """the real `cargo metadata` on small crates with an optional path dependency"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_crate(self.root, "depx", deps="", features="")
        self.dep = f'depx = {{ path = "{self.root / "depx"}", optional = true }}\n'

    def tearDown(self):
        self.tmp.cleanup()

    def pkg(self, features: str, name: str = "pkg") -> Path:
        return write_crate(self.root, name, deps=self.dep, features=features) / "Cargo.toml"

    # the checks of each case, as functions of the helpers, so a mutant can be run through them
    def implicit_case(self, universe_of) -> None:
        m = self.pkg('py = []\n')
        universe = universe_of(m)
        self.assertIn("depx", universe)
        self.assertEqual(sg.feature_specs({"features": {"py": []}}, None, universe)[-1][2], ["depx", "py"])

    def closure_case(self, closure) -> None:
        feats = sg.features_from_metadata(sg.cargo_metadata(self.pkg('py = ["depx"]\n')), "pkg")
        self.assertEqual(closure(feats, {"py"}), {"py", "depx"})
        universe = sg.universe_from_features(feats)
        self.assertEqual(sg.uncovered_features(universe, feats, [["py"]], {}), [])

    def test_an_optional_dep_no_feature_names_is_in_the_universe_and_the_all_features_set(self):
        self.implicit_case(sg.feature_universe)

    def test_a_dep_named_by_a_feature_is_covered_through_the_closure(self):
        self.closure_case(sg.feature_closure)

    def test_a_dep_named_with_dep_is_not_a_feature(self):
        m = self.pkg('py = ["dep:depx"]\n')
        self.assertEqual(sg.feature_universe(m), ["py"])
        feats = sg.features_from_metadata(sg.cargo_metadata(m), "pkg")
        self.assertEqual(sg.feature_closure(feats, {"py"}), {"py"})
        self.assertEqual(sg.unknown_features(feats, ["depx"]), ["depx"])

    def test_the_closure_follows_slash_forms(self):
        feats = {"default": ["a"], "a": ["b", "dep:z"], "b": [], "depx": ["dep:depx"], "s": ["depx/f"],
                 "w": ["depx?/f"], "n": ["serde/derive"]}
        self.assertEqual(sg.feature_closure(feats, {"a"}), {"a", "b"})
        self.assertEqual(sg.feature_closure(feats, {"s"}), {"s", "depx"})
        self.assertEqual(sg.feature_closure(feats, {"w"}), {"w"})
        self.assertEqual(sg.feature_closure(feats, {"n"}), {"n"})
        universe = sg.universe_from_features(feats)
        self.assertEqual(sg.uncovered_features(universe, feats, [["s", "a"]], {"w": "why"}), ["n"])
        self.assertEqual(sg.uncovered_features(universe, feats, [], {}), sorted(set(feats) - {"default"}))
        self.assertEqual(sg.unknown_features(feats, ["default", "depx", "nope"]), ["nope"])

    def test_a_workspace_selects_the_package_by_name(self):
        ws = self.root / "ws"
        (ws / "m" / "src").mkdir(parents=True)
        (ws / "Cargo.toml").write_text('[workspace]\nmembers = ["m"]\nresolver = "2"\n', encoding="utf-8")
        (ws / "m" / "Cargo.toml").write_text(f'[package]\nname = "m"\nversion = "0.1.0"\nedition = "2021"\n\n'
                                             f'[dependencies]\n{self.dep}', encoding="utf-8")
        (ws / "m" / "src" / "lib.rs").write_text("", encoding="utf-8")
        self.assertEqual(sg.feature_universe(ws / "Cargo.toml", "m"), ["depx"])
        self.assertEqual(sg.feature_universe(ws / "m" / "Cargo.toml"), ["depx"])
        with self.assertRaises(sg.MetadataError):  # a virtual workspace with no name
            sg.feature_universe(ws / "Cargo.toml")
        with self.assertRaises(sg.MetadataError):
            sg.feature_universe(ws / "Cargo.toml", "absent")

    def test_a_metadata_failure_is_an_error(self):
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "Cargo.toml").write_text("[package\n", encoding="utf-8")
        for m in (broken / "Cargo.toml", self.root / "none" / "Cargo.toml"):
            with self.assertRaises(sg.MetadataError):
                sg.feature_universe(m)
        with self.assertRaises(sg.MetadataError):
            sg.features_from_metadata({"packages": [{"name": "pkg"}]}, "pkg")

    def test_non_json_output_is_an_error(self):
        import unittest.mock as um
        done = sg.subprocess.CompletedProcess([], 0, stdout="warning: not json", stderr="")
        with um.patch.object(sg.subprocess, "run", return_value=done), self.assertRaises(sg.MetadataError):
            sg.cargo_metadata(self.pkg(""))

    def test_mutant_universe_from_the_features_table_is_red(self):
        def keys_only(m):
            return sorted(f for f in sg.tomllib.loads(Path(m).read_text())["features"] if f != "default")
        with self.assertRaises(AssertionError):
            self.implicit_case(keys_only)

    def test_mutant_closure_of_raw_names_is_red(self):
        with self.assertRaises(AssertionError):
            self.closure_case(lambda feats, names: set(names))


class ImplicitFeatureEndToEnd(unittest.TestCase):
    """the real tool: the only public item is behind an optional dependency no feature names"""

    def test_removing_it_under_a_minor_step_is_refused_in_the_all_features_set(self):
        if shutil.which("cargo-semver-checks") is None:
            self.fail("cargo-semver-checks is not installed (CI installs it; locally: cargo install cargo-semver-checks)")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_crate(root, "depx")
            dep = f'depx = {{ path = "{root / "depx"}", optional = true }}\n'
            item = '#[cfg(feature = "depx")]\npub fn only() -> u32 {\n    1\n}\n'
            base = write_crate(root / "b", "semverprobe", "1.0.0", dep, "", item)
            new = write_crate(root / "n", "semverprobe", "1.1.0", dep, "", "")
            universe = sg.feature_universe(new / "Cargo.toml")
            self.assertEqual(universe, ["depx"])

            def gate(universe):
                return sg.semver_problems(
                    new, "Cargo.toml", "semverprobe", "1.1.0", {}, None, latest_fixture("semverprobe", "1.0.0"),
                    run=lambda repo, mp, baseline, features, td: sg.run_semver_checks(
                        repo, mp, ["--baseline-root", str(base), *baseline[2:]], features, td),
                    target_dir=root / "target", universe=universe)
            p, n, _ = gate(universe)
            self.assertEqual(n, 2, p)  # default features and none: the item is not there
            self.assertEqual(len(p), 1, p)
            self.assertIn("(all features)", p[0])
            self.assertIn("function_missing", p[0])
            # the [features] keys (none here) would compare no all-features set and pass
            self.assertEqual(gate([])[:2], ([], 2))


if __name__ == "__main__":
    unittest.main()
