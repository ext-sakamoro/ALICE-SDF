#!/usr/bin/env python3
"""Tests of scripts/id_move_check.py: the rules on fixed values, then end to end on a
throwaway crate in a throwaway git repository (built with cargo)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import id_move_check as imc  # noqa: E402

A, B, C = "a1" * 32, "b2" * 32, "c3" * 32
CFG = imc.Config("probe-crate", "Cargo.toml", (), True, ("v{version}",),
                 (imc.IdSpec("probe_crate::SEMANTICS_ID", "probe_crate::PREVIOUS_SEMANTICS_IDS"),))


def ids(value, previous=()):
    return {"probe_crate::SEMANTICS_ID": None if value is None else {"id": value, "previous": list(previous)}}


class Rules(unittest.TestCase):
    """one test per branch of the rule"""
    LOG = f"SEMANTICS_ID moves from {A[:8]}… to {B[:8]}…"

    @staticmethod
    def depends_on(*values):
        """a stand-in for the behaviour check: the named test depends on these old values"""
        def verify(spec, old):
            if old in values:
                return ["tests/golden.rs::previous"], []
            return [], [f"{spec.path}: previous-tests tests/golden.rs::previous still passes with {old[:8]} taken out"]
        return verify

    def test_unmoved_passes(self):
        self.assertEqual(imc.judge("0.3.0", "0.3.1", ids(A), ids(A), "", CFG)[0], [])

    def test_a_breaking_version_with_the_old_value_kept_and_tested_passes(self):
        self.assertEqual(imc.judge("0.3.0", "0.4.0", ids(A), ids(B, [A]), self.LOG, CFG, self.depends_on(A))[0], [])

    def test_no_bump_with_the_old_value_kept_and_tested_passes(self):
        # the old bits are reproduced: not a change of meaning, no bump needed
        problems, report = imc.judge("0.5.0", "0.5.1", ids(A), ids(B, [A]), self.LOG, CFG, self.depends_on(A))
        self.assertEqual(problems, [])
        self.assertIn("exercised by tests/golden.rs::previous", report[0])

    def test_the_old_value_kept_but_not_shown_to_matter_fails(self):
        problems = imc.judge("0.5.0", "0.5.1", ids(A), ids(B, [A]), self.LOG, CFG, self.depends_on())[0]
        self.assertTrue(any("still passes with" in p for p in problems), problems)
        # with a bump too: the claim in the list is still checked
        problems = imc.judge("0.5.0", "0.6.0", ids(A), ids(B, [A]), self.LOG, CFG, self.depends_on())[0]
        self.assertTrue(any("still passes with" in p for p in problems), problems)
        # and no verifier at all is a failure, not a pass
        problems = imc.judge("0.5.0", "0.5.1", ids(A), ids(B, [A]), self.LOG, CFG)[0]
        self.assertTrue(any("nothing tested the claim" in p for p in problems), problems)

    def test_the_named_test_must_be_a_test_that_runs(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        (d / "tests").mkdir()
        (d / "tests" / "golden.rs").write_text("#[test]\nfn ok() {}\n#[test]\n#[ignore]\nfn skipped() {}\nfn helper() {}\n")
        self.assertIsNone(imc._test_fn_problem(d, "tests/golden.rs::ok", "PREVIOUS"))
        self.assertIn("#[ignore]", imc._test_fn_problem(d, "tests/golden.rs::skipped", "PREVIOUS"))
        self.assertIn("no #[test]", imc._test_fn_problem(d, "tests/golden.rs::helper", "PREVIOUS"))
        self.assertIn("no fn absent", imc._test_fn_problem(d, "tests/golden.rs::absent", "PREVIOUS"))
        self.assertIn("does not exist", imc._test_fn_problem(d, "tests/nope.rs::x", "PREVIOUS"))

    def test_the_old_value_is_taken_out_of_each_written_form(self):
        hex_form = f'pub const P: &[[u8; 32]] = &[\n    hex32("{C}"),\n    hex32("{A}"),\n];'
        out = imc.remove_value(hex_form, A)
        self.assertNotIn(A, out)
        self.assertIn(C, out)
        bytes_form = "pub const P: &[[u8; 32]] = &[[\n    " + ", ".join(f"0x{A[k:k+2]}" for k in range(0, 64, 2)) + ",\n]];"
        out = imc.remove_value(bytes_form, A)
        self.assertNotEqual(out, bytes_form)
        self.assertIsNone(imc.remove_value(out, A))
        self.assertEqual(imc.remove_value("pub const P: &[[u8; 32]] = &[[0xa1; 32]];", "a1" * 32),
                         "pub const P: &[[u8; 32]] = &[[0x00; 32]];")
        self.assertIsNone(imc.remove_value("pub const P: &[[u8; 32]] = &[make_old()];", A))

    def test_no_bump_and_the_old_value_not_kept_fails(self):
        problems = imc.judge("0.3.0", "0.3.1", ids(A), ids(B, [C]), self.LOG, CFG, self.depends_on(A))[0]
        self.assertTrue(any("not a breaking step" in p for p in problems), problems)
        problems = imc.judge("1.2.0", "1.3.0", ids(A), ids(B, []), self.LOG, CFG)[0]
        self.assertTrue(any("not a breaking step" in p for p in problems), problems)

    def test_no_previous_list_and_no_bump_fails(self):
        no_list = imc.Config("probe-crate", "Cargo.toml", (), True, ("v{version}",),
                             (imc.IdSpec("probe_crate::SEMANTICS_ID", None),))
        problems = imc.judge("0.3.0", "0.3.1", ids(A), ids(B, [A]), self.LOG, no_list, self.depends_on(A))[0]
        self.assertTrue(any("not a breaking step" in p for p in problems), problems)
        self.assertEqual(imc.judge("0.3.0", "0.4.0", ids(A), ids(B), self.LOG, no_list)[0], [])

    def test_a_breaking_version_without_the_old_value_kept_passes(self):
        # a change of meaning: the old data is refused, so the list need not hold it
        self.assertEqual(imc.judge("0.3.0", "0.4.0", ids(A), ids(B, []), self.LOG, CFG)[0], [])

    def test_a_changelog_missing_one_value_fails(self):
        for log in (f"new id {B[:8]}", f"old id {A[:8]}"):
            problems = imc.judge("0.3.0", "0.4.0", ids(A), ids(B, [A]), log, CFG, self.depends_on(A))[0]
            self.assertTrue(any("CHANGELOG.md does not name" in p for p in problems), (log, problems))

    def test_the_bump_first_message_names_the_fix(self):
        problems = imc.judge("0.6.0", "0.6.0", ids(A), ids(B), self.LOG, CFG)[0]
        self.assertTrue(any('bump the version (set version = "0.7.0"' in p for p in problems), problems)
        problems = imc.judge("3.1.0", "3.1.2", ids(A), ids(B), self.LOG, CFG)[0]
        self.assertTrue(any('set version = "4.0.0"' in p for p in problems), problems)

    # while publication is frozen the version does not move: a **Breaking:** entry of [Unreleased]
    # naming both values is the announcement instead
    BREAKING = f"## [Unreleased]\n\n### Changed\n\n- **Breaking:** clamp follows G2, SEMANTICS_ID moves {A[:8]} -> {B[:8]}\n\n## [0.6.0]\n"

    def test_a_breaking_entry_naming_both_values_announces_the_move(self):
        for base, head in (("0.6.0", "0.6.0"), ("0.5.1", "0.5.2")):
            problems, report = imc.judge(base, head, ids(A), ids(B), self.BREAKING, CFG)
            self.assertEqual(problems, [], (base, head))
            self.assertTrue(any("announced by a **Breaking:** entry" in r for r in report), report)

    def test_a_breaking_entry_that_wraps_onto_the_next_line_counts(self):
        log = f"## [Unreleased]\n\n- **Breaking:** the identifier moves from {A[:8]}\n  to {B[:8]}\n"
        self.assertEqual(imc.judge("0.6.0", "0.6.0", ids(A), ids(B), log, CFG)[0], [])

    def test_without_a_breaking_entry_naming_both_the_move_fails(self):
        for log in (
            # an ordinary entry, not marked breaking
            f"## [Unreleased]\n\n- SEMANTICS_ID moves {A[:8]} -> {B[:8]}\n",
            # a breaking entry naming one value, the other named elsewhere
            f"## [Unreleased]\n\n- **Breaking:** clamp moves to {B[:8]}\n- the old id was {A[:8]}\n",
            # the breaking entry in a released section, not [Unreleased]
            f"## [Unreleased]\n\n## [0.6.0]\n\n- **Breaking:** {A[:8]} -> {B[:8]}\n",
            # the marker inside an entry, not at its start
            f"## [Unreleased]\n\n- moves {A[:8]} -> {B[:8]} (**Breaking:** not the entry's start)\n",
        ):
            problems = imc.judge("0.6.0", "0.6.0", ids(A), ids(B), log, CFG)[0]
            self.assertTrue(any("bump the version" in p for p in problems), (log, problems))

    def test_a_coloured_cargo_test_output_is_counted(self):
        # CARGO_TERM_COLOR=always: libtest colours `ok` / `FAILED`, cargo resets the charset
        plain = "test result: ok. 3 passed; 0 failed; 0 ignored\n"
        coloured = "test result: \x1b[32mok\x1b(B\x1b[m. 3 passed; 1 failed; 0 ignored\n"
        self.assertEqual(imc.tests_ran(plain), 3)
        self.assertEqual(imc.tests_ran(coloured), 4)

    def test_a_version_bump_without_a_move_passes(self):
        self.assertEqual(imc.judge("0.6.0", "0.7.0", ids(A), ids(A), "", CFG)[0], [])

    def test_an_unreadable_registry_fails_on_main_and_tags_and_warns_on_ci_branches(self):
        p = imc.unreadable_base_policy
        self.assertEqual(p({"GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_NAME": "push"}), "fail")
        self.assertEqual(p({"GITHUB_REF": "refs/tags/v1.0.0", "GITHUB_EVENT_NAME": "push"}), "fail")
        self.assertEqual(p({}), "fail")
        self.assertEqual(p({"GITHUB_REF": "refs/heads/ci/x", "GITHUB_EVENT_NAME": "push"}), "warn")
        self.assertEqual(p({"GITHUB_REF": "refs/pull/3/merge", "GITHUB_EVENT_NAME": "pull_request"}), "warn")
        self.assertEqual(p({"GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_NAME": "schedule"}), "fail")

    def test_every_32_byte_constant_is_an_identifier_or_exempt(self):
        found = {"SEMANTICS_ID": ["src/lib.rs:1"], "FORMAT_ID": ["src/container.rs:9"]}
        cfg = imc.Config("p", "Cargo.toml", (), True, ("v{version}",), (imc.IdSpec("p::SEMANTICS_ID", None),))
        problems = imc.completeness(cfg, found)
        self.assertTrue(any(p.startswith("FORMAT_ID (src/container.rs:9)") for p in problems), problems)
        ok = imc.Config("p", "Cargo.toml", (), True, ("v{version}",), cfg.ids,
                        (("FORMAT_ID", "a key, not a stamp of what was computed"),))
        self.assertEqual(imc.completeness(ok, found), [])
        stale = imc.Config("p", "Cargo.toml", (), True, ("v{version}",), cfg.ids,
                           (("FORMAT_ID", "a key, not a stamp of what was computed"), ("GONE", "removed long ago, stale")))
        self.assertTrue(any("GONE" in p and "stale" in p for p in imc.completeness(stale, found)))
        self.assertEqual(imc.completeness(cfg, {}), ["no const / static item of type [u8; 32] found in the crate (compared nothing)"])

    def test_breaking_steps(self):
        self.assertTrue(imc.is_breaking("0.10.0", "0.11.0"))
        self.assertTrue(imc.is_breaking("0.10.3", "1.0.0"))
        self.assertTrue(imc.is_breaking("3.1.0", "4.0.0"))
        self.assertFalse(imc.is_breaking("0.10.0", "0.10.1"))
        self.assertFalse(imc.is_breaking("3.1.0", "3.2.0"))
        self.assertFalse(imc.is_breaking("0.11.0", "0.11.0"))

    def test_a_config_without_identifiers_fails_as_comparing_nothing(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        (d / "scripts").mkdir()
        (d / "scripts" / "id_move.toml").write_text('package = "x"\nmanifest = "Cargo.toml"\ntags = ["v{version}"]\n')
        with self.assertRaisesRegex(imc.CheckError, "compared nothing"):
            imc.load_config(d)


LIB = """pub const SEMANTICS_ID: [u8; 32] = [{b}; 32];
pub const PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[{prev}];
"""


E0599 = ("error[E0599]: no associated function or constant named `{item}` found for struct `{ty}` "
         "in the current scope\n --> /tmp/id-move-probe-x/src/main.rs:3:61\n"
         "error: could not compile `id-move-probe-1` (bin \"id-move-probe-1\") due to 1 previous error\n")


class ProbeAbsent(unittest.TestCase):
    """When a failed probe means the identifier is absent at the base (new), and when it is an error."""

    def absent(self, path: str, stderr: str) -> bool:
        try:
            return imc.probe_says_absent(path, stderr)
        except Exception as e:  # it judges, it never raises: a crash would be an error, not a verdict
            self.fail(f"probe_says_absent raised {e!r}")

    def test_a_missing_associated_const_of_the_named_type_is_absent(self):
        self.assertTrue(self.absent("c::World::DOMAIN", E0599.format(item="DOMAIN", ty="World")))
        # a generic or path-qualified type is the same type
        self.assertTrue(self.absent("c::World::DOMAIN", E0599.format(item="DOMAIN", ty="c::World<T>")))
        self.assertTrue(self.absent(
            "c::World::DOMAIN", E0599.format(item="DOMAIN", ty="World").replace(
                "no associated function or constant", "no associated item")))

    def test_a_coloured_e0599_is_read_like_a_plain_one(self):
        # CARGO_TERM_COLOR=always (as CI sets it) wraps rustc's lines in escape sequences
        coloured = E0599.format(item="DOMAIN", ty="World").replace(
            "error[E0599]: ", "\x1b[1m\x1b[91merror[E0599]\x1b[0m\x1b[1m: ").replace(
            "error: could not", "\x1b[1m\x1b[91merror\x1b[0m: could not")
        self.assertTrue(self.absent("c::World::DOMAIN", coloured))
        self.assertFalse(self.absent("c::World::DOMAIN", coloured.replace("`DOMAIN`", "`OTHER`")))
        other = "\x1b[1m\x1b[91merror[E0308]\x1b[0m\x1b[1m: mismatched types\x1b[0m\n"
        self.assertFalse(self.absent("c::World::DOMAIN", other + coloured))

    def test_an_unrelated_e0599_is_not_absent(self):
        # another item, another type, a substring of the name: each is an error, not a new identifier
        self.assertFalse(self.absent("c::World::DOMAIN", E0599.format(item="OTHER", ty="World")))
        self.assertFalse(self.absent("c::World::DOMAIN", E0599.format(item="DOMAIN", ty="Other")))
        self.assertFalse(self.absent("c::World::DOMAIN", E0599.format(item="DOMAIN_V2", ty="World")))
        self.assertFalse(self.absent("c::World::DOMAIN", E0599.format(item="DOMAIN", ty="MyWorld")))
        # a free constant's path has no type segment to match
        self.assertFalse(self.absent("c::DOMAIN", E0599.format(item="DOMAIN", ty="c")))

    def test_a_failure_with_no_compile_error_or_not_from_the_probe_is_not_absent(self):
        self.assertFalse(self.absent("c::World::DOMAIN", "error: linker `cc` not found\n"))
        not_probe = E0599.format(item="DOMAIN", ty="World").replace("id-move-probe", "some-crate")
        self.assertFalse(self.absent("c::World::DOMAIN", not_probe))

    def test_a_matching_e0599_next_to_another_error_is_not_absent(self):
        other = "error[E0308]: mismatched types\n --> src/lib.rs:2:1\n"
        self.assertFalse(self.absent("c::World::DOMAIN", other + E0599.format(item="DOMAIN", ty="World")))

    def test_an_unresolved_path_is_absent_and_a_build_error_is_not(self):
        self.assertTrue(self.absent(
            "c::SEMANTICS_ID",
            "error[E0425]: cannot find value `SEMANTICS_ID` in crate `c`\n --> /tmp/id-move-probe-x/src/main.rs:3:1\n"))
        self.assertFalse(self.absent(
            "c::SEMANTICS_ID", "error[E0308]: mismatched types\nerror: could not compile `c`\n"))


@unittest.skipUnless(shutil.which("cargo") and shutil.which("git"), "needs cargo and git")
class EndToEnd(unittest.TestCase):
    """A crate released as v0.3.0 with SEMANTICS_ID = [0x11; 32], then changed in the working tree."""

    @classmethod
    def setUpClass(cls):
        cls.target = Path(tempfile.mkdtemp(prefix="id-move-test-target-"))
        os.environ["ID_MOVE_TARGET_DIR"] = str(cls.target)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ID_MOVE_TARGET_DIR", None)
        shutil.rmtree(cls.target, ignore_errors=True)

    def repo(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix="id-move-test-"))
        self.addCleanup(shutil.rmtree, d, True)
        (d / "src").mkdir()
        (d / "scripts").mkdir()
        (d / "scripts" / "id_move.toml").write_text(
            'package = "probe-crate"\nmanifest = "Cargo.toml"\ntags = ["v{version}"]\n'
            '[[id]]\npath = "probe_crate::SEMANTICS_ID"\nprevious = "probe_crate::PREVIOUS_SEMANTICS_IDS"\n'
            'previous-tests = ["tests/previous.rs::old"]\n')
        self.write(d, "0.3.0", "0x11", "")
        (d / "CHANGELOG.md").write_text("# Changelog\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "release"], ["tag", "v0.3.0"]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)
        return d

    def write(self, d: Path, version: str, byte: str, prev: str):
        (d / "Cargo.toml").write_text(f'[package]\nname = "probe-crate"\nversion = "{version}"\nedition = "2021"\n\n[workspace]\n')
        (d / "src" / "lib.rs").write_text(LIB.format(b=byte, prev=prev))

    def test_rustdoc_finds_every_form_of_a_32_byte_item(self):
        d = Path(tempfile.mkdtemp(prefix="id-move-forms-"))
        self.addCleanup(shutil.rmtree, d, True)
        (d / "src").mkdir()
        (d / "Cargo.toml").write_text('[package]\nname = "forms"\nversion = "0.1.0"\nedition = "2021"\n\n[workspace]\n')
        (d / "src" / "lib.rs").write_text(
            "pub const ZZ_A: [ u8 ; 32 ] = [0; 32];\n"
            "pub const ZZ_B:\n    [u8; 32] = [0; 32];\n"
            "pub(crate) const ZZ_C: [u8; 32] = [0; 32];\n"
            "pub const ZZ_D: &[u8; 32] = &[0; 32];\n"
            "pub static ZZ_E: [u8; 0x20] = [0; 32];\n"
            "pub const ZZ_F: [u8; 32usize] = [0; 32];\n"
            "const N: usize = 32;\npub const ZZ_G: [u8; N] = [0; 32];\n"
            "pub type Digest = [u8; 32];\npub const ZZ_H: Digest = [0; 32];\n"
            "mod inner { #[allow(dead_code)] pub const ZZ_I: [u8; 32] = [0; 32]; }\n"
            "const ZZ_J: [u8; 32] = [0; 32];\n"
            "pub const NOT_16: [u8; 16] = [0; 16];\npub const NOT_I8: [i8; 32] = [0; 32];\n"
            "#[cfg(feature = \"extra\")]\npub const ZZ_K: [u8; 32] = [0; 32];\n"
            "pub struct ZzH;\nimpl ZzH { pub const ID: [u8; 32] = [0; 32]; }\n"
            "pub trait Stamp { const TID: [u8; 32]; }\nimpl Stamp for ZzH { const TID: [u8; 32] = [2; 32]; }\n")
        (d / "Cargo.toml").write_text('[package]\nname = "forms"\nversion = "0.1.0"\nedition = "2021"\n\n'
                                      '[features]\nextra = []\n\n[workspace]\n')
        cfg = imc.Config("forms", "Cargo.toml", (), True, ("v{version}",), (imc.IdSpec("forms::ZZ_A", None),))
        # a re-export of another crate's constant: here `core::f64::consts::PI` stands in for an upstream id
        with open(d / "src" / "lib.rs", "a") as f:
            f.write("pub use core::f64::consts::PI as UPSTREAM_ID;\nuse core::f64::consts::E;\n#[allow(dead_code)] const _E: f64 = E;\n"
                    "pub mod zz_glob { pub use core::f64::consts::*; }\n")
        found = imc.id_constants(d, cfg)
        self.assertEqual(sorted(found), sorted([f"ZZ_{c}" for c in "ABCDEFGHIJK"] + ["ZzH::ID", "ZzH::TID", "Stamp::TID", "UPSTREAM_ID",
                                               "core::f64::consts::*"]))
        self.assertIn("re-export of core::f64::consts::PI", found["UPSTREAM_ID"][0])
        self.assertEqual(found["ZZ_B"], ["src/lib.rs:2"])
        problems = imc.completeness(cfg, found)
        self.assertEqual(len(problems), 15, problems)
        # an associated const is listed by Type::NAME
        listed = imc.Config("forms", "Cargo.toml", (), True, ("v{version}",),
                            (imc.IdSpec("forms::ZZ_A", None), imc.IdSpec("forms::ZzH::ID", None)))
        self.assertFalse(any(p.startswith("ZzH::ID ") for p in imc.completeness(listed, found)))
        # without the feature the gated item is not seen: rustdoc runs with every feature
        narrow = imc.Config("forms", "Cargo.toml", (), True, ("v{version}",), cfg.ids, (), ())
        self.assertNotIn("ZZ_K", imc.id_constants(d, narrow))

    def test_unmoved_passes(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x11", "")
        problems, report = imc.check(d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("unchanged (11111111)", report[1])

    def golden(self, d: Path, value_byte: str, depends: bool = True):
        """the named previous test: one that depends on the old value being in the list, or (the
        form a review found passing) one that only writes the value out"""
        (d / "tests").mkdir(exist_ok=True)
        body = (f"assert!(probe_crate::PREVIOUS_SEMANTICS_IDS.contains(&[0x{value_byte}; 32]));" if depends
                else f'let _ = "{value_byte * 32}";')
        (d / "tests" / "previous.rs").write_text(f"#[test]\nfn old() {{ {body} }}\n")

    def test_moved_without_a_bump_and_not_kept_fails(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        problems = imc.check(d, None, "tag")[0]
        self.assertTrue(any("not a breaking step" in p for p in problems), problems)

    def test_moved_without_a_bump_kept_and_tested_passes(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "[0x11; 32]")
        self.golden(d, "11")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        problems, report = imc.check(d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("exercised by tests/previous.rs", report[1])

    def test_a_second_run_with_the_same_target_dir_is_not_fooled_by_the_changed_copy(self):
        # the changed copy of the first run leaves its build in the shared target dir; the next copy
        # must rebuild from its own (unchanged) sources
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "[0x11; 32]")
        self.golden(d, "11")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        self.assertEqual(imc.check(d, None, "tag")[0], [])
        self.assertEqual(imc.check(d, None, "tag")[0], [])

    def test_previous_tests_build_with_the_doc_features(self):
        # two features that cannot be on together: the named test must build with doc-features
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "[0x11; 32]")
        cargo = (d / "Cargo.toml").read_text().replace("[workspace]", '[features]\na = []\nb = []\n\n[workspace]')
        (d / "Cargo.toml").write_text(cargo)
        with open(d / "src" / "lib.rs", "a") as f:
            f.write('#[cfg(all(feature = "a", feature = "b"))]\ncompile_error!("a and b are mutually exclusive");\n')
        self.golden(d, "11")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        cfg = (d / "scripts" / "id_move.toml").read_text()
        # every feature on: the crate does not build at all
        with self.assertRaisesRegex(imc.CheckError, "rustdoc failed"):
            imc.check(d, None, "tag")
        (d / "scripts" / "id_move.toml").write_text('doc-features = ["a"]\n' + cfg)
        self.assertEqual(imc.check(d, None, "tag")[0], [])

    def test_an_empty_doc_features_list_builds_the_default_features_only(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "[0x11; 32]")
        cargo = (d / "Cargo.toml").read_text().replace("[workspace]", '[features]\ndefault = []\na = []\nb = []\n\n[workspace]')
        (d / "Cargo.toml").write_text(cargo)
        with open(d / "src" / "lib.rs", "a") as f:
            f.write('#[cfg(all(feature = "a", feature = "b"))]\ncompile_error!("a and b are mutually exclusive");\n')
        self.golden(d, "11")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        cfg = (d / "scripts" / "id_move.toml").read_text()
        (d / "scripts" / "id_move.toml").write_text("doc-features = []\n" + cfg)
        self.assertEqual(imc.check(d, None, "tag")[0], [])

    def test_a_copy_that_does_not_build_without_the_value_is_not_a_dependence(self):
        # a const assertion on the list breaks the build once the value is taken out: no test ran
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "[0x11; 32]")
        with open(d / "src" / "lib.rs", "a") as f:
            f.write("const _: () = assert!(PREVIOUS_SEMANTICS_IDS[0][0] == 0x11);\n")
        (d / "tests").mkdir(exist_ok=True)
        (d / "tests" / "previous.rs").write_text("#[test]\nfn old() {}\n")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        problems = imc.check(d, None, "tag")[0]
        self.assertTrue(any("did not run with 11111111 taken out" in p for p in problems), problems)

    def test_moved_kept_with_a_test_that_only_names_the_value_fails(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "[0x11; 32]")
        self.golden(d, "11", depends=False)
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        problems = imc.check(d, None, "tag")[0]
        self.assertTrue(any("still passes with 11111111 taken out" in p for p in problems), problems)

    def test_moved_kept_with_a_named_test_that_already_fails_fails(self):
        # a test failing as it is would also "fail without the old value": it must pass first
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "[0x11; 32]")
        (d / "tests").mkdir(exist_ok=True)
        (d / "tests" / "previous.rs").write_text("#[test]\nfn old() { assert!(probe_crate::PREVIOUS_SEMANTICS_IDS.is_empty()); }\n")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        problems = imc.check(d, None, "tag")[0]
        self.assertTrue(any("does not pass as it is" in p for p in problems), problems)

    def test_moved_kept_with_no_named_test_file_fails(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "[0x11; 32]")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        problems = imc.check(d, None, "tag")[0]
        self.assertTrue(any("tests/previous.rs does not exist" in p for p in problems), problems)

    def test_moved_with_a_bump_and_not_kept_passes(self):
        d = self.repo()
        self.write(d, "0.4.0", "0x22", "")
        (d / "CHANGELOG.md").write_text("SEMANTICS_ID 11111111… -> 22222222…\n")
        problems, report = imc.check(d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("moved 11111111 -> 22222222", report[1])

    def test_an_identifier_the_release_did_not_have_is_new(self):
        d = self.repo()
        # rewrite history: the release had no SEMANTICS_ID, and the base reads no previous list
        (d / "src" / "lib.rs").write_text("pub const OTHER: u8 = 1;\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["commit", "-qam", "release without the id"], ["tag", "-f", "v0.3.0"]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)
        self.write(d, "0.3.1", "0x11", "")
        problems, report = imc.check(d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("new since the base", report[1])

    def test_an_associated_const_the_release_did_not_have_is_new(self):
        d = self.repo()
        # the release has the type but not its associated const; the working tree adds it as an id
        (d / "src" / "lib.rs").write_text("pub struct World;\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["commit", "-qam", "release without the const"], ["tag", "-f", "v0.3.0"]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)
        (d / "scripts" / "id_move.toml").write_text(
            'package = "probe-crate"\nmanifest = "Cargo.toml"\ntags = ["v{version}"]\n'
            '[[id]]\npath = "probe_crate::World::DOMAIN"\n')
        (d / "Cargo.toml").write_text('[package]\nname = "probe-crate"\nversion = "0.3.1"\nedition = "2021"\n\n[workspace]\n')
        (d / "src" / "lib.rs").write_text("pub struct World;\nimpl World { pub const DOMAIN: [u8; 32] = [7; 32]; }\n")
        try:
            problems, report = imc.check(d, None, "tag")
        except imc.CheckError as e:
            self.fail(f"a new associated const was not taken as new: {e}")
        self.assertEqual(problems, [])
        self.assertIn("new since the base", report[1])

    def test_a_base_that_does_not_build_is_an_error_not_a_new_identifier(self):
        d = self.repo()
        (d / "src" / "lib.rs").write_text(LIB.format(b="0x11", prev="") + "pub fn broken() -> u8 { \"x\" }\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["commit", "-qam", "broken release"], ["tag", "-f", "v0.3.0"]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)
        self.write(d, "0.3.1", "0x22", "")
        with self.assertRaisesRegex(imc.CheckError, "did not run"):
            imc.check(d, None, "tag")

    def test_the_tag_tree_keeps_export_ignored_files_and_leaves_the_index_alone(self):
        d = self.repo()
        # a module the crate needs, marked export-ignore at the release: `git archive` would drop it
        (d / "src" / "lib.rs").write_text("mod extra;\npub use extra::*;\n")
        (d / "src" / "extra.rs").write_text(LIB.format(b="0x11", prev=""))
        (d / ".gitattributes").write_text("src/extra.rs export-ignore\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["add", "-A"], ["commit", "-qm", "release with an export-ignored module"], ["tag", "-f", "v0.3.0"]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)
        (d / "src" / "extra.rs").write_text(LIB.format(b="0x11", prev=""))
        self.write(d, "0.3.1", "0x11", "")
        before = subprocess.run(["git", "-C", str(d), "status", "--porcelain"], capture_output=True, text=True).stdout
        problems, report = imc.check(d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("unchanged (11111111)", report[1])
        after = subprocess.run(["git", "-C", str(d), "status", "--porcelain"], capture_output=True, text=True).stdout
        # the build of the working tree may write its Cargo.lock; nothing else of the repository moves
        self.assertEqual(before, "".join(ln for ln in after.splitlines(True) if "Cargo.lock" not in ln))
        self.assertEqual([p.name for p in d.parent.glob(f".id-move-base-{os.getpid()}")], [])

    def renamed(self, byte: str, base_package: bool = True) -> Path:
        """released as probe-crate 0.3.0, then renamed to probe-crate-core 1.0.0 (lib probe_crate_core)"""
        d = self.repo()
        (d / "Cargo.toml").write_text('[package]\nname = "probe-crate-core"\nversion = "1.0.0"\nedition = "2021"\n\n[workspace]\n')
        (d / "src" / "lib.rs").write_text(LIB.format(b=byte, prev="[0x11; 32]"))
        self.golden(d, "11")
        (d / "tests" / "previous.rs").write_text((d / "tests" / "previous.rs").read_text().replace("probe_crate::", "probe_crate_core::"))
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        (d / "scripts" / "id_move.toml").write_text(
            'package = "probe-crate-core"\n' + ('base-package = "probe-crate"\n' if base_package else "")
            + 'manifest = "Cargo.toml"\ntags = ["v{version}"]\n'
            '[[id]]\npath = "probe_crate_core::SEMANTICS_ID"\nprevious = "probe_crate_core::PREVIOUS_SEMANTICS_IDS"\n'
            'previous-tests = ["tests/previous.rs::old"]\n')
        return d

    def test_a_renamed_package_reads_the_base_under_its_old_name(self):
        # the tag of the old name is found, and one probe path resolves in the base and in the tree
        problems, report = imc.check(self.renamed("0x11"), None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("base v0.3.0", report[0])
        self.assertIn("unchanged (11111111)", report[1])
        problems, report = imc.check(self.renamed("0x22"), None, "tag")
        self.assertEqual(problems, [], report)
        self.assertIn("11111111", report[1])
        self.assertIn("22222222", report[1])

    def test_a_renamed_package_reads_the_registry_base_under_its_old_name(self):
        # a registry (a directory source in a private CARGO_HOME) that holds only the old name
        # probe-crate 0.3.0: the base has to be asked for under base-package, or it is not found
        d = self.renamed("0x11")
        reg = Path(tempfile.mkdtemp(prefix="id-move-registry-"))
        crate = reg / "vendor" / "probe-crate"
        (crate / "src").mkdir(parents=True)
        manifest = subprocess.run(["git", "-C", str(d), "show", "v0.3.0:Cargo.toml"], check=True,
                                  capture_output=True, text=True).stdout
        (crate / "Cargo.toml").write_text(manifest.replace("\n[workspace]\n", "\n"))
        lib = subprocess.run(["git", "-C", str(d), "show", "v0.3.0:src/lib.rs"], check=True,
                             capture_output=True, text=True).stdout
        (crate / "src" / "lib.rs").write_text(lib)
        (crate / ".cargo-checksum.json").write_text('{"files": {}, "package": null}')
        home = reg / "cargo-home"
        home.mkdir()
        (home / "config.toml").write_text('[source.crates-io]\nreplace-with = "fixture"\n\n'
                                          f'[source.fixture]\ndirectory = {json.dumps(str(reg / "vendor"))}\n')
        old = os.environ.get("CARGO_HOME")
        os.environ["CARGO_HOME"] = str(home)
        try:
            for base_from in ("registry", "auto"):
                problems, report = imc.check(d, None, base_from)
                self.assertEqual(problems, [], report)
                self.assertIn("probe-crate 0.3.0 from the registry", report[0])
                self.assertIn("unchanged (11111111)", report[1])
        finally:
            if old is None:
                os.environ.pop("CARGO_HOME", None)
            else:
                os.environ["CARGO_HOME"] = old
            shutil.rmtree(reg, ignore_errors=True)

    def test_a_renamed_package_without_base_package_finds_no_release(self):
        with self.assertRaisesRegex(imc.CheckError, "no release tag"):
            imc.check(self.renamed("0x11", base_package=False), None, "tag")

    def test_base_package_must_be_a_package_name(self):
        d = self.renamed("0x11")
        cfg = d / "scripts" / "id_move.toml"
        cfg.write_text(cfg.read_text().replace('base-package = "probe-crate"', 'base-package = "probe crate"'))
        with self.assertRaisesRegex(imc.CheckError, "base-package"):
            imc.load_config(d)

    def test_no_release_tag_fails(self):
        d = self.repo()
        subprocess.run(["git", "-C", str(d), "tag", "-d", "v0.3.0"], check=True, capture_output=True)
        with self.assertRaisesRegex(imc.CheckError, "no release tag"):
            imc.check(d, None, "tag")



    def commit(self, d: Path, msg: str):
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["add", "-A"], ["commit", "-qm", msg]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)

    def test_bump_base_named_in_unreleased_passes(self):
        d = self.repo()
        self.write(d, "0.4.0", "0x22", "")
        (d / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n- 11111111 -> 22222222\n")
        self.commit(d, "bump")
        problems, report = imc.check(d, None, "bump", "HEAD")
        self.assertEqual(problems, [])
        self.assertIn("version bump", report[0])
        self.assertIn("(0.3.0)", report[0])
        self.assertTrue(any("moved 11111111 -> 22222222" in r for r in report), report)

    def test_bump_base_named_only_in_an_older_section_fails(self):
        d = self.repo()
        self.write(d, "0.4.0", "0x22", "")
        (d / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n## [0.3.0]\n\n- 11111111 -> 22222222\n")
        self.commit(d, "bump")
        problems = imc.check(d, None, "bump", "HEAD")[0]
        self.assertTrue(any("does not name 11111111, 22222222" in p for p in problems), problems)

    def test_bump_base_without_a_breaking_step_fails(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "")
        (d / "CHANGELOG.md").write_text("## [Unreleased]\n- 11111111 -> 22222222\n")
        self.commit(d, "bump")
        problems = imc.check(d, None, "bump", "HEAD")[0]
        self.assertTrue(any("not a breaking step" in p for p in problems), problems)

    def test_a_second_move_inside_the_unreleased_version_is_judged_against_the_version_before_the_bump(self):
        # the version moved on main after the tag; the identifier moves again later, unpublished
        d = self.repo()
        (d / "README.md").write_text("x\n")
        self.commit(d, "doc")
        self.write(d, "0.4.0", "0x22", "")
        (d / "CHANGELOG.md").write_text("## [Unreleased]\n- 11111111 -> 22222222\n")
        self.commit(d, "bump")
        self.write(d, "0.4.0", "0x33", "")
        (d / "CHANGELOG.md").write_text("## [Unreleased]\n- 11111111 -> 22222222\n- 22222222 -> 33333333\n")
        self.commit(d, "move again")
        ref, kind = imc.choose_base(d, imc.load_config(d), "auto", "HEAD")
        self.assertEqual(kind, "bump")
        problems, report = imc.check(d, None, "auto", "HEAD")
        self.assertEqual(problems, [])
        self.assertTrue(any("moved 11111111 -> 33333333" in r for r in report), report)
        # a move left out of [Unreleased] is red
        (d / "CHANGELOG.md").write_text("## [Unreleased]\n- 11111111 -> 22222222\n")
        problems = imc.check(d, None, "auto", "HEAD")[0]
        # judged against both the release and the bump base, each problem naming its base
        self.assertEqual([b[1] for b in imc.choose_bases(d, imc.load_config(d), "auto", "HEAD")], ["tag", "bump"])
        self.assertTrue(any(p.startswith("[tag base v0.3.0] ") and "does not name 33333333" in p for p in problems),
                        problems)
        self.assertTrue(any(p.startswith("[bump base ") and "does not name 33333333" in p for p in problems),
                        problems)

    def test_a_move_since_the_release_left_out_of_the_new_section_is_red_against_the_bump_base_only(self):
        # named under the released section but not where HEAD's release is described: the tag base
        # (which reads the whole file) passes, the bump base does not
        d = self.repo()
        (d / "README.md").write_text("x\n")
        self.commit(d, "doc")
        self.write(d, "0.4.0", "0x22", "")
        (d / "CHANGELOG.md").write_text("## [Unreleased]\n\n## [0.3.0]\n- 11111111 -> 22222222\n")
        self.commit(d, "bump")
        problems = imc.check(d, None, "auto", "HEAD")[0]
        self.assertTrue(problems)
        self.assertTrue(all(p.startswith("[bump base ") for p in problems), problems)

    def test_auto_takes_the_tag_on_a_tie(self):
        # the tag is the commit before the bump: the same base, read as the release
        d = self.repo()
        self.write(d, "0.4.0", "0x11", "")
        self.commit(d, "bump")
        self.assertEqual(imc.choose_base(d, imc.load_config(d), "auto", "HEAD"), ("v0.3.0", "tag"))

    def test_auto_takes_the_tag_when_it_is_newer_than_the_bump(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x11", "")
        self.commit(d, "release 0.3.1")
        subprocess.run(["git", "-C", str(d), "tag", "v0.3.1"], check=True, capture_output=True)
        (d / "README.md").write_text("x\n")
        self.commit(d, "doc")
        # the bump base is the commit before 0.3.1 (= v0.3.0), older than the tag v0.3.1
        self.assertEqual(imc.choose_base(d, imc.load_config(d), "auto", "HEAD"), ("v0.3.1", "tag"))

    def test_a_crate_without_a_version_before_has_every_identifier_new(self):
        d = self.repo()
        problems, report = imc.check(d, None, "bump", "HEAD")
        self.assertEqual(problems, [])
        self.assertIn("no base", report[0])
        self.assertTrue(any("new since the base (11111111)" in r for r in report), report)

    def test_no_main_and_no_tag_is_an_error(self):
        d = self.repo()
        subprocess.run(["git", "-C", str(d), "tag", "-d", "v0.3.0"], check=True, capture_output=True)
        with self.assertRaises(imc.CheckError):
            imc.choose_base(d, imc.load_config(d), "auto", "no-such-ref")


class ChangelogSections(unittest.TestCase):
    """against a bump base a move is named where HEAD's release is described"""

    TEXT = ("# Changelog\n\n## [Unreleased]\n\n- u1\n\n## [0.4.0] - 2026-10-01\n\n- h1\n\n"
            "## [0.3.0]\n\n- old1\n")

    def test_unreleased_and_the_head_version_only(self):
        got = imc.changelog_sections(self.TEXT, "0.4.0")
        self.assertIn("u1", got)
        self.assertIn("h1", got)
        self.assertNotIn("old1", got)

    def test_another_version_is_not_read(self):
        got = imc.changelog_sections(self.TEXT, "0.5.0")
        self.assertIn("u1", got)
        self.assertNotIn("h1", got)
        self.assertNotIn("old1", got)

    def test_a_v_prefix_and_no_brackets(self):
        text = "## v0.4.0\n- x\n## 0.3.0\n- y\n"
        self.assertIn("x", imc.changelog_sections(text, "0.4.0"))
        self.assertNotIn("y", imc.changelog_sections(text, "0.4.0"))


@unittest.skipUnless(shutil.which("cargo") and shutil.which("git"), "needs cargo and git")
class ReExportedPreviousList(unittest.TestCase):
    """The crate stamps an upstream identifier and re-exports the upstream previous list; the
    upstream is a sibling repository pinned in .github/alice-siblings"""

    ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")

    @classmethod
    def setUpClass(cls):
        cls.target = Path(tempfile.mkdtemp(prefix="id-move-reexport-target-"))
        os.environ["ID_MOVE_TARGET_DIR"] = str(cls.target)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ID_MOVE_TARGET_DIR", None)
        shutil.rmtree(cls.target, ignore_errors=True)

    def git(self, d: Path, *args: str) -> str:
        return subprocess.run(["git", "-C", str(d), *args], check=True, env=self.ENV,
                              capture_output=True, text=True).stdout.strip()

    def setUp(self):
        self.parent = Path(tempfile.mkdtemp(prefix="id-move-reexport-"))
        self.up = self.parent / "ALICE-Upstream"
        (self.up / "src").mkdir(parents=True)
        (self.up / "Cargo.toml").write_text('[package]\nname = "alice-upstream"\nversion = "0.11.0"\nedition = "2021"\n\n[workspace]\n')
        self.upstream_list("[0x11; 32]")
        self.git(self.up, "init", "-q")
        self.git(self.up, "config", "maintenance.auto", "false")
        self.git(self.up, "add", "-A")
        self.git(self.up, "commit", "-qm", "one")
        self.first = self.git(self.up, "rev-parse", "HEAD")
        (self.up / "README").write_text("two\n")
        self.git(self.up, "add", "-A")
        self.git(self.up, "commit", "-qm", "two")
        self.d = self.parent / "ALICE-Probe"
        (self.d / "src").mkdir(parents=True)
        (self.d / "tests").mkdir()
        (self.d / ".github").mkdir()
        (self.d / "Cargo.toml").write_text(
            '[package]\nname = "probe-crate"\nversion = "0.6.0"\nedition = "2021"\n\n[dependencies]\n'
            'alice-upstream = { path = "../ALICE-Upstream", version = ">=0.11, <0.12" }\n\n[workspace]\n')
        (self.d / "src" / "lib.rs").write_text("pub use alice_upstream::{PREVIOUS_SEMANTICS_IDS, SEMANTICS_ID};\n")
        self.previous_test("[0x11; 32]")
        self.pin(self.git(self.up, "rev-parse", "HEAD"))
        self.git(self.d, "init", "-q")
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "probe")
        self.cfg = imc.Config("probe-crate", "Cargo.toml", (), True, ("v{version}",),
                              (imc.IdSpec("probe_crate::SEMANTICS_ID", "probe_crate::PREVIOUS_SEMANTICS_IDS",
                                          ("tests/previous.rs::old",)),))

    def tearDown(self):
        shutil.rmtree(self.parent, ignore_errors=True)

    def upstream_list(self, prev: str):
        (self.up / "src" / "lib.rs").write_text(
            "pub const SEMANTICS_ID: [u8; 32] = [0x22; 32];\n"
            f"pub const PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[{prev}];\n")

    def previous_test(self, value: str):
        (self.d / "tests" / "previous.rs").write_text(
            f"#[test]\nfn old() {{ assert!(probe_crate::PREVIOUS_SEMANTICS_IDS.contains(&{value})); }}\n")

    def pin(self, sha: str):
        (self.d / ".github" / "alice-siblings").write_text(f"ALICE-Upstream {sha}\n")

    def verify(self):
        doc = imc.rustdoc_json(self.d, self.cfg)
        return imc.PreviousVerifier(self.d, self.cfg, doc, self.target)(self.cfg.ids[0], "11" * 32)

    def test_the_old_value_is_taken_out_of_the_pinned_sibling(self):
        before = (self.up / "src" / "lib.rs").read_text()
        passed, problems = self.verify()
        self.assertEqual(problems, [])
        self.assertEqual(passed, ["tests/previous.rs::old"])
        self.assertEqual((self.up / "src" / "lib.rs").read_text(), before)
        self.assertFalse(any(p.name.startswith(".id-move-previous-") for p in self.parent.iterdir()))

    def test_a_sibling_without_the_old_value_fails(self):
        self.upstream_list("[0x33; 32]")
        self.git(self.up, "commit", "-qam", "other value")
        self.pin(self.git(self.up, "rev-parse", "HEAD"))
        self.previous_test("[0x33; 32]")
        passed, problems = self.verify()
        self.assertEqual(passed, [])
        self.assertTrue(any("11111111 is not written in ../ALICE-Upstream/src/lib.rs" in p for p in problems), problems)

    def test_a_test_that_does_not_depend_on_the_sibling_value_fails(self):
        (self.d / "tests" / "previous.rs").write_text("#[test]\nfn old() { let _ = probe_crate::PREVIOUS_SEMANTICS_IDS; }\n")
        passed, problems = self.verify()
        self.assertEqual(passed, [])
        self.assertTrue(any("still passes" in p or "names no" in p or "does not" in p for p in problems), problems)

    def test_a_sibling_at_another_commit_than_the_pin_fails(self):
        self.pin(self.first)
        passed, problems = self.verify()
        self.assertEqual(passed, [])
        self.assertTrue(any("but .github/alice-siblings pins" in p for p in problems), problems)

    def test_a_dirty_sibling_fails(self):
        (self.up / "README").write_text("dirty\n")
        passed, problems = self.verify()
        self.assertTrue(any("uncommitted changes" in p for p in problems), problems)

    def test_a_re_export_from_an_unlisted_crate_fails(self):
        # the listed sibling is a real checkout of another crate: only the name match tells them apart
        other = self.parent / "ALICE-Other"
        (other / "src").mkdir(parents=True)
        (other / "Cargo.toml").write_text('[package]\nname = "alice-other"\nversion = "0.1.0"\nedition = "2021"\n\n[workspace]\n')
        (other / "src" / "lib.rs").write_text("pub const PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[[0x11; 32]];\n")
        self.git(other, "init", "-q")
        self.git(other, "add", "-A")
        self.git(other, "commit", "-qm", "other")
        (self.d / ".github" / "alice-siblings").write_text(f"ALICE-Other {self.git(other, 'rev-parse', 'HEAD')}\n")
        passed, problems = self.verify()
        self.assertEqual(passed, [])
        self.assertTrue(any("not a sibling repository listed" in p for p in problems), problems)
        (self.d / ".github" / "alice-siblings").unlink()
        passed, problems = self.verify()
        self.assertTrue(any(".github/alice-siblings is missing" in p for p in problems), problems)

@unittest.skipUnless(shutil.which("cargo") and shutil.which("git"), "needs cargo and git")
class SiblingPointsBack(unittest.TestCase):
    """A workspace member path-depends on a sibling repository that path-depends back on a crate of
    this repository (ALICE-Probe/robot -> ../../ALICE-Back -> ../ALICE-Probe/probe): a copy of the
    repository built next to the root holds two packages of the same name and version"""

    ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    REF = "tests/previous.rs::old"

    @classmethod
    def setUpClass(cls):
        cls.target = Path(tempfile.mkdtemp(prefix="id-move-back-target-"))
        os.environ["ID_MOVE_TARGET_DIR"] = str(cls.target)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ID_MOVE_TARGET_DIR", None)
        shutil.rmtree(cls.target, ignore_errors=True)

    def git(self, d: Path, *args: str) -> str:
        return subprocess.run(["git", "-C", str(d), *args], check=True, env=self.ENV,
                              capture_output=True, text=True).stdout.strip()

    def setUp(self):
        self.parent = Path(tempfile.mkdtemp(prefix="id-move-back-"))
        self.back = self.parent / "ALICE-Back"
        (self.back / "src").mkdir(parents=True)
        (self.back / "Cargo.toml").write_text(
            '[package]\nname = "alice-back"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n'
            'probe-crate = { path = "../ALICE-Probe/probe" }\n\n[workspace]\n')
        (self.back / "src" / "lib.rs").write_text("pub use probe_crate::SEMANTICS_ID;\n")
        self.git(self.back, "init", "-q")
        self.git(self.back, "add", "-A")
        self.git(self.back, "commit", "-qm", "back")
        self.d = self.parent / "ALICE-Probe"
        for sub in ("probe/src", "probe/tests", "robot/src", "scripts", ".github"):
            (self.d / sub).mkdir(parents=True)
        (self.d / ".github" / "alice-siblings").write_text(
            f"ALICE-Back {self.git(self.back, 'rev-parse', 'HEAD')}\n")
        (self.d / "Cargo.toml").write_text('[workspace]\nmembers = ["probe", "robot"]\nresolver = "2"\n')
        (self.d / "probe" / "Cargo.toml").write_text(
            '[package]\nname = "probe-crate"\nversion = "0.3.0"\nedition = "2021"\n')
        (self.d / "probe" / "src" / "lib.rs").write_text(LIB.format(b="0x22", prev="[0x11; 32]"))
        (self.d / "probe" / "tests" / "previous.rs").write_text(
            "#[test]\nfn old() { assert!(probe_crate::PREVIOUS_SEMANTICS_IDS.contains(&[0x11; 32])); }\n")
        (self.d / "robot" / "Cargo.toml").write_text(
            '[package]\nname = "robot"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n'
            'alice-back = { path = "../../ALICE-Back" }\n')
        (self.d / "robot" / "src" / "lib.rs").write_text("pub use alice_back::SEMANTICS_ID;\n")
        (self.d / "scripts" / "id_move.toml").write_text(
            'package = "probe-crate"\nmanifest = "probe/Cargo.toml"\ntags = ["v{version}"]\n'
            '[[id]]\npath = "probe_crate::SEMANTICS_ID"\nprevious = "probe_crate::PREVIOUS_SEMANTICS_IDS"\n'
            f'previous-tests = ["{self.REF}"]\n')
        (self.d / "CHANGELOG.md").write_text("# Changelog\n")
        self.git(self.d, "init", "-q")
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "release")
        self.git(self.d, "tag", "v0.3.0")
        self.cfg = imc.load_config(self.d)

    def tearDown(self):
        shutil.rmtree(self.parent, ignore_errors=True)

    def verify(self):
        doc = imc.rustdoc_json(self.d / "probe", self.cfg)
        return imc.PreviousVerifier(self.d, self.cfg, doc, self.target)(self.cfg.ids[0], "11" * 32)

    def test_the_fixture_collides_in_a_copy_next_to_the_root(self):
        # the layout the copies had before: the sibling's `../ALICE-Probe` reaches the working tree
        old = self.parent / ".ALICE-Probe.id-move-previous-old"
        imc.copy_tracked(self.d, old)
        r = subprocess.run(["cargo", "metadata", "--format-version", "1", "--offline",
                            "--manifest-path", str(old / "Cargo.toml")], capture_output=True, text=True,
                           env=dict(os.environ, CARGO_TARGET_DIR=str(self.target)))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("package collision in the lockfile", r.stderr)

    def test_the_siblings_are_read_from_the_manifests(self):
        self.assertEqual(imc.path_dep_siblings(self.d, self.parent), ["ALICE-Back"])

    def test_previous_tests_run_against_the_copy(self):
        before = self.git(self.d, "status", "--porcelain")
        passed, problems = self.verify()
        self.assertEqual(problems, [])
        self.assertEqual(passed, [self.REF])
        after = self.git(self.d, "status", "--porcelain")
        # the build of the working tree may write its Cargo.lock; nothing else of the repository moves
        self.assertEqual(before, "\n".join(ln for ln in after.splitlines() if "Cargo.lock" not in ln))
        self.assertEqual(sorted(p.name for p in self.parent.iterdir()), ["ALICE-Back", "ALICE-Probe"])

    def test_the_base_tree_is_read_with_a_sibling_that_points_back(self):
        # the base identifiers come from a probe build of the crate alone (no other member), so the
        # base tree next to the root does not collide; this pins that it stays so
        (self.d / "probe" / "Cargo.toml").write_text(
            '[package]\nname = "probe-crate"\nversion = "0.3.1"\nedition = "2021"\n')
        problems, report = imc.check(self.d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("unchanged (22222222)", report[1])
        self.assertEqual(sorted(p.name for p in self.parent.iterdir()), ["ALICE-Back", "ALICE-Probe"])

    def test_a_named_sibling_that_is_not_checked_out_fails(self):
        doc = imc.rustdoc_json(self.d / "probe", self.cfg)
        self.back.rename(self.parent / "elsewhere")
        passed, problems = imc.PreviousVerifier(self.d, self.cfg, doc, self.target)(self.cfg.ids[0], "11" * 32)
        self.assertEqual(passed, [])
        self.assertTrue(any("names ../ALICE-Back, which is not checked out" in p for p in problems), problems)


@unittest.skipUnless(shutil.which("cargo") and shutil.which("git"), "needs cargo and git")
class BaseTreeAtItsPins(unittest.TestCase):
    """The base tree path-depends on a sibling that has since taken a major version: the base is
    built against the sibling commit the base tree pins in .github/alice-siblings, not today's
    checkout"""

    ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")

    @classmethod
    def setUpClass(cls):
        cls.target = Path(tempfile.mkdtemp(prefix="id-move-pins-target-"))
        os.environ["ID_MOVE_TARGET_DIR"] = str(cls.target)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ID_MOVE_TARGET_DIR", None)
        shutil.rmtree(cls.target, ignore_errors=True)

    def git(self, d: Path, *args: str) -> str:
        return subprocess.run(["git", "-C", str(d), *args], check=True, env=self.ENV,
                              capture_output=True, text=True).stdout.strip()

    def up_version(self, version: str):
        (self.up / "Cargo.toml").write_text(
            f'[package]\nname = "alice-up"\nversion = "{version}"\nedition = "2021"\n\n[workspace]\n')
        self.git(self.up, "add", "-A")
        self.git(self.up, "commit", "-qm", version)
        return self.git(self.up, "rev-parse", "HEAD")

    def probe(self, version: str, up_req: str, pin: str | None):
        (self.d / "Cargo.toml").write_text(
            f'[package]\nname = "probe-crate"\nversion = "{version}"\nedition = "2021"\n\n[dependencies]\n'
            f'alice-up = {{ path = "../ALICE-Up", version = "{up_req}" }}\n\n[workspace]\n')
        if pin is None:
            (self.d / ".github" / "alice-siblings").unlink(missing_ok=True)
        else:
            (self.d / ".github" / "alice-siblings").write_text(f"ALICE-Up {pin}\n")

    def setUp(self):
        self.parent = Path(tempfile.mkdtemp(prefix="id-move-pins-"))
        self.up = self.parent / "ALICE-Up"
        (self.up / "src").mkdir(parents=True)
        (self.up / "src" / "lib.rs").write_text("pub fn f() {}\n")
        self.git(self.up, "init", "-q")
        self.git(self.up, "config", "maintenance.auto", "false")
        self.v1 = self.up_version("1.0.0")
        self.d = self.parent / "ALICE-Probe"
        (self.d / "src").mkdir(parents=True)
        (self.d / "scripts").mkdir()
        (self.d / ".github").mkdir()
        (self.d / "scripts" / "id_move.toml").write_text(
            'package = "probe-crate"\nmanifest = "Cargo.toml"\ntags = ["v{version}"]\n'
            '[[id]]\npath = "probe_crate::SEMANTICS_ID"\n')
        (self.d / "src" / "lib.rs").write_text("pub const SEMANTICS_ID: [u8; 32] = [0x11; 32];\n")
        (self.d / "CHANGELOG.md").write_text("# Changelog\n")
        self.probe("0.3.0", "^1", self.v1)
        self.git(self.d, "init", "-q")
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "release")
        self.git(self.d, "tag", "v0.3.0")
        # the sibling takes a major version and the working tree follows it
        self.v2 = self.up_version("2.0.0")
        self.probe("0.4.0", "^2", self.v2)

    def tearDown(self):
        shutil.rmtree(self.parent, ignore_errors=True)

    def test_the_base_tree_builds_against_the_sibling_it_pins(self):
        problems, report = imc.check(self.d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("unchanged (11111111)", report[1])
        self.assertEqual(self.git(self.up, "rev-parse", "HEAD"), self.v2)
        self.assertEqual(sorted(p.name for p in self.parent.iterdir()), ["ALICE-Probe", "ALICE-Up"])

    def test_a_base_tree_path_dependency_without_a_pin_fails(self):
        self.probe("0.3.0", "^1", None)
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "no pin")
        self.git(self.d, "tag", "-f", "v0.3.0")
        self.probe("0.4.0", "^2", self.v2)
        # the pin file is in the history of the base but not at it: a broken base, never a fallback
        with self.assertRaises(imc.CheckError) as e:
            imc.check(self.d, None, "tag")
        self.assertIn("broken base", str(e.exception))

    def test_a_pin_that_is_not_in_the_sibling_fails(self):
        self.probe("0.3.0", "^1", "ab" * 20)
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "bad pin")
        self.git(self.d, "tag", "-f", "v0.3.0")
        self.probe("0.4.0", "^2", self.v2)
        with self.assertRaises(imc.CheckError) as e:
            imc.check(self.d, None, "tag")
        self.assertIn("abababababab", str(e.exception))


@unittest.skipUnless(shutil.which("git"), "needs git")
class PrintSiblings(unittest.TestCase):
    """--print-siblings lists, before any sibling is cloned, every sibling commit a base tree can be
    built against: on a chain ALICE-Probe -> ALICE-Up -> ALICE-Mid -> ALICE-Low (each a path
    dependency of the one before), the list holds all three levels, and every sibling the check's
    own walk lays out next to the base is in it"""

    ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(self, d: Path, *args: str) -> str:
        return subprocess.run(["git", "-C", str(d), *args], check=True, env=self.ENV,
                              capture_output=True, text=True).stdout.strip()

    def repo(self, name: str, dep: str | None) -> str:
        d = self.parent / name
        (d / "src").mkdir(parents=True, exist_ok=True)
        (d / "src" / "lib.rs").write_text("pub fn f() {}\n")
        deps = f'\n[dependencies]\nnext = {{ path = "../{dep}" }}\n' if dep else ""
        n = len(list(d.glob("rev-*")))
        (d / f"rev-{n}").write_text("")
        (d / "Cargo.toml").write_text(f'[package]\nname = "{name.lower()}"\nversion = "1.0.0"\n{deps}')
        if not (d / ".git").exists():
            self.git(d, "init", "-q", "-b", "main")
        self.git(d, "add", "-A")
        self.git(d, "commit", "-qm", name)
        return self.git(d, "rev-parse", "HEAD")

    def release(self, pins: str | None, dep: str | None = "ALICE-Up") -> str:
        deps = f'\n[dependencies]\nup = {{ path = "../{dep}" }}\n' if dep else ""
        (self.d / "Cargo.toml").write_text(f'[package]\nname = "probe-crate"\nversion = "0.3.0"\n{deps}')
        f = self.d / ".github" / "alice-siblings"
        if pins is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(pins)
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "release", "--allow-empty")
        self.git(self.d, "tag", "-f", "v0.3.0")
        return self.git(self.d, "rev-parse", "HEAD")

    def setUp(self):
        self.parent = Path(tempfile.mkdtemp(prefix="id-move-print-siblings-"))
        self.low = self.repo("ALICE-Low", None)
        self.mid = self.repo("ALICE-Mid", "ALICE-Low")
        self.up = self.repo("ALICE-Up", "ALICE-Mid")
        self.d = self.parent / "ALICE-Probe"
        (self.d / "scripts").mkdir(parents=True)
        (self.d / ".github").mkdir()
        (self.d / "scripts" / "id_move.toml").write_text(
            'package = "probe-crate"\nmanifest = "Cargo.toml"\ntags = ["v{version}"]\n'
            '[[id]]\npath = "probe_crate::SEMANTICS_ID"\n')
        self.git(self.d, "init", "-q", "-b", "main")
        self.pins = f"# siblings\nALICE-Up {self.up}\nALICE-Mid {self.mid}\nALICE-Low {self.low}\n"

    def tearDown(self):
        shutil.rmtree(self.parent, ignore_errors=True)

    def printed(self, *extra: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = imc.main(["--root", str(self.d), "--base-from", "tag", "--print-siblings", *extra])
        return rc, out.getvalue(), err.getvalue()

    def test_every_level_of_the_chain_is_listed(self):
        self.release(self.pins)
        rc, out, err = self.printed()
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.splitlines(), sorted([f"ALICE-Low {self.low}", f"ALICE-Mid {self.mid}",
                                                   f"ALICE-Up {self.up}"]))

    def test_the_list_holds_what_the_walk_lays_out(self):
        # the check's own walk over the base tree, at depth 3: each sibling it writes next to the
        # base, at the commit it took, is a line of the list
        ref = self.release(self.pins)
        listed = set(imc.base_sibling_pins(self.d, [ref]))
        work = Path(tempfile.mkdtemp(prefix="id-move-print-siblings-work-"))
        try:
            fresh = work / "fresh"
            wt = fresh / "ALICE-Probe"
            wt.mkdir(parents=True)
            env = dict(self.ENV, GIT_INDEX_FILE=str(work / "base.index"))
            for args in (["read-tree", ref], ["checkout-index", "--all", "--force", f"--prefix={wt}/"]):
                subprocess.run(["git", "-C", str(self.d), *args], check=True, env=env, capture_output=True)
            lines = imc.materialize_pinned_siblings(wt, self.parent, work, self.d, ref)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        walked = set()
        for line in lines:
            m = re.match(r"\.\./(ALICE-[A-Za-z]+) at ([0-9a-f]{12})", line)
            walked.add((m.group(1), m.group(2)))
        self.assertEqual(len(walked), 3, lines)
        self.assertEqual(walked, {(r, sha[:12]) for r, sha in listed})

    def test_two_bases_are_listed_together_once(self):
        first = self.release(self.pins)
        mid2 = self.repo("ALICE-Mid", "ALICE-Low")
        second = self.release(f"ALICE-Up {self.up}\nALICE-Mid {mid2}\nALICE-Low {self.low}\n")
        got = imc.base_sibling_pins(self.d, [first, second])
        self.assertEqual(got, sorted([("ALICE-Low", self.low), ("ALICE-Mid", self.mid), ("ALICE-Mid", mid2),
                                      ("ALICE-Up", self.up)]))

    def test_a_base_without_path_dependencies_lists_nothing(self):
        self.release(None, dep=None)
        rc, out, err = self.printed()
        self.assertEqual((rc, out), (0, ""), err)

    def test_a_reached_sibling_without_a_pin_fails(self):
        self.release(f"ALICE-Mid {self.mid}\nALICE-Low {self.low}\n")
        rc, out, err = self.printed()
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("path-depends on ../ALICE-Up but its .github/alice-siblings has no pin", err)

    def test_a_malformed_pin_fails(self):
        self.release(self.pins + "ALICE-Low abc\n")
        rc, _, err = self.printed()
        self.assertEqual(rc, 1)
        self.assertIn("not one '<ALICE-Repo> <40-hex SHA>' line", err)

    def test_a_base_from_before_the_pin_file_whose_sibling_cannot_be_fetched_fails(self):
        self.release(None)
        os.environ["ALICE_SIBLINGS_GIT_BASE"] = (self.parent / "no-remote").as_uri()
        try:
            rc, out, err = self.printed()
        finally:
            os.environ.pop("ALICE_SIBLINGS_GIT_BASE", None)
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("reaches ../ALICE-Up, whose main cannot be fetched", err)

    def test_a_base_whose_history_had_the_pin_file_fails(self):
        self.release(self.pins)
        self.release(None)
        rc, _, err = self.printed()
        self.assertEqual(rc, 1)
        self.assertIn("broken base", err)

    def test_base_names_the_base(self):
        ref = self.release(self.pins)
        self.release(None, dep=None)
        rc, out, err = self.printed("--base", ref)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out.splitlines()), 3)


@unittest.skipUnless(shutil.which("git"), "needs git")
class PrintSiblingsBeforePins(unittest.TestCase):
    """A base from before .github/alice-siblings: --print-siblings resolves each sibling it reaches, at
    any depth, at the newest commit of the sibling's main committed at or before the base commit, by
    fetching that main (here from file:// remotes, no network), and the result is what the check's
    own walk takes. The chain ALICE-Probe -> ALICE-Up -> ALICE-Mid -> ALICE-Low; every sibling has a
    commit before the base and one after it, and ALICE-Up's later commit names another sibling"""

    BEFORE, BASE, AFTER = "2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z"

    def env(self, date: str) -> dict:
        return dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                    GIT_COMMITTER_EMAIL="t@t", GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date,
                    GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(self, d: Path, *args: str, date: str = BEFORE) -> str:
        return subprocess.run(["git", "-C", str(d), *args], check=True, env=self.env(date),
                              capture_output=True, text=True).stdout.strip()

    def commit(self, name: str, dep: str | None, date: str) -> str:
        d = self.parent / name
        (d / "src").mkdir(parents=True, exist_ok=True)
        (d / "src" / "lib.rs").write_text(f"// {date}\n")
        deps = f'\n[dependencies]\nnext = {{ path = "../{dep}" }}\n' if dep else ""
        (d / "Cargo.toml").write_text(f'[package]\nname = "{name.lower()}"\nversion = "1.0.0"\n{deps}')
        if not (d / ".git").exists():
            bare = self.remotes / name
            bare.mkdir(parents=True)
            self.git(bare, "init", "-q", "--bare", "-b", "main")
            self.git(d, "init", "-q", "-b", "main")
            self.git(d, "config", "maintenance.auto", "false")
            self.git(d, "remote", "add", "origin", str(bare))
        self.git(d, "add", "-A", date=date)
        self.git(d, "commit", "-qm", date, date=date)
        self.git(d, "pu" + "sh", "-q", "origin", "HEAD:refs/heads/main", date=date)
        return self.git(d, "rev-parse", "HEAD")

    def setUp(self):
        self.parent = Path(tempfile.mkdtemp(prefix="id-move-prepin-print-"))
        self.remotes = self.parent / "remote"
        self.low1 = self.commit("ALICE-Low", None, self.BEFORE)
        self.mid1 = self.commit("ALICE-Mid", "ALICE-Low", self.BEFORE)
        self.up1 = self.commit("ALICE-Up", "ALICE-Mid", self.BEFORE)
        self.d = self.parent / "ALICE-Probe"
        (self.d / "scripts").mkdir(parents=True)
        (self.d / ".github").mkdir()
        (self.d / "scripts" / "id_move.toml").write_text(
            'package = "probe-crate"\nmanifest = "Cargo.toml"\ntags = ["v{version}"]\n'
            '[[id]]\npath = "probe_crate::SEMANTICS_ID"\n')
        (self.d / "Cargo.toml").write_text(
            '[package]\nname = "probe-crate"\nversion = "0.3.0"\n\n[dependencies]\nup = { path = "../ALICE-Up" }\n')
        self.git(self.d, "init", "-q", "-b", "main")
        self.git(self.d, "add", "-A", date=self.BASE)
        self.git(self.d, "commit", "-qm", "release", date=self.BASE)
        self.git(self.d, "tag", "v0.3.0", date=self.BASE)
        self.ref = self.git(self.d, "rev-parse", "HEAD")
        # after the base: every sibling moves on, and ALICE-Up then names ALICE-Other instead
        self.commit("ALICE-Other", None, self.AFTER)
        self.commit("ALICE-Low", None, self.AFTER)
        self.commit("ALICE-Mid", "ALICE-Low", self.AFTER)
        self.commit("ALICE-Up", "ALICE-Other", self.AFTER)
        os.environ["ALICE_SIBLINGS_GIT_BASE"] = self.remotes.as_uri()

    def tearDown(self):
        os.environ.pop("ALICE_SIBLINGS_GIT_BASE", None)
        shutil.rmtree(self.parent, ignore_errors=True)

    def expected(self) -> list[tuple[str, str]]:
        return sorted([("ALICE-Low", self.low1), ("ALICE-Mid", self.mid1), ("ALICE-Up", self.up1)])

    def test_every_level_is_taken_at_its_main_as_of_the_base(self):
        self.assertEqual(imc.base_sibling_pins(self.d, [self.ref]), self.expected())

    def test_the_cli_prints_the_lines(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = imc.main(["--root", str(self.d), "--base-from", "tag", "--print-siblings"])
        self.assertEqual(rc, 0, err.getvalue())
        self.assertEqual(out.getvalue().splitlines(), [f"{r} {sha}" for r, sha in self.expected()])

    def test_the_same_base_gives_the_same_lines(self):
        self.assertEqual(imc.base_sibling_pins(self.d, [self.ref]), imc.base_sibling_pins(self.d, [self.ref]))

    def test_the_list_is_what_the_walk_takes(self):
        # the check's own walk, with each sibling checked out next to the repository (as the action
        # leaves it): every sibling it lays out, at the commit it took, is a printed line, and back
        listed = {(r, sha[:12]) for r, sha in imc.base_sibling_pins(self.d, [self.ref])}
        work = Path(tempfile.mkdtemp(prefix="id-move-prepin-print-work-"))
        try:
            wt = work / "fresh" / "ALICE-Probe"
            wt.mkdir(parents=True)
            env = dict(self.env(self.BASE), GIT_INDEX_FILE=str(work / "base.index"))
            for args in (["read-tree", self.ref], ["checkout-index", "--all", "--force", f"--prefix={wt}/"]):
                subprocess.run(["git", "-C", str(self.d), *args], check=True, env=env, capture_output=True)
            lines = imc.materialize_pinned_siblings(wt, self.parent, work, self.d, self.ref)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        walked = {re.match(r"\.\./(ALICE-[A-Za-z]+) at ([0-9a-f]{12})", line).groups() for line in lines}
        self.assertEqual(len(walked), 3, lines)
        self.assertEqual(walked, listed)

    def test_a_private_sibling_fails_naming_it(self):
        (self.d / ".github" / "alice-siblings-private").write_text("ALICE-Mid\n")
        with self.assertRaises(imc.CheckError) as e:
            imc.base_sibling_pins(self.d, [self.ref])
        self.assertIn("reaches ../ALICE-Mid, which .github/alice-siblings-private lists", str(e.exception))

    def test_a_sibling_that_cannot_be_fetched_fails_naming_it(self):
        shutil.rmtree(self.remotes / "ALICE-Low")
        with self.assertRaises(imc.CheckError) as e:
            imc.base_sibling_pins(self.d, [self.ref])
        self.assertIn("reaches ../ALICE-Low, whose main cannot be fetched", str(e.exception))

    def test_a_sibling_with_no_commit_by_the_base_date_fails(self):
        self.git(self.d, "commit", "-q", "--amend", "--no-edit", date="2026-01-01T00:00:00Z")
        ref = self.git(self.d, "rev-parse", "HEAD")
        with self.assertRaises(imc.CheckError) as e:
            imc.base_sibling_pins(self.d, [ref])
        self.assertIn("../ALICE-Up has no commit of its origin main committed at or before", str(e.exception))


@unittest.skipUnless(shutil.which("cargo") and shutil.which("git"), "needs cargo and git")
class GitRevPreviousList(unittest.TestCase):
    """The crate re-exports the previous list of a dependency it takes by git + rev (not a path
    sibling): the old value is taken out of an archive of that rev laid out next to the copy, which
    the copy uses through a [patch] of the git source"""

    ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    REF = "tests/previous.rs::old"

    @classmethod
    def setUpClass(cls):
        cls.target = Path(tempfile.mkdtemp(prefix="id-move-gitrev-target-"))
        os.environ["ID_MOVE_TARGET_DIR"] = str(cls.target)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ID_MOVE_TARGET_DIR", None)
        shutil.rmtree(cls.target, ignore_errors=True)

    def git(self, d: Path, *args: str) -> str:
        return subprocess.run(["git", "-C", str(d), *args], check=True, env=self.ENV,
                              capture_output=True, text=True).stdout.strip()

    def upstream_list(self, prev: str):
        (self.up / "src" / "lib.rs").write_text(
            "pub const SEMANTICS_ID: [u8; 32] = [0x22; 32];\n"
            f"pub const PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[{prev}];\n")

    def consumer(self, rev: str,
                 test_body: str = "assert!(probe_crate::PREVIOUS_SEMANTICS_IDS.contains(&[0x11; 32]));"):
        (self.d / "Cargo.toml").write_text(
            '[package]\nname = "probe-crate"\nversion = "0.6.0"\nedition = "2021"\n\n[dependencies]\n'
            f'alice-zz = {{ git = "{self.url}", rev = "{rev}", version = "0.11" }}\n\n[workspace]\n')
        (self.d / "tests" / "previous.rs").write_text(f"#[test]\nfn old() {{ {test_body} }}\n")

    def setUp(self):
        self.parent = Path(tempfile.mkdtemp(prefix="id-move-gitrev-"))
        bare = self.parent / "remote" / "ALICE-Zz.git"
        bare.mkdir(parents=True)
        self.git(bare, "init", "-q", "--bare")
        self.up = self.parent / "ALICE-Zz"
        (self.up / "src").mkdir(parents=True)
        (self.up / "Cargo.toml").write_text(
            '[package]\nname = "alice-zz"\nversion = "0.11.0"\nedition = "2021"\n\n[workspace]\n')
        self.upstream_list("[0x11; 32]")
        self.git(self.up, "init", "-q")
        self.git(self.up, "config", "maintenance.auto", "false")
        self.git(self.up, "add", "-A")
        self.git(self.up, "commit", "-qm", "one")
        self.first = self.git(self.up, "rev-parse", "HEAD")
        (self.up / "README").write_text("two\n")
        self.git(self.up, "add", "-A")
        self.git(self.up, "commit", "-qm", "two")
        self.rev = self.git(self.up, "rev-parse", "HEAD")
        self.git(self.up, "remote", "add", "origin", str(bare))
        self.git(self.up, "pu" + "sh", "-q", "origin", "HEAD:refs/heads/main")
        self.git(self.up, "fetch", "-q", "origin")
        # cargo fetches the rev from the clone itself; the checker reads the clone's origin branches
        self.url = self.up.as_uri()
        self.d = self.parent / "ALICE-Probe"
        (self.d / "src").mkdir(parents=True)
        (self.d / "tests").mkdir()
        (self.d / "src" / "lib.rs").write_text("pub use alice_zz::{PREVIOUS_SEMANTICS_IDS, SEMANTICS_ID};\n")
        self.consumer(self.rev)
        self.git(self.d, "init", "-q")
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "probe")
        self.cfg = imc.Config("probe-crate", "Cargo.toml", (), True, ("v{version}",),
                              (imc.IdSpec("probe_crate::SEMANTICS_ID", "probe_crate::PREVIOUS_SEMANTICS_IDS",
                                          (self.REF,)),))

    def tearDown(self):
        shutil.rmtree(self.parent, ignore_errors=True)

    def verify(self):
        doc = imc.rustdoc_json(self.d, self.cfg)
        return imc.PreviousVerifier(self.d, self.cfg, doc, self.target)(self.cfg.ids[0], "11" * 32)

    def test_the_old_value_is_taken_out_of_the_pinned_rev(self):
        # the shared clone's HEAD is elsewhere: the checker neither reads nor moves it
        self.git(self.up, "checkout", "-q", "--detach", self.first)
        before = (self.up / "src" / "lib.rs").read_text()
        passed, problems = self.verify()
        self.assertEqual(problems, [])
        self.assertEqual(passed, [self.REF])
        self.assertEqual(self.git(self.up, "rev-parse", "HEAD"), self.first)
        self.assertEqual((self.up / "src" / "lib.rs").read_text(), before)
        self.assertEqual(sorted(p.name for p in self.parent.iterdir()), ["ALICE-Probe", "ALICE-Zz", "remote"])

    def test_a_renamed_git_dependency_is_followed(self):
        # the dependency is taken under another name (`key = { package = ..., git, rev }`) and the
        # list is re-exported through that name: the checker finds the git source by the rename too
        (self.d / "Cargo.toml").write_text(
            '[package]\nname = "probe-crate"\nversion = "0.6.0"\nedition = "2021"\n\n[dependencies]\n'
            f'zz_alias = {{ package = "alice-zz", git = "{self.url}", rev = "{self.rev}", version = "0.11" }}\n\n'
            "[workspace]\n")
        (self.d / "src" / "lib.rs").write_text("pub use zz_alias::{PREVIOUS_SEMANTICS_IDS, SEMANTICS_ID};\n")
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "renamed")
        passed, problems = self.verify()
        self.assertEqual(problems, [])
        self.assertEqual(passed, [self.REF])

    def test_a_dependency_only_in_its_remote_is_fetched_from_its_url(self):
        # no clone of the dependency next to the repository or in the home directory (a CI runner):
        # the checker reads the rev from the dependency's own url, and the copy built with it
        # resolves the package there, not at ../<Repo>
        bare = (self.parent / "remote" / "ALICE-Zz.git").as_uri()
        shutil.rmtree(self.up)
        self.consumer(self.rev)
        (self.d / "Cargo.toml").write_text((self.d / "Cargo.toml").read_text().replace(self.url, bare))
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "remote only")
        passed, problems = self.verify()
        self.assertEqual(problems, [])
        self.assertEqual(passed, [self.REF])
        self.assertEqual(sorted(p.name for p in self.parent.iterdir()), ["ALICE-Probe", "remote"])

    def test_a_dependency_only_in_its_remote_still_needs_a_test_that_depends_on_the_old_value(self):
        bare = (self.parent / "remote" / "ALICE-Zz.git").as_uri()
        shutil.rmtree(self.up)
        self.consumer(self.rev, "let _ = probe_crate::PREVIOUS_SEMANTICS_IDS;")
        (self.d / "Cargo.toml").write_text((self.d / "Cargo.toml").read_text().replace(self.url, bare))
        self.git(self.d, "add", "-A")
        self.git(self.d, "commit", "-qm", "remote only, weak test")
        passed, problems = self.verify()
        self.assertEqual(passed, [])
        self.assertTrue(problems, "a test that passes without the old value is refused")

    def test_a_dependency_whose_url_cannot_be_fetched_fails_closed(self):
        # no clone anywhere and a url that does not answer (as a private repository without
        # credentials): a problem naming the fetch, not a pass and not a skipped comparison
        # (cargo itself reached the dependency, as it does with its own credentials: the rustdoc JSON
        # is built first, then the manifest names a url the checker cannot fetch)
        doc = imc.rustdoc_json(self.d, self.cfg)
        shutil.rmtree(self.up)
        gone = (self.parent / "no-such-remote" / "ALICE-Zz.git").as_uri()
        (self.d / "Cargo.toml").write_text((self.d / "Cargo.toml").read_text().replace(self.url, gone))
        passed, problems = imc.PreviousVerifier(self.d, self.cfg, doc, self.target)(self.cfg.ids[0], "11" * 32)
        self.assertEqual(passed, [])
        self.assertTrue(any("fetching ALICE-Zz failed" in p for p in problems), problems)

    def test_a_rev_not_on_an_origin_branch_fails(self):
        (self.up / "NOTES").write_text("local only\n")
        self.git(self.up, "add", "-A")
        self.git(self.up, "commit", "-qm", "local")
        self.consumer(self.git(self.up, "rev-parse", "HEAD"))
        passed, problems = self.verify()
        self.assertEqual(passed, [])
        self.assertTrue(any("not on an origin branch" in p for p in problems), problems)

    def test_a_test_that_does_not_depend_on_the_old_value_fails(self):
        self.consumer(self.rev, "let _ = probe_crate::PREVIOUS_SEMANTICS_IDS;")
        passed, problems = self.verify()
        self.assertEqual(passed, [])
        self.assertTrue(any("still passes" in p for p in problems), problems)


@unittest.skipUnless(shutil.which("cargo") and shutil.which("git"), "needs cargo and git")
class BaseBeforePins(unittest.TestCase):
    """A base tree from before the repository had .github/alice-siblings: each sibling it reaches
    by path is taken at the newest commit of its origin main (first parent) whose committer date is
    not later than the base commit's, what CI cloned when the base was built"""

    def env(self, date: str) -> dict:
        # no system or global git config: a host's init.defaultBranch (or anything else) must not
        # decide what the fixture repositories look like
        return dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                    GIT_COMMITTER_EMAIL="t@t", GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date,
                    GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    @classmethod
    def setUpClass(cls):
        cls.target = Path(tempfile.mkdtemp(prefix="id-move-prepin-target-"))
        os.environ["ID_MOVE_TARGET_DIR"] = str(cls.target)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ID_MOVE_TARGET_DIR", None)
        shutil.rmtree(cls.target, ignore_errors=True)

    def git(self, d: Path, *args: str, date: str = "2026-01-01T00:00:00Z") -> str:
        return subprocess.run(["git", "-C", str(d), *args], check=True, env=self.env(date),
                              capture_output=True, text=True).stdout.strip()

    def up_commit(self, version: str, date: str) -> str:
        (self.up / "Cargo.toml").write_text(
            f'[package]\nname = "alice-up"\nversion = "{version}"\nedition = "2021"\n\n[workspace]\n')
        self.git(self.up, "add", "-A", date=date)
        self.git(self.up, "commit", "-qm", version, date=date)
        self.git(self.up, "pu" + "sh", "-q", "origin", "HEAD:refs/heads/main", date=date)
        return self.git(self.up, "rev-parse", "HEAD")

    def probe(self, version: str, up_req: str):
        (self.d / "Cargo.toml").write_text(
            f'[package]\nname = "probe-crate"\nversion = "{version}"\nedition = "2021"\n\n[dependencies]\n'
            f'alice-up = {{ path = "../ALICE-Up", version = "{up_req}" }}\n\n[workspace]\n')

    def release(self, date: str):
        self.git(self.d, "add", "-A", date=date)
        self.git(self.d, "commit", "-qm", "release", date=date)
        self.git(self.d, "tag", "-f", "v0.3.0", date=date)

    def setUp(self):
        self.parent = Path(tempfile.mkdtemp(prefix="id-move-prepin-"))
        bare = self.parent / "remote" / "ALICE-Up.git"
        bare.mkdir(parents=True)
        self.git(bare, "init", "-q", "--bare", "-b", "main")
        self.up = self.parent / "ALICE-Up"
        (self.up / "src").mkdir(parents=True)
        (self.up / "src" / "lib.rs").write_text("pub fn f() {}\n")
        self.git(self.up, "init", "-q", "-b", "main")
        self.git(self.up, "config", "maintenance.auto", "false")
        self.git(self.up, "remote", "add", "origin", str(bare))
        self.v1 = self.up_commit("1.0.0", "2026-02-01T00:00:00Z")
        self.d = self.parent / "ALICE-Probe"
        (self.d / "src").mkdir(parents=True)
        (self.d / "scripts").mkdir()
        (self.d / ".github").mkdir()
        (self.d / "scripts" / "id_move.toml").write_text(
            'package = "probe-crate"\nmanifest = "Cargo.toml"\ntags = ["v{version}"]\n'
            '[[id]]\npath = "probe_crate::SEMANTICS_ID"\n')
        (self.d / "src" / "lib.rs").write_text("pub const SEMANTICS_ID: [u8; 32] = [0x11; 32];\n")
        (self.d / "CHANGELOG.md").write_text("# Changelog\n")
        self.probe("0.3.0", "^1")
        self.git(self.d, "init", "-q", "-b", "main")
        self.release("2026-03-01T00:00:00Z")
        # later: the sibling takes a major version, and the working tree follows it with a pin
        self.v2 = self.up_commit("2.0.0", "2026-04-01T00:00:00Z")
        self.probe("0.4.0", "^2")
        (self.d / ".github" / "alice-siblings").write_text(f"ALICE-Up {self.v2}\n")

    def tearDown(self):
        shutil.rmtree(self.parent, ignore_errors=True)

    def test_each_sibling_is_taken_at_its_main_as_of_the_base_commit(self):
        problems, report = imc.check(self.d, None, "tag")
        self.assertEqual(problems, [])
        self.assertTrue(any(f"../ALICE-Up at {self.v1[:12]}" in r and "base predates pins" in r for r in report),
                        report)
        self.assertEqual(sorted(p.name for p in self.parent.iterdir()), ["ALICE-Probe", "ALICE-Up", "remote"])

    def test_the_summary_counts_bases_not_sibling_lines(self):
        import contextlib
        import io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(imc.main(["--root", str(self.d), "--base-from", "tag"]), 0)
        out = buf.getvalue()
        self.assertIn("sibling ../ALICE-Up", out)
        self.assertIn("against 1 base(s), 0 problems", out)

    def test_a_base_whose_history_has_the_pin_file_but_not_at_the_base_fails(self):
        (self.d / ".github" / "alice-siblings").write_text(f"ALICE-Up {self.v1}\n")
        self.git(self.d, "add", "-A", date="2026-02-15T00:00:00Z")
        self.git(self.d, "commit", "-qm", "pins", date="2026-02-15T00:00:00Z")
        (self.d / ".github" / "alice-siblings").unlink()
        self.probe("0.3.0", "^1")
        self.release("2026-03-02T00:00:00Z")
        self.probe("0.4.0", "^2")
        (self.d / ".github" / "alice-siblings").write_text(f"ALICE-Up {self.v2}\n")
        with self.assertRaises(imc.CheckError) as e:
            imc.check(self.d, None, "tag")
        self.assertIn("broken base", str(e.exception))

    def test_a_base_after_a_removal_fails_even_when_the_file_is_added_again_later(self):
        # add -> remove -> base -> add again: the newest add is not an ancestor of the base, but
        # the base's history had the file, so the fallback does not apply
        (self.d / ".github" / "alice-siblings").write_text(f"ALICE-Up {self.v1}\n")
        self.git(self.d, "add", "-A", date="2026-02-15T00:00:00Z")
        self.git(self.d, "commit", "-qm", "pins", date="2026-02-15T00:00:00Z")
        (self.d / ".github" / "alice-siblings").unlink()
        self.probe("0.3.0", "^1")
        self.release("2026-03-02T00:00:00Z")
        self.probe("0.4.0", "^2")
        (self.d / ".github" / "alice-siblings").write_text(f"ALICE-Up {self.v2}\n")
        self.git(self.d, "add", "-A", date="2026-04-02T00:00:00Z")
        self.git(self.d, "commit", "-qm", "pins again", date="2026-04-02T00:00:00Z")
        with self.assertRaises(imc.CheckError) as e:
            imc.check(self.d, None, "tag")
        self.assertIn("broken base", str(e.exception))

    def test_a_pin_file_only_on_a_merged_side_branch_falls_back_by_date(self):
        # the file was added and removed on a side branch merged before the base; main's
        # first-parent line never had it, so CI built the base without pins: the date fallback
        pin = self.d / ".github" / "alice-siblings"
        pin.unlink()
        self.probe("0.3.0", "^1")
        self.git(self.d, "checkout", "-q", "-b", "side")
        pin.write_text(f"ALICE-Up {self.v1}\n")
        self.git(self.d, "add", "-A", date="2026-03-02T00:00:00Z")
        self.git(self.d, "commit", "-qm", "pins on a side branch", date="2026-03-02T00:00:00Z")
        pin.unlink()
        self.git(self.d, "add", "-A", date="2026-03-03T00:00:00Z")
        self.git(self.d, "commit", "-qm", "pins removed on the side branch", date="2026-03-03T00:00:00Z")
        self.git(self.d, "checkout", "-q", "-")
        self.git(self.d, "merge", "-q", "--no-ff", "-m", "merge side", "side", date="2026-03-04T00:00:00Z")
        self.git(self.d, "tag", "-f", "v0.3.0", date="2026-03-04T00:00:00Z")
        self.probe("0.4.0", "^2")
        pin.write_text(f"ALICE-Up {self.v2}\n")
        problems, report = imc.check(self.d, None, "tag")
        self.assertEqual(problems, [])
        self.assertTrue(any(f"../ALICE-Up at {self.v1[:12]}" in r and "base predates pins" in r for r in report),
                        report)

    def test_the_sibling_is_read_on_its_first_parent_line_not_a_merged_side_branch(self):
        # a side-branch commit dated before the base, merged into main after it: main as of the
        # base date was v1, not the side commit
        self.git(self.up, "checkout", "-q", "-b", "side", self.v1)
        (self.up / "src" / "side.rs").write_text("// side\n")
        self.git(self.up, "add", "-A", date="2026-02-20T00:00:00Z")
        self.git(self.up, "commit", "-qm", "side", date="2026-02-20T00:00:00Z")
        side = self.git(self.up, "rev-parse", "HEAD")
        self.git(self.up, "checkout", "-q", "main")
        self.git(self.up, "merge", "-q", "--no-ff", "-m", "merge side", "side", date="2026-04-10T00:00:00Z")
        self.git(self.up, "pu" + "sh", "-q", "origin", "HEAD:refs/heads/main", date="2026-04-10T00:00:00Z")
        problems, report = imc.check(self.d, None, "tag")
        self.assertEqual(problems, [])
        self.assertTrue(any(f"../ALICE-Up at {self.v1[:12]}" in r for r in report), report)
        self.assertFalse(any(f"../ALICE-Up at {side[:12]}" in r for r in report), report)

    def test_a_shallow_sibling_fails(self):
        shutil.rmtree(self.up)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "main",
                        (self.parent / "remote" / "ALICE-Up.git").as_uri(), str(self.up)],
                       check=True, capture_output=True, env=self.env("2026-05-01T00:00:00Z"))
        # the sibling really is a shallow clone with its tree checked out, so the check below can
        # only fail for the reason this test names
        self.assertEqual(self.git(self.up, "rev-parse", "--is-shallow-repository"), "true")
        self.assertTrue((self.up / "Cargo.toml").is_file())
        with self.assertRaises(imc.CheckError) as e:
            imc.check(self.d, None, "tag")
        self.assertIn("shallow", str(e.exception))

    def test_no_sibling_commit_before_the_base_fails(self):
        # the release commit dated before the sibling's first commit
        self.git(self.d, "commit", "-q", "--amend", "--no-edit", "--only", date="2026-01-15T00:00:00Z")
        self.git(self.d, "tag", "-f", "v0.3.0", date="2026-01-15T00:00:00Z")
        with self.assertRaises(imc.CheckError) as e:
            imc.check(self.d, None, "tag")
        self.assertIn("no commit of its origin main", str(e.exception))


def main() -> int:
    """the suite inside a temporary directory of its own (also the $TMPDIR of what it runs): a test
    that leaves a file or directory behind fails the run, naming it (the suite runs in every ALICE
    crate that carries this file, so a leak grows each machine's temporary directory without end)"""
    scratch = Path(tempfile.mkdtemp(prefix="id-move-tests-"))
    old_tempdir, old_env = tempfile.tempdir, os.environ.get("TMPDIR")
    tempfile.tempdir = str(scratch)
    os.environ["TMPDIR"] = str(scratch)
    try:
        ok = unittest.main(exit=False).result.wasSuccessful()
        left = sorted(p.name for p in scratch.iterdir())
    finally:
        tempfile.tempdir = old_tempdir
        if old_env is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_env
        shutil.rmtree(scratch, ignore_errors=True)
    if left:
        print(f"test_id_move_check: the tests left {len(left)} entries in the temporary directory: "
              f"{', '.join(left[:10])}", file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
