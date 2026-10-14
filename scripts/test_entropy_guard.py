#!/usr/bin/env python3
"""Tests for scripts/entropy_guard.py: the repository passes, each entropy or
clock source in production code is reported, test modules are exempt, and the
allowed counts are exact."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entropy_guard as eg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="entropy-guard-"))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


class EntropyGuard(unittest.TestCase):
    def setUp(self):
        self._allowed = dict(eg.ALLOWED)
        eg.ALLOWED.clear()

    def tearDown(self):
        eg.ALLOWED.clear()
        eg.ALLOWED.update(self._allowed)

    def problems(self, files):
        return eg.check(tree(files))[0]

    def test_the_repository_passes(self):
        eg.ALLOWED.update(self._allowed)
        problems, files, patterns = eg.check(ROOT)
        self.assertEqual(problems, [])
        self.assertGreater(files, 10)
        self.assertEqual(patterns, len(eg.PATTERNS))

    def test_every_source_in_production_code_is_reported(self):
        for call in ["getrandom::getrandom(&mut b)", "rand::rngs::OsRng", "rand::thread_rng()",
                     "rand::random::<u64>()", "SecureRng::from_entropy()", "DpNoise::try_from_entropy(1.0, 1.0)",
                     "std::time::SystemTime::now()", "std::time::Instant::now()",
                     "std::collections::hash_map::RandomState::new()", "uuid::Uuid::new_v4()"]:
            p = self.problems({"src/a.rs": f"pub fn f() {{ let _ = {call}; }}\n"})
            self.assertEqual(len(p), 1, (call, p))

    def test_a_test_module_is_exempt(self):
        src = "pub fn f() {}\n\n#[cfg(test)]\nmod tests {\n    fn t() {\n        let _ = std::time::Instant::now();\n    }\n}\n"
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_cfg_not_test_is_not_exempt(self):
        # `not(test)` contains the word "test" but means the opposite of it;
        # a regex keyed on the substring "test" would wrongly treat this as a
        # test module and hide a real production call.
        src = ('pub fn f() {}\n\n#[cfg(not(test))]\nmod prod_only {\n'
               '    fn t() {\n        let _ = std::time::Instant::now();\n    }\n}\n')
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x for x in p), p)

    def test_code_after_a_test_module_is_checked_again(self):
        src = "#[cfg(test)]\nmod tests {\n    fn t() {}\n}\n\npub fn f() { let _ = rand::thread_rng(); }\n"
        p = self.problems({"src/a.rs": src})
        self.assertEqual(len(p), 1, p)
        self.assertIn("src/a.rs:6", p[0])

    def test_a_definition_strings_and_comments_are_not_calls(self):
        src = ('pub fn from_entropy() -> u64 { 0 }\n'
               'pub const S: &str = "thread_rng()";\n'
               '// SystemTime::now() is not used here\n')
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_an_allowed_site_needs_its_exact_count(self):
        src = "pub fn f() { let _ = std::time::SystemTime::now(); }\n"
        eg.ALLOWED[("src/a.rs", "SystemTime::now")] = (1, "reason")
        self.assertEqual(self.problems({"src/a.rs": src}), [])
        eg.ALLOWED[("src/a.rs", "SystemTime::now")] = (2, "reason")
        self.assertTrue(any("1 call sites, ALLOWED says 2" in x for x in self.problems({"src/a.rs": src})))

    def test_an_allowed_entry_needs_a_reason(self):
        src = "pub fn f() { let _ = std::time::SystemTime::now(); }\n"
        eg.ALLOWED[("src/a.rs", "SystemTime::now")] = (1, "  ")
        self.assertTrue(any("has no reason" in x for x in self.problems({"src/a.rs": src})))

    def test_a_stale_allowed_entry_fails(self):
        eg.ALLOWED[("src/a.rs", "OsRng")] = (1, "reason")
        p = self.problems({"src/a.rs": "pub fn f() {}\n"})
        self.assertTrue(any("matches nothing" in x for x in p), p)

    def test_an_aliased_clock_is_found(self):
        src = "use std::time::Instant as Clock;\npub fn f() { let _ = Clock::now(); }\n"
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x and "src/a.rs:2" in x for x in p), p)

    def test_a_call_split_over_lines_is_found(self):
        src = "pub fn f() {\n    let _ = std::time::Instant\n        ::now();\n}\n"
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x and "src/a.rs:2" in x for x in p), p)

    def test_random_state_new_is_found(self):
        src = "pub fn f() { let _ = std::collections::hash_map::RandomState::new(); }\n"
        self.assertTrue(any(x.startswith("RandomState::new") for x in self.problems({"src/a.rs": src})))

    def test_uuid_new_v4_is_found(self):
        src = "pub fn f() { let _ = uuid::Uuid::new_v4(); }\n"
        self.assertTrue(any(x.startswith("Uuid::new_v4") for x in self.problems({"src/a.rs": src})))

    def test_unix_epoch_elapsed_is_found(self):
        src = "pub fn f() { let _ = std::time::UNIX_EPOCH.elapsed(); }\n"
        self.assertTrue(any(x.startswith("UNIX_EPOCH") for x in self.problems({"src/a.rs": src})))

    def test_attributes_between_cfg_test_and_the_module_keep_it_exempt(self):
        src = ("pub fn f() {}\n\n#[cfg(test)]\n// timing only\n#[allow(clippy::disallowed_methods)]\n"
               "mod tests {\n    fn t() { let _ = std::time::Instant::now(); }\n}\n")
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_main_exits_1_when_it_scans_no_file(self):
        root = tree({"README.md": "x\n"})
        saved = eg.ROOT
        try:
            eg.ROOT = root
            with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
                code = eg.main()
        finally:
            eg.ROOT = saved
        self.assertEqual(code, 1)
        self.assertIn("compared nothing", err.getvalue())

    def test_no_source_file_compares_nothing(self):
        _, files, _ = eg.check(tree({"README.md": "x\n"}))
        self.assertEqual(files, 0)


if __name__ == "__main__":
    unittest.main()
