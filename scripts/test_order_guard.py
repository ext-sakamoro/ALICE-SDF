#!/usr/bin/env python3
"""Tests of scripts/order_guard.py on throwaway trees."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import order_guard  # noqa: E402

PLANTS = {
    "HashMap": "let m: std::collections::HashMap<u32, f64> = Default::default();",
    "HashSet": "let s = std::collections::HashSet::<u32>::new();",
    "parallel": "let t: f64 = xs.par_iter().sum();",
    "threads": "let h = std::thread::spawn(|| 1);",
    "address order": "let a = p as *const u8 as usize;",
}


class OrderGuardTest(unittest.TestCase):
    def tree(self, files: dict[str, str]) -> Path:
        d = Path(tempfile.mkdtemp())
        (d / "src").mkdir()
        for name, text in files.items():
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return d

    def test_clean_tree_passes(self):
        d = self.tree({"src/lib.rs": "pub fn f(m: &std::collections::BTreeMap<u32, f64>) -> f64 { 0.0 }\n"})
        self.assertEqual(order_guard.check(d)[0], [])

    def test_every_pattern_is_found(self):
        self.assertEqual(set(PLANTS), set(order_guard.PATTERNS))
        for name, line in PLANTS.items():
            problems, _, _ = order_guard.check(self.tree({"src/lib.rs": f"fn f() {{ {line} }}\n"}))
            self.assertTrue(any(f"src/lib.rs::{name}" in p for p in problems), (name, problems))

    def test_listed_site_passes_and_drift_fails(self):
        src = {"src/store.rs": "use std::collections::HashMap;\npub struct S { m: HashMap<u32, u32> }\n"}
        h1 = order_guard.line_hash("use std::collections::HashMap;")
        h2 = order_guard.line_hash("pub struct S { m: HashMap<u32, u32> }")
        ok = ('[allow]\n"src/store.rs::HashMap" = { count = 2, '
              f'lines = ["{h1}", "{h2}"], reason = "keyed lookup only, never iterated" }}\n')
        self.assertEqual(order_guard.check(self.tree(dict(src, **{"scripts/order_allowlist.toml": ok})))[0], [])
        wrong = ok.replace("count = 2", "count = 1")
        problems = order_guard.check(self.tree(dict(src, **{"scripts/order_allowlist.toml": wrong})))[0]
        self.assertTrue(any("says 1" in p for p in problems), problems)
        stale = ok + '"src/gone.rs::HashSet" = { count = 1, reason = "removed some time ago" }\n'
        problems = order_guard.check(self.tree(dict(src, **{"scripts/order_allowlist.toml": stale})))[0]
        self.assertTrue(any("stale" in p for p in problems), problems)

    def test_a_rewritten_allowed_line_fails_until_its_hash_is_listed(self):
        # the count stays the same, the content does not: an ordered collect turned into
        # appends in completion order on the one allowed line
        before = "pub fn f(v: &[u32]) -> Vec<u32> { v.par_iter().map(|x| x + 1).collect() }\n"
        after = ("pub fn f(v: &[u32]) -> Vec<u32> { let m = std::sync::Mutex::new(Vec::new()); "
                 "v.par_iter().for_each(|x| m.lock().unwrap().push(x + 1)); m.into_inner().unwrap() }\n")
        allow = ('[allow]\n"src/lib.rs::parallel" = { count = 1, '
                 f'lines = ["{order_guard.line_hash(before)}"], reason = "indexed collect keeps the order" }}\n')
        self.assertEqual(order_guard.check(self.tree({"src/lib.rs": before, "scripts/order_allowlist.toml": allow}))[0], [])
        problems = order_guard.check(self.tree({"src/lib.rs": after, "scripts/order_allowlist.toml": allow}))[0]
        self.assertTrue(any("not the reviewed ones" in p for p in problems), problems)
        # an entry without lines fails as well
        bare = allow.replace(f'lines = ["{order_guard.line_hash(before)}"], ', "")
        problems = order_guard.check(self.tree({"src/lib.rs": before, "scripts/order_allowlist.toml": bare}))[0]
        self.assertTrue(any("not the reviewed ones" in p for p in problems), problems)

    def test_every_rayon_parallel_method_is_found(self):
        for call in ("v.par_windows(2).map(|w| w[0] * w[1]).sum::<f64>()", "v.par_chunks_exact(4).count()",
                     "v.par_sort_unstable()", "v.par_drain(..).count()", "v.par_split(|x| *x == 0.0).count()",
                     "s.into_par_iter().count()", "it.par_bridge().count()",
                     # through the trait, no dot and no rayon token
                     "ParallelIterator::sum::<f64>(ParallelSlice::par_windows(v, 2).map(|w| w[0] * w[1]))",
                     "ParallelIterator::sum::<f64>(IntoParallelRefIterator::par_iter(v).copied())",
                     "IndexedParallelIterator::len(&it)", "<Vec<f64> as FromParallelIterator<f64>>::from_par_iter(it)"):
            problems = order_guard.check(self.tree({"src/lib.rs": f"pub fn f() {{ let _ = {call}; }}\n"}))[0]
            self.assertTrue(any("src/lib.rs::parallel" in p for p in problems), (call, problems))

    def test_import_forms_of_threads_and_hash_modules_are_found(self):
        for line in ("use std::thread::{self as t};", "use std::thread::{spawn};", "use std::thread::*;",
                     "use std::{thread as t};", "use std::*;", "use std::thread as t;",
                     "mod zz { pub use std::thread as pool; }", "fn f(i: std::collections::hash_set::Iter<u64>) {}",
                     "fn f(i: std::collections::hash_map::Keys<u8, u8>) {}"):
            problems, _, _ = order_guard.check(self.tree({"src/lib.rs": line + "\n"}))
            self.assertTrue(problems, line)

    def test_sleeping_the_calling_thread_is_not_a_combination_order(self):
        d = self.tree({"src/lib.rs": "fn f() { std::thread::sleep(std::time::Duration::from_millis(1)); }\n"})
        self.assertEqual(order_guard.check(d)[0], [])

    def test_an_order_dependency_fails_unless_listed(self):
        cargo = '[package]\nname = "x"\n[dependencies]\nrayon = "1"\nfast = { package = "hashbrown", version = "0.14" }\n'
        problems, _, _ = order_guard.check(self.tree({"src/lib.rs": "\n", "Cargo.toml": cargo}))
        self.assertTrue(any("Cargo.toml::rayon" in p for p in problems), problems)
        self.assertTrue(any("Cargo.toml::hashbrown" in p for p in problems), problems)

    def test_test_modules_and_comments_are_not_production(self):
        text = ("#[cfg(test)]\nmod tests {\n    fn f() { let m = std::collections::HashMap::<u8, u8>::new(); }\n}\n"
                "// HashMap in a comment\npub const DOC: &str = \"HashSet in a string\";\n")
        self.assertEqual(order_guard.check(self.tree({"src/lib.rs": text}))[0], [])

    def test_no_source_file_fails(self):
        d = Path(tempfile.mkdtemp())
        (d / "src").mkdir()
        self.assertEqual(order_guard.main(["--root", str(d)]), 1)


if __name__ == "__main__":
    unittest.main()
