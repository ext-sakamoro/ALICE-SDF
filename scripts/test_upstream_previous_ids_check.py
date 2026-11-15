#!/usr/bin/env python3
"""oracles of upstream_previous_ids_check.py: another crate's PREVIOUS_*_SEMANTICS_IDS is never indexed by position

An upstream crate empties its list of previous identifiers on a major fold (`&[]` is a correct state); a downstream
test that reads `upstream::PREVIOUS_X_SEMANTICS_IDS[0]` then panics with index out of bounds on every OS. The checker
refuses every positional read of another crate's list; iterating it, `contains`, `len` and `is_empty` stay allowed.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import upstream_previous_ids_check as up  # noqa: E402

LIST = "alice_physics::PREVIOUS_PHYSICS_SEMANTICS_IDS"


def findings(src: str, own: set[str] = frozenset({"alice_edge"})) -> list[str]:
    return [f.kind for f in up.scan_text("src/x.rs", src, set(own))]


class Positional(unittest.TestCase):
    def test_indexing_another_crates_list_is_refused(self):
        self.assertEqual(findings(f"let a = {LIST}[0];"), ["index"])
        self.assertEqual(findings(f"let a = {LIST} [ 1 ];"), ["index"])
        self.assertEqual(findings(f"let a = &{LIST}[1..];"), ["index"])
        self.assertEqual(findings(f"let a = ::{LIST}[0];"), ["index"])

    def test_every_positional_accessor_is_refused(self):
        for tail, kind in ((".get(0)", "get"), (".first()", "first"), (".last()", "last"),
                           (".iter().nth(0)", "nth"), (".iter().next()", "next"), (".split_first()", "first"),
                           (".iter().rev().next()", "next"), (".iter().rev().nth(1)", "nth")):
            self.assertEqual(findings(f"let a = {LIST}{tail};"), [kind], tail)

    def test_more_positional_forms(self):
        for src, kind in ((f"if let [a, ..] = {LIST} {{}}", "pattern"), (f"let [a, b] = *&{LIST};", "pattern"),
                          (f"match {LIST} {{ [a, ..] => a, [] => x }}", "pattern"),
                          (f"let a = {LIST}.iter().take(1);", "take"), (f"let a = {LIST}.iter().skip(1).next();", "skip"),
                          (f"let a = {LIST}.into_iter().next();", "next"), (f"let a = {LIST}.split_at(1);", "split"),
                          (f"let a = {LIST}.iter().copied().last();", "next"),
                          (f"let a = {LIST}\n    .iter()\n    .next();", "next"), (f"assert_eq!({LIST}[0], x);", "index")):
            got = findings(src)
            self.assertEqual(len(got), 1, src)
            if kind in ("pattern", "index"):
                self.assertEqual(got, [kind], src)
        self.assertEqual(findings(f"let [a, b] = {LIST};"), ["pattern"])
        # a renamed dependency is another crate too
        self.assertEqual(findings("let a = phys::PREVIOUS_PHYSICS_SEMANTICS_IDS[0];"), ["index"])

    def test_the_verifiers_missed_forms_are_refused(self):
        for src in (f"let a = {LIST}.iter().take(1);", f"for x in {LIST}.iter().take(1) {{}}",
                    f"let a = {LIST}.iter().skip(0).next();", f"let a = {LIST}.into_iter().next();",
                    f"let a = {LIST}.split_at(1);", f"if let [a, ..] = {LIST} {{}}",
                    f"match {LIST} {{ [x, ..] => x, [] => y }}", f"let v = {LIST}.to_vec();\nlet a = v[0];",
                    f"let v: Vec<_> = {LIST}.iter().copied().collect();\nlet a = v.first();",
                    f"let p = {LIST};\nlet q = p;\nlet a = q[0];", f"let a = {LIST}.iter().peekable();",
                    f"let a = {LIST} == other;"):
            self.assertTrue(findings(src), src)

    def test_whole_uses_through_iterators_and_bindings_are_allowed(self):
        for src in (f"assert!({LIST}.iter().any(|x| x == &id));", f"let n = {LIST}.iter().filter(|x| ok(x)).count();",
                    f"let v: Vec<_> = {LIST}.iter().map(|x| hex(x)).collect();", f"assert_eq!({LIST}, &[]);",
                    f"accept(&{LIST}, cur);", f"let v = {LIST}.to_vec();\nfor x in &v {{}}",
                    f"let p = {LIST};\nassert!(p.contains(&id));", f"for (i, x) in {LIST}.iter().enumerate() {{}}",
                    f"const UP: &[[u8; 32]] = {LIST};\nlet n = UP.len();"):
            self.assertEqual(findings(src), [], src)

    def test_a_re_export_under_the_own_crate_is_followed(self):
        texts = {"src/ids.rs": f"pub use {LIST} as UPSTREAM_PREVIOUS;\n",
                 "src/lib.rs": "let a = crate::ids::UPSTREAM_PREVIOUS[0];\nlet b = crate::ids::UPSTREAM_PREVIOUS.len();\n"}
        rex = up.reexports(texts, {"alice_edge"})
        self.assertEqual(rex, {"UPSTREAM_PREVIOUS"})
        self.assertEqual([f.kind for f in up.scan_text("src/lib.rs", texts["src/lib.rs"], {"alice_edge"}, rex)],
                         ["index"])
        self.assertEqual(up.reexports({"a.rs": "pub use crate::PREVIOUS_EDGE_SEMANTICS_IDS as P;"}, {"alice_edge"}),
                         set())

    def test_iteration_membership_and_size_are_allowed(self):
        for src in (f"for id in {LIST} {{ accept(id); }}", f"for id in {LIST}.iter() {{}}",
                    f"assert!({LIST}.contains(&x));", f"let n = {LIST}.len();", f"if {LIST}.is_empty() {{}}",
                    f"let v: Vec<_> = {LIST}.iter().copied().collect();", f"check(&cur, {LIST});"):
            self.assertEqual(findings(src), [], src)

    def test_the_own_crates_list_is_not_checked(self):
        for src in ("let a = PREVIOUS_EDGE_SEMANTICS_IDS[0];", "let a = crate::PREVIOUS_EDGE_SEMANTICS_IDS[0];",
                    "let a = self::PREVIOUS_EDGE_SEMANTICS_IDS[0];", "let a = super::ids::PREVIOUS_EDGE_SEMANTICS_IDS[0];",
                    "let a = alice_edge::PREVIOUS_EDGE_SEMANTICS_IDS[0];",
                    "let a = alice_edge::law::PREVIOUS_EDGE_SEMANTICS_IDS.first();"):
            self.assertEqual(findings(src), [], src)
        # a workspace member is the same repository: its CI sees its own list
        self.assertEqual(findings(f"let a = {LIST}[0];", {"alice_edge", "alice_physics"}), [])

    def test_an_imported_name_and_its_alias_are_followed(self):
        self.assertEqual(findings(f"use {LIST};\nlet a = PREVIOUS_PHYSICS_SEMANTICS_IDS[0];"), ["index"])
        self.assertEqual(findings(f"use {LIST} as PREV;\nlet a = PREV.first();"), ["first"])
        self.assertEqual(findings("use alice_physics::{law::LawRef, PREVIOUS_PHYSICS_SEMANTICS_IDS as P};\n"
                                  "let a = P[0];"), ["index"])
        self.assertEqual(findings("use alice_physics::*;\nlet a = PREVIOUS_PHYSICS_SEMANTICS_IDS[0];"), ["index"])
        # a local of the same name that is not the import: an own list defined here
        self.assertEqual(findings("pub const PREVIOUS_PHYSICS_SEMANTICS_IDS: &[[u8; 32]] = &[];\n"
                                  "let a = PREVIOUS_PHYSICS_SEMANTICS_IDS[0];"), [])

    def test_a_let_bound_list_is_followed(self):
        self.assertEqual(findings(f"let prev = {LIST};\nlet a = prev[0];"), ["index"])
        self.assertEqual(findings(f"let prev: &[[u8; 32]] = &{LIST};\nlet a = prev.first();"), ["first"])
        self.assertEqual(findings(f"let prev = {LIST};\nfor id in prev {{}}"), [])
        self.assertEqual(findings("let prev = own_list();\nlet a = prev[0];"), [])

    def test_comments_and_strings_do_not_count_but_doc_examples_do(self):
        self.assertEqual(findings(f"// was {LIST}[0]\nlet a = 1;"), [])
        self.assertEqual(findings(f'let s = "{LIST}[0]";'), [])
        self.assertEqual(findings(f"/* {LIST}[0] */ let a = 1;"), [])
        self.assertEqual(findings(f"/// ```\n/// let a = {LIST}[0];\n/// ```\npub fn f() {{}}"), ["index"])  # a doctest runs
        # prose in a doc comment names the list; it is not code
        self.assertEqual(findings(f"/// one of [`{LIST}`] or `{LIST}[0]`\npub fn f() {{}}"), [])
        self.assertEqual(findings(f"//! listed in `{LIST}`, see\n//! ```\n//! assert!({LIST}.contains(&x));\n//! ```\n"), [])

    def test_the_finding_names_the_line_and_the_fix(self):
        fs = up.scan_text("tests/t.rs", f"fn a() {{}}\n\nlet x = {LIST}[0];\n", {"alice_edge"})
        self.assertEqual([(f.path, f.line) for f in fs], [("tests/t.rs", 3)])
        text = up.render(fs[0])
        self.assertIn("tests/t.rs:3", text)
        self.assertIn("contains", text)


class Cli(unittest.TestCase):
    def repo(self, files: dict[str, str]) -> Path:
        d = Path(tempfile.mkdtemp(prefix="prev-ids-"))
        self.addCleanup(subprocess.run, ["rm", "-rf", str(d)])
        subprocess.run(["git", "-C", str(d), "init", "-q"], check=True)
        for k, v in files.items():
            p = d / k
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(v)
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        return d

    def run_check(self, d: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(HERE / "upstream_previous_ids_check.py"), "--repo", str(d)],
                              capture_output=True, text=True)

    def test_a_finding_fails_and_a_clean_tree_passes_with_its_count(self):
        manifest = '[package]\nname = "alice-edge"\nversion = "0.1.0"\n'
        bad = self.repo({"Cargo.toml": manifest, "src/lib.rs": f"pub fn f() {{ let _ = {LIST}[0]; }}\n"})
        r = self.run_check(bad)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("src/lib.rs:1", r.stderr)
        good = self.repo({"Cargo.toml": manifest, "src/lib.rs": f"pub fn f() -> usize {{ {LIST}.len() }}\n"})
        r = self.run_check(good)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1 file", r.stderr)

    def test_no_rust_file_scanned_fails(self):
        d = self.repo({"README.md": "x\n"})
        r = self.run_check(d)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("0 files", r.stderr)

    def test_a_path_through_an_own_module_is_own(self):
        manifest = '[package]\nname = "alice-edge"\nversion = "0.1.0"\n'
        d = self.repo({"Cargo.toml": manifest, "src/lib.rs": "mod ids;\npub use ids::PREVIOUS_EDGE_SEMANTICS_IDS;\n"
                       "pub fn f() -> [u8; 32] { ids::PREVIOUS_EDGE_SEMANTICS_IDS[0] }\n",
                       "src/ids.rs": "pub const PREVIOUS_EDGE_SEMANTICS_IDS: &[[u8; 32]] = &[[0; 32]];\n"})
        self.assertEqual(self.run_check(d).returncode, 0, self.run_check(d).stderr)

    def test_workspace_members_are_own_crates(self):
        d = self.repo({"Cargo.toml": '[workspace]\nmembers = ["crates/*"]\n',
                       "crates/core/Cargo.toml": '[package]\nname = "alice-physics"\nversion = "1.0.0"\n',
                       "crates/app/Cargo.toml": '[package]\nname = "alice-app"\nversion = "1.0.0"\n',
                       "crates/app/src/main.rs": f"fn main() {{ let _ = {LIST}[0]; }}\n"})
        self.assertEqual(up.own_crates(d), {"alice_physics", "alice_app"})
        self.assertEqual(self.run_check(d).returncode, 0)


if __name__ == "__main__":
    unittest.main()
