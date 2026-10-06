#!/usr/bin/env python3
"""scripts/wiring_guard.py の oracle.

検査器は「比較件数 0 でも green になる」型の空振りが最も危ないので、
(1) 配線済みで通る (2) 未配線で落ちる (3) 落ちるべき 5 つの形 (test のみ参照 /
tests/ のみ参照 / doc 言及のみ / use 行のみ / 自身の定義のみ) で落ちる
(4) マーカーと baseline の欠落・古さで落ちる (5) 検査対象 0 件で落ちる
を fixture crate で固定する。期待値は fixture の構造から決まり、検査器を呼んで作らない。

run: python3 scripts/test_wiring_guard.py
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiring_guard as wg  # noqa: E402


def crate(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="wiring-fixture-"))
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body), encoding="utf-8")
    return root


def unwired(vs):
    return {v.key.split("::")[-1] for v in vs if v.kind == "unwired"}


def kinds(vs):
    return {v.kind for v in vs}


LIB = "pub mod a;\npub mod b;\n"


class UnwiredItems(unittest.TestCase):
    def test_an_item_called_from_another_production_module_is_wired(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn helper() {}\n",
            "src/b.rs": "pub fn entry() { crate::a::helper(); }\n",
            "examples/e.rs": "fn main() { mycrate::b::entry(); }\n",  # entry 自身が配線済でないと helper も配線済にならない
        })
        self.assertNotIn("helper", unwired(wg.check(r)))

    def test_an_item_nobody_calls_is_unwired(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn lonely() {}\n", "src/b.rs": "pub fn x() {}\n"})
        self.assertIn("lonely", unwired(wg.check(r)))

    def test_pub_crate_items_are_checked_too(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub(crate) fn lonely_c() {}\n", "src/b.rs": "pub fn x() {}\n"})
        self.assertIn("lonely_c", unwired(wg.check(r)))

    def test_struct_enum_const_trait_are_checked(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub struct Sx;\npub enum Ex { A }\npub const CX: u32 = 1;\npub trait Tx {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertTrue({"Sx", "Ex", "CX", "Tx"} <= unwired(wg.check(r)))

    def test_referenced_only_inside_cfg_test_does_not_count(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn only_tested() {}\n",
            "src/b.rs": """
                pub fn x() {}
                #[cfg(test)]
                mod tests {
                    #[test]
                    fn t() { crate::a::only_tested(); }
                }
            """,
        })
        self.assertIn("only_tested", unwired(wg.check(r)))

    def test_referenced_only_from_the_tests_directory_does_not_count(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn only_integration() {}\n",
            "src/b.rs": "pub fn x() {}\n",
            "tests/t.rs": "fn t() { mycrate::a::only_integration(); }\n",
        })
        self.assertIn("only_integration", unwired(wg.check(r)))

    def test_a_mention_in_a_doc_comment_or_string_does_not_count(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn ghosted() {}\n",
            "src/b.rs": '/// see [`ghosted`] for details\npub fn x() { let _ = "ghosted"; }\n',
        })
        self.assertIn("ghosted", unwired(wg.check(r)))

    def test_a_use_line_alone_does_not_count(self):
        r = crate({
            "src/lib.rs": LIB + "pub use a::reexported;\n",
            "src/a.rs": "pub fn reexported() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertIn("reexported", unwired(wg.check(r)))

    def test_a_call_from_examples_or_benches_counts(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn demo_entry() {}\n",
            "src/b.rs": "pub fn x() {}\n",
            "examples/e.rs": "fn main() { mycrate::a::demo_entry(); }\n",
        })
        self.assertNotIn("demo_entry", unwired(wg.check(r)))

    def test_private_items_are_not_checked(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "fn private_lonely() {}\n", "src/b.rs": "pub fn x() {}\n"})
        self.assertNotIn("private_lonely", unwired(wg.check(r)))


class Markers(unittest.TestCase):
    def test_allow_unwired_with_a_reason_silences_the_item(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-UNWIRED: public entry point reached from downstream crates\npub fn api() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertNotIn("api", unwired(wg.check(r)))

    def test_allow_unwired_with_a_short_reason_is_rejected(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-UNWIRED: later\npub fn api() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertIn("bad_marker", kinds(wg.check(r)))

    def test_a_stale_allow_unwired_marker_is_rejected(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-UNWIRED: public entry point reached from downstream crates\npub fn api() {}\n",
            "src/b.rs": "pub fn x() { crate::a::api(); }\n",
            "examples/e.rs": "fn main() { mycrate::b::x(); }\n",
        })
        self.assertIn("stale_marker", kinds(wg.check(r)))

    def test_allow_dead_code_without_a_marker_is_rejected(self):
        r = crate({"src/lib.rs": LIB + "#![allow(dead_code)]\n", "src/a.rs": "", "src/b.rs": "pub fn x() {}\n"})
        self.assertIn("dead_code", kinds(wg.check(r)))

    def test_allow_dead_code_with_a_marker_passes(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-DEAD: reserved variants awaiting solver integration\n#![allow(dead_code)]\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertNotIn("dead_code", kinds(wg.check(r)))

    def test_allow_dead_code_marker_with_a_short_reason_is_rejected(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-DEAD: todo\n#[allow(dead_code)]\nfn f() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertIn("bad_marker", kinds(wg.check(r)))

    def test_a_stale_allow_dead_marker_is_rejected(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-DEAD: reserved variants awaiting solver integration\nfn f() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertIn("stale_marker", kinds(wg.check(r)))


class Baseline(unittest.TestCase):
    FILES = {"src/lib.rs": LIB, "src/a.rs": "pub fn legacy() {}\n", "src/b.rs": "pub fn x() {}\n"}

    def test_an_item_in_the_baseline_is_tolerated(self):
        r = crate(self.FILES)
        self.assertNotIn("legacy", unwired(wg.check(r, "unwired src/a.rs::legacy\n")))

    def test_an_item_not_in_the_baseline_fails(self):
        r = crate(self.FILES)
        self.assertIn("legacy", unwired(wg.check(r, "")))

    def test_a_stale_baseline_entry_fails_so_the_ratchet_tightens(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn legacy() {}\n", "src/b.rs": "pub fn x() { crate::a::legacy(); }\n", "examples/e.rs": "fn main() { mycrate::b::x(); }\n"})
        self.assertIn("stale_baseline", kinds(wg.check(r, "unwired src/a.rs::legacy\n")))

    def test_dead_code_count_may_not_grow_beyond_the_baseline(self):
        body = "#![allow(dead_code)]\n#[allow(dead_code)]\nfn f() {}\n"
        r = crate({"src/lib.rs": LIB, "src/a.rs": body, "src/b.rs": "pub fn x() {}\n"})
        self.assertIn("dead_code", kinds(wg.check(r, "dead_code src/a.rs 1\n")))
        self.assertNotIn("dead_code", kinds(wg.check(r, "dead_code src/a.rs 2\n")))


class EmptyScan(unittest.TestCase):
    def test_scanning_zero_items_is_a_failure_not_a_green(self):
        r = crate({"src/lib.rs": "fn private_only() {}\n"})
        self.assertIn("empty_scan", kinds(wg.check(r)))

    def test_a_missing_src_directory_is_a_failure(self):
        r = crate({"README.md": "x\n"})
        self.assertIn("empty_scan", kinds(wg.check(r)))


def keys(vs, kind="unwired"):
    return {v.key for v in vs if v.kind == kind}


EX = "examples/e.rs"


class TransitiveWiring(unittest.TestCase):
    """未配線の item の本体からしか参照されない item も未配線 (不動点反復)."""

    CHAIN = {
        "src/lib.rs": LIB,
        "src/a.rs": "pub fn a_fn() { b_helper(); }\nfn b_helper() { crate::b::c_fn(); }\n",
        "src/b.rs": "pub fn c_fn() {}\n",
    }

    def test_a_chain_hanging_off_an_unwired_pub_is_unwired(self):
        got = unwired(wg.check(crate(self.CHAIN)))
        self.assertTrue({"a_fn", "c_fn"} <= got, got)

    def test_a_private_helper_is_a_node_but_is_never_reported(self):
        got = unwired(wg.check(crate(self.CHAIN)))
        self.assertNotIn("b_helper", got)

    def test_allow_unwired_on_the_head_makes_the_rest_of_the_chain_wired(self):
        files = dict(self.CHAIN)
        files["src/a.rs"] = "// ALLOW-UNWIRED: public entry point reached from downstream crates\n" + files["src/a.rs"]
        vs = wg.check(crate(files))
        self.assertNotIn("c_fn", unwired(vs))
        self.assertNotIn("a_fn", unwired(vs))
        self.assertNotIn("stale_marker", kinds(vs))

    def test_a_call_to_the_head_from_examples_wires_the_whole_chain(self):
        files = dict(self.CHAIN)
        files[EX] = "fn main() { mycrate::a::a_fn(); }\n"
        self.assertEqual(unwired(wg.check(crate(files))) & {"a_fn", "c_fn"}, set())

    def test_a_live_chain_of_depth_five_is_fully_wired(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn l1() { l2(); }\nfn l2() { l3(); }\nfn l3() { crate::b::l4(); }\n",
            "src/b.rs": "pub fn l4() { l5(); }\nfn l5() { l6(); }\npub fn l6() {}\n",
            EX: "fn main() { mycrate::a::l1(); }\n",
        })
        self.assertEqual(unwired(wg.check(r)), set())

    def test_mutual_recursion_with_no_outside_caller_is_unwired_and_terminates(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn ping() { pong(); }\npub fn pong() { ping(); }\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertTrue({"ping", "pong"} <= unwired(wg.check(r)))

    def test_mutual_recursion_called_from_examples_is_wired(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn ping() { pong(); }\npub fn pong() { ping(); }\n",
            "src/b.rs": "pub fn x() {}\n",
            EX: "fn main() { mycrate::a::ping(); }\n",
        })
        self.assertEqual(unwired(wg.check(r)) & {"ping", "pong"}, set())

    def test_self_recursion_alone_is_unwired(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn rec(n: u32) { if n > 0 { rec(n - 1); } }\n", "src/b.rs": "pub fn x() {}\n"})
        self.assertIn("rec", unwired(wg.check(r)))

    def test_a_dead_private_fn_does_not_keep_its_callee_alive(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn helper() {}\n", "src/b.rs": "fn dead() { crate::a::helper(); }\npub fn x() {}\n"})
        self.assertIn("helper", unwired(wg.check(r)))

    def test_trait_impl_members_are_roots_so_their_callees_are_wired(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn helper() {}\n",
            "src/b.rs": "struct T;\nimpl core::fmt::Display for T {\n    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result { crate::a::helper(); Ok(()) }\n}\npub fn x() {}\n",
        })
        self.assertNotIn("helper", unwired(wg.check(r)))

    def test_a_user_trait_impl_member_is_a_root_for_its_callees(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn helper() {}\n",
            "src/b.rs": "pub trait Tr { fn run_it(&self); }\npub struct T;\nimpl Tr for T {\n    fn run_it(&self) { crate::a::helper(); }\n}\n",
        })
        self.assertNotIn("helper", unwired(wg.check(r)))

    def test_code_outside_src_is_a_root_even_in_a_function_nobody_calls(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn helper() {}\n",
            "src/b.rs": "pub fn x() {}\n",
            "examples/e.rs": "fn dead_helper() { mycrate::a::helper(); }\nfn main() {}\n",
        })
        self.assertNotIn("helper", unwired(wg.check(r)))

    def test_a_reference_at_module_level_outside_any_body_is_a_root(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn helper() {}\n", "src/b.rs": "register!(crate::a::helper);\npub fn x() {}\n"})
        self.assertNotIn("helper", unwired(wg.check(r)))

    def test_exempt_attribute_items_are_roots_for_their_callees(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn helper() {}\n",
            "src/b.rs": '#[no_mangle]\npub extern "C" fn ffi() { crate::a::helper(); }\n',
        })
        self.assertNotIn("helper", unwired(wg.check(r)))

    def test_a_type_used_only_in_an_unwired_signature_is_unwired(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub struct Cfg;\n",
            "src/b.rs": "pub fn make() -> crate::a::Cfg { crate::a::Cfg }\n",
        })
        self.assertTrue({"Cfg", "make"} <= unwired(wg.check(r)))

    def test_a_reference_in_the_last_character_of_a_body_belongs_to_that_body(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn z() {}\n", "src/b.rs": "pub fn top() -> fn() {crate::a::z}\n"})
        self.assertTrue({"z", "top"} <= unwired(wg.check(r)))
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn z() {}\n", "src/b.rs": "pub fn top() -> fn() {crate::a::z}\n",
                   EX: "fn main() { mycrate::b::top(); }\n"})
        self.assertEqual(unwired(wg.check(r)), set())

    def test_a_reference_right_after_a_body_belongs_to_the_outside(self):
        # `}` の直後 (同じ行) の module 直下の参照は、直前の fn の本体に含めない
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn zz() {}\n", "src/b.rs": "pub fn top() {}crate::a::zz;\n"})
        self.assertNotIn("zz", unwired(wg.check(r)))

    def test_the_stale_baseline_check_sees_the_transitive_entries(self):
        r = crate(self.CHAIN)
        base = "unwired src/a.rs::a_fn\nunwired src/b.rs::c_fn\n"
        self.assertEqual(keys(wg.check(r, base)), set())
        self.assertEqual(keys(wg.check(r, "unwired src/a.rs::a_fn\n")), {"src/b.rs::c_fn"})


class NameCollisions(unittest.TestCase):
    """local 変数 / 引数 / フィールドと同名の pub fn は、それだけでは配線済にならない."""

    def run_with(self, user_src: str, extra: dict[str, str] | None = None):
        files = {
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn flip() {}\n",
            "src/b.rs": user_src,
            EX: "fn main() { mycrate::b::x(); }\n",
        }
        files.update(extra or {})
        return unwired(wg.check(crate(files)))

    def test_a_let_binding_is_not_a_reference(self):
        self.assertIn("flip", self.run_with("pub fn x() { let flip = 1; let _ = flip + 1; }\n"))

    def test_a_let_mut_binding_is_not_a_reference(self):
        self.assertIn("flip", self.run_with("pub fn x() { let mut flip = 1; flip += 1; }\n"))

    def test_a_tuple_pattern_binding_is_not_a_reference(self):
        self.assertIn("flip", self.run_with("pub fn x() { let (flip, other) = (1, 2); let _ = flip + other; }\n"))

    def test_an_if_let_binding_is_not_a_reference(self):
        self.assertIn("flip", self.run_with("pub fn x(o: Option<u8>) { if let Some(flip) = o { let _ = flip; } }\n"))

    def test_a_function_parameter_is_not_a_reference(self):
        self.assertIn("flip", self.run_with("pub fn x(flip: bool) -> bool { flip }\n"))

    def test_a_closure_parameter_is_not_a_reference(self):
        self.assertIn("flip", self.run_with("pub fn x() { let _f = |flip: i32| flip + 1; let _g = |flip| flip; }\n"))

    def test_a_for_loop_variable_is_not_a_reference(self):
        self.assertIn("flip", self.run_with("pub fn x() { for flip in 0..3 { let _ = flip; } }\n"))

    def test_a_field_definition_and_field_access_are_not_references(self):
        src = "pub struct S { pub flip: bool }\npub fn x(s: &S) -> bool { s.flip }\n"
        self.assertIn("flip", self.run_with(src))

    def test_a_struct_literal_field_and_shorthand_are_not_references(self):
        src = "pub struct S { pub flip: bool }\npub fn x() -> S { let v = true; S { flip: v } }\npub fn y(flip: bool) -> S { S { flip } }\n"
        self.assertIn("flip", self.run_with(src))

    def test_a_method_call_does_not_wire_a_free_fn_of_the_same_name(self):
        src = "pub struct S;\nimpl S { pub fn flip(&self) {} }\npub fn x(s: &S) { s.flip(); }\n"
        files = {"src/lib.rs": LIB, "src/a.rs": "pub fn flip() {}\n", "src/b.rs": src,
                 EX: "fn main() { mycrate::b::x(&mycrate::b::S); }\n"}
        self.assertIn("src/a.rs::flip", keys(wg.check(crate(files))))

    def test_an_actual_call_wires_the_fn(self):
        self.assertNotIn("flip", self.run_with("pub fn x() { crate::a::flip(); }\n"))

    def test_a_bare_call_wires_the_fn(self):
        self.assertNotIn("flip", self.run_with("use crate::a::flip;\npub fn x() { flip(); }\n"))

    def test_a_turbofish_call_wires_the_fn(self):
        r = self.run_with("pub fn x() { crate::a::flip::<u8>(); }\n", {"src/a.rs": "pub fn flip<T>() {}\n"})
        self.assertNotIn("flip", r)

    def test_passing_the_fn_path_as_a_value_wires_it(self):
        self.assertNotIn("flip", self.run_with("pub fn x() { call(crate::a::flip); }\n"))

    def test_a_bare_fn_pointer_argument_wires_it(self):
        self.assertNotIn("flip", self.run_with("use crate::a::flip;\npub fn x() { apply(flip); }\n"))

    def test_a_local_shadow_followed_by_a_real_call_elsewhere_still_wires_it(self):
        src = "pub fn x() { let flip = 1; let _ = flip; }\npub fn y() { crate::a::flip(); }\n"
        files = {"src/lib.rs": LIB, "src/a.rs": "pub fn flip() {}\n", "src/b.rs": src,
                 EX: "fn main() { mycrate::b::x(); mycrate::b::y(); }\n"}
        self.assertNotIn("flip", unwired(wg.check(crate(files))))

    def test_a_method_is_wired_only_by_a_method_call_or_a_path(self):
        files = {"src/lib.rs": LIB, "src/a.rs": "pub struct S;\nimpl S { pub fn flip(&self) {} }\n",
                 "src/b.rs": "pub fn x(s: &crate::a::S) { let flip = 1; let _ = flip; }\n",
                 EX: "fn main() { mycrate::b::x(&mycrate::a::S); }\n"}
        self.assertIn("flip", unwired(wg.check(crate(files))))
        files["src/b.rs"] = "pub fn x(s: &crate::a::S) { s.flip(); }\n"
        self.assertNotIn("flip", unwired(wg.check(crate(files))))
        files["src/b.rs"] = "pub fn x(s: &crate::a::S) { crate::a::S::flip(s); }\n"
        self.assertNotIn("flip", unwired(wg.check(crate(files))))

    def method_files(self, user_src: str):
        return {"src/lib.rs": LIB, "src/a.rs": "pub struct S;\nimpl S { pub fn flip(&self) {} }\n", "src/b.rs": user_src,
                EX: "fn main() { mycrate::b::x(&mycrate::a::S); }\n"}

    def test_a_field_access_does_not_wire_a_method_of_the_same_name(self):
        files = self.method_files("pub struct P { pub flip: bool }\npub fn x(_s: &crate::a::S, p: &P) -> bool { p.flip }\n")
        self.assertIn("flip", unwired(wg.check(crate(files))))

    def test_a_bare_call_of_a_local_closure_does_not_wire_a_method(self):
        files = self.method_files("pub fn x(_s: &crate::a::S) -> u8 { let flip = |a: u8| a; flip(1) }\n")
        self.assertIn("flip", unwired(wg.check(crate(files))))

    def test_a_struct_pattern_field_shorthand_is_not_a_reference(self):
        src = "pub struct S { pub flip: bool }\npub fn x(S { flip }: S) { let _ = (); }\n"
        self.assertIn("flip", self.run_with(src))

    def test_uppercase_items_keep_the_plain_occurrence_rule(self):
        files = {"src/lib.rs": LIB, "src/a.rs": "pub const LIMIT: u32 = 3;\n",
                 "src/b.rs": "pub fn x() -> u32 { crate::a::LIMIT }\n", EX: "fn main() { mycrate::b::x(); }\n"}
        self.assertNotIn("LIMIT", unwired(wg.check(crate(files))))


class Robustness(unittest.TestCase):
    """検査器自身が例外終了せず、明示的な違反か empty_scan になる."""

    def test_an_empty_source_tree_is_an_explicit_empty_scan(self):
        self.assertIn("empty_scan", kinds(wg.check(crate({"src/lib.rs": ""}))))

    def test_an_unclosed_brace_is_an_explicit_violation_not_an_exception(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn broken() {\n  if x {\n", "src/b.rs": "pub fn x() {}\n"})
        vs = wg.check(r)
        self.assertIn("unbalanced_braces", kinds(vs))
        self.assertEqual(wg.main(["--root", str(r), "--baseline", "none"]), 1)

    def test_non_utf8_bytes_do_not_raise(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn ok() {}\n", "src/b.rs": "pub fn x() {}\n"})
        (r / "src/a.rs").write_bytes(b"pub fn ok() {}\n// \xff\xfe\x80 broken\n")
        vs = wg.check(r)
        self.assertIsInstance(vs, list)
        self.assertIn("ok", unwired(vs))

    def test_a_huge_file_finishes_quickly(self):
        import time

        body = "".join(f"pub fn f{i}() {{ if true {{ g{i}(); }} }}\nfn g{i}() {{}}\n" for i in range(20000))
        r = crate({"src/lib.rs": LIB, "src/a.rs": body, "src/b.rs": "pub fn x() {}\n"})
        # ⚠️ monotonic, not time(): wall clock steps (machine sleep, NTP) make a
        # time-based gate report an elapsed time it never spent, and the failure
        # looks like a performance regression rather than a clock jump. Measured
        # on 2026-10-02: this assertion read 943 s while the whole 79-test suite
        # finished in 4.7 s.
        t0 = time.monotonic()
        vs = wg.check(r)
        self.assertLess(time.monotonic() - t0, 30)
        self.assertEqual(len([v for v in vs if v.kind == "unwired"]), 20001)

    def test_a_pathological_header_without_a_terminator_does_not_hang(self):
        import time

        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn a\n" * 5000, "src/b.rs": "pub fn x() {}\n"})
        # ⚠️ monotonic, not time(): wall clock steps (machine sleep, NTP) make a
        # time-based gate report an elapsed time it never spent, and the failure
        # looks like a performance regression rather than a clock jump. Measured
        # on 2026-10-02: this assertion read 943 s while the whole 79-test suite
        # finished in 4.7 s.
        t0 = time.monotonic()
        wg.check(r)
        self.assertLess(time.monotonic() - t0, 30)

    def test_no_src_directory_is_an_empty_scan(self):
        self.assertIn("empty_scan", kinds(wg.check(crate({"README.md": "x\n"}))))

    def test_a_binary_garbage_file_does_not_raise(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn lonely() {}\n", "src/b.rs": "pub fn x() {}\n"})
        (r / "src/c.rs").write_bytes(bytes(range(256)) * 50)
        self.assertIsInstance(wg.check(r), list)


class Workspace(unittest.TestCase):
    """Cargo workspace: member の src/ を走査し、参照は repo 全体の corpus で数える."""

    WS = '[workspace]\nmembers = ["alpha", "beta"]\nresolver = "2"\n'

    def test_a_call_from_a_sibling_member_wires_the_item(self):
        r = crate({
            "Cargo.toml": self.WS,
            "alpha/src/lib.rs": "pub fn shared() {}\n",
            "beta/src/lib.rs": "pub fn x() { alpha::shared(); }\n",
        })
        vs = wg.check(r)
        self.assertIn("beta/src/lib.rs::x", keys(vs))
        self.assertIn("alpha/src/lib.rs::shared", keys(vs))  # x が未配線なので shared も未配線 (推移)
        (r / "beta/examples").mkdir()
        (r / "beta/examples/e.rs").write_text("fn main() { beta::x(); }\n")  # x が配線済になれば shared も配線済
        self.assertEqual(keys(wg.check(r)), set())

    def test_keys_are_relative_to_the_repo_root(self):
        r = crate({
            "Cargo.toml": self.WS,
            "alpha/src/lib.rs": "pub fn lonely_a() {}\n",
            "beta/src/lib.rs": "pub fn lonely_b() {}\n",
        })
        self.assertEqual(keys(wg.check(r)), {"alpha/src/lib.rs::lonely_a", "beta/src/lib.rs::lonely_b"})

    def test_a_member_without_src_is_skipped_not_fatal(self):
        r = crate({
            "Cargo.toml": '[workspace]\nmembers = ["alpha", "gamma"]\n',
            "alpha/src/lib.rs": "pub fn lonely_a() {}\n",
            "gamma/README.md": "no src here\n",
        })
        vs = wg.check(r)
        self.assertIn("alpha/src/lib.rs::lonely_a", keys(vs))
        self.assertNotIn("empty_scan", kinds(vs))

    def test_empty_scan_only_when_no_member_defines_anything(self):
        r = crate({"Cargo.toml": self.WS, "alpha/src/lib.rs": "fn private() {}\n", "beta/README.md": "x\n"})
        self.assertIn("empty_scan", kinds(wg.check(r)))
        r2 = crate({"Cargo.toml": self.WS, "alpha/src/lib.rs": "fn private() {}\n", "beta/src/lib.rs": "pub fn d() {}\n"})
        self.assertNotIn("empty_scan", kinds(wg.check(r2)))

    def test_a_glob_member_is_expanded(self):
        r = crate({
            "Cargo.toml": '[workspace]\nmembers = ["crates/*"]\n',
            "crates/a/src/lib.rs": "pub fn def_a() {}\n",
            "crates/b/src/lib.rs": "pub fn caller() { a::def_a(); }\n",
            "crates/b/examples/e.rs": "fn main() { b::caller(); }\n",
            "crates/notes.txt": "not a crate\n",
        })
        self.assertEqual(keys(wg.check(r)), set())
        (r / "crates/b/examples/e.rs").unlink()
        self.assertEqual(keys(wg.check(r)), {"crates/b/src/lib.rs::caller", "crates/a/src/lib.rs::def_a"})

    def test_tests_and_target_under_a_member_are_still_excluded(self):
        r = crate({
            "Cargo.toml": self.WS,
            "alpha/src/lib.rs": "pub fn only_tested() {}\n",
            "beta/src/lib.rs": "pub fn x() {}\n",
            "beta/tests/t.rs": "fn t() { alpha::only_tested(); }\n",
            "alpha/target/gen.rs": "fn g() { only_tested(); }\n",
        })
        self.assertIn("alpha/src/lib.rs::only_tested", keys(wg.check(r)))

    def test_a_root_package_that_is_also_a_workspace_is_scanned_too(self):
        r = crate({
            "Cargo.toml": '[package]\nname = "root"\n[workspace]\nmembers = ["alpha"]\n',
            "src/lib.rs": "pub fn root_lonely() {}\n",
            "alpha/src/lib.rs": "pub fn alpha_lonely() {}\n",
        })
        self.assertEqual(keys(wg.check(r)), {"src/lib.rs::root_lonely", "alpha/src/lib.rs::alpha_lonely"})

    def test_a_transitive_chain_across_members(self):
        r = crate({
            "Cargo.toml": self.WS,
            "alpha/src/lib.rs": "pub fn deep() {}\n",
            "beta/src/lib.rs": "pub fn top() { mid(); }\nfn mid() { alpha::deep(); }\n",
        })
        self.assertEqual(keys(wg.check(r)), {"alpha/src/lib.rs::deep", "beta/src/lib.rs::top"})

    def test_an_excluded_member_is_not_scanned(self):
        r = crate({
            "Cargo.toml": '[workspace]\nmembers = ["crates/*"]\nexclude = ["crates/skip"]\n',
            "crates/a/src/lib.rs": "pub fn lonely_a() {}\n",
            "crates/skip/src/lib.rs": "pub fn lonely_skip() {}\n",
        })
        self.assertEqual(keys(wg.check(r)), {"crates/a/src/lib.rs::lonely_a"})

    def test_markers_and_the_baseline_use_the_repo_relative_key(self):
        r = crate({
            "Cargo.toml": self.WS,
            "alpha/src/lib.rs": "// ALLOW-UNWIRED: public entry point reached from downstream crates\npub fn api() {}\npub fn legacy() {}\n",
            "beta/src/lib.rs": "pub fn x() {}\n",
        })
        vs = wg.check(r, "unwired alpha/src/lib.rs::legacy\nunwired beta/src/lib.rs::x\n")
        self.assertEqual(keys(vs), set())


class RawPointerTypes(unittest.TestCase):
    """`*const T` は型であって `const` item ではない (item と数えると後続の method を抱え込む)."""

    SRC = textwrap.dedent("""\
        pub struct V { ptr: *mut f32 }
        impl V {
            pub const fn as_ptr(&self) -> *const f32 {
                self.ptr
            }
            pub fn as_mut_slice(&mut self) -> &mut [f32] {
                &mut []
            }
        }
        impl core::ops::DerefMut for V {
            fn deref_mut(&mut self) -> &mut [f32] {
                self.as_mut_slice()
            }
        }
        """)

    def test_a_method_after_a_raw_pointer_return_type_is_wired_by_a_trait_impl_call(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": self.SRC, "src/b.rs": "pub fn x() {}\n"})
        self.assertNotIn("as_mut_slice", unwired(wg.check(r)))
        # as_ptr has no caller: the fixture still checks something
        self.assertIn("as_ptr", unwired(wg.check(r)))

    def test_a_raw_pointer_type_is_not_an_item_node(self):
        code = "impl V {\n    pub fn p(&self) -> *const f32 { self.ptr }\n    pub fn q(&self) {}\n}\n"
        nodes = wg.parse_nodes(code, "src/a.rs", True, 0)
        self.assertEqual([n.kind for n in nodes if n.kind == "const"], [])
        q = next(n for n in nodes if n.name == "q")
        self.assertEqual(q.parent.kind, "impl")


class SameNameInOneFile(unittest.TestCase):
    """同じ file の同名 item は所有する型で key を分ける (baseline の 1 行が両方を覆わない)."""

    SRC = textwrap.dedent("""\
        pub struct A;
        pub struct B;
        impl A {
            pub fn high_quality() -> Self { A }
        }
        impl B {
            pub fn high_quality() -> Self { B }
        }
        pub fn make() -> A { A::high_quality() }
        """)

    def files(self):
        return {"src/lib.rs": LIB, "src/a.rs": self.SRC, "src/b.rs": "pub fn x() { crate::a::make(); }\n",
                "examples/e.rs": "fn main() { mycrate::b::x(); }\n"}

    def test_the_keys_name_the_owning_type(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": self.SRC.replace("pub fn make() -> A { A::high_quality() }\n", ""),
                   "src/b.rs": "pub fn x() {}\n"})
        keys = {v.key for v in wg.check(r) if v.kind == "unwired"}
        self.assertIn("src/a.rs::A::high_quality", keys)
        self.assertIn("src/a.rs::B::high_quality", keys)
        self.assertNotIn("src/a.rs::high_quality", keys)

    def test_baseline_lines_are_unique(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": self.SRC.replace("pub fn make() -> A { A::high_quality() }\n", ""),
                   "src/b.rs": "pub fn x() {}\n"})
        lines = [ln for ln in wg.build_baseline(r).splitlines() if ln.startswith("unwired ")]
        self.assertEqual(len(lines), len(set(lines)))
        self.assertIn("unwired src/a.rs::A::high_quality", lines)

    def test_an_unqualified_line_does_not_cover_both(self):
        # the old form `src/a.rs::high_quality` covered both items; it no longer matches
        r = crate({"src/lib.rs": LIB, "src/a.rs": self.SRC.replace("pub fn make() -> A { A::high_quality() }\n", ""),
                   "src/b.rs": "pub fn x() {}\n"})
        vs = wg.check(r, "unwired src/a.rs::high_quality\nunwired src/b.rs::x\n")
        self.assertIn("stale_baseline", kinds(vs))
        self.assertEqual({v.key for v in vs if v.kind == "unwired"},
                         {"src/a.rs::A::high_quality", "src/a.rs::B::high_quality"})

    def test_a_free_fn_and_a_method_of_the_same_name_are_told_apart(self):
        src = "pub struct H;\nimpl H {\n    pub fn extract(&self) {}\n}\npub fn extract() {}\n"
        # `.extract()` can only be the method (a free fn is never called with a dot), so the
        # free fn stays unwired and is reported under its own key
        r = crate({"src/lib.rs": LIB, "src/a.rs": src, "src/b.rs": "pub fn x(h: &crate::a::H) { h.extract(); }\n",
                   "examples/e.rs": "fn main() { mycrate::b::x(&mycrate::a::H); }\n"})
        keys = {v.key for v in wg.check(r) if v.kind == "unwired"}
        self.assertEqual(keys, {"src/a.rs::extract"})

    def test_generic_arguments_are_kept_only_when_the_type_name_is_not_enough(self):
        src = ("pub struct Vol<T>(T);\nimpl Vol<f32> {\n    pub fn sample(&self) {}\n}\n"
               "impl Vol<u8> {\n    pub fn sample(&self) {}\n}\n")
        r = crate({"src/lib.rs": LIB, "src/a.rs": src, "src/b.rs": "pub fn x() {}\n"})
        keys = {v.key for v in wg.check(r) if v.kind == "unwired"}
        self.assertIn("src/a.rs::Vol<f32>::sample", keys)
        self.assertIn("src/a.rs::Vol<u8>::sample", keys)

    def test_impl_self_type(self):
        self.assertEqual(wg.impl_self_type(" <'a, F> Hermite<'a, F>\nwhere\n    F: Fn(),\n"), "Hermite<'a,F>")
        self.assertEqual(wg.impl_self_type(" core::ops::Deref for Aligned "), "Aligned")
        self.assertEqual(wg.impl_self_type(" Volume3D<f32> "), "Volume3D<f32>")

    def test_an_allow_unwired_marker_applies_to_the_qualified_key(self):
        src = self.SRC.replace("pub fn make() -> A { A::high_quality() }\n", "").replace(
            "impl B {\n", "impl B {\n    // ALLOW-UNWIRED: kept for a planned preset table\n")
        r = crate({"src/lib.rs": LIB, "src/a.rs": src, "src/b.rs": "pub fn x() {}\n"})
        keys = {v.key for v in wg.check(r) if v.kind == "unwired"}
        self.assertIn("src/a.rs::A::high_quality", keys)
        self.assertNotIn("src/a.rs::B::high_quality", keys)


if __name__ == "__main__":
    unittest.main(verbosity=2)
