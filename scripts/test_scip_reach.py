#!/usr/bin/env python3
"""Tests for scripts/scip_reach.py.

Each case writes a few Rust source files and a hand-built SCIP index (encoded
here, so rust-analyzer is not needed) into a temporary directory, then checks
the level the analysis assigns. One rule per case; breaking that rule in
scip_reach.py turns exactly that case red.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scip_reach as sr  # noqa: E402

P = "rust-analyzer cargo demo 0.1.0 "


# --- tiny SCIP encoder ---------------------------------------------------------

def _v(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _f(field: int, payload) -> bytes:
    if isinstance(payload, int):
        return _v(field << 3 | 0) + _v(payload)
    if isinstance(payload, str):
        payload = payload.encode()
    return _v(field << 3 | 2) + _v(len(payload)) + payload


def _packed(xs) -> bytes:
    return b"".join(_v(x) for x in xs)


class Doc:
    def __init__(self, path: str, text: str):
        self.path, self.lines = path, text.split("\n")
        self.occ: list[bytes] = []
        self.syms: list[bytes] = []

    def _pos(self, line: int, token: str, nth: int = 0) -> list[int]:
        start = -1
        for _ in range(nth + 1):
            start = self.lines[line].index(token, start + 1)
        return [line, start, start + len(token)]

    def define(self, sym: str, line: int, token: str, end_line: int | None = None, nth: int = 0) -> "Doc":
        """Definition of `sym` at `token`; its body spans line..end_line."""
        r = self._pos(line, token, nth)
        end_line = line if end_line is None else end_line
        enc = [line, 0, end_line, len(self.lines[end_line])]
        self.occ.append(_f(1, _packed(r)) + _f(2, P + sym) + _f(3, 1) + _f(7, _packed(enc)))
        return self

    def ref(self, sym: str, line: int, token: str, nth: int = 0) -> "Doc":
        self.occ.append(_f(1, _packed(self._pos(line, token, nth))) + _f(2, P + sym))
        return self

    def implements(self, sym: str, target: str, external: bool = False) -> "Doc":
        tgt = ("rust-analyzer cargo std 1.0.0 " + target) if external else P + target
        rel = _f(1, tgt) + _f(3, 1)
        self.syms.append(_f(1, P + sym) + _f(4, rel))
        return self

    def encode(self) -> bytes:
        body = _f(1, self.path) + b"".join(_f(2, o) for o in self.occ) + b"".join(_f(3, s) for s in self.syms)
        return _f(2, body)


def build(docs: list[Doc]) -> Path:
    """Sources plus the three indexes. A Doc under fuzz/ goes to fuzz.scip with the
    prefix removed, as rust-analyzer indexes the fuzz crate from its own root."""
    d = Path(tempfile.mkdtemp())
    for doc in docs:
        p = d / doc.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(doc.lines), encoding="utf-8")
    crate = [doc for doc in docs if not doc.path.startswith("fuzz/")]
    fuzz = [doc for doc in docs if doc.path.startswith("fuzz/")]
    (d / "target" / "scip").mkdir(parents=True)
    (d / "target" / "scip" / "native.scip").write_bytes(b"".join(doc.encode() for doc in crate))
    (d / "target" / "scip" / "wasm.scip").write_bytes(b"")
    blob = b""
    for doc in fuzz:
        full = doc.path
        doc.path = full[len("fuzz/"):]
        blob += doc.encode()
        doc.path = full
    (d / "target" / "scip" / "fuzz.scip").write_bytes(blob)
    return d


def levels(docs: list[Doc]) -> dict[str, str]:
    d = build(docs)
    s = d / "target" / "scip"
    return sr.analyze(d, [s / "native.scip", s / "wasm.scip", s / "fuzz.scip"]).level


# --- fixtures -----------------------------------------------------------------

LIB = """pub fn used() {}
pub fn unused() {}
pub fn helper() {}
pub fn via_binding() { helper(); }
trait Tr { fn run(&self); }
impl Tr for X { fn run(&self) {} }"""


def lib_doc() -> Doc:
    return (Doc("src/lib.rs", LIB)
            .define("used().", 0, "used")
            .define("unused().", 1, "unused")
            .define("helper().", 2, "helper")
            .define("via_binding().", 3, "via_binding")
            .ref("helper().", 3, "helper")
            # a private trait and its impl: no pub item, one trait-impl link (the
            # Main guards need one), so the levels above are unchanged
            .define("Tr#run().", 4, "run")
            .define("impl#[X][Tr]run().", 5, "run"))


class Levels(unittest.TestCase):
    def test_example_only_is_l1_and_unreferenced_is_l0(self):
        ex = Doc("examples/demo.rs", "fn main() { used(); }").ref("used().", 0, "used")
        lv = levels([lib_doc(), ex])
        self.assertEqual(lv["src/lib.rs::used"], "L1")
        self.assertEqual(lv["src/lib.rs::unused"], "L0")

    def test_binding_reaches_transitively_and_is_live(self):
        ffi = Doc("src/ffi/mod.rs", "pub extern \"C\" fn api() { via_binding(); }").ref("via_binding().", 0, "via_binding")
        lv = levels([lib_doc(), ffi])
        self.assertEqual(lv["src/lib.rs::via_binding"], "live")
        self.assertEqual(lv["src/lib.rs::helper"], "live")
        self.assertNotIn("src/ffi/mod.rs::api", lv)  # binding surface is a root, not reported

    def test_same_name_on_different_types_is_not_conflated(self):
        text = "pub struct A;\npub struct B;\nimpl A { pub fn update(&self) {} }\nimpl B { pub fn update(&self) {} }"
        src = (Doc("src/m.rs", text)
               .define("m/impl#[A]update().", 2, "update")
               .define("m/impl#[B]update().", 3, "update"))
        ex = Doc("examples/e.rs", "fn main() { a.update(); }").ref("m/impl#[A]update().", 0, "update")
        lv = levels([src, ex])
        # one key per impl type: reaching A::update must not hide that B::update is
        # unreached (keyed by name alone, the pair was one item and read as reached)
        self.assertEqual(lv["src/m.rs::A::update"], "L1")
        self.assertEqual(lv["src/m.rs::B::update"], "L0")
        self.assertNotIn("src/m.rs::update", lv)

    def test_ledger_compares_with_the_wiring_baseline_by_file_and_name(self):
        # wiring-baseline.txt has no type in its keys: `src/m.rs::update` there
        # stands for both methods, and counts as reached only if both are
        text = "pub struct A;\npub struct B;\nimpl A { pub fn update(&self) {} }\nimpl B { pub fn update(&self) {} }"
        src = (Doc("src/m.rs", text)
               .define("m/impl#[A]update().", 2, "update")
               .define("m/impl#[B]update().", 3, "update"))
        ex = Doc("examples/e.rs", "fn main() { a.update(); }").ref("m/impl#[A]update().", 0, "update")
        d = build([src, ex])
        s_ = d / "target" / "scip"
        a = sr.analyze(d, [s_ / "native.scip", s_ / "wasm.scip", s_ / "fuzz.scip"])
        text_out = sr.report(a, {"src/m.rs::update"})
        # B::update is unreached and its file::name is in the wiring baseline, so it is
        # not "L0 here but not in the baseline"
        self.assertIn("### L0 here but not in the baseline (0)", text_out)
        self.assertIn("### In the baseline but reached here (0)", text_out)
        self.assertIn("- `src/m.rs::B::update`", text_out.split("## L0 — unreached")[1])

    def test_impl_header_is_not_a_use_of_the_type(self):
        text = "pub struct Foo;\nimpl Foo {\n    fn x(&self) {}\n}"
        src = Doc("src/m.rs", text).define("m/Foo#", 0, "Foo").ref("m/Foo#", 1, "Foo")
        self.assertEqual(levels([src])["src/m.rs::Foo"], "L0")

    def test_reference_inside_cfg_test_is_ignored(self):
        text = "pub fn target() {}\n#[cfg(test)]\nmod tests {\n    fn t() { super::target(); }\n}"
        src = Doc("src/m.rs", text).define("m/target().", 0, "target").ref("m/target().", 3, "target")
        self.assertEqual(levels([src])["src/m.rs::target"], "L0")

    def test_reference_inside_comment_is_ignored(self):
        text = "pub fn target() {}\n// see target() for details"
        src = Doc("src/m.rs", text).define("m/target().", 0, "target").ref("m/target().", 1, "target")
        self.assertEqual(levels([src])["src/m.rs::target"], "L0")

    def test_pub_use_reexport_is_not_a_use(self):
        lib = Doc("src/lib.rs", "pub use m::target;").ref("m/target().", 0, "target")
        src = Doc("src/m.rs", "pub fn target() {}").define("m/target().", 0, "target")
        self.assertEqual(levels([lib, src])["src/m.rs::target"], "L0")

    def test_module_level_code_is_a_root(self):
        text = "pub fn target() {}\nconst_assert!(target());"
        src = Doc("src/m.rs", text).define("m/target().", 0, "target").ref("m/target().", 1, "target")
        self.assertEqual(levels([src])["src/m.rs::target"], "live")

    def test_call_through_trait_reaches_the_implementation(self):
        text = ("pub trait Tr { fn run(&self); }\npub fn inner() {}\n"
                "impl Tr for X { fn run(&self) { inner(); } }\npub fn go(t: &dyn Tr) { t.run(); }")

        def src(with_relationship: bool) -> Doc:
            d = (Doc("src/m.rs", text)
                 .define("m/Tr#run().", 0, "run")
                 .define("m/inner().", 1, "inner")
                 .define("m/impl#[X][Tr]run().", 2, "run")
                 .ref("m/inner().", 2, "inner")
                 .define("m/go().", 3, "go")
                 .ref("m/Tr#run().", 3, "run"))
            return d.implements("m/impl#[X][Tr]run().", "m/Tr#run().") if with_relationship else d

        ex = Doc("examples/e.rs", "fn main() { go(&x); }").ref("m/go().", 0, "go")
        # go -> Tr::run -> X::run -> inner, through the SCIP relationship when present
        self.assertEqual(levels([src(True), ex])["src/m.rs::inner"], "L1")
        # and from the impl symbol name alone: rust-analyzer emits no relationships,
        # so this is the case that occurs in practice
        self.assertEqual(levels([src(False), ex])["src/m.rs::inner"], "L1")

    def test_fuzz_target_is_a_caller(self):
        # the fuzz crate is indexed from fuzz/: its paths gain the prefix and its
        # references count like an example's
        fz = Doc("fuzz/fuzz_targets/f.rs", "fuzz_target!(|d| { unused(); });").ref("unused().", 0, "unused")
        self.assertEqual(levels([lib_doc()])["src/lib.rs::unused"], "L0")
        self.assertEqual(levels([lib_doc(), fz])["src/lib.rs::unused"], "L1")

    def test_fuzz_references_count_whatever_the_index_order(self):
        # fuzz references are counted against the crate's own definitions, so the
        # fuzz index must be read after native / wasm even when passed first
        fz = Doc("fuzz/fuzz_targets/f.rs", "fuzz_target!(|d| { used(); });").ref("used().", 0, "used")
        d = build([lib_doc(), fz])
        s = d / "target" / "scip"
        a = sr.analyze(d, [s / "fuzz.scip", s / "native.scip", s / "wasm.scip"])
        self.assertEqual(a.fuzz_refs, 1)

    def test_impl_link_needs_the_same_trait_and_method(self):
        text = ("pub trait Tr { fn run(&self); }\npub fn inner() {}\n"
                "impl Other for X { fn run(&self) { inner(); } }\npub fn go(t: &dyn Tr) { t.run(); }\n"
                "impl Tr for Y { fn stop(&self) { inner(); } }")
        src = (Doc("src/m.rs", text)
               .define("m/Tr#run().", 0, "run")
               .define("m/Other#run().", 0, "Tr")       # an in-crate trait of another name
               .define("m/inner().", 1, "inner")
               .define("m/impl#[X][Other]run().", 2, "run")
               .ref("m/inner().", 2, "inner")
               .define("m/go().", 3, "go")
               .ref("m/Tr#run().", 3, "run")
               .define("m/impl#[Y][Tr]stop().", 4, "stop")
               .ref("m/inner().", 4, "inner"))
        ex = Doc("examples/e.rs", "fn main() { go(&x); }").ref("m/go().", 0, "go")
        # Other::run and Tr::stop are not Tr::run: reaching Tr::run reaches neither
        self.assertEqual(levels([src, ex])["src/m.rs::inner"], "L0")

    def test_trait_name_with_path_and_generics_is_matched(self):
        text = ("pub trait Conv<T> { fn conv(&self); }\npub fn inner() {}\n"
                "impl Conv<u8> for X { fn conv(&self) { inner(); } }\npub fn go(t: &dyn Conv<u8>) { t.conv(); }")
        src = (Doc("src/m.rs", text)
               .define("m/Conv#conv().", 0, "conv")
               .define("m/inner().", 1, "inner")
               .define("m/impl#[X][`crate::m::Conv<u8>`]conv().", 2, "conv")
               .ref("m/inner().", 2, "inner")
               .define("m/go().", 3, "go")
               .ref("m/Conv#conv().", 3, "conv"))
        ex = Doc("examples/e.rs", "fn main() { go(&x); }").ref("m/go().", 0, "go")
        self.assertEqual(levels([src, ex])["src/m.rs::inner"], "L1")

    def test_impl_of_external_trait_is_a_root_without_a_relationship(self):
        text = "pub fn helper() {}\npub struct X;\nimpl Drop for X {\n    fn drop(&mut self) { helper(); }\n}"
        src = (Doc("src/m.rs", text)
               .define("m/helper().", 0, "helper")
               .define("m/impl#[X][Drop]drop().", 3, "drop", 3)
               .ref("m/helper().", 3, "helper"))
        self.assertEqual(levels([src])["src/m.rs::helper"], "live")

    def test_impl_of_a_type_nothing_reaches_is_not_reached(self):
        # rapid type analysis: reaching Tr::run reaches Y::run (Y is used) but not
        # X::run (no value of X exists), so only what Y::run calls is reached
        text = ("pub trait Tr { fn run(&self); }\npub fn via_x() {}\npub fn via_y() {}\n"
                "pub struct X;\npub struct Y;\n"
                "impl Tr for X { fn run(&self) { via_x(); } }\nimpl Tr for Y { fn run(&self) { via_y(); } }\n"
                "pub fn go() { let y = Y; y.run(); }")
        src = (Doc("src/m.rs", text)
               .define("m/Tr#run().", 0, "run")
               .define("m/via_x().", 1, "via_x")
               .define("m/via_y().", 2, "via_y")
               .define("m/X#", 3, "X")
               .define("m/Y#", 4, "Y")
               .define("m/impl#[X][Tr]run().", 5, "run")
               .ref("m/via_x().", 5, "via_x")
               .define("m/impl#[Y][Tr]run().", 6, "run")
               .ref("m/via_y().", 6, "via_y")
               .define("m/go().", 7, "go")
               .ref("m/Y#", 7, "Y")
               .ref("m/Tr#run().", 7, "run"))
        ex = Doc("examples/e.rs", "fn main() { go(); }").ref("m/go().", 0, "go")
        lv = levels([src, ex])
        self.assertEqual(lv["src/m.rs::via_y"], "L1")
        self.assertEqual(lv["src/m.rs::via_x"], "L0")

    def test_type_built_only_through_its_constructor_is_reached(self):
        # `Y::new()` is the only use of Y: the call names `impl#[Y]new().`, not `Y#`
        text = ("pub trait Tr { fn run(&self); }\npub fn via_y() {}\npub struct Y;\n"
                "impl Y { pub fn new() -> Y { Y } }\nimpl Tr for Y { fn run(&self) { via_y(); } }\n"
                "pub fn go(t: &dyn Tr) { t.run(); }\npub fn make() { go(&Y::new()); }")
        src = (Doc("src/m.rs", text)
               .define("m/Tr#run().", 0, "run")
               .define("m/via_y().", 1, "via_y")
               .define("m/Y#", 2, "Y")
               .define("m/impl#[Y]new().", 3, "new")
               .define("m/impl#[Y][Tr]run().", 4, "run")
               .ref("m/via_y().", 4, "via_y")
               .define("m/go().", 5, "go")
               .ref("m/Tr#run().", 5, "run")
               .define("m/make().", 6, "make")
               .ref("m/go().", 6, "go")
               .ref("m/impl#[Y]new().", 6, "new"))
        ex = Doc("examples/e.rs", "fn main() { make(); }").ref("m/make().", 0, "make")
        self.assertEqual(levels([src, ex])["src/m.rs::via_y"], "L1")

    def test_external_trait_impl_waits_for_its_type(self):
        # `impl Drop for X` runs only if an X exists: unreached X -> not a root
        text = "pub fn helper() {}\npub struct X;\nimpl Drop for X {\n    fn drop(&mut self) { helper(); }\n}\npub fn make() { let _x = X; }"

        def src(made_by_example: bool) -> list[Doc]:
            d = (Doc("src/m.rs", text)
                 .define("m/helper().", 0, "helper")
                 .define("m/X#", 1, "X")
                 .define("m/impl#[X][Drop]drop().", 3, "drop", 3)
                 .ref("m/helper().", 3, "helper")
                 .define("m/make().", 5, "make")
                 .ref("m/X#", 5, "X"))
            ex = Doc("examples/e.rs", "fn main() { make(); }").ref("m/make().", 0, "make")
            return [d, ex] if made_by_example else [d]
        self.assertEqual(levels(src(False))["src/m.rs::helper"], "L0")
        self.assertEqual(levels(src(True))["src/m.rs::helper"], "L1")

    def test_trait_name_helper(self):
        self.assertEqual(sr._trait_name("TgsHooks"), "TgsHooks")
        self.assertEqual(sr._trait_name("`From<crate::solver_tgs::ImpulseCacheStats>`"), "From")
        self.assertEqual(sr._trait_name("`core::ops::Add<Self>`"), "Add")
        m = sr.IMPL_METHOD_RE.search(P + "m/impl#[`[Fix128; 3]`][`Add<Self>`]add().")
        self.assertEqual((m.group(2), m.group(3)), ("`Add<Self>`", "add"))

    def test_impl_of_external_trait_is_a_root(self):
        text = "pub fn helper() {}\npub struct X;\nimpl Drop for X {\n    fn drop(&mut self) { helper(); }\n}"
        src = (Doc("src/m.rs", text)
               .define("m/helper().", 0, "helper")
               .define("m/impl#[X][Drop]drop().", 3, "drop", 3)
               .implements("m/impl#[X][Drop]drop().", "core/ops/Drop#drop().", external=True)
               .ref("m/helper().", 3, "helper"))
        self.assertEqual(levels([src])["src/m.rs::helper"], "live")

    def test_reached_struct_reaches_its_field_types(self):
        text = "pub struct Cfg;\npub struct W {\n    pub cfg: Cfg,\n}"
        src = (Doc("src/m.rs", text)
               .define("m/Cfg#", 0, "Cfg")
               .define("m/W#", 1, "W", 3)
               .define("m/W#cfg.", 2, "cfg")
               .ref("m/Cfg#", 2, "Cfg"))
        ex = Doc("examples/e.rs", "fn main() { let w: W; }").ref("m/W#", 0, "W")
        self.assertEqual(levels([src, ex])["src/m.rs::Cfg"], "L1")
        self.assertEqual(levels([src])["src/m.rs::Cfg"], "L0")

    def test_tests_directory_is_not_a_root(self):
        t = Doc("tests/t.rs", "fn t() { used(); }").ref("used().", 0, "used")
        self.assertEqual(levels([lib_doc(), t])["src/lib.rs::used"], "L0")


class MacroGenerated(unittest.TestCase):
    """Types a macro_rules invocation generates, and the members its body writes.
    rust-analyzer defines the type at the invocation's argument token, emits no
    definition for the members and records no reference from the expanded body."""

    TEXT = ("pub struct Entry { pub hash: u64 }\n"
            "macro_rules! gen { ($n:ident) => { pub struct $n; impl $n { "
            "pub fn top_k(&self) -> Vec<Entry> { vec![] } pub fn fresh() -> Self { $n } } } }\n"
            "gen!(H5);\n"
            "gen!(H10);")

    def analysis(self, *refs: str):
        src = (Doc("src/m.rs", self.TEXT)
               .define("m/Entry#", 0, "Entry")
               .define("m/Entry#hash.", 0, "hash")
               .define("m/H5#", 2, "H5")
               .define("m/H10#", 3, "H10"))
        ex = Doc("examples/e.rs", "fn main() { x.top_k(); }")
        for r in refs:
            ex.ref(r, 0, "top_k")
        d = build([src, ex])
        s_ = d / "target" / "scip"
        return sr.analyze(d, [s_ / "native.scip", s_ / "wasm.scip", s_ / "fuzz.scip"])

    def test_each_generated_type_and_member_is_an_item(self):
        a = self.analysis("m/impl#[H5]top_k().")
        self.assertEqual(a.level["src/m.rs::H5::top_k"], "L1")
        self.assertEqual(a.level["src/m.rs::H5::fresh"], "L0")
        # the same member of the other invocation is a separate item
        self.assertEqual(a.level["src/m.rs::H10::top_k"], "L0")
        self.assertIn("src/m.rs::H10", a.level)
        self.assertEqual(a.generated_items, 2 * (2 + 1))
        # accounted for, so no longer listed as unindexed
        self.assertEqual(a.unindexed, [])

    def test_a_reached_member_reaches_what_the_body_names(self):
        self.assertEqual(self.analysis("m/impl#[H5]top_k().").level["src/m.rs::Entry"], "L1")
        self.assertEqual(self.analysis().level["src/m.rs::Entry"], "L0")

    SIBLINGS = ("macro_rules! sk { ($n:ident, $c:ident) => { pub struct $n; impl $n { "
                "pub const M: usize = 4; "
                "pub fn insert(&mut self) { self.insert_hash(Self::M as u64) } "
                "pub fn insert_hash(&mut self, h: u64) { helper_of(self) } "
                "fn helper_of(&self) { self.registers(); let _ = $c::new(); } "
                "pub fn registers(&self) {} "
                "pub fn unused_one(&self) {} } } }\n"
                "macro_rules! cm { ($n:ident) => { pub struct $n; impl $n { pub fn new() -> Self { $n } pub fn other() {} } } }\n"
                "sk!(S4, C8);\n"
                "cm!(C8);\n"
                "cm!(C9);")

    def sibling_levels(self):
        src = (Doc("src/m.rs", self.SIBLINGS)
               .define("m/S4#", 2, "S4")
               .define("m/C8#", 3, "C8")
               .define("m/C9#", 4, "C9"))
        ex = Doc("examples/e.rs", "fn main() { s.insert(); }").ref("m/impl#[S4]insert().", 0, "insert")
        return levels([src, ex])

    def test_a_reached_member_reaches_the_siblings_its_own_body_calls(self):
        lv = self.sibling_levels()
        self.assertEqual(lv["src/m.rs::S4::insert_hash"], "L1")  # self.insert_hash(..)
        self.assertEqual(lv["src/m.rs::S4::M"], "L1")            # Self::M
        self.assertEqual(lv["src/m.rs::S4::registers"], "L1")    # through the private helper_of
        # a member no reached body names stays unreached (not "every member of the body")
        self.assertEqual(lv["src/m.rs::S4::unused_one"], "L0")

    def test_a_metavariable_path_reaches_that_member_of_the_generated_types(self):
        lv = self.sibling_levels()
        self.assertEqual(lv["src/m.rs::C8::new"], "L1")    # $c::new()
        self.assertEqual(lv["src/m.rs::C8::other"], "L0")
        # only the type this invocation passed as `$c`, not every type with a `new`
        self.assertEqual(lv["src/m.rs::C9::new"], "L0")

    def test_a_reached_generated_type_reaches_its_field_types(self):
        text = ("pub struct Cell { pub v: u8 }\n"
                "macro_rules! g { ($n:ident) => { pub struct $n { c: Cell } impl $n { pub fn f(&self) {} } } }\n"
                "g!(G1);")
        src = Doc("src/m.rs", text).define("m/Cell#", 0, "Cell").define("m/G1#", 2, "G1")
        ex = Doc("examples/e.rs", "fn main() { let _x: G1; }").ref("m/G1#", 0, "G1")
        lv = levels([src, ex])
        self.assertEqual(lv["src/m.rs::Cell"], "L1")
        self.assertEqual(lv["src/m.rs::G1::f"], "L0")

    def test_split_members(self):
        parts, rest = sr.split_members("pub struct $n; impl $n { pub const M: u8 = 1; pub fn a(&self) { if x { b() } } fn c() {} }")
        self.assertEqual(sorted(parts), ["M", "a", "c"])
        self.assertIn("if x { b() }", parts["a"])
        self.assertNotIn("b()", rest)

    def test_a_used_field_reaches_its_type(self):
        # reading `entry.hash` shows an Entry exists, although nothing names `Entry#`
        src = (Doc("src/m.rs", "pub struct Entry { pub hash: u64 }\npub struct Other { pub n: u8 }")
               .define("m/Entry#", 0, "Entry")
               .define("m/Entry#hash.", 0, "hash")
               .define("m/Other#", 1, "Other")
               .define("m/Other#n.", 1, "n"))
        ex = Doc("examples/e.rs", "fn main() { let _ = e.hash; }").ref("m/Entry#hash.", 0, "hash")
        lv = levels([src, ex])
        self.assertEqual(lv["src/m.rs::Entry"], "L1")
        self.assertEqual(lv["src/m.rs::Other"], "L0")


class Main(unittest.TestCase):
    def test_missing_index_fails(self):
        d = Path(tempfile.mkdtemp())
        self.assertEqual(sr.main(["--root", str(d)]), 1)

    # each zero-count guard is tested with the other two conditions satisfied,
    # so removing one guard turns exactly one case green-when-it-should-fail
    EX = staticmethod(lambda: Doc("examples/e.rs", "fn main() { used(); }").ref("used().", 0, "used"))
    FFI = staticmethod(lambda: Doc("src/ffi/mod.rs", "pub extern \"C\" fn api() { used(); }").ref("used().", 0, "used"))
    FUZZ = staticmethod(lambda: Doc("fuzz/fuzz_targets/f.rs", "fuzz_target!(|d| { used(); });").ref("used().", 0, "used"))

    def test_all_three_conditions_met_passes(self):
        self.assertEqual(sr.main(["--root", str(build([lib_doc(), self.EX(), self.FFI(), self.FUZZ()]))]), 0)

    def test_no_example_references_fails(self):
        self.assertEqual(sr.main(["--root", str(build([lib_doc(), self.FFI(), self.FUZZ()]))]), 1)

    def test_no_binding_references_fails(self):
        self.assertEqual(sr.main(["--root", str(build([lib_doc(), self.EX(), self.FUZZ()]))]), 1)

    def test_no_items_fails(self):
        self.assertEqual(sr.main(["--root", str(build([self.EX(), self.FFI(), self.FUZZ()]))]), 1)

    def test_no_fuzz_reference_fails(self):
        self.assertEqual(sr.main(["--root", str(build([lib_doc(), self.EX(), self.FFI()]))]), 1)
        # a fuzz target that references only outside crates does not count either
        other = Doc("fuzz/fuzz_targets/f.rs", "fn main() { other(); }").ref("other_crate/other().", 0, "other")
        self.assertEqual(sr.main(["--root", str(build([lib_doc(), self.EX(), self.FFI(), other]))]), 1)

    def test_no_trait_impl_link_fails(self):
        bare = (Doc("src/lib.rs", "pub fn used() {}").define("used().", 0, "used"))
        self.assertEqual(sr.main(["--root", str(build([bare, self.EX(), self.FFI(), self.FUZZ()]))]), 1)
        # the same crate with one link passes, so the guard is the only difference
        linked = (Doc("src/lib.rs", "pub fn used() {}\ntrait Tr { fn run(&self); }\nimpl Tr for X { fn run(&self) {} }")
                  .define("used().", 0, "used").define("Tr#run().", 1, "run").define("impl#[X][Tr]run().", 2, "run"))
        self.assertEqual(sr.main(["--root", str(build([linked, self.EX(), self.FFI(), self.FUZZ()]))]), 0)

    def test_writes_ledger_and_compares_with_baseline(self):
        ex = Doc("examples/demo.rs", "fn main() { used(); }").ref("used().", 0, "used")
        ffi = Doc("src/ffi/mod.rs", "pub extern \"C\" fn api() { via_binding(); }").ref("via_binding().", 0, "via_binding")
        d = build([lib_doc(), ex, ffi, Main.FUZZ()])
        (d / "scripts").mkdir()
        (d / "scripts" / "wiring-baseline.txt").write_text("unwired src/lib.rs::via_binding\n", encoding="utf-8")
        out = d / "ledger.md"
        self.assertEqual(sr.main(["--root", str(d), "--write", str(out)]), 0)
        text = out.read_text(encoding="utf-8")
        self.assertIn("| L0 | not reached by any non-test code, examples included | 1 |", text)
        self.assertIn("- `src/lib.rs::unused`", text)
        self.assertIn("`src/lib.rs::via_binding` (live)", text)  # baseline says unwired, a binding reaches it


class Unindexed(unittest.TestCase):
    """pub items the index has no definition for (macro_rules bodies)."""

    TEXT = ("pub fn used() {}\nmacro_rules! gen { ($n:ident) => { impl $n { pub fn generated(&self) {} } } }\n"
            "trait Tr { fn run(&self); }\nimpl Tr for X { fn run(&self) {} }")

    def crate(self, baseline: str | None) -> Path:
        src = (Doc("src/lib.rs", self.TEXT)
               .define("used().", 0, "used")
               .define("Tr#run().", 2, "run")
               .define("impl#[X][Tr]run().", 3, "run"))
        d = build([src, Main.EX(), Main.FFI(), Main.FUZZ()])
        if baseline is not None:
            (d / "scripts").mkdir()
            (d / "scripts" / "integration-baseline.txt").write_text(baseline, encoding="utf-8")
        return d

    def test_item_in_a_macro_body_is_listed_and_an_indexed_one_is_not(self):
        d = self.crate(None)
        s_ = d / "target" / "scip"
        a = sr.analyze(d, [s_ / "native.scip", s_ / "wasm.scip", s_ / "fuzz.scip"])
        self.assertEqual(a.unindexed, ["src/lib.rs::generated"])

    def test_unlisted_unindexed_item_fails_the_check(self):
        d = self.crate("# none\n")
        self.assertEqual(sr.main(["--root", str(d), "--check-baseline"]), 1)

    def test_listed_unindexed_item_passes_and_a_stale_one_fails(self):
        d = self.crate("unindexed: src/lib.rs::generated\n")
        self.assertEqual(sr.main(["--root", str(d), "--check-baseline"]), 0)
        d = self.crate("unindexed: src/lib.rs::generated\nunindexed: src/lib.rs::gone\n")
        self.assertEqual(sr.main(["--root", str(d), "--check-baseline"]), 1)

    def test_write_baseline_records_the_unindexed_items(self):
        d = self.crate(None)
        (d / "scripts").mkdir()
        self.assertEqual(sr.main(["--root", str(d), "--write-baseline"]), 0)
        text = (d / "scripts" / "integration-baseline.txt").read_text(encoding="utf-8")
        self.assertIn("unindexed: src/lib.rs::generated\n", text)
        self.assertEqual(sr.main(["--root", str(d), "--check-baseline"]), 0)


class Baseline(unittest.TestCase):
    """--check-baseline: a ratchet on L0 items, like scripts/wiring-baseline.txt."""

    def crate(self, baseline: str | None) -> Path:
        ex = Doc("examples/demo.rs", "fn main() { used(); }").ref("used().", 0, "used")
        ffi = Doc("src/ffi/mod.rs", "pub extern \"C\" fn api() { via_binding(); }").ref("via_binding().", 0, "via_binding")
        d = build([lib_doc(), ex, ffi, Main.FUZZ()])  # L0: unused
        if baseline is not None:
            (d / "scripts").mkdir()
            (d / "scripts" / "integration-baseline.txt").write_text(baseline, encoding="utf-8")
        return d

    def check(self, baseline):
        return sr.main(["--root", str(self.crate(baseline)), "--check-baseline"])

    def test_baseline_listing_every_l0_passes(self):
        self.assertEqual(self.check("# header\nsrc/lib.rs::unused\n"), 0)

    def test_a_new_l0_item_fails(self):
        self.assertEqual(self.check("# nothing recorded\n"), 1)

    def test_a_stale_entry_fails(self):
        self.assertEqual(self.check("src/lib.rs::unused\nsrc/lib.rs::used\n"), 1)  # used is L1 now

    def test_missing_baseline_fails(self):
        self.assertEqual(self.check(None), 1)

    def test_write_baseline_then_check_round_trips(self):
        d = self.crate(None)
        (d / "scripts").mkdir()
        self.assertEqual(sr.main(["--root", str(d), "--write-baseline"]), 0)
        text = (d / "scripts" / "integration-baseline.txt").read_text(encoding="utf-8")
        self.assertIn("src/lib.rs::unused\n", text)
        self.assertEqual(sr.main(["--root", str(d), "--check-baseline"]), 0)


class KeepGraph(unittest.TestCase):
    """analyze(keep_graph=True): the same levels, plus the reach function and roots."""

    def docs(self):
        ex = Doc("examples/demo.rs", "fn main() { used(); }").ref("used().", 0, "used")
        ffi = Doc("src/ffi/mod.rs", "pub extern \"C\" fn api() { via_binding(); }").ref("via_binding().", 0, "via_binding")
        return [lib_doc(), ex, ffi]

    def run_analyze(self, keep):
        d = build(self.docs())
        s = d / "target" / "scip"
        return sr.analyze(d, [s / "native.scip", s / "wasm.scip", s / "fuzz.scip"], keep_graph=keep) if keep is not None \
            else sr.analyze(d, [s / "native.scip", s / "wasm.scip", s / "fuzz.scip"])

    def test_without_the_keyword_nothing_changes(self):
        plain, kept = self.run_analyze(None), self.run_analyze(True)
        self.assertEqual(plain.level, kept.level)
        self.assertEqual((plain.example_refs, plain.binding_refs), (kept.example_refs, kept.binding_refs))
        self.assertFalse(hasattr(plain, "reach"))
        self.assertFalse(hasattr(self.run_analyze(False), "reach"))

    def test_graph_and_roots_are_kept(self):
        a = self.run_analyze(True)
        P = "rust-analyzer cargo demo 0.1.0 "
        self.assertIn(P + "via_binding().", a.roots_binding)
        self.assertNotIn(P + "used().", a.roots_binding)
        self.assertIn(P + "used().", a.roots_example)
        self.assertTrue(a.roots_binding <= a.roots_core)
        self.assertEqual(set(a.roots_by_binding), {"src/ffi/"})
        self.assertEqual(a.roots_by_binding["src/ffi/"], a.roots_binding)

    def test_reach_answers_for_other_roots(self):
        a = self.run_analyze(True)
        P = "rust-analyzer cargo demo 0.1.0 "
        from_binding = a.reach(a.roots_binding)
        self.assertIn(P + "helper().", from_binding)       # via_binding -> helper
        self.assertNotIn(P + "used().", from_binding)      # only an example calls it
        self.assertEqual(a.reach({P + "used()."}), {P + "used()."})



class Bindings(unittest.TestCase):
    """SDF bindings are directories (src/ffi/, src/python/, src/godot/) and one file (src/wasm.rs)."""

    def test_files_under_a_binding_directory_are_that_binding(self):
        self.assertEqual(sr.binding_of("src/ffi/eval.rs"), "src/ffi/")
        self.assertEqual(sr.binding_of("src/python/helpers.rs"), "src/python/")
        self.assertEqual(sr.binding_of("src/godot/mod.rs"), "src/godot/")
        self.assertEqual(sr.binding_of("src/wasm.rs"), "src/wasm.rs")

    def test_other_files_are_not_bindings(self):
        for rel in ("src/ffi.rs", "src/ffi_helpers.rs", "src/wasm_util.rs", "src/eval/mod.rs", "examples/ffi/x.rs"):
            self.assertIsNone(sr.binding_of(rel), rel)

    def test_a_reference_from_a_nested_binding_file_is_a_root(self):
        py = Doc("src/python/node.rs", "pub fn api() { via_binding(); }").ref("via_binding().", 0, "via_binding")
        lv = levels([lib_doc(), py])
        self.assertEqual(lv["src/lib.rs::via_binding"], "live")

if __name__ == "__main__":
    unittest.main()
