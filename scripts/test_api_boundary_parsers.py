#!/usr/bin/env python3
"""Tests for scripts/api_boundary_parsers.py: the pure core (judge, definitions, owner_and_name) over a
small call graph written by hand in SCIP symbol form.

usage: python3 scripts/test_api_boundary_parsers.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_boundary_parsers as abp  # noqa: E402

V = "rust-analyzer cargo"
FROM_WGSL = f"{V} demo 1.0.0 compiled/msl/impl#[MslShader]from_wgsl()."
HELPER = f"{V} demo 1.0.0 compiled/msl/compile()."
GEN = f"{V} demo 1.0.0 shader/generate()."
VIA_PTR = f"{V} demo 1.0.0 shader/compile_with()."
PARSE_STR = f"{V} naga 24.0.0 front/wgsl/parse_str()."
TO_STRING = f"{V} alloc 1.0.0 string/ToString#to_string()."
DES_NEW = f"{V} serde_json 1.0.0 de/impl#[`Deserializer<R>`]new()."
FROM_STR_JSON = f"{V} serde_json 1.0.0 de/from_str()."
DERIVED_DE = f"{V} demo 1.0.0 tree/impl#[SdfTree][`Deserialize<'de>`]deserialize()."
READ_TREE = f"{V} demo 1.0.0 tree/read_tree()."
READ_TREE2 = f"{V} demo 1.0.0 tree/read_tree_derived()."
FFI_LOAD = f"{V} demo 1.0.0 ffi/io_ffi/alice_load()."
PY_LOAD = f"{V} demo 1.0.0 python/functions/load_text()."
WASM_EVAL = f"{V} demo 1.0.0 wasm/eval_text()."
WASM_VERSION = f"{V} demo 1.0.0 wasm/version()."
IO_VERSION = f"{V} demo 1.0.0 io/version()."
TRAIT_DECL = f"{V} demo 1.0.0 lang/Lang#decl()."
TRAIT_IMPL = f"{V} demo 1.0.0 lang/impl#[Wgsl][Lang]decl()."
FROM_STR = f"{V} demo 1.0.0 lang/impl#[ShaderLanguage][`FromStr`]from_str()."

# from_wgsl -> compile -> naga parse_str; generate -> to_string; compile_with calls through a
# function pointer it was given (the reference is to the parameter, so no edge to the parser)
EDGES = {
    FROM_WGSL: {HELPER},
    HELPER: {PARSE_STR},
    GEN: {TO_STRING},
    VIA_PTR: set(),
    TRAIT_DECL: set(),
    TRAIT_IMPL: {PARSE_STR},
    FROM_STR: set(),
    READ_TREE: {DES_NEW},          # serde_json::Deserializer::new(StrRead::new(s)) then deserialize
    READ_TREE2: {DERIVED_DE},      # T::deserialize(..) on a derive's impl
    FFI_LOAD: {FROM_STR_JSON},     # an extern "C" reader
    PY_LOAD: {FROM_STR_JSON},      # the Rust function a pyo3 wrapper calls
    WASM_EVAL: {PARSE_STR},        # a wasm32-only export
    WASM_VERSION: {TO_STRING},     # a wasm32-only export that parses nothing ...
    IO_VERSION: {FROM_STR_JSON},   # ... and another function of the same name that does
}
ALL = set(EDGES) | {PARSE_STR, TO_STRING, DES_NEW, FROM_STR_JSON, DERIVED_DE}
# the stored-form readers of the library clippy configuration (a subset, as the script reads them)
STORED = ["serde_json::Deserializer::new", "serde_json::de::from_str", "serde::de::Deserialize::deserialize",
          "bincode::serde::decode_from_slice"]
PARSERS = ["naga front/wgsl/parse_str()."]


def reach(start):
    seen, stack = set(start), list(start)
    while stack:
        for t in EDGES.get(stack.pop(), ()):
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


def run(listed, parsers=PARSERS, exempt=None, unindexed=None, stored=STORED):
    defs = abp.definitions(ALL, "demo")
    shorts = {abp.short(s) for s in ALL}
    by_desc = {}
    for sym in ALL:
        sh = abp.short(sym)
        if sh.startswith("demo "):
            by_desc.setdefault(sh[5:], set()).add(sym)
    return abp.judge(listed, parsers, exempt or {}, unindexed or {}, defs, reach, shorts, stored, by_desc,
                     ["naga"], {})


BASE = {
    "fn::MslShader::from_wgsl": "pending: compiles shader text from outside; no identifier yet",
    "fn::shader::generate": "local: generates shader text from a tree of this process",
}


class Keys(unittest.TestCase):
    def test_owner_and_name(self):
        self.assertEqual(abp.owner_and_name("fn::io::load"), (None, "load"))
        self.assertEqual(abp.owner_and_name("fn::MslShader::from_wgsl"), ("MslShader", "from_wgsl"))
        self.assertEqual(abp.owner_and_name("fn::<FromStr for ShaderLanguage>::from_str"),
                         ("FromStr for ShaderLanguage", "from_str"))
        self.assertEqual(abp.owner_and_name("fn::decl"), (None, "decl"))

    def test_definitions_index_methods_trait_impls_and_free_functions(self):
        d = abp.definitions(ALL, "demo")
        self.assertEqual(d[("MslShader", "from_wgsl")], {FROM_WGSL})
        self.assertEqual(d[(None, "generate")], {GEN})
        self.assertEqual(d[("FromStr for ShaderLanguage", "from_str")], {FROM_STR})
        self.assertIn(TRAIT_IMPL, d[("Wgsl", "decl")])
        self.assertNotIn(("naga", "parse_str"), d)  # other crates are not this crate's definitions


class StoredFormReaders(unittest.TestCase):
    def test_paths_match_their_index_symbols(self):
        shorts = {abp.short(s) for s in ALL}
        self.assertEqual(abp.match_path("serde_json::Deserializer::new", shorts),
                         {"serde_json de/impl#[`Deserializer<R>`]new()."})
        self.assertEqual(abp.match_path("serde_json::de::from_str", shorts), {"serde_json de/from_str()."})
        # a trait method matches its impls in any crate (a derive's deserialize here)
        self.assertEqual(abp.match_path("serde::de::Deserialize::deserialize", shorts),
                         {"demo tree/impl#[SdfTree][`Deserialize<'de>`]deserialize()."})
        self.assertEqual(abp.match_path("bincode::serde::decode_from_slice", shorts), set())

    def test_a_local_entry_reaching_deserializer_new_fails(self):
        problems, _ = run(dict(BASE, **{"fn::tree::read_tree": "local: builds a tree from its arguments"}))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("reaches `serde_json de/impl#[`Deserializer<R>`]new().`", problems[0])

    def test_a_local_entry_reaching_a_derived_deserialize_fails(self):
        problems, _ = run(dict(BASE, **{"fn::tree::read_tree_derived": "local: builds a tree from its arguments"}))
        self.assertEqual(len(problems), 1, problems)

    def test_without_the_derived_set_those_readers_pass(self):
        # the hand list alone does not name serde_json: the derived set is what catches them
        problems, _ = run(dict(BASE, **{"fn::tree::read_tree": "local: builds a tree from its arguments"}),
                          stored=["serde_json::de::from_str"])
        self.assertEqual(problems, [])

    def test_a_stored_form_set_that_matches_nothing_fails(self):
        problems, _ = run(BASE, stored=["serde_json::nothing_like_this"])
        self.assertTrue(any("matched the index" in p for p in problems), problems)


class StoredFormSetFromTheConfiguration(unittest.TestCase):
    SCRIPTS = Path(__file__).resolve().parent

    def _root(self, tmp: str, config: str | None) -> Path:
        import shutil
        root = Path(tmp)
        (root / "scripts").mkdir()
        shutil.copy(self.SCRIPTS / "clippy_lib_hidden_check.py", root / "scripts")
        if config is not None:
            (root / ".github" / "clippy-lib").mkdir(parents=True)
            (root / ".github" / "clippy-lib" / "clippy.toml").write_text(config, encoding="utf-8")
        return root

    def test_the_readers_are_read_from_the_configuration(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, 'disallowed-methods = [\n'
                              '  { path = "serde_json::from_str", reason = "a bare read of a stored form: x" },\n'
                              '  { path = "f32::max", reason = "the sign of zeros" },\n]\n')
            stored, problem = abp.stored_form_readers(root)
            self.assertEqual((stored, problem), (["serde_json::from_str"], None))

    def test_a_configuration_without_stored_form_readers_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, 'disallowed-methods = [\n  { path = "f32::max", reason = "the sign of zeros" },\n]\n')
            stored, problem = abp.stored_form_readers(root)
            self.assertEqual(stored, [])
            self.assertIn("names no stored-form reader", problem)
            # main() stops there with exit 1 (before the index is read)
            (root / "scripts" / "api_boundary.toml").write_text("parsers = []\n", encoding="utf-8")
            (root / "Cargo.toml").write_text('[package]\nname = "demo"\n', encoding="utf-8")
            import contextlib, io
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(abp.main(["--root", str(root), "--index", str(root / "none")]), 1)
            self.assertIn("names no stored-form reader", err.getvalue())

    def test_a_missing_configuration_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, problem = abp.stored_form_readers(self._root(tmp, None))
            self.assertIn("missing", problem)


class ParserCratesAreComplete(unittest.TestCase):
    OPEN = f"{V} image 0.25.0 images/dynimage/open()."
    SAVE = f"{V} image 0.25.0 images/dynimage/impl#[DynamicImage]save()."
    LOAD_IMG = f"{V} demo 1.0.0 terrain/load_heightmap()."

    def _run(self, listed, parsers, crates=("naga", "image"), not_parser=None):
        edges = dict(EDGES, **{self.LOAD_IMG: {self.OPEN, self.SAVE}})
        allsyms = ALL | {self.OPEN, self.SAVE, self.LOAD_IMG}

        def reach2(start):
            seen, stack = set(start), list(start)
            while stack:
                for t in edges.get(stack.pop(), ()):
                    if t not in seen:
                        seen.add(t)
                        stack.append(t)
            return seen

        defs = abp.definitions(allsyms, "demo")
        shorts = {abp.short(x) for x in allsyms}
        return abp.judge(listed, parsers, {}, {}, defs, reach2, shorts, STORED, {},
                         list(crates) if crates is not None else None, not_parser or {})

    def test_a_parse_like_function_missing_from_the_list_fails(self):
        # image::open is called (by load_heightmap) but only parse_str is listed
        problems, _ = self._run(BASE, PARSERS)
        self.assertTrue(any("`image images/dynimage/open().` is a parse-like function" in p for p in problems),
                        problems)
        self.assertFalse(any("save" in p for p in problems), problems)  # not a parse-like name

    def test_listed_it_is_followed_and_a_local_caller_fails(self):
        listed = dict(BASE, **{"fn::terrain::load_heightmap": "local: builds a heightmap from numbers"})
        problems, _ = self._run(listed, PARSERS + ["image images/dynimage/open()."])
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("fn::terrain::load_heightmap: decided `local` but reaches `image images/dynimage/open().`",
                      problems[0])

    def test_a_reasoned_not_parser_entry_passes_and_a_stale_one_fails(self):
        ok, _ = self._run(BASE, PARSERS, not_parser={"image images/dynimage/open()": "x" * 3})
        self.assertTrue(any("is a parse-like function" in p for p in ok), ok)  # wrong key: still missing
        ok, _ = self._run(BASE, PARSERS, not_parser={"image images/dynimage/open().": "opens a file for reading pixels"})
        self.assertEqual(ok, [])
        stale, _ = self._run(BASE, PARSERS + ["image images/dynimage/open()."],
                             not_parser={"image images/dynimage/load().": "a function no longer called"})
        self.assertTrue(any("[parser_not_parser] image images/dynimage/load().: not in the index" in p
                            for p in stale), stale)

    def test_a_missing_parser_crates_key_fails(self):
        problems, _ = self._run(BASE, PARSERS + ["image images/dynimage/open()."], crates=None)
        self.assertTrue(any("`parser_crates` is missing" in p for p in problems), problems)


class OtherEntryKinds(unittest.TestCase):
    def test_abi_binding_and_target_entries_are_followed(self):
        for key in ("abi::ffi::io_ffi::alice_load", "binding::pyo3::load_text",
                    "target::wasm32-unknown-unknown::wasm::eval_text"):
            problems, stats = run(dict(BASE, **{key: "local: plain numbers and handles of this process"}))
            self.assertEqual(len(problems), 1, (key, problems))
            self.assertIn(f"{key}: decided `local` but reaches", problems[0])
            self.assertEqual(stats["resolved"], 3)

    def test_an_entry_resolves_by_its_module_path_before_its_name(self):
        # wasm::version is clean; io::version of the same name reads JSON. The target prefix is
        # stripped so the exact path wins; matching by name alone would take both
        for key in ("target::wasm32-unknown-unknown::wasm::version", "fn::wasm::version"):
            problems, _ = run(dict(BASE, **{key: "local: returns this build's version string"}))
            self.assertEqual(problems, [], key)

    def test_impl_and_reexport_entries_are_counted_not_followed(self):
        listed = dict(BASE, **{"impl::Deserialize for SdfTree": "local: the type's field layout only",
                               "reexport::glam::Vec3": "local: a plain vector type of the API"})
        problems, stats = run(listed)
        self.assertEqual(problems, [])
        self.assertEqual(stats["not_followed"], {"impl": 1, "reexport": 1})


class Judge(unittest.TestCase):
    def test_a_consistent_manifest_passes(self):
        problems, stats = run(BASE)
        self.assertEqual(problems, [])
        self.assertEqual(stats["resolved"], 2)

    def test_a_local_entry_that_reaches_a_parser_fails(self):
        # the MslShader::from_wgsl mistake: decided local, compiles outside WGSL with naga
        listed = dict(BASE, **{"fn::MslShader::from_wgsl": "local: generates shader text from a tree"})
        problems, _ = run(listed)
        self.assertEqual(len(problems), 1)
        self.assertIn("fn::MslShader::from_wgsl: decided `local` but reaches `naga front/wgsl/parse_str().`",
                      problems[0])

    def test_the_same_entry_with_a_producer_form_decision_passes(self):
        for decision in ("pending: compiles shader text from outside; no identifier yet",
                         "handshake: some_test"):
            problems, _ = run(dict(BASE, **{"fn::MslShader::from_wgsl": decision}))
            self.assertEqual(problems, [], decision)

    def test_an_exemption_with_a_reason_passes_and_a_stale_one_fails(self):
        listed = dict(BASE, **{"fn::MslShader::from_wgsl": "local: parses only text this crate wrote"})
        ok, stats = run(listed, exempt={"fn::MslShader::from_wgsl": "parses only text this crate wrote"})
        self.assertEqual(ok, [])
        self.assertEqual(stats["exempt"], 1)
        # exempting an entry that reaches no parser, or one that is not local, is stale
        p1, _ = run(BASE, exempt={"fn::shader::generate": "generates text, nothing to parse"})
        self.assertTrue(any("reaches no listed parser any more" in p for p in p1), p1)
        p2, _ = run(BASE, exempt={"fn::MslShader::from_wgsl": "compiles outside text on purpose"})
        self.assertTrue(any("is not `local`" in p for p in p2), p2)
        p3, _ = run(BASE, exempt={"fn::gone::away": "an entry that no longer exists"})
        self.assertTrue(any("not an entry of [boundary]" in p for p in p3), p3)

    def test_a_short_reason_fails(self):
        listed = dict(BASE, **{"fn::MslShader::from_wgsl": "local: x"})
        problems, _ = run(listed, exempt={"fn::MslShader::from_wgsl": "short"})
        self.assertTrue(any("reason shorter than 12" in p for p in problems), problems)

    def test_a_missing_parser_list_fails(self):
        # the ratchet: this repository adopted the rule, so dropping the key is not a pass
        problems, _ = run(BASE, parsers=None)
        self.assertEqual(len(problems), 1)
        self.assertIn("`parsers` is missing", problems[0])

    def test_an_empty_parser_list_fails(self):
        problems, _ = run(BASE, parsers=[])
        self.assertTrue(any("`parsers` is empty" in p for p in problems), problems)

    def test_a_parser_the_index_does_not_contain_fails(self):
        problems, _ = run(BASE, parsers=PARSERS + ["naga front/wgsl/parse_strr()."])
        self.assertTrue(any("`naga front/wgsl/parse_strr().` is not in the index" in p for p in problems),
                        problems)

    def test_an_unresolved_entry_must_be_listed_and_a_resolved_one_must_not(self):
        listed = dict(BASE, **{"fn::macro::generated": "local: generated by a macro of this crate"})
        problems, _ = run(listed)
        self.assertTrue(any("fn::macro::generated: the index has no definition" in p for p in problems), problems)
        ok, stats = run(listed, unindexed={"fn::macro::generated": "a macro body the index does not see"})
        self.assertEqual(ok, [])
        self.assertEqual(stats["unindexed"], 1)
        stale, _ = run(BASE, unindexed={"fn::shader::generate": "a macro body the index does not see"})
        self.assertTrue(any("resolves it now (stale)" in p for p in stale), stale)

    def test_a_trait_method_named_without_its_trait_follows_every_method_of_that_name(self):
        listed = {"fn::decl": "local: shader text builders over names and expressions"}
        problems, _ = run(listed)
        self.assertTrue(any("fn::decl: decided `local` but reaches" in p for p in problems), problems)

    def test_nothing_resolved_fails(self):
        problems, _ = run({"fn::nowhere::f": "local: does not exist in the index at all"},
                          unindexed={"fn::nowhere::f": "not in the index for this test"})
        self.assertTrue(any("compared nothing" in p for p in problems), problems)

    def test_a_call_through_a_function_pointer_is_not_seen(self):
        # known residual (module docstring): compile_with(f) calls the parser through `f`, but the
        # index records a reference to the parameter only, so no edge leads to the parser and a
        # `local` decision passes. Kept as a test so the gap stays visible; if the walk learns to
        # follow it, this test fails and the docstring must change with it.
        problems, _ = run(dict(BASE, **{"fn::shader::compile_with": "local: runs the caller's compiler"}))
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
