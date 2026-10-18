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
}
ALL = set(EDGES) | {PARSE_STR, TO_STRING}
PARSERS = ["naga front/wgsl/parse_str()."]


def reach(start):
    seen, stack = set(start), list(start)
    while stack:
        for t in EDGES.get(stack.pop(), ()):
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


def run(listed, parsers=PARSERS, exempt=None, unindexed=None):
    defs = abp.definitions(ALL, "demo")
    shorts = {abp.short(s) for s in ALL}
    return abp.judge(listed, parsers, exempt or {}, unindexed or {}, defs, reach, shorts)


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
