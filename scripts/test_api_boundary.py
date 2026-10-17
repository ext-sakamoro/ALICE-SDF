#!/usr/bin/env python3
"""Tests of scripts/api_boundary.py on hand-built rustdoc JSON (format 60).

The shapes are the ones rustdoc emits for the forms listed in each test name;
the end-to-end run against the real crate is the CI step itself.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_boundary  # noqa: E402

U8 = {"primitive": "u8"}
U32 = {"primitive": "u32"}


def ref(t):
    return {"borrowed_ref": {"lifetime": None, "is_mutable": False, "type": t}}


def path(name, *args, constraints=()):
    return {"resolved_path": {"path": name, "id": None, "args": {"angle_bracketed": {
        "args": [{"type": a} for a in args], "constraints": list(constraints)}}}}


def bound(trait):
    return {"trait_bound": {"trait": trait["resolved_path"], "generic_params": [], "modifier": "none"}}


def fn(name, inputs=(), output=None, generics=None, abi="Rust"):
    return {"crate_id": 0, "name": name, "visibility": "public", "inner": {"function": {
        "sig": {"inputs": [[f"p{i}", t] for i, t in enumerate(inputs)], "output": output, "is_c_variadic": False},
        "generics": generics or {"params": [], "where_predicates": []},
        "header": {"is_const": False, "is_unsafe": False, "is_async": False, "abi": abi}}}}


def doc(items: dict, version: int = 60) -> dict:
    return {"format_version": version, "root": "0", "index": {str(k): v for k, v in items.items()}, "paths": {}}


class EnumerateTest(unittest.TestCase):
    def keys(self, items):
        return api_boundary.enumerate_boundary(doc(items))

    def test_integer_only_fn_is_not_listed(self):
        self.assertEqual(self.keys({1: fn("plain", [U32], U32)}), [])

    def test_byte_shapes_in_parameters_are_listed(self):
        shapes = {
            "slice": ref({"slice": U8}),
            "array": {"array": {"type": U8, "len": "8"}},
            "vec": path("Vec", U8),
            "boxed": path("Box", {"slice": U8}),
            "cow": path("Cow", {"slice": U8}),
            "str": ref({"primitive": "str"}),
            "string": path("String"),
            "chars": ref({"slice": {"primitive": "char"}}),
            "as_ref": {"impl_trait": [bound(path("AsRef", {"slice": U8}))]},
            "iterator": {"impl_trait": [bound(path("Iterator", constraints=[
                {"name": "Item", "args": None, "binding": {"equality": {"type": U8}}}]))]},
            "dyn_read": ref({"dyn_trait": {"traits": [{"trait": path("Read")["resolved_path"],
                                                        "generic_params": []}], "lifetime": None}}),
            "tuple": {"tuple": [U32, ref({"slice": U8})]},
        }
        for name, t in shapes.items():
            self.assertEqual(self.keys({1: fn(name, [t])}), [f"fn::{name}"], name)

    def test_byte_return_type_is_listed(self):
        self.assertEqual(self.keys({1: fn("to_bytes", [], path("Vec", U8))}), ["fn::to_bytes"])

    def test_generic_bound_and_where_clause_are_listed(self):
        g = {"params": [{"name": "B", "kind": {"type": {"bounds": [bound(path("AsRef", {"slice": U8}))],
                                                        "default": None, "is_synthetic": False}}}],
             "where_predicates": []}
        w = {"params": [{"name": "B", "kind": {"type": {"bounds": [], "default": None,
                                                        "is_synthetic": False}}}],
             "where_predicates": [{"bound_predicate": {"type": {"generic": "B"},
                                                       "bounds": [bound(path("BufRead"))],
                                                       "generic_params": []}}]}
        self.assertEqual(self.keys({1: fn("g", [{"generic": "B"}], None, g)}), ["fn::g"])
        self.assertEqual(self.keys({1: fn("w", [{"generic": "B"}], None, w)}), ["fn::w"])

    def test_type_alias_is_followed(self):
        alias = {"crate_id": 0, "name": "Blob", "inner": {"type_alias": {"type": path("Vec", U8),
                                                                          "generics": {}}}}
        arg = {"resolved_path": {"path": "Blob", "id": 2, "args": None}}
        self.assertEqual(self.keys({1: fn("takes_alias", [arg]), 2: alias}), ["fn::takes_alias"])

    def test_extern_abi_is_always_listed(self):
        self.assertEqual(self.keys({1: fn("as_x", [U32], U32, abi={"C": {"unwind": False}})}), ["abi::as_x"])

    def test_serde_and_conversion_impls_are_listed_and_blanket_impls_are_not(self):
        def impl(trait, for_, blanket=None, items=()):
            return {"crate_id": 0, "inner": {"impl": {"trait": trait["resolved_path"], "for": for_,
                                                      "items": list(items), "blanket_impl": blanket,
                                                      "is_synthetic": False}}}
        t = path("Thing")
        items = {
            1: impl(path("Deserialize"), t),
            2: impl(path("Serialize"), t, blanket={"generic": "T"}),
            3: impl(path("TryFrom", ref({"slice": U8})), t),
            4: impl(path("From", U32), t),
            5: impl(path("FromStr"), t),
        }
        self.assertEqual(self.keys(items), ["impl::Deserialize for Thing", "impl::FromStr for Thing",
                                            "impl::TryFrom for Thing"])

    def test_reexport_of_an_outside_item_is_listed(self):
        use = {"crate_id": 0, "visibility": "public", "inner": {"use": {"source": "core::str::from_utf8",
                                                                          "name": "from_utf8", "id": 99,
                                                                          "is_glob": False}}}
        self.assertEqual(self.keys({1: use}), ["reexport::core::str::from_utf8"])

    def test_repeated_keys_are_numbered(self):
        self.assertEqual(self.keys({1: fn("new", [U8]), 2: fn("new", [U8])}), ["fn::new", "fn::new#2"])

    def test_unsupported_format_stops(self):
        with self.assertRaises(SystemExit):
            api_boundary.enumerate_boundary(doc({}, version=61))


class CheckTest(unittest.TestCase):
    def root(self, test_src: str = "") -> Path:
        d = Path(tempfile.mkdtemp())
        (d / "tests").mkdir()
        (d / "tests" / "t.rs").write_text(test_src, encoding="utf-8")
        return d

    def test_listed_items_pass_and_drift_fails(self):
        d = doc({1: fn("read_it", [ref({"slice": U8})]), 2: fn("send", [path("String")])})
        root = self.root("fn read_it_refuses_another_identifier() {}\n")
        ok = {"fn::read_it": "handshake: read_it_refuses_another_identifier",
              "fn::send": "local: a label chosen by the caller"}
        self.assertEqual(api_boundary.check(root, d, ok)[0], [])
        missing = dict(ok)
        del missing["fn::send"]
        self.assertTrue(any("fn::send" in p and "not in" in p for p in api_boundary.check(root, d, missing)[0]))
        stale = dict(ok, **{"fn::gone": "local: was removed some time ago"})
        self.assertTrue(any("stale" in p for p in api_boundary.check(root, d, stale)[0]))
        no_test = dict(ok, **{"fn::read_it": "handshake: no_such_test"})
        self.assertTrue(any("no_such_test" in p for p in api_boundary.check(root, d, no_test)[0]))
        bad = dict(ok, **{"fn::send": "short"})
        self.assertTrue(any("must start" in p for p in api_boundary.check(root, d, bad)[0]))

    def test_nothing_enumerated_fails(self):
        problems, _ = api_boundary.check(self.root(), doc({1: fn("plain", [U32])}), {})
        self.assertEqual(problems, ["enumerated no boundary item (compared nothing)"])



C_ABI = {"C": {"unwind": False}}


def impl_of(type_name, items, trait=None):
    return {"crate_id": 0, "name": None, "visibility": "default", "inner": {"impl": {
        "for": path(type_name), "trait": trait, "items": items, "blanket_impl": None,
        "is_synthetic": False}}}


def with_paths(d: dict, paths: dict) -> dict:
    d["paths"] = {str(k): {"path": v} for k, v in paths.items()}
    return d


class PrivateRunTest(unittest.TestCase):
    """the second rustdoc run with --document-private-items: abi anywhere, and the bindings"""

    def test_an_exported_symbol_in_a_private_module_is_listed_from_the_private_run(self):
        public = doc({1: fn("plain", [U32])})
        private = with_paths(doc({1: fn("plain", [U32]), 2: fn("lib_compile", [U32], abi=C_ABI)}),
                             {2: ["lib", "ffi", "compilation", "lib_compile"]})
        self.assertEqual(api_boundary.enumerate_boundary(public, private),
                         ["abi::ffi::compilation::lib_compile"])

    def test_with_a_private_run_a_public_abi_fn_is_listed_once(self):
        f = fn("exported", [U32], abi=C_ABI)
        public = with_paths(doc({1: f}), {1: ["lib", "ffi", "exported"]})
        private = with_paths(doc({1: f}), {1: ["lib", "ffi", "exported"]})
        self.assertEqual(api_boundary.enumerate_boundary(public, private), ["abi::ffi::exported"])

    def test_pyo3_wrappers_are_bindings_not_abi(self):
        private = doc({
            1: fn("__pyfunction_to_mesh", [U32]),
            2: impl_of("PySdfNode", [3]),
            3: fn("__pymethod_union__", [U32]),
            4: fn("__pyo3_init", [], abi=C_ABI),
            5: fn("helper", [U32]),
        })
        keys = api_boundary.enumerate_boundary(doc({}), private, ("pyo3",))
        self.assertEqual(keys, ["binding::pyo3::PySdfNode::union", "binding::pyo3::__pyo3_init",
                                "binding::pyo3::to_mesh"])

    def test_pyo3_wrappers_are_not_bindings_unless_declared(self):
        private = doc({1: fn("__pyfunction_to_mesh", [U32])})
        self.assertEqual(api_boundary.enumerate_boundary(doc({}), private), [])

    def test_a_target_run_lists_the_functions_only_that_target_has(self):
        native = with_paths(doc({1: fn("shared", [U32])}), {1: ["lib", "shared"]})
        target = with_paths(doc({1: fn("shared", [U32]), 2: fn("eval_at", [U32])}),
                            {1: ["lib", "shared"], 2: ["lib", "wasm", "eval_at"]})
        keys = api_boundary.enumerate_boundary(native, doc({}), (), ({"name": "wasm32", "doc": target},))
        self.assertEqual(keys, ["target::wasm32::wasm::eval_at"])

    def test_a_target_run_without_its_own_functions_fails(self):
        native = with_paths(doc({1: fn("shared", [ref({"slice": U8})])}), {1: ["lib", "shared"]})
        target = with_paths(doc({1: fn("shared", [U32])}), {1: ["lib", "shared"]})
        root = Path(tempfile.mkdtemp())
        problems, _ = api_boundary.check(root, native, {"fn::shared": "local: bytes the caller owns"},
                                         doc({}), (), ({"name": "wasm32", "doc": target},))
        self.assertTrue(any("wasm32 run lists no function" in p for p in problems), problems)


class DeclaredBindingsAndPendingTest(unittest.TestCase):
    def root(self) -> Path:
        d = Path(tempfile.mkdtemp())
        (d / "tests").mkdir()
        (d / "tests" / "t.rs").write_text("", encoding="utf-8")
        return d

    def test_a_declared_generator_without_entry_points_fails(self):
        private = doc({1: fn("exported", [U32], abi=C_ABI)})
        listed = {"abi::exported": "local: integers the host passes in"}
        problems, _ = api_boundary.check(self.root(), doc({}), listed, private, ("pyo3",))
        self.assertTrue(any("pyo3 is declared but no entry point" in p for p in problems), problems)

    def test_an_unknown_generator_fails(self):
        private = doc({1: fn("exported", [U32], abi=C_ABI)})
        listed = {"abi::exported": "local: integers the host passes in"}
        problems, _ = api_boundary.check(self.root(), doc({}), listed, private, ("cxx",))
        self.assertTrue(any("unknown generator" in p for p in problems), problems)

    def pending_problems(self, status):
        d = doc({1: fn("read_saved", [ref({"slice": U8})])})
        listed = {"fn::read_saved": "pending: reads this crate's saved file, which has no identifier yet"}
        return api_boundary.check(self.root(), d, listed, g7_status=status)[0]

    def test_pending_is_allowed_while_g7_is_not_pass(self):
        self.assertEqual(self.pending_problems("unverified"), [])
        self.assertEqual(self.pending_problems("fail-known"), [])

    def test_pending_with_g7_pass_fails(self):
        self.assertTrue(any("pending entries while" in p for p in self.pending_problems("pass")))

    def test_pending_without_a_conformance_manifest_fails(self):
        self.assertTrue(any("pending entries while" in p for p in self.pending_problems(None)))

    def test_a_short_pending_reason_fails(self):
        d = doc({1: fn("read_saved", [ref({"slice": U8})])})
        problems = api_boundary.check(self.root(), d, {"fn::read_saved": "pending: later"},
                                      g7_status="unverified")[0]
        self.assertTrue(any("reason shorter" in p for p in problems), problems)

    def test_g7_status_is_read_from_the_conformance_manifest(self):
        r = self.root()
        (r / "alice-law-conformance.toml").write_text('[G7]\nstatus = "unverified"\n', encoding="utf-8")
        self.assertEqual(api_boundary.g7_status(r), "unverified")
        self.assertIsNone(api_boundary.g7_status(self.root()))


class MissingTargetTest(unittest.TestCase):
    """--skip-missing-targets (preflight): an uninstalled target is skipped out loud; without
    the flag (CI) the run fails"""

    def setUp(self):
        import contextlib
        import io
        from unittest import mock
        self.mock, self.io, self.ctx = mock, io, contextlib
        self.root = Path(tempfile.mkdtemp())
        (self.root / "scripts").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "scripts" / "api_boundary.toml").write_text(
            'features = ""\n'
            'targets = [{ target = "wasm32-unknown-unknown", features = "wasm" }]\n\n'
            '[boundary]\n'
            '"fn::read_it" = "local: bytes the caller owns and keeps"\n'
            '"target::wasm32-unknown-unknown::wasm::eval_at" = "local: numbers in, numbers out"\n',
            encoding="utf-8")
        native = with_paths(doc({1: fn("read_it", [ref({"slice": U8})])}), {1: ["lib", "read_it"]})
        self.native = native

    def fake_rustdoc(self, missing: bool):
        target_doc = with_paths(doc({1: fn("read_it", [U32]), 2: fn("eval_at", [U32])}),
                                {1: ["lib", "read_it"], 2: ["lib", "wasm", "eval_at"]})

        def run(root, features, private=False, target=None):
            if target is not None:
                if missing:
                    raise SystemExit("api_boundary: rustdoc failed")
                return target_doc
            return self.native
        return run

    def run_main(self, missing: bool, flag: bool):
        installed = "" if missing else "wasm32-unknown-unknown\n"
        rustup = self.mock.Mock(stdout=installed)
        out = self.io.StringIO()
        with self.mock.patch.object(api_boundary, "rustdoc", self.fake_rustdoc(missing)), \
                self.mock.patch.object(api_boundary.subprocess, "run", return_value=rustup), \
                self.ctx.redirect_stdout(out), self.ctx.redirect_stderr(self.io.StringIO()):
            try:
                rc = api_boundary.main(["--root", str(self.root)] + (["--skip-missing-targets"] if flag else []))
            except SystemExit as e:
                rc = f"exit: {e}"
        return rc, out.getvalue()

    def test_a_missing_target_is_skipped_out_loud_with_the_flag(self):
        rc, out = self.run_main(missing=True, flag=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("PREFLIGHT-SKIP: api_boundary target wasm32-unknown-unknown", out)

    def test_a_missing_target_fails_without_the_flag(self):
        rc, _ = self.run_main(missing=True, flag=False)
        self.assertNotEqual(rc, 0)

    def test_an_installed_target_is_checked_with_the_flag(self):
        rc, out = self.run_main(missing=False, flag=True)
        self.assertEqual(rc, 0, out)
        self.assertNotIn("PREFLIGHT-SKIP", out)
        self.assertIn("1 target", out)

if __name__ == "__main__":
    unittest.main()
