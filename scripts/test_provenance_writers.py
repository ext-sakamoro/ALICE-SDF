#!/usr/bin/env python3
"""Tests of scripts/provenance_writers.py: each failure it promises, on a small tree."""

from __future__ import annotations

import atexit
import shutil
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance_writers as pw  # noqa: E402

SRC = """pub fn export_a(p: &Path) {}
pub fn export_a_with_provenance(p: &Path, id: &Provenance) {}
pub fn import_a(p: &Path) {}
fn export_private() {}
pub fn save_b() {}

#[cfg(test)]
mod tests {
    pub fn export_in_tests() {}
}
"""
TEST = (
    'const WRITERS: &[(&str, &str, Write)] = &[\n    ("src/io/a.rs::export_a", "a.x", |m, p, id| {}),\n];\n'
    "fn round_trip() { io::export_a_with_provenance(p, id); }\n"
)


def _mkdtemp(**kwargs) -> str:
    """tempfile.mkdtemp, removed when the interpreter exits (after a failing test too)"""
    d = tempfile.mkdtemp(**kwargs)
    atexit.register(shutil.rmtree, d, True)
    return d


def tree(
    src: str = SRC,
    test: str = TEST,
    pending: str = "expected-writers 2\nsrc/io/a.rs::save_b\n",
    not_writers: str | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    root = _mkdtemp()
    os.makedirs(os.path.join(root, "src/io"))
    os.makedirs(os.path.join(root, "tests"))
    os.makedirs(os.path.join(root, "scripts"))
    files = [("src/io/a.rs", src), (pw.TEST_FILE, test), (pw.PENDING_FILE, pending)]
    if not_writers is not None:
        files.append((pw.NOT_WRITERS_FILE, not_writers))
    files += list((extra or {}).items())
    for rel, text in files:
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
            fh.write(text)
    return root


def macos_writer_tree() -> str:
    """A git tree with a SCIP index of its HEAD: a writer compiled only for macOS (its
    definition and its call of std::fs::write are in macos.scip only)."""
    import subprocess
    import test_scip_reach as tsr
    text = "pub fn save_mac(p: &Path, b: &[u8]) { std::fs::write(p, b); }"
    mac = tsr.Doc("src/metal.rs", text, indexes=("macos.scip",)).define("metal/save_mac().", 0, "save_mac")
    mac.occ.append(tsr._f(1, tsr._packed(mac._pos(0, "write"))) + tsr._f(2, "rust-analyzer cargo std 1.0.0 fs/write()."))
    wasm = tsr.Doc("src/wasm.rs", "pub fn in_wasm() {}", indexes=("wasm.scip",)).define("wasm/in_wasm().", 0, "in_wasm")
    d = tsr.build([tsr.lib_doc(), mac, wasm])
    git = ["git", "-C", str(d), "-c", "user.name=T", "-c", "user.email=t@example.invalid"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "c"], check=True)
    head = subprocess.run(git[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    (d / "target" / "scip" / "commit").write_text(f"{head} clean\n", encoding="utf-8")
    return str(d)


class MacosIndex(unittest.TestCase):
    def test_a_writer_only_the_macos_index_has_reaches_its_sink(self):
        root = macos_writer_tree()
        out, commit, problem = pw.reach_candidates(root, os.path.join(root, "target", "scip"))
        self.assertIsNone(problem)
        self.assertEqual(out.get("src/metal.rs::save_mac"), {"std fs/write()."})

    def test_without_the_macos_index_it_is_a_problem(self):
        root = macos_writer_tree()
        os.remove(os.path.join(root, "target", "scip", "macos.scip"))
        out, _commit, problem = pw.reach_candidates(root, os.path.join(root, "target", "scip"))
        self.assertEqual(out, {})
        self.assertIn("macos.scip", problem or "")


def errors(root: str, reached: dict[str, set[str]] | None = None) -> list[str]:
    return pw.check(root, reached)[0]


class Writers(unittest.TestCase):
    def test_the_small_tree_passes(self):
        self.assertEqual(errors(tree()), [])
        self.assertEqual(pw.check(tree())[1], 2)

    def test_variants_private_fns_importers_and_the_test_module_are_not_writers(self):
        self.assertEqual(sorted(pw.writers(tree())), ["src/io/a.rs::export_a", "src/io/a.rs::save_b"])

    def test_a_new_writer_fails_twice_count_and_coverage(self):
        e = errors(tree(src=SRC.replace("#[cfg(test)]", "pub fn write_c() {}\n\n#[cfg(test)]")))
        self.assertTrue(any("3 writers found" in x for x in e), e)
        self.assertTrue(any("write_c: neither covered" in x for x in e), e)

    def test_a_variant_with_no_base_writer_fails(self):
        src = SRC.replace("#[cfg(test)]", "pub fn export_orphan_with_stamp(p: &Path) {}\n\n#[cfg(test)]")
        test = TEST + "fn t() { io::export_orphan_with_stamp(p); }\n"
        e = errors(tree(src=src, test=test))
        self.assertTrue(any("export_orphan_with_stamp: extends no writer" in x for x in e), e)

    def test_a_variant_the_test_never_calls_fails(self):
        src = SRC.replace("#[cfg(test)]", "pub fn export_a_with_stamp(p: &Path) {}\n\n#[cfg(test)]")
        e = errors(tree(src=src))
        self.assertTrue(any("export_a_with_stamp: not called" in x for x in e), e)
        e = errors(tree(src=src, test=TEST + "fn t() { io::export_a_with_stamp(p); }\n"))
        self.assertEqual(e, [])

    def test_qualified_pub_fns_are_writers(self):
        for q in ["unsafe ", 'extern "C" ', "const ", "async ", "pub(crate) "[:0]]:
            src = SRC.replace("#[cfg(test)]", f"pub {q}fn export_q() {{}}\n\n#[cfg(test)]")
            e = errors(tree(src=src))
            self.assertTrue(any("export_q: neither covered" in x for x in e), (q, e))

    def test_a_stray_cfg_test_item_hides_nothing_after_it(self):
        src = SRC.replace("pub fn save_b() {}", "#[cfg(test)]\nconst P: u8 = 0;\npub fn save_b() {}\npub fn export_late() {}")
        e = errors(tree(src=src))
        self.assertTrue(any("export_late: neither covered" in x for x in e), e)

    def test_a_writer_inside_a_cfg_test_module_is_not_counted(self):
        self.assertNotIn("src/io/a.rs::export_in_tests", pw.writers(tree()))

    def test_a_variant_named_only_in_a_comment_or_string_is_not_called(self):
        src = SRC.replace("#[cfg(test)]", "pub fn export_a_with_stamp(p: &Path) {}\n\n#[cfg(test)]")
        for t in ["// io::export_a_with_stamp(p)\n", "/* export_a_with_stamp() */\n", 'const S: &str = "export_a_with_stamp(p)";\n']:
            e = errors(tree(src=src, test=TEST + t))
            self.assertTrue(any("export_a_with_stamp: not called" in x for x in e), (t, e))

    def test_a_removed_writer_fails_on_the_count(self):
        e = errors(tree(src=SRC.replace("pub fn save_b() {}\n", ""), pending="expected-writers 2\n"))
        self.assertTrue(any("1 writers found" in x for x in e), e)

    def test_a_pending_line_already_covered_fails(self):
        e = errors(tree(pending="expected-writers 2\nsrc/io/a.rs::save_b\nsrc/io/a.rs::export_a\n"))
        self.assertTrue(any("export_a is covered" in x for x in e), e)

    def test_a_pending_line_that_is_no_writer_fails(self):
        e = errors(tree(pending="expected-writers 2\nsrc/io/a.rs::save_b\nsrc/io/a.rs::gone\n"))
        self.assertTrue(any("gone is not a writer" in x for x in e), e)

    def test_a_covered_name_that_is_no_writer_fails(self):
        e = errors(tree(test=TEST.replace("export_a", "export_z")))
        self.assertTrue(any("export_z is not a writer" in x for x in e), e)
        self.assertTrue(any("export_a: neither covered" in x for x in e), e)

    def test_a_missing_count_fails(self):
        self.assertTrue(any("expected-writers" in x for x in errors(tree(pending="src/io/a.rs::save_b\n"))))

    def test_a_duplicate_pending_line_fails(self):
        e = errors(tree(pending="expected-writers 2\nsrc/io/a.rs::save_b\nsrc/io/a.rs::save_b\n"))
        self.assertTrue(any("listed twice" in x for x in e), e)

    def test_no_writer_found_fails(self):
        self.assertEqual(errors(tree(src="pub fn import_only() {}\n")), ["no writer found (compared nothing)"])

    def test_main_returns_1_on_error_and_0_when_clean(self):
        self.assertEqual(pw.main(["x", tree()]), 0)
        self.assertEqual(pw.main(["x", tree(src="")]), 1)

    def test_a_writer_name_anywhere_under_src_is_a_candidate(self):
        # a new module outside src/io with an encoder: counted, so it must be classified
        e = errors(tree(extra={"src/mesh/b.rs": "pub fn encode_foo(m: &Mesh) -> Vec<u8> { Vec::new() }\n"}))
        self.assertTrue(any("src/mesh/b.rs::encode_foo: neither covered" in x for x in e), e)
        self.assertTrue(any("3 writers found" in x for x in e), e)
        for name in ["bake_to_grid", "serialize", "persist_all", "dump_state", "to_vec"]:
            e = errors(tree(extra={"src/b.rs": f"pub fn {name}() {{}}\n"}))
            self.assertTrue(any(f"src/b.rs::{name}: neither covered" in x for x in e), (name, e))

    def test_a_write_sink_makes_any_name_a_candidate(self):
        for body in [
            "pub fn store(w: &mut impl Write) {}\n",
            "pub fn store<W: std::io::Write>(w: W) {}\n",
            "pub fn store(w: &mut dyn Write) {}\n",
            "pub fn store(p: &Path) { std::fs::write(p, b\"x\").unwrap(); }\n",
            "pub fn store(p: &Path) { let f = File::create(p); }\n",
            "pub fn store(p: &Path) { let f = OpenOptions::new(); }\n",
        ]:
            e = errors(tree(extra={"src/b.rs": body}))
            self.assertTrue(any("src/b.rs::store: neither covered" in x for x in e), (body, e))
        # a fmt::Write into a String inside the body is no sink, and no writer name
        e = errors(tree(extra={"src/b.rs": "pub fn hex(&self) -> String { let mut s = String::new(); write!(s, \"x\"); s }\n"}))
        self.assertEqual(e, [])

    def test_an_array_return_type_does_not_end_the_signature(self):
        e = errors(tree(extra={"src/b.rs": "pub fn to_bytes(&self) -> [u8; 4] { [0; 4] }\n"}))
        self.assertTrue(any("src/b.rs::to_bytes: neither covered" in x for x in e), e)

    def test_a_delegating_fn_listed_as_not_a_writer_passes(self):
        b = "pub fn export_a(p: &str) { use crate::io::export_a as io_export_a; io_export_a(p) }\n"
        root = tree(extra={"src/py.rs": b}, not_writers="src/py.rs::export_a delegates to export_a\n")
        self.assertEqual(errors(root), [])
        self.assertEqual(pw.check(root)[1], 2)

    def test_a_delegating_fn_that_stops_calling_its_writer_fails(self):
        b = "pub fn export_a(p: &str) { other(p) }\n"
        e = errors(tree(extra={"src/py.rs": b}, not_writers="src/py.rs::export_a delegates to export_a\n"))
        self.assertTrue(any("does not call it" in x for x in e), e)

    def test_a_delegation_to_no_writer_fails(self):
        b = "pub fn export_x(p: &str) { import_a(p) }\n"
        e = errors(tree(extra={"src/py.rs": b}, not_writers="src/py.rs::export_x delegates to import_a\n"))
        self.assertTrue(any("which is no covered or pending writer" in x for x in e), e)

    def test_a_short_reason_fails_and_a_long_one_passes(self):
        x = {"src/b.rs": "pub fn serialize() {}\n"}
        e = errors(tree(extra=x, not_writers="src/b.rs::serialize in memory\n"))
        self.assertTrue(any("at least five words" in y for y in e), e)
        self.assertEqual(errors(tree(extra=x, not_writers="src/b.rs::serialize packs words for an in-process upload\n")), [])

    def test_a_not_writer_line_that_names_no_candidate_fails(self):
        e = errors(tree(not_writers="src/io/a.rs::import_a reads a file and writes nothing at all\n"))
        self.assertTrue(any("import_a is not a candidate" in x for x in e), e)

    def test_a_not_writer_also_pending_or_listed_twice_fails(self):
        line = "src/io/a.rs::save_b writes nothing that is stored anywhere\n"
        e = errors(tree(not_writers=line))
        self.assertTrue(any("save_b is also covered or pending" in x for x in e), e)
        e = errors(tree(not_writers=line * 2))
        self.assertTrue(any("listed twice: src/io/a.rs::save_b" in x for x in e), e)

    def test_a_stored_form_return_type_makes_any_name_a_candidate(self):
        for sig in ["pub fn pack70(&self) -> Vec<u8> { Vec::new() }",
                    "pub fn pack70() -> [u8; 32] { [0; 32] }",
                    "pub fn pack70(&self) -> &[u8] { &[] }",
                    "pub fn pack70() -> Box<[u8]> { Box::new([]) }",
                    "pub fn pack70() -> Result<Vec<u8>, E> { Ok(Vec::new()) }",
                    "pub fn to_json_string() -> Result<String, E> { Ok(String::new()) }",
                    "pub fn as_text(&self) -> String { String::new() }"]:
            name = re.search(r"fn (\w+)", sig).group(1)
            e = errors(tree(extra={"src/b.rs": sig + "\n"}))
            self.assertTrue(any(f"src/b.rs::{name}: neither covered" in x for x in e), (sig, e))
        # text from a fn that is not to_* / as_* is no stored form by itself
        self.assertEqual(errors(tree(extra={"src/b.rs": "pub fn describe() -> String { String::new() }\n"})), [])

    def test_a_variant_counts_its_base_whatever_the_names(self):
        src = ("pub fn render(t: &T) -> u32 { 0 }\n"
               "pub fn render_with_provenance(t: &T, p: &Provenance) -> u32 { 0 }\n")
        e = errors(tree(extra={"src/b.rs": src}))
        self.assertTrue(any("src/b.rs::render: neither covered" in x for x in e), e)
        self.assertTrue(any("render_with_provenance: not called" in x for x in e), e)

    def test_a_concrete_writer_type_or_file_options_is_a_sink(self):
        for body in ["pub fn emit70(w: &mut std::io::BufWriter<std::fs::File>) {}\n",
                     "pub fn emit70(f: File) {}\n",
                     "pub fn emit70(p: &Path) { let f = std::fs::File::options().write(true).open(p); }\n",
                     "pub fn emit70(p: &Path) { std::fs::copy(p, p); }\n"]:
            e = errors(tree(extra={"src/b.rs": body}))
            self.assertTrue(any("src/b.rs::emit70: neither covered" in x for x in e), (body, e))

    def test_an_entry_the_call_graph_shows_writing_is_a_candidate(self):
        # a trait-impl method has no `pub`, so the text never sees it; the index does
        src = "pub trait Ex { fn export70(&self); }\nimpl Ex for S { fn export70(&self) { w() } }\n"
        x = {"src/b.rs": src}
        self.assertEqual(errors(tree(extra=x)), [])
        e = errors(tree(extra=x), reached={"src/b.rs::export70": {"std fs/write()."}})
        self.assertTrue(any("src/b.rs::export70: neither covered" in y for y in e), e)
        line = "src/b.rs::export70 writes a scratch file that is removed before it returns\n"
        self.assertEqual(errors(tree(extra=x, not_writers=line), reached={"src/b.rs::export70": set()}), [])
        # without an index the line is checked to name a fn of the file; with one, to be a candidate
        self.assertEqual(errors(tree(extra=x, not_writers=line)), [])
        e = errors(tree(extra=x, not_writers=line), reached={})
        self.assertTrue(any("export70 is not a candidate" in y for y in e), e)
        e = errors(tree(not_writers=line))
        self.assertTrue(any("export70 is not a candidate" in y for y in e), e)

    def test_reach_hits_keeps_the_entries_that_reach_a_sink(self):
        graph = {"a": {"b"}, "b": {"std fs/write()."}, "c": {"d"}}

        def reach(start):
            live, stack = set(start), list(start)
            while stack:
                for t in graph.get(stack.pop(), ()):
                    if t not in live:
                        live.add(t)
                        stack.append(t)
            return live

        hits = pw.reach_hits({"src/x.rs::T::a": {"a"}, "src/x.rs::c": {"c"}}, reach, {"std fs/write()."})
        self.assertEqual(hits, {"src/x.rs::T::a": {"std fs/write()."}})
        self.assertEqual(pw.legacy("src/x.rs::T::a"), "src/x.rs::a")
        self.assertRegex("std fs/impl#[File]create().", pw.SINK_SYM)
        self.assertRegex("std fs/impl#[OpenOptions]open().", pw.SINK_SYM)
        self.assertRegex("std io/buffered/bufwriter/impl#[`BufWriter<W>`][Write]write_all().", pw.SINK_SYM)
        self.assertIsNone(pw.SINK_SYM.match("std fs/impl#[File]open()."))
        self.assertIsNone(pw.SINK_SYM.match("std fs/read()."))

    def test_any_fn_of_the_command_line_tool_with_a_sink_is_a_candidate(self):
        # the CLI's fns are private, yet each writes a file the user keeps
        body = "fn cmd_mesh(p: &Path) { std::fs::write(p, obj).unwrap(); }\nfn cmd_info() {}\n"
        e = errors(tree(extra={"src/bin/main.rs": body}))
        self.assertTrue(any("src/bin/main.rs::cmd_mesh: neither covered" in x for x in e), e)
        self.assertTrue(any("3 writers found" in x for x in e), e)
        self.assertFalse(any("cmd_info" in x for x in e), e)
        # a private fn elsewhere under src/ stays out of the text scan
        self.assertEqual(errors(tree(extra={"src/b.rs": body})), [])
        # a CLI fn that delegates to a library writer is classified like any other candidate
        line = "src/bin/main.rs::cmd_mesh delegates to export_a\n"
        ok = "fn cmd_mesh(p: &Path) { export_a(p).unwrap(); }\n"
        self.assertEqual(errors(tree(extra={"src/bin/main.rs": ok}, not_writers=line)), [])
        self.assertEqual(errors(tree(extra={"src/bin/main.rs": ok}, not_writers=line),
                                reached={"src/bin/main.rs::cmd_mesh": {"std fs/write()."}}), [])
        e = errors(tree(extra={"src/bin/main.rs": ok}), reached={"src/bin/main.rs::cmd_mesh": {"std fs/write()."}})
        self.assertTrue(any("src/bin/main.rs::cmd_mesh: neither covered" in x for x in e), e)

    def test_any_fn_of_an_example_with_a_sink_is_a_candidate_and_a_table_covers_it(self):
        body = "fn write_law(p: &Path) { std::fs::write(p, b).unwrap(); }\nfn main() { write_law(p) }\n"
        e = errors(tree(extra={"examples/bake/stored.rs": body}))
        self.assertTrue(any("examples/bake/stored.rs::write_law: neither covered" in x for x in e), e)
        self.assertFalse(any("::main" in x for x in e), e)
        # a helper's name or return type is no evidence in a program: only a sink is
        helpers = "fn encode_srgb(c: f32) -> u8 { 0 }\nfn render(&self) -> Vec<u8> { vec![] }\n"
        self.assertEqual(errors(tree(extra={"examples/icon.rs": helpers, "src/bin/main.rs": helpers})), [])
        table = 'const T: &[(&str, &str)] = &[("examples/bake/stored.rs::write_law", "x")];\n'
        ok = tree(extra={"examples/bake/stored.rs": body, "tests/test_bake.rs": table}, pending="expected-writers 3\nsrc/io/a.rs::save_b\n")
        self.assertEqual(errors(ok), [])

    def test_a_fn_of_the_command_line_tool_is_an_index_entry(self):
        sym = "rust-analyzer cargo alice-sdf 7.0.0 main/cmd_to_mesh()."
        self.assertEqual(pw.bin_fn_key("src/bin/main.rs", sym), "src/bin/main.rs::cmd_to_mesh")
        self.assertIsNone(pw.bin_fn_key("src/io/a.rs", sym))
        self.assertIsNone(pw.bin_fn_key("src/bin/main.rs", "rust-analyzer cargo alice-sdf 7.0.0 main/Cli#"))

    def test_every_writer_a_delegation_names_is_checked(self):
        src = SRC.replace("#[cfg(test)]", "pub fn save(p: &Path) { if j { save_b() } else { export_a(p) } }\n\n#[cfg(test)]")
        line = "src/io/a.rs::save delegates to export_a (and to save_b for a JSON path)\n"
        self.assertEqual(errors(tree(src=src, not_writers=line)), [])
        # the second branch writes through another fn: only the second name catches it
        bad = src.replace("save_b() }", "save_b_with_provenance() }")
        e = errors(tree(src=bad, not_writers=line))
        self.assertTrue(any("delegates to `save_b`, but its body does not call it" in x for x in e), e)
        self.assertFalse(any("`export_a`" in x for x in e), e)
        e = errors(tree(src=src, not_writers="src/io/a.rs::save delegates to export_a (and to nothing_here for x)\n"))
        self.assertTrue(any("`nothing_here`, which is no covered or pending writer" in x for x in e), e)

    def test_a_cli_fn_starts_from_what_its_own_body_references(self):
        pre = "rust-analyzer cargo alice-sdf 7.0.0 "
        occ = [
            {"symbol": pre + "main().", "roles": 1, "range": [1, 3, 7], "enc": [1, 0, 4, 1]},
            {"symbol": pre + "cmd_a().", "roles": 0, "range": [2, 4, 9]},
            {"symbol": pre + "cmd_a().", "roles": 1, "range": [6, 3, 8], "enc": [6, 0, 9, 1]},
            {"symbol": pre + "io/obj/export_obj().", "roles": 0, "range": [7, 4, 14]},
            {"symbol": "local 3", "roles": 0, "range": [8, 4, 5]},
            {"symbol": pre + "outside().", "roles": 0, "range": [12, 0, 7]},
        ]
        got = pw.bin_entries("src/bin/main.rs", occ)
        self.assertEqual(got, {
            # main calls cmd_a, whose references it takes along
            "src/bin/main.rs::main": {pre + "main().", pre + "cmd_a().", pre + "io/obj/export_obj()."},
            "src/bin/main.rs::cmd_a": {pre + "cmd_a().", pre + "io/obj/export_obj()."},
        })
        self.assertEqual(pw.bin_entries("src/bin/main.rs", occ[:0]), {})
        self.assertEqual(pw.bin_entries("src/io/a.rs", occ), {})

    def test_a_reference_in_a_nested_cli_fn_belongs_to_the_innermost_fn(self):
        pre = "rust-analyzer cargo alice-sdf 7.0.0 "
        occ = [
            {"symbol": pre + "cmd_outer().", "roles": 1, "range": [1, 3, 12], "enc": [1, 0, 10, 1]},
            # a fn nested in cmd_outer's body, which cmd_outer never calls
            {"symbol": pre + "cmd_outer/helper().", "roles": 1, "range": [3, 7, 13], "enc": [3, 4, 6, 5]},
            {"symbol": pre + "io/obj/export_obj().", "roles": 0, "range": [4, 8, 18]},
            {"symbol": pre + "info().", "roles": 0, "range": [8, 4, 8]},
        ]
        got = pw.bin_entries("src/bin/main.rs", occ)
        self.assertEqual(got["src/bin/main.rs::helper"], {pre + "cmd_outer/helper().", pre + "io/obj/export_obj()."})
        self.assertEqual(got["src/bin/main.rs::cmd_outer"], {pre + "cmd_outer().", pre + "info()."})

    def test_two_impls_of_one_name_share_a_key_and_both_must_delegate(self):
        src = ("impl A {\n    pub fn export_a(&self) { export_a(p) }\n}\n"
               "impl B {\n    pub fn export_a(&self) { other() }\n}\n")
        e = errors(tree(extra={"src/py.rs": src}, not_writers="src/py.rs::export_a delegates to export_a\n"))
        self.assertTrue(any("does not call it" in x for x in e), e)

    def test_a_table_in_any_test_file_covers(self):
        extra = {"tests/test_other.rs": 'const T: &[(&str, &str)] = &[("src/io/a.rs::save_b", "b")];\n'}
        e = errors(tree(extra=extra))
        self.assertTrue(any("save_b is covered by a table in tests/" in x for x in e), e)
        self.assertEqual(errors(tree(extra=extra, pending="expected-writers 2\n")), [])

    def test_this_repository_passes(self):
        self.assertEqual(errors(pw.ROOT), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
