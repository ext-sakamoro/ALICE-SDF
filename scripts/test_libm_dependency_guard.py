#!/usr/bin/env python3
"""Tests for scripts/libm_dependency_guard.py: a tree without libm passes,
libm declared in every table form without the ban fails at its line, the ban
plus scripts/libm_ban_check.py passes, either half alone fails, look-alikes do
not trigger, and an empty tree is exit 2."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import libm_dependency_guard as g  # noqa: E402

PKG = '[package]\nname = "x"\nversion = "0.1.0"\nedition = "2021"\n\n'
BAN = ('disallowed-methods = [\n'
       '    { path = "f32::sin", reason = "r" },\n'
       '    { path = "libm::sin", reason = "r" },\n'
       ']\n')
NO_BAN = 'disallowed-methods = [\n    { path = "f32::sin", reason = "r" },\n]\n'
BAN_SCRIPT = {"scripts/libm_ban_check.py": "# ban check\n"}


def tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="libm-guard-"))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def run(files: dict[str, str]) -> tuple[int, str, str]:
    root = tree(files)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = g.main(["--root", str(root)])
    return rc, out.getvalue(), err.getvalue()


# (manifest body after [package], line of the declaration in the whole file)
FORMS = {
    "plain": ('[dependencies]\nlibm = "0.2"\n', 7),
    "inline": ('[dependencies]\nlibm = { version = "0.2", default-features = false }\n', 7),
    "dotted": ('[dependencies]\nlibm.version = "0.2"\n', 7),
    "table": ('[dependencies.libm]\nversion = "0.2"\n', 6),
    "renamed-inline": ('[dependencies]\nm = { package = "libm", version = "0.2" }\n', 7),
    "renamed-table": ('[dependencies.m]\npackage = "libm"\nversion = "0.2"\n', 7),
    "dev": ('[dev-dependencies]\nlibm = "0.2"\n', 7),
    "dev-legacy": ('[dev_dependencies]\nlibm = "0.2"\n', 7),
    "build": ('[build-dependencies]\nlibm = "0.2"\n', 7),
    "build-legacy": ('[build_dependencies]\nlibm = "0.2"\n', 7),
    "target": ("[target.'cfg(target_os = \"none\")'.dependencies]\nlibm = \"0.2\"\n", 7),
    "target-renamed": ('[target.wasm32-unknown-unknown.dev-dependencies]\n'
                       'm = { package = "libm", version = "0.2" }\n', 7),
    "workspace": ('[workspace.dependencies]\nlibm = "0.2"\n', 7),
    "patch": ('[patch.crates-io]\nlibm = { path = "../libm" }\n', 7),
    "replace": ('[replace]\n"libm:0.2.8" = { path = "../libm" }\n', 7),
}


class Guard(unittest.TestCase):
    def test_clean_tree_passes(self):
        rc, out, err = run({"Cargo.toml": PKG + '[dependencies]\nserde = "1"\n',
                            "clippy.toml": NO_BAN})
        self.assertEqual(rc, 0, err)
        self.assertIn("1 manifests read, 0 declare libm, 0 problems", out)

    def test_every_form_without_ban_fails_at_its_line(self):
        for name, (body, line) in FORMS.items():
            with self.subTest(form=name):
                rc, out, err = run({"Cargo.toml": PKG + body, "clippy.toml": NO_BAN})
                self.assertEqual(rc, 1, f"{name}: {out}{err}")
                self.assertIn(f"Cargo.toml:{line}: libm is a direct dependency but "
                              "clippy.toml bans no libm:: path", err)
                self.assertIn("1 declare libm", out)

    def test_every_form_without_any_clippy_toml_fails(self):
        for name, (body, line) in FORMS.items():
            with self.subTest(form=name):
                rc, _, err = run({"Cargo.toml": PKG + body, **BAN_SCRIPT})
                self.assertEqual(rc, 1, name)
                self.assertIn(f"Cargo.toml:{line}: libm is a direct dependency", err)

    def test_every_form_with_full_ban_passes(self):
        for name, (body, _) in FORMS.items():
            with self.subTest(form=name):
                rc, out, err = run({"Cargo.toml": PKG + body, "clippy.toml": BAN,
                                    **BAN_SCRIPT})
                self.assertEqual(rc, 0, f"{name}: {err}")
                self.assertIn("1 declare libm, 0 problems", out)

    def test_string_form_of_ban_entry_counts(self):
        rc, _, err = run({"Cargo.toml": PKG + FORMS["plain"][0],
                          "clippy.toml": 'disallowed-methods = ["libm::exp"]\n', **BAN_SCRIPT})
        self.assertEqual(rc, 0, err)

    def test_ban_without_ban_check_script_fails(self):
        rc, _, err = run({"Cargo.toml": PKG + FORMS["plain"][0], "clippy.toml": BAN})
        self.assertEqual(rc, 1)
        self.assertIn("Cargo.toml:7: libm is a direct dependency but "
                      "scripts/libm_ban_check.py is missing", err)
        self.assertNotIn("bans no libm::", err)

    def test_ban_check_script_without_ban_fails(self):
        rc, _, err = run({"Cargo.toml": PKG + FORMS["plain"][0], "clippy.toml": NO_BAN,
                          **BAN_SCRIPT})
        self.assertEqual(rc, 1)
        self.assertIn("bans no libm::", err)
        self.assertNotIn("is missing", err)

    def test_libm_path_under_other_key_is_not_a_ban(self):
        clippy = 'disallowed-types = [{ path = "libm::Libm", reason = "r" }]\n'
        rc, _, err = run({"Cargo.toml": PKG + FORMS["plain"][0], "clippy.toml": clippy,
                          **BAN_SCRIPT})
        self.assertEqual(rc, 1)
        self.assertIn("bans no libm::", err)

    def test_nested_clippy_toml_without_ban_fails(self):
        rc, _, err = run({"Cargo.toml": PKG + FORMS["plain"][0], "clippy.toml": BAN,
                          "sub/clippy.toml": NO_BAN, **BAN_SCRIPT})
        self.assertEqual(rc, 1)
        self.assertIn("sub/clippy.toml:0: no libm:: path under disallowed-methods", err)

    def test_clippy_toml_under_dot_directory_without_ban_fails(self):
        rc, _, err = run({"Cargo.toml": PKG + FORMS["plain"][0], "clippy.toml": BAN,
                          ".github/clippy-lib/clippy.toml": NO_BAN, **BAN_SCRIPT})
        self.assertEqual(rc, 1)
        self.assertIn(".github/clippy-lib/clippy.toml:0: no libm:: path", err)

    def test_nested_ban_without_root_clippy_toml_fails(self):
        rc, _, err = run({"Cargo.toml": PKG + FORMS["plain"][0],
                          "sub/clippy.toml": BAN, **BAN_SCRIPT})
        self.assertEqual(rc, 1)
        self.assertIn("bans no libm::", err)

    def test_lookalikes_do_not_trigger(self):
        body = ('# libm = "0.2"\n'
                '[dependencies]\n'
                'libm-foo = "1"\n'
                'foo-libm = "1"\n'
                'm = { package = "libm-foo", version = "1" }\n'
                'n = { version = "1", features = ["libm"] }  # libm\n'
                'o = { version = "1" }  # was: package = "libm"\n'
                'note = { path = "x", description = "uses libm" }\n'
                '[features]\n'
                'libm = ["n/libm"]\n'
                '[package.metadata.x]\n'
                'libm = "0.2"\n')
        rc, out, err = run({"Cargo.toml": PKG + body})
        self.assertEqual(rc, 0, err)
        self.assertIn("0 declare libm, 0 problems", out)

    def test_multiline_string_mentioning_libm_does_not_trigger(self):
        body = 'description = """\n[dependencies]\nlibm = "0.2"\n"""\n'
        rc, out, err = run({"Cargo.toml": PKG.replace("[package]\n", "[package]\n" + body)})
        self.assertEqual(rc, 0, err)
        self.assertIn("0 declare libm", out)

    def test_zero_manifests_is_exit_2(self):
        rc, out, _ = run({"README.md": "libm\n"})
        self.assertEqual(rc, 2)
        self.assertIn("0 manifests read", out)

    def test_nested_member_manifest_is_read(self):
        rc, out, err = run({"Cargo.toml": PKG + '[workspace]\nmembers = ["crates/a"]\n',
                            "crates/a/Cargo.toml": PKG + FORMS["plain"][0]})
        self.assertEqual(rc, 1)
        self.assertIn("2 manifests read, 1 declare libm", out)
        self.assertIn("crates/a/Cargo.toml:7: libm is a direct dependency", err)

    def test_skipped_directories_are_not_read(self):
        rc, out, _ = run({"Cargo.toml": PKG,
                          "target/x/Cargo.toml": PKG + FORMS["plain"][0],
                          "node_modules/y/Cargo.toml": PKG + FORMS["plain"][0],
                          ".hidden/Cargo.toml": PKG + FORMS["plain"][0]})
        self.assertEqual(rc, 0)
        self.assertIn("1 manifests read, 0 declare libm", out)

    def test_parser_and_scan_disagreement_is_reported(self):
        # an inline table spread over lines the scan cannot follow (TOML 1.1
        # parsers accept it, TOML 1.0 parsers reject it): either way it is
        # reported, never passed, even with the full ban in place
        body = '[dependencies]\nm = { version = "0.2",\n  package = "libm" }\n'
        rc, _, err = run({"Cargo.toml": PKG + body, "clippy.toml": BAN, **BAN_SCRIPT})
        self.assertEqual(rc, 1, err)
        self.assertTrue("unlocated" in err or "unparsable" in err, err)

    def test_unlocated_declaration_still_requires_the_ban(self):
        orig = g._located
        g._located = lambda text: {}
        try:
            rc, out, err = run({"Cargo.toml": PKG + FORMS["plain"][0]})
        finally:
            g._located = orig
        self.assertEqual(rc, 1)
        self.assertIn("1 declare libm", out)
        self.assertIn("Cargo.toml:0: libm dependency unlocated", err)
        self.assertIn("Cargo.toml:0: libm is a direct dependency but clippy.toml bans", err)

    def test_scan_only_hit_is_reported(self):
        orig = g._semantic
        g._semantic = lambda doc: set()
        try:
            rc, _, err = run({"Cargo.toml": PKG + FORMS["plain"][0]})
        finally:
            g._semantic = orig
        self.assertEqual(rc, 1)
        self.assertIn("Cargo.toml:7: unlocated (line scan finds libm, parser does not)", err)

    def test_unparsable_manifest_is_a_problem(self):
        rc, _, err = run({"Cargo.toml": "[dependencies\n"})
        self.assertEqual(rc, 1)
        self.assertIn("Cargo.toml:0: unparsable manifest", err)

    def test_this_repository(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = g.main([])
        self.assertEqual(rc, 0, err.getvalue())
        self.assertRegex(out.getvalue(), r"libm_dependency_guard: [1-9]\d* manifests read")


if __name__ == "__main__":
    unittest.main()
