#!/usr/bin/env python3
"""Tests for scripts/abi_decl_check.py: each runs the script against a small repository tree
(src/ffi exports and a tests/ declaration) and checks the exit status and the report.

usage: python3 scripts/test_abi_decl_check.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "abi_decl_check.py"

EXPORTS = """
#[no_mangle]
pub extern "C" fn alice_sdf_repeat_finite(node: SdfHandle, cx: u32, cy: u32, cz: u32, sx: f32, sy: f32, sz: f32) -> SdfHandle {
    todo!()
}

#[no_mangle]
pub extern "C" fn alice_sdf_load_abm(path: *const c_char) -> MeshHandle {
    todo!()
}

#[no_mangle]
pub extern "C" fn alice_sdf_free(node: SdfHandle) {
}
"""

GOOD_DECLS = """
extern "C" {
    fn alice_sdf_repeat_finite(node: SdfHandle, cx: u32, cy: u32, cz: u32, sx: f32, sy: f32, sz: f32) -> SdfHandle;
    fn alice_sdf_load_abm(path: *const c_char) -> MeshHandle;
    fn alice_sdf_free(node: SdfHandle);
}
"""


def run(decls: str, exports: str = EXPORTS) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "scripts").mkdir()
        shutil.copy(SCRIPT, root / "scripts" / SCRIPT.name)
        (root / "src" / "ffi").mkdir(parents=True)
        (root / "src" / "ffi" / "mod.rs").write_text(exports, encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests" / "decl.rs").write_text(decls, encoding="utf-8")
        return subprocess.run([sys.executable, str(root / "scripts" / SCRIPT.name)],
                              capture_output=True, text=True)


class AbiDeclCheck(unittest.TestCase):
    def test_matching_declarations_pass(self):
        r = run(GOOD_DECLS)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("3 compared, 0 mismatched", r.stdout)

    def test_swapped_argument_groups_fail(self):
        # the Windows-only failure of 2026-09-30: floats before the integers
        r = run(GOOD_DECLS.replace("cx: u32, cy: u32, cz: u32, sx: f32, sy: f32, sz: f32) -> SdfHandle;",
                                   "sx: f32, sy: f32, sz: f32, cx: u32, cy: u32, cz: u32) -> SdfHandle;"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("ABI-DECL-MISMATCH", r.stdout)
        self.assertIn("alice_sdf_repeat_finite", r.stdout)

    def test_another_return_type_fails(self):
        r = run(GOOD_DECLS.replace("fn alice_sdf_load_abm(path: *const c_char) -> MeshHandle;",
                                   "fn alice_sdf_load_abm(path: *const c_char) -> *mut std::ffi::c_void;"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("-> *mut std::ffi::c_void", r.stdout)
        self.assertIn("-> MeshHandle", r.stdout)

    def test_a_dropped_return_type_fails(self):
        r = run(GOOD_DECLS.replace("fn alice_sdf_load_abm(path: *const c_char) -> MeshHandle;",
                                   "fn alice_sdf_load_abm(path: *const c_char);"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("-> ()", r.stdout)

    def test_a_return_type_on_a_unit_export_fails(self):
        r = run(GOOD_DECLS.replace("fn alice_sdf_free(node: SdfHandle);",
                                   "fn alice_sdf_free(node: SdfHandle) -> u32;"))
        self.assertEqual(r.returncode, 1)

    def test_whitespace_in_a_return_type_does_not_matter(self):
        r = run(GOOD_DECLS.replace("-> MeshHandle;", "->   MeshHandle ;"))
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_an_undeclared_export_name_fails(self):
        r = run(GOOD_DECLS.replace("fn alice_sdf_free(", "fn alice_sdf_free_all("))
        self.assertEqual(r.returncode, 1)
        self.assertIn("exports no such symbol", r.stdout)

    def test_nothing_compared_fails(self):
        r = run("// no extern block here\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn("nothing was compared", r.stdout)

    def test_no_exports_fails(self):
        r = run(GOOD_DECLS, exports="// no exports\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no exports found", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
