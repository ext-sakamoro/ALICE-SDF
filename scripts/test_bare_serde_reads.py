#!/usr/bin/env python3
"""Tests for scripts/bare_serde_reads.py: a stored-form site is found on a statement and on a
function, with the lines it covers (string literals and comments with braces do not end it
early); a site with an unknown status, a short reason or `expect` fails; a tree with no source
file or no site fails; and the repository passes. That clippy flags a bare read in each
spelling is the canary's (scripts/clippy_lib_gate.py)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "bare_serde_reads.py"
sys.path.insert(0, str(HERE))
import bare_serde_reads as b  # noqa: E402


def attr(status_and_why: str, kind: str = "allow") -> str:
    return f'#[{kind}(\n    clippy::disallowed_methods,\n    reason = "stored form {status_and_why}"\n)]\n'


STATEMENT = ("pub fn f(s: &str) -> Option<Tree> {\n"
             "    " + attr("checked: the identifier goes through sdf_load_verdict").replace("\n", "\n    ").rstrip()
             + "\n    let v: Value = serde_json::from_str(s)\n        .map_err(|e| format!(\"{e} }}\"))\n        .ok()?;\n"
             "    let _later = serde_json::from_str::<u8>(s);\n    None\n}\n")
FUNCTION = (attr("pending: a reader that still has to check its identifier")
            + "fn g(s: &str) -> Value {\n    // a } in a comment\n    serde_json::from_str(s).unwrap()\n}\nfn h() {}\n")


class Sites(unittest.TestCase):
    def test_a_statement_site_covers_the_statement_only(self):
        (s,) = b.sites(STATEMENT)
        self.assertEqual((s["kind"], s["status"]), ("allow", "checked"))
        self.assertEqual((s["first"], s["last"]), (2, 8))
        self.assertNotIn("_later", STATEMENT.split("\n")[s["last"] - 1])

    def test_a_function_site_covers_the_function(self):
        (s,) = b.sites(FUNCTION)
        self.assertEqual((s["status"], s["first"], s["last"]), ("pending", 1, 8))

    def test_another_lint_or_reason_is_not_a_site(self):
        self.assertEqual(b.sites('#[allow(clippy::disallowed_methods, reason = "libm for a preview")]\nfn f() {}\n'), [])
        self.assertEqual(b.sites('#[allow(clippy::disallowed_types, reason = "stored form checked: x y z w v")]\n'), [])


class Run(unittest.TestCase):
    def run_check(self, files, make_src=True):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            if make_src:
                (root / "src").mkdir()
            for name, text in files.items():
                (root / "src" / name).write_text(text, encoding="utf-8")
            p = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root)], capture_output=True, text=True)
        return p.returncode, p.stdout

    def test_reviewed_sites_pass_and_are_counted(self):
        rc, out = self.run_check({"a.rs": STATEMENT, "b.rs": FUNCTION})
        self.assertEqual(rc, 0, out)
        self.assertIn("2 stored-form sites (1 checked, 0 other-type, 1 pending, 0 problems)", out)

    def test_a_bad_site_fails(self):
        for text in (attr("allowed: an unknown status for this read") + "fn f() {}\n",
                     attr("checked: short") + "fn f() {}\n",
                     attr("checked: an expect is unfulfilled under the root list", "expect") + "fn f() {}\n"):
            with self.subTest(text=text):
                rc, out = self.run_check({"a.rs": text})
                self.assertEqual(rc, 1, out)
                self.assertIn("error:", out)

    def test_no_site_fails(self):
        rc, out = self.run_check({"a.rs": "pub fn f() {}\n"})
        self.assertEqual(rc, 1, out)
        self.assertIn("compared nothing", out)

    def test_no_source_file_fails(self):
        rc, out = self.run_check({}, make_src=False)
        self.assertEqual(rc, 1, out)
        self.assertIn("0 scanned", out)

    def test_the_repository_passes(self):
        p = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout)


if __name__ == "__main__":
    unittest.main()
