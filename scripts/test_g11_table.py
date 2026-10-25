#!/usr/bin/env python3
"""Tests of scripts/g11_table.sh (python3 scripts/test_g11_table.py)

The script fetches ALICE-Law's scripts/g11_table_check.py at the revision .github/g11-table-pin
names. In an export of HEAD it passes; with one row of conformance/g11.toml removed it fails (a pass
gate without its row); a pin with a wrong SHA-256 or a malformed pin fails before anything is
compared."""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bash() -> str:
    """Git's bash on Windows (`bash` on PATH there can be the WSL launcher, which has no
    distribution on the runners), the PATH one elsewhere."""
    if os.name == "nt":
        exec_path = subprocess.run(["git", "--exec-path"], check=True, capture_output=True, text=True).stdout.strip()
        for up in (3, 2):
            candidate = Path(exec_path).parents[up - 1] / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError(f"no bash.exe of Git near {exec_path}")
    found = shutil.which("bash")
    if found is None:
        raise RuntimeError("no bash on PATH")
    return found


class G11Table(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Path(self.tmp.name) / "tree"
        self.tree.mkdir()
        # no line-ending conversion: on Windows git archive would otherwise write CRLF
        archive = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "HEAD"], cwd=ROOT,
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(self.tree)], input=archive, check=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self):
        return subprocess.run([bash(), "scripts/g11_table.sh"], cwd=self.tree, capture_output=True, text=True)

    def test_the_committed_table_passes(self):
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertRegex(r.stdout, r"[1-9][0-9]* rows, [1-9][0-9]* pass gates, 0 findings")

    def test_a_pass_gate_without_its_row_fails(self):
        table = self.tree / "conformance" / "g11.toml"
        text = table.read_text(encoding="utf-8")
        rows = re.split(r"(?m)^(?=\[\[row\]\]$)", text)
        own = [i for i, r in enumerate(rows) if re.search(r'(?m)^gate = "G6"$', r)]
        self.assertTrue(own, "no G6 row to remove")
        del rows[own[0]]
        table.write_text("".join(rows), encoding="utf-8", newline="\n")
        r = self.run_script()
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("G6", r.stdout + r.stderr)

    def test_a_wrong_checker_sha256_fails(self):
        pin = self.tree / ".github" / "g11-table-pin"
        text = pin.read_text(encoding="utf-8")
        wrong = re.sub(r"(checker-sha256 )([0-9a-f])", lambda m: m.group(1) + ("1" if m.group(2) == "0" else "0"), text)
        self.assertNotEqual(wrong, text)
        pin.write_text(wrong, encoding="utf-8", newline="\n")
        r = self.run_script()
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("has sha256", r.stderr)

    def test_a_malformed_pin_fails(self):
        (self.tree / ".github" / "g11-table-pin").write_text("rev nothing\n", encoding="utf-8", newline="\n")
        r = self.run_script()
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("needs", r.stderr)


if __name__ == "__main__":
    unittest.main()
