#!/usr/bin/env python3
"""Tests of scripts/law_ratchets.sh (python3 scripts/test_law_ratchets.py)

The script fetches ALICE-Law at the revision .github/law-ratchets-pin names and runs its no-panic
ratchet, structural lint and added-function pins on one tree. An export of HEAD passes with every
count read; one production `unwrap()` more is a finding (exit 1); a pin with a wrong SHA-256 or a
malformed pin fails before anything is read (exit 2); a tree with no production Rust code compared
nothing and fails (exit 2); a tree that does not depend on alice-det-math skips the added-function
check and says so."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bash() -> str:
    """Git's bash on Windows (`bash` on PATH there can be the WSL launcher), the PATH one elsewhere."""
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


class LawRatchets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Path(self.tmp.name) / "tree"
        self.tree.mkdir()
        archive = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "HEAD"], cwd=ROOT,
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(self.tree)], input=archive, check=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, tree=None):
        # the script reads its pin from the copy; the tree under test is passed explicitly (the copy has no git)
        return subprocess.run([bash(), "scripts/law_ratchets.sh", str(tree or self.tree)], cwd=self.tree,
                              capture_output=True, text=True)

    def test_the_committed_tree_passes_and_reads_every_count(self):
        r = self.run_script()
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertRegex(out, r"no_panic_ratchet: 1 repositories read, [1-9][0-9]* production files, [0-9]+ sites, 0 finding")
        self.assertRegex(out, r"structural_lint: 1 repositories read, [1-9][0-9]* production files, .* 0 finding")
        self.assertRegex(out, r"added_function_pins: .*1 repositories read, 1 use alice_det_math, 0 finding")

    def test_one_more_production_unwrap_is_a_finding(self):
        lib = self.tree / "src" / "lib.rs"
        with lib.open("a", encoding="utf-8", newline="\n") as f:
            f.write('\npub fn law_ratchets_test_site() -> u32 { "7".parse::<u32>().unwrap() }\n')
        r = self.run_script()
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertRegex(r.stdout + r.stderr, r"unwrap [0-9]+ > baseline [0-9]+ \(a new site\)")

    def test_a_wrong_checker_sha256_fails_before_reading(self):
        pin = self.tree / ".github" / "law-ratchets-pin"
        text = pin.read_text(encoding="utf-8")
        line = next(l for l in text.splitlines() if l.startswith("no-panic-checker-sha256 "))
        digest = line.split()[1]
        flipped = ("0" if digest[0] != "0" else "1") + digest[1:]
        pin.write_text(text.replace(line, f"no-panic-checker-sha256 {flipped}"), encoding="utf-8", newline="\n")
        r = self.run_script()
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("scripts/no_panic_ratchet.py at", r.stderr)
        self.assertNotIn("repositories read", r.stdout)

    def test_a_malformed_pin_fails(self):
        (self.tree / ".github" / "law-ratchets-pin").write_text("rev nothing\n", encoding="utf-8", newline="\n")
        r = self.run_script()
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("law-ratchets-pin: needs", r.stderr)

    def test_a_tree_with_no_production_code_compared_nothing(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        (empty / "Cargo.toml").write_text('[package]\nname = "x"\n', encoding="utf-8", newline="\n")
        r = self.run_script(empty)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("does not depend on alice-det-math", r.stdout)


if __name__ == "__main__":
    unittest.main()
