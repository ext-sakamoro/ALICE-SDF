#!/usr/bin/env python3
"""Tests of scripts/shared_files.sh (python3 scripts/test_shared_files.py)

The script fetches ALICE-Law at the pinned revision. Without an argument it checks the
committed tree, which passes; given an export of HEAD with a flipped byte in a shared copy it
fails, and a pin with a wrong SHA-256 or a malformed pin fails before anything is compared."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# shared copies the flipped-byte test may use, in order (not every crate carries each)
SHARED_PROBES = ("scripts/run_tests.sh", "scripts/build_config_guard.py")


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


class SharedFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Path(self.tmp.name) / "tree"
        self.tree.mkdir()
        # no line-ending conversion: on Windows git archive would otherwise write CRLF
        archive = subprocess.run(["git", "-C", str(ROOT), "-c", "core.autocrlf=false", "-c", "core.eol=lf",
                                  "archive", "HEAD"], check=True, capture_output=True)
        subprocess.run(["tar", "-x", "-C", str(self.tree)], input=archive.stdout, check=True)
        for rel in ("scripts/shared_files.sh", ".github/shared-files-pin"):
            shutil.copy(ROOT / rel, self.tree / rel)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args: str, script_root: Path | None = None):
        root = script_root or self.tree
        r = subprocess.run([bash(), str(root / "scripts/shared_files.sh"), *args], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def test_the_committed_tree_passes_and_compares_something(self):
        code, out = self.run_script(script_root=ROOT)
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"copies [1-9][0-9]*, pending [0-9]+, 0 finding")

    def test_an_export_of_head_passes(self):
        code, out = self.run_script(str(self.tree))
        self.assertEqual(code, 0, out)

    def test_a_flipped_byte_in_a_shared_copy_fails(self):
        # a file of a uniform row every crate that runs this check carries (the first present)
        rel = next(r for r in SHARED_PROBES if (self.tree / r).is_file())
        f = self.tree / rel
        data = bytearray(f.read_bytes())
        data[-1] ^= 1
        f.write_bytes(bytes(data))
        code, out = self.run_script(str(self.tree))
        self.assertEqual(code, 1, out)
        self.assertIn(rel, out)
        # exactly the flipped file: a red for every file would also be exit 1
        self.assertIn(", 1 finding(s)", out)

    def test_a_wrong_checker_sha256_in_the_pin_fails(self):
        pin = self.tree / ".github/shared-files-pin"
        lines = pin.read_text(encoding="utf-8").splitlines()
        lines = [f"checker-sha256 {'0' * 64}" if l.startswith("checker-sha256 ") else l for l in lines]
        pin.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(self.run_script(str(self.tree))[0], 2)

    def test_a_malformed_pin_fails(self):
        (self.tree / ".github/shared-files-pin").write_text("rev main\n", encoding="utf-8")
        self.assertEqual(self.run_script(str(self.tree))[0], 2)


if __name__ == "__main__":
    unittest.main()
