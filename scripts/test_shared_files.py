#!/usr/bin/env python3
"""Tests of scripts/shared_files.sh (python3 scripts/test_shared_files.py)

The script fetches ALICE-Law at the pinned revision. Without an argument it checks the
committed tree, which passes; given an export of HEAD with a flipped byte in a shared copy it
fails, and a pin with a wrong SHA-256 or a malformed pin fails before anything is compared.

The same file runs in every crate that carries the script, public (listed in ALICE-Law's table)
or private (`visibility private` in its pin, never listed): which shared copies a tree holds and
which repository names the table lists are read from the table at the pinned revision, not
written here."""
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / ".github/shared-files-pin"
# the copies these tests rewrite themselves: never the probe of a flipped byte
OWN = ("scripts/shared_files.sh", "scripts/test_shared_files.py")


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


def pin_value(key: str) -> str:
    values = [l.split()[1] for l in PIN.read_text(encoding="utf-8").splitlines()
              if l.startswith(key + " ") and len(l.split()) == 2]
    if len(values) != 1:
        raise RuntimeError(f"{PIN}: no single '{key} <value>' line")
    return values[0]


def pinned_table() -> list[tuple[str, list[str]]]:
    """(path, listed repositories) of each row of ALICE-Law's table at the pinned revision, refused
    unless its SHA-256 is the pin's (the script fetches the same bytes)"""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        subprocess.run(["git", "-C", tmp, "fetch", "-q", "--depth", "1", "https://github.com/ext-sakamoro/ALICE-Law",
                        pin_value("rev")], check=True)
        data = subprocess.run(["git", "-C", tmp, "show", "FETCH_HEAD:conformance/shared-files.toml"], check=True,
                              capture_output=True).stdout
    if hashlib.sha256(data).hexdigest() != pin_value("table-sha256"):
        raise RuntimeError("the table at the pinned revision is not the pin's table-sha256")
    rows = []
    for block in data.decode("utf-8").split("[[file]]")[1:]:
        path = re.search(r'^path = "([^"]+)"$', block, re.M)
        repos = re.search(r"^repos = \[([^\]]*)\]$", block, re.M)
        if path is None or repos is None:
            raise RuntimeError(f"a table row without a path or a repos line: {block[:80]!r}")
        rows.append((path.group(1), re.findall(r'"([^"]+)"', repos.group(1))))
    if not rows:
        raise RuntimeError("the pinned table has no rows")
    return rows


class SharedFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pinned_table()

    def present(self, tree: Path) -> list[str]:
        """the table paths this tree holds"""
        return [p for p, _ in self.rows if (tree / p).is_file()]

    def probes(self, tree: Path) -> list[str]:
        """the table paths this tree holds whose row records bytes (some repository holds it: a row no
        repository holds is per-repository data, not compared), in table order"""
        compared = [p for p, repos in self.rows if repos and p not in OWN]
        found = [p for p in compared if (tree / p).is_file()]
        self.assertTrue(found, f"{tree} holds no shared copy whose row records bytes")
        return found

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
        # the first shared copy this tree holds whose row records bytes
        rel = self.probes(self.tree)[0]
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

    def write_pin(self, name: str, visibility: str | None) -> None:
        """The committed pin with another repository name and, if given, a visibility line."""
        pin = self.tree / ".github/shared-files-pin"
        lines = [l for l in pin.read_text(encoding="utf-8").splitlines()
                 if not l.startswith(("name ", "visibility "))]
        lines.insert(0, f"name {name}")
        if visibility is not None:
            lines.append(f"visibility {visibility}")
        pin.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def unlisted_name(self) -> str:
        """a repository name the table lists nowhere (as a private holder's)"""
        listed = {r for _, repos in self.rows for r in repos}
        name = "ALICE-PrivateProbe"
        while name in listed:
            name += "X"
        return name

    def private_tree(self) -> Path:
        """A private holder's tree: every table path the committed tree holds (shared copies and
        per-repository data alike, as a private repository carries them), at the committed bytes,
        and nothing else."""
        tree = Path(self.tmp.name) / "private"
        for rel in self.present(self.tree):
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.tree / rel, tree / rel)
        return tree

    def test_a_private_repository_compares_its_copies_with_the_rows(self):
        # a name the table never lists: the private mode compares each table path present
        self.write_pin(self.unlisted_name(), "private")
        code, out = self.run_script(str(self.private_tree()))
        self.assertEqual(code, 0, out)
        self.assertIn("private copy, the row's bytes", out)
        self.assertRegex(out, r"copies [1-9][0-9]*, pending [0-9]+, 0 finding")

    def test_a_flipped_byte_fails_in_the_private_mode_too(self):
        self.write_pin(self.unlisted_name(), "private")
        tree = self.private_tree()
        rel = self.probes(tree)[0]
        f = tree / rel
        data = bytearray(f.read_bytes())
        data[-1] ^= 1
        f.write_bytes(bytes(data))
        code, out = self.run_script(str(tree))
        self.assertEqual(code, 1, out)
        self.assertIn(rel, out)
        self.assertIn(", 1 finding(s)", out)

    def test_every_shared_copy_of_a_private_holder_is_compared(self):
        # a flipped byte in each probe of the private tree is a finding on its own (not only the first)
        self.write_pin(self.unlisted_name(), "private")
        for rel in self.probes(self.private_tree()):
            with self.subTest(rel=rel):
                tree = self.private_tree()
                f = tree / rel
                data = bytearray(f.read_bytes())
                data[-1] ^= 1
                f.write_bytes(bytes(data))
                code, out = self.run_script(str(tree))
                self.assertEqual(code, 1, out)
                self.assertIn(f"{rel}: sha256 ", out)
                self.assertIn(", 1 finding(s)", out)
                shutil.rmtree(tree)

    def test_a_listed_public_repository_cannot_use_the_private_mode(self):
        # a repository the table lists (the first named by a row), so the private mode would dodge
        # its rows: refused
        name = next(r for _, repos in self.rows for r in repos)
        self.write_pin(name, "private")
        code, out = self.run_script(str(self.tree))
        self.assertEqual(code, 2, out)

    def test_a_pin_without_a_visibility_line_is_public(self):
        # no line is the public mode: a name the table never lists then holds shared paths the table
        # neither lists nor exempts for it, which fails (the private mode is never the default)
        self.write_pin(self.unlisted_name(), None)
        code, out = self.run_script(str(self.private_tree()))
        self.assertEqual(code, 1, out)
        self.assertNotIn("private copy", out)

    def test_an_unknown_visibility_fails(self):
        self.write_pin(self.unlisted_name(), "secret")
        self.assertEqual(self.run_script(str(self.tree))[0], 2)

    def test_a_visibility_line_without_a_value_fails(self):
        # a bare `visibility` line must not read as public
        self.write_pin(self.unlisted_name(), "")
        self.assertEqual(self.run_script(str(self.private_tree()))[0], 2)

    def test_the_private_mode_comparing_nothing_fails(self):
        self.write_pin(self.unlisted_name(), "private")
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        (empty / "README.md").write_text("nothing shared\n", encoding="utf-8")
        self.assertEqual(self.run_script(str(empty))[0], 2)


if __name__ == "__main__":
    unittest.main()
