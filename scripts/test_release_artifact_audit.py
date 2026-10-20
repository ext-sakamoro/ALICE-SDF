#!/usr/bin/env python3
"""Tests of scripts/release_artifact_audit.py on artifacts built here: a small Rust program
compiled with a home directory in its panic paths is a finding, the same program built with
the path remapped and the symbols stripped is not; a private name (held only as a hash), an
internal word, a token-like string, a member of a zip and a UTF-16 path are findings; nothing to
compare exits 2.

The checker is ALICE-Law's at the pin (.github/public-text-pin): fetched, or the copy named by
ALICE_PUBLIC_TEXT_CHECKER (its SHA-256 is checked either way). The Rust program needs `rustc`.

usage: python3 scripts/test_release_artifact_audit.py
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import release_artifact_audit as rea  # noqa: E402

# a home directory that is not a placeholder (the checker does not count runner, user, ...)
PLANTED_HOME = "/Users/plantedname"
PROGRAM = 'fn main() {\n    let v: Vec<u32> = std::env::args().map(|a| a.len() as u32).collect();\n' \
          '    if v.len() > 7 { panic!("too many arguments: {}", v[7]); }\n    println!("{}", v.len());\n}\n'
# an internal word, assembled so this file does not carry it as a word
INTERNAL_WORD = "sub" + "agent"
# a token's shape, assembled for the same reason (and so no scanner sees a credential here)
TOKEN = "gh" + "p_" + "Q7" * 18


class Checker:
    module = None
    path: Path | None = None
    tmp: Path | None = None

    @classmethod
    def get(cls):
        if cls.module is None:
            _, sha = rea.read_pin()
            given = os.environ.get("ALICE_PUBLIC_TEXT_CHECKER")
            if given:
                cls.path = Path(given)
            else:
                cls.tmp = Path(tempfile.mkdtemp(prefix="rea-checker-"))
                cls.path = rea.fetch_checker(rea.read_pin()[0], cls.tmp)
            cls.module = rea.load_checker(cls.path, sha)
        return cls.module


def run(*args: str) -> subprocess.CompletedProcess:
    Checker.get()
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "release_artifact_audit.py"),
                           "--checker", str(Checker.path), *args], capture_output=True, text=True, timeout=300)


class Built(unittest.TestCase):
    """the same program, built twice"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="rea-build-"))
        src = cls.tmp / "src"
        src.mkdir()
        (src / "main.rs").write_text(PROGRAM)

        def build(name: str, *flags: str) -> Path:
            out = cls.tmp / name
            subprocess.run(["rustc", "--edition", "2021", "-C", "opt-level=2", *flags, "-o", str(out),
                            str(src / "main.rs")], check=True, capture_output=True, cwd=cls.tmp, timeout=300)
            return out

        cls.planted = build("planted", f"--remap-path-prefix={src}={PLANTED_HOME}/project/src")
        cls.clean = build("clean", f"--remap-path-prefix={src}=src", "-C", "strip=symbols")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, True)

    def test_the_planted_home_path_is_in_the_bytes(self) -> None:
        # the fixture itself: otherwise a green below proves nothing
        self.assertIn(PLANTED_HOME.encode(), self.planted.read_bytes())
        self.assertNotIn(PLANTED_HOME.encode(), self.clean.read_bytes())

    def test_a_binary_with_a_home_path_is_a_finding(self) -> None:
        p = run(str(self.planted))
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("binary home path", p.stdout)
        self.assertNotIn("plantedname", p.stdout + p.stderr, "a finding names the kind, never the text")

    def test_a_remapped_and_stripped_binary_passes(self) -> None:
        p = run(str(self.clean))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertRegex(p.stdout, r"1 artifacts, 1 files \(0 same as a tracked file\), [1-9][0-9]* strings compared, 0 findings")

    def test_a_member_of_a_zip_is_read(self) -> None:
        z = self.tmp / "plugin.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as w:
            w.write(self.clean, "lib/clean")
            w.write(self.planted, "lib/planted")
        p = run(str(z))
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("plugin.zip!lib/planted: ", p.stdout)
        self.assertNotIn("plugin.zip!lib/clean: ", p.stdout)

    def test_a_directory_is_every_file_under_it(self) -> None:
        d = self.tmp / "dist"
        (d / "a").mkdir(parents=True)
        shutil.copy(self.clean, d / "clean")
        shutil.copy(self.planted, d / "a" / "planted")
        p = run(str(d))
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("2 artifacts", p.stdout)
        self.assertIn("planted: ", p.stdout)


class Planted(unittest.TestCase):
    """each kind on its own, read in-process"""

    def audit(self, data: bytes, name: str = "artifact") -> rea.Audit:
        a = rea.Audit(Checker.get())
        a.read(name, name, data, 0, [rea.BUDGET])
        return a

    def test_a_private_name_held_as_a_hash_is_a_finding(self) -> None:
        c = Checker.get()
        fake = "zqvplantedproject"
        h = hashlib.sha256(fake.encode()).hexdigest()
        data = b"\0\0header\0" + f"built for {fake} release".encode() + b"\0"
        self.assertEqual(self.audit(data).findings, [], "the name is not private until its hash is held")
        c.PRIVATE_NAME_HASHES.add(h)
        try:
            self.assertTrue(any("private name" in f for f in self.audit(data).findings))
        finally:
            c.PRIVATE_NAME_HASHES.discard(h)

    def test_an_internal_word_is_a_finding(self) -> None:
        data = b"\0\0" + f"spawned by the {INTERNAL_WORD} runner".encode() + b"\0"
        self.assertTrue(any("agent process" in f for f in self.audit(data).findings))

    def test_a_token_like_string_is_a_finding(self) -> None:
        for data in (b"\0\0auth=" + TOKEN.encode() + b"\0", ("const k = '" + TOKEN + "';\n").encode()):
            with self.subTest(binary=b"\0" in data):
                f = self.audit(data).findings
                self.assertIn("artifact: 1 github token", f)
                self.assertFalse(any(TOKEN in x for x in f))
        # the shape needs its length: a prefix alone is not a token
        self.assertEqual(self.audit(b"\0\0ghp_short\0padding text\0").findings, [])

    def test_a_home_path_in_utf16_is_a_finding(self) -> None:
        data = b"\0\0" + (PLANTED_HOME + "/project/src/main.rs").encode("utf-16-le") + b"\0\0"
        self.assertTrue(any("binary home path" in f for f in self.audit(data).findings))

    def test_a_home_path_in_a_web_bundle_is_a_finding(self) -> None:
        data = f'//# sourceURL=file://{PLANTED_HOME}/project/web/app.js\nconsole.log(1);\n'.encode()
        self.assertTrue(any("home path" in f for f in self.audit(data, "app.js").findings))

    def test_a_container_that_cannot_be_read_is_a_finding(self) -> None:
        f = self.audit(b"PK\x03\x04 not really a zip", "broken.zip").findings
        self.assertTrue(any("unscanned container" in x for x in f), f)

    def test_an_empty_container_is_a_finding(self) -> None:
        # nothing read out of it is not "nothing found in it"
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w"):
            pass
        t = io.BytesIO()
        with tarfile.open(fileobj=t, mode="w"):
            pass
        # an empty tar has no ustar header: it is a container by its name, one no reader opens
        for name, data, kind in (("empty.zip", z.getvalue(), "zip, empty"), ("empty.tar", t.getvalue(), "unknown container")):
            with self.subTest(name=name):
                f = self.audit(data, name).findings
                self.assertEqual(f, [f"{name}: {kind}: unscanned container"])

    def test_containers_nested_too_deep_are_a_finding(self) -> None:
        data = b"\0\0clean bytes here\0"
        for k in range(rea.DEPTH + 1):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as w:
                w.writestr(f"m{k}", data)
            data = buf.getvalue()
        f = self.audit(data, "deep.zip").findings
        self.assertTrue(any("nested too deep" in x for x in f), f)

    def test_a_copy_of_a_tracked_file_is_left_to_the_public_text_check(self) -> None:
        data = b"\0\0" + f"spawned by the {INTERNAL_WORD} runner".encode() + b"\0"
        a = rea.Audit(Checker.get(), {rea.blob_id(data)})
        a.read("pkg", "pkg", data, 0, [rea.BUDGET])
        self.assertEqual((a.files, a.same_as_tracked, a.strings, a.findings), (1, 1, 0, []))
        # one byte more is another file, read as any other
        a = rea.Audit(Checker.get(), {rea.blob_id(data)})
        a.read("pkg", "pkg", data + b"x", 0, [rea.BUDGET])
        self.assertTrue(any("agent process" in f for f in a.findings))

    def test_the_blob_id_is_git_s(self) -> None:
        readme = ROOT / "README.md"
        want = subprocess.run(["git", "-C", str(ROOT), "hash-object", str(readme)], check=True, capture_output=True,
                              text=True).stdout.strip()
        self.assertEqual(rea.blob_id(readme.read_bytes()), want)

    def test_the_strings_are_counted(self) -> None:
        a = self.audit(b"\0\0first string\0second string\0")
        self.assertEqual((a.files, a.strings, a.findings), (1, 2, []))


def git_bash() -> str:
    """The bash the release workflows run the script with (`shell: bash`). On Windows that is
    Git Bash: a bare `bash` there is System32's WSL launcher, which Windows finds before PATH."""
    if os.name != "nt":
        return "bash"
    exec_path = Path(subprocess.run(["git", "--exec-path"], check=True, capture_output=True,
                                    text=True).stdout.strip())  # <git>/mingw64/libexec/git-core
    for bash in (exec_path.parents[2] / "bin" / "bash.exe", exec_path.parents[2] / "usr" / "bin" / "bash.exe"):
        if bash.is_file():
            return str(bash)
    raise AssertionError(f"Git Bash not found next to {exec_path}")


class RemapFlags(unittest.TestCase):
    SCRIPT = ROOT / "scripts" / "release_remap_flags.sh"

    def setUp(self) -> None:
        # a cargo home in the native form of the OS (C:\... on Windows), as CARGO_HOME is set there
        self.tmp = Path(tempfile.mkdtemp(prefix="rea-remap-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def flags(self, cargo_home: Path) -> subprocess.CompletedProcess:
        return subprocess.run([git_bash(), str(self.SCRIPT)], capture_output=True, text=True, timeout=60,
                              env=dict(os.environ, CARGO_HOME=str(cargo_home)))

    def test_the_checkout_and_the_cargo_home_are_remapped(self) -> None:
        home = self.tmp / "plantedname" / ".cargo"
        p = self.flags(home)
        self.assertEqual(p.returncode, 0, p.stderr)
        # rustc records native paths, so the prefixes are native too (the script converts on Windows)
        self.assertEqual(p.stdout.split(), [f"--remap-path-prefix={ROOT.parent}=/src",
                                            f"--remap-path-prefix={home}=/cargo",
                                            "--remap-path-prefix=/root/.cargo=/cargo"])

    def test_a_path_with_a_space_exits_2(self) -> None:
        p = self.flags(self.tmp / "planted name" / ".cargo")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)


class NothingCompared(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rea-empty-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_no_artifact_exits_2(self) -> None:
        for args in ((), (str(self.tmp),)):
            with self.subTest(args=args):
                p = run(*args)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn("no artifact given", p.stderr)

    def test_a_missing_path_exits_2(self) -> None:
        p = run(str(self.tmp / "absent"))
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)

    def test_no_string_read_exits_2(self) -> None:
        f = self.tmp / "noise.bin"
        f.write_bytes(bytes(range(32)) * 4)
        p = run(str(f))
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("no string compared", p.stderr)

    def test_a_checker_with_another_hash_exits_2(self) -> None:
        other = self.tmp / "public_text_check.py"
        other.write_bytes(Checker.path.read_bytes() + b"\n")
        f = self.tmp / "a.txt"
        f.write_text("plain text\n")
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "release_artifact_audit.py"), "--checker",
                            str(other), str(f)], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("says", p.stderr)


if __name__ == "__main__":
    unittest.main()
