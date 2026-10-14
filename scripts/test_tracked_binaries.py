#!/usr/bin/env python3
"""Tests of scripts/tracked_binaries.py: a tracked built binary fails (by extension in any
letter case, by bundle, or by its header), the allow list waives only the paths it names
and needs a reason, a listed path that is gone fails, an unreadable tracked file fails,
and a repository with no tracked file is "compared nothing" (exit 2), not a pass.

usage: python3 scripts/test_tracked_binaries.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tracked_binaries as tb  # noqa: E402


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", *args], cwd=cwd, check=True,
                   capture_output=True)


def pe(off: int = 0x80) -> bytes:
    """a minimal PE image: the DOS header names the offset of `PE\\0\\0`"""
    head = bytearray(b"MZ" + b"\x00" * (off - 2))
    head[0x3C:0x40] = off.to_bytes(4, "little")
    return bytes(head) + b"PE\x00\x00" + b"\x00" * 16


class TrackedBinaries(unittest.TestCase):
    def setUp(self) -> None:
        self.r = Path(tempfile.mkdtemp(prefix="tracked-binaries-test-"))
        self.addCleanup(shutil.rmtree, self.r, True)
        git("init", "-q", cwd=self.r)
        self.add("src/lib.rs", b"pub fn f() {}\n")

    def add(self, path: str, data: bytes) -> None:
        p = self.r / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        git("add", path, cwd=self.r)

    def drop(self, path: str) -> None:
        git("rm", "-q", "--cached", path, cwd=self.r)

    def run_check(self) -> int:
        return tb.main([str(self.r)])

    def test_a_clean_tree_passes(self) -> None:
        self.add("README.md", b"# x\n")
        self.add("Main.class", b"\xca\xfe\xba\xbe\x00\x00\x00\x41rest")  # (a Java class, version 65: not a library)
        self.assertEqual(self.run_check(), 0)

    def test_nothing_tracked_is_not_a_pass(self) -> None:
        self.drop("src/lib.rs")
        self.assertEqual(self.run_check(), 2)

    def test_a_tracked_binary_fails_by_name(self) -> None:
        for path in ("app/jniLibs/arm64-v8a/libx.so", "lib/libx.dylib", "lib/libx.a", "out/x.aar", "X.DLL",
                     "bin/x.exe", "web/x.wasm", "addon/x.node", "lib/LIBX.SO",
                     "Frameworks/X.xcframework/Info.plist", "Frameworks/X.XCFramework/Info.plist"):
            with self.subTest(path=path):
                self.add(path, b"not a header\n")
                self.assertEqual(self.run_check(), 1)
                self.drop(path)

    def test_a_binary_without_its_name_fails_by_header(self) -> None:
        for name, data in (("elf", b"\x7fELF\x02\x01\x01\x00"), ("macho64", b"\xcf\xfa\xed\xfe\x07\x00\x00\x01"),
                           ("macho32", b"\xce\xfa\xed\xfe\x07\x00\x00\x00"), ("fat", b"\xca\xfe\xba\xbe\x00\x00\x00\x02"),
                           ("fat64", b"\xca\xfe\xba\xbf\x00\x00\x00\x02"), ("ar", b"!<arch>\n"),
                           ("wasm", b"\x00asm\x01\x00\x00\x00"), ("pe", pe())):
            with self.subTest(name=name):
                self.add(f"bin/{name}.bin", data + b"rest")
                self.assertEqual(self.run_check(), 1)
                self.drop(f"bin/{name}.bin")

    def test_mz_without_a_pe_header_is_not_a_binary(self) -> None:
        self.add("notes/mz.txt", b"MZ is a word too\n" + b"x" * 80)
        self.assertEqual(self.run_check(), 0)

    def test_the_allow_list_waives_only_what_it_names(self) -> None:
        self.add("vendor/libkeep.so", b"\x7fELF")
        self.add(".github/tracked-binaries.txt", b"vendor/libkeep.so  # a vendored runtime the build cannot make\n")
        self.assertEqual(self.run_check(), 0)
        # a listed binary next to an unlisted one: the unlisted one still fails
        self.add("lib/x.so", b"\x7fELF")
        self.assertEqual(self.run_check(), 1)
        self.drop("lib/x.so")
        # an entry without a reason
        self.add(".github/tracked-binaries.txt", b"vendor/libkeep.so\n")
        self.assertEqual(self.run_check(), 1)
        # an entry whose file is no longer tracked
        self.add(".github/tracked-binaries.txt", b"vendor/libkeep.so # r\nvendor/gone.so # r\n")
        self.assertEqual(self.run_check(), 1)

    @unittest.skipIf(os.name == "nt" or os.geteuid() == 0, "file modes do not deny reading here")
    def test_an_unreadable_tracked_file_fails(self) -> None:
        self.add("data/blob", b"\x7fELF")
        (self.r / "data/blob").chmod(0)
        self.addCleanup((self.r / "data/blob").chmod, 0o644)
        self.assertEqual(self.run_check(), 1)

    @unittest.skipIf(os.name == "nt", "symbolic links need extra rights here")
    def test_a_tracked_symlink_is_judged_by_its_name_not_its_target(self) -> None:
        # a link is not followed: a link to text, and a link to an untracked ELF image, are both not binaries
        os.symlink("src/lib.rs", self.r / "link")
        git("add", "link", cwd=self.r)
        self.assertEqual(self.run_check(), 0)
        (self.r / "build.out").write_bytes(b"\x7fELFrest")
        os.symlink("build.out", self.r / "link2")
        git("add", "link2", cwd=self.r)
        self.assertEqual(self.run_check(), 0)
        # its name still counts
        os.symlink("src/lib.rs", self.r / "libx.so")
        git("add", "libx.so", cwd=self.r)
        self.assertEqual(self.run_check(), 1)

    def test_a_submodule_is_not_read_as_a_file(self) -> None:
        # a submodule is a directory in the work tree (its own repository checks its own files)
        sub = self.r / "vendor" / "sub"
        sub.mkdir(parents=True)
        git("init", "-q", cwd=sub)
        (sub / "a.txt").write_bytes(b"a\n")
        git("add", "a.txt", cwd=sub)
        git("commit", "-qm", "a", cwd=sub)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=sub, capture_output=True, text=True,
                             check=True).stdout.strip()
        git("update-index", "--add", "--cacheinfo", f"160000,{sha},vendor/sub", cwd=self.r)
        self.assertEqual(self.run_check(), 0)

    def test_this_repository_passes(self) -> None:
        self.assertEqual(tb.main([str(HERE.parent)]), 0)


if __name__ == "__main__":
    unittest.main()
