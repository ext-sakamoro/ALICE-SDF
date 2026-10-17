#!/usr/bin/env python3
"""Tests for scripts/build_flags_check.py: the `Running` lines are read with
and without colour, each rule fires on a rustc command line, the target-feature
policy passes, a run that shows no command line, none for this crate, or this
crate as Fresh fails, a failing command fails, the target directory is always
a fresh one and is removed afterwards, and no temporary directory is left
behind."""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_flags_check as b  # noqa: E402

ESC = "\x1b"


def running(cmd: str, colour: bool = False) -> str:
    word = f"{ESC}[1m{ESC}[92m     Running{ESC}[0m" if colour else "     Running"
    return f"{word} `{cmd}`"


def rustc(crate: str, extra: str = "") -> str:
    return (f"rustc --crate-name {crate} --edition=2021 src/lib.rs --crate-type lib "
            f"--emit=dep-info,metadata -C opt-level=3 {extra}").strip()


class Parse(unittest.TestCase):
    def test_clean_run(self):
        out = "\n".join([running(rustc("libc", "-Dwarnings")), running(rustc("alice_db", "-Dwarnings"), True)])
        findings, seen, own, fresh = b.parse(out, "alice_db")
        self.assertEqual((findings, seen, own, fresh), ([], 2, 1, False))

    def test_colour_and_charset_escapes_are_stripped(self):
        out = running(rustc("alice_db", "-C target-cpu=native"), True) + f"{ESC}(B\n"
        findings, seen, own, _ = b.parse(out, "alice_db")
        self.assertEqual(seen, 1)
        self.assertEqual(own, 1)
        self.assertEqual(findings, ["alice_db: target-cpu-native: target-cpu=native"])

    def test_each_rule(self):
        for extra, rule in [("-C target-cpu=native", "target-cpu-native"),
                            ("-Ctarget_cpu=native", "target-cpu-native"),
                            ("-C llvm-args=-enable-unsafe-fp-math", "llvm-fast-math"),
                            ("-C llvm-args=-denormal-fp-math=preserve-sign", "llvm-fast-math"),
                            ("-C llvm-args=-fp-contract=fast", "llvm-fast-math"),
                            ("-C target-feature=+avx,-sse2", "x87-float"),
                            ("'-C' 'target-feature=-sse'", "x87-float")]:
            with self.subTest(extra=extra):
                findings, *_ = b.parse(running(rustc("dep", extra)) + "\n" + running(rustc("alice_db")), "alice_db")
                self.assertEqual(len(findings), 1, findings)
                self.assertIn(f"dep: {rule}:", findings[0])

    def test_policy_and_look_alikes_pass(self):
        extra = "-C target-cpu=x86-64-v2 -C target-feature=+avx2,+fma,-avx,-sse4.1 -C target-feature=+dotprod"
        findings, *_ = b.parse(running(rustc("alice_db", extra)), "alice_db")
        self.assertEqual(findings, [])

    def test_fresh_and_missing(self):
        _, seen, own, fresh = b.parse("       Fresh alice-db v0.5.1 (/x)\n", "alice_db")
        self.assertEqual((seen, own, fresh), (0, 0, True))
        _, seen, own, _ = b.parse(running(rustc("libc")), "alice_db")
        self.assertEqual((seen, own), (1, 0))


FAKE = textwrap.dedent('''
    import sys
    lines = {lines!r}
    for l in lines:
        print(l, file=sys.stderr)
    assert "-v" in sys.argv and "--target-dir" in sys.argv
    sys.exit({rc})
''')


# every temporary directory of this run, the fake repositories and the target
# directories build_flags_check.main makes, lives under one base directory, which
# must be empty again at the end (tearDownModule fails otherwise) and is then removed
BASE = Path(tempfile.mkdtemp(prefix="build-flags-tests-"))


def tearDownModule():
    left = sorted(p.name for p in BASE.iterdir())
    shutil.rmtree(BASE, ignore_errors=True)
    if left:
        raise AssertionError(f"{len(left)} temporary directories were left behind: {left[:5]}")


def run_fake(lines, rc=0, extra_args=()):
    d = Path(tempfile.mkdtemp(prefix="repo-", dir=BASE))
    targets = BASE / "targets"
    targets.mkdir(exist_ok=True)
    saved = tempfile.tempdir
    tempfile.tempdir = str(targets)  # main() makes its fresh target directory here
    try:
        (d / "Cargo.toml").write_text('[package]\nname = "alice-db"\nversion = "0.1.0"\n', encoding="utf-8")
        fake = d / "fake_cargo.py"
        fake.write_text(FAKE.format(lines=list(lines), rc=rc), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = b.main(["--root", str(d), "--", sys.executable, str(fake), *extra_args])
        left = sorted(p.name for p in targets.iterdir())
        return code, out.getvalue() + (f"\nTARGET LEFT: {left}" if left else "")
    finally:
        tempfile.tempdir = saved
        shutil.rmtree(d)
        shutil.rmtree(targets, ignore_errors=True)


class Main(unittest.TestCase):
    def test_clean(self):
        code, out = run_fake([running(rustc("libc")), running(rustc("alice_db"))])
        self.assertEqual(code, 0, out)
        self.assertIn("2 rustc invocations (1 for alice_db), 0 problems", out)

    def test_finding(self):
        code, out = run_fake([running(rustc("alice_db", "-C target-cpu=native"))])
        self.assertEqual(code, 1, out)
        self.assertIn("alice_db: target-cpu-native:", out)

    def test_nothing_seen_fails(self):
        code, out = run_fake(["   Compiling alice-db v0.1.0"])
        self.assertEqual(code, 1, out)
        self.assertIn("no rustc command line was seen", out)

    def test_own_crate_missing_fails(self):
        code, out = run_fake([running(rustc("libc"))])
        self.assertEqual(code, 1, out)
        self.assertIn("no rustc command line for alice_db", out)

    def test_own_crate_fresh_fails(self):
        code, out = run_fake([running(rustc("libc")), "       Fresh alice-db v0.1.0 (/x)"])
        self.assertEqual(code, 1, out)
        self.assertIn("alice_db was Fresh", out)

    def test_failing_command_fails(self):
        code, out = run_fake([running(rustc("alice_db"))], rc=101)
        self.assertEqual(code, 1, out)
        self.assertIn("the command failed (exit 101)", out)

    def test_the_fresh_target_dir_is_removed(self):
        for lines, rc in (([running(rustc("alice_db"))], 0), ([running(rustc("alice_db"))], 101), ([], 0)):
            with self.subTest(rc=rc, lines=len(lines)):
                _, out = run_fake(lines, rc=rc)
                self.assertNotIn("TARGET LEFT", out)

    def test_target_dir_is_chosen_here(self):
        code, _ = run_fake([running(rustc("alice_db"))], extra_args=("--target-dir", "target"))
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
