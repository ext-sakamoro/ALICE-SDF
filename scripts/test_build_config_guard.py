#!/usr/bin/env python3
"""Tests for scripts/build_config_guard.py: a clean tree passes, every rule
fires in every file kind it applies to (one mutant per rule and place), a flag
reached through a variable, matrix, env entry, anchor, continued line or
wrapper fires, a comment or a YAML name does not while a string does, the
target-feature policy passes, the src intrinsics fire, the allowlist covers
exact lines with an exact count and goes stale, the script's own files and
test data need an exclude, an unreadable or unparseable file fails, and a tree
with nothing to read (or no root Cargo.toml) is exit 2."""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_config_guard as g  # noqa: E402

PKG = '[package]\nname = "x"\nversion = "0.1.0"\nedition = "2021"\n'
NATIVE = "-C target-cpu=native"


# every temporary tree of this run lives under one base directory, which must be
# empty again at the end (tearDownModule fails otherwise) and is then removed
BASE = Path(tempfile.mkdtemp(prefix="build-config-guard-tests-"))


def tearDownModule():
    left = sorted(p.name for p in BASE.iterdir())
    shutil.rmtree(BASE, ignore_errors=True)
    if left:
        raise AssertionError(f"{len(left)} temporary trees were left behind: {left[:5]}")


def tree(files: dict[str, str | bytes]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="tree-", dir=BASE))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            p.write_bytes(text)
        else:
            p.write_text(text, encoding="utf-8")
    return root


def run(files: dict[str, str | bytes]) -> tuple[int, str, str]:
    root = tree(files)
    try:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = g.main(["--root", str(root)])
        return rc, out.getvalue(), err.getvalue()
    finally:
        shutil.rmtree(root)


def red(test, files, where):
    rc, out, _ = run(files)
    test.assertEqual(rc, 1, out)
    test.assertIn(where, out)


# (file, text holding the flag on line `line`)
PLACES = {
    "cargo-config": (".cargo/config.toml", '[build]\nrustflags = ["-C", "{flag}"]\n', 2),
    "cargo-config-legacy": (".cargo/config", '[target.x86_64-unknown-linux-gnu]\nrustflags = ["{flag}"]\n', 2),
    "cargo-config-cfg": (".cargo/config.toml", "[target.'cfg(target_arch = \"x86_64\")']\nrustflags = [\"{flag}\"]\n", 2),
    "nested-cargo-config": ("crates/a/.cargo/config.toml", '[env]\nRUSTFLAGS = "{flag}"\n', 2),
    "fuzz-cargo-config": ("fuzz/.cargo/config.toml", '[build]\nrustflags = [\n  "-C",\n  "{flag}",\n]\n', 4),
    "cargo-alias": (".cargo/config.toml", '[alias]\nb = "rustc --release -- {flag}"\n', 2),
    "cargo-profile": ("Cargo.toml", 'cargo-features = ["profile-rustflags"]\n' + PKG
                      + '\n[profile.release]\nrustflags = ["{flag}"]\n', 8),
    "pyproject": ("pyproject.toml", '[tool.maturin]\nrustc-args = ["{flag}"]\n', 2),
    "cargo-make": ("Makefile.toml", '[env]\nRUSTFLAGS = "{flag}"\n', 2),
    "workflow-env": (".github/workflows/ci.yml", 'env:\n  RUSTFLAGS: "-C {flag}"\njobs: {{}}\n', 2),
    "workflow-run": (".github/workflows/x.yaml", 'jobs:\n  a:\n    steps:\n      - run: RUSTFLAGS="{flag}" cargo build\n', 4),
    "action": (".github/actions/setup/action.yml", 'runs:\n  using: composite\n  env:\n    CARGO_ENCODED_RUSTFLAGS: "{flag}"\n', 4),
    "compose": ("docker-compose.yml", 'services:\n  b:\n    environment:\n      - RUSTFLAGS={flag}\n', 4),
    "devcontainer": (".devcontainer/devcontainer.json", '{{\n  "containerEnv": {{ "RUSTFLAGS": "{flag}" }}\n}}\n', 2),
    "makefile": ("Makefile", 'build:\n\tRUSTFLAGS="{flag}" cargo build\n', 2),
    "mk": ("make/rules.mk", 'X := 1\nRUSTFLAGS += {flag}\n', 2),
    "justfile": ("justfile", 'build:\n    RUSTFLAGS="{flag}" cargo build\n', 2),
    "shell": ("scripts/preflight.sh", '#!/usr/bin/env bash\nexport RUSTFLAGS="{flag}"\n', 2),
    "shell-subshell": ("ci/run.sh", '#!/usr/bin/env bash\n( export RUSTFLAGS="{flag}"; cargo build )\n', 2),
    "powershell": ("scripts/b.ps1", '$x = 1\n$env:RUSTFLAGS = "{flag}"\n', 2),
    "batch": ("b.cmd", '@echo off\nset RUSTFLAGS={flag}\n', 2),
    "envrc": (".envrc", 'use flake\nexport RUSTFLAGS="{flag}"\n', 2),
    "dotenv": (".env", 'A=1\nRUSTFLAGS={flag}\n', 2),
    "dockerfile": ("Dockerfile", 'FROM rust\nRUN RUSTFLAGS="{flag}" cargo build --release\n', 2),
    "nix": ("shell.nix", '{{ pkgs }}:\npkgs.mkShell {{ RUSTFLAGS = "{flag}"; }}\n', 2),
    "python-anywhere": ("tools/b.py", 'import os\nos.environ["RUSTFLAGS"] = "{flag}"\n', 2),
    "xtask": ("xtask/src/main.rs", 'fn main() {{\n    Command::new("cargo").env("RUSTFLAGS", "{flag}");\n}}\n', 2),
    "build-rs": ("build.rs", 'fn main() {{\n    println!("cargo:rustc-flags={flag}");\n}}\n', 2),
    "build-rs-cc": ("build.rs", 'fn main() {{\n    cc::Build::new().flag_if_supported("{flag}").compile("x");\n}}\n', 2),
}

FLAGS = {
    "target-cpu-native": ["target-cpu=native", "-Ctarget-cpu=native", "target-cpu = native",
                          "target_cpu=native", "-Ctarget_cpu=native", "-C\\x1ftarget-cpu=native"],
    "llvm-fast-math": ["llvm-args=-enable-unsafe-fp-math", "llvm-args=-enable-no-nans-fp-math",
                       "llvm-args=-enable-no-infs-fp-math", "llvm-args=-enable-no-signed-zeros-fp-math",
                       "llvm_args=-fp-contract=fast", "llvm-args=-denormal-fp-math=preserve-sign"],
    "c-fast-math": ["-march=native", "-mcpu=native", "-mfma", "-ffast-math", "-Ofast",
                    "-funsafe-math-optimizations", "-fassociative-math", "-freciprocal-math",
                    "-ffinite-math-only", "-fno-signed-zeros", "-ffp-contract=fast", "-ffp-model=fast",
                    "-fdenormal-fp-math=positive-zero", "-mdaz-ftz", "/fp:fast", "-fp-model fast",
                    "-mno-sse2", "-mfpmath=387"],
    "x87-float": ["target-feature=-sse2", "target-feature=+avx,-sse", "target_feature=-sse2"],
}


class Clean(unittest.TestCase):
    def test_minimal_tree_passes(self):
        rc, out, _ = run({"Cargo.toml": PKG})
        self.assertEqual(rc, 0, out)
        self.assertIn("1 files read", out)

    def test_ordinary_flags_and_the_target_feature_policy_pass(self):
        rc, out, _ = run({
            "Cargo.toml": PKG + '\n[profile.release]\nlto = "fat"\nopt-level = 3\n',
            ".github/workflows/ci.yml": 'env:\n  RUSTFLAGS: "-Dwarnings -C target-feature=+avx2,+fma"\n',
            ".cargo/config.toml": '[build]\nrustflags = ["-C", "target-cpu=x86-64-v2"]\n',
            "build.rs": 'fn main() { cc::Build::new().flag("-O2").flag("-march=x86-64-v2").compile("x"); }\n',
            "scripts/a.sh": 'RUSTFLAGS="-C target-feature=-avx2,-avx,-fma,-sse4.1" cargo test\n',
            "scripts/b.sh": 'RUSTFLAGS="-C target-feature=+dotprod" cargo test\n',
            "src/lib.rs": '#[target_feature(enable = "fma")]\npub unsafe fn f(a: f64) -> f64 { a.mul_add(a, a) }\n',
        })
        self.assertEqual(rc, 0, out)

    def test_counts_per_category(self):
        rc, out, _ = run({
            "Cargo.toml": PKG, ".cargo/config.toml": "[build]\n", "build.rs": "fn main() {}\n",
            "src/lib.rs": "\n", ".github/workflows/ci.yml": "jobs: {}\n", "a.json": "{}\n",
            "scripts/a.sh": "true\n",
        })
        self.assertEqual(rc, 0, out)
        self.assertIn("7 files read (cargo-config 1, toml 1, build-rs 1, rust 1, yaml 1, json 1, "
                      "script 1, wrapper 0)", out)


class EveryRuleEveryPlace(unittest.TestCase):
    def test_each_flag_fires_at_its_line(self):
        for rule, flags in FLAGS.items():
            for flag in flags:
                for place, (rel, body, line) in PLACES.items():
                    files = {"Cargo.toml": PKG}
                    files[rel] = body.format(flag=flag)
                    with self.subTest(rule=rule, flag=flag, place=place):
                        red(self, files, f"{rel}:{line}: {rule}:")


class Indirection(unittest.TestCase):
    """A flag that reaches the build through something else is still found."""

    def test_variable_then_use(self):
        red(self, {"Cargo.toml": PKG, "scripts/b.sh": f'F="{NATIVE}"\nRUSTFLAGS="$F" cargo build --release\n'},
            "scripts/b.sh:1: target-cpu-native:")

    def test_matrix_value(self):
        wf = ('jobs:\n  b:\n    strategy:\n      matrix:\n        cpu: [native]\n'
              '    env:\n      RUSTFLAGS: -C target-cpu=${{ matrix.cpu }}\n')
        red(self, {"Cargo.toml": PKG, ".github/workflows/ci.yml": wf},
            ".github/workflows/ci.yml:7: target-cpu-unresolved:")

    def test_unresolved_spellings(self):
        for text in ("RUSTFLAGS=-Ctarget-cpu=$CPU cargo b\n", 'RUSTFLAGS="-C target-cpu=${CPU}" cargo b\n',
                     "all:\n\tRUSTFLAGS=-Ctarget-cpu=$(CPU) cargo b\n", "set RUSTFLAGS=-Ctarget-cpu=%CPU%\n",
                     'RUSTFLAGS="-C target-feature=$FEATURES" cargo b\n'):
            with self.subTest(text=text):
                red(self, {"Cargo.toml": PKG, "b.sh": text}, "unresolved:")

    def test_continued_line(self):
        wf = 'jobs:\n  b:\n    steps:\n      - run: |\n          RUSTFLAGS="-C \\\n          target-cpu=native" cargo build\n'
        red(self, {"Cargo.toml": PKG, ".github/workflows/ci.yml": wf}, ".github/workflows/ci.yml:6: target-cpu-native:")

    def test_env_indirection_and_anchor(self):
        wf = ('env:\n  CPU: "-C target-cpu=native"\njobs:\n  b:\n    env:\n      RUSTFLAGS: ${{ env.CPU }}\n')
        red(self, {"Cargo.toml": PKG, ".github/workflows/ci.yml": wf}, ".github/workflows/ci.yml:2: target-cpu-native:")
        wf = 'x: &rf "-C target-cpu=native"\njobs:\n  b:\n    env:\n      RUSTFLAGS: *rf\n'
        red(self, {"Cargo.toml": PKG, ".github/workflows/ci.yml": wf}, ".github/workflows/ci.yml:1: target-cpu-native:")

    def test_wrapper_without_extension(self):
        files = {"Cargo.toml": PKG, ".cargo/config.toml": '[build]\nrustc-wrapper = "scripts/w"\n',
                 "scripts/w": '#!/bin/sh\nexec "$@" -C target-cpu=native\n'}
        rc, out, _ = run(files)
        self.assertEqual(rc, 1, out)
        self.assertIn("scripts/w:2: target-cpu-native:", out)
        self.assertIn("wrapper 1", out)
        files = {"Cargo.toml": PKG, ".github/workflows/ci.yml": "env:\n  RUSTC_WRAPPER: tools/wrap\n",
                 "tools/wrap": '#!/bin/sh\nexec "$@" -Ofast\n'}
        red(self, files, "tools/wrap:2: c-fast-math:")

    def test_mentions_are_findings(self):
        # a mention in an echo or a message configures nothing, but is reported: it is
        # recorded in the allowlist with its reason, never passed silently
        red(self, {"Cargo.toml": PKG, "scripts/a.sh": 'echo "never use -C target-cpu=native"\n'},
            "scripts/a.sh:1: target-cpu-native:")


class Comments(unittest.TestCase):
    def test_hash_comments_and_yaml_names_do_not_fire(self):
        rc, out, _ = run({
            "Cargo.toml": PKG + '# RUSTFLAGS="-C target-cpu=native" cargo build --release\n',
            ".cargo/config.toml": '[build]\n# rustflags = ["-C", "target-cpu=native"]\n'
                                  '[profile.release]\nopt-level = 3 # target-cpu=native\n',
            ".github/workflows/ci.yml": ('env:\n  RUSTFLAGS: "-Dwarnings"  # never -C target-cpu=native\n'
                                         '  # RUSTFLAGS: -C target-cpu=native\njobs:\n  a:\n'
                                         '    name: no target-cpu=native here\n    steps:\n'
                                         '      - name: build without -Ofast\n        run: cargo build\n'),
            "scripts/a.sh": '#!/bin/sh\n  # export RUSTFLAGS=-ffast-math\necho ${#x}\n',
            "a.json": '{\n  // "RUSTFLAGS": "-C target-cpu=native"\n  "a": 1 /* -Ofast */\n}\n',
            "b.bat": "rem set RUSTFLAGS=-C target-cpu=native\n:: -Ofast\n",
        })
        self.assertEqual(rc, 0, out)

    def test_hash_inside_string_is_not_a_comment(self):
        red(self, {"Cargo.toml": PKG, "scripts/a.sh": 'X="a # b -C target-cpu=native"\n'},
            "scripts/a.sh:1: target-cpu-native:")

    def test_rust_comments_do_not_fire_but_strings_do(self):
        build = (
            'fn main() {\n'
            '    // println!("cargo:rustc-flags=-C target-cpu=native");\n'
            '    /* cc::Build::new().flag("-ffast-math");\n'
            '       /* nested */ still a comment -Ofast */\n'
            '    let s = "// not a comment -ffast-math";\n'
            '    let r = r#"-C target-cpu=native"#;\n'
            '}\n')
        rc, out, _ = run({"Cargo.toml": PKG, "build.rs": build})
        self.assertEqual(rc, 1, out)
        self.assertIn("build.rs:5: c-fast-math:", out)
        self.assertIn("build.rs:6: target-cpu-native:", out)
        for n in (2, 3, 4):
            self.assertNotIn(f"build.rs:{n}:", out)


class Source(unittest.TestCase):
    def test_fast_intrinsics_fire_in_src(self):
        for text in ("#![feature(core_intrinsics)]\n", "#![feature(float_algebraic)]\n",
                     "let y = unsafe { core::intrinsics::fadd_fast(a, b) };\n",
                     "let y = unsafe { std::intrinsics::fmul_fast(a, b) };\n",
                     "let y = core::intrinsics::fadd_algebraic(a, b);\n",
                     "let y = a.algebraic_mul(b);\n", "use core::intrinsics::{fdiv_fast, frem_fast};\n"):
            with self.subTest(text=text):
                red(self, {"Cargo.toml": PKG, "crates/m/src/x.rs": "\n" + text}, "crates/m/src/x.rs:2: fast-intrinsic:")

    def test_comments_and_non_src_do_not_fire(self):
        rc, out, _ = run({"Cargo.toml": PKG,
                          "src/lib.rs": "// fadd_fast is never used\n/* #![feature(core_intrinsics)] */\n",
                          "tests/t.rs": "#![feature(core_intrinsics)]\n",
                          "src/fast.rs": "pub fn add_fast_path() {}\n"})
        self.assertEqual(rc, 0, out)


class Allowlist(unittest.TestCase):
    LEG = ('jobs:\n  t:\n    strategy:\n      matrix:\n'
           '        rustflags: ["", "-C target-cpu=native"]\n')
    LINE = 'rustflags: ["", "-C target-cpu=native"]'
    REASON = "test leg: results must not depend on the CPU; not a shipped build"

    def files(self, body, leg=None):
        return {"Cargo.toml": PKG, ".github/workflows/ci.yml": leg or self.LEG,
                "scripts/build_config_allowlist.toml": body}

    def entry(self, line=None, count=None, reason=None, path=".github/workflows/ci.yml"):
        s = f'[[allow]]\npath = "{path}"\nline = \'{line or self.LINE}\'\nreason = "{reason or self.REASON}"\n'
        return s + (f"count = {count}\n" if count is not None else "")

    def test_allow_covers_and_reports(self):
        rc, out, _ = run(self.files(self.entry()))
        self.assertEqual(rc, 0, out)
        self.assertIn("allowed .github/workflows/ci.yml:5: target-cpu-native:", out)
        self.assertIn("allow entry 1: .github/workflows/ci.yml: 1 of 1 lines", out)
        self.assertIn("1 allowed, 0 problems", out)

    def test_without_allow_the_leg_fails(self):
        red(self, {"Cargo.toml": PKG, ".github/workflows/ci.yml": self.LEG}, ".github/workflows/ci.yml:5:")

    def test_a_new_occurrence_in_the_same_file_is_a_finding(self):
        leg = self.LEG + '  release:\n    env:\n      RUSTFLAGS: -C target-cpu=native\n'
        rc, out, _ = run(self.files(self.entry(), leg))
        self.assertEqual(rc, 1, out)
        self.assertIn(".github/workflows/ci.yml:8: target-cpu-native:", out)
        # a substring of the line is not the line
        red(self, self.files(self.entry(line="target-cpu=native")), "covers 0 lines, count is 1")

    def test_count_must_match(self):
        leg = self.LEG + '  u:\n    strategy:\n      matrix:\n        rustflags: ["", "-C target-cpu=native"]\n'
        red(self, self.files(self.entry(), leg), "covers 2 lines, count is 1")
        rc, out, _ = run(self.files(self.entry(count=2), leg))
        self.assertEqual(rc, 0, out)
        red(self, self.files(self.entry(count=3), leg), "covers 2 lines, count is 3")

    def test_stale_or_bad_entries_fail(self):
        cases = {
            "other path": self.entry(path="scripts/a.sh"),
            "empty reason": self.entry(reason=" "),
            "no reason": f'[[allow]]\npath = ".github/workflows/ci.yml"\nline = \'{self.LINE}\'\n',
            "unknown key": self.entry() + 'scope = "x"\n',
            "old text key": f'[[allow]]\npath = ".github/workflows/ci.yml"\ntext = "native"\nreason = "r"\n',
            "zero count": self.entry(count=0),
            "string count": self.entry() + 'count = "1"\n',
            "not toml": "[[allow]\n",
            "unknown table": '[[permit]]\npath = "x"\nreason = "r"\n',
        }
        for name, body in cases.items():
            with self.subTest(name=name):
                red(self, self.files(body), "allowlist:")

    def test_line_matching_only_a_comment_covers_nothing(self):
        leg = self.LEG.replace('native"]', 'native"]  # cpu-independence')
        rc, out, _ = run(self.files(self.entry(line=self.LINE + "  # cpu-independence"), leg))
        self.assertEqual(rc, 1, out)
        self.assertIn("covers 0 lines", out)
        self.assertIn(".github/workflows/ci.yml:5: target-cpu-native:", out)

    def test_exclude(self):
        fixture = {"Cargo.toml": PKG,
                   "testdata/flags.toml": 'cases = ["-C target-cpu=native"]\n'}
        red(self, fixture, "testdata/flags.toml:1:")
        fixture["scripts/build_config_allowlist.toml"] = (
            '[[exclude]]\npath = "testdata/flags.toml"\nreason = "test data, read by no build"\n')
        rc, out, _ = run(fixture)
        self.assertEqual(rc, 0, out)
        self.assertIn("exclude: testdata/flags.toml: not read", out)
        for body in ('[[exclude]]\npath = "testdata/missing.toml"\nreason = "r"\n',
                     '[[exclude]]\npath = "README.md"\nreason = "r"\n',
                     '[[exclude]]\npath = "testdata/flags.toml"\nreason = ""\n',
                     '[[exclude]]\npath = "testdata/flags.toml"\nreason = "r"\nline = "x"\n'):
            with self.subTest(body=body):
                f = dict(fixture, **{"README.md": "x\n", "scripts/build_config_allowlist.toml": body})
                red(self, f, "allowlist:")

    def test_own_files_and_test_data_need_an_exclude(self):
        files = {"Cargo.toml": PKG,
                 "scripts/build_config_guard.py": 'R = "-C target-cpu=native"\n',
                 "scripts/test_build_config_guard.py": 'F = "-Ofast"\n',
                 "scripts/test_preflight.py": f'CASE = \'RUSTFLAGS="{NATIVE}" cargo test\'\n'}
        rc, out, _ = run(files)
        self.assertEqual(rc, 1, out)
        for rel in ("scripts/build_config_guard.py", "scripts/test_build_config_guard.py", "scripts/test_preflight.py"):
            self.assertIn(f"{rel}:1:", out)
        files["scripts/build_config_allowlist.toml"] = "".join(
            f'[[exclude]]\npath = "{rel}"\nreason = "holds the flags as patterns or test data"\n'
            for rel in ("scripts/build_config_guard.py", "scripts/test_build_config_guard.py",
                        "scripts/test_preflight.py"))
        rc, out, _ = run(files)
        self.assertEqual(rc, 0, out)


class Scope(unittest.TestCase):
    def test_skipped_dirs_and_documents_are_not_read(self):
        rc, out, _ = run({
            "Cargo.toml": PKG,
            "target/debug/build/x/build.rs": 'fn main() { let _ = "-ffast-math"; }\n',
            "node_modules/a/Makefile": "all:\n\tcc -Ofast\n",
            "docs/notes.md": "never use -C target-cpu=native\n",
            "README.md": 'RUSTFLAGS="-C target-cpu=native" cargo build\n',
            "notes.txt": "-ffast-math\n",
        })
        self.assertEqual(rc, 0, out)
        self.assertIn("1 files read", out)


class FailClosed(unittest.TestCase):
    def test_unreadable_file_fails(self):
        red(self, {"Cargo.toml": PKG, ".cargo/config.toml": b"\xff\xfe rustflags\n"}, ".cargo/config.toml:0: unreadable:")

    def test_unparseable_toml_fails(self):
        for rel in ("Cargo.toml", ".cargo/config.toml", "pyproject.toml", "fuzz/Cargo.toml", "Makefile.toml"):
            with self.subTest(rel=rel):
                files = {"Cargo.toml": PKG, rel: "[build\nrustflags = 1\n"}
                red(self, files, f"{rel}:0: unparseable:")

    def test_nothing_read_is_exit_2(self):
        rc, out, err = run({"README.md": "x\n"})
        self.assertEqual(rc, 2, out + err)
        self.assertIn("0 files read", out)

    def test_no_root_manifest_is_exit_2(self):
        rc, out, err = run({"crates/a/Cargo.toml": PKG, ".github/workflows/ci.yml": "jobs: {}\n"})
        self.assertEqual(rc, 2, out + err)


class Closure(unittest.TestCase):
    """A flag value that is not a plain literal is unresolved, whatever builds it."""

    def test_built_values_are_unresolved(self):
        cases = [
            (".cargo/config.toml", '[build]\nrustflags = ["-C", "target-cpu=nat\\u0069ve"]\n', 2, "target-cpu-unresolved"),
            ("scripts/b.sh", "RUSTFLAGS=-Ctarget-cpu=nat'ive' cargo build\n", 1, "target-cpu-unresolved"),
            ("scripts/b.sh", "RUSTFLAGS=$'-C target-cpu=nat\\x69ve' cargo build\n", 1, "target-cpu-unresolved"),
            (".github/workflows/ci.yml", 'env:\n  RUSTFLAGS: "-C target-cpu=nat\\x69ve"\n', 2, "target-cpu-unresolved"),
            ("tools/b.py", 'import os\nos.environ["RUSTFLAGS"] = "-C target-cpu=" + "native"\n', 2, "target-cpu-unresolved"),
            ("tools/c.py", 'F = "-C target-cpu=nat" + "ive"\n', 1, "target-cpu-unresolved"),
            ("tools/d.py", 'F = "-C target-cpu=%s" % cpu\n', 1, "target-cpu-unresolved"),
            ("tools/e.py", 'F = f"-C target-cpu={cpu}"\n', 1, "target-cpu-unresolved"),
            ("xtask/src/main.rs", 'fn main() { let f = format!("-C target-cpu={c}"); }\n', 1, "target-cpu-unresolved"),
            ("scripts/b.sh", 'RUSTFLAGS="-C target-cpu=" cargo build\n', 1, "target-cpu-unresolved"),
            ("scripts/b.sh", 'RUSTFLAGS=-Ctarget-cpu=\\\nnative cargo build\n', 1, "target-cpu-unresolved"),
            ("scripts/b.sh", 'RUSTFLAGS="-C target-feature=+avx2,$EXTRA" cargo build\n', 1, "target-feature-unresolved"),
            ("scripts/b.sh", "RUSTFLAGS=\"-C llvm-args=$OPTS\" cargo build\n", 1, "llvm-args-unresolved"),
            ("scripts/b.sh", "RUSTFLAGS='-C target-feature=-ss'e2 cargo build\n", 1, "target-feature-unresolved"),
        ]
        for rel, text, line, rule in cases:
            with self.subTest(rel=rel, text=text):
                red(self, {"Cargo.toml": PKG, rel: text}, f"{rel}:{line}: {rule}:")

    def test_plain_values_are_judged(self):
        rc, out, _ = run({"Cargo.toml": PKG,
                          "scripts/a.sh": 'RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build\n'
                                          "RUSTFLAGS='-C target-feature=+avx2,+fma' cargo build\n"
                                          "RUSTFLAGS=-Ctarget-cpu=apple-m1 cargo build\n",
                          ".cargo/config.toml": '[build]\nrustflags = ["-C", "target-cpu=x86-64-v3", "-C", "llvm-args=-x86-asm-syntax=intel"]\n'})
        self.assertEqual(rc, 0, out)
        red(self, {"Cargo.toml": PKG, "scripts/a.sh": 'RUSTFLAGS="-C target-cpu=native"\n'}, "target-cpu-native:")

    def test_decoded_toml_values(self):
        files = {"Cargo.toml": PKG, ".cargo/config.toml": '[build]\nrustflags = ["-C", "\\u0074arget-cpu=native"]\n'}
        red(self, files, ".cargo/config.toml:0: target-cpu-native: target-cpu=native (decoded value)")
        files = {"Cargo.toml": PKG, "Makefile.toml": '[env]\nRUSTFLAGS = "-C target\\u002dcpu=native"\n'}
        red(self, files, "Makefile.toml:0: target-cpu-native:")

    def test_decoded_json_values(self):
        files = {"Cargo.toml": PKG, ".devcontainer/devcontainer.json":
                 '{ "containerEnv": { "RUSTFLAGS": "-C \\u0074arget-cpu=native" } }\n'}
        red(self, files, ".devcontainer/devcontainer.json:0: target-cpu-native:")

    def test_rust_cfg_predicates_are_not_flags(self):
        rc, out, _ = run({"Cargo.toml": PKG, "src/lib.rs":
                          '#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]\n'
                          'pub fn f() {}\n'
                          'pub const A: bool = cfg!(target_feature = "avx2");\n'
                          '#[cfg_attr(not(target_feature = "fma"), allow(dead_code))]\nfn g() {}\n'})
        self.assertEqual(rc, 0, out)
        red(self, {"Cargo.toml": PKG, "src/lib.rs": 'pub const F: &str = "-Ctarget_feature=-sse2";\n'}, "x87-float:")


class FlagsFromCommands(unittest.TestCase):
    def test_values_from_a_command_or_a_file(self):
        cases = [
            ("scripts/b.sh", 'RUSTFLAGS="$(cat ci/flags.txt)" cargo build\n'),
            ("scripts/b.sh", "RUSTFLAGS=$(cat ci/flags.txt) cargo build\n"),
            ("scripts/b.sh", 'export RUSTFLAGS="$(<ci/flags.txt)"\n'),
            ("scripts/b.sh", "RUSTFLAGS=`cat ci/flags.txt` cargo build\n"),
            (".github/workflows/ci.yml", 'jobs:\n  b:\n    steps:\n      - run: echo "RUSTFLAGS=$(cat f)" >> "$GITHUB_ENV"\n'),
            ("tools/b.py", 'import os\nos.environ["RUSTFLAGS"] = open("ci/flags.txt").read()\n'),
            ("tools/c.py", 'os.environ["RUSTFLAGS"] = Path("f").read_text()\n'),
            ("xtask/src/main.rs", 'fn main() { cmd.env("RUSTFLAGS", std::fs::read_to_string("f").unwrap()); }\n'),
            ("scripts/b.sh", "CARGO_ENCODED_RUSTFLAGS=$(tr ' ' '\\037' < f) cargo build\n"),
        ]
        for rel, text in cases:
            with self.subTest(text=text):
                red(self, {"Cargo.toml": PKG, rel: text}, "rustflags-unresolved:")

    def test_other_substitutions_on_the_line_are_not_the_value(self):
        rc, out, _ = run({"Cargo.toml": PKG, "scripts/a.sh":
                          'RUSTFLAGS="-Dwarnings" WRAPPER="$(rustup which clippy-driver)" cargo build\n'
                          'RUSTFLAGS="$RUSTFLAGS -Dwarnings" cargo build\n'})
        self.assertEqual(rc, 0, out)


class ExcludeRefusal(unittest.TestCase):
    NATIVE_LINE = 'RUSTFLAGS="-C target-cpu=native" cargo build\n'

    def test_build_read_files_cannot_be_excluded(self):
        cases = {
            ".cargo/config.toml": '[build]\nrustflags = ["-C", "target-cpu=native"]\n',
            "crates/a/.cargo/extra.toml": '[build]\nrustflags = ["-C", "target-cpu=native"]\n',
            "crates/a/Cargo.toml": PKG + '[profile.release]\nrustflags = ["-Ctarget-cpu=native"]\n',
            "pyproject.toml": '[tool.maturin]\nrustc-args = ["-Ctarget-cpu=native"]\n',
            "build.rs": 'fn main() { println!("cargo:rustc-flags=-C target-cpu=native"); }\n',
            ".github/workflows/ci.yml": 'env:\n  RUSTFLAGS: -C target-cpu=native\n',
            "tools/x/action.yml": 'runs:\n  env:\n    RUSTFLAGS: -C target-cpu=native\n',
            "ci/flags.sh": self.NATIVE_LINE,
        }
        for rel, text in cases.items():
            with self.subTest(rel=rel):
                files = {"Cargo.toml": PKG, rel: text,
                         "scripts/build_config_allowlist.toml": f'[[exclude]]\npath = "{rel}"\nreason = "x"\n'}
                if rel == "ci/flags.sh":
                    files[".github/workflows/b.yml"] = "jobs:\n  b:\n    steps:\n      - run: ci/flags.sh\n"
                rc, out, _ = run(files)
                self.assertEqual(rc, 1, out)
                self.assertIn("is refused", out)

    def test_wrapper_cannot_be_excluded(self):
        files = {"Cargo.toml": PKG, ".cargo/config.toml": '[build]\nrustc-wrapper = "tools/w"\n',
                 "tools/w": '#!/bin/sh\nexec "$@" -C target-cpu=native\n',
                 "scripts/build_config_allowlist.toml": '[[exclude]]\npath = "tools/w"\nreason = "x"\n'}
        rc, out, _ = run(files)
        self.assertEqual(rc, 1, out)
        self.assertIn("tools/w:2: target-cpu-native:", out)

    def test_documents_data_and_the_guard_can_be_excluded(self):
        files = {"Cargo.toml": PKG,
                 "alice-law-conformance.toml": 'note = "a mutant sets -C target-cpu=native"\n',
                 "scripts/build_config_guard.py": 'R = "-C target-cpu=native"\n',
                 "scripts/test_build_config_guard.py": 'F = "-Ofast"\n',
                 ".github/workflows/ci.yml": "jobs:\n  b:\n    steps:\n      - run: python3 scripts/build_config_guard.py\n",
                 "scripts/build_config_allowlist.toml": "".join(
                     f'[[exclude]]\npath = "{rel}"\nreason = "quotes flags as data"\n'
                     for rel in ("alice-law-conformance.toml", "scripts/build_config_guard.py",
                                 "scripts/test_build_config_guard.py"))}
        rc, out, _ = run(files)
        self.assertEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main(verbosity=1)
