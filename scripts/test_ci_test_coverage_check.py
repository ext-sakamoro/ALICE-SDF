#!/usr/bin/env python3
"""scripts/ci_test_coverage_check.py の oracle.

検査器が「走っていない feature 付き test を走っていると読む」形の空振りを固定する.
期待値は fixture の構造 (どの file / cfg をどの step が走らせるか) から決まり、検査器を呼んで作らない.

run: python3 scripts/test_ci_test_coverage_check.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ci_test_coverage_check', HERE / 'ci_test_coverage_check.py')
cov = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cov  # dataclass が sys.modules から module を引く
spec.loader.exec_module(cov)

CARGO = '''\
[package]
name = "demo"
version = "0.1.0"

[features]
default = ["cli"]
cli = []
gpu = ["dep:wgpu"]
glsl = []
physics = ["dep:alice-physics"]
volume = []
gi = []
aaa = ["volume", "gi"]
'''

PLAIN = '#[test]\nfn a() {}\n'
GATED_GPU = '#![cfg(feature = "gpu")]\n#[test]\nfn a() {}\n'
GATED_ALL = '#![cfg(all(feature = "glsl", feature = "gpu"))]\n#[test]\nfn a() {}\n'


def workflow(steps: str, on: str = 'on:\n  push:\n    branches: [main]\n', extra_job: str = '') -> str:
    return on + '\njobs:\n  test:\n    runs-on: ubuntu-latest\n' + extra_job + '    steps:\n' + textwrap.indent(
        textwrap.dedent(steps), ' ' * 6)


class Repo:
    def __init__(self, tests: dict[str, str], wf: str | dict[str, str]):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / 'Cargo.toml').write_text(CARGO, encoding='utf-8')
        (root / 'tests').mkdir()
        for name, body in tests.items():
            (root / 'tests' / f'{name}.rs').write_text(body, encoding='utf-8')
        wfd = root / '.github' / 'workflows'
        wfd.mkdir(parents=True)
        for name, body in ({'ci.yml': wf} if isinstance(wf, str) else wf).items():
            (wfd / name).write_text(body, encoding='utf-8')
        self.root = root

    def check(self):
        return cov.check(self.root)

    def close(self):
        self.tmp.cleanup()


class CoverageTest(unittest.TestCase):
    def run_case(self, tests, wf):
        r = Repo(tests, wf)
        self.addCleanup(r.close)
        return r.check()

    def kinds(self, violations):
        return sorted(v.split(':', 1)[0] for v in violations)

    # ---- file 単位

    def test_plain_file_run_by_tests_is_covered(self):
        v, stats = self.run_case({'t': PLAIN}, workflow('- run: cargo test --tests\n'))
        self.assertEqual(v, [])
        self.assertEqual(stats['files'], 1)
        self.assertEqual(stats['invocations'], 1)

    def test_gated_file_under_default_features_is_unrun(self):
        # 実測した形: `cargo test --tests` は default feature なので gpu 付き file は 0 本になる
        v, _ = self.run_case({'g': GATED_GPU}, workflow('- run: cargo test --tests\n'))
        self.assertEqual(self.kinds(v), ['unrun'])
        self.assertIn('tests/g.rs', v[0])

    def test_gated_file_with_feature_and_test_flag_is_covered(self):
        v, _ = self.run_case({'g': GATED_GPU}, workflow('- run: cargo test --features gpu --test g\n'))
        self.assertEqual(v, [])

    def test_all_requires_every_feature(self):
        v, _ = self.run_case({'g': GATED_ALL}, workflow('- run: cargo test --features gpu --test g\n'))
        self.assertEqual(self.kinds(v), ['unrun'])
        v, _ = self.run_case({'g': GATED_ALL}, workflow('- run: cargo test --features "glsl,gpu" --test g\n'))
        self.assertEqual(v, [])

    def test_feature_flag_spellings(self):
        for flag in ('--features=glsl,gpu', '-F "glsl gpu"', '--features glsl --features gpu', '--all-features'):
            with self.subTest(flag=flag):
                v, _ = self.run_case({'g': GATED_ALL}, workflow(f'- run: cargo test {flag} --test g\n'))
                self.assertEqual(v, [])

    def test_meta_feature_is_expanded(self):
        body = '#![cfg(all(feature = "volume", feature = "gi"))]\n#[test]\nfn a() {}\n'
        v, _ = self.run_case({'g': body}, workflow('- run: cargo test --features aaa --test g\n'))
        self.assertEqual(v, [])

    def test_other_test_target_does_not_cover(self):
        v, _ = self.run_case({'g': GATED_GPU, 'h': PLAIN},
                             workflow('- run: cargo test --features gpu --test h\n'))
        self.assertEqual(self.kinds(v), ['unrun'])
        self.assertIn('tests/g.rs', v[0])

    def test_lib_and_doc_only_do_not_cover(self):
        for cmd in ('cargo test --lib --features gpu', 'cargo test --doc --features gpu',
                    'cargo test --bins --features gpu'):
            with self.subTest(cmd=cmd):
                v, _ = self.run_case({'g': GATED_GPU}, workflow(f'- run: {cmd}\n'))
                self.assertEqual(self.kinds(v), ['unrun'])

    def test_no_target_flag_runs_every_test_file(self):
        v, _ = self.run_case({'g': GATED_GPU}, workflow('- run: cargo test --features gpu\n'))
        self.assertEqual(v, [])

    def test_harness_args_after_double_dash_are_ignored(self):
        v, _ = self.run_case({'g': GATED_GPU},
                             workflow('- run: cargo test --test g -- --features gpu --nocapture\n'))
        self.assertEqual(self.kinds(v), ['unrun'])

    def test_no_default_features_drops_default(self):
        body = '#![cfg(feature = "cli")]\n#[test]\nfn a() {}\n'
        v, _ = self.run_case({'g': body}, workflow('- run: cargo test --no-default-features --tests\n'))
        self.assertEqual(self.kinds(v), ['unrun'])
        v, _ = self.run_case({'g': body}, workflow('- run: cargo test --tests\n'))
        self.assertEqual(v, [])

    # ---- 別 package / 別 directory

    def test_step_working_directory_does_not_cover(self):
        # 実物: mobile/uniffi-wrapper で `cargo test --tests` が走る step がある
        steps = ('- run: cargo test --tests\n'
                 '- working-directory: sub\n  run: cargo test --tests --features gpu\n')
        v, _ = self.run_case({'g': GATED_GPU}, workflow(steps))
        self.assertEqual(self.kinds(v), ['unrun'])

    def test_job_default_working_directory_does_not_cover(self):
        extra = '    defaults:\n      run:\n        working-directory: sub\n'
        wf = {'ci.yml': workflow('- run: cargo test --tests\n'),
              'sub.yml': workflow('- run: cargo test --features gpu --test g\n', extra_job=extra)}
        v, stats = self.run_case({'g': GATED_GPU}, wf)
        self.assertEqual(self.kinds(v), ['unrun'])
        self.assertEqual(stats['invocations'], 1)

    def test_other_package_does_not_cover(self):
        for cmd in ('cargo test -p other --features gpu --test g',
                    'cargo test --manifest-path sub/Cargo.toml --features gpu --test g'):
            with self.subTest(cmd=cmd):
                v, _ = self.run_case({'g': GATED_GPU}, workflow(f'- run: {cmd}\n'))
                self.assertEqual(self.kinds(v), ['unrun'])
        v, _ = self.run_case({'g': GATED_GPU}, workflow('- run: cargo test -p demo --features gpu --test g\n'))
        self.assertEqual(v, [])

    # ---- run の書き方

    def test_folded_block_scalar(self):
        # 実物: gpu-parity job の `run: >` で --test を複数行に分けて書いている
        steps = '- name: gpu\n  run: >\n    cargo test --features "gpu"\n    --test g\n    -- --nocapture\n'
        v, _ = self.run_case({'g': GATED_GPU}, workflow(steps))
        self.assertEqual(v, [])

    def test_literal_block_with_continuation_and_chain(self):
        steps = ('- run: |\n    set -e\n    cargo build\n'
                 '    cargo test \\\n      --features gpu \\\n      --test g\n')
        v, _ = self.run_case({'g': GATED_GPU}, workflow(steps))
        self.assertEqual(v, [])
        v, _ = self.run_case({'g': GATED_GPU}, workflow('- run: cargo build && cargo test --features gpu --test g\n'))
        self.assertEqual(v, [])

    def test_workflow_not_on_push_or_pr_is_ignored(self):
        wf = workflow('- run: cargo test --features gpu --test g\n', on='on:\n  workflow_dispatch:\n')
        v, stats = self.run_case({'g': GATED_GPU}, {'ci.yml': workflow('- run: cargo test --tests\n'), 'manual.yml': wf})
        self.assertEqual(self.kinds(v), ['unrun'])
        self.assertEqual(stats['invocations'], 1)

    # ---- 関数単位の cfg

    def test_item_cfg_needs_its_own_feature(self):
        body = PLAIN + '#[cfg(feature = "gpu")]\n#[test]\nfn b() {}\n'
        v, _ = self.run_case({'t': body}, workflow('- run: cargo test --tests\n'))
        self.assertEqual(self.kinds(v), ['unrun_item'])
        self.assertIn('tests/t.rs:3', v[0])
        v, _ = self.run_case({'t': body}, workflow('- run: cargo test --tests\n- run: cargo test --features gpu --test t\n'))
        self.assertEqual(v, [])

    def test_item_cfg_not_feature_is_covered_by_default(self):
        body = PLAIN + '#[cfg(not(feature = "gpu"))]\n#[test]\nfn b() {}\n'
        v, _ = self.run_case({'t': body}, workflow('- run: cargo test --tests\n'))
        self.assertEqual(v, [])

    def test_item_cfg_without_feature_is_not_checked(self):
        body = PLAIN + '#[cfg(target_os = "macos")]\n#[test]\nfn b() {}\n#[cfg(test)]\nmod m {}\n'
        v, stats = self.run_case({'t': body}, workflow('- run: cargo test --tests\n'))
        self.assertEqual(v, [])
        self.assertEqual(stats['cfgs'], 0)

    def test_item_cfg_combined_with_file_cfg(self):
        body = GATED_GPU + '#[cfg(feature = "glsl")]\n#[test]\nfn b() {}\n'
        wf = workflow('- run: cargo test --features gpu --test g\n- run: cargo test --features glsl --test g\n')
        v, _ = self.run_case({'g': body}, wf)
        self.assertEqual(self.kinds(v), ['unrun_item'])

    def test_multiline_and_commented_cfg(self):
        body = '#![cfg(all(\n    feature = "glsl",\n    feature = "gpu"\n))]\n// #[cfg(feature = "physics")]\n' + PLAIN
        v, _ = self.run_case({'g': body}, workflow('- run: cargo test --features gpu --test g\n'))
        self.assertEqual(self.kinds(v), ['unrun'])
        v, stats = self.run_case({'g': body}, workflow('- run: cargo test --features glsl,gpu --test g\n'))
        self.assertEqual(v, [])
        self.assertEqual(stats['cfgs'], 1)

    def test_platform_predicate_is_assumed_satisfiable(self):
        body = '#![cfg(all(feature = "gpu", target_os = "macos"))]\n' + PLAIN
        v, _ = self.run_case({'g': body}, workflow('- run: cargo test --features gpu --test g\n'))
        self.assertEqual(v, [])

    # ---- 空振り

    def test_no_test_files_fails(self):
        v, _ = self.run_case({}, workflow('- run: cargo test --tests\n'))
        self.assertEqual(self.kinds(v), ['empty_scan'])

    def test_no_cargo_test_fails(self):
        v, _ = self.run_case({'t': PLAIN}, workflow('- run: cargo build\n'))
        self.assertEqual(self.kinds(v), ['empty_scan'])

    # ---- cfg 式

    def test_eval_cfg(self):
        f = {'a'}
        self.assertIs(cov.eval_cfg(cov.parse_cfg('feature = "a"'), f), True)
        self.assertIs(cov.eval_cfg(cov.parse_cfg('feature = "b"'), f), False)
        self.assertIs(cov.eval_cfg(cov.parse_cfg('any(feature = "b", feature = "a")'), f), True)
        self.assertIs(cov.eval_cfg(cov.parse_cfg('not(feature = "b")'), f), True)
        self.assertIs(cov.eval_cfg(cov.parse_cfg('all(feature = "a", unix)'), f), None)
        self.assertIs(cov.eval_cfg(cov.parse_cfg('all(feature = "b", unix)'), f), False)
        with self.assertRaises(ValueError):
            cov.parse_cfg('feature = "a" garbage')


class RealRepoTest(unittest.TestCase):
    """実 repo を走査して、0 件で空振りしていないことだけを確かめる (違反の有無は本体の実行が判定する)."""

    def test_real_repo_scans_something(self):
        _, stats = cov.check(HERE.parent)
        self.assertGreater(stats['files'], 0)
        self.assertGreater(stats['invocations'], 0)
        self.assertGreater(stats['cfgs'], 0)


if __name__ == '__main__':
    unittest.main()
