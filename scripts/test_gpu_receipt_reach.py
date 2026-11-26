#!/usr/bin/env python3
"""scripts/gpu_receipt_reach.py の oracle

The expected sets come from the fixture's paths (which file names which module), not from the checker.

run: python3 scripts/test_gpu_receipt_reach.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gpu_receipt_reach', HERE / 'gpu_receipt_reach.py')
gr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gr
spec.loader.exec_module(gr)


class ReachTest(unittest.TestCase):
    """gpu_receipt_reach: every crate file reached from the files under the rule's paths must be under a path.

    The expected sets come from the fixture's paths (which file names which module), not from the checker."""

    BASE = {
        'src/lib.rs': 'pub mod field;\npub mod law;\npub mod types;\npub mod io;\npub use types::Aabb;\n',
        'src/field/mod.rs': 'pub mod gpu;\nuse crate::law::LawMinMax;\n',
        'src/field/gpu.rs': 'use super::*;\nconst S: &str = include_str!("march.wgsl");\n'
                            'fn f() -> crate::Aabb { todo!() }\n',
        'src/field/march.wgsl': '// shader\n',
        'src/law.rs': 'pub struct LawMinMax;\n',
        'src/types/mod.rs': 'mod containers;\npub use containers::Aabb;\n',
        'src/types/containers.rs': 'pub struct Aabb;\n',
        'src/io/mod.rs': 'pub struct IoError;\n// crate::io::never is a comment, not a use\n',
        'tests/test_field_a.rs': 'use demo::{field::gpu, io::IoError};\n#[test]\nfn a() {}\n',
    }
    FULL = ('path src/field/**\npath tests/test_field_*\npath src/law.rs\npath src/types/mod.rs\n'
            'path src/types/containers.rs\npath src/io/mod.rs\n')

    def reach(self, rule, files=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, body in (files or self.BASE).items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(body, encoding='utf-8')
        (root / 'scripts').mkdir(exist_ok=True)
        (root / gr.GPU_RECEIPT_RULE).write_text(rule, encoding='utf-8')
        return gr.gpu_receipt_reach(root, 'demo')

    def missing(self, problems):
        return sorted({p.split()[1] for p in problems})

    def test_a_rule_listing_every_reached_file_holds(self):
        problems, walked = self.reach(self.FULL)
        self.assertEqual(problems, [])
        self.assertEqual(walked, 7)                     # every .rs under a hand-kept path is walked

    def test_dropping_any_one_path_line_is_a_finding_for_that_file(self):
        # the mutant 07 asked for: each path line, dropped on its own, turns red on the file it covered
        for line, want in (('path src/law.rs', ['src/law.rs']),                       # use crate::law::…
                           ('path src/types/mod.rs', ['src/types/mod.rs']),           # crate::Aabb via lib.rs
                           ('path src/types/containers.rs', ['src/types/containers.rs']),  # re-export target
                           ('path src/io/mod.rs', ['src/io/mod.rs'])):                # from the test's use tree
            with self.subTest(line=line):
                problems, _ = self.reach(self.FULL.replace(line + '\n', ''))
                self.assertEqual(self.missing(problems), want)

    def test_an_included_shader_outside_the_rule_is_a_finding(self):
        files = dict(self.BASE)
        files['src/field/gpu.rs'] = 'const S: &str = include_str!("../shaders/march.wgsl");\n'
        files['src/shaders/march.wgsl'] = '// shader\n'
        problems, _ = self.reach(self.FULL, files)
        self.assertEqual(self.missing(problems), ['src/shaders/march.wgsl'])

    def test_the_walk_is_transitive(self):
        # law.rs is reached (not hand-kept) and itself reaches deep.rs: the walk must go on from a reached file
        files = dict(self.BASE)
        files['src/law.rs'] = 'pub struct LawMinMax;\nuse crate::deep::X;\n'
        files['src/deep.rs'] = 'pub struct X;\n'
        rule = ('path src/field/**\npath tests/test_field_*\n' + gr.REACH_BEGIN + '\n'
                + ''.join(f'path {b}\n' for b in ('src/io/mod.rs', 'src/law.rs', 'src/types/containers.rs',
                                                   'src/types/mod.rs')) + gr.REACH_END + '\n')
        problems, walked = self.reach(rule, files)
        self.assertEqual(walked, 3)
        self.assertEqual(self.missing(problems), ['src/deep.rs'])

    def test_lib_rs_is_never_required_and_paths_in_comments_and_strings_are_not_uses(self):
        files = dict(self.BASE)
        files['src/field/mod.rs'] = ('pub mod gpu;\nuse crate::law::LawMinMax;\n'
                                     '// crate::secret::Y\nconst M: &str = "crate::secret::Y";\n')
        files['src/secret.rs'] = 'pub struct Y;\n'
        problems, _ = self.reach(self.FULL, files)
        self.assertEqual(problems, [])

    def test_the_generated_block_must_equal_the_walk_both_ways(self):
        seeds = 'path src/field/**\npath tests/test_field_*\n'
        block = ['src/io/mod.rs', 'src/law.rs', 'src/types/containers.rs', 'src/types/mod.rs']
        rule = seeds + gr.REACH_BEGIN + '\n' + ''.join(f'path {b}\n' for b in block) + gr.REACH_END + '\nsuite x 1\n'
        problems, walked = self.reach(rule)
        self.assertEqual(problems, [])
        self.assertEqual(walked, 3)                     # field/mod.rs, field/gpu.rs, the test
        for drop in block:                              # a reached file missing from the block
            with self.subTest(drop=drop):
                problems, _ = self.reach(rule.replace(f'path {drop}\n', ''))
                self.assertEqual(self.missing(problems), [drop])
                self.assertIn('would land without the GPU receipt', problems[0])
        problems, _ = self.reach(rule.replace(gr.REACH_END, 'path src/old.rs\n' + gr.REACH_END))
        self.assertEqual(self.missing(problems), ['src/old.rs'])     # listed, no longer reached
        self.assertIn('stale', problems[0])

    def test_a_marker_that_does_not_match_or_a_block_left_open_refuses(self):
        seeds = 'path src/field/**\npath tests/test_field_*\n'
        good = seeds + gr.REACH_BEGIN + '\npath src/law.rs\n' + gr.REACH_END + '\n'
        for rule in (good.replace('gpu_receipt_reach.py', 'other.py'),          # the begin marker misspelt
                     good.replace(gr.REACH_END, '# end  reach'),                 # the end marker misspelt
                     good.replace(gr.REACH_END + '\n', ''),                      # left open
                     good + gr.REACH_BEGIN + '\n' + gr.REACH_END + '\n',          # a second block
                     seeds + gr.REACH_END + '\n',                                # an end without a begin
                     seeds + '# reach: hand list\npath src/law.rs\n'):           # a lone look-alike marker
            with self.subTest(rule=rule[-60:]):
                problems, walked = self.reach(rule)
                self.assertEqual(walked, 0)
                self.assertEqual(len(problems), 1)
                self.assertIn('reach', problems[0])

    def test_write_reach_produces_the_block_the_check_accepts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, body in self.BASE.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(body, encoding='utf-8')
        (root / 'scripts').mkdir()
        (root / gr.GPU_RECEIPT_RULE).write_text('# head\npath src/field/**\npath tests/test_field_*\nsuite x 1\n',
                                                 encoding='utf-8')
        self.assertEqual(gr.gpu_receipt_reach(root, 'demo')[0] != [], True)   # red before
        self.assertEqual(gr.write_reach(root, 'demo'), 4)
        self.assertEqual(gr.gpu_receipt_reach(root, 'demo'), ([], 3))
        text = (root / gr.GPU_RECEIPT_RULE).read_text(encoding='utf-8')
        self.assertLess(text.index(gr.REACH_END), text.index('suite x 1'))    # before the suites
        self.assertEqual(gr.write_reach(root, 'demo'), 4)                      # idempotent
        self.assertEqual((root / gr.GPU_RECEIPT_RULE).read_text(encoding='utf-8'), text)

    def test_nothing_compared_fails_and_no_rule_is_no_finding(self):
        problems, walked = self.reach('path docs/**\n')
        self.assertEqual(walked, 0)
        self.assertTrue(any('the walk read nothing' in p for p in problems))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.assertEqual(gr.gpu_receipt_reach(Path(tmp.name), 'demo'), ([], 0))


class RealRepo(unittest.TestCase):
    def test_the_repository_walks_something_and_holds(self):
        problems, walked = gr.gpu_receipt_reach(HERE.parent, gr.crate_name(HERE.parent))
        self.assertGreater(walked, 0)
        self.assertEqual(problems, [])


if __name__ == '__main__':
    unittest.main()
