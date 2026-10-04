#!/usr/bin/env python3
"""scripts/gen-wiring-status.py と scripts/gen-oracle-status.py の oracle.

台帳の生成器は「違反を 1 件も拾えないのに green」型の空振りが最も危ない
(2026-10-03 実測: wiring 側は wiring_guard の実出力の形を読めず、新しい違反が
あっても必ず「違反なし」と判定していた) ので、実際の出力の形を食わせて固定する.
期待値は fixture の構造 (何件書いたか) から決まり、生成器を呼んで作らない.

run: python3 scripts/test_gen_status.py
"""
from __future__ import annotations

import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wiring = load("gen_wiring_status", "gen-wiring-status.py")
oracle = load("gen_oracle_status", "gen-oracle-status.py")

# wiring_guard が実際に stderr に出す形 (scripts/wiring_guard.py の print と同じ書式)
GUARD_BAD = """\
unwired: src/a.rs::foo: foo: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/a.rs::bar: bar: 未配線の item の本体 (または自己再帰) からしか参照されていない
dead_code: src/b.rs: #[allow(dead_code)] に理由が無い
stale_baseline: src/c.rs::baz: baseline にあるが既に配線済 / 消えている 行を消す
unbalanced_braces: src/d.rs: 波括弧が閉じていない (本体の範囲を切り出せない)
wiring-guard: 5 violation(s)
"""
GUARD_OK = "wiring-guard: ok\n"


class WiringParse(unittest.TestCase):
    def test_each_violation_kind_is_counted_from_the_real_output_format(self):
        v = wiring.parse_violations(GUARD_BAD)
        self.assertEqual(len(v["unwired"]), 2)
        self.assertEqual(len(v["dead_code"]), 1)
        self.assertEqual(len(v["stale_baseline"]), 1)
        self.assertEqual(len(v["unbalanced_braces"]), 1)

    def test_the_summary_line_is_not_a_violation(self):
        v = wiring.parse_violations(GUARD_OK)
        self.assertEqual(sum(len(x) for x in v.values()), 0)
        v = wiring.parse_violations("wiring-guard: 7 violation(s)\n")
        self.assertEqual(sum(len(x) for x in v.values()), 0)

    def test_a_clean_run_reports_no_new_violations(self):
        base = {"dead_code": [], "unwired": []}
        report = wiring.generate_markdown_report(base, wiring.parse_violations(GUARD_OK))
        self.assertIn("All clear", report)
        self.assertNotIn("NEW", report)

    def test_a_new_violation_is_reported_as_new(self):
        base = {"dead_code": [], "unwired": []}
        report = wiring.generate_markdown_report(base, wiring.parse_violations(GUARD_BAD))
        self.assertIn("NEW violations detected", report)
        self.assertIn("src/a.rs::foo", report)
        self.assertIn("src/c.rs::baz", report)


    def test_each_kind_alone_is_enough_to_report_a_new_violation(self):
        base = {"dead_code": [], "unwired": []}
        for line, word in [
            ("dead_code: src/b.rs: x", "Dead Code Guard"),
            ("unwired: src/a.rs::f: x", "Unwired Items"),
            ("stale_baseline: src/c.rs::g: x", "Stale Baseline"),
            ("unbalanced_braces: src/d.rs: x", "Syntax Errors"),
        ]:
            r = wiring.generate_markdown_report(base, wiring.parse_violations(line + "\n"))
            self.assertIn("NEW violations detected", r, line)
            self.assertIn(word, r, line)


class WiringReport(unittest.TestCase):
    BASE = {
        "dead_code": ["dead_code src/x.rs 2"],
        "unwired": [
            "unwired src/x.rs::a",
            "unwired src/x.rs::b",
            "unwired src/y.rs::c",
        ],
    }

    def test_the_report_is_a_pure_function_of_its_inputs(self):
        # 時刻を含むと内容が同じでも毎回 commit される (実測: bot commit が push ごとに出る)
        a = wiring.generate_markdown_report(self.BASE, wiring.parse_violations(GUARD_OK))
        b = wiring.generate_markdown_report(self.BASE, wiring.parse_violations(GUARD_OK))
        self.assertEqual(a, b)
        self.assertNotIn("Last updated", a)

    def test_the_counts_match_the_baseline(self):
        r = wiring.generate_markdown_report(self.BASE, wiring.parse_violations(GUARD_OK))
        self.assertIn("4 baseline items", r)  # 1 + 3
        self.assertIn("Dead Code (1)", r)
        self.assertIn("Unwired Items (3)", r)

    def test_a_per_file_count_table_is_shown(self):
        r = wiring.generate_markdown_report(self.BASE, wiring.parse_violations(GUARD_OK))
        self.assertIn("| `src/x.rs` | 3 |", r)  # dead_code 1 行 (件数 2 ではなく 1 行) + unwired 2
        self.assertIn("| `src/y.rs` | 1 |", r)

    def test_the_exit_code_does_not_hide_the_report(self):
        # 違反があっても report は書く (違反の gate は CI の wiring_guard step の仕事)
        self.assertEqual(wiring.exit_code(wiring.parse_violations(GUARD_BAD)), 0)


def fixture_tests(body: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix="oracle-fixture-"))
    (root / "tests").mkdir()
    (root / "tests" / "t.rs").write_text(textwrap.dedent(body), encoding="utf-8")
    return root / "tests" / "t.rs"


class OracleParse(unittest.TestCase):
    def names(self, body: str):
        return {t["name"]: t for t in oracle.extract_test_metadata(fixture_tests(body))}

    def test_a_plain_test_is_counted_and_not_ignored(self):
        t = self.names("#[test]\nfn a() {}\n")
        self.assertFalse(t["a"]["is_ignored"])

    def test_a_one_line_ignore_keeps_its_reason(self):
        t = self.names('#[test]\n#[ignore = "runtime: 40 s"]\nfn a() {}\n')
        self.assertTrue(t["a"]["is_ignored"])
        self.assertEqual(t["a"]["ignore_reason"], "runtime: 40 s")

    def test_a_multi_line_ignore_reason_is_read_whole(self):
        # 実物: tests/analytic_large_rotation.rs の「the red is correct: ...」(`\` 行継続)
        t = self.names(
            '#[test]\n#[ignore = "the red is correct: a rigid rotation must carry \\\n'
            '            no stress, and the solver cannot deliver that yet. \\\n'
            '            Remove this attribute when it can"]\nfn a() {}\n'
        )
        self.assertTrue(t["a"]["is_ignored"])
        self.assertEqual(
            t["a"]["ignore_reason"],
            "the red is correct: a rigid rotation must carry no stress, and the "
            "solver cannot deliver that yet. Remove this attribute when it can",
        )

    def test_a_bare_ignore_is_pending(self):
        t = self.names("#[test]\n#[ignore]\nfn a() {}\n")
        self.assertTrue(t["a"]["is_ignored"])
        self.assertEqual(t["a"]["ignore_reason"], "pending")

    def test_ignore_before_test_is_seen(self):
        t = self.names('#[ignore = "slow"]\n#[test]\nfn a() {}\n')
        self.assertTrue(t["a"]["is_ignored"])
        self.assertEqual(t["a"]["ignore_reason"], "slow")

    def test_a_test_far_from_its_fn_is_not_dropped(self):
        # #[test] と fn の間が 10 行を超えても (長い should_panic / doc comment) 落とさない
        doc = "".join(f"/// line {i}\n" for i in range(15))
        t = self.names(f"#[test]\n{doc}#[should_panic(expected = \"x\")]\nfn far() {{}}\n")
        self.assertIn("far", t)

    def test_an_ignore_on_the_next_test_does_not_leak_back(self):
        t = self.names('#[test]\nfn a() {}\n\n#[test]\n#[ignore = "b only"]\nfn b() {}\n')
        self.assertFalse(t["a"]["is_ignored"])
        self.assertTrue(t["b"]["is_ignored"])

    def test_a_helper_fn_without_test_attr_is_not_a_test(self):
        t = self.names("fn helper() {}\n\n#[test]\nfn real() {}\n")
        self.assertEqual(set(t), {"real"})

    def test_an_ignore_does_not_leak_to_the_next_fn(self):
        # #[ignore] を持つ test の後ろの helper fn / 次の test に属性が持ち越されない
        t = self.names('#[test]\n#[ignore = "x"]\nfn a() {}\n\nfn helper() {}\n\n#[test]\nfn b() {}\n')
        self.assertEqual(set(t), {"a", "b"})
        self.assertTrue(t["a"]["is_ignored"])
        self.assertFalse(t["b"]["is_ignored"])

    def test_a_reason_keeps_inner_spacing_collapsed(self):
        t = self.names('#[test]\n#[ignore = "a   b\t c"]\nfn a() {}\n')
        self.assertEqual(t["a"]["ignore_reason"], "a b c")

    def test_a_cfg_test_mod_marker_in_a_comment_is_not_a_test(self):
        t = self.names("// #[test]\n// fn not_a_test() {}\n#[test]\nfn real() {}\n")
        self.assertEqual(set(t), {"real"})


class OracleClassify(unittest.TestCase):
    def test_a_red_by_design_test_is_not_called_pending(self):
        self.assertEqual(oracle.classify_ignored("the red is correct: ..."), "red")
        self.assertEqual(oracle.classify_ignored("src gap: measured 2026-10-02"), "red")
        self.assertEqual(oracle.classify_ignored("known defect: AUD-A-S1W1-001: sign flipped"), "red")

    def test_runtime_and_diagnostic_tests_are_gated(self):
        for reason in ["runtime: about 40 s in release", "diagnostic: the table behind x",
                       "manual: about 4 h in release", "25,600 tets at cell 0.25; run with --release"]:
            self.assertEqual(oracle.classify_ignored(reason), "gated", reason)

    def test_a_bare_ignore_is_pending(self):
        self.assertEqual(oracle.classify_ignored("pending"), "pending")

    def test_the_report_is_a_pure_function_and_has_honest_wording(self):
        cat = {
            "implemented": [{"test_name": "ok", "file": "a.rs", "ignore_reason": ""}],
            "partial": [],
            "pending": [
                {"test_name": "r", "file": "a.rs", "ignore_reason": "the red is correct: x"},
                {"test_name": "g", "file": "a.rs", "ignore_reason": "runtime: 5 s"},
                {"test_name": "p", "file": "a.rs", "ignore_reason": "pending"},
            ],
        }
        a = oracle.generate_markdown_report(cat)
        self.assertEqual(a, oracle.generate_markdown_report(cat))
        self.assertNotIn("Last updated", a)
        # 「passing」を名乗らない (数えているのは「#[ignore] でない」だけ)
        self.assertNotIn("passing", a)
        self.assertIn("Red by design (1)", a)
        self.assertIn("Gated (1)", a)
        self.assertIn("Pending (1)", a)
        self.assertIn("**Total** | **4**", a)


if __name__ == "__main__":
    unittest.main(verbosity=2)
