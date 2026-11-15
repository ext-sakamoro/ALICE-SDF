#!/usr/bin/env python3
"""Tests for scripts/gen_eval_rec.py: the generated `eval_rec` is what `combine` and `descend` give,
and every variant of `SdfNode` is covered exactly once

The repository's files pass. Copies with a change fail: a hand edit of the generated block, a law
arm of `combine` the block does not have, a variant of the enum no group covers, a kind in two
groups, and an enum with no variant (nothing compared).

Author: Moroya Sakamoto
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_eval_rec as g  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class GenEvalRec(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "src" / "eval").mkdir(parents=True)
        (d / "src" / "types").mkdir(parents=True)
        self.src = d / "src" / "eval" / "mod.rs"
        self.types = d / "src" / "types" / "mod.rs"
        shutil.copy(ROOT / "src" / "eval" / "mod.rs", self.src)
        shutil.copy(ROOT / "src" / "types" / "mod.rs", self.types)
        for f in ("rustfmt.toml", ".rustfmt.toml"):
            if (ROOT / f).exists():
                shutil.copy(ROOT / f, d / f)
        self.saved = (g.SRC, g.TYPES)
        g.SRC, g.TYPES = self.src, self.types

    def tearDown(self) -> None:
        g.SRC, g.TYPES = self.saved
        self.tmp.cleanup()

    def check(self) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            return g.main(["gen_eval_rec.py", "--check"])

    def edit(self, path: Path, old: str, new: str) -> None:
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def test_the_repository_passes(self) -> None:
        self.assertEqual(self.check(), 0)
        compared, problems = g.coverage(self.src.read_text(encoding="utf-8"))
        self.assertGreater(compared, 100)
        self.assertEqual(problems, [])

    def test_a_hand_edit_of_the_block_fails(self) -> None:
        self.edit(self.src, "sdf_union(\n", "sdf_intersection(\n")
        self.assertEqual(self.check(), 1)

    def test_a_law_arm_the_block_lacks_fails(self) -> None:
        # a new law arm in combine (here: Union's law also under a renamed copy) changes the block
        self.edit(
            self.src,
            "        SdfNode::Union { .. } => sdf_union(d1, d2),",
            "        SdfNode::Union { .. } => sdf_union(d1, d2),\n        SdfNode::Xunion { .. } => sdf_union(d1, d2),",
        )
        self.assertEqual(self.check(), 1)

    def test_a_variant_no_group_covers_fails(self) -> None:
        self.edit(self.types, "pub enum SdfNode {\n", "pub enum SdfNode {\n    Uncovered,\n")
        self.assertEqual(self.check(), 1)
        _, problems = g.coverage(self.src.read_text(encoding="utf-8"))
        self.assertTrue(any("Uncovered" in p for p in problems), problems)

    def test_a_kind_in_two_groups_fails(self) -> None:
        # Translate also listed as a leaf
        self.edit(self.src, "fn eval_leaf(node: &SdfNode, point: Vec3) -> f32 {\n    match node {\n",
                  "fn eval_leaf(node: &SdfNode, point: Vec3) -> f32 {\n    match node {\n        SdfNode::Translate { .. } => 0.0,\n")
        _, problems = g.coverage(self.src.read_text(encoding="utf-8"))
        self.assertTrue(any("Translate" in p for p in problems), problems)
        self.assertEqual(self.check(), 1)

    def test_an_enum_with_no_variant_fails(self) -> None:
        text = self.types.read_text(encoding="utf-8")
        start = text.index("pub enum SdfNode {")
        end = text.index("\n}\n", start)
        self.types.write_text(text[:start] + "pub enum SdfNode {" + text[end:], encoding="utf-8")
        self.assertEqual(self.check(), 1)


if __name__ == "__main__":
    unittest.main()
