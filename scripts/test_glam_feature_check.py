#!/usr/bin/env python3
"""Tests of scripts/glam_feature_check.py on hand-written cargo metadata."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glam_feature_check as g  # noqa: E402


def meta(*glams: tuple[str, list[str]], other: bool = True) -> dict:
    packages, nodes = [], []
    for version, feats in glams:
        pid = f"registry+https://github.com/rust-lang/crates.io-index#glam@{version}"
        packages.append({"id": pid, "name": "glam"})
        nodes.append({"id": pid, "features": feats})
    if other:
        packages.append({"id": "path+file:///x#alice-sdf@7.0.0", "name": "alice-sdf"})
        nodes.append({"id": "path+file:///x#alice-sdf@7.0.0", "features": ["fast-math"]})
    return {"packages": packages, "resolve": {"nodes": nodes}}


def run(m: dict) -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        p.write_text(json.dumps(m), encoding="utf-8")
        return g.main(["--metadata", str(p)])


class GlamFeatureCheck(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(run(meta(("0.29.3", ["default", "serde", "std"]))), 0)

    def test_fast_math_fails(self):
        self.assertEqual(run(meta(("0.29.3", ["default", "fast-math", "std"]))), 1)

    def test_one_of_several_versions(self):
        # a transitive older glam with the feature is the same defect
        self.assertEqual(run(meta(("0.29.3", ["std"]), ("0.24.2", ["fast-math", "std"]))), 1)
        problems, n = g.check(meta(("0.29.3", ["std"]), ("0.24.2", ["fast-math"])))
        self.assertEqual(n, 2)
        self.assertEqual(len(problems), 1)
        self.assertIn("glam@0.24.2", problems[0])

    def test_feature_on_another_crate_does_not_count(self):
        # only glam's own feature set is read
        self.assertEqual(run(meta(("0.29.3", ["std"]), other=True)), 0)

    def test_no_glam_is_compared_nothing(self):
        self.assertEqual(run(meta(other=True)), 2)


if __name__ == "__main__":
    unittest.main()
