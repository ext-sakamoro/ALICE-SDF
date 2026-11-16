#!/usr/bin/env python3
"""Tests of scripts/gpu_slot_guard_check.py `builds`: the cargo JSON of a test build is counted the way cargo writes it."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "gpu_slot_guard_check.py"


def artifact(name, kind, features, test=False, executable=None):
    return json.dumps({
        "reason": "compiler-artifact",
        "package_id": f"path+file:///x#{name}@7.0.0",
        "target": {"name": name, "kind": kind},
        "features": features,
        "profile": {"test": test},
        "executable": executable,
    })


def run(lines):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("\n".join(lines) + "\n")
    r = subprocess.run([sys.executable, str(SCRIPT), "builds", f.name], capture_output=True, text=True)
    return r.returncode, r.stdout


class Builds(unittest.TestCase):
    def test_an_rlib_and_cdylib_library_with_the_feature_is_counted(self):
        code, out = run([
            artifact("alice_sdf", ["cdylib", "rlib"], ["gpu", "gpu-slot-guard"]),
            artifact("t", ["test"], [], test=True, executable="/x/t"),
        ])
        self.assertEqual(code, 0, out)
        self.assertIn("1 alice-sdf library artifacts", out)

    def test_a_library_without_the_feature_fails(self):
        code, out = run([
            artifact("alice_sdf", ["cdylib", "rlib"], ["gpu"]),
            artifact("t", ["test"], [], test=True, executable="/x/t"),
        ])
        self.assertEqual(code, 1, out)

    def test_no_library_artifact_fails(self):
        code, out = run([artifact("t", ["test"], [], test=True, executable="/x/t")])
        self.assertEqual(code, 1, out)
        self.assertIn("nothing was checked", out)


if __name__ == "__main__":
    unittest.main()
