#!/usr/bin/env python3
"""Tests of scripts/unreal_engines.py: which engines one Unreal workflow run builds.

The rule (from the module doc): main and schedule build every engine; a ci/ push builds every
engine when the change is engine-specific and the newest otherwise; bad roots build nothing and
fail. Every expectation below is written from that rule, not from the script's output.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unreal_engines as ue  # noqa: E402

ROOTS = ["E:\\UE_5.7", "E:\\UE_5.8"]
CI = "refs/heads/ci/20261113-x"
MAIN = "refs/heads/main"


def run(*args: str, output: bool = True) -> tuple[int, str, str, str]:
    """(exit, stdout, stderr, the GITHUB_OUTPUT file's text)"""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out"
        out.write_text("", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_OUTPUT"}
        if output:
            env["GITHUB_OUTPUT"] = str(out)
        p = subprocess.run([sys.executable, str(HERE / "unreal_engines.py"), *args],
                           capture_output=True, text=True, env=env, check=False)
        return p.returncode, p.stdout, p.stderr, out.read_text(encoding="utf-8")


def engines(output_text: str) -> list[str]:
    lines = [l for l in output_text.splitlines() if l.startswith("engines=")]
    assert len(lines) == 1, output_text
    return json.loads(lines[0][len("engines="):])


def args(event: str, ref: str, specific: str, roots: list[str] | str = ROOTS) -> list[str]:
    r = roots if isinstance(roots, str) else json.dumps(roots)
    return ["--roots", r, "--event", event, "--ref", ref, "--specific", specific]


class Selection(unittest.TestCase):
    def test_main_builds_every_engine_even_when_not_specific(self):
        code, _, _, out = run(*args("push", MAIN, "false"))
        self.assertEqual(code, 0)
        self.assertEqual(engines(out), ROOTS)

    def test_schedule_builds_every_engine(self):
        code, _, _, out = run(*args("schedule", MAIN, "false"))
        self.assertEqual(code, 0)
        self.assertEqual(engines(out), ROOTS)

    def test_an_engine_specific_ci_change_builds_every_engine(self):
        code, _, _, out = run(*args("push", CI, "true"))
        self.assertEqual(code, 0)
        self.assertEqual(engines(out), ROOTS)

    def test_any_other_ci_change_builds_the_newest_engine_only(self):
        code, _, _, out = run(*args("push", CI, "false"))
        self.assertEqual(code, 0)
        self.assertEqual(engines(out), ["E:\\UE_5.8"])

    def test_newest_is_by_version_not_by_position(self):
        # the newest is neither the first nor the last entry
        roots = ["E:\\UE_5.7", "D:\\UE_5.10", "E:\\UE_5.8"]
        code, _, _, out = run(*args("push", CI, "false", roots))
        self.assertEqual(code, 0)
        self.assertEqual(engines(out), ["D:\\UE_5.10"])
        code, _, _, out = run(*args("push", CI, "false", ["E:\\UE_5.8", "E:\\UE_5.7"]))
        self.assertEqual(code, 0)
        self.assertEqual(engines(out), ["E:\\UE_5.8"])

    def test_a_root_without_a_version_is_older_than_one_with(self):
        self.assertEqual(ue.newest(["E:\\Engine", "E:\\UE_5.7"]), "E:\\UE_5.7")
        self.assertEqual(ue.newest(["E:\\UE_5.7", "E:\\Engine"]), "E:\\UE_5.7")
        self.assertEqual(ue.newest(["E:\\A", "E:\\B"]), "E:\\B")

    def test_one_root_is_built_in_every_case(self):
        for event, ref, spec in [("push", CI, "false"), ("push", CI, "true"),
                                 ("push", MAIN, "false"), ("schedule", MAIN, "false")]:
            code, _, _, out = run(*args(event, ref, spec, ["E:\\UE_5.7"]))
            self.assertEqual(code, 0)
            self.assertEqual(engines(out), ["E:\\UE_5.7"], (event, ref, spec))

    def test_without_github_output_the_line_goes_to_stdout(self):
        code, stdout, _, _ = run(*args("push", CI, "false"), output=False)
        self.assertEqual(code, 0)
        self.assertEqual(engines(stdout), ["E:\\UE_5.8"])


class Refusals(unittest.TestCase):
    def assert_refused(self, *a: str) -> None:
        code, _, err, out = run(*a)
        self.assertEqual(code, 1, err)
        # a refusal with its reason, not a crash (a traceback also exits 1)
        self.assertTrue(err.startswith("unreal_engines: "), err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("engines=", out)

    def test_empty_or_malformed_roots_build_nothing_and_fail(self):
        for roots in ["[]", "not json", '"E:\\\\UE_5.7"', "[1, 2]", '[""]', "{}"]:
            for event, ref in [("push", CI), ("push", MAIN), ("schedule", MAIN)]:
                with self.subTest(roots=roots, event=event, ref=ref):
                    self.assert_refused(*args(event, ref, "false", roots))

    def test_an_unexpected_event_fails(self):
        for event in ["pull_request", "workflow_dispatch", ""]:
            with self.subTest(event=event):
                self.assert_refused(*args(event, CI, "false"))

    def test_specific_must_be_a_boolean_word(self):
        for spec in ["", "yes", "1", "True"]:
            with self.subTest(spec=spec):
                self.assert_refused(*args("push", CI, spec))


class Log(unittest.TestCase):
    def test_the_log_names_versions_not_install_paths(self):
        code, stdout, _, _ = run(*args("push", MAIN, "false"))
        self.assertEqual(code, 0)
        self.assertIn("UE 5.7", stdout)
        self.assertIn("UE 5.8", stdout)
        self.assertNotIn("E:", stdout)


if __name__ == "__main__":
    unittest.main(verbosity=1)
