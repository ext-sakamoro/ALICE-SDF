#!/usr/bin/env python3
"""Tests for scripts/gpu_receipt_run.py: the rule it reads, the counts it reads from libtest and its verdict."""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpu_receipt_run as gr  # noqa: E402


class Run(unittest.TestCase):
    def test_passed_is_read_from_the_result_line(self):
        log = "running 5 tests\n...\ntest result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
        self.assertEqual(gr.passed(log), 5)
        self.assertEqual(gr.passed("test result: FAILED. 3 passed; 2 failed;\n"), 3)
        self.assertEqual(gr.passed("error: could not compile\n"), 0)
        # coloured libtest output (CARGO_TERM_COLOR=always): `ok` / `FAILED` wrapped in escape codes
        self.assertEqual(gr.passed("test result: \x1b[32mok\x1b[0m. 5 passed; 0 failed; 0 ignored\n"), 5)
        self.assertEqual(gr.passed("test result: \x1b[31mFAILED\x1b[0m. 3 passed; 2 failed;\n"), 3)

    def test_every_suite_at_its_min_and_rc_0_is_a_receipt(self):
        want = {"a": 2, "b": 1}
        self.assertEqual(gr.judge(want, {"a": {"passed": 2, "rc": 0}, "b": {"passed": 3, "rc": 0}}), [])

    def test_a_count_under_min_a_failure_or_a_missing_suite_is_not(self):
        want = {"a": 2, "b": 1}
        self.assertIn("a: 1 passed, under its min 2",
                      gr.judge(want, {"a": {"passed": 1, "rc": 0}, "b": {"passed": 1, "rc": 0}}))
        self.assertIn("b: rc 101", gr.judge(want, {"a": {"passed": 2, "rc": 0}, "b": {"passed": 1, "rc": 101}}))
        self.assertIn("b: not run", gr.judge(want, {"a": {"passed": 2, "rc": 0}}))

    def test_the_rule_needs_a_path_and_suites_with_positive_mins(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "scripts").mkdir()
            f = Path(d) / gr.RULE
            f.write_text("path src/**\nsuite a 2\n", encoding="utf-8")
            self.assertEqual(gr.suites(Path(d)), {"a": 2})
            for bad in ("suite a 2\n", "path src/**\n", "path src/**\nsuite a 0\n", "path src/**\nsuite a 1\nsuite a 2\n"):
                f.write_text(bad, encoding="utf-8")
                with self.assertRaises(SystemExit, msg=bad):
                    gr.suites(Path(d))

    def test_the_receipt_is_a_run_directory_with_the_log_and_its_hash(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as d:
            got = {"a": {"passed": 2, "rc": 0}}
            r = gr.write_receipt(Path(d), "ALICE-SDF", "a" * 40, "r1", "nb1", "0x10de\trow", got, "$ cargo test\nok\n")
            self.assertEqual(r, Path(d) / "receipts" / f"ALICE-SDF-{'a' * 40}-r1")
            res = json.loads((r / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(res["run_log_sha256"], hashlib.sha256((r / "run.log").read_bytes()).hexdigest())
            self.assertEqual((res["repo"], res["tree"], res["run_id"], res["notebook"], res["adapter"], res["suites"]),
                             ("ALICE-SDF", "a" * 40, "r1", "nb1", "0x10de\trow", got))
            with self.assertRaises(FileExistsError):    # a run id is written once
                gr.write_receipt(Path(d), "ALICE-SDF", "a" * 40, "r1", "nb1", "row", got, "x")

    def test_a_software_adapter_gets_no_receipt(self):
        real = "0x10de\t0x20b5\tVulkan\tNVIDIA A100 80GB PCIe\t580.196.07\t326.44\t2026-11-26\tcmd"
        self.assertIsNone(gr.software_adapter(real))
        self.assertIsNone(gr.software_adapter("0x106b\t0x0\tMetal\tApple M2 Pro\t0\t120.0"))
        for row in ("0x10005\t0x0\tVulkan\tllvmpipe (LLVM 17.0.6, 256 bits)\t0.0.1\t9000.0",
                    "0x10de\t0x0\tVulkan\tlavapipe\t1\t9000.0",        # by name, whatever the vendor
                    "0x1ae0\t0xc0de\tVulkan\tSwiftShader Device (Subzero)\t5.0.0\t9000.0",
                    "0x1414\t0x8c\tDx12\tMicrosoft Basic Render Driver\t10\t9000.0",
                    "0x10005\t0x0\tVulkan\tMesa device\t0.0.1\t9000.0"):    # by vendor, whatever the name
            self.assertIsNotNone(gr.software_adapter(row), row)
        self.assertIn("not a speed row", gr.software_adapter("no tabs here"))

    def test_no_adapter_names_the_machine_type_and_the_probe_error(self):
        err = "error: XDG_RUNTIME_DIR not set\nadapter speed: Failed to create GPU device: Parent device is lost\n"
        msg = gr.no_adapter("NVIDIA A100-SXM4-80GB", 1, err)
        self.assertTrue(msg.startswith("gpu-receipt: no hardware adapter on this machine (NVIDIA A100-SXM4-80GB)"))
        self.assertIn("Parent device is lost", msg)
        self.assertIn("rc 1", msg)

    def test_the_probe_failure_reaches_the_no_adapter_refusal(self):
        # adapter_row is the runner's first step: a probe with no row refuses there with the machine type (no suite runs)
        class P:
            returncode, stdout, stderr = 101, "", "adapter speed: Failed to create GPU device: Parent device is lost\n"
        real = gr.subprocess.run
        gr.subprocess.run = lambda cmd, **k: P() if cmd[0] == "cargo" else real(cmd, **k)
        try:
            with self.assertRaises(SystemExit) as e:
                gr.adapter_row(HERE.parent, {})
        finally:
            gr.subprocess.run = real
        self.assertIn("no hardware adapter on this machine (", str(e.exception))
        self.assertIn("Parent device is lost", str(e.exception))

    def test_machine_type_reads_nvidia_smi_and_falls_back(self):
        class P:
            def __init__(self, rc, out):
                self.returncode, self.stdout = rc, out
        self.assertEqual(gr.machine_type(lambda *a, **k: P(0, "NVIDIA A100-SXM4-80GB\n")), "NVIDIA A100-SXM4-80GB")
        self.assertEqual(gr.machine_type(lambda *a, **k: P(0, "NVIDIA RTX A6000\nNVIDIA RTX A6000\n")), "NVIDIA RTX A6000")
        self.assertIn("no nvidia-smi GPU", gr.machine_type(lambda *a, **k: P(9, "")))

        def missing(*a, **k):
            raise FileNotFoundError("nvidia-smi")
        self.assertIn("no nvidia-smi GPU", gr.machine_type(missing))

    def test_the_rule_covers_the_device_code_and_itself(self):
        text = (HERE / "gpu-receipt-suites.txt").read_text(encoding="utf-8")
        paths = {ln.split()[1] for ln in text.splitlines() if ln.startswith("path ")}
        for need in ("src/sheet_field/**", "src/gpu_context.rs", "Cargo.toml", "Cargo.lock",
                     "scripts/gpu-receipt-suites.txt", "scripts/gpu_receipt_run.py"):
            self.assertIn(need, paths)

    def test_the_repository_rule_parses(self):
        self.assertGreater(len(gr.suites(HERE.parent)), 0)


if __name__ == "__main__":
    unittest.main()
