#!/usr/bin/env python3
"""Oracle for scripts/workflow_tool_pins.py: an unpinned tool install fails, a pinned one passes, an empty tree fails.

Shared byte-identical by the ALICE crates that carry it (this file and scripts/workflow_tool_pins.py, ALICE-Law's
shared table); change every copy together."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workflow_tool_pins as wtp  # noqa: E402

HEAD = "name: x\non:\n  push:\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n"
SHA = "fe3acb53a98b62c772db35de75ce0e4595e7ca8c"


def run_step(step: str) -> str:
    return HEAD + "      - run: " + step + "\n"


def action(ref: str = SHA, tool: str = "cargo-deny@0.22.0", fallback: str | None = "none") -> str:
    w = f"        with:\n          tool: {tool}\n" + (f"          fallback: {fallback}\n" if fallback else "")
    return HEAD + f"      - uses: taiki-e/install-action@{ref}\n" + w


def run(*texts: str) -> int:
    with tempfile.TemporaryDirectory() as d:
        for i, t in enumerate(texts):
            Path(d, f"w{i}.yml").write_text(t, encoding="utf-8")
        return wtp.main([d])


class CargoInstall(unittest.TestCase):
    def test_the_fleet_s_unpinned_forms_fail(self):
        for cmd in ("cargo install --locked cargo-deny", "cargo install cargo-audit --locked",
                    "cargo +nightly install cargo-fuzz --locked", "cargo install cargo-audit cargo-deny --locked"):
            with self.subTest(cmd=cmd):
                self.assertEqual(run(run_step(cmd)), 1)

    def test_a_version_without_a_pinned_toolchain_fails(self):
        self.assertEqual(run(run_step("cargo install --locked --version 0.13.2 cargo-fuzz")), 1)

    def test_a_floating_toolchain_fails(self):
        for chain in ("+stable", "+nightly", "+beta", "+1.95"):
            with self.subTest(chain=chain):
                self.assertEqual(run(run_step(f"cargo {chain} install --locked --version 0.13.2 cargo-fuzz")), 1)

    def test_a_pinned_toolchain_and_exact_version_pass(self):
        for cmd in ("cargo +1.95.0 install --locked --version 0.13.2 cargo-fuzz",
                    "cargo +1.95.0 install --locked cargo-fuzz@0.13.2",
                    "cargo +1.95.0 install --locked --version=0.13.2 cargo-fuzz",
                    "cargo +1.95.0 install --locked --version =4.1.2 cargo-ndk"):
            with self.subTest(cmd=cmd):
                self.assertEqual(run(run_step(cmd)), 0)

    def test_a_range_version_fails(self):
        for v in ("^0.13", "0.13", "~0.13.2", "0.13.*"):
            with self.subTest(v=v):
                self.assertEqual(run(run_step(f"cargo +1.95.0 install --locked --version '{v}' cargo-fuzz")), 1)

    def test_a_continued_line_is_read_whole(self):
        step = "|\n          cargo +1.95.0 install --locked \\\n            --version 0.13.2 cargo-fuzz"
        self.assertEqual(run(HEAD + "      - run: " + step + "\n"), 0)
        bad = "|\n          cargo install --locked \\\n            cargo-fuzz"
        self.assertEqual(run(HEAD + "      - run: " + bad + "\n"), 1)

    def test_an_install_after_another_command_is_found(self):
        self.assertEqual(run(run_step("echo hi && cargo install cargo-deny")), 1)

    def test_a_path_install_is_exempt_and_a_git_install_needs_a_rev(self):
        self.assertEqual(run(run_step("cargo install --path tools/gen")), 0)
        self.assertEqual(run(run_step("cargo +1.95.0 install --git https://example.invalid/x tool")), 1)
        self.assertEqual(run(run_step(f"cargo +1.95.0 install --git https://example.invalid/x --rev {SHA} tool")), 0)

    def test_binstall_needs_an_exact_version(self):
        self.assertEqual(run(run_step("cargo binstall -y cargo-deny")), 1)
        self.assertEqual(run(run_step("cargo binstall -y cargo-deny@0.22.0")), 0)

    def test_an_inline_version_that_is_not_exact_fails(self):
        for c in ("cargo-deny@0.22", "cargo-deny@latest", "cargo-deny@^0.22.0"):
            with self.subTest(c=c):
                self.assertEqual(run(run_step(f"cargo binstall -y {c}")), 1)

    def test_one_version_for_two_crates_fails(self):
        self.assertEqual(run(run_step("cargo +1.95.0 install --locked --version 0.22.0 cargo-deny cargo-audit")), 1)

    def test_an_env_prefix_does_not_hide_the_install(self):
        self.assertEqual(run(run_step("RUSTFLAGS=-Dwarnings cargo install cargo-deny")), 1)


class InstallAction(unittest.TestCase):
    def test_a_pinned_step_passes(self):
        self.assertEqual(run(action()), 0)

    def test_a_tag_ref_fails(self):
        self.assertEqual(run(action(ref="v2")), 1)

    def test_a_tool_without_an_exact_version_fails(self):
        for tool in ("cargo-deny", "cargo-deny@0.22", "cargo-deny@latest"):
            with self.subTest(tool=tool):
                self.assertEqual(run(action(tool=tool)), 1)

    def test_several_tools_are_each_checked(self):
        self.assertEqual(run(action(tool="cargo-deny@0.22.0,cargo-audit@0.23.0")), 0)
        self.assertEqual(run(action(tool="cargo-deny@0.22.0,cargo-audit")), 1)

    def test_a_step_without_a_tool_input_fails(self):
        self.assertEqual(run(HEAD + f"      - uses: taiki-e/install-action@{SHA}\n        with:\n          fallback: none\n"), 1)

    def test_a_missing_or_other_fallback_fails(self):
        self.assertEqual(run(action(fallback=None)), 1)
        self.assertEqual(run(action(fallback="cargo-binstall")), 1)


# a CLI that must equal the crate the lock resolves takes its version from a step that reads Cargo.lock
LOCK_STEP_AWK_EQ = (
    "      - id: wbg\n        run: |\n          set -euo pipefail\n"
    "          v=$(awk '/^\\[\\[package\\]\\]/{p=0} $0==\"name = \\\"wasm-bindgen\\\"\"{p=1} "
    "p&&/^version = /{gsub(/\"/,\"\",$3); print $3; exit}' Cargo.lock)\n"
    "          [ -n \"$v\" ] || { echo \"error: no wasm-bindgen in Cargo.lock\" >&2; exit 1; }\n"
    "          echo \"version=$v\" >> \"$GITHUB_OUTPUT\"\n")
LOCK_STEP_ANCHORED = (
    "      - id: wbg\n        run: |\n          set -euo pipefail\n"
    "          v=$(awk '/^name = \"wasm-bindgen\"$/{getline; gsub(/\"/, \"\", $3); print $3; exit}' Cargo.lock)\n"
    "          test -n \"${v}\"\n"
    "          echo \"version=${v}\" >> \"${GITHUB_OUTPUT}\"\n")
FROM_LOCK = ("      - uses: taiki-e/install-action@" + SHA + "\n        with:\n"
             "          tool: wasm-bindgen@${{ steps.wbg.outputs.version }}\n          fallback: none\n")


class LockPinned(unittest.TestCase):
    def test_the_two_fleet_forms_are_accepted(self):
        for step in (LOCK_STEP_AWK_EQ, LOCK_STEP_ANCHORED):
            with self.subTest(step=step[:60]):
                self.assertEqual(run(HEAD + step + FROM_LOCK), 0)

    def test_an_empty_value_that_does_not_stop_fails(self):
        no_stop = LOCK_STEP_ANCHORED.replace("          set -euo pipefail\n", "").replace('          test -n "${v}"\n', "")
        self.assertEqual(run(HEAD + no_stop + FROM_LOCK), 1)
        # test -n without set -e does not stop the step
        self.assertEqual(run(HEAD + LOCK_STEP_ANCHORED.replace("          set -euo pipefail\n", "") + FROM_LOCK), 1)

    def test_a_substring_match_fails(self):
        sub = LOCK_STEP_ANCHORED.replace('/^name = \"wasm-bindgen\"$/', '/name = \"wasm-bindgen/')
        self.assertEqual(run(HEAD + sub + FROM_LOCK), 1)

    def test_a_step_id_mismatch_fails(self):
        self.assertEqual(run(HEAD + LOCK_STEP_ANCHORED.replace("id: wbg", "id: other") + FROM_LOCK), 1)

    def test_an_id_on_two_steps_or_on_a_step_without_run_fails(self):
        self.assertEqual(run(HEAD + LOCK_STEP_ANCHORED + LOCK_STEP_ANCHORED + FROM_LOCK), 1)
        uses_step = "      - id: wbg\n        uses: actions/checkout@" + SHA + "\n"
        self.assertEqual(run(HEAD + uses_step + FROM_LOCK), 1)

    def test_a_step_that_does_not_read_the_lock_fails(self):
        no_lock = LOCK_STEP_ANCHORED.replace(" Cargo.lock)", " versions.txt)")
        self.assertEqual(run(HEAD + no_lock + FROM_LOCK), 1)

    def test_an_output_key_the_step_does_not_write_fails(self):
        self.assertEqual(run(HEAD + LOCK_STEP_ANCHORED + FROM_LOCK.replace("outputs.version", "outputs.v")), 1)


class Tree(unittest.TestCase):
    def test_no_workflow_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(wtp.main([d]), 1)

    def test_a_workflow_without_installs_passes(self):
        self.assertEqual(run(run_step("cargo test")), 0)

    def test_one_bad_workflow_among_good_ones_fails(self):
        self.assertEqual(run(action(), run_step("cargo install cargo-deny")), 1)

    def test_without_the_reader_the_check_fails(self):
        saved, wtp.cp = wtp.cp, None
        try:
            self.assertEqual(run(action()), 1)
        finally:
            wtp.cp = saved


if __name__ == "__main__":
    unittest.main()
