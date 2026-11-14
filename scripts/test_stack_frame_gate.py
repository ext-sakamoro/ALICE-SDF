#!/usr/bin/env python3
"""Tests of scripts/stack_frame_gate.py

Author: Moroya Sakamoto
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stack_frame_gate as g  # noqa: E402

ARM64 = """
0000000100080db0 <__RNvMs_NtNtCslj63CBXhRGC_9alice_sdf8compiled8compilerNtB4_8Compiler12compile_node>:
100080db0:     \tsub\tsp, sp, #0x80
100080db4:     \tstp\tx29, x30, [sp, #0x70]
100080db8:     \tadd\tx29, sp, #0x70
100080dbc:     \tbl\t0x100000000 <foo>
100080dc0:     \tsub\tsp, sp, #0x4000
0000000100090000 <__RNCNvNtCslj63CBXhRGC_9alice_sdf8interval14eval_lipschitz0B5_>:
100090000:     \tsub\tsp, sp, #0x20
00000001000a0000 <__RNvNtCslj63CBXhRGC_9alice_sdf8interval14eval_lipschitz>:
1000a0000:     \tstp\tx28, x27, [sp, #-0x20]!
1000a0004:     \tstp\tx29, x30, [sp, #0x10]
1000a0008:     \tadd\tx29, sp, #0x10
1000a000c:     \tsub\tx9, sp, #0x1e0
1000a0010:     \tand\tsp, x9, #0xffffffffffffffc0
00000001000b0000 <__ZN9alice_sdf8compiled8compiler16validate_subtree17h0123456789abcdefE>:
1000b0000:     \tstp\tx29, x30, [sp, #-0x10]!
1000b0004:     \tsub\tsp, sp, #0x1, lsl #12 ; =0x1000
00000001000c0000 <__RNvNtCslj63CBXhRGC_9alice_sdf4eval4eval>:
1000c0000:     \tmov\tx9, #0x5000
1000c0004:     \tsub\tsp, sp, x9
"""

INLINED_COPY = """
000000010000754c <__RNvNtCslj63CBXhRGC_9alice_sdf4eval4evalCsfPly4OVopP2_22test_eval_stack_budget>:
10000754c:     \tsub\tsp, sp, #0x130
"""

X86 = """
0000000000012340 <_ZN9alice_sdf4eval4eval17h0011223344556677E>:
   12340:\tpush   %rbp
   12341:\tpush   %r15
   12343:\tsub    $0x48,%rsp
   12347:\tcall   12000 <x>
0000000000012400 <_RNvNtCs1_9alice_sdf8interval14eval_lipschitz.llvm.4242>:
   12400:\tpush   %rbx
   12401:\tand    $0xffffffffffffffc0,%rsp
"""


def _uleb(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _sleb(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        done = (n == 0 and not b & 0x40) or (n == -1 and b & 0x40)
        out.append(b | (0 if done else 0x80))
        if done:
            return bytes(out)


def _section(sid, payload):
    return bytes([sid]) + _uleb(len(payload)) + payload


def _name(s):
    b = s.encode()
    return _uleb(len(b)) + b


def wasm_module(frame):
    """two functions: `_RNvCs1_9alice_sdf4eval` with an `frame`-byte prologue, `leaf` without"""
    body1 = b"\x00" + b"\x23\x00" + b"\x41" + _sleb(frame) + b"\x6b" + b"\x21\x00\x0b"
    body2 = b"\x00" + b"\x41\x01\x1a\x0b"
    code = _uleb(2) + _uleb(len(body1)) + body1 + _uleb(len(body2)) + body2
    fnames = _uleb(2) + _uleb(0) + _name("_RNvCs1_9alice_sdf4eval") + _uleb(1) + _name("leaf")
    gnames = _uleb(1) + _uleb(0) + _name("__stack_pointer")
    names = _name("name") + b"\x01" + _uleb(len(fnames)) + fnames + b"\x07" + _uleb(len(gnames)) + gnames
    return b"\0asm\x01\0\0\0" + _section(10, code) + _section(0, names)


class Native(unittest.TestCase):
    def frames(self, text):
        return {s: g.native_frame(i) for s, i in g.objdump_functions(text).items()}

    def test_arm64_prologues(self):
        f = self.frames(ARM64)
        rep, fails = g.gate(f, ["alice_sdf::compiled::compiler::Compiler::compile_node"], 512)
        self.assertEqual(fails, [])
        self.assertIn("128 B", rep[0])  # stops at the first call: the later sub is not the prologue
        # re-aligned frame: 0x20 + 0x1e0 + up to 63 bytes of alignment
        _, fails = g.gate(f, ["alice_sdf::interval::eval_lipschitz"], 574)
        self.assertEqual(len(fails), 1)
        self.assertIn("575 B", fails[0])
        # legacy mangling; `lsl #12`
        rep, fails = g.gate(f, ["alice_sdf::compiled::compiler::validate_subtree"], 8192)
        self.assertEqual(fails, [])
        self.assertIn(f"{16 + 4096} B", rep[0])

    def test_inline_copy_in_another_crate(self):
        rep, fails = g.gate(self.frames(INLINED_COPY), ["alice_sdf::eval::eval"], 512)
        self.assertEqual(fails, [])
        self.assertIn("304 B", rep[0])

    def test_closure_does_not_match(self):
        f = self.frames(ARM64)
        rep, _ = g.gate(f, ["alice_sdf::interval::eval_lipschitz"], 10_000)
        self.assertEqual(len(rep), 1)  # the closure `...eval_lipschitz0B5_` is not counted

    def test_register_sized_frame_fails(self):
        _, fails = g.gate(self.frames(ARM64), ["alice_sdf::eval::eval"], 10_000)
        self.assertEqual(len(fails), 1)
        self.assertIn("not readable", fails[0])

    def test_x86_prologues(self):
        f = self.frames(X86)
        rep, fails = g.gate(f, ["alice_sdf::eval::eval"], 512)
        self.assertEqual(fails, [])
        self.assertIn(f"{8 + 8 + 0x48 + 8} B", rep[0])
        rep, fails = g.gate(f, ["alice_sdf::interval::eval_lipschitz"], 512)
        self.assertEqual(fails, [])
        self.assertIn(f"{8 + 63 + 8} B", rep[0])  # `.llvm.<n>` clone matched

    def test_over_budget_missing_and_empty_fail(self):
        f = self.frames(X86)
        _, fails = g.gate(f, ["alice_sdf::eval::eval"], 87)
        self.assertEqual(len(fails), 1)
        _, fails = g.gate(f, ["alice_sdf::compiled::compiler::Compiler::compile_node"], 512)
        self.assertIn("not found", fails[0])
        _, fails = g.gate(f, [], 512)
        self.assertEqual(fails, ["no function to measure"])


class Wasm(unittest.TestCase):
    def test_shadow_stack_prologue(self):
        for frame in (16, 128, 4096, 70000):
            f = g.wasm_functions(wasm_module(frame))
            self.assertEqual(f["_RNvCs1_9alice_sdf4eval"], frame)
            self.assertEqual(f["leaf"], 0)

    def test_gate_on_module_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.wasm"
            p.write_bytes(wasm_module(640))
            rc = g.main([str(p), "--wasm", "--budget", "512", "--function", "alice_sdf::eval"])
            self.assertEqual(rc, 1)
            p.write_bytes(wasm_module(96))
            rc = g.main([str(p), "--wasm", "--budget", "512", "--function", "alice_sdf::eval"])
            self.assertEqual(rc, 0)

    def test_not_wasm(self):
        with self.assertRaises(ValueError):
            g.wasm_functions(b"\x7fELF....")


if __name__ == "__main__":
    unittest.main()
