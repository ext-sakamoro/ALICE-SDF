#!/usr/bin/env python3
"""Stack frame gate: the bytes one call of each recursive tree walker reserves, read from a binary.

A tree walker that recurses once per tree level (``eval``, the bytecode compiler, its validation,
its stack depths, the Lipschitz bound) uses its frame once per level: a frame over the budget
makes a deep tree overflow a thread or a wasm module's stack. The runtime canaries
(``tests/test_eval_stack_budget.rs``, ``tests/test_compile_stack_budget.rs``) bound the whole
path; this gate names the function and its size, from the prologue the compiler emitted:

- native (``objdump -d`` output): arm64 ``stp ..., [sp, #-N]!`` / ``str ..., [sp, #-N]!`` /
  ``sub sp, sp, #N`` (``lsl #12``) / ``sub x9, sp, #N`` + ``and sp, x9, #mask`` (re-alignment
  adds up to the alignment); x86-64 ``push`` (8 each) / ``sub $N, %rsp`` / ``and $-A, %rsp``,
  plus the 8-byte return address
- wasm (the module itself): ``global.get <stack pointer>; i32.const N; i32.sub`` near the start
  of the body (the shadow stack in linear memory; a function without it reserves 0)

A function is matched by the path segments of its mangled symbol (legacy ``_ZN...17h<hash>E`` and
v0 ``_R...`` both spell ``<len><ident>``), anchored at the end, so closures inside it do not match.
Exit 1 when a frame is over the budget, a function is not found, a prologue cannot be read, or no
function was asked for.

Author: Moroya Sakamoto
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Frame:
    symbol: str
    bytes: int | None  # None: the prologue could not be read


def segment_pattern(path: str) -> re.Pattern[str]:
    """`alice_sdf::eval::eval` -> the mangled segments `9alice_sdf` .. `4eval` .. `4eval`, at the end"""
    # in order, anything between (v0 puts the impl's type path between the module and the
    # method), each length not preceded by another digit; legacy ends with the hash
    # `17h<16 hex>E`; a v0 copy of an `#[inline]` function made in another crate ends with that
    # crate (`Cs<hash>_<len><crate>`); LLVM may clone a function as `<sym>.llvm.<n>`
    segs = ".*".join(rf"(?<![0-9]){len(p)}{re.escape(p)}" for p in path.split("::"))
    tail = r"(?:17h[0-9a-f]{16}E|Cs[0-9A-Za-z]*_[0-9]+\w+)?(?:\.llvm\.\d+)?$"
    return re.compile(segs + tail)


# ---------------------------------------------------------------- native (objdump text)

_FUNC = re.compile(r"^[0-9a-f]+ <(.+)>:$")


def objdump_functions(text: str, limit: int | None = 16) -> dict[str, list[str]]:
    """symbol (leading underscores of the Mach-O spelling removed) -> its first `limit`
    instructions (all of them with `limit=None`)"""
    funcs: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for line in text.splitlines():
        m = _FUNC.match(line.strip())
        if m:
            cur = funcs.setdefault(m.group(1).lstrip("_"), [])
            continue
        if cur is not None and (limit is None or len(cur) < limit) and ":" in line:
            ins = line.split(":", 1)[1].strip()
            if ins:
                cur.append(re.sub(r"\s+", " ", ins))
    return funcs


_CALL = re.compile(r"^(?:bl|b|call[q]?|jmp[q]?) .*<_*([^<>+]+)>$")


def calls_itself(symbol: str, instructions: list[str]) -> bool:
    """a call or a tail jump to the function's own entry (`<sym>`, not a jump inside it,
    which objdump spells `<sym+0x..>`)"""
    own = symbol.lstrip("_")
    for ins in instructions:
        m = _CALL.match(ins)
        if m and m.group(1) == own:
            return True
    return False


def _imm(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def native_frame(instructions: list[str]) -> int | None:
    """Bytes the prologue reserves; None when a size is held in a register (unreadable)"""
    total, x86, pending_x9 = 0, False, None
    for ins in instructions:
        if m := re.match(r"stp \S+, \S+, \[sp, #-(\w+)\]!", ins) or re.match(
            r"str \S+, \[sp, #-(\w+)\]!", ins
        ):
            total += _imm(m.group(1))
        elif m := re.match(r"sub sp, sp, #(\w+)(?:, lsl #12)?", ins):
            total += _imm(m.group(1)) * (4096 if "lsl #12" in ins else 1)
        elif re.match(r"sub sp, sp, [wx]\d+", ins):
            return None
        elif m := re.match(r"sub x9, sp, #(\w+)", ins):
            pending_x9 = _imm(m.group(1))
        elif (m := re.match(r"and sp, x9, #(\w+)", ins)) and pending_x9 is not None:
            mask = _imm(m.group(1)) & 0xFFFF_FFFF_FFFF_FFFF
            total += pending_x9 + ((~mask) & 0xFFFF_FFFF_FFFF_FFFF)
            pending_x9 = None
        elif re.match(r"push[q]? %", ins):
            total, x86 = total + 8, True
        elif m := re.match(r"sub[q]? \$(\w+), ?%rsp", ins):
            total, x86 = total + _imm(m.group(1)), True
        elif m := re.match(r"and[q]? \$(-?\w+), ?%rsp", ins):
            mask = _imm(m.group(1)) & 0xFFFF_FFFF_FFFF_FFFF  # -64 or 0xffffffffffffffc0
            total, x86 = total + ((~mask) & 0xFFFF_FFFF_FFFF_FFFF), True
        elif re.match(r"sub[q]? %\w+, ?%rsp", ins):
            return None
        elif re.match(r"(bl|b|ret|call|jmp)\b", ins):
            break
    return total + (8 if x86 else 0)


# ---------------------------------------------------------------- wasm (binary module)


def _uleb(b: bytes, i: int) -> tuple[int, int]:
    v = shift = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return v, i


def _sleb(b: bytes, i: int) -> tuple[int, int]:
    v = shift = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            if c & 0x40:
                v -= 1 << shift
            return v, i


def wasm_functions(data: bytes) -> dict[str, int]:
    """function name -> shadow stack bytes its prologue reserves"""
    if data[:4] != b"\0asm":
        raise ValueError("not a wasm module")
    i, n_imported, bodies, names, global_names = 8, 0, [], {}, {}
    while i < len(data):
        sid, i = data[i], i + 1
        size, i = _uleb(data, i)
        sec, i = data[i : i + size], i + size
        if sid == 2:  # imports: count the imported functions (they come first in the index space)
            n, j = _uleb(sec, 0)
            for _ in range(n):
                for _ in range(2):
                    ln, j = _uleb(sec, j)
                    j += ln
                kind, j = sec[j], j + 1
                if kind == 0:
                    n_imported += 1
                    _, j = _uleb(sec, j)
                elif kind == 1:  # table: reftype + limits
                    j += 1
                    flag, j = sec[j], j + 1
                    _, j = _uleb(sec, j)
                    if flag & 1:
                        _, j = _uleb(sec, j)
                elif kind == 2:  # memory limits
                    flag, j = sec[j], j + 1
                    _, j = _uleb(sec, j)
                    if flag & 1:
                        _, j = _uleb(sec, j)
                elif kind == 3:  # global: valtype + mut
                    j += 2
                else:  # tag
                    j += 1
                    _, j = _uleb(sec, j)
        elif sid == 10:
            n, j = _uleb(sec, 0)
            for _ in range(n):
                ln, j = _uleb(sec, j)
                bodies.append(sec[j : j + ln])
                j += ln
        elif sid == 0:
            ln, j = _uleb(sec, 0)
            if sec[j : j + ln] != b"name":
                continue
            j += ln
            while j < len(sec):
                sub, j = sec[j], j + 1
                sz, j = _uleb(sec, j)
                part, j = sec[j : j + sz], j + sz
                if sub not in (1, 7):
                    continue
                n, k = _uleb(part, 0)
                for _ in range(n):
                    idx, k = _uleb(part, k)
                    ln2, k = _uleb(part, k)
                    (names if sub == 1 else global_names)[idx] = part[k : k + ln2].decode(
                        "utf-8", "replace"
                    )
                    k += ln2
    sp = next((g for g, n in global_names.items() if n == "__stack_pointer"), 0)
    out: dict[str, int] = {}
    for k, body in enumerate(bodies):
        name = names.get(n_imported + k)
        if name is None:
            continue
        out[name] = _wasm_prologue(body, sp)
    return out


def _wasm_prologue(body: bytes, sp: int) -> int:
    n, j = _uleb(body, 0)
    for _ in range(n):  # local declarations: count + valtype
        _, j = _uleb(body, j)
        j += 1
    seen_sp = False
    for _ in range(12):
        if j >= len(body):
            break
        op, j = body[j], j + 1
        if op == 0x23:  # global.get
            g, j = _uleb(body, j)
            seen_sp = g == sp
        elif op in (0x20, 0x21, 0x22, 0x24):  # local.get/set/tee, global.set
            _, j = _uleb(body, j)
        elif op == 0x41:  # i32.const
            v, j = _sleb(body, j)
            if seen_sp and j < len(body) and body[j] == 0x6B:  # i32.sub
                return v
        else:
            break
    return 0


# ---------------------------------------------------------------- gate


def measure(binary: Path, kind: str, objdump: str) -> dict[str, int | None]:
    if kind == "wasm":
        return dict(wasm_functions(binary.read_bytes()))
    text = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {s: native_frame(ins) for s, ins in objdump_functions(text).items()}


def gate(frames: dict[str, int | None], paths: list[str], budget: int) -> tuple[list[str], list[str]]:
    """(report lines, failures)"""
    report, fails = [], []
    if not paths:
        fails.append("no function to measure")
    for path in paths:
        pat = segment_pattern(path)
        hits = [(s, b) for s, b in frames.items() if pat.search(s)]
        if not hits:
            fails.append(f"{path}: not found in the binary")
            continue
        for sym, b in hits:
            if b is None:
                fails.append(f"{path}: prologue not readable ({sym})")
            else:
                report.append(f"{path}: {b} B per call ({sym})")
                if b > budget:
                    fails.append(f"{path}: {b} B per call > budget {budget} B")
    return report, fails


def recursion_gate(text: str, paths: list[str]) -> tuple[list[str], list[str]]:
    """(report lines, failures): each named function is found and does not call itself"""
    report, fails = [], []
    if not paths:
        fails.append("no function to check")
    funcs = objdump_functions(text, limit=None)
    for path in paths:
        pat = segment_pattern(path)
        hits = [(sym, ins) for sym, ins in funcs.items() if pat.search(sym)]
        if not hits:
            fails.append(f"{path}: not found in the binary")
            continue
        for sym, ins in hits:
            if calls_itself(sym, ins):
                fails.append(f"{path}: calls itself ({sym})")
            else:
                report.append(f"{path}: does not call itself ({len(ins)} instructions, {sym})")
    return report, fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("binary", type=Path)
    ap.add_argument("--wasm", action="store_true", help="the binary is a wasm module")
    ap.add_argument("--budget", type=int, help="bytes per call (recursive functions)")
    ap.add_argument(
        "--no-recursion",
        action="store_true",
        help="check instead that each function does not call itself (native only)",
    )
    ap.add_argument("--function", action="append", default=[], help="Rust path, repeatable")
    ap.add_argument("--objdump", default="objdump")
    a = ap.parse_args(argv)
    if a.no_recursion:
        if a.wasm:
            ap.error("--no-recursion reads native disassembly")
        text = subprocess.run(
            [a.objdump, "-d", "--no-show-raw-insn", str(a.binary)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        report, fails = recursion_gate(text, a.function)
        for line in report:
            print(f"stack-frame: {line}")
        for line in fails:
            print(f"stack-frame: FAIL {line}", file=sys.stderr)
        print(f"stack-frame: {len(report)} checked, {len(fails)} failure(s), no self-calls")
        return 1 if fails else 0
    if a.budget is None:
        ap.error("--budget is required without --no-recursion")
    frames = measure(a.binary, "wasm" if a.wasm else "native", a.objdump)
    report, fails = gate(frames, a.function, a.budget)
    for line in report:
        print(f"stack-frame: {line}")
    for line in fails:
        print(f"stack-frame: FAIL {line}", file=sys.stderr)
    print(f"stack-frame: {len(report)} measured, {len(fails)} failure(s), budget {a.budget} B")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
