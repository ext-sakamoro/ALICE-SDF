#!/usr/bin/env python3
"""Compare every `extern "C"` declaration of an exported symbol against the
definition in `src/ffi/`, as an ordered list of types.

Why this exists
---------------
`scripts/unreal-abi-check.sh` step 2a compares `include/alice_sdf.h` and
`bindings/AliceSdf.cs` against the Rust exports. That leaves a third
declaration site unchecked: the `extern "C"` blocks that tests and examples
write so they can call the real exported symbol instead of a private module.

2026-09-30 (run 36680136653): `tests/test_binding_oracle.rs` declared

    fn alice_sdf_repeat_finite(node, sx: f32, sy: f32, sz: f32,
                                     cx: u32, cy: u32, cz: u32)

while the export takes the three `u32` first. **Windows was the only CI job
that went red**; macOS and Linux passed.

The reason is the calling convention, and it is why a human review of the two
lists is not enough:

* **SysV** (macOS, Linux) keeps *separate* registers for integers (RDI, RSI,
  RDX, RCX, R8, R9) and for floats (XMM0-7), and advances the two counters
  independently. Swapping the integer group with the float group leaves every
  value in the register the callee reads, so the program behaves correctly and
  the test passes.
* **Microsoft x64** assigns the first four arguments by *position* — RCX/RDX/
  R8/R9 or XMM0-3 for slot 0..3, the rest on the stack — so the same swap makes
  the callee read unrelated registers and stack slots.

So an argument-order error in a declaration is invisible on SysV by
construction. Only a comparison of the ordered type lists catches it, which is
what this script does. Comparing the *set* of types is not enough.

Usage
-----
    python3 scripts/abi_decl_check.py            # repo root
    python3 scripts/abi_decl_check.py --verbose

Exit status 0 when every declaration matches, 1 otherwise. A mismatch prints
the marker ABI-DECL-MISMATCH so the failure is greppable.

Author: Moroya Sakamoto
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "ABI-DECL-MISMATCH"

DECL_RE = re.compile(r"fn\s+(alice_sdf_\w+)\s*\((.*?)\)\s*(?:->\s*[^;{]+)?;", re.S)
DEF_RE = re.compile(r'extern\s+"C"\s+fn\s+(alice_sdf_\w+)\s*\((.*?)\)\s*(?:->|\{)', re.S)


def arg_types(args: str) -> list[str]:
    """Ordered argument types. Splits on top-level commas so `[f32; 16]` and
    generic parameters survive."""
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in args:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    out = []
    for p in parts:
        p = p.strip()
        if not p or ":" not in p:
            continue
        # normalise whitespace so `*const  f32` and `*const f32` agree
        out.append(" ".join(p.split(":", 1)[1].split()))
    return out


def extern_blocks(text: str):
    """Yield the body of each `extern "C" { ... }` block."""
    for m in re.finditer(r'extern\s+"C"\s*\{', text):
        i = m.end()
        depth = 1
        start = i
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        yield text[start : i - 1]


def main() -> int:
    verbose = "--verbose" in sys.argv
    root = Path(__file__).resolve().parent.parent

    # definitions
    defs: dict[str, list[str]] = {}
    for p in sorted((root / "src" / "ffi").rglob("*.rs")):
        for m in DEF_RE.finditer(p.read_text()):
            defs[m.group(1)] = arg_types(m.group(2))
    if not defs:
        print(f"{MARKER}: no exports found under src/ffi — the scan is broken")
        return 1

    # declarations in tests/ and examples/
    problems: list[str] = []
    checked = 0
    sites = 0
    for base in ("tests", "examples"):
        d = root / base
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.rs")):
            text = p.read_text()
            if 'extern "C"' not in text:
                continue
            for body in extern_blocks(text):
                for m in DECL_RE.finditer(body):
                    name, got = m.group(1), arg_types(m.group(2))
                    sites += 1
                    want = defs.get(name)
                    rel = p.relative_to(root)
                    if want is None:
                        problems.append(
                            f"  {rel}: {name} is declared but src/ffi exports no such symbol"
                        )
                        continue
                    checked += 1
                    if got != want:
                        problems.append(
                            f"  {rel}: {name}\n"
                            f"      declared: ({', '.join(got)})\n"
                            f"      exported: ({', '.join(want)})"
                        )
                    elif verbose:
                        print(f"  ok {rel}: {name}")

    print(
        f"abi_decl_check: {len(defs)} exports, {sites} declarations found, "
        f"{checked} compared, {len(problems)} mismatched"
    )
    if problems:
        print(f"{MARKER}: an ordered type list disagrees with the export.")
        print(
            "  An argument-order error here is invisible on SysV (separate integer\n"
            "  and float register banks) and breaks on Microsoft x64 (slots assigned\n"
            "  by position). See the module docstring."
        )
        for line in problems:
            print(line)
        return 1
    if checked == 0:
        print(f"{MARKER}: nothing was compared — the declaration scan found no symbols")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
