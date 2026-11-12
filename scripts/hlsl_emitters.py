#!/usr/bin/env python3
"""Every source file that produces HLSL / USF text is in the Unreal both-engines list.

usage: hlsl_emitters.py [--list] [ROOT]

The Unreal workflow builds every installed engine for a ci/ change only when the change hits
.github/unreal-engines-both.txt (scripts/unreal_engines.py); engine versions differ in their
shader compilers, so a change to HLSL text must build every engine before it lands. This check
finds the files under src/ that produce HLSL text and fails when one of them is not matched by
a pattern of that list, so a new emitter cannot fall into the newest-engine-only class.

A file under src/ produces HLSL text when a line of it, outside a `feature = "hlsl"` cfg,
  * holds HLSL code: `float2(` .. `float4x4(`, `asfloat(`, `asuint(`, `[numthreads`, `cbuffer`,
    `RWStructuredBuffer`, a `: SV_<semantic>`, a declaration `float3 name =`, or the type names
    as strings (`"float3"`, a dialect table);
  * names the dialect as a string (`"hlsl"`, a per-dialect renderer);
  * defines an HLSL emitter: `fn <name>_hlsl` / `fn hlsl_<name>`, a `const` / `static
    <NAME>_HLSL`, an `impl .. for HlslLang`, or code generic over `L: ShaderLang` (it emits
    HLSL when `L` is `HlslLang`);
  * renders the shader template (`render_template(`) or tags every emitted shader
    (`tagged_shader(` / `tagged_shader_with(`), or names a `.usf` / `.ush` file.

Exit 1 when an emitter is not in the list, when no emitter is found (the scan compared nothing),
or when the list has no pattern. `--list` prints the emitters found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import changed_paths as cp  # noqa: E402

LIST = Path(".github/unreal-engines-both.txt")

EMIT = re.compile(
    r"\bfloat[234](?:x[234])?\("
    r"|\basfloat\(|\basuint\("
    r"|\[numthreads|\bcbuffer\b|\bRWStructuredBuffer\b"
    r"|:\s*SV_[A-Za-z]+"
    r"|\bfloat[234] [a-z_][a-z0-9_]* *="
    r'|"float[234]"'
    r'|"hlsl"'
    r"|\bfn [a-z0-9_]*_hlsl\b|\bfn hlsl_[a-z0-9_]+"
    r"|\b(?:const|static) [A-Z0-9_]*_HLSL\b"
    r"|\bfor HlslLang\b|\bL: ShaderLang\b"
    r"|\brender_template\(|\btagged_shader(?:_with)?\("
    r'|\.usf"|\.ush"'
)
CFG = re.compile(r'feature\s*=\s*"hlsl"')


def emits(text: str) -> bool:
    return any(EMIT.search(CFG.sub("", line)) for line in text.splitlines())


def emitters(root: Path) -> list[str]:
    out = []
    for p in sorted((root / "src").rglob("*.rs")):
        if emits(p.read_text(encoding="utf-8")):
            out.append(p.relative_to(root).as_posix())
    return out


def check(root: Path) -> tuple[list[str], list[str]]:
    """(findings, emitters found)"""
    pats = cp.read_patterns(str(root / LIST), [])
    if not pats:
        return [f"{LIST}: no pattern"], []
    rules = cp.compile_patterns(pats)
    found = emitters(root)
    if not found:
        return ["no HLSL emitter found under src/ (the scan compared nothing)"], found
    missing = [f for f in found if not cp.selected(f, rules)]
    return [f"{f}: produces HLSL text but is not in {LIST} (add it)" for f in missing], found


def main(argv: list[str]) -> int:
    show = "--list" in argv
    rest = [a for a in argv if a != "--list"]
    root = Path(rest[0]) if rest else HERE.parent
    findings, found = check(root)
    if show:
        for f in found:
            print(f)
    for f in findings:
        print(f"hlsl_emitters: {f}", file=sys.stderr)
    print(f"hlsl_emitters: {len(found)} emitter file(s), {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
