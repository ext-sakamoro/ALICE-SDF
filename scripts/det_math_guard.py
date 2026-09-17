#!/usr/bin/env python3
"""3.1.0 bit-exact guard: no platform libm call and no fused `mul_add` in the
evaluator and law directories.

Every transcendental there goes through `alice_det_math` and `a * b + c` is
two roundings; a stray `x.sin()` or `.mul_add(` silently reintroduces
per-target bits (`tests/test_det_parity.rs` / `test_det_golden.rs` would
catch the value drift, this catches the cause). Test modules (from
`#[cfg(test)] mod` on) may call libm as the reference. `src/compiled/real.rs`
is excluded: its generic laws call the `Real` trait methods (`x.sin_cos()`),
which the `f32` / `f32x8` impls wire to alice_det_math.

The remaining directories (npr / mesh / texture / gi / io / terrain) move
over in 3.2.0, after which this becomes a `clippy.toml` `disallowed-methods`
gate for the whole crate.
"""
import re
import subprocess
import sys

DIRS = [
    "src/primitives/", "src/modifiers/", "src/operations/", "src/eval/",
    "src/compiled/", "src/raycast/", "src/sdf2d.rs", "src/crispy.rs",
]
EXCLUDE = {"src/compiled/real.rs"}
# `Real` trait method calls in generic laws (`R: Real`): `(x).exp()` on `R`
# dispatches to alice_det_math through the `f32` / `f32x8` impls
GENERIC_OK = re.compile(r"R::(one|zero|splat)\(|<R: Real>|: R\b")
CALL = re.compile(
    r"\.(sin|cos|sin_cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|exp|exp2|ln|log|log2|log10|powf|powi|cbrt|hypot|mul_add)\("
)
TEST_MOD = re.compile(r"#\[cfg\(test\)\]\s*(pub\s+)?mod\s")

files = subprocess.run(["git", "ls-files"] + DIRS, capture_output=True, text=True, check=True).stdout.split()
hits = []
for f in files:
    if not f.endswith(".rs") or f in EXCLUDE:
        continue
    src = open(f, encoding="utf-8").read()
    m = TEST_MOD.search(src)
    head = src if not m else src[: m.start()]
    for i, line in enumerate(head.split("\n"), 1):
        if line.lstrip().startswith("//"):
            continue
        if CALL.search(line) and not GENERIC_OK.search(line):
            hits.append(f"{f}:{i}: {line.strip()}")
if hits:
    print("❌ platform libm / mul_add call in a bit-exact directory (use alice_det_math::*, a * b + c):")
    print("\n".join(hits))
    sys.exit(1)
print(f"✓ No platform libm / mul_add in the evaluator and law directories ({len(files)} files)")
