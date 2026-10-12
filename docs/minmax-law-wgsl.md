# WGSL min/max and the signed-zero Law (design note)

The tree, SIMD, bytecode and Cranelift-JIT evaluators pin `min`/`max` to one
ordered IEEE-754-2019 rule (`maximumNumber`/`minimumNumber`): NaN returns the
other operand, and a both-zero result follows `-0 < +0` (so `max` is `+0` iff
either operand is `+0`, `min` is `-0` iff either is `-0`, `max(-0,-0) = -0`).
See `law_max_f32`/`law_min_f32` in `src/compiled/real.rs`.

The WGSL transpiler currently emits the builtins `max(a,b)` / `min(a,b)` in the
Law paths (`src/compiled/wgsl/transpiler.rs`, e.g. `smooth_min`/`smooth_max`).
Per the WGSL spec the sign of a zero result from the builtins is
implementation-defined, so a GPU evaluation can disagree with the CPU
evaluators on the sign of a zero.

## Scope of the hazard on the GPU path

GPU output is rendered, not persisted: it does not feed the content hash
(`src/cache/mod.rs` quantizes to `i32`, collapsing `±0`) nor the SVO
serialization (CPU-built). The CPU↔GPU parity oracle
(`tests/test_npr_bytecode_gpu_parity.rs`) compares within a tolerance, which
absorbs a `±0` difference. In the `smooth_min`/`smooth_max` laws the clamp
result is squared (`h*h`), so a `±0` from the clamp is absorbed before it
reaches the output. So the GPU `±0` is latent today, not observed.

## The explicit-compare form (for Law paths only)

To make the GPU path agree with the CPU Law by construction (not by the
absorbing arithmetic that happens to follow), emit explicit compares instead of
the builtin in Law-bearing generated code:

```wgsl
fn law_max(a: f32, b: f32) -> f32 {
  // NaN -> other; both-zero -> +0 iff either is +0 (-0 < +0 ordered)
  if (a != a || b > a) { return b; }
  if (b != b || a > b || sign(a) >= 0.0) { return a; }
  return b;
}
fn law_min(a: f32, b: f32) -> f32 {
  if (a != a || b < a) { return b; }
  if (b != b || a < b || sign(a) < 0.0) { return a; }
  return b;
}
```

Notes for the implementer:
- `a != a` is the WGSL NaN test (`isNan` is not guaranteed under fast-math
  relaxations; the self-inequality is robust).
- `sign(a)` in WGSL returns `0.0` for both `+0` and `-0`, so it cannot
  distinguish the zero sign. Use the bit test instead:
  `(bitcast<u32>(a) & 0x80000000u) == 0u` for "is `+0`/non-negative".
  Rewrite the zero-sign branches with the bitcast; the `sign()` form above is a
  placeholder that does NOT capture `-0` and must be replaced before use.
- Keep the builtins everywhere else (color, UI, non-Law math): the explicit
  form is only for the transpiled Law paths, matched by the same lint policy
  that governs the CPU evaluators.
- Add the mixed-lane `±0` case to the GPU parity oracle (different zero sign per
  invocation) and assert bit-equality with the CPU Law for those seeds, so the
  gate has teeth rather than relying on the absorbing arithmetic.

## Priority

Lower than the CPU routing: no persisted-meaning change on the GPU path today.
Land after the CPU-side gate + `prim_table_simd` routing, with the parity oracle
extended to the `±0`/NaN seeds.
