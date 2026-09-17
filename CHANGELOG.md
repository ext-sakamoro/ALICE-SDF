# Changelog

All notable changes to ALICE-SDF are documented in this file.

For releases prior to v1.5.0 (v0.1.0 – v1.3.0), see [CHANGELOG-history.md](CHANGELOG-history.md).

## [Unreleased]

### Fixed — `Terrain` shader emit was invalid GLSL/WGSL/HLSL, and its law drifted from the CPU

The `Terrain` arm in `transpiler_common.rs` wrote raw GLSL (`float`, `vec2`,
`for(int`, an undefined `vnoise`) regardless of the target language; naga
rejected it in GLSL (unknown function) and WGSL (invalid syntax). The CPU
law separately hashed with `fract(sin(·)·43758)`, the same GPU/CPU-divergent
pattern `SurfaceRoughness` moved away from for `hash_noise_3d` (PCG lattice
hash) before this.

- `SdfNode::Terrain`'s 3-octave fbm now samples `hash_noise_3d` on the xz
  plane (y held at 0, degenerating the trilinear blend to bilinear) on the
  CPU (`eval/mod.rs`) **and** through `ensure_helper("hash_noise")` +
  portable `ShaderLang` ops in the shader emit — same law, one source, no
  language-specific string left in `transpiler_common.rs`.
- `TERRAIN_FBM_GRAD` (the `eval_lipschitz` bound) updated 3.208 → 6.42: the
  new noise's range is `[-1, 1]` (was `[0, 1]`), doubling the per-axis
  gradient bound; `lipschitz_claim_bounds_every_difference_quotient`
  verifies it empirically.
- `Terrain` added to the shared GPU-oracle corpus
  (`tests/common/corpus.rs`) — it was the one node in the "144 nodes
  transpile, corpus oracle covers them" (3.0.0) claim the corpus never
  actually exercised. `every_corpus_node_matches_cpu_through_wgsl` /
  `_through_glsl` now cover it; `test_det_golden.rs` gained its pin.
  `test_det_parity.rs` and `test_evaluator_opcode_parity.rs` (which iterate
  the same corpus but exercise the bytecode compiler) now skip nodes with
  no bytecode law — `CompileError::UnsupportedPrimitive` — instead of
  assuming every corpus entry compiles.

### Fixed — `Elongate` shader law disagreed with every other evaluator

The tree evaluator, VM bytecode (`real::elongate`) and both Cranelift JITs
all implement IQ's cheap elongate — `q = p - clamp(p, -a, a)`, the child
evaluated as-is at `q` — but the GLSL/WGSL/HLSL transpilers emitted IQ's
*exact* elongate (`q = max(abs(p) - a, 0)` plus a `min(max(q), 0)` box
correction on the returned distance), a different law that disagreed by up
to 1.0 on interior points (`elongate(1,2,3, sphere(1))` at
`(0.011, 0.666, 0.515)`: cpu `-1.0`, shader `-1.99`). The shader emit now
matches the other four evaluators; `every_grammar_construct_matches_cpu_on_gpu`
(alice-lol) and the corpus GPU-parity tests confirm it.

## [v3.1.0] - 2026-09-17

### Changed — cross-platform bit-exact evaluation (alice-det-math)

Distances move in the last ulp everywhere a law calls a transcendental
(twist / bend / TPMS / polar / helix / smooth-exp / …) and wherever
`mul_add` was used; no law changed its meaning. Goldens re-pinned.

- Every transcendental in the evaluator and law directories (`primitives` /
  `modifiers` / `operations` / `eval` / `compiled` / `raycast` / `sdf2d`) goes
  through [`alice-det-math`](https://crates.io/crates/alice-det-math) 0.2
  (scalar and `f32x8`, the crate `alice-physics` 1.4 uses), and `a * b + c`
  is two roundings everywhere (`mul_add` removed — it fuses on FMA hardware
  and not elsewhere). `Real` for `f32` / `f32x8` (`sin_cos` / `atan2` / `exp`
  / `ln` / `round` / `mul_add`) dispatches to it; the SIMD table's own `wide`
  sin / cos are gone.
- The tree evaluator, compiled scalar, `f32x8` SIMD, BVH and Cranelift SIMD
  JIT are **bit-identical** to each other (`tests/test_det_parity.rs`, 144
  corpus nodes × 266 points, `to_bits()` equality) and across x86_64 /
  aarch64 / wasm32 (`tests/test_det_golden.rs`, one SHA-256 per corpus node
  on a libm-free grid; verified on aarch64 and x86_64 via Rosetta locally,
  CI lanes pin it). `Scale` in the tree evaluator uses `p * (1/s)` like the
  compiled evaluators (was `p / s`).
- SIMD JIT: `simd_sincos_approx` (unreduced Taylor, 4e-6 abs error, and its
  `bitselect` masks were never bitcast — twist / bend did not compile on
  Cranelift 0.113, which the tolerance test skipped silently) is replaced by
  the det_math law emitted as IR; `fma` → `fmul` + `fadd`; the length,
  smooth-blend, rotate, cone, rounded-cone and pyramid arms follow the law's
  operation order (`(x*x + y*y) + z*z`, `h = max(1 - |a-b|·rk, 0)`,
  quaternion rotation, division by `k2·k2` / `m2`). Ellipsoid has no JIT arm
  any more: the scalar law is the exact Eberly distance, the JIT arm was the
  IQ approximation (up to 30% off). The scalar tree JIT (`JitCompiledSdf`)
  is held to 1e-5 only (3.2.0).
- Shaders: `alice_atan2` helper (WGSL / GLSL / HLSL) returns the CPU law's
  exact constants on the axes (`atan2(±0, x<0) = ±π`, `atan2(y, ±0) = ±π/2`),
  so polar / polygon / helix sector ties on an axis snap like the CPU
  (`polar_axis_ties_gpu_match_cpu`, WGSL and GLSL). Hardcoded constants are
  printed with round-trip precision (`6.2831855`, was `{:.6}` → `6.283185`,
  which alone flipped a sector at π).
- `security-audit.yml` / preflight: `scripts/det_math_guard.py` fails on any
  libm method call or `mul_add` in the bit-exact directories.
- `benches/sdf_eval.rs`: `transcendental_laws` group (6 laws × scalar / SIMD /
  JIT). SIMD is faster than 3.0.0 (gyroid 121 → 20 µs / 4096 pts, twist 22 →
  12, exp-smooth 23 → 24); scalar is 1.3–2.4× slower (libm 1.5 ns vs det_math
  3.6 ns per `sin`) — the price of the guarantee.

### Fixed

- `tests/test_relaxed_tracing.rs`: the judge re-evaluated a hit with the
  un-renormalised direction (the marcher normalises), which crossed the 1e-4
  band on a grazing gyroid ray by 2e-9.

### Added — VRChat package: host-side parity of the Mochi collider

- `examples/vrchat_mochi_golden.rs` prints the Mochi scene from
  `alice_sdf::eval`; `vrchat-package/HostTests~/MochiParity` compiles the
  UdonSharp collider against a UnityEngine stub and checks its
  `EvaluateSdf` against that golden (1521 points, 1e-5) plus a
  grab / split / merge scenario. `scripts/vrchat-host-parity.sh`, CI job
  `vrchat-host` (setup-dotnet), preflight step.

### Changed — VRChat package: Mochi sample

- Same features, tighter code: ground / mochi shading blends by the
  smooth-union factor (no seam at the neck), LOD-scaled normal epsilon,
  value-noise ground, soft contact shadow, light / fog as material
  properties; the collider owns `blendK` / `groundK` and pushes them to
  the material, hand state is indexed by hand, settle is frame-rate
  independent. Host-verified (glslang HLSL, .NET compile, `EvaluateSdf`
  vs `alice_sdf::eval` parity 6e-8 on 1521 points); not compiled in
  Unity here (see `vrchat-package/CHANGELOG.md`).

## [v3.0.0] - 2026-09-16

Every node kind is transpiled, `ShaderLang` is sealed (the reason for the
major bump), IFS / skinning made sound on the CPU.

### Changed — breaking: `ShaderLang` is sealed

- `compiled::transpiler_common::ShaderLang` is now sealed, like `Real`
  since 2.0: only `WgslLang`, `GlslLang` and `HlslLang` implement it. It
  was left unsealed in 2.0 by oversight; it gains a required method every
  time a node kind needs new syntax (four in this release), and an
  implementation has to supply every helper the emitted laws reference,
  so external implementations were never workable. cargo-semver-checks
  flags both the new required methods and the sealing as major, hence
  3.0.0 rather than 2.2.0. No dependent on crates.io or in the ALICE
  repositories implements the trait; consumers of `to_wgsl` /
  `to_glsl` / `to_hlsl` / `GpuEvaluator` are unaffected.

### Added — every node kind is transpiled (IFS, skinning, lattice, heightmap)

- The four kinds the shaders used to pass through unchanged are now emitted
  in WGSL / GLSL / HLSL: `IFS` (transforms and iterations unrolled as
  literals, `ifs_fold_with_scale` law, child ÷ accumulated scale),
  `SdfSkinning` (each bone's two column-major transforms, weighted mean),
  `LatticeDeform` (control points as a module-scope array, the FFD as a
  module-scope function called three times for the central-difference
  correction), `HeightmapDisplacement` (the map as a module-scope array,
  dominant-axis projection, bilinear sample). `shader_unsupported_nodes`
  now returns an empty list; the corpus oracles cover all 144 nodes.
- `ShaderLang` gained `cast_int` / `decl_int` / `global_float_array` /
  `global_vec3_fn`; `GenericTranspiler::globals` collects module-scope
  declarations that each language emits before `sdf_eval`.

### Fixed — IFS / skinning on the CPU (found by non-identity corpus entries)

- The corpus IFS and skinning entries used identity matrices, so nothing
  had checked the laws with real transforms. With a scale / rotate IFS and
  a two-bone skin: the scalar / SIMD / BVH VM dropped the IFS scale
  correction the tree applies (`/ max(scale, 1e-6)`, now applied at
  `PopTransform`); the interval evaluator treated both as identity maps
  (skinning is one affine map `A p + b` — pushed through exactly; IFS
  uses the hull of the box and its images, divided by the scale range);
  `eval_lipschitz` claimed `2 · L(child)` for IFS (unsound by 280× — the
  nearest-image choice jumps, so it is `INFINITY` like domain repetition,
  pinned set 14 → 16) and `L(child)` for skinning (now `L(child) · ‖A‖`).

## [v2.1.0] - 2026-09-16

Corpus-wide GPU execution oracle (WGSL + GLSL) and the 15 shader laws it
caught, rounded cylinder fixed on the CPU, step budget scaled with the
Lipschitz bound (the gyroid "8.3 % miss"), a slimmer scalar-VM frame, one
VRChat shader source.

### Changed — VRChat package: one shader source

- `vrchat-package/Runtime/Shaders/` is the only copy; the legacy
  `Assets/AliceSDF/Shaders/` fork (which alone had the PBR surface and the
  material-id ops) is merged into it and deleted, together with the June
  `.unitypackage` snapshot. Merged by hand, not yet compiled in Unity
  (see `vrchat-package/CHANGELOG.md`).

### Fixed — 15 shader laws that differed from the CPU (found by the new corpus GPU oracle)

- `tests/test_gpu_law_parity.rs` now runs **every corpus node** through the
  WGSL path *and* through the GLSL path (`GpuEvaluator::from_glsl_compute`,
  wgpu's `glsl` feature / naga glsl-in), on 1024 random points each. The
  hand-picked WGSL tests had covered the laws touched by specific fixes;
  the blanket run found 16 of 142 nodes drifting on WGSL (up to 3.4) and
  18 on GLSL. Fixed to the CPU law in all three transpilers: heart (the
  shaders had a different implicit-cubic heart), pie, vesica (axis and
  sign), box frame (typos in the `max` operands), lidinoid / IWP / FRD
  (different surfaces), columns union / intersection / subtraction (one
  helper per language, `hg_sdf` law; the GLSL helper also used `half`,
  a reserved word), bend (rotation sign), extrude (2-D child in XY, slab
  on Z), displacement (frequency 5, not 10), octant mirror (abs *and*
  sort), icosahedral symmetry (was a pass-through; now the fold).
- **Rounded cylinder was wrong on the CPU**: `sdf_rounded_cylinder` and
  the SIMD table carried IQ's `− 2·ra` literally, so `radius = 0.4`
  rendered as 0.8 on the CPU while the shaders (and the docs) meant 0.4.
  CPU and BVH AABB now use `ρ − radius + round_radius`.
- `compiled::shader_unsupported_nodes(&node)` / `SHADER_UNSUPPORTED`: the
  four node kinds the transpilers pass through unchanged (LatticeDeform,
  HeightmapDisplacement, SdfSkinning, IFS — per-node data with no shader
  binding); the emitted shader carries a comment, the oracles skip them.
- Exact ties (a sector boundary at `atan2 = π`, a columns cell boundary)
  are platform-dependent on the GPU: the corpus oracles use random points,
  `repeat_laws_gpu_match_cpu_at_ties` keeps pinning the WGSL tie behaviour.

### Changed — scalar VM transform frame slimmed

- `eval_compiled` (scalar) pushed opcode + 4 params + aux window on every
  transform; the frame is now the point plus the pushing instruction's
  index, read back at `PopTransform`. Against the tree walker on CSG
  scenes (release, single point): 5 primitives 1.02×, 10 → 1.19×
  (was 1.35×), 20 → 1.17×, 40 → 0.91× (was 1.05×). The 9/16 self-review's
  "still 30 % slower" is this push / pop pair per transform; the scalar VM
  is the front of the SIMD batch path (`eval_compiled_batch_simd`, ~4×
  the tree), which is where compiled evaluation pays.

### Changed — step budget scales with the Lipschitz bound

- `RaymarchConfig::with_bound` multiplies `max_steps` by the bound (steps
  are `d / L`, so the same count covers `1 / L` of the distance) and the
  default budget is 256 (was 128); `high_quality` 512; `relaxed` uses the
  same scaled budget. The 9/16 self-review's "gyroid: 8.3 % of rays lost,
  unchanged in 2.0.0" was budget exhaustion, not a law problem: on 3000
  random rays through `gyroid(1.0, 0.1)` the 1.x default (128 steps at
  L = √3) lost 0.7 %, 444 steps lose 0, and the only case left
  (`gyroid(2.0, 0.1)`: 2 / 2540) is a ray grazing the shell and creeping by
  ≈ ε per step, which 4096 steps resolve. Pinned by
  `tpms_default_budget_random_rays`. A ray that stops inside the ε band
  while grazing (`f = 7e-6`) is a hit by definition even when the first
  sign change is further along.

## [v2.0.0] - 2026-09-16

Taper is a distance bound (with a phantom-free singular plane), the four
breaking changes deferred through 1.x (sealed `Real`, private `CompiledSdf`
fields, `dep:` features, no `lazy_static` feature), dual contouring without
fins, one noise law for texture fitting with GPU parity.

### Changed — **breaking**

- `SdfNode::Taper` gained `reach: [f32; 2]` (the child's `[r_xz, r_y]`
  from its AABB, computed by `SdfNode::taper`); hand-written `Taper { .. }`
  literals must add it (`[f32::INFINITY; 2]` = no cone bound). Serialized
  trees from 1.x load with that default.
- `compiled::real::Real` is sealed (`f32` / `f32x8` only, as documented
  since 1.12.0).
- `CompiledSdf` fields are private: `instructions()`, `aux_data()`,
  `node_count()`, `lipschitz()`; `#[non_exhaustive]` removed.
- Optional dependencies are enabled with `dep:` — `--features wgpu` /
  `clap` / `pyo3` / `numpy` / `cranelift-*` / `pollster` / `bytemuck` /
  `futures-channel` no longer exist (use `gpu` / `cli` / `python` / `jit`);
  `image` stays a named feature (heightmap import). The `lazy_static`
  compatibility feature is gone.
- `optimize`: a taper with factor 0 is dropped as the identity (it used to
  drop factor **1**, which is not the identity).

### Changed — taper is a distance bound (law change, same shape)

- `Taper` returned the child's distance at the tapered point, which is
  not a parent-space distance: default tracing lost 4.9 % of the rays on
  the shrinking side. It now returns `real::taper_bound`: the child
  distance divided by the Jacobian norm over a ball (the map is a
  perspective projection with centre `(0, 1/f, 0)`; the norm grows
  towards that plane, so the ball is capped at half the distance to it),
  combined with the signed distance to the cone ∩ slab that contains the
  shape (from the child's reach). The second term is what keeps the plane
  `y = 1/f` from becoming a phantom surface — the Jacobian term alone goes
  to 0 there and *every* ray crossing the plane stopped on it (508 / 508 in
  the new `taper_singular_plane_is_not_a_surface`). Same law on the tree
  evaluator, compiled scalar / SIMD, interval arithmetic and the three
  shader helpers (`alice_taper_bound`); GPU parity and naga validation
  cover it. `tests/test_relaxed_tracing.rs`: taper moved from the pinned
  (6 %) to the exact set with four scenes, including ones whose singular
  plane lies inside the ray box. `eval_lipschitz` stays `INFINITY` for
  taper (no finite global constant). The BVH AABB of a taper is the cone's
  box (`r_xz (1 + |f| r_y)`), which the old `expand(extent · |f|)`
  undershot for shapes larger than 1.

### Changed — texture-fit uses the crate's PCG noise; GPU parity of the emitted shader

- The texture module had its own value noise (`fract(sin(dot) · 43758.5)`)
  duplicated in the emitted WGSL / HLSL / GLSL under the same
  `hash_noise_3d` name as the SDF transpilers' helper — a second law that
  no GPU reproduces exactly and that clashed when both shaders were pasted
  together. It now uses `modifiers::hash_noise_3d` (PCG lattice hash) on
  the CPU (scalar and SIMD) and embeds the transpilers' own helper text
  (`modifiers::HASH_NOISE_{WGSL,GLSL,HLSL}`, now public). **Fits made
  before this version reconstruct differently and must be regenerated.**
- Oracle `tests/test_texture_shader_gpu_parity.rs` (CI `gpu-parity` job):
  the emitted WGSL rendered through `GpuEvaluator` matches `reconstruct`
  to 3e-7 over a 4-octave result with rotated and axis-aligned octaves.

### Fixed — dual contouring fins at grid-tangent surfaces

- Where the surface is tangent to a grid plane (torus inner equator, sphere
  or cylinder radius on a plane) the cells on both sides see a sign change
  and both get a dual vertex; the quads between the two rows were thin fins
  folded whichever diagonal was chosen, and pointed inward.
  `triangulate_quads` now collapses a quad that folds on both diagonals
  along its shorter pair of opposite edges (a local edge collapse inside the
  cell). `tests/test_dual_contouring_invariants.rs` no longer skips small
  triangles and checks torus / sphere / cylinder at five resolutions.

## [v1.13.0] - 2026-09-16

Oracle tests for the paths that had none (dual contouring, non-Lipschitz tracing, SVO ray query, NPR colour laws, neural SDF, texture fitting, the Python binding) and the fixes they found, plus a local `scripts/preflight.sh` that reproduces every CI gate before a push.

### Fixed — texture-fit (found by the new oracle; the feature had no CI test step)

- The scalar noise (`hash_noise_3d_cpu`, `eval_octave`) had drifted from
  the SIMD lanes: the 1.12.0 clippy pass rewrote it with `mul_add`, and
  `fract(sin(dot) · 43758.5)` turns a 1-ulp difference in `dot` into a
  different corner value (`test_simd_matches_scalar` had been failing;
  no CI step ran the `texture-fit` tests). Scalar and SIMD now share one
  operation order, `#[allow(clippy::suboptimal_flops)]` with the reason.
- The fitter could not recover a texture that *is* one octave of its own
  law (NMSE 0.63 from a single Nelder-Mead start: the DCT band index is a
  coarse frequency estimate and the cost is periodic in phase). Each
  octave now scans 4 frequency scales × 4 phase quadrants with a short
  budget and refines the best start with the full budget; the same
  texture fits to NMSE 0.08 / 29 dB in one octave.
- Padded SIMD lanes of the subsampled cost (sample counts that are not a
  multiple of 8) contributed `amp² · noise(phase)²` each; masked out.
- DC bias accumulated in f32 (third decimal off on large images); f64.
- `TextureFitConfig::tileable` is documented as having no effect (the
  hash lattice does not wrap); `FrequencyBand::frequency` is documented
  as the DCT-II index, not cycles per image.
- Public: `texture::{reconstruct, eval_octave, nelder_mead, OptimizeResult}`.
- Oracle `tests/test_texture_fit_oracle.rs`: Nelder-Mead on a quadratic
  bowl / Rosenbrock / monotonicity, `eval_octave` vs an independent
  evaluation, noise range and continuity, flat image → bias only,
  synthesized octave recovered with reported PSNR ≡ PSNR of `reconstruct`
  and NMSE ≡ MSE / Var, more octaves never worse, determinism, padded vs
  aligned grid, and naga validation of the emitted WGSL / GLSL. CI runs
  the `texture-fit` lib tests and this oracle; clippy covers the module.
  Pending: a GPU parity run of the emitted shader against `reconstruct`.

### Added — Python binding smoke oracle in CI

- `python/tests/smoke.py` + ci.yml `python-smoke` job (`maturin develop
  --features python`, no pytest): every assertion has a closed-form
  answer — unit sphere / box distances, `eval_batch` ≡ |p| − 1,
  compiled ≡ tree, mesh vertices on the sphere with outward winding and
  volume within 5 % of 4/3 π, JSON / `.asdf` round trips. The binding had
  no CI coverage before (release-wheels only builds it).

### Changed — `lazy_static` compatibility feature, local CI preflight

- The FFI registries use `std::sync::LazyLock`; the `lazy_static` optional
  dependency is gone. Because an optional dependency is an implicit public
  feature, the name stays as an empty `lazy_static` feature that `ffi`
  still enables (cargo-semver-checks `feature_missing` /
  `feature_no_longer_enables_feature` are major breaks). Removed in 2.0.
- `scripts/preflight.sh`: every hard gate of ci.yml / security-audit.yml
  / fuzz.yml as the commands CI runs (actionlint, fmt, the three strict
  clippy sets plus an x86_64 cross-lint, MSRV 1.85 checks, feature builds,
  wasm32, rustdoc, semver-checks against crates.io, cargo-deny, machete,
  stub guard, fuzz build; `--quick` skips only the test suites). The
  pre-push hook runs it and blocks the push on failure; the semver break
  above reached CI because this file did not exist yet.

### Fixed — dual contouring (found by the new invariant oracle)

- **Every dual-contouring triangle was wound inward** (sphere res 32:
  0 outward / 3714 inward, signed volume −4.23): the orientation branch per
  edge axis was inverted, the mirror image of the marching-cubes finding.
  Non-planar quads are now split along the diagonal whose two triangles both
  face the quad's mean vertex normal (a fixed diagonal folded triangles on
  the inner ring of a torus), and "inside" is `d < 0` everywhere so a face
  lying exactly on a grid plane no longer produces in-plane quads with an
  arbitrary orientation (the CSG-subtract box lost triangles that way).
  Oracle: `tests/test_dual_contouring_invariants.rs` — outward winding,
  closed, vertices on the surface, signed volume within 5 % of the analytic
  sphere / torus, and the sharp-feature property (box vertices on its faces,
  corners within a quarter cell, volume error below marching cubes').
  Known residue: ~0.5 %-of-a-cell slivers where a surface is tangent to a
  grid plane (documented in the test, QEF clamping is backlog).

### Fixed — sparse voxel octree ray query (found by the new oracle)

- `SparseVoxelOctree::ray_query` sphere-traced with the raw node distance
  (sampled at the node centre, so up to a half diagonal too large inside
  the node) and declared a hit only at `|dist| < 0.001` on a
  piecewise-constant field — 59 of 256 rays through a CSG scene overshot
  and were lost. It now steps by `dist − half_diag(leaf)` (a safe bound for
  a 1-Lipschitz field), treats "within a half diagonal" as the surface
  lying in this leaf and locates the crossing by bisecting the sign of the
  query: 0 misses, every hit within two finest leaves of the analytic
  crossing. Oracle: `tests/test_svo_query_oracle.rs` (query error bounded
  by the leaf size derived from the subdivision rule, error decreasing with
  depth, ray query vs scan, linearisation preserving every node).
- `ffi` registries use `std::sync::LazyLock`; the `lazy_static` dependency
  is gone (the `ffi` feature keeps its name).
- CI: the strict clippy job lints every feature that builds on Linux (the
  crate-wide policy only covered the default + shader set before); the
  feature-gated SVO oracle runs in the test matrix.

### Changed — neural SDF default learning rate (found by the new oracle)

- `NeuralSdfConfig::default().learning_rate` is 1e-2 (was 1e-3). Measured
  against the analytic unit sphere on held-out points, the old default left
  the network at RMSE 0.26 after its 100 epochs (a quarter of the radius,
  11 % of points far from the surface with the wrong sign); 1e-2 reaches
  0.08 in the same time and 0.03 at 300 epochs. Oracle:
  `tests/test_neural_oracle.rs` (RMSE vs analytic, sign agreement, more
  epochs not worse, seed determinism, lossless save / load, `eval_batch` ≡
  `eval`).
- NPR colour laws: `tests/test_npr_analytic.rs` — exact toon band levels,
  ramp / rim / vignette / outline endpoints and monotonicity, posterize
  idempotence, palette endpoints, compiled pipeline ≡ closed-form
  composition (all laws were already correct).

### Added — tracing oracle for the non-Lipschitz laws

- `tests/test_relaxed_tracing.rs::non_lipschitz_laws_default_tracing`:
  domain repetition (`RepeatInfinite` / `RepeatFinite` / `PolarRepeat`) of a
  child symmetric inside its cell traces with 0 mismatches against the scan
  oracle (the common case is a distance field even though `eval_lipschitz`
  cannot prove it); an off-centre repeated child and a tapered box are
  pinned at documented miss-rate ceilings (taper over-estimates on its
  shrinking side, 4.9 % of rays — making the taper law a bound by dividing
  by the local Jacobian norm is backlog).

## [v1.12.0] - 2026-09-15

Bridge features back on crates.io (P15), crate-wide clippy pedantic + nursery policy, every CI test step a hard gate (blocking fuzz seed replay, semver-checks), colour-program stack validation, and the evaluator / marcher / law work that followed the 2026-09-15 maintainer self-review.

### Fixed — panics reachable from untrusted input

- `GpuColorProgram::deserialize` accepted an unbalanced stack program
  (every opcode decoded, operands missing), which then panicked in
  `CompiledColorPipeline::eval`. `ColorOp::stack_effect` is the single
  `(pops, pushes)` table, `CompiledColorPipeline::validate` simulates the
  stack, and `deserialize` returns `DeserializeError::Stack` for an
  unbalanced program; `eval` documents that it panics on one (programs from
  `compile` / `deserialize` never are). The remaining `unwrap` / `expect`
  sites in production paths were audited: infallible `write!` to `String`,
  slices with a checked length, documented-panic APIs with `try_` twins.
- `DeserializeError` and the new `StackError` are `#[non_exhaustive]`
  (error enums grow as validation improves). Adding `Stack` to the exhaustive
  `DeserializeError` is the one-time break this release accepts; the two
  corresponding cargo-semver-checks lints are downgraded to warnings in
  `Cargo.toml` (removed after the 1.12.0 publish) so the CI gate stays hard
  for everything else.
- Texture optimiser / spectrum sorts use `total_cmp` (a NaN cost no longer
  panics the sort).
- `Real` is documented as implemented for `f32` / `f32x8` only (required
  methods are added as laws need them; a private `Sealed` supertrait comes
  with 2.0).

### Changed — clippy policy: pedantic + nursery for the whole crate

- `Cargo.toml [lints.clippy]` now sets `pedantic` and `nursery` to warn
  with every exception listed and justified there (the former `lib.rs`
  allow list moved into it, so tests / benches / examples share the bar),
  and CI's clippy jobs run with `-D warnings`. Landing the policy fixed
  ~480 lib findings: `use_self`, `const fn`, `midpoint`, `mul_add` at 100
  sites (演算の掟 §2-4), a never-read Vec in the IGES writer, contains +
  insert, a decimal bitmask, `hypot`, an integer loop for the stairs step
  index, complete `Debug` impls; two `suspicious_operation_groupings` sites
  are documented false positives.

### Changed — CI gates (review R2-7 follow-up)

- `Test (AAA meta)` and `cargo-semver-checks` are hard gates (both were
  `continue-on-error`). The AAA step had been hiding `npr::scene_composer`
  tests that built GLSL / HLSL sources without those transpiler features;
  each test is now gated on the feature it needs.
- Fuzz: the committed regression seeds are replayed in a dedicated blocking
  step (a known crash regressing fails the job). Until now the seed branch
  never ran in CI — the path was spelled `fuzz/seeds/…` from inside
  `fuzz/`, so the directory test was always false. The time-boxed
  exploration run and the coverage job stay informational by design
  (documented in the workflows).

### Added — bridge features restored (roadmap P15)

- `physics` (alice-physics 1.1), `codec` (alice-codec 0.1.2), `asp`
  (libasp 1.0) and `sdf-cache` (alice-cache 0.2) resolve to the sibling
  crates on crates.io again — they were removed for the 1.7.7 publish while
  those crates were path-only. API drift since then was two `Result`s in
  the codec quantiser (buffers are sized to each other, so the error is
  impossible and is `expect`ed with that invariant) and a missing doc
  comment; the redundant `unsafe impl Send / Sync for CompiledSdfField` is
  gone (the type derives both). `font` stays an inert gate until alice-font
  publishes. New: ASP I-packet round-trip test.
- CI: a `bridges` job builds and tests each bridge feature on its own with
  the real crates.io dependencies, and the main test job runs all four
  together as a hard gate (that step was `continue-on-error` against
  features that did not exist). The `alice-stubs` action and every stub
  workaround in the workflows are removed — there are no path
  dependencies left to satisfy.

## [v1.11.0] - 2026-09-15

Maintainer self-review landing (two rounds, independent Linux x86_64 environment, 2026-09-15): every finding fixed with an oracle test on its path; the Lipschitz bound is applied by every marcher; seven primitive laws are now exact. Shape changes (Egg apex, Horseshoe legs, BlobbyCross arms) and the marching-cubes index order flip are listed under Fixed.

### Added

- `fuzz/fuzz_targets/fuzz_eval_parity.rs`: builds arbitrary primitive / CSG /
  modifier trees and asserts tree ≡ compiled scalar ≡ compiled SIMD at random
  points plus cell / sector / base-plane ties; `fuzz/seeds/<target>/` keeps
  every crash it found as a committed regression input that CI replays first.
- FFI: every exported `extern "C"` function (175) now runs through
  `ffi_guard`, which catches a Rust panic inside the call and returns the
  function's sentinel (null handle, `f32::MAX`, `0`, `false`,
  `SdfResult_Unknown`) instead of unwinding into the host — on Rust 1.81+ that
  unwind aborts Unity / Unreal / the Python interpreter. The message is kept
  per thread and read with the new `alice_sdf_last_error()` /
  `alice_sdf_clear_last_error()` (`include/alice_sdf.h` updated; no new
  `SdfResult` variant, the enum is exhaustive and 1.x stays semver-minor).
- `mesh::MeshInputError` and non-panicking `try_stripify`,
  `try_encode_index_buffer`, `try_encode_filter_oct_i16`,
  `try_decode_filter_oct_i16_in_place`, `try_encode_filter_quat_i16`; the
  existing functions keep their signature and forward to the `try_*` form
  (their `# Panics` contract is unchanged).

### Fixed — evaluation-path parity (found by the new `fuzz_eval_parity` target, 9 findings in its first hour)

- `Scale` / `ScaleNonUniform` in the compiled scalar, SIMD, BVH and JIT-SIMD
  paths multiplied every *leaf* distance by the factor; the tree, JIT-scalar
  and shader paths scale the blended result. The two agree only when every
  operator above the leaves is linear — `Scale(ExpSmoothUnion)` was 21% off,
  `Scale(SmoothUnion / Chamfer / Stairs / Round / Onion)` likewise. The scale
  is now applied when the `Scale` frame pops on every path.
- `Real::signum` is `x < 0 ? -1 : 1` on every path (`f32` used `f32::signum`,
  whose `-0.0 → -1` flipped the pyramid distance sign at its base centre
  against SIMD / JIT); the GLSL / WGSL / HLSL pyramid and hex-prism helpers
  emit the same conditional instead of `sign()` (0 at 0).
- Exponential smooth union / intersection / subtraction use the stable form
  `min(a, b) ∓ k·ln(1 + e^{-|a-b|/k})` on the CPU paths and in the shader
  text: the textbook `-k·ln(e^{-a/k} + e^{-b/k})` underflowed to `ln(0)` for
  `d ≫ k` (`+inf` on libm, NaN on the SIMD polynomial, `-ln(1e-10)·k` in the
  shaders' clamp).
- SIMD `atan2` is per-lane libm: the `wide` polynomial is a few ulp off and
  an odd polar-repeat count puts `atan2(0, -x) = π` exactly on a sector tie.
- Every repeat / polar / helix law is one function: the tree evaluator's
  `modifier_repeat_infinite` / `modifier_repeat_finite` / `modifier_polar_repeat`
  / `modifier_taper` and its `Rotate` arm now call the generic
  `compiled::real` laws (`p * (1 / s)` operand form, two-cross-product
  rotation) instead of keeping scalar copies that rounded differently by an
  ulp and crossed a cell / sector boundary (four nested polar repeats
  amplified it to a whole sector).
- `Taper` denominator `1 - f·y` is kept away from its singular plane
  (`|den| ≥ 1e-6`, sign preserved) on every path (the tree gave NaN, SIMD a
  finite value); the transpilers emitted `1 + f·y` — a *mirrored* taper — and
  multiplied the child distance by `den`, neither of which any CPU path does.
- Shader `RepeatFinite` clamped the cell index to `±count` instead of the CPU
  `±count/2` (twice the extent).
- `tests/test_gpu_law_parity.rs` (feature `gpu`, Metal-verified): taper,
  repeat, polar, pyramid / hex sign, scale-after-blend and exp-smooth laws
  agree with `eval` to 5e-7 relative on the GPU.

### Fixed — self-review 2026-09-15 (round 1: sphere tracing / Lipschitz / CI oracle)

- Over-relaxed sphere tracing (`RaymarchConfig::fast()` ω = 1.2,
  `RaymarchConfig::relaxed()` ω = 1.6, any `omega > 1`) never retreated: on an
  overshoot it advanced from the overshot position, left the ray behind the
  surface with `d < 0` and crawled `min_step` until `max_steps` — a unit
  sphere lost 76 % of its rays at ω = 1.6, a torus 86 %. The tree, compiled
  and JIT marchers (and `raymarch_detailed`, which ignored `omega`) now share
  one `RelaxedStepper` implementing Keinert et al. 2014: when the unbounding
  spheres of two consecutive samples do not overlap, or the sign of `d`
  flips, the ray retreats to the last safe point `t_prev + |d_prev| / L` and
  continues unrelaxed; the overshoot check runs before the `|d| < ε` hit test
  so a relaxed step that lands just past a thin feature is not reported as a
  hit. `raymarch_relaxed` / `raymarch_detailed` are now re-exported from
  `raycast` (they were unreachable dead code). Oracle:
  `tests/test_relaxed_tracing.rs` (24 × 24 rays vs a 0.5 mm scan + bisection,
  6 shapes × 4 configs × 3 paths, 0 hit/miss mismatches; relaxed tracing
  takes fewer steps than plain tracing at grazing incidence).
- `interval::eval_lipschitz` was unsound: the nine triply periodic minimal
  surfaces (Gyroid, Schwarz P, diamond, Neovius, Lidinoid, IWP, FRD,
  Fischer–Koch S, PMY) sat in the "exact SDF, L = 1" arm although they are
  implicit trigonometric functions with |∇F| up to 7 (Neovius) — so
  `RaymarchConfig::relaxed` stepped past their surface and Neovius / IWP could
  not be rendered; the noise / displacement bounds ignored the gradient of the
  offset field (`sin(5x)…` is ×5, sine displacement dropped its frequency,
  Perlin |∇| ≤ 3.5, value-noise fbm ≤ 3√3 per octave); chamfer / stairs /
  engrave (`(a + b)/√2`) and pipe (`√(a² + b²)`) are √2-Lipschitz, not 1; and
  the twist / bend factor assumed a radius of 10 with the wrong norm
  (`√(1 + v²)` instead of the shear singular value `v/2 + √(1 + v²/4)`) — it is
  now taken from the child's AABB (the plane / unbounded-child fallback keeps
  10). The bound is now defined on the exterior `{f ≥ 0}` (what sphere tracing
  needs) and every claim is analytic or a pinned numerical supremum; laws that
  are not Lipschitz there — domain repetition (`RepeatInfinite` / `RepeatFinite`
  / `PolarRepeat`) with an arbitrary child, `Taper` (singular plane),
  `ColumnsUnion` family and `LatticeDeform` (jumps), `HeightmapDisplacement`
  (dominant-axis switch), `SweepBezier` (nearest-parameter jumps), and the
  `Ellipsoid` / `Egg` / `Horseshoe` / `BlobbyCross` / `Stairs` / `Helix`
  primitives (their laws jump or grow unboundedly, see the follow-up entries)
  — return `f32::INFINITY` instead of a guess, and `RaymarchConfig::relaxed`
  falls back to plain tracing for them. Oracles:
  `lipschitz_claim_bounds_every_difference_quotient` (every corpus node,
  13 directions × 2 step sizes × 2600 points, difference quotient ≤ claim) and
  `lipschitz_claims_are_finite_where_the_law_is_lipschitz` in
  `tests/test_evaluator_opcode_parity.rs`; `tests/test_relaxed_tracing.rs::
  tpms_trace_correctly_with_lipschitz_bound` (six TPMS, 0 mismatches with the
  bound, Neovius / IWP demonstrably lose rays without it).
- `interval::eval_interval` returned `EVERYTHING` for the nine TPMS
  surfaces, so interval-based pruning silently did nothing on any scene
  containing one (review SDF-R2-5); they now use the centre sample ± L·ρ
  with the Lipschitz constants above (finite and sound, pinned by
  `tpms_intervals_are_finite_and_sound`).
- `RaymarchConfig::min_step` is now applied in field units (divided by
  `lipschitz` like the step itself). A floor in ray units moved a sample by up
  to `L·min_step` in field value and, for `L·min_step > ε`, carried it across
  the `|d| < ε` hit band into the interior (Neovius at L = 7 lost 16 % of its
  rays that way even with the correct bound). With `min_step ≤ epsilon` plain
  tracing can no longer overshoot at all.

### Added — Lipschitz bound applied by every marcher

- `CompiledSdf` is now `#[non_exhaustive]` (construct it with `compile` /
  `try_compile`; its fields stay readable). The new `lipschitz` field would
  otherwise have broken an exhaustive struct literal — no known consumer
  builds one, since `instructions` comes from the compiler — and the
  attribute keeps later fields semver-minor. Related: `Real` staying
  unsealed until 2.0 is tracked in the roadmap.

- `CompiledSdf::lipschitz` / `JitCompiledSdf::lipschitz()` record
  `eval_lipschitz(node)` at compile time, and `RaymarchConfig::with_bound`
  raises a configuration's `lipschitz` to a finite bound. The config-less
  entry points (`raymarch`, `raymarch_batch*`, `render_depth`,
  `render_normals`, `raymarch_compiled`, `raymarch_simd_8`, `raymarch_jit*`,
  the `*_with_config` compiled / JIT variants) now step by `d / L`, so a
  TPMS surface (L = √3 … 7) traces correctly without the caller knowing it
  is not a distance field: Neovius / IWP / Gyroid report 0 false hits
  against the scan oracle on every path (`default_entry_points_apply_the_
  lipschitz_bound`), where the plain `t += d` lost most rays. Trees with no
  finite bound keep stepping by `d`. `raymarch` on a tree computes the bound
  per call (a tree walk; twist / bend children cost an AABB pass) — the batch
  / render functions compute it once and the compiled marchers not at all.

### Fixed — laws that were not distance fields (found by the Lipschitz property test)

- `Egg` was a three-branch approximation that reported **positive**
  distances for interior points on the axis (`egg(1, 0.5)` at (0, 0.1, 0) =
  +0.9) and jumped by `ra − rb` across the origin. It is now Inigo Quilez's
  exact `sdEgg` (disc of radius `ra` below y = 0, arcs of radius
  `2(ra − rb)`, apex cap `rb` at `y = √3(ra − rb) + ra`) on the CPU and in
  the WGSL / GLSL / HLSL helpers (GPU ↔ CPU 4.2e-7); `eval_lipschitz` is 1
  again and the interval bounding sphere covers the apex. **Shape change**:
  the apex moved from `y = ra` to `y = √3(ra − rb) + ra`.

- `Horseshoe` mixed an `abs(qx)` leg mirror with the width / thickness
  terms and was not a distance field (difference quotients up to √2). It
  is now IQ's exact `sdHorseshoe` (band of half-width `width` around an
  arc of `radius` opened by `angle`, legs of `half_length`) extruded by
  `thickness`, mirrored in the three shader helpers (GPU ↔ CPU 3.0e-7);
  `eval_lipschitz` is 1 again. **Shape change**: legs now end flat at
  `half_length` and the band is symmetric about its centre line.

- `BlobbyCross` was a home-grown "sqrt blend" that jumped by up to 23× the
  sample spacing between its two regions. It is now IQ's exact
  `sdBlobbyCross` (nearest parameter on the parabola arms from the
  depressed cubic, `he = 0.5`) on `|xz| / size`, extruded along Y; CPU and
  the three shader helpers are mirrored (GPU ↔ CPU 8e-5, `pow` / `acos`
  domain), the interval arm is the exact-SDF form and `eval_lipschitz` is
  1. **Shape change**: arms are parabola segments reaching `±size`.

- `SweepBezier` found the nearest curve parameter with 5 samples + Newton on
  the CPU / SIMD paths and with 4 Newton steps from `t = 0.5` in the
  shaders — three different laws, all jumping between local minima
  (difference quotients 800× the sample spacing). The distance is now IQ's
  closed-form `sdBezier` (Cardano / trigonometric cubic roots, degenerate
  curve → segment) in `modifiers::sweep::bezier_distance_2d`, used per lane
  by the SIMD path and emitted as one `bezier_distance_2d` helper per shader
  language (GPU ↔ CPU 4.3e-5); `eval_lipschitz` is the child's bound.

- `Stairs` compared only the step boxes {si − 1, si, si + 1, sj} and so
  missed the nearest step for points above or beside the staircase, jumping
  by up to 84× the sample spacing where the candidate set changed. It now
  takes the exact minimum over all `n_steps` boxes (CPU and the three
  shader helpers, GPU ↔ CPU 2.4e-7); `eval_lipschitz` is 1.

- `Helix` measured the distance to the helix point at the query's own
  azimuth (an over-estimate, and undefined on the axis, where the field
  jumped by 15× the sample spacing). It now finds the true nearest curve
  point by Newton from the same-azimuth candidates of the three nearest
  wraps (brute-force agreement 2e-3 over three pitch / radius ratios,
  continuous on the axis, azimuth pinned to 0 there because GPU `atan2(0, 0)`
  is NaN); the three shader helpers mirror it (GPU ↔ CPU 1.7e-6) and
  `eval_lipschitz` is 1.

- `Ellipsoid` was Inigo Quilez's `k0·(k0 − 1)/k1` approximation: not a
  distance bound (its gradient grows like `(max r / min r)⁴` far from the
  surface, so sphere tracing could skip an anisotropic ellipsoid) and
  discontinuous at the centre. It is now the exact signed distance
  (Eberly's robust nearest-point algorithm: axes sorted, point folded into
  the first orthant, bisection for the Lagrange parameter with the
  lower-dimensional reductions for axis-plane queries), on the CPU, per
  lane in the SIMD path, and as one `sdf_ellipsoid` helper per shader
  language (GPU ↔ CPU 3.3e-7 including on-axis queries and a 10:1 flat
  ellipsoid); brute-force agreement 4e-3 (sampling-limited) and
  `eval_lipschitz` is 1. Cost: up to 64 bisection steps per evaluation on
  this primitive only.

### Changed — compiled evaluator speed (review SDF-R2-4)

- `eval_compiled` zero-filled its three evaluator stacks (≈ 3.4 KB for f32,
  ≈ 6 KB for the SIMD lanes) on every call, a ≈ 21 ns fixed cost that made
  the "recommended" scalar VM slower than the tree walker for anything
  under ~30 nodes (sphere 30 ns vs 8.6 ns, 20-node CSG 82 ns vs 63 ns). The
  stacks are now uninitialised slots written before they are read (the
  stack discipline of the bytecode; debug builds assert every read):
  sphere 4.0 ns, 20-node CSG 58 ns. The SIMD batch path shares the gain.
- `sdf_to_mesh` / `marching_cubes` evaluate the grid through the compiled
  SIMD batch evaluator (tree ≡ compiled by the parity corpus), falling back
  to the tree walker only for trees the compiler rejects: the 20-node scene
  at res 128 goes from 60 ms to 38 ms (bench `marching_cubes/complex`).

### Fixed — self-review 2026-09-15 (transpiler validation, found by the new naga oracle)

- The five GDF polyhedra (`Tetrahedron`, `Dodecahedron`, `Icosahedron`,
  `TruncatedOctahedron`, `TruncatedIcosahedron`) transpiled to a call of
  `sdf_<name>(...)` that no transpiler defined — every WGSL / GLSL / HLSL
  shader containing one failed to compile on the GPU. The helpers are now
  emitted in all three languages, mirroring `primitives::gdf_vectors`
  (GPU ↔ CPU ≤ 3.4e-7 on Metal).
- `ColumnsUnion` emitted a truncated declaration (`var d3_a2 = mi    var
  d3_m = …`) left behind by an abandoned string-building attempt; the arm
  now emits the law once. Its modulo was WGSL `%` / HLSL `fmod` (truncated,
  sign of the dividend) while the CPU law and GLSL `mod` are floor modulo —
  a whole column period of drift for negative operands; `modulo_expr` is
  floor modulo in every language.
- GLSL `PolarRepeat` emitted `atan2(y, x)`, which GLSL does not have
  (`atan(y, x)`); the walker now goes through `ShaderLang::atan2_expr`.
- Each transpiler's `generate_shader` kept a second, hand-copied table of
  helper sources that silently skipped unknown names (`_ => {}`) — that is
  how the polyhedra went missing. The single `helper_source` table is now
  the only source and an unregistered helper panics at transpile time.
- GPU marching cubes failed to build its Pass 3 pipeline on DX12 (Windows,
  FXC `X4505: Sum of temp registers and indexable temp registers exceeds
  limit of 4096`): the 4096-entry triangle table was a WGSL module `const`
  that naga's HLSL backend lowers into indexable temporaries. It is now a
  read-only storage buffer (`TRI_TABLE`, binding 5). Hidden until now by
  the `continue-on-error` on the shader test step.
- Oracle: `tests/test_transpiler_naga_validate.rs` parses **and validates**
  (`naga::valid::Validator`) the WGSL and GLSL of every corpus node
  (features `gpu` / `gpu,glsl`); the corpus moved to `tests/common/corpus.rs`
  so every integration test can share it.

### Fixed — self-review 2026-09-15 (round 2: marching cubes output)

- **Every marching-cubes triangle was wound inward** (CPU `marching_cubes` /
  `sdf_to_mesh`, the compiled and adaptive variants, and the GPU compute
  path): `CORNER_OFFSETS` numbered the cube with its second and third axes
  swapped relative to the Bourke / Lorensen edge and triangle tables, a
  mirror image of the table's cube. A unit sphere at res 32 had 3608 of 3608
  triangles facing inward and a signed volume of −4.088 (truth +4.189); STL
  facets stored the (correct) averaged vertex normal next to a contradicting
  vertex order, so slicers that use the winding read every export as an
  inside-out solid. The corner numbering now matches the tables (0–3 on the
  y = 0 face, 4–7 on y = 1) on both CPU and GPU. **Breaking for consumers
  that compensated for the flip** (e.g. rendered with front-face culling set
  to CW, or negated normals from `(b − a) × (c − a)`): mesh topology and
  vertex positions are unchanged, only the index order per triangle.
- `sdf_to_mesh` was not closed on grids aligned with the surface: a grid
  edge shared by four cells was interpolated from each cell's local endpoint
  order, so the four copies differed in their last bits and vertex
  deduplication could not merge them (16–64 open edges at res ≥ 64). Edge
  vertices are now interpolated from the lexicographically smaller corner in
  every cell (bit-identical), and triangles that collapse when a grid corner
  sits exactly on the iso-level are dropped after deduplication
  (`mesh::remove_degenerate_triangles`, also exported). Oracle:
  `tests/test_mesh_orientation.rs` — all triangles outward against ∇f,
  signed volume positive and within 5 % of the analytic sphere / torus,
  zero open edges, vertex count = distinct positions, STL round trip facet
  normal ∥ winding; the GPU variant runs under `--features gpu-mesh`
  (Metal-verified).

### Changed

- FFI handle registries tolerate a poisoned lock (`PoisonError::into_inner`):
  a caught panic while a registry lock was held no longer turns every later
  FFI call into an error.
- CI: the `unity` / `unreal` meta-feature builds are hard gates
  (`continue-on-error` removed).
- CI: new `gpu-parity` job (ubuntu + Mesa lavapipe software Vulkan) runs
  the GPU ↔ CPU law / noise parity tests, the naga shader validation and
  the GPU marching-cubes orientation test with `ALICE_SDF_REQUIRE_GPU=1`,
  which turns "no adapter → skip" into a failure. Until now the only
  correctness oracle for the transpilers had never executed in CI; the
  `Test (shader transpilers)` step is also a hard gate (its
  `continue-on-error` is removed).

## [v1.10.3] - 2026-09-15

### Fixed

- Cell-boundary rounding now agrees on every evaluation path. The repeat /
  polar-repeat / helix laws snapped with `round`, whose tie direction differs
  per path (`f32::round` ties away from zero; `wide::f32x8::round` ties to even
  on AVX / NEON but away from zero on the SSE2 fallback; Cranelift `nearest`,
  WGSL `round` tie to even; GLSL `round` is implementation-defined; HLSL
  `round` ties away). A point on a cell boundary — every marching-cubes grid
  whose step divides the spacing has them — was folded into a different cell
  per path, changing the distance by a whole cell (1.2 for
  `sphere(0.3).translate(0.6,0,0).repeat_infinite(2,2,2)` at x = ±1: scalar
  1.3 / SIMD 0.1 / JIT 1.3). The canonical rule is now `floor(x + 0.5)`
  (`crispy::round_half_up`, `Real::round_half_up`) in the tree evaluator, the
  generic scalar / SIMD stack machine, the interval evaluator, both JITs and
  the GLSL / WGSL / HLSL transpilers. `Real::round` is unchanged but documented
  as not path-safe at ties.
- `PolarRepeat` tree evaluation used a different law from the compiled paths
  (`%` fold with a `+100·sector` offset) and picked a different sector at exact
  sector boundaries; it now delegates to the same `sector` / `count / TAU`
  round-trick law the compiler bakes into the instruction, and the shader
  transpilers snap with `angle * (n / TAU)` (same operands) instead of
  `angle / sector`.
- Parity corpus: tie sample points and offset (asymmetric) repeat children
  added (`tests/test_evaluator_opcode_parity.rs`); new
  `tests/test_round_tie_parity.rs` pins tree / compiled / SIMD / JIT agreement
  at cell boundaries and asserts the shader text uses `floor(x + 0.5)`.
- Test gating: `tests/noise_shader_validate.rs` and the two WGSL
  `npr::scene_composer` tests require the `gpu` feature (they use
  `WgslShader`) and are now gated on it, so `--features glsl,hlsl` without
  `gpu` compiles and passes.

### Changed

- `rust-version` corrected from `1.75` to `1.85`: the declared MSRV had been
  false since the lockfile picked up `clap_lex 1.1.0` (edition 2024), so
  `cargo check` on 1.75 failed at manifest parsing for the default `cli`
  feature. 1.85 is verified for the default and docs.rs feature sets; a CI
  `msrv` job now pins the declared toolchain.
- docs.rs builds with `glsl, hlsl, jit, svo, terrain, destruction, gi, ffi`
  (`[package.metadata.docs.rs]`); previously only the default feature was
  documented, hiding the transpilers, the JIT and the AAA modules.
- Cargo.toml `description` no longer references 1.7.7 / 1.8.0 for the bridge
  features (they remain path / git only).
- CI: strict clippy runs with `--all-targets` (tests, benches, examples).
- `resolver = "3"` (MSRV-aware dependency resolution, cargo 1.84+): `cargo
  update` no longer selects dependencies whose `rust-version` exceeds the
  crate's, so the lockfile cannot drift away from the declared MSRV again.
- Two `Option::map_or(true, ..)` sites rewritten as `is_none_or` — clippy's
  `unnecessary_map_or` had been silenced by the false 1.75 MSRV.

## [v1.10.2] - 2026-09-14

### Fixed

- `SdfNode` now drops iteratively (`src/types/drop.rs`): children are moved onto an explicit heap stack and released one `Arc` at a time, so freeing a tree no longer recurses once per level. A 2,400-deep `subtract` nest (the shape ALICE-LOL's stdlib products produce) overflowed a 2 MB thread stack on drop; the regression test now builds and drops 100,000-deep chains in a 256 KB thread. Shared subtrees (`Arc::clone`) are left to their last owner as before. Recursive `clone` / `node_count` / evaluators are unchanged.
- crates.io metadata: `homepage` / `documentation` (docs.rs) added to `Cargo.toml` — the crates.io page had no Documentation link before 1.10.2 (published 2026-09-15 together with 1.10.0 / 1.10.1 changes).
- README: bridge-feature notes no longer reference "v1.7.7 / v1.8.0"; bridges remain unavailable on crates.io releases (1.7.7 → 1.10.x).

## [v1.10.1] - 2026-09-14

### Changed — 1.10 Phase 2 (G1 + G2): one law per basic primitive and CSG operator

- The 13 basic primitives (`sphere` / `box3d` / `cylinder` / `torus` / `plane` / `capsule` / `cone` / `ellipsoid` / `rounded_cone` / `pyramid` / `octahedron` / `hex_prism` / `link`) and all 24 CSG binary operators (`union` … `tongue`, smooth / chamfer / stairs / columns families, `exp_smooth_*`) now have a single generic body `sdf_x_r<R: Real>` in `primitives::*` / `operations::*`. The existing scalar functions (`sdf_sphere(Vec3, f32)` etc.) are unchanged in signature and delegate to the generic law; the scalar and SIMD evaluator tables call the same generic function. Branches became `Real::select` (`cone`, `rounded_cone`, `pyramid`, `octahedron`, `ellipsoid` centre, `columns` early-out), `hypot` became `sqrt(x²+y²)` (≤ 1 ulp difference).
- Removed the SIMD-only `smooth_min_simd_rk` / `chamfer_min_simd` / `stairs_min_simd` / `eval_per_lane_binary` helpers (the generic laws replace them; `columns_*` and `exp_smooth_*` are now SIMD-native instead of per-lane).
- Added `Real::signum`.

## [v1.10.0] - 2026-09-14

### Changed — 1.10 Phase 1: one stack machine for scalar and SIMD

- **`compiled::real::Real`** — new scalar abstraction implemented for `f32` and `wide::f32x8` (`sqrt` / `abs` / `floor` / `round` / `min` / `max` / `sin_cos` / `atan2` / `exp` / `ln` / comparisons + `select` / per-lane `map` escape hatches) with `Vec3R<R>`. Every transform, modifier and post-processing law of the bytecode evaluator now exists once, as a generic function in `compiled::real` (`rotate_inverse` / `twist` / `bend` / `repeat_*` / `elongate` / `mirror` / `octant_mirror` / `revolution` / `extrude_*` / `taper` / `polar_repeat` / `shear` / `sweep_bezier` / `exp_smooth_*`), unit-tested against the canonical `modifiers::*` laws on both instantiations.
- **`compiled::eval_core::eval_bytecode<R>`** — the stack machine is generic; `eval_compiled` is `eval_bytecode::<f32>`, `eval_compiled_simd` is `eval_bytecode::<f32x8>`, `eval_compiled_bvh` shares the `f32` instantiation. The 2,250-line hand-written SIMD evaluator and the 1,400-line scalar evaluator are gone; the only per-instantiation code left is the leaf-primitive / CSG-binary law table (`compiled::prim_table::PrimTable`, bodies moved verbatim into `prim_table_scalar.rs` / `prim_table_simd.rs`). Phase 2 folds those into generic `sdf_x<R: Real>` laws.
- SIMD `LatticeDeform` now evaluates the lattice once per lane instead of twice (the per-lane escape returns point and Jacobian together).
- Performance (Apple Silicon, A/B against 1.9.2, min of 3 interleaved rounds): the generic evaluator is faster on every compiled path — `sphere` 38.4 → 30.2 ns (−21%), `translate×5` −17%, `rotate×5` −11%, `round×5` −13%, `twist×5` −9%, BVH sparse scene −5…−11%, 8-lane SIMD `translate×5` −9%, `twist×5` −23%, SoA 10k −5%. Getting there required three fixes worth recording: push frames as a direct struct literal (no temporary), leave the value stack untouched at `PopTransform` for point-only frames, and keep per-lane frame data (Extrude's z, LatticeDeform's Jacobian) in a side array instead of inflating every frame by two `R`s. New `benches/frame_cost.rs` guards exactly this.
- Public API unchanged: `eval_compiled*`, `eval_compiled_simd`, `eval_compiled_batch_simd(_parallel)`, `eval_gradient_simd`, `eval_distance_and_gradient_simd`, `Vec3x8`, `Quatx8`, the SoA entry points and the BVH entry points keep their signatures. `compiled::real` is public so downstream code can write `Real`-generic SDF laws.

### Fixed

- `Noise` evaluated Perlin gradient noise on the CPU / SIMD / bytecode paths but the shader transpilers emitted `hash_noise_3d` value noise. The transpilers now emit `perlin_noise_3d` — a verbatim port of `modifiers::perlin_noise_3d` (xor-multiply lattice hash of the `i32` cell coordinates and seed, 16-entry gradient LUT, quintic fade) — in GLSL / WGSL / HLSL, so every path renders the same Perlin field. `tests/test_gpu_noise_parity.rs` (feature `gpu`) measures |GPU − CPU| ≤ 7.2e-7 for `Noise` as well; `tests/noise_shader_validate.rs` (feature `glsl`) parses the generated WGSL and GLSL with naga. Shader-rendered `Noise` patterns change from value noise to the intended Perlin.
- `SurfaceRoughness` evaluated a different noise on every path: the CPU used its own `sin`-hash value-noise fbm (amplitude 0.5, per-octave rotation) while the GLSL / WGSL / HLSL `hash_noise_3d` helpers used a `sin`-hash fbm with amplitude 1.0 and no rotation, and `sin` of large arguments differs between GPU and CPU anyway. The noise law now exists once: `modifiers::surface_roughness::{hash_noise_3d, fbm}` (PCG lattice hash over the corner bits + seed, trilinear smoothstep blend, `2v - 1`; fbm = `Σ 0.5^i · noise(p · 2^i, 42)`) and all three shader helpers emit the identical function with `floatBitsToUint` / `bitcast` / `asuint`. `tests/test_gpu_noise_parity.rs` (feature `gpu`, skips without an adapter) measures |GPU − CPU| ≤ 7.2e-7 on 2048 points. The interval widening is `|amplitude| · (2 − 2^(1 − octaves))`. Realised roughness patterns change (statistics unchanged); the `Noise` node still pairs CPU Perlin with shader value noise and is tracked separately.
- `interval::eval_interval` violated its own contract (`lo ≤ sdf(p) ≤ hi`) for 32 node kinds — 2,045 violations on the 120-node parity corpus. Bounding-sphere lower bounds assumed a radius the shape exceeded or an exact distance the evaluation does not provide (`RoundedCylinder`, `HexPrism`, `Tube`, `TriangularPrism`, `Tetrahedron` / `Dodecahedron` / `Icosahedron` / `Truncated*`, `BoxFrame`, `Ellipsoid`, the extruded 2D shapes, …), `Taper` / `Shear` / `RepeatFinite` / `OctantMirror` / `IcosahedralSymmetry` / `LatticeDeform` mis-modelled their domain maps, and `Chamfer*` / `Stairs*` / `ExpSmooth*` under-estimated the blend widening. Arms now use a Lipschitz form (`eval(centre) ± L·half_diagonal`, sound for every 1-Lipschitz law), exact interval arithmetic on the actual formulas (`Superellipsoid`, `BlobbyCross`, `Tunnel`, `Helix`, chamfer), or interval images of the domain maps. `tests/test_evaluator_opcode_parity.rs::interval_eval_contains_point_values` pins the contract on the whole corpus; `analytic_gradient_matches_numerical` does the same for `eval::gradient` (already sound).
- `modifiers::surface_roughness` hash was `fract(sin(h)) * 43758` (±43758) instead of `fract(sin(h) * 43758)`, and `value_noise` hid it with a trailing `fract()` that turned the "smooth value noise" into a discontinuous sawtooth in (-1, 1); fbm therefore left `[-1, 1]` and the roughness interval was unsound. Fixed at the source (GLSL-style non-negative fract, no wrap); the shader paths already used their own `hash_noise_3d` fbm and are unaffected.
- `examples/gpu_eval.rs` did not build with `--features gpu` (`WgslShader::transpile` gained a `mode` argument). CI now builds the feature-gated examples (`cargo build --examples --features "glsl,hlsl,gpu"`) so example API drift fails fast.

## [v1.9.2] - 2026-09-14

**One law per opcode, every path** — closes the follow-ups left open by 1.9.1: the shader / JIT backends now share the CPU sign convention, the BVH is an annotation pass over the main compiler instead of a third hand-copied compiler, the JIT SIMD compilers fail loudly instead of emitting `f32::MAX`, and the parity corpus covers six evaluation paths plus an AABB-conservativeness oracle. First crates.io release since 1.9.0 (1.9.1 was never published; its notes below are included).

### Fixed

- **`Plane` sign in GLSL / WGSL / HLSL / BlinkScript transpilers and the Cranelift JITs** (`dot(p, n) + d`) now matches `sdf_plane` and every CPU evaluator (`dot(p, n) - d`, "distance from origin"). **Shader output changes for scenes using `Plane`** — flip the sign of `distance` if you relied on the old convention.
- **`sdf_regular_polygon` was unbounded**: the sector fold only handled `z >= 0` and the 2D term was a half-plane distance, so half the plane evaluated as "inside". Replaced with the exact regular-polygon law (circumradius `radius`, XZ plane) in Rust and in all three shader helper libraries. Caught by the new AABB-conservativeness test.
- **`LatticeDeform` outside its bounding box**: points were clamped to the boundary (every outside point mapped to the same deformed point) and the `0.1` Jacobian floor inflated distances 10x. Outside points now pass through unchanged with correction `1.0` (standard FFD). Tree result changes for scenes evaluating `LatticeDeform` outside the lattice.
- **BVH compiler was a third hand-copy of the compile law** (68 arms, no AABB laws for 60+ primitives, so it rejected them). `CompiledSdfBvh::try_compile` now reuses `CompiledSdf::try_compile` and computes AABBs with the instruction-driven walker in `compiled::refit` — every tree the main compiler accepts, the BVH accepts (the nine kinds 1.9.1 rejected included). `refit::primitive_aabb` / `csg_binary_aabb` / `transform_or_modifier_aabb` are exhaustive over `OpCode`.
- **`refit` scene AABB** was the union of every intermediate push (an inner child's untransformed bound leaked into the scene bound after `refit_all`); it is now the root's AABB.
- **`refit` `RoundedCone` AABB** ignored the end-cap spheres (`y` range is `[-hh - r1, hh + r2]`).
- **JIT SIMD (`jit::JitSimdSdf` / `JitSimdSdfDynamic`) silently pushed `f32::MAX` for the 82 opcodes without a codegen arm** and treated `Noise` as a no-op. `compile` now returns `Err("JIT SIMD: no codegen arm for opcode …")` so callers fall back to the interpreter instead of rendering nothing.
- **Cranelift JIT law drift** caught by the extended corpus: `Ellipsoid` centre (`0` instead of `-min(radii)`), `Engrave` (`0.5` instead of `1/√2`), `RepeatFinite` (clamped to `±count` instead of `±count/2`), `Bend` rotation sign, and unreduced degree-5 Taylor sin/cos (>10% error for `|x| > 2` in `Twist` / `Bend` / `PolarRepeat`) — both JITs now use range reduction to `[-π/2, π/2]` plus degree-9/8 series (≤ 3e-5 abs error).

### Added

- `compiled::OpKind` + `OpCode::kind()` — exhaustive stack-machine classification (`Primitive` / `Binary` / `Transform` / `Modifier` / `PopTransform` / `End`); `is_primitive` / `is_binary_op` / `is_transform` / `is_modifier` are now derived from it instead of numeric ranges.
- `CompiledSdfBvh::aux_data` — the BVH now carries the side buffer (heightmaps, lattices, bones, IFS matrices, polygon vertices). **Struct-literal constructors of `CompiledSdfBvh` outside the crate must add the field.**
- `tests/test_evaluator_opcode_parity.rs` now compares six paths (tree / scalar / SIMD / BVH / Cranelift JIT / JIT SIMD, the last two under `--features jit`) and adds `primitive_and_scene_aabbs_are_conservative` (grid-samples every corpus node: `sdf(p) ≤ 0 ⇒ p ∈ scene AABB`).

### Changed

- `compiled::jit_simd::JitSimd` is **deprecated** and is now a thin wrapper over `compiled::jit::JitSimdSdf` (it was a 2,500-line divergent copy). `compile` / `eval` / `eval_soa` keep their signatures.
- `refit::RefitError::UnsupportedOpcode` is no longer produced (every opcode has an AABB law); the variant is kept for API compatibility.

## [v1.9.1] - 2026-09-14

**Compiled evaluator parity** — every compiled evaluation path now executes the same law as the tree evaluator, and the opcode dispatch is exhaustive by construction.

### Fixed

- **Scalar / BVH evaluators silently mis-evaluated 11 / 18 opcodes.** `eval_compiled` lacked arms for `Circle2D` / `Rect2D` / `RoundedRect2D` / `Segment2D` / `Polygon2D` / `Annular2D` / `ExpSmoothUnion` / `ExpSmoothIntersection` / `ExpSmoothSubtraction` / `Shear` / `Animated`; `eval_compiled_bvh` additionally lacked `IFS` / `SdfSkinning` / `LatticeDeform` / `HeightmapDisplacement` / `IcosahedralSymmetry` / `ProjectiveTransform` / `SurfaceRoughness`. A `_ =>` fallback evaluated unknown primitives as a unit sphere, unknown binary ops as plain `min`, and unknown modifiers as a no-op (which for `Shear` then underflowed the coordinate stack at `PopTransform`). Both evaluators are now thin wrappers over a single exhaustive stack machine (`compiled::eval_scalar_core`), so a new `OpCode` without an evaluator arm is a compile error.
- **`Polygon2D` lost its vertices at compile time** and evaluated as a unit sphere on every compiled path. The compiler now serialises vertices into `aux_data` and all paths evaluate the real polygon.
- **`SineDisplacement` lost its frequency at compile time** (collapsed to the legacy `Displacement` law with frequency 5). `Instruction::displacement` now carries `[amplitude, fx, fy, fz]` and a new `Instruction::sine_displacement` preserves the per-axis frequency.
- **`LatticeDeform` ignored the Jacobian correction** on compiled paths; the tree law `eval(child, q) / correction` is now applied at `PopTransform` (per lane on SIMD).
- **BVH compiler silently replaced unsupported nodes with `sphere(0.001)`** (`IFS` / `SdfSkinning` / `LatticeDeform` / `HeightmapDisplacement` / `IcosahedralSymmetry` / `ProjectiveTransform` / `SurfaceRoughness` / `SineDisplacement` / `Terrain`). `CompiledSdfBvh::try_compile` now rejects them with `CompileError::UnsupportedPrimitive`; `BvhCompiler::compile_node` is exhaustive.
- **BVH compiler dropped `OctantMirror`** (compiled the child only). It now emits the modifier with a symmetric cube AABB.
- **`CompiledSdf` compiled `Terrain` to a silent sphere**; `try_compile` now rejects it with `UnsupportedPrimitive("Terrain")`.
- **SIMD (`eval_compiled_simd`) diverged from the scalar law** on: `Plane` (`dot + d` instead of `dot - d`), `Ellipsoid` (0 instead of `-min(radii)` at the centre), `RepeatFinite` (clamped to `±count` instead of `±count/2`), `HeightmapDisplacement` (added instead of subtracted the displacement, sine fallback when aux missing), `SurfaceRoughness` (ad-hoc FBM instead of `modifiers::surface_roughness`), `Segment2D` / `Polygon2D` (bounding-sphere fallback), `ExpSmooth*` (Schraudolph exp + a Padé ln whose linear term was ~2x off → up to 35% error), and every trig-based opcode (Bhaskara I sin/cos with 1.6e-3 abs error, 3.8e-3 rad atan2 — amplified to >10% on TPMS / `Twist` / `Bend` / `PolarRepeat`). SIMD now uses `wide`'s `sin` / `cos` / `atan2` and calls the shared scalar law per lane where no exact SIMD form exists.

- **`--features font` (and therefore `--all-features`) did not build on crates.io.** The empty `font` feature gated `font_bridge`, which imports the undeclared `alice_font` crate, so every user enabling it — and `cargo-semver-checks`, which enables all features — hit `unresolved import alice_font`. The feature is kept (removing it would be a semver-major break) but is now inert: `font_bridge` and the `text_to_3d_demo` example additionally require `--cfg alice_font_bridge` plus a local `alice-font` path dep.
- **`godot` feature did not build**: `to_glsl` called a non-existent `compiled::glsl::transpile_glsl`; it now uses `GlslShader::transpile(node, GlslTranspileMode::Hardcoded).source`.
- **`MeshRepair::repair_all` left non-manifold edges behind.** `merge_duplicate_vertices` can collapse two vertices of a sliver triangle (new zero-area face) and make two neighbouring slivers reference the same three vertices (duplicate face → edge with 4 incident triangles). The old order (`degenerate → merge → fix_normals`) never cleaned what the merge created — a marching-cubes `sphere(10)` at resolution 96 kept 96 non-manifold edges after repair (measured 2026-09-14 in the text-to-print 3MF export path). `repair_all` now runs `degenerate → merge → degenerate → remove_duplicate_triangles → fix_normals` and the sphere / table regression test asserts 0 non-manifold and 0 boundary edges.

### Added

- `MeshRepair::remove_duplicate_triangles` — drops winding-insensitive duplicate faces and index-collapsed triangles (keeps the first occurrence).
- `SdfNode::box3d_half_extents(hx, hy, hz)` — half-extent spelling of the plain box, matching `rounded_box` and the LOL DSL `box3d`. `box3d` (full dimensions) and `rounded_box` (half-extents) docs now state the asymmetry explicitly; neither signature changes.
- `primitives::{sdf_circle_2d, sdf_rect_2d, sdf_rounded_rect_2d, sdf_segment_2d, sdf_annular_2d, sdf_polygon_2d, sdf_polygon_2d_xy, sdf_polygon_2d_flat, extrude_2d}` — single-source 2D-extruded primitive laws used by tree / scalar / BVH / SIMD.
- `operations::{sdf_exp_smooth_union, sdf_exp_smooth_intersection, sdf_exp_smooth_subtraction}` — blend-width (`d/k`) exponential smooth laws (distinct from the rate-based `smooth_min_exp`).
- `modifiers::modifier_shear` — inverse shear law shared by all evaluators.
- `Instruction::sine_displacement(amplitude, fx, fy, fz)`.
- `tests/test_evaluator_opcode_parity.rs` — 120-node corpus covering every compilable `SdfNode` variant, compared across tree / scalar / SIMD / BVH at 8 sample points, plus a guard that the corpus reaches all 124 emitted opcodes and that unsupported nodes are rejected loudly.

### Changed

- `compiled::eval` and `compiled::eval_bvh` are now thin wrappers; the stack machine lives in `compiled::eval_scalar_core` (~2,300 lines of duplicated dispatch removed).
- `CompiledSdfBvh::try_compile` rejects the nine node kinds listed under Fixed (previously accepted and mis-compiled).

### Known limitations

- (both resolved in 1.9.2) The shader / JIT `Plane` sign and the `LatticeDeform` outside-bbox law.

## [v1.9.0] - 2026-09-13

**NPR compiled pipeline + SIMD batch + GPU bytecode** — Phase 12-D / 13 / 14 landing as additive minor bump on top of 1.8.0 NPR module foundation. Plus rustdoc broken-intra-doc-link fix and 6-issue clippy cleanup in test code.

### Added

- **Phase 14** — GPU bytecode serialisation and WGSL evaluator emitter. New public API:
  - `npr::compiled_color::gpu_opcode_tag` — stable `u32` tag constants for all 17 native opcodes (public so the WGSL evaluator's constants stay in lockstep with Rust)
  - `npr::compiled_color::gpu_palette_source_tag` — stable `u32` tag constants for `PaletteSource`
  - `npr::compiled_color::GpuColorProgram` — upload-ready flat `[u32]` bytecode stream with `as_words` / `byte_len` / `deserialize` (round-trip check that decodes the stream back into a `CompiledColorPipeline`)
  - `npr::compiled_color::SerializeError` / `DeserializeError` — non-panicking error surface (`Fallback` rejected up front; unknown opcode / truncated payload / unknown palette source detected on decode)
  - `npr::compiled_color::opcode_word_count` — payload-word count lookup keyed by opcode tag
  - `CompiledColorPipeline::serialize() -> Result<GpuColorProgram, SerializeError>` — encode the CPU-side opcode stream into a GPU-uploadable buffer
  - `npr::compiled_color::emit_wgsl_bytecode_evaluator() -> String` — canonical WGSL source that defines `AliceNprBytecodeCtx` + `alice_npr_eval_bytecode(program_len, ctx) -> vec3<f32>` (stack depth 32). The caller supplies `fn alice_npr_load(index: u32) -> u32`, decoupling the evaluator from any specific bind-group layout and avoiding the `unrestricted_pointer_parameters` WGSL extension.
- Round-trip tests: `serialize_all_native_variants_roundtrip` covers every native opcode; `serialize_deep_composition_roundtrip` covers a nine-level composition tree. Encode → decode → scalar `eval` matches the original pipeline lane-for-lane.
- Naga validation tests (new `tests/npr_bytecode_wgsl_validate.rs`): the emitted evaluator wrapped in a minimal fragment-shader entry point parses (`naga::front::wgsl::parse_str`) and passes full semantic validation (`naga::valid::Validator` with `ValidationFlags::all()`).
- **Phase 12-D** — `CompiledColorPipeline` native opcode coverage extended to all 17 current `NprColorNode` variants. New `ColorOp` variants: `Multiply` / `Add` / `OutlineOver` / `Fresnel` / `Saturate` / `Bloom` / `PosterizeColor` / `Vignette` / `Palette3` / `Palette5` / `Hatch` / `Tonemap` / `SpeedLine`. A well-formed pipeline compiled from any current DSL surface now contains zero `Fallback` opcodes; the `Fallback` opcode is preserved as a forward-compat seam for future variants.
- **Phase 13** — 8-lane SIMD batch evaluator via `wide::f32x8`. New public API:
  - `npr::compiled_color::NprColorBatch8` — SoA 8-lane RGB colour batch with `splat` / `from_vec3s` / `to_vec3s` / `lerp` / `scale` / `mul_componentwise` / `add_vec3x8` / `dot_scalar` / `max_channel`
  - `npr::compiled_color::NprBatchContext8` — SoA 8-lane shading context (derived scalars only: `n_dot_l` / `n_dot_v` / `sdf` / `uv_x` / `uv_y` / `time`), built from `[NprColorContext; 8]` via `from_contexts`
  - `CompiledColorPipeline::eval_batch8(&NprBatchContext8) -> NprColorBatch8` — evaluates the same bytecode across 8 lanes in parallel
  - SIMD-native path for 14 opcodes (`PushConstant` / `Toon` / `SoftToon` / `TwoTone` / `Multiply` / `Add` / `Scale` / `OutlineOver` / `Saturate` / `Bloom` / `PosterizeColor` / `Vignette` / `Palette3` / `Hatch` / `Tonemap`); per-lane scalar over the SoA batch for the remaining 3 (`Fresnel` uses `powf`, `SpeedLine` uses `atan2`, `Palette5` walks a 4-segment palette).
- Benchmarks: `bench_color_pipeline` gains `deep_composition_eval` / `deep_composition_compiled_eval` (6-level composition touching `Toon` + `OutlineOver` + `Fresnel` + `Vignette` + `Saturate` + `Tonemap`) plus P13's `toon_batch8_eval` / `toon_with_outline_batch8_eval` / `deep_composition_batch8_eval` per-call figures.
- Tests: `npr::compiled_color::tests` gains 16 native-opcode coverage tests plus `all_current_variants_compile_without_fallback` regression guard, and 19 batched tests (`vec3x8_from_vec3s_roundtrip` + `batch_matches_scalar_for_*` for every current variant + full-variant composition regression) asserting `eval_batch8` == 8 x `eval` per lane.

### Changed

- `npr::dsl::palette_source_scalar` promoted to `pub(crate)` so `compiled_color::ColorOp::{Palette3, Palette5}` can share the tree-eval scalar-source semantics.
- `prelude` re-exports `NprBatchContext8` and `NprColorBatch8` from `npr::compiled_color`.

### Performance (Apple Silicon, Phase 13)

Per-lane cost of the batched path (total time / 8):

| Pipeline | Tree eval (scalar) | Compiled scalar | Batch8 per-lane | Batch8 vs tree |
|----------|--------------------|-----------------|-----------------|----------------|
| `toon` shallow | 4.98 ns | 21.3 ns | 5.2 ns | ~1.04× (parity) |
| `toon` + outline | ~5.0 ns | 21.7 ns | 5.5 ns | ~1.10× |
| Deep 6-level composition | 18.2 ns | 28.9 ns | **11.4 ns** | **0.63×** |

The deep-composition case is the first regime where the compiled pipeline beats the tree walker outright. Shallow `toon` remains parity because opcode-fetch overhead dominates trivial arithmetic.

## [v1.8.0] - 2026-09-13

**NPR module landing** — a new procedural NPR (Non-Photorealistic Rendering) subsystem across 12 phases (P1 through P12-A) landing as `alice_sdf::npr`. The 5 bridge dependencies (`alice-codec` / `alice-physics` / `alice-cache` / `alice-font` / `alice-asp`) remain trimmed as in 1.7.7 because they have not yet been published to crates.io; scheduled restoration in a future release once upstream publishes.

### Added

- **`npr` module** — Procedural NPR (Non-Photorealistic Rendering) primitives across 9 categories, all closed-form and texture-free (Phase 2 Law-only compliant)
  - `npr::toon` — `toon_ramp`, `soft_toon_ramp`, `two_tone`, `posterize_color`
  - `npr::outline` — `distance_field_outline{,_soft}`, `curvature_outline`, `depth_step_outline`, `composite_outline`
  - `npr::sky` — `sky_gradient_bands`, `puffy_cloud_layer`, `distance_color_quantize`, `light_shaft_beam`, `sun_disc`
  - `npr::rim` — `fresnel_rim`, `procedural_matcap` (2x2 palette bilinear, no texture), `stylized_specular`
  - `npr::hatch` — `hatch_lines`, `cross_hatch`, `paper_grain`, `pencil_shade`
  - `npr::distortion` — `hand_drawn_jitter`, `sketch_wobble`, `line_boil`
  - `npr::palette` — `palette_gradient`, `time_of_day`, `season_palette`
  - `npr::composition` — `vignette`, `bloom_toon`, `chromatic_offsets`
  - `npr::motion` — `speed_line`, `impact_flash`
  - `npr::noise` — `NoiseField` trait + `HashNoise` deterministic hash-based value noise + `PerlinNoise` gradient noise + `WorleyNoise` cellular noise + `SimplexNoise` skewed-lattice gradient noise + `fbm` multi-octave composer
  - `npr::sdf_integration` — Adapters that consume `SdfNode` via `eval`, `eval_normal`, and `autodiff::mean_curvature`: `curvature_outline_from_node`, `distance_outline_from_node`, `toon_shade_from_node`, `soft_toon_shade_from_node`
  - `npr::dsl` — `NprColorNode` expression tree + `NprColorContext` for composing NPR primitives into a color pipeline
  - `npr::shader_glue` — Core (14 primitives) + palette (`sky_gradient_bands_3`, `palette_gradient_5`, `time_of_day`, `season_palette`) GLSL / WGSL / HLSL helper string constants + `helpers_for` / `palette_helpers_for` / `full_helpers_for(ShaderLanguage)` dispatch
- All NPR items re-exported from the `prelude` module
- `examples/npr_toon_demo.rs` — 9-category primitive tour
- `examples/npr_background_scene.rs` — Shadertoy-style raymarching background scene composing multiple NPR primitives
- `benches/npr_primitives.rs` — Criterion benchmarks across all 9 categories plus noise and DSL evaluation
- `npr::scene_composer::SceneShaderBuilder` (feature-gated: `glsl` / `hlsl` / `gpu`) — Builder that composes the NPR helper library, the transpiled SDF evaluator, and a canonical raymarching `main()` per shader language into a single shader source string
  - `.with_pipeline(NprColorNode)` — Replace the default `soft_toon + composite_outline` hit-branch colour block with a custom `NprColorNode` expression tree
- `npr::dsl_shader::transpile_npr_color_node` — Transpile an `NprColorNode` DSL tree into a shader-language snippet (`NprShaderSnippet`) usable across GLSL / WGSL / HLSL
- `NprColorNode` new variants: `Multiply` / `Add` / `Scale` / `Fresnel` / `Saturate` / `Bloom` / `PosterizeColor` / `Vignette` / `Palette3` with builder helpers (`.multiply`, `.plus`, `.scale`, `.with_fresnel`, `.saturate`, `.bloom`, `.posterize`, `.vignetted`)
- `PaletteSource` enum (`NDotL` / `NDotV` / `Sdf` / `UvY`) driving `Palette3`
- `NprColorContext.uv: Vec2` + `NprShaderContext.uv: &str` for UV-dependent variants
- `alice_saturate(color, factor)` added to `NPR_GLSL_HELPERS` / `NPR_WGSL_HELPERS` / `NPR_HLSL_HELPERS`
- `alice_palette_gradient_3(t, c0, c1, c2)` added to `NPR_*_PALETTE_HELPERS`
- `NprColorNode::Hatch { base, angle_rad, density, thickness, ink }` variant with `.with_hatch` builder helper
- `alice_hatch_lines(uv, angle_rad, density, thickness)` added to `NPR_GLSL_HELPERS` / `NPR_WGSL_HELPERS` / `NPR_HLSL_HELPERS`
- `NprColorNode::Palette5 { source, c0..c4 }` variant reusing `alice_palette_gradient_5`
- `NprColorNode::Tonemap { child, exposure }` variant with `.tonemap_reinhard` builder helper
- `NprColorNode::SpeedLine { base, focus, count, thickness, ink }` variant with `.with_speed_lines` builder helper
- `alice_tonemap_reinhard(color, exposure)` and `alice_speed_line(uv, focus, count, thickness)` added to `NPR_GLSL_HELPERS` / `NPR_WGSL_HELPERS` / `NPR_HLSL_HELPERS`
- `NprColorContext.time: f32` + `NprShaderContext.time: &str` (canonical `"iTime"`) for animation
- `PaletteSource::TimeCycle` — `fract(time)` driver for cyclic palettes
- `SceneShaderBuilder` shader output now declares an `iTime` uniform (`layout(binding=0) uniform SceneUniforms.iTime` in GLSL, `SceneUniforms.iTime` in WGSL aliased as `iTime` in `fs_main`, `cbuffer SceneCB.iTime` in HLSL)
- `npr::compiled_color::CompiledColorPipeline` — Host-side bytecode compilation of `NprColorNode` trees into a flat `ColorOp` stream evaluated by a small stack machine. Currently natively supports `Constant` / `Toon` / `SoftToon` / `TwoTone` / `Scale`; other variants use a transparent `Fallback` opcode that delegates to the recursive tree walker. Ships now to lock in the API ahead of SIMD / GPU integration; on shallow trees scalar bytecode is presently slower than tree eval (measured on Apple Silicon: 3.5 ns vs 19 ns for `toon`)
- `benches/npr_primitives.rs::bench_color_pipeline` gains `toon_compiled_eval` and `toon_with_outline_compiled_eval` benchmarks that compare the compiled pipeline against tree evaluation
- `NprShaderContext.n_dot_v` field for Fresnel-driven pipelines; canonical scene shader now declares `ndv = -dot(n, ray_dir)` in the hit branch
- `tests/npr_shader_validate.rs` — Naga-based validation of `SceneShaderBuilder` GLSL and WGSL output (default pipeline + `.with_pipeline` custom trees), plus `naga::valid::Validator` semantic validation on the full-variant WGSL pipeline
- `alice_sun_disc` added to `NPR_GLSL_HELPERS` / `NPR_WGSL_HELPERS` / `NPR_HLSL_HELPERS`
- `examples/npr_scene_shader.rs` — Emit a fully-composed shader for a small CSG scene via `SceneShaderBuilder`
- `.github/workflows/npr-bench.yml` — Benchmark regression watchdog that compares NPR primitive latency between PR head and `main` baseline

## [v1.7.7] - 2026-09-12

**crates.io landing** — first release published to https://crates.io/crates/alice-sdf Absorbs the Unreleased mesh-optimization batch plus the 1.7.4-1.7.6 preparation work (bridge trim + security fixes + fuzz + CI hardening)

### Security

- **RUSTSEC-2025-0020** (pyo3 `PyString::from_object` buffer overflow) — resolved by pyo3 `0.23 → 0.29` major bump
- **RUSTSEC-2026-0177** (pyo3 `PyCFunction::new_closure` `Sync` missing) — resolved by pyo3 `0.23 → 0.29`
- **RUSTSEC-2025-0141** (bincode 1.x unmaintained) — resolved by bincode `1.3 → 2.0`; wire format compat kept via `config::legacy()` so existing `.asdf` files remain readable

### Removed (temporary, restoration scheduled in 1.8.0)

- **5 optional path deps + associated features**: `alice-codec`, `alice-physics`, `libasp` (ALICE-Streaming-Protocol), `alice-cache`, `alice-font` are removed from `[dependencies]`, and the matching features `codec` / `physics` / `asp` / `sdf-cache` / `font` from `[features]` The corresponding `src/*_bridge.rs` modules remain `#[cfg(feature = "...")]`-gated and simply do not compile on crates.io 1.7.7 Users who need the bridges keep using `path` / `git` deps against the sibling repos
- Previously prepared as v1.7.4 (2026-07-23) but that tag/publish was skipped; this release folds the trim + subsequent 1.7.5 / 1.7.6 (internal) hardening into a single crates.io landing

### Added — Mesh optimization batch (23 methods absorbed from zeux/meshoptimizer)

Large batch of mesh-optimization work absorbing 23 methods from the
zeux/meshoptimizer C++ library, adding meshopt binary-compatible codecs,
`EXT_meshopt_compression` glTF integration, vertex filters, triangle
stripification, and Nanite-style meshlet clusters No breaking API changes;
all additions are opt-in

### Added

#### meshopt binary-compatible codec

- **`mesh::meshopt_index_codec`** — indexcodec v1 port (EdgeFIFO +
  VertexFIFO + 16-entry `codeaux` table + 4-mode encoding: edge FIFO
  match, codeaux fast path, full triangle encode, reset detection)
  Public API: `encode_index_buffer(indices) -> Vec<u8>`,
  `decode_index_buffer(bytes, index_count) -> Result<Vec<u32>, CodecError>`
- **`mesh::meshopt_vertex_codec`** — vertexcodec v0/v1 port (16-byte
  groups + bit widths 0/1/2/4/8 + control byte 4-mode: bit-encoded,
  zero, literal, XOR+rotate channel) Public API:
  `encode_vertex_buffer(data, size)`,
  `encode_vertex_buffer_level(data, size, level)` with `level` selecting
  `0=scalar / 2=u8-u16 estimate / 3=u8-u16-u32 XOR+rot estimate`,
  `decode_vertex_buffer(bytes, count, size)`
- **`estimate_rotate` heuristic** — 8-rotation bit-consistency search
  matching meshopt `estimateRotate`, activated at `level >= 3`
- **`tests/meshopt_reference_vectors.rs`** — cross-verification against
  8 fixtures generated by the meshoptimizer v0.24+ C++ library
  (`tri_single`, `strip_small`, `seq_100`, `large_500`, `uniform`),
  proving binary compatibility of the Rust decoder with C++-encoded bytes

#### glTF `EXT_meshopt_compression`

- **`io::meshopt_gltf`** module — compact GLB writer applying meshopt
  encoding to POSITION / NORMAL / TEXCOORD_0 / JOINTS_0 / WEIGHTS_0 /
  indices `MeshoptGltfConfig { export_normals, export_uvs, level,
  double_sided }` with public API `export_glb_meshopt` /
  `export_glb_meshopt_bytes` and skinned variants
  `export_glb_meshopt_skinned` / `export_glb_meshopt_bytes_skinned`
- **`MeshoptSkinning { joints: Vec<[u8; 4]>, weights: Vec<[u8; 4]> }`**
  — external per-vertex bone indices + weights for glTF skinning without
  extending the `Vertex` struct (backward compat)
- **`GltfConfig::meshopt_compress: bool` + `meshopt_level: u8`** — enable
  the meshopt path from the existing `export_glb` / `export_glb_bytes`
  entry points via delegation to `io::meshopt_gltf`; existing
  KHR_mesh_quantization / material / bufferView paths are unaffected
  when the option is off

#### Vertex filters

- **`mesh::meshopt_filter`** module — three encoders + matching in-place
  decoders:
  - **Octahedral** (`encode_filter_oct_i16` / `decode_filter_oct_i16_in_place`)
    — unit-vector projection for normals/tangents, 50–75% smaller than
    raw `f32×3` storage with <1% angular error
  - **Quaternion** (`encode_filter_quat_i16` / `decode_filter_quat_i16_in_place`)
    — largest-component + cyclic-swizzle storage, double-cover discards sign
  - **Exponential** (`encode_filter_exp_u32` / `decode_filter_exp_u32_in_place`)
    — per-lane mantissa (24 bit) + shared exponent (8 bit) pack

#### Mesh optimization primitives

- **`mesh::stripifier`** — Evans-Skiena-Varshney greedy strip generation
  (`stripify(indices, vertex_count, restart_index)` /
  `unstripify(strip, restart_index)`) with 8-triangle lookahead buffer,
  primitive-restart or degenerate-triangle joining Empirical index
  reduction ~48% on closed sphere meshes
- **`mesh::meshlet`** — Nanite-style meshlet clustering
  (`build_meshlets_scan` V1 and `build_meshlets_adjacency` V2 with
  `MeshletConfig::quality()` enabling `adjacency_grow=true` +
  `cone_weight=0.25`) Emits Vulkan `VK_EXT_mesh_shader` /
  DirectX 12 mesh-shader-ready cluster data
- **`ClusterBounds` + `NormalCone { axis, cutoff_cos, apex }`** —
  cluster culling data with `cone_apex` computed via
  `NormalCone::from_normals_and_positions` for tighter backface rejection
- **`mesh::overdraw::optimize_overdraw`** — view-independent triangle
  sort preserving vertex-cache clusters, plus
  `optimize_overdraw_with_views` for custom view directions
- **`mesh::spatial_order::optimize_spatial_order`** — 30-bit Morton
  Z-order spatial locality reorder for BVH build speedup
- **`mesh::optimize::optimize_vertex_fetch`** + **`compute_atvr`** —
  vertex-fetch order + Average Transformed Vertex Ratio metric

#### Quantization + mesh codec

- **`mesh::quantization`** — snorm/unorm i8/i16 encode/decode helpers +
  IEEE 754 binary16 (`half_encode` / `half_decode`)
- **`mesh::mesh_codec`** — custom varint delta codec (independent from
  meshopt binary format) with header `b"ASDF"`, LEB128 varint, zigzag
  signed delta, per-slot triangle index delta encoding, per-byte
  position stream delta Typical index 2–3×, regular position 4–8×
  compression
- **glTF quantization integration** —
  `GltfConfig::quantize_positions` refactored to
  center-based `snorm_i16_encode` (full i16 range, 2× precision vs the
  legacy `[0, 32767]` half-range mapping); new
  `GltfConfig::quantize_normals` (`SBYTE snorm`),
  `quantize_uvs` (`USHORT unorm`), `quantize_colors` (`UBYTE unorm`),
  `quantize_tangents` (`SBYTE snorm`) with unified
  `KHR_mesh_quantization` extension trigger

#### Simplifier / decimation

- **`DecimateConfig::lock_vertices: Vec<bool>`** — per-vertex lock mask
  (meshopt `lockVertices` equivalent) for LOD-seam preservation
- **`DecimateConfig::error_absolute: bool`** (default `true`) — when
  `false`, `max_error` is scaled by the mesh AABB diagonal so the
  threshold applies proportionally to the mesh size, matching the
  meshopt `simplifier.cpp` non-`SimplifyErrorAbsolute` semantics

#### Mesh repair + UV metrics

- **`MeshRepair::orient_faces`** — BFS + signed-volume face reorientation
  for consistent winding
- **`MeshRepair::fill_holes`** — connected-component + triangle-fan hole
  filling
- **`MeshRepair::drop_specks`** — Union-Find + `min_ratio` (`f32`)
  small-island removal
- **`compute_uv_density`** — per-face texel-density measurement with
  `UvDensityReport`, `RECOMMENDED_MIN_TEXELS_PER_FACE = 30.0`, and
  `WARN_LOW_DENSITY_RATIO = 0.05`

#### glTF materials

- **`GltfConfig::double_sided`** — force `doubleSided: true` on all
  materials, addresses back-face culling of mixed-orientation faces from
  Dual Contouring / Marching Cubes output

### References

- zeux/meshoptimizer (MIT) v0.24+ — indexcodec, vertexcodec, stripifier,
  simplifier, clusterizer, overdrawoptimizer, spatialorder,
  vfetchoptimizer, quantization, vertexfilter
- Evans, Skiena, Varshney "Optimizing Triangle Strips for Fast
  Rendering" (1996)
- Cigolle et al "A Survey of Efficient Representations for Independent
  Unit Vectors" (2014)
- Fabian Giesen "Simple lossless index buffer compression" (2013)
- Conor Stokes "Vertex Cache Optimised Index Buffer Compression" (2014)

### Added — Morphology (SDF offset + tolerance fit check for 3-D-print clearance)

- **`morphology` module** (~300 LOC): SDF morphological operations for CAD tolerance / print-clearance workflows
  - `eval_offset(node, point, radius)`: canonical signed offset (exact for `A ⊕ B_r` dilate when `r > 0`, `A ⊖ B_r` erode when `r < 0`)
  - `eval_offset_batch` / `eval_offset_batch_parallel`: batch variants matching the existing `shell` module API surface
  - `tolerance_fits(inner, outer, tolerance, samples, half_extent)`: sample-based test that `inner ⊂ outer ⊕ B_tolerance`
  - `tolerance_max_violation(...)`: worst-case penetration depth (ALICE-Bamboo safety validator uses this for auto-adjusting clearance)
  - Tests: 10 unit (offset scalar/batch/parallel + tolerance-fits accept/reject + violation reporting + panic paths); library total 1311 → 1321 passing
  - Note: set-theoretic `open` / `close` compositions do not reduce to closed-form SDF on arbitrary shapes; flagged as future work in module docs

### Fixed

- **Fuzz-found DoS in `load_asdf`** (`src/io/asdf.rs`): valid ASDF magic + malformed body triggered a bincode 2 `decode_from_slice` `Vec` capacity-overflow panic (attacker-controlled `.asdf` could abort the process). Fixed by `bincode::config::legacy().with_limit::<256 MB>()` allocation cap + panic → `Err(IoError::Serialization)` graceful conversion. Wire format compat preserved (encode side bit-exact) Regression test `test_malformed_body_no_panic` uses the exact fuzz artifact
- **`cargo fmt` regression** (`src/python/*.rs` × 4 sites): `Python::detach(|| ...)` single-line collapse was missed in the pyo3 `0.23 → 0.29` migration; applied and CI restored
- **CI `stub-guard` regex false-positive**: trait default methods with `panic!("... not implemented by this backend")` were being flagged as unshipped stubs; regex tightened to `panic!\([^)]*STUB` (uppercase-only), `todo!` / `unimplemented!` still detected

### Changed — CI hardening

- **`actions/checkout` `@v4 → @v5`** across all workflows (Node.js 20 deprecation), 20+ sites
- **`security-audit.yml` — 2 new informational jobs**: `coverage` (`cargo-llvm-cov`) and `semver-checks` (`cargo-semver-checks`) Semver-checks runs `continue-on-error: true` because 1.8.0 physics/font restoration is a planned major-bump event
- **`security-audit` path filter expanded**: `.github/actions/**` and sibling crate `Cargo.toml` under `examples/*/` and `bindings/**/` now trigger the workflow
- **`alice-stubs` action Cargo.toml template**: `license = "MIT OR Apache-2.0"` added so `cargo-deny` licenses check passes for CI-generated bridge stub crates (stubs are ephemeral, not shipped)
- **`deny.toml`**: `[[licenses.exceptions]]` for `alice-physics` (AGPL-3.0, internal sibling crate) added to bypass mechanical SPDX rejection under `--all-features`; redundant empty `exceptions = []` removed
- **`machete` CI job**: `alice-stubs` step wired for path-dep resolution + `[package.metadata.cargo-machete].ignored` added to 3 Cargo.toml files (`alice-sdf` / `alice-sdf-wasm` / `alice-sdf-bevy`) to suppress false positives for feature-gated planned deps
- **`cargo-fuzz` scaffold + Fuzz workflow** (`.github/workflows/fuzz.yml`) — 3 fuzz targets (`fuzz_sdf_eval` / `fuzz_asdf_decode` / `fuzz_bincode_roundtrip`), nightly toolchain override, matrix parallel, daily `03:00 UTC` schedule, `workflow_dispatch` with `duration_seconds` input Local 5-second smoke: 609k / 52k / 524k executions, 0 crashes each Day-1 real DoS bug catch (see Fixed above) validated the ROI

## [v1.7.4] - 2026-07-23

_Prepared but never tagged / published; the trim plus subsequent 1.7.5 / 1.7.6 (internal) hardening were folded into v1.7.7 (2026-09-12) Preserved below for historical accuracy of the initial trim plan_


### Removed (temporary, restoration scheduled in 1.8.0)

- **5 optional path deps + associated features**: `alice-codec`,
  `alice-physics`, `libasp` (ALICE-Streaming-Protocol), `alice-cache`,
  `alice-font` were removed from `[dependencies]`, and the matching
  features `codec` / `physics` / `asp` / `sdf-cache` / `font` from
  `[features]`, so the crate can be published to crates.io without
  waiting on the transitive dep chain (15+ crates deep). The
  corresponding `src/*_bridge.rs` modules are unchanged and remain
  `#[cfg(feature = "...")]`-gated — they simply never activate on
  crates.io 1.7.4. Users who need the bridges keep using `path`/`git`
  deps against the sibling repos as before.

### Scheduled restoration (1.8.0)

Once `alice-crypto` / `alice-analytics` / `alice-ml` / `alice-db` /
`alice-cache` / `alice-codec` / `alice-physics` / `libasp` /
`alice-font` (and their own transitive deps) reach crates.io, 1.8.0
will restore the 5 features and dep entries with `version =` pins so
`cargo add alice-sdf --features physics` starts working on
downstream consumers.

### Fixed

- Keyword `signed-distance-function` (24 chars) → `distance-field`
  (14 chars) to satisfy the crates.io 20-char limit.
- Description expanded to note the temporary bridge removal.

## [v1.7.3] - 2026-07-04

### Changed

- **`wgpu` dependency: 23 → 24** — GPU features (`gpu`, `volume`, `gpu-mesh`) の内部 wgpu を major bump。API 表面は不変、`Instance::new()` が `wgpu::InstanceDescriptor` を値渡しから参照渡しに変わったため内部 5 箇所 (`src/mesh/gpu_marching_cubes.rs` / `src/compiled/wgsl/gpu_eval.rs`) で `Instance::new(&desc)` に変更。ALICE-TRT v0.8.0 と wgpu version を揃えて **単一 `GpuDevice` を alice-sdf + alice-trt 間で共有可能** に (下流 crate が両方使う場合の VRAM 節約 + wgpu type mismatch 解消)
- **README** — Engine integrations 列挙を `Unreal Engine 5 / 6` に更新 (英語/日本語)

### Added

- **Unreal Engine 6.0 (UE6) support** — `unreal-plugin/AliceSDF.uplugin` の `EngineVersion` を `6.0.0` に bump (UE6-main `f602d4b` time point)。UE5.5+ で導入された最新 RHI API (= `FRHIBatchedShaderParameters` / `FRHIBufferCreateDesc::CreateVertex/CreateIndex` / 4 引数 `SubscribeToPostProcessingPass` / `DispatchComputeShader` / `IMPLEMENT_GLOBAL_SHADER` / `LAYOUT_FIELD` / `FSceneViewExtensionBase` / `GScreenRectangleVertexBuffer` 等) が UE6 にも残存、`UE_DEPRECATED(6.x)` 0 件確認、Build.cs / `.cpp` / `.h` / `.usf` 改変ゼロで論理互換。実機 UE6 Editor build 検証は別途実施推奨

### Backwards compatibility

- Public API 変更なし
- **注**: `--features gpu` (または `volume` / `gpu-mesh`) を有効化する下流 crate は自身の `wgpu` を 24 に揃える必要あり (同 major でないと `wgpu::Device` / `wgpu::Buffer` の型が別種扱い)。ALICE-Metaverse など path dep で追従する crate は自動同期

## [v1.7.2] - 2026-06-08

### Added

- **core clippy-strict CI** — `clippy` job を informational から `-D warnings` 化 (no-default-features + glsl/hlsl/gpu の 2 matrix)。新 lint 混入を即 CI fail で発見
- **Pre-built wheel CI** (`.github/workflows/release-wheels.yml`) — tag push (`v*`) で linux-x86_64 / linux-aarch64 / macos-arm64 / macos-x86_64 / windows-x86_64 の wheel を maturin で abi3-py310 ビルドして Release に attach。1 wheel で Python 3.10–3.13 をカバー
- **REST server smoke test** CI job — `/version` / `/eval` / `/op` / `/mesh` / `/splat` / `/vox` の全 endpoint を curl で叩く
- **WASM build** CI job — `cargo build --target wasm32-unknown-unknown --features wasm` で artifact 生成検証
- **Three.js TypeScript type-check** CI job — `tsc --noEmit` で TypeScript 健全性確認
- **Mobile sample compile** CI job (macOS runner) — iOS は xcodebuild build-for-testing、Android は `gradlew assembleDebug` でリグレッション検出
- **visionOS XCFramework support** — `mobile/packaging/ios/build-xcframework.sh --with-visionos` で `aarch64-apple-visionos` / `aarch64-apple-visionos-sim` slice を追加 (nightly + `-Z build-std`)
- **REST server hardening**:
  - `Authorization: Bearer <ALICE_SDF_TOKEN>` middleware (env が空でなければ全 endpoint で必須化、`/` `/version` は除外)
  - `tower_governor` レート制限 (per-IP、デフォルト 20 RPS / burst 60、`ALICE_SDF_RPS` / `ALICE_SDF_BURST` で上書き可能)
  - `RequestBodyLimitLayer` で 1 MiB JSON body 上限
- **`docs/USAGE.md` / `docs/USAGE_JP.md`** — README から詳細セクション 1675 行を移動
- **`docs/PUBLISH.md`** — crates.io 配布戦略の現状とロードマップを明文化

### Changed

- **STEP / IGES README claim 是正** (`README.md` / `README_JP.md`) — 「Fusion 360 / SolidWorks / OnShape / Rhino / AutoCAD / FreeCAD 互換」を撤回。実態は `POLY_LOOP` + `FACE_OUTER_BOUND` の faceted mesh / Entity 134+136 FEM mesh で、`MANIFOLD_SOLID_BREP` を要求する CAD ツールでは開けない可能性がある旨を明記
- **REST server resolution / size 検証** — 旧 silent `clamp(8, 192)` を `400 Bad Request` に変更 (out-of-range を明示的にエラー返却)
- **README 分割**: 2585 → 913 行 (35%)、JP も同様
- `pyproject.toml`: `requires-python = ">=3.9"` → `">=3.10"` (abi3-py310 と整合)
- `pyproject.toml`: project version 0.1.0 → 1.7.2 (Cargo.toml と同期)

### Fixed

- `src/python/compiled.rs`: 未使用の `source_node` field 削除 (`dead_code` warning 除去)
- `src/io/iges.rs`: `format!()` を str literal に置換 (clippy `useless_format`)
- `src/io/vox.rs`: `cfg.size.min(256).max(1)` → `cfg.size.clamp(1, 256)` (clippy `manual_clamp`)

### Compatibility

- Mobile: iOS / Android **+ visionOS** (XCFramework スクリプトに追加)
- Unreal Engine: 5.7.0 〜 5.7.4 / 5.8.0-preview-1 (変更なし)

## [v1.7.1] - 2026-06-08

### Added

- **REST server endpoint 拡張** (`server/`) — `POST /mesh` (Marching Cubes vertices+normals+indices)、`POST /splat` (3D Gaussian Splats、`format=bytes` で base64 32-byte stream)、`POST /vox` (voxel 配列) を追加。`/version` が全 endpoint を列挙
- **OpenXR `SceneFrame` / `SphereBeacon` / `RayHit` API** (`bindings/openxr/`) — フレーム 1 回分のシーン状態を builder style で組み立て、head/left/right の raycast と手メッシュ→beacon 最小距離をワンメソッドで取得
- **OpenXR `examples/quest_demo.rs`** — Meta Quest 風の 60-frame loop 完全実装サンプル
- **visionOS `makeSDFMeshEntity` / `makeBlobEntity`** — 任意 SDF closure を voxel-fill で評価し RealityKit `ModelEntity` 化 / 2 球 smooth-union を Rust `AliceSDFFramework` 直呼出で blob 生成
- **visionOS `AliceSDFFramework` 統合** — Swift 側 SDF 計算を Rust UniFFI コアにルーティング (`sdfSphere` / `opSmoothUnion` / `sphereBatch` / `aliceSdfVersion`)、`canImport(AliceSDFFramework)` でフォールバック実装も保持

### Changed

- **STEP / IGES export を Marching Cubes 化** (`src/io/step.rs` / `src/io/iges.rs`) — 旧 naive voxel quads を `mesh::sdf_to_mesh` (実 MC アルゴリズム) に置換。res=16 で <100 verts → 数千 verts の品質向上
- **PyO3 を `abi3-py310` に固定** — Python 3.10 / 3.11 / 3.12 / 3.13 を 1 つの `.so` でサポートし、再ビルド不要に
- **CHANGELOG split**: v0.1.0 – v1.3.0 を `CHANGELOG-history.md` に分離 (本ファイルの肥大化対策)
- **CI `clippy-strict` トリガ拡張**: `paths-filter` で `code` (core src/**) 変更時も mobile wrapper の strict clippy を実行 (uniffi-wrapper は alice-sdf core を path dep として再 clippy するため、core の変更が見落とされる設計ミスを修正)
- **CI mobile job**: `cargo test --tests` (debug プロファイル) を追加し、12 公開 wrapper 関数を 26 統合テストで網羅
- README (英日) の Python 節に abi3 / pre-built wheel 説明を追加

### Fixed

- `src/io/vdb.rs`: `VdbError::Io` / `VdbError::InvalidBounds` の missing variant docs (clippy strict 対応)

### Quality

- **Core**: 1,093 tests passing (+10 vs v1.7.0 — STEP/IGES 各 +3 quality tests + Bevy +7 + OpenXR +8 + mobile +22 + server +4)
- **OpenXR**: 4 → 12 tests
- **Bevy**: 4 → 11 tests (normals 単位長 / vertex bounds / annulus / cap planes / plugin build)
- **mobile/uniffi-wrapper**: 4 unit + **26 integration tests** (全 12 公開関数を網羅)
- **server**: 0 → 4 tests
- 全 clippy-strict (`-D warnings`) pass: openxr / mobile / bevy

## [v1.7.0] - 2026-06-06

### Added

#### 3D / Modern rendering

- **3D Gaussian Splatting I/O** (`src/io/splat.rs`) — Inria 3DGS 互換 `.splat` バイナリ (32 bytes/splat: pos + scale + RGBA + compressed quat) の読書き、`sdf_to_splats()` で SDF 表面近傍を Gaussian Splat 化、4 tests
- **MagicaVoxel I/O** (`src/io/vox.rs`) — `.vox` v150 RIFF (SIZE + XYZI chunks) の読書き、`sdf_to_vox()` で SDF を voxelize、4 tests
- **STEP AP203 export** (`src/io/step.rs`) — ISO 10303-21 ASCII Faceted BREP、Fusion 360 / SolidWorks / OnShape / Rhino / FreeCAD 互換、2 tests
- **IGES export** (`src/io/iges.rs`) — IGES ASCII Entity 134 (Node) + 136 (Finite Element) で三角形メッシュ表現、Rhino / AutoCAD 互換、2 tests

#### Web / Mobile / XR

- **WebXR raymarching helpers** (`src/wasm.rs` 拡張) — `raymarch_sphere` / `raymarch_two_spheres_smooth` / `sphere_batch_flat` で VR/AR コントローラ・ハンドメッシュ用 SDF クエリ
- **Three.js / React Three Fiber TypeScript wrapper** (`bindings/threejs/`) — `@alice-sdf/threejs` npm パッケージ、`AliceSDF` クラス + `createSliceTexture()` Three.js helper + `<AliceSDFSlicePlane>` R3F コンポーネント + WebXR 統合例
- **OpenXR native helpers** (`bindings/openxr/`) — `XrPose` 変換 + `raymarch_sphere` + ハンドメッシュバッチ評価、Meta Quest / PC VR / Apple Vision Pro 対応、3 tests
- **visionOS Swift Package** (`mobile/swift-package-visionos/`) — Apple Vision Pro 用 RealityKit ヘルパー (`makeSphereEntity` / `makeBoxEntity`)、`AliceSDFFramework` (XCFramework) を再利用

#### DCC ツール統合

- **Blender Add-on** (`bindings/blender/`) — Blender 4.0 / 4.2 LTS / 4.4+ プラグイン: `.asdf` Import operator + sphere/box/torus 生成 + N-panel UI
- **Houdini Python plugin** (`bindings/houdini/`) — Houdini 20.0 / 20.5 / 21+ 用 Python SOP body + 自動 install.sh (python3.10libs/3.11libs 検出)
- **Maya Python plugin** (`bindings/maya/`) — Autodesk Maya 2024 / 2025 / 2026+ 用 Python module + MFnMesh + `register_menu()`
- **Nuke Python plugin** (`bindings/nuke/`) — Foundry Nuke 15.x / 16.x 用 Python module + Volume export + Slice render
- **Cinema 4D Python plugin** (`bindings/cinema4d/`) — Maxon Cinema 4D 2024 / 2025 / 2026+ 用 Python module + PolygonObject 生成

#### Cloud / Server

- **REST API server** (`server/`) — `axum` 0.7 + `tokio` 1.40、`POST /eval` (primitive 評価) + `POST /op` (operation) 公開、`alicelaw.net/sdf-metaverse` バックエンド向け

### Changed

- README (英日) に Web/VFX/Bevy/Splat/Vox/Blender/Houdini/Maya/Nuke/Cinema 4D/Three.js セクション追加
- DCC ツールの対応バージョンを各 README で **後方互換維持 + 新バージョン明示** (Maya 2024-2026、Houdini 20.0/20.5/21、Nuke 15.x/16.x、Blender 4.0/4.2/4.4)
- `AliceSDF.uplugin`: VersionName 1.6.0 → 1.7.0、Version 3 → 4

### Fixed

- `src/io/vox.rs` / `src/io/iges.rs`: pub struct field の missing docs (clippy strict 対応)
- `src/io/iges.rs`: unused `mut` / 使われない変数を削除
- `src/eval/mod.rs`: `43758.5453` の f32 過剰精度を `43758.547` に修正 (clippy strict `excessive_precision` 対応)

## [v1.6.0] - 2026-06-06

### Added

- **`wasm` feature** — WebAssembly bindings (browser): wasm-bindgen + js-sys。`sdf_sphere` / `sdf_box` / `sdf_torus_w` / `sdf_cylinder_w` / `sdf_plane_w` / 6 op + `render_sphere_slice_rgba` を JavaScript から呼び出し可能。`cargo build --target wasm32-unknown-unknown --features wasm` で動作
- **`openvdb` feature** — OpenVDB Float Grid I/O (Houdini / Maya / Nuke 等の VFX/DCC ツール連携): `bake_dense_grid()` / `bake_to_vdb()` / `load_dense_grid_from_vdb()`。vdb-rs 0.6 ベース。`io::vdb` モジュール、4 tests
- **Bevy plugin** (`bindings/bevy/alice-sdf-bevy/`) — Bevy 0.18 用 ECS 統合: `AliceSdfPlugin` + `SdfShape` Component (Sphere/Box/Torus/Cylinder)、Mesh 自動生成 system、`examples/sphere_demo.rs`、4 tests
- **CI matrix 拡張**: `wasm` / `openvdb` / `bevy` の 3 ジョブ追加、`physics` strict 化 (continue-on-error 削除 + 実 ALICE-Physics clone + 1088 tests カバー)

### Changed

- README (英日) に "Web (WebAssembly) / VFX (OpenVDB) / Bevy エンジン" セクション追加
- `AliceSDF.uplugin`: VersionName 1.5.0 → 1.6.0、Version 2 → 3

### Quality

- 全 CI matrix green: macOS ARM64 / Linux x86_64 / Windows x86_64 + Mobile + Physics strict + wasm + openvdb + bevy + clippy + clippy-strict + fmt
- 全 strict job pass (continue-on-error なし)

## [v1.5.0] - 2026-06-06

### Added

- **Mobile SDK (iOS / Android)** — `mobile/` 配下に [UniFFI](https://mozilla.github.io/uniffi-rs/) ベースの Swift / Kotlin 公開 SDK
  - `mobile/uniffi-wrapper/` — UDL 定義 + Rust ラッパークレート (`sdfSphere` / `sdfBox` / `sdfTorus` / `sdfCylinder` / `sdfPlane` / `sdfRoundedBox` + 6 op + `sphereBatch` + version)
  - `mobile/packaging/ios/build-xcframework.sh` — `AliceSDF.xcframework` (device 44MB + sim fat 88MB) 自動生成
  - `mobile/packaging/android/build-aar.sh` — 4 ABI `libuniffi_alice_sdf.so` (250-400KB) + Kotlin bindings 自動生成
  - `mobile/swift-package/Package.swift` — SwiftPM パッケージ (binaryTarget + Swift bindings 2層)
  - `mobile/samples/ios-swiftui/` — SwiftUI サンプルアプリ (xcodegen + Bridging Header 方式)
  - `mobile/samples/android-compose/` — Jetpack Compose サンプルアプリ (AGP 8.5.2 + Kotlin 2.0)
  - 実機検証: iPhone 17 Pro Simulator (iOS 26.0) + Pixel 6 Emulator (Android 14 / API 34) で iOS と Android 数値完全一致 (sphere d=0.2806, smooth union=0.2056)
- **Rendering metaverse features** — RenderConfig に分光レンダリング / 破壊 / VFX / マイクロ法線 / インテリアマッピング / dual SDF 還元
- **`WgslShader::transpile_material()`** — WithMaterial サブツリーからマテリアル評価関数を WGSL 生成
- **Terrain primitive** — 地形プリミティブ + フルレンダリングパイプライン
- **`examples/sword.lol`** — LOL DSL で記述した剣のサンプル
- **Unreal Engine 5.8 互換性確認** — `AliceSDF.uplugin` に `"EngineVersion": "5.7.0"` 明示、UE 5.7.0 〜 5.7.4 stable + 5.8.0-preview-1 で改修不要を実証 (Shader Parameter API: `FRHIBatchedShaderParameters` + `SetBatchedShaderParameters` + `FRHIBufferCreateDesc::CreateStructured` + `FRHIViewDesc::CreateBufferSRV/UAV`)
- **README リンク**: ALICE SDF Metaverse demo (https://alicelaw.net/sdf-metaverse) + alicelaw.net repo を Related Projects に追加 (英日)

### Changed

- **CI/CD 大規模強化**:
  - `concurrency: cancel-in-progress` で連続 push 時の前 run 自動 cancel
  - `dorny/paths-filter@v3` で README/docs only push の full CI skip
  - `.github/actions/alice-stubs` composite action で dep stub 生成を DRY 化 (60行 × 2 jobs)
  - `mobile` job 新規: iOS 3 target + Android 4 ABI cross-compile + Swift/Kotlin bindings 生成検証 + gpu (Metal) feature ビルド
  - `clippy-strict` job: mobile/uniffi-wrapper のみ `RUSTFLAGS="-Dwarnings"` 厳格
  - `nick-fields/retry@v3`: cargo build 3 リトライ (HTTP/2 framing layer 一過性失敗対策)
  - `CARGO_NET_RETRY=5` + `CARGO_HTTP_MULTIPLEXING=false` env
  - `fmt` job 拡張: core + mobile/uniffi-wrapper 両方
- **Author email**: `Moroya Sakamoto <sakamoro@alicelaw.net>` に統一 (Cargo.toml authors)

### Fixed

- **`optimize.rs`**: 4 パスを値渡し化、未最適化ノードの deep clone 除去 (perf)
- **BVH**: `split_off` 化、mipchain clone 除去、abm `read_to_end` 事前確保 (perf)
- **`check_min_tests`**: 算術エラー修正、cargo 失敗時の安全な skip
- **`ecosystem-tests` schedule**: 削除 (CI では兄弟クレート不在で動作不可)
- **`transpile_material`** ヘルパー重複定義の排除
- **cargo fmt** 差分修正 (CI rustfmt 互換)

### Quality

- **1,379 tests passing** (src/ 1,375 + mobile/uniffi-wrapper 4), 0 failed (+205 from v1.3.0)
- 0 clippy pedantic+nursery warnings (core)
- 0 clippy `-D warnings` (mobile wrapper、strict mode)
- 0 fmt diffs (core + mobile)
- CI matrix: macOS ARM64 + Linux x86_64 + Windows x86_64 + macOS Mobile cross-compile

### Compatibility

| Platform | Status |
|----------|--------|
| Linux x86_64 / aarch64 | 🟢 |
| macOS Apple Silicon / Intel | 🟢 |
| Windows x86_64 | 🟢 |
| **iOS aarch64 / sim** | 🟢 v1.5.0 新規 |
| **Android arm64-v8a / armv7 / x86_64 / x86** | 🟢 v1.5.0 新規 |
| Unreal Engine 5.7.0 〜 5.7.4 (stable) | 🟢 |
| Unreal Engine 5.8.0-preview-1 | 🟢 改修不要見込み |
