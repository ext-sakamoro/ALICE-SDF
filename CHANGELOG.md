# Changelog

All notable changes to ALICE-SDF are documented in this file.

For releases prior to v1.5.0 (v0.1.0 – v1.3.0), see [CHANGELOG-history.md](CHANGELOG-history.md).

## [Unreleased]

## [v1.10.2] - 2026-09-14

### Fixed

- `SdfNode` now drops iteratively (`src/types/drop.rs`): children are moved onto an explicit heap stack and released one `Arc` at a time, so freeing a tree no longer recurses once per level. A 2,400-deep `subtract` nest (the shape ALICE-LOL's stdlib products produce) overflowed a 2 MB thread stack on drop; the regression test now builds and drops 100,000-deep chains in a 256 KB thread. Shared subtrees (`Arc::clone`) are left to their last owner as before. Recursive `clone` / `node_count` / evaluators are unchanged.

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
