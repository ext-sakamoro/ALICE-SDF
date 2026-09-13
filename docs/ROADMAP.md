# ALICE-SDF Roadmap

Canonical roadmap tracking phases, current state, and open decisions for the
ALICE-SDF crate. Primary source of truth — the `MEMORY.md` index and per-session
memory files reference this document rather than duplicating the phase list.

- **Current version**: `v1.8.0` (crates.io landing scheduled after bridge dep restoration)
- **Repo**: `ext-sakamoro/ALICE-SDF`
- **License**: Dual (see `LICENSE`, `LICENSE-COMMUNITY`)
- **MSRV**: `1.75`

---

## Current position (2026-09-13)

`v1.8.0` landed on `main` (`76c0232`). The release absorbs the full **NPR
module landing** across 12 phases (P1 → P12-A) plus the `CompiledColorPipeline`
stack machine (Phase 12-C). All bridge deps (`alice-codec` / `alice-physics` /
`alice-cache` / `alice-font` / `alice-asp`) remain trimmed as in `v1.7.7`
because they are not yet published to crates.io.

The `npr` module is now the largest single subsystem in the crate and has:

- 9 primitive categories (toon / outline / sky / rim / hatch / distortion /
  palette / composition / motion) plus the `noise` trait family
- A colour-composition DSL (`NprColorNode`) with 17 variants, tree evaluator,
  builder API, shader transpiler, and host-side bytecode compiler
- A scene shader builder (`SceneShaderBuilder`) emitting GLSL / WGSL / HLSL
- Naga-based semantic validation in `tests/npr_shader_validate.rs`
- Criterion benchmarks (`benches/npr_primitives.rs`) with a CI regression
  watchdog (`.github/workflows/npr-bench.yml`)

---

## Phase log

Legend: ✅ landed · 🚧 in progress · ⏳ planned · 💤 deferred

### NPR module (P1 – P12)

| Phase | Status | Commit | Summary |
|-------|--------|--------|---------|
| P1 | ✅ | `76259cc` | MVP: `toon` / `outline` / `sky` primitive category |
| P2 | ✅ | `cf5f3b3` | Remaining 6 categories: `rim` / `hatch` / `distortion` / `palette` / `composition` / `motion` |
| P3 | ✅ | `6f2fd3a` | `NprColorNode` DSL + shader glue + autodiff link + noise trait + `prelude` |
| P4 | ✅ | `cf4ef9c` | `PerlinNoise` / `WorleyNoise` + shader palette variants + criterion benches |
| P5 | ✅ | `eb07c94` | `SimplexNoise` + `SceneShaderBuilder` + `sun_disc` shader glue + npr-bench CI |
| P6 | ✅ | `e8907b9` | `NprColorNode` shader transpiler + `SceneShaderBuilder.with_pipeline` |
| P7 | ✅ | `d7eab1d` | Variant expansion: `Multiply` / `Add` / `Scale` / `Fresnel` + naga WGSL/GLSL validation |
| P8 | ✅ | `7c6b00c` | Variant expansion: `Saturate` / `Bloom` / `PosterizeColor` + `alice_saturate` helper |
| P9 | ✅ | `84d89b9` | UV-dependent variants: `Vignette` / `Palette3` + `PaletteSource` enum |
| P10 | ✅ | `ae01ec2` | `Hatch` variant + `alice_hatch_lines` helper + naga validation |
| P11 | ✅ | `8a2f60d` | `Palette5` / `Tonemap` / `SpeedLine` + `tonemap_reinhard` / `speed_line` helper |
| P12-A | ✅ | `e36d82a` | Time context: `NprColorContext.time` / `NprShaderContext.time` + `iTime` uniform + `TimeCycle` palette source |
| P12-B | ✅ | `76c0232` | `v1.7.7 → v1.8.0` release cut + CHANGELOG polish |
| P12-C | ✅ | `18c84da` | `CompiledColorPipeline` stack machine — 5 native opcodes + `Fallback` tree eval delegation + bench |
| P12-D | ✅ | `d7f30cf` | Native opcode coverage extended to all 17 current `NprColorNode` variants (`Multiply` / `Add` / `OutlineOver` / `Fresnel` / `Saturate` / `Bloom` / `PosterizeColor` / `Vignette` / `Palette3` / `Palette5` / `Hatch` / `Tonemap` / `SpeedLine`); `Fallback` retained as forward-compat seam only. Deep-composition bench: 32.5 ns compiled vs 18.8 ns tree (~1.73× on 6-level tree, down from ~5-6× on shallow) |
| P13 | ✅ | pending | 8-lane SIMD batch evaluator (`wide::f32x8`, SoA `NprColorBatch8` + `NprBatchContext8`, `CompiledColorPipeline::eval_batch8`). 14 opcodes SIMD-native; 3 (`Fresnel` / `SpeedLine` / `Palette5`) fall back to per-lane scalar over the SoA batch. Deep 6-level composition: **11.4 ns/lane batched** vs 18.2 ns tree — first regime where the compiled pipeline beats tree eval outright |

### Post-1.8.0 candidates
| P14 | ⏳ | GPU-side bytecode serialisation — emit `ColorOp` stream into a uniform / storage buffer that the shader interpreter walks, enabling arbitrary DSL trees without per-tree shader recompile |
| P15 | ⏳ | Bridge dep restoration — re-add `alice-codec` / `alice-physics` / `alice-cache` / `alice-font` / `alice-asp` once each is on crates.io. Corresponding features (`codec` / `physics` / `asp` / `sdf-cache` / `font`) return to `[features]` |

### Deeper follow-ups (not scheduled)

- **CSG bytecode unification** — merge `npr::compiled_color::CompiledColorPipeline` with the existing `SdfNode` bytecode compiler in `src/compiled/`. Requires adding colour variants to `Instruction` / `Opcode` and extending the SIMD / stack-based interpreters.
- **`.wgsl` compute-shader offload of NPR** — evaluate the compiled colour bytecode inside the same compute pass that raymarches the scene, avoiding CPU / GPU boundary crossings entirely.
- **Autodiff / gradient path over `NprColorNode`** — currently the DSL is a forward-only evaluator; a gradient pass would let colour-composition parameters participate in optimisation loops.

---

## Open questions (ADR-pending)

1. **P12-D vs P13 first?** — Native opcode expansion (P12-D) is mechanical and improves fallback percentage; SIMD (P13) is where bytecode starts to beat tree eval. If a downstream consumer needs SIMD immediately, P13 could jump the queue.
2. **Bytecode format stability** — the `ColorOp` enum is currently `pub` and part of the public API. Before P14 (GPU serialisation) we should decide whether to freeze the bytecode format (dedicated wire encoding, versioned) or keep it as an internal ABI.
3. **Naga validation coverage** — `tests/npr_shader_validate.rs` currently validates the default pipeline plus a full-variant custom tree. Expanding to fuzz-driven random DSL trees would surface transpile bugs in rarely-exercised composition paths.
4. **Bridge dep restoration order** — once upstream crates publish, restore them one-by-one (verify each bridge builds) rather than in a single `v1.9.0` batch.

---

## Decision record (ADR)

### ADR-001 — Keep `Fallback` opcode in the compiled pipeline (2026-09-13, P12-C)

**Context**: `CompiledColorPipeline::compile` covers 5 of 17 `NprColorNode`
variants natively; the remaining 12 could either (a) block compilation, (b)
recurse via a `Fallback` opcode that delegates to `NprColorNode::eval`, or (c)
be inlined at compile time.

**Decision**: Ship `Fallback(Box<NprColorNode>)`. Uncovered variants transparently
work today, and future P12-D revisions can replace `Fallback` with native
opcodes without breaking the pipeline API.

**Consequences**: `native_op_count` / `fallback_op_count` accessors let benches
and callers monitor coverage. On shallow trees the compiled path is currently
slower than tree eval (measured: `toon` 3.5 ns tree vs 19 ns compiled on Apple
Silicon), but the API is stable ahead of SIMD / GPU work.

### ADR-002 — `iTime` as the canonical time-uniform name (2026-09-13, P12-A)

**Context**: Shadertoy compatibility and community familiarity vs. an
ALICE-specific naming convention.

**Decision**: Use `iTime` (Shadertoy convention) as the canonical uniform name
across GLSL / WGSL / HLSL output. `NprShaderContext.time` defaults to `"iTime"`
but callers can override.

**Consequences**: Shadertoy → ALICE porting stays low-friction. Existing
ALICE-SDF uniforms already coexist under `SceneUniforms` / `SceneCB`.

### ADR-003 — Trim bridge deps through v1.8.0 (2026-09-12, P12-B)

**Context**: `alice-codec` / `alice-physics` / `alice-cache` / `alice-font` /
`alice-asp` are not yet on crates.io. Publishing `v1.8.0` with `path` deps
would either block crates.io publish or force those crates to publish first.

**Decision**: Keep the bridge modules `#[cfg(feature = "...")]`-gated but drop
the `[dependencies]` and matching feature entries for the crates.io release.
Users needing bridges continue with `path` / `git` deps against sibling repos.

**Consequences**: `v1.8.0` publishes cleanly; bridge restoration deferred to
P15 once upstream crates are published.

---

## Cross-references

- CHANGELOG: `CHANGELOG.md` (post-1.5.0) / `CHANGELOG-history.md` (pre-1.5.0)
- Architecture: `ARCHITECTURE.md`
- API reference: `API.md` / `docs/API_REFERENCE.md`
- Bambu 3MF export path (canonical is `alice-bamboo`, not this crate): see `CLAUDE.md`
- Personal memory: `success_alice_sdf_phase_12_landing.md` (Phase 12 A/B/C session summary)
