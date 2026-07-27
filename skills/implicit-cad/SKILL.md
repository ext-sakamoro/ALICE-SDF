---
name: alice-implicit-cad
description: Create implicit CAD models via ALICE-SDF's 126-construct Signed Distance Function library (72 primitives, 24 CSG ops, 7 transforms, 23 modifiers). Supports 7 evaluation modes (interpret / VM / SIMD 8-wide / BVH / SoA / JIT / GPU), GLSL/WGSL/HLSL shader transpile for Unity/UE5/UE6/Godot/WebGPU/VRChat, and mesh export to GLB/OBJ/STL/PLY/3MF/FBX/USD/Nanite. Prefer this skill for SDF field composition, raymarching, high-primitive-count implicit models, and shader-native workflows. For LLM DSL-driven text→3D with GBNF-constrained decoding, use `alice-lol-sdf` skill instead.
---

# ALICE Implicit CAD (SDF-based)

Provenance: maintained in [ext-sakamoro/ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF). This skill is a thin wrapper — install the parent `alice-sdf` crate (`cargo add alice-sdf`) or clone the repo for full runtime.

## Purpose

Author 3D geometry as **Signed Distance Functions** (SDF) — mathematical field definitions — rather than polygon meshes or parametric CAD. Emit browser/engine-native shaders (GLSL / WGSL / HLSL) or extract meshes via Marching Cubes / Dual Contouring / V-HACD for downstream tooling.

## When to use this skill

Use this skill when the task calls for:

- **SDF field composition** — smooth booleans, TPMS gyroid/schwarz/diamond/neovius, distance-field modifiers, procedural CSG
- **Raymarch-native output** — GLSL / WGSL / HLSL emitted directly for browser or engine (WebGPU / Vulkan / DirectX / OpenGL)
- **Engine integration** — Unity / Unreal Engine 5-6 / Godot / VRChat / WebAssembly bindings
- **High-primitive-count implicit models** where mesh export is a secondary artifact
- **Compression-first delivery** — SDF tree JSON is 10-1000x smaller than equivalent polygon mesh
- **Advanced SDF features** — Neural SDF approximation, Interval Arithmetic AABB pruning, Analytic Gradient, Automatic Differentiation, Auto Tight AABB, mean/gaussian/principal curvature

Do **not** use this skill for:

- **STEP-first parametric CAD** with mating/joint/assembly (use `earthtojake/text-to-cad` `cad` skill or CADQuery / build123d directly — SDF is not the right primitive for BREP mechanical CAD)
- **DSL-driven text→3D via LLM** — use `alice-lol-sdf` skill (LOL DSL + GBNF constrained decoding gives syntax-error-free LLM output)
- **URDF / SRDF robot description** — use `earthtojake/text-to-cad` `urdf` / `srdf` skills
- **G-code slicing** for FDM print — use a slicer skill (e.g. `bambu-labs`)

## Core concepts

An ALICE-SDF model is an `SdfNode` tree. Two authoring paths:

1. **JSON tree** — declarative, LLM-friendly, `alice_sdf::SdfNode::from_json()` compiles
2. **Rust builder** — `alice_sdf::builder::sphere(1.0).union(...)` or `alice-lol` DSL (see `alice-lol-sdf` skill)

The tree evaluates to a signed distance field: `f(p: Vec3) → f32` where `f(p) < 0` inside, `> 0` outside, `= 0` on the surface.

### Available primitives (72)

sphere, box3d, rounded_box, cylinder, torus, cone, capsule, ellipsoid, plane, octahedron, rounded_cone, pyramid, hex_prism, link, capped_cone, capped_torus, rounded_cylinder, tube, barrel, heart, egg, helix, tetrahedron, box_frame, diamond, star_polygon, cross_shape, triangle, bezier, triangular_prism, cut_sphere, cut_hollow_sphere, death_star, solid_angle, rhombus, horseshoe, vesica, infinite_cylinder, infinite_cone, gyroid, chamfered_cube, schwarz_p, superellipsoid, rounded_x, pie, trapezoid, parallelogram, tunnel, uneven_capsule, arc_shape, moon, blobby_cross, parabola_segment, regular_polygon, stairs_prim, dodecahedron, icosahedron, truncated_octahedron, truncated_icosahedron, diamond_surface, neovius, lidinoid, iwp, frd, fischer_koch_s, pmy, circle_2d, rect_2d, segment_2d, rounded_rect_2d, annular_2d.

### Available CSG operations (24)

union, smooth_union, intersection, smooth_intersection, subtract, smooth_subtract, chamfer_union, chamfer_intersection, chamfer_subtraction, stairs_union, stairs_intersection, stairs_subtraction, columns_union, columns_intersection, columns_subtraction, exp_smooth_union, exp_smooth_intersection, exp_smooth_subtraction, xor, pipe, engrave, groove, tongue.

### Transforms (7) + Modifiers (23)

translate, rotate, scale, scale_non_uniform, mirror, polar_repeat, shear +
round, onion, twist, bend, repeat, elongate, revolution, extrude, taper, displacement, noise, repeat_finite, octant_mirror, icosahedral_symmetry, with_material, surface_roughness, sweep_bezier, shell, lattice_infill (TPMS), diamond_infill, schwarz_infill.

See `references/syntax.md` for full argument details.

## Workflow

1. **Write a modeling brief** — dimensions (mm assumed unless otherwise specified), coordinate frame (Y-up right-hand default), procedural material intent, target export format.
2. **Author the SDF tree** as an ALICE-SDF native file (`.asdf` binary or `.asdf.json`) or via the `alice-lol` DSL (recommended for LLM-generated content — see `alice-lol-sdf` skill).
3. **Validate**: `scripts/generate.sh <input.asdf.json>` runs `alice-sdf info` and reports tree structure, node count, bounds.
4. **Export mesh** (visualization / engine): `scripts/export.sh <input.asdf.json> --output part.{glb|obj|fbx|usda|abc}`. Format is auto-detected from extension.
5. **3D print output**: `scripts/print.sh <input.asdf.json> --output part.{stl|3mf|obj}` — uses higher default resolution (128) suited for fabrication.
6. **Shader transpile** (GLSL / WGSL / HLSL) — the current `alice-sdf` CLI does **not** expose transpile as a subcommand. Use the library API from Rust:
   ```rust
   use alice_sdf::{glsl::to_glsl, wgsl::to_wgsl, hlsl::to_hlsl};
   let src = to_glsl(&node); // or to_wgsl / to_hlsl
   ```
   Requires `--features glsl|gpu|hlsl` respectively on the parent crate.
7. **Verify** by importing the mesh into the target engine (Unity / UE5 / Godot / Blender) or opening the GLB in a browser viewer.

## Scripts

From this skill directory:

```bash
scripts/generate.sh <input.asdf.json>                          # validate + info dump
scripts/export.sh <input.asdf.json> --output <path.{glb|obj|fbx|usda|abc}> [--resolution N] [--bounds F]
scripts/print.sh <input.asdf.json> --output <path.{stl|3mf|obj}> [--resolution N] [--bounds F]
```

Scripts are thin wrappers over `cargo run --release --features cli --bin alice-sdf -- <subcommand>`. They assume the ALICE-SDF workspace is available at `../..` (default install layout when this skill lives at `<crate>/skills/implicit-cad/`).

Use `scripts/<name>.sh --help` for the full command interface.

## Handoff

After generating a mesh artifact (`.glb`, `.stl`, `.3mf`, `.obj`, `.fbx`, `.usda`, `.abc`), always report the file path to the caller. If a downstream viewer skill (`$cad-viewer` or equivalent) is installed, hand off the path for interactive preview.

Do not use this skill to open a live browser viewer directly — mesh export is the deliverable; visualization is a separate concern (delegated to the engine or a viewer skill). Snapshot / preview generation is intentionally out of scope for this skill.

## Non-negotiables

- Coordinate frame: Y-up right-hand unless the target engine forces otherwise (Unity is Y-up left-hand; ALICE-SDF handles the flip on export).
- Units: millimeters for 3D print outputs (STL / 3MF). Meters for engine assets by default (Godot / Unreal / Unity all Y-up meters).
- Do not silently swallow SDF parse or compile errors — surface them so the LLM can repair the tree.
- Do not emit invalid GLSL / WGSL / HLSL — the transpiler validates before write; if `--shader` fails, report the offending node.

## Related skills

- **`alice-lol-sdf`** (companion) — LOL DSL front-end with GBNF-constrained LLM decoding. Prefer for text→3D via LLM.
- **`earthtojake/text-to-cad` `implicit-cad`** — smaller SDF library (~20 primitives), browser JS module format (`.implicit.js`), targets browser-only workflows. ALICE `alice-implicit-cad` is a strict superset in primitive count and engine integration breadth.

## References

- `references/syntax.md` — Full 126-construct reference with argument shapes
- `references/shader-targets.md` — GLSL / WGSL / HLSL emit differences and gotchas
- Parent crate docs: `~/ALICE-SDF/README.md`, `~/ALICE-SDF/API.md`, `~/ALICE-SDF/ARCHITECTURE.md`
