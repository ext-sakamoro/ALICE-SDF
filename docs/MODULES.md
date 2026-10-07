# Module overview

Every public module of `alice-sdf`, grouped by area, with a one-line summary.
API details are on [docs.rs](https://docs.rs/alice-sdf); test status is in
[`oracle-status.md`](oracle-status.md).

- **Feature**: the Cargo feature the module needs. Blank means it is always
  available. **AGPL** marks a feature that links an `AGPL-3.0-or-later` crate
  (see [License](../README.md#license)).
- **Example**: a program in [`examples/`](../examples/) that uses the module
  (`cargo run --release --example <name>`).

`scripts/readme_sync.py --check` fails CI when a public module is missing from
this file or listed twice, or when a linked file does not exist.

## Contents

- [The tree and its laws](#the-tree-and-its-laws)
- [Evaluation](#evaluation)
- [Analysis and validation](#analysis-and-validation)
- [Meshing and file formats](#meshing-and-file-formats)
- [Shading, materials and animation](#shading-materials-and-animation)
- [Voxel worlds](#voxel-worlds)
- [Bindings](#bindings)
- [Bridges to other crates](#bridges-to-other-crates)
- [Shader and source transpilers](#shader-and-source-transpilers)

## The tree and its laws

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `types` | `SdfNode` (the shape tree: primitives, operations, transforms, modifiers), `SdfTree`, `SdfCategory` |  | [`hello_sphere`](../examples/hello_sphere.rs) |
| `primitives` | distance functions of the primitive shapes, including the 2D shapes extruded to 3D and the TPMS surfaces |  | [`hello_sphere`](../examples/hello_sphere.rs) |
| `operations` | CSG operations: union, intersection, subtraction and their smooth, chamfer, stairs, exponential and column variants, XOR, morph |  | [`csg_operations`](../examples/csg_operations.rs) |
| `transforms` | translate, rotate, scale, non-uniform scale, projective transform, lattice deformation, skinning |  | [`new_transforms`](../examples/new_transforms.rs) |
| `modifiers` | twist, bend, repetition, mirror, revolution, extrusion, displacement, symmetry folds, IFS and other domain modifiers |  | [`new_modifiers_demo`](../examples/new_modifiers_demo.rs) |
| `shell` | variable-thickness offset surface (shell) |  |  |
| `morphology` | opening and closing of a field (removing thin features and filling small gaps) |  |  |
| `sdf2d` | 2D distance functions (circle, rectangle, Bézier, glyph outlines) with bilinear sampling |  |  |
| `optimize` | tree simplification: identity transforms and modifiers removed, nested transforms merged |  |  |
| `diff` | structural diff and patch between two trees (undo / redo, network sync) |  |  |
| `constraint` | Gauss-Newton solver for parametric constraints (fixed, distance, sum, ratio) |  |  |
| `prelude` | commonly used types and functions in one import |  |  |

## Evaluation

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `eval` | tree evaluator (`eval`), normals, gradients and the Lipschitz bound of a tree |  | [`hello_sphere`](../examples/hello_sphere.rs) |
| `compiled` | `CompiledSdf`: the tree flattened to bytecode, with scalar, 8-wide SIMD and BVH evaluators; the JIT and the transpilers live under it (see [below](#shader-and-source-transpilers)) |  |  |
| `soa` | structure-of-arrays point buffers for batch SIMD evaluation |  |  |
| `incremental` | changes a parameter of a `CompiledSdf` in place, without recompiling |  | [`incremental_param_index`](../examples/incremental_param_index.rs) |
| `interval` | interval arithmetic over a box, with bounds rounded outward |  |  |
| `autodiff` | forward-mode automatic differentiation with dual numbers; Hessian and mean curvature |  |  |
| `tight_aabb` | the smallest box that contains the surface, from interval bounds propagated through the tree |  |  |
| `fidelity` | what a field's distance claim is worth: where it is a true distance and where only a bound |  |  |
| `raycast` | sphere tracing, including relaxed and Lipschitz-adaptive stepping |  |  |
| `neural` | small MLP trained to approximate a tree, in pure Rust |  |  |

## Analysis and validation

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `validity` | printability: an erosion proof with a three-valued verdict, wall thickness per triangle, overhang |  |  |
| `measure` | volume, surface area and centre of mass by Monte Carlo with a fixed seed, with the standard error |  |  |
| `collision` | contact between two fields on a grid, with interval pruning |  |  |
| `heatmap` | cross-section images of a field with four colour maps |  |  |
| `llm_schema` | JSON schema of the tree format, and validation of trees produced by a language model |  |  |

## Meshing and file formats

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `mesh` | marching cubes (uniform and adaptive), dual contouring, decimation, LOD chains, manifold repair, UV unwrap, meshlets, meshopt-compatible codecs; GPU marching cubes with `gpu` |  | [`export_mesh`](../examples/export_mesh.rs) |
| `io` | `.asdf` / `.asdf.json` trees; OBJ, glTF (`.glb`), FBX, USD, Alembic, STL, PLY, 3MF, STEP, IGES, MagicaVoxel, Gaussian splat, Unity and UE5 mesh export; OpenVDB with `openvdb` |  | [`export_mesh`](../examples/export_mesh.rs) |
| `cache` | mesh cache keyed by the tree hash, with LRU eviction |  |  |

## Shading, materials and animation

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `material` | PBR materials (metallic-roughness) |  |  |
| `npr` | non-photorealistic shading built from the field, its normal and the light and view directions |  | [`npr_toon_demo`](../examples/npr_toon_demo.rs) |
| `animation` | keyframe tracks over node parameters |  |  |
| `texture` | fits a bitmap texture with procedural noise formulas | `texture-fit` |  |
| `volume` | bakes a field into a 3D texture | `volume` |  |
| `gi` | cone-traced global illumination over a sparse voxel octree | `gi` |  |

## Voxel worlds

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `svo` | sparse voxel octree built from a field | `svo` |  |
| `destruction` | voxel destruction (carving, debris) | `destruction` |  |
| `terrain` | heightmap terrain with erosion and caves | `terrain` |  |

## Bindings

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `ffi` | C ABI used by the C / C++ header, Unity, Unreal Engine and the mobile packages | `ffi` |  |
| `python` | Python bindings (PyO3 + NumPy) | `python` |  |
| `godot` | Godot 4 GDExtension | `godot` |  |
| `wasm` | WebAssembly bindings (built only for `wasm32`) | `wasm` | [`wasm-demo`](../examples/wasm-demo/) |

## Bridges to other crates

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `physics_bridge` | `CompiledSdf` as an `alice_physics::SdfField`, so a shape can be a collider | `physics` (AGPL) |  |
| `sim_bridge` | simulation modifiers of `alice-physics` (thermal, pressure, erosion, fracture, phase change) applied to a field | `physics` (AGPL) |  |
| `codec_bridge` | voxelises a field and compresses the volume with `alice-codec` | `codec` (AGPL) |  |
| `cache_bridge` | caches evaluation results with `alice-cache` | `sdf-cache` (AGPL) |  |
| `asp_bridge` | packs a tree into ALICE Streaming Protocol packets (`libasp`) | `asp` |  |
| `font_bridge` | glyph outlines from `alice-font` to 2D / 3D fields; also needs `--cfg alice_font_bridge` and a local `alice-font` | `font` | [`text_to_3d_demo`](../examples/text_to_3d_demo.rs) |

## Shader and source transpilers

These are sub-modules of `compiled`. Every target is generated from the same
tree; the GPU targets are compared with the CPU evaluator in CI within a
tolerance (see [Determinism](../README.md#determinism)).

| Path | Output | Feature |
|------|--------|---------|
| `compiled::glsl` | GLSL (Unity, OpenGL, Vulkan, Shadertoy) | `glsl` |
| `compiled::wgsl` | WGSL (WebGPU) | `gpu` |
| `compiled::hlsl` | HLSL (Unreal Engine, DirectX) | `hlsl` |
| `compiled::msl` | Metal Shading Language, converted from the WGSL output with naga | `msl` |
| `compiled::blinkscript` | BlinkScript for Nuke | `blinkscript` |
| `compiled::rust` | dependency-free Rust source (`fn sdf`, `fn normal`), bit-identical to `eval_compiled` | `rust` |
| `compiled::jit`, `compiled::jit_simd` | native code through Cranelift (scalar and 8-wide) | `jit` |
