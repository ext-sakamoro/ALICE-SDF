# ALICE-SDF Architecture

## Overview

ALICE-SDF implements a multi-stage evaluation pipeline for Signed Distance Functions. An SDF tree (AST) is compiled into a flat bytecode representation, then evaluated via stack-based VM, SIMD batch, or GPU compute.

```
SdfNode Tree (AST)
     │
     ▼
┌─────────────┐
│  Compiler    │  AST → Instructions + aux_data
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│              CompiledSdf                         │
│  ┌──────────────────┐  ┌──────────────────────┐ │
│  │ instructions[]   │  │ aux_data: Vec<f32>   │ │
│  │ (opcode, params, │  │ (matrices, control   │ │
│  │  aux_offset,     │  │  points, heightmaps) │ │
│  │  aux_len)        │  │                      │ │
│  └──────────────────┘  └──────────────────────┘ │
└──────────┬──────────────────────────────────────┘
           │
     ┌─────┼─────────┬──────────┬──────────┐
     ▼     ▼         ▼          ▼          ▼
  Scalar  SIMD     BVH        JIT       GPU
  eval    8-wide   pruning    native    WGSL
```

## Core Data Structures

### SdfNode (AST)

`src/types/mod.rs` — 126-variant enum representing the SDF tree:

| Category | Count | Examples |
|----------|-------|---------|
| Primitive | 72 | Sphere, Box3d, Torus, Gyroid, Heart, Helix |
| Operation | 24 | Union, SmoothUnion, ChamferSubtraction, XOR, Morph |
| Transform | 7 | Translate, Rotate, Scale, ProjectiveTransform, LatticeDeform, SdfSkinning |
| Modifier | 23 | Twist, Bend, Mirror, IcosahedralSymmetry, IFS, HeightmapDisplacement |

### Instruction

`src/compiled/instruction.rs` — Fixed-size bytecode instruction:

```
┌────────────────────────────────────────────────┐
│ OpCode (u8) │ flags (u8) │ child_count (u16)  │
│ params[7]: [f32; 7]   (28 bytes)              │
│ skip_offset: u32                               │
│ aux_offset: u32  │  aux_len: u32              │
└────────────────────────────────────────────────┘
```

- `params[7]` — inline parameters (radius, smoothness factor, etc.)
- `aux_offset` / `aux_len` — pointer into `CompiledSdf.aux_data` for variable-length data

### Auxiliary Data Buffer

Operations requiring more than 7 floats of parameters serialize their data into `aux_data`:

| Operation | aux_data contents | Size |
|-----------|-------------------|------|
| ProjectiveTransform | inv_matrix [f32; 16] | 16 floats |
| LatticeDeform | [nx, ny, nz, bbox_min(3), bbox_max(3), control_points...] | 9 + N*3 floats |
| SdfSkinning | [bone_count, (inv_bind[16], cur_pose[16], weight) × N] | 1 + N*33 floats |
| IFS | [transform_count, mat4[0..16], mat4[16..32], ...] | 1 + N*16 floats |
| HeightmapDisplacement | [width, height, heightmap_data...] | 2 + W*H floats |

## Evaluation Pipeline

### 1. Interpreted (eval/)

Recursive tree walker. Simplest, used for debugging and as reference implementation.

```
eval(node, point) → match node {
    Sphere { r } → point.length() - r,
    Union { a, b } → min(eval(a, point), eval(b, point)),
    Translate { child, offset } → eval(child, point - offset),
    ...
}
```

### 2. Compiled Scalar (compiled/eval.rs)

Stack-based VM with two stacks:
- **Value stack** — intermediate SDF distances
- **Coordinate stack** — pushed/popped by transforms and modifiers (CoordFrame)

```
for (inst_idx, inst) in instructions {
    match inst.opcode {
        Sphere    → push distance to value_stack
        Union     → pop 2, push min
        Translate → push CoordFrame, modify point
        PopTransform → pop CoordFrame, restore point, apply post-processing
    }
}
```

PopTransform handles post-processing for:
- HeightmapDisplacement (bilinear heightmap sampling)
- SurfaceRoughness (FBM noise addition)
- ProjectiveTransform (Lipschitz correction)

### 3. Compiled SIMD 8-wide (compiled/eval_simd.rs)

Evaluates 8 points simultaneously using `f32x8` / `F32x8` SIMD types.
For operations that cannot be vectorized (IcosahedralSymmetry, IFS, SdfSkinning, etc.), falls back to per-lane scalar dispatch:

```rust
for lane in 0..8 {
    let pt = Vec3::new(p.x[lane], p.y[lane], p.z[lane]);
    let (q, _) = scalar_function(pt, &aux_params);
    result.x[lane] = q.x;
    result.y[lane] = q.y;
    result.z[lane] = q.z;
}
```

### 4. BVH-accelerated (compiled/eval_bvh.rs)

AABB pruning for complex scenes. Skips subtrees where the query point is provably outside the bounding box.

### 5. JIT Native (compiled/jit/)

Cranelift-based JIT compilation to native machine code. Both scalar and SIMD 8-wide variants.

### 6. GPU Compute (compiled/wgsl/)

WGSL compute shaders via wgpu. Used for massive batch evaluation (1M+ points).

## Shader Transpilers

Three transpilers share a common `GenericTranspiler<L: ShaderLang>` framework:

| Target | Module | Output | Use Case |
|--------|--------|--------|----------|
| GLSL | `compiled/glsl/` | `.glsl` / `.hlsl` (Unity) | Unity Shader Graph, OpenGL, Vulkan |
| WGSL | `compiled/wgsl/` | `.wgsl` | WebGPU, wgpu, ALICE-View |
| HLSL | `compiled/hlsl/` | `.hlsl` / `.ush` | UE5 Material Function, DirectX |

Each transpiler walks the SdfNode tree and emits target-language SDF evaluation code with:
- Primitive distance functions
- CSG operations
- Transform coordinate manipulation
- Modifier domain warping

## Mesh Generation

| Method | Module | Description |
|--------|--------|-------------|
| Marching Cubes | `mesh/sdf_to_mesh.rs` | Standard MC with Z-slab parallelization |
| Adaptive MC | `mesh/sdf_to_mesh.rs` | Octree-based, surface-adaptive subdivision |
| Dual Contouring | `mesh/dual_contouring.rs` | QEF vertex placement, sharp edge preservation |
| GPU MC | `mesh/gpu_marching_cubes.rs` | wgpu compute shader Marching Cubes |

## I/O Formats

15 formats supported across import/export. See README.md for the full matrix.

Key binary formats:
- **ASDF** (`io/asdf.rs`) — Native SDF binary with CRC32 integrity
- **ABM** (`io/abm.rs`) — ALICE Binary Mesh with LOD chain support
- **Nanite** (`io/nanite.rs`) — UE5 Nanite hierarchical cluster format

## Module Map

```
src/
├── types/              # SdfNode enum (126 variants), constructors, categories
├── primitives/         # Mathematical SDF formulas (72 primitives)
├── operations/         # CSG operations (24 ops)
├── transforms/         # Spatial transforms (projective, lattice, skinning)
├── modifiers/          # Domain modifiers (IFS, icosahedral, heightmap, roughness)
├── eval/               # Recursive interpreter + gradient
├── compiled/
│   ├── opcode.rs       # OpCode enum
│   ├── instruction.rs  # Instruction struct (params + aux_offset/aux_len)
│   ├── compiler.rs     # AST → bytecode + aux_data serialization
│   ├── eval.rs         # Scalar VM evaluator
│   ├── eval_simd.rs    # SIMD 8-wide evaluator
│   ├── eval_bvh.rs     # BVH-accelerated evaluator
│   ├── eval_soa.rs     # SoA batch evaluator
│   ├── glsl/           # GLSL transpiler
│   ├── wgsl/           # WGSL transpiler + GPU eval
│   ├── hlsl/           # HLSL transpiler
│   ├── jit/            # Cranelift JIT (scalar + SIMD)
│   └── transpiler_common.rs  # Shared transpiler framework
├── mesh/               # Marching Cubes, Dual Contouring, decimation, LOD, BVH
├── io/                 # 15 format importers/exporters
├── neural.rs           # Neural SDF (MLP)
├── interval.rs         # Interval arithmetic evaluation
├── collision.rs        # SDF-to-SDF collision
├── autodiff.rs         # Dual Number forward-mode AD
├── animation.rs        # Keyframe animation system
├── material.rs         # PBR material system
├── python/             # PyO3 bindings
├── ffi/                # C/C++/C# FFI
├── godot/              # Godot GDExtension
└── svo/                # Sparse Voxel Octree
```
