# ALICE-SDF Architecture

Deep dive into the 13-layer architecture of ALICE-SDF.

## Design Philosophy

> "Don't send polygons. Send the law of shapes."

ALICE-SDF represents 3D geometry as mathematical functions (SDFs) rather than polygon meshes. This enables:

- **10-1000x compression** vs. traditional mesh formats
- **Infinite resolution** at any zoom level
- **Exact boolean operations** without mesh intersection artifacts
- **Unified evaluation** across CPU, GPU, and WASM targets

---

## 13-Layer Architecture

Every SDF primitive, operation, and modifier is implemented across all 13 layers:

```
                    ┌─────────────────────────────────┐
                    │         SdfNode (AST)            │  Layer 1: Type System
                    │   Sphere | Box | Union | ...     │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       Primitives (Math)          │  Layer 2: SDF Formulas
                    │   eval_sphere(), eval_box()      │  (Inigo Quilez)
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     Recursive Interpreter        │  Layer 3: Tree Walk
                    │   eval(&node, point) → f32       │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────▼────────────────────────┐
              │              Compiler Pipeline              │
              │  ┌────────┐  ┌────────┐  ┌──────────────┐  │
              │  │ OpCode │→ │ Instr  │→ │   Compiler   │  │  Layers 4-6
              │  │ Enum   │  │ 32-byte│  │  AST → Code  │  │
              │  └────────┘  └────────┘  └──────────────┘  │
              └────────────────────┬────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────────┐
        │                  Evaluation Engines                      │
        │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
        │  │ VM Eval  │ │ SIMD 8x  │ │ BVH Eval │ │ SoA Eval │   │  Layers 7-9
        │  │ (Stack)  │ │ (Vec3x8) │ │ (AABB)   │ │ (Cache)  │   │
        │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
        └────────────────────┬────────────────────────────────────┘
                              │
        ┌─────────────────────▼───────────────────────────────┐
        │              Shader Transpilers                      │
        │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │
        │  │   GLSL   │ │   WGSL   │ │   HLSL   │             │  Layers 10-12
        │  │ (Unity)  │ │ (WebGPU) │ │ (UE5/DX) │             │
        │  └──────────┘ └──────────┘ └──────────┘             │
        └─────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────▼───────────────────────────────┐
        │              JIT Native Code                         │
        │  ┌──────────────────────────────────────────────┐   │
        │  │  Cranelift → x86-64 / AArch64 Machine Code   │   │  Layer 13
        │  │  AVX2 / NEON SIMD native instructions         │   │
        │  └──────────────────────────────────────────────┘   │
        └─────────────────────────────────────────────────────┘
```

### Layer Details

| Layer | File | Description | Key Types |
|-------|------|-------------|-----------|
| 1 | `types.rs` | SDF node AST with 13 primitives, 6 operations, 3 transforms, 7 modifiers | `SdfNode`, `SdfTree`, `Aabb`, `Ray`, `Hit` |
| 2 | `primitives/` | Mathematical distance functions (Inigo Quilez formulas) | `eval_sphere()`, `eval_box3d()`, etc. |
| 3 | `eval/` | Recursive tree-walk interpreter with parallel batch | `eval()`, `eval_batch_parallel()` |
| 4 | `compiled/opcode.rs` | Instruction set (37 opcodes) | `OpCode` enum |
| 5 | `compiled/instruction.rs` | 32-byte aligned instruction encoding | `Instruction` struct |
| 6 | `compiled/compiler.rs` | AST to instruction compiler (single-pass) | `CompiledSdf::compile()` |
| 7 | `compiled/eval.rs` | Stack-based VM evaluator | `eval_compiled()` |
| 8 | `compiled/eval_simd.rs` | SIMD 8-wide evaluator (Vec3x8, AVX2/NEON) | `eval_compiled_simd()` |
| 9 | `compiled/eval_bvh.rs` | BVH spatial pruning evaluator | `eval_compiled_bvh()` |
| 10 | `compiled/glsl/` | GLSL transpiler (Unity, OpenGL, Vulkan) | `GlslShader` |
| 11 | `compiled/wgsl/` | WGSL transpiler (WebGPU) + GPU compute | `WgslShader`, `GpuEvaluator` |
| 12 | `compiled/hlsl/` | HLSL transpiler (UE5, DirectX) | `HlslShader` |
| 13 | `compiled/jit/` | JIT native code via Cranelift | `JitEvaluator` |

---

## Data Flow

### Shape Creation to Evaluation

```
User Code                    Internal                      Output
---------                    --------                      ------

SdfNode::sphere(1.0)    →   SdfNode::Sphere{r=1.0}
  .subtract(box)         →   SdfNode::Subtraction{a,b}
  .twist(0.5)            →   SdfNode::Twist{child,s=0.5}
                                     │
                         ┌───────────┼───────────┐
                         │           │           │
                    Interpreted  Compiled    Transpiled
                         │           │           │
                    eval(&node,p) compile()  to_hlsl()
                         │           │           │
                    → f32         → CompiledSdf  → String
                                     │
                              eval_compiled()
                                     │
                                  → f32
```

### Mesh Pipeline

```
SdfNode
  │
  ├── sdf_to_mesh() ──────────────────────┐
  │   (Marching Cubes, Z-slab parallel)    │
  │                                        ▼
  ├── adaptive_marching_cubes() ──────→  Mesh
  │   (Octree, surface-adaptive)         │ vertices: Vec<Vertex>
  │                                      │ indices: Vec<u32>
  │                                      │
  │                              ┌───────┼────────────────────┐
  │                              │       │                    │
  │                        Processing  Physics          Export
  │                              │       │                    │
  │                     ┌────────┤    ┌──┴───┐         ┌──────┤
  │                     │        │    │      │         │      │
  │                 decimate  optimize │   convex   export  export
  │                     │     cache   │   decomp    _glb   _fbx
  │                 validate  dedup  aabb  v-hacd    │      │
  │                  repair   lmap  bsphere hull    .glb   .fbx
  │                     │        │    │      │
  │                     ▼        ▼    ▼      ▼
  │                  Mesh     Mesh  Collision Physics
  │                 (clean) (optimized) Data   Data
  │
  ├── mesh_to_sdf() ──────→ SdfNode (capsule tree)
  └── mesh_to_sdf_exact() ─→ MeshSdf (BVH queries)
```

---

## Module Map

```
src/
├── lib.rs                 # Prelude, re-exports, integration tests
├── types.rs               # SdfNode enum, SdfTree, Aabb, Ray, Hit
├── material.rs            # PBR material system, MaterialLibrary
├── animation.rs           # Keyframe, Track, Timeline, AnimatedSdf
├── soa.rs                 # SoA memory layout (SoAPoints, SoADistances)
│
├── primitives/            # Layer 2: Mathematical SDF formulas
│   ├── mod.rs
│   ├── sphere.rs          # |p| - r
│   ├── box3d.rs           # max(|p| - h, 0) + min(max(|px|-hx, ...), 0)
│   ├── cylinder.rs        # Capped cylinder
│   ├── torus.rs           # length(vec2(length(p.xz)-R, p.y)) - r
│   ├── capsule.rs         # Line segment distance - radius
│   ├── plane.rs           # dot(p, n) - d
│   ├── cone.rs            # Finite cone
│   ├── ellipsoid.rs       # Approximate ellipsoid
│   ├── rounded_cone.rs    # Rounded cone (two radii)
│   ├── pyramid.rs         # Square pyramid
│   ├── octahedron.rs      # Regular octahedron
│   ├── hex_prism.rs       # Hexagonal prism
│   └── link.rs            # Chain link
│
├── operations/            # Boolean CSG
│   ├── union.rs           # min(a, b)
│   ├── intersection.rs    # max(a, b)
│   ├── subtraction.rs     # max(a, -b)
│   └── smooth.rs          # Polynomial smooth min/max
│
├── transforms/            # Spatial transforms
│   ├── translate.rs       # p - offset
│   ├── rotate.rs          # quat * p
│   └── scale.rs           # eval(p/s) * s
│
├── modifiers/             # Domain deformations
│   ├── twist.rs           # Rotate XZ by Y*strength
│   ├── bend.rs            # Curve X axis
│   ├── repeat.rs          # Modulo-based repetition
│   ├── noise.rs           # Perlin noise displacement
│   ├── extrude.rs         # 2D → 3D extrusion
│   ├── revolution.rs      # 2D → 3D revolution
│   └── mirror.rs          # Axis symmetry
│
├── eval/                  # Layer 3: Evaluation
│   ├── mod.rs             # eval(), normal(), gradient()
│   └── parallel.rs        # eval_batch_parallel(), eval_grid()
│
├── raycast/               # Ray-SDF intersection
│   ├── mod.rs             # raycast(), ambient_occlusion(), soft_shadow()
│   └── march.rs           # raymarch(), render_depth(), render_normals()
│
├── mesh/                  # Mesh conversion and processing
│   ├── mod.rs             # Vertex, Triangle, module exports
│   ├── sdf_to_mesh.rs     # Marching cubes + adaptive MC
│   ├── mesh_to_sdf.rs     # Capsule approx + BVH exact
│   ├── bvh.rs             # Mesh BVH for exact SDF
│   ├── hermite.rs         # Hermite data for dual contouring
│   ├── primitive_fitting.rs # Detect geometric primitives
│   ├── nanite.rs          # UE5 Nanite cluster generation
│   ├── lod.rs             # LOD chain (resolution + decimation)
│   ├── decimate.rs        # QEM decimation with attribute preservation
│   ├── manifold.rs        # Mesh validation and repair
│   ├── optimize.rs        # Vertex cache optimization
│   ├── lightmap.rs        # Lightmap UV generation
│   └── collision.rs       # AABB, sphere, convex hull, V-HACD
│
├── io/                    # File I/O
│   ├── mod.rs             # save(), load(), get_info()
│   ├── asdf.rs            # Binary format (.asdf)
│   ├── json.rs            # JSON format (.asdf.json)
│   ├── obj.rs             # Wavefront OBJ import/export
│   ├── gltf.rs            # glTF 2.0 (.glb) export
│   └── fbx.rs             # FBX 7.4 export
│
├── compiled/              # Layers 4-13: Compiled pipeline
│   ├── mod.rs             # Module documentation
│   ├── opcode.rs          # 37 opcodes
│   ├── instruction.rs     # 32-byte instruction encoding
│   ├── compiler.rs        # AST → instructions
│   ├── eval.rs            # Stack VM evaluator
│   ├── eval_simd.rs       # SIMD 8-wide (Vec3x8)
│   ├── eval_soa.rs        # Structure-of-Arrays batch
│   ├── eval_bvh.rs        # BVH-accelerated evaluator
│   ├── aabb.rs            # AABB computation for BVH
│   ├── simd.rs            # Vec3x8 SIMD abstraction
│   ├── glsl/              # GLSL transpiler
│   │   ├── mod.rs
│   │   └── transpiler.rs
│   ├── wgsl/              # WGSL transpiler + GPU compute
│   │   ├── mod.rs
│   │   ├── transpiler.rs
│   │   └── gpu_eval.rs
│   ├── hlsl/              # HLSL transpiler
│   │   ├── mod.rs
│   │   └── transpiler.rs
│   ├── jit/               # JIT native code (scalar + SIMD)
│   │   ├── mod.rs
│   │   ├── codegen.rs     # JIT scalar codegen (all 13 primitives)
│   │   ├── runtime.rs
│   │   └── simd.rs        # JIT SIMD 8-wide (F32X4 dual-lane)
│   └── jit_simd.rs        # JIT SIMD standalone backend (all 13 primitives)
│
├── ffi/                   # C/C++/C# FFI
│   ├── mod.rs
│   ├── api.rs             # 50+ extern "C" functions
│   ├── registry.rs        # Thread-safe handle registry
│   └── types.rs           # FFI type definitions
│
├── python.rs              # PyO3 Python bindings
│
└── bin/
    └── main.rs            # CLI tool
```

---

## Performance Characteristics

### Evaluation Throughput (Apple M3 Max)

```
Layer 3  (Interpreter):   307 M/s    ████████████░░░░░░░░
Layer 7  (VM):            350 M/s    ██████████████░░░░░░
Layer 8  (SIMD):          800 M/s    ████████████████████████████████░░░░
Layer 9  (BVH):           500 M/s*   ████████████████████████░░░░░░░░
Layer 13 (JIT):           977 M/s    ████████████████████████████████████████
GPU (WGSL):               101 M/s**  ████░░░░░░░░░░░░░░░░

* BVH advantage increases with scene complexity
** GPU advantage increases with batch size (>1M points)
```

### Memory Layout

```
SdfNode (AST)        CompiledSdf (VM)      SIMD (Vec3x8)
┌──────────┐         ┌──────────┐          ┌──────────┐
│ enum tag │         │ instr[0] │ 32B      │ x: f32x8 │ 32B
│ children │ heap    │ instr[1] │ 32B      │ y: f32x8 │ 32B
│ params   │ alloc   │ instr[2] │ 32B      │ z: f32x8 │ 32B
│ ...      │         │ ...      │          └──────────┘
└──────────┘         │ stack    │ 16 deep   96 bytes per
Variable size        └──────────┘           8-point batch
                     Flat, cache-friendly
```

### Instruction Format (32 bytes)

```
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ opcode │ flags  │ pad    │ pad    │  param0         │  param1         │
│  u8    │  u8    │  u8    │  u8    │  f32            │  f32            │
├────────┴────────┴────────┴────────┼────────┬────────┼────────┬────────┤
│  param2                           │ param3 │ param4 │ param5 │ pad    │
│  f32                              │  f32   │  f32   │  f32   │  f32   │
└───────────────────────────────────┴────────┴────────┴────────┴────────┘
                                    32 bytes (cache line aligned)
```

---

## Thread Safety

| Component | Thread Safety | Notes |
|-----------|--------------|-------|
| `SdfNode` | `Send + Sync` | Immutable tree, safe to share |
| `CompiledSdf` | `Send + Sync` | Read-only bytecode |
| `eval*` functions | Thread-safe | No mutable state |
| FFI handles | Thread-safe | Protected by `DashMap` |
| `Mesh` (mutable ops) | Single-threaded | `decimate()`, `optimize_vertex_cache()` |
| Rayon parallelism | Automatic | `eval_batch_parallel`, mesh generation |

---

## Adding a New Primitive

To add a new primitive (e.g., `Tetrahedron`), implement across all layers:

1. **Layer 1** (`types.rs`): Add `Tetrahedron { size: f32 }` variant to `SdfNode`
2. **Layer 2** (`primitives/tetrahedron.rs`): Implement `eval_tetrahedron(p, size) -> f32`
3. **Layer 3** (`eval/mod.rs`): Add match arm in `eval()`
4. **Layer 4** (`compiled/opcode.rs`): Add `OpCode::Tetrahedron`
5. **Layer 5** (`compiled/instruction.rs`): Add instruction encoding
6. **Layer 6** (`compiled/compiler.rs`): Add compilation case
7. **Layer 7** (`compiled/eval.rs`): Add VM evaluation case
8. **Layer 8** (`compiled/eval_simd.rs`): Add SIMD evaluation
9. **Layer 9** (`compiled/eval_bvh.rs`): Add AABB computation
10. **Layer 10** (`compiled/glsl/transpiler.rs`): Add GLSL generation
11. **Layer 11** (`compiled/wgsl/transpiler.rs`): Add WGSL generation
12. **Layer 12** (`compiled/hlsl/transpiler.rs`): Add HLSL generation
13. **Layer 13** (`compiled/jit/codegen.rs`): Add JIT code generation

Also update: serialization (`io/`), FFI (`ffi/api.rs`), Python (`python.rs`), tests.

---

## Dependencies

| Crate | Purpose | Version |
|-------|---------|---------|
| `glam` | Fast vector math (Vec3, Quat, Mat4) | 0.29 |
| `rayon` | Data parallelism | 1.10 |
| `wide` | SIMD types (f32x8) | 0.7 |
| `serde` | Serialization framework | 1.0 |
| `bincode` | Binary serialization | 1.3 |
| `thiserror` | Error handling | 2.0 |
| `clap` | CLI argument parsing | 4.5 |
| `pyo3` | Python bindings (optional) | 0.22 |
| `cranelift-*` | JIT compilation (optional) | 0.112 |
| `wgpu` | WebGPU compute (optional) | 23 |
| `lazy_static` | FFI handle registry (optional) | 1.5 |

---

Author: Moroya Sakamoto
