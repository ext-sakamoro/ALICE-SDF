# ALICE-SDF Architecture

## Module Structure

```
alice-sdf/
├── src/
│   ├── types/          # SdfNode enum, operations, transforms
│   ├── eval/           # Tree-walker evaluator
│   ├── compiled/       # Performance-critical execution paths
│   │   ├── mod.rs      # CompiledSdf VM (flat instruction array)
│   │   ├── simd.rs     # 8-wide SIMD batch evaluation
│   │   ├── jit/        # Cranelift JIT compiler
│   │   ├── wgsl/       # WGSL transpiler + GPU evaluator
│   │   ├── hlsl/       # HLSL transpiler (Unreal/DirectX)
│   │   └── glsl/       # GLSL transpiler (Unity/OpenGL)
│   ├── mesh/           # Mesh generation
│   │   ├── sdf_to_mesh.rs     # CPU Marching Cubes
│   │   ├── gpu_marching_cubes.rs  # GPU Marching Cubes (wgpu)
│   │   ├── dual_contouring.rs # Dual Contouring
│   │   ├── nanite.rs          # Nanite-style LOD
│   │   └── lod.rs             # LOD chain generation
│   ├── material.rs     # PBR materials (metallic-roughness)
│   ├── collision.rs    # Contact detection, manifold, CCD
│   ├── autodiff.rs     # Dual numbers, curvature analysis
│   ├── animation.rs    # Morph, keyframes, timeline
│   ├── neural.rs       # Neural SDF (MLP approximation)
│   ├── interval.rs     # Interval arithmetic for pruning
│   ├── constraint.rs   # Gauss-Newton constraint solver
│   ├── sdf2d.rs        # 2D SDF primitives
│   ├── shell.rs        # Shell/offset operations
│   ├── diff.rs         # Tree diffing/patching
│   ├── ffi/            # C ABI bindings
│   └── python/         # PyO3 bindings
├── examples/           # Runnable examples
├── benches/            # Criterion benchmarks
├── bindings/           # C# (Unity) bindings
├── unreal-plugin/      # UE5 plugin source
└── vrchat-package/     # VRChat/UdonSharp package
```

## Evaluation Pipeline

```
SdfNode (tree)
    |
    |-->  eval()              Direct tree-walk (simple, flexible)
    |
    |-->  CompiledSdf         Flat instruction array (cache-friendly)
    |     |-->  eval_compiled()        Single point
    |     |-->  eval_compiled_8wide()  8 points (SIMD)
    |     +-->  eval_compiled_rayon()  Parallel batch
    |
    |-->  JIT (cranelift)     Native machine code (fastest single-point)
    |
    |-->  GpuEvaluator        WebGPU compute shader (millions of points)
    |
    +-->  Transpile           Shader source code output
          |-->  WGSL          WebGPU
          |-->  HLSL          DirectX / Unreal Engine
          +-->  GLSL          OpenGL / Unity
```

## Performance Tiers

| Method | Latency (1 point) | Throughput (1M points) | Use Case |
|--------|-------------------|------------------------|----------|
| eval() | ~200ns | ~5M pts/s | Editing, debugging |
| eval_compiled() | ~50ns | ~20M pts/s | Real-time single query |
| eval_compiled_8wide() | ~80ns/8pts | ~100M pts/s | Batch CPU evaluation |
| JIT | ~30ns | ~30M pts/s | Hot-path single query |
| GPU | ~10ms setup | ~1B pts/s | Massive parallel eval |

## Game Engine Integration

```
                    +--- FFI (.h) --> C/C++ (native)
                    |
alice-sdf (Rust) ---+--- FFI (.cs) --> Unity (C# P/Invoke)
                    |
                    +--- HLSL --> Unreal Engine compute shader
                    |
                    +--- UdonSharp --> VRChat (C# codegen)
```
