# ALICE-SDF Cookbook

Practical recipes for common SDF tasks.

## Basic Shapes

```rust
use alice_sdf::prelude::*;

let sphere = SdfNode::sphere(1.0);
let box3d = SdfNode::box3d(1.0, 0.5, 0.5);
let cylinder = SdfNode::cylinder(0.5, 2.0);
let torus = SdfNode::torus(1.0, 0.3);
let plane = SdfNode::plane(0.0, 1.0, 0.0, 0.0); // Y-up
```

## CSG Operations

```rust
// Boolean
let union = a.clone().union(b.clone());
let intersection = a.clone().intersection(b.clone());
let subtraction = a.clone().subtract(b.clone());

// Smooth blending (k = blend radius)
let smooth = a.smooth_union(b, 0.3);
```

## Transforms

```rust
let moved = shape.translate(1.0, 2.0, 0.0);
let rotated = shape.rotate_y(std::f32::consts::FRAC_PI_4);
let scaled = shape.scale(2.0);
```

## Evaluate Distance

```rust
use alice_sdf::eval::eval;
use glam::Vec3;

let d = eval(&node, Vec3::new(1.0, 0.0, 0.0));
// d < 0: inside, d == 0: surface, d > 0: outside
```

## Compiled SDF (Faster Evaluation)

```rust
use alice_sdf::compiled::{CompiledSdf, eval_compiled};
use glam::Vec3;

let compiled = CompiledSdf::compile(&node);
let d = eval_compiled(&compiled, Vec3::new(1.0, 0.0, 0.0));
```

## SIMD Batch Evaluation (8 Points at Once)

```rust
use alice_sdf::compiled::eval_compiled_8wide;

let xs = [0.0_f32; 8];
let ys = [0.0; 8];
let zs = [1.0; 8];
let distances = eval_compiled_8wide(&compiled, &xs, &ys, &zs);
```

## GPU Evaluation (Millions of Points)

```rust
// Requires feature "gpu"
use alice_sdf::compiled::GpuEvaluator;
use glam::Vec3;

let gpu = GpuEvaluator::new(&node)?;
let points: Vec<Vec3> = /* your points */;
let distances = gpu.eval_batch(&points)?;
```

## Mesh Generation (Marching Cubes)

```rust
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig};
use glam::Vec3;

let config = MarchingCubesConfig {
    resolution: 128,
    ..Default::default()
};
let mesh = sdf_to_mesh(&node, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
println!("Vertices: {}, Triangles: {}", mesh.vertices.len(), mesh.indices.len() / 3);
```

## Shader Transpilation

```rust
// WGSL (WebGPU) — requires feature "gpu"
use alice_sdf::compiled::{WgslShader, TranspileMode};
let wgsl = WgslShader::transpile(&node, TranspileMode::Hardcoded);
println!("{}", wgsl.code);

// HLSL (DirectX/Unreal) — requires feature "hlsl"
use alice_sdf::compiled::{HlslShader, HlslTranspileMode};
let hlsl = HlslShader::transpile(&node, HlslTranspileMode::Hardcoded);

// GLSL (OpenGL/Unity) — requires feature "glsl"
use alice_sdf::compiled::{GlslShader, GlslTranspileMode};
let glsl = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
```

## Materials (PBR)

```rust
use alice_sdf::material::{Material, StandardMaterials, material_lerp};

let gold = StandardMaterials::gold();
let chrome = StandardMaterials::chrome();
let blend = material_lerp(&gold, &chrome, 0.5);
```

## Collision Detection

```rust
use alice_sdf::collision::{sdf_contact_points, compute_manifold, sdf_ccd};
use glam::Vec3;

// Contact points between two SDF shapes
let contacts = sdf_contact_points(&node_a, &node_b, resolution);

// Contact manifold
let manifold = compute_manifold(&contacts);

// Continuous collision detection (ray vs SDF)
let toi = sdf_ccd(&node, Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 10.0);
```

## Automatic Differentiation

```rust
use alice_sdf::autodiff::{eval_with_gradient, principal_curvatures, gaussian_curvature};
use glam::Vec3;

let (distance, gradient) = eval_with_gradient(&node, Vec3::new(1.0, 0.0, 0.0));
let (k1, k2) = principal_curvatures(&node, Vec3::new(1.0, 0.0, 0.0));
let kg = gaussian_curvature(&node, Vec3::new(1.0, 0.0, 0.0));
```

## Animation / Morph

```rust
use alice_sdf::animation::{morph, Timeline, Track, Keyframe};

// Morph between two shapes
let morphed = morph(&sphere, &box3d, 0.5); // 50% blend

// Keyframe animation
let track = Track::new("blend")
    .add_keyframe(Keyframe::new(0.0, 0.0))
    .add_keyframe(Keyframe::new(1.0, 1.0));
let timeline = Timeline::new().add_track(track);
```

## JSON Serialization

```rust
// Save
let json = serde_json::to_string_pretty(&node).unwrap();
std::fs::write("shape.asdf.json", &json).unwrap();

// Load
let json = std::fs::read_to_string("shape.asdf.json").unwrap();
let node: SdfNode = serde_json::from_str(&json).unwrap();
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| (default) | Core SDF eval, mesh, compiled VM, SIMD |
| `jit` | Cranelift JIT compilation |
| `gpu` | WebGPU evaluation + WGSL transpiler |
| `hlsl` | HLSL shader transpiler |
| `glsl` | GLSL shader transpiler |
| `ffi` | C/C++/C# FFI bindings |
| `unity` | Unity integration (FFI + GLSL) |
| `unreal` | Unreal Engine integration (FFI + HLSL) |
| `aaa` | All premium features |
