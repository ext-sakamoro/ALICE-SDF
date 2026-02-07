# ALICE-SDF Quick Start Guide

Get up and running with ALICE-SDF in 5 minutes.

## Table of Contents

- [Rust (Library)](#rust-library)
- [CLI (Command Line)](#cli-command-line)
- [Python](#python)
- [Unity](#unity)
- [Unreal Engine 5](#unreal-engine-5)
- [VRChat](#vrchat)
- [Godot](#godot)
- [WebAssembly (Browser)](#webassembly-browser)

---

## Rust (Library)

### 1. Add Dependency

```bash
cargo add alice-sdf
```

### 2. Create an SDF Shape

```rust
use alice_sdf::prelude::*;

fn main() {
    // Create a sphere with a box subtracted
    let shape = SdfNode::sphere(1.0)
        .subtract(SdfNode::box3d(0.5, 0.5, 0.5));

    // Evaluate distance at a point
    let distance = eval(&shape, Vec3::new(0.5, 0.0, 0.0));
    println!("Distance: {}", distance);
}
```

### 3. Generate a Mesh

```rust
use alice_sdf::prelude::*;

fn main() {
    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);

    // Generate mesh at resolution 64
    let config = MarchingCubesConfig::aaa(64);
    let mesh = sdf_to_mesh(
        &shape,
        Vec3::splat(-2.0),
        Vec3::splat(2.0),
        &config,
    );

    println!("Vertices: {}, Triangles: {}",
        mesh.vertex_count(), mesh.triangle_count());

    // Export to glTF
    export_glb(&mesh, "output.glb", &GltfConfig::aaa(), None).unwrap();
}
```

### 4. Use Materials

```rust
use alice_sdf::prelude::*;

fn main() {
    let mut mat_lib = MaterialLibrary::new();
    let gold_id = mat_lib.add(Material::metal("Gold", 1.0, 0.766, 0.336, 0.3));

    let shape = SdfNode::sphere(1.0).with_material(gold_id);

    let mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0),
        &MarchingCubesConfig::aaa(64));

    export_glb(&mesh, "gold_sphere.glb", &GltfConfig::aaa(), Some(&mat_lib)).unwrap();
}
```

---

## CLI (Command Line)

### 1. Build

```bash
cd ALICE-SDF
cargo build --release --features cli
```

### 2. Create a Demo Shape

```bash
cargo run --release -- demo -o demo.asdf
```

### 3. Convert to Mesh

```bash
cargo run --release -- to-mesh demo.asdf -o model.obj --resolution 64
```

### 4. Inspect an ASDF File

```bash
cargo run --release -- info demo.asdf
```

### 5. Benchmark

```bash
cargo run --release --features "cli,jit" -- bench demo.asdf --points 1000000
```

---

## Python

### 1. Install

```bash
pip install alice-sdf
# Or build from source:
cd ALICE-SDF
maturin develop --features python
```

### 2. Create and Evaluate

```python
import alice_sdf as sdf
import numpy as np

# Create shape
sphere = sdf.SdfNode.sphere(1.0)
box = sdf.SdfNode.box3d(0.5, 0.5, 0.5)
shape = sphere.subtract(box)

# Evaluate at a single point
distance = shape.eval(0.5, 0.0, 0.0)
print(f"Distance: {distance}")

# Batch evaluate (NumPy)
points = np.random.randn(10000, 3).astype(np.float32)
distances = sdf.eval_batch(shape, points)
print(f"Inside: {(distances < 0).sum()} points")
```

### 3. Generate Mesh

```python
vertices, indices = sdf.to_mesh(
    shape,
    bounds_min=(-2.0, -2.0, -2.0),
    bounds_max=(2.0, 2.0, 2.0),
    resolution=64
)
print(f"Vertices: {vertices.shape}, Indices: {indices.shape}")
```

### 4. Save / Load

```python
sdf.save_sdf(shape, "shape.asdf")
loaded = sdf.load_sdf("shape.asdf")
```

See [Python Guide](PYTHON_GUIDE.md) for Blender integration and advanced usage.

---

## Unity

### 1. Build Rust Library

```bash
cd ALICE-SDF
cargo build --release --features unity
```

### 2. Copy Plugin to Unity

```bash
# macOS
cp target/release/libalice_sdf.dylib unity-sdf-universe/Assets/Plugins/

# Windows
copy target\release\alice_sdf.dll unity-sdf-universe\Assets\Plugins\

# Linux
cp target/release/libalice_sdf.so unity-sdf-universe/Assets/Plugins/
```

### 3. Use in C# Script

```csharp
using UnityEngine;

public class SdfExample : MonoBehaviour
{
    void Start()
    {
        // Create shape
        var sphere = AliceSdf.Sphere(1.0f);
        var box = AliceSdf.Box(0.5f, 0.5f, 0.5f);
        var shape = AliceSdf.SmoothUnion(sphere, box, 0.2f);

        // Compile for fast evaluation
        var compiled = AliceSdf.Compile(shape);

        // Evaluate distance
        float dist = AliceSdf.EvalCompiled(compiled, transform.position);
        Debug.Log($"Distance: {dist}");

        // Clean up
        AliceSdf.FreeCompiled(compiled);
        AliceSdf.Free(shape);
        AliceSdf.Free(sphere);
        AliceSdf.Free(box);
    }
}
```

### 4. Open Demo

Open `unity-sdf-universe/` in Unity 2022.3+, load `Assets/Scenes/SdfUniverse.unity`, press Play.

See [Unity Setup Guide](../unity-sdf-universe/SETUP_GUIDE.md) for detailed instructions.

---

## Unreal Engine 5

### 1. Build Plugin DLL

```bash
cd ALICE-SDF

# Windows (UE5 primary platform)
cargo build --release --features unreal --target x86_64-pc-windows-msvc
# Output: target/x86_64-pc-windows-msvc/release/alice_sdf.dll

# macOS
cargo build --release --features unreal
# Output: target/release/libalice_sdf.dylib

# Linux
cargo build --release --features unreal --target x86_64-unknown-linux-gnu
# Output: target/x86_64-unknown-linux-gnu/release/libalice_sdf.so
```

### 2. Set Up Plugin

Copy `unreal-plugin/AliceSDF/` to your UE5 project's `Plugins/` directory, then copy the built DLL to `Plugins/AliceSDF/Binaries/`.

### 3. Generate HLSL Shader

```rust
use alice_sdf::prelude::*;
use alice_sdf::compiled::{HlslShader, HlslTranspileMode};

let shape = SdfNode::sphere(1.0)
    .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);

let shader = HlslShader::transpile(&shape, HlslTranspileMode::Hardcoded);
let hlsl_code = shader.to_ue5_custom_node();
// Paste into UE5 Custom Material Expression node
```

See [Unreal Engine Guide](UNREAL_ENGINE.md) for detailed setup.

---

## VRChat

### 1. Import Package

In Unity (VRChat Creator Companion), import `vrchat-package/` as a local package.

### 2. Use ALICE-Baker

1. Open `Window > ALICE-SDF > Baker`
2. Drag your `.asdf.json` file into the inspector
3. Click **Bake** to generate shader + Udon + prefab

### 3. Drop into World

Drag the generated prefab into your VRChat world scene. Player collision and VR interactions work automatically.

See `vrchat-package/README.md` for full documentation.

---

## Godot

### 1. Export glTF from ALICE-SDF

```bash
# CLI approach (outputs OBJ format)
cargo run --release -- to-mesh input.asdf -o model.obj --resolution 64

# Or from Rust code (exports glTF binary)
export_glb(&mesh, "model.glb", &GltfConfig::aaa(), Some(&mat_lib)).unwrap();
```

### 2. Import in Godot

Drag `model.glb` into your Godot project. Godot natively supports glTF 2.0 with PBR materials.

### 3. Real-time SDF (GDExtension)

Link `libalice_sdf` via GDExtension for real-time SDF evaluation. See [Godot Guide](GODOT_GUIDE.md).

---

## WebAssembly (Browser)

### 1. Build

```bash
cd ALICE-SDF/examples/wasm-demo
wasm-pack build --target web
```

### 2. Serve

```bash
python3 -m http.server 8080
# Open http://localhost:8080
```

### 3. Use in JavaScript

```javascript
import init, { SdfEvaluator } from './pkg/alice_sdf_wasm.js';

await init();

const evaluator = new SdfEvaluator("sphere");
evaluator.set_params(1.0, 0.0);  // radius
const distance = evaluator.eval(0.5, 0.0, 0.0);
```

See [WASM Guide](WASM_GUIDE.md) for WebGPU integration and deployment.

---

## Next Steps

- [Architecture](ARCHITECTURE.md) - Understand the 13-layer design
- [API Reference](API_REFERENCE.md) - Complete function reference
- [Performance Tuning](../README.md#performance) - Benchmark and optimize

---

Author: Moroya Sakamoto
