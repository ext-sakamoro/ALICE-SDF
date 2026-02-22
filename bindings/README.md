# ALICE-SDF Unity Integration

C# P/Invoke bindings for ALICE-SDF. 72 primitives, 24 CSG operations, 7 transforms, 23 modifiers (126 total SDF variants) with compiled batch evaluation, shader generation, and mesh export.

Author: Moroya Sakamoto

## Installation

### 1. Copy the C# bindings

Copy `AliceSdf.cs` to your Unity project's `Assets/` folder (or any subfolder).

### 2. Copy the native library

Copy the platform-specific native library to `Assets/Plugins/`:

| Platform | Library | Destination |
|----------|---------|-------------|
| macOS (ARM64) | `libalice_sdf.dylib` | `Assets/Plugins/macOS/` |
| macOS (Intel) | `libalice_sdf.dylib` | `Assets/Plugins/macOS/` |
| Windows x64 | `alice_sdf.dll` | `Assets/Plugins/x86_64/` |
| Linux x64 | `libalice_sdf.so` | `Assets/Plugins/x86_64/` |
| Android | `libalice_sdf.so` | `Assets/Plugins/Android/` |
| iOS | Built-in (`__Internal`) | Link via Xcode |

Pre-built libraries are available on the [GitHub Releases](https://github.com/ext-sakamoro/ALICE-SDF/releases) page.

### 3. Build from source (optional)

```bash
cargo build --release --features unity
```

## Quick Start

```csharp
using AliceSdfUnity;
using UnityEngine;

public class SdfDemo : MonoBehaviour
{
    void Start()
    {
        // Create shapes
        var sphere = AliceSdf.Sphere(1.0f);
        var box = AliceSdf.Box(0.5f, 0.5f, 0.5f);
        var shape = AliceSdf.SmoothUnion(sphere, box, 0.2f);

        // Evaluate distance
        float dist = AliceSdf.Eval(shape, new Vector3(0.5f, 0f, 0f));
        Debug.Log($"Distance: {dist}");

        // Compile for fast batch evaluation
        var compiled = AliceSdf.Compile(shape);
        float distCompiled = AliceSdf.EvalCompiled(compiled, new Vector3(0.5f, 0f, 0f));

        // Generate GLSL for Shader Graph
        string glsl = AliceSdf.ToGlsl(shape);

        // Clean up
        AliceSdf.FreeCompiled(compiled);
        AliceSdf.Free(shape);
        AliceSdf.Free(box);
        AliceSdf.Free(sphere);
    }
}
```

## v1.1.0 New Features

### Advanced Transforms

```csharp
// Projective transform with Matrix4x4
var projected = AliceSdf.ProjectiveTransform(shape, invMatrix, 1.5f);

// Projective transform with raw float array (16 floats, column-major)
float[] mat = new float[16];
var projected2 = AliceSdf.ProjectiveTransform(shape, mat, 1.5f);

// Lattice Free-Form Deformation
uint nx = 4, ny = 4, nz = 4;
float[] controlPoints = new float[nx * ny * nz * 3]; // x,y,z per point
var deformed = AliceSdf.LatticeDeform(shape, controlPoints, nx * ny * nz,
    nx, ny, nz, new Vector3(-2, -2, -2), new Vector3(2, 2, 2));

// Skeletal skinning (33 floats per bone: 16 inv_bind + 16 cur_pose + 1 weight)
float[] boneData = new float[2 * 33]; // 2 bones
var skinned = AliceSdf.Skinning(shape, boneData, 2);
```

### Advanced Modifiers

```csharp
// 120-fold icosahedral symmetry
var gem = AliceSdf.IcosahedralSymmetry(shape);

// IFS fractal (2 transform matrices, 5 iterations)
float[] transforms = new float[2 * 16]; // 2 column-major 4x4 matrices
var fractal = AliceSdf.IFS(shape, transforms, 2, 5);

// Heightmap displacement
float[] heightmap = new float[256 * 256];
var displaced = AliceSdf.HeightmapDisplacement(shape, heightmap, 256, 256, 0.1f, 1.0f);

// FBM surface roughness
var rough = AliceSdf.SurfaceRoughness(shape, 4.0f, 0.05f, 4);
```

## Batch Evaluation

For best performance, compile once and evaluate many points:

```csharp
var compiled = AliceSdf.Compile(shape);

// Batch evaluate (SIMD + multi-threaded on Rust side)
Vector3[] points = new Vector3[10000];
// ... fill points ...
float[] distances = AliceSdf.EvalCompiledBatch(compiled, points);

AliceSdf.FreeCompiled(compiled);
```

## Shader Generation

Generate shader code for Unity Shader Graph Custom Function nodes:

```csharp
// GLSL for Unity Shader Graph (HDRP/URP)
string glsl = AliceSdf.ToGlsl(shape);

// HLSL for Custom Function nodes
string hlsl = AliceSdf.ToHlsl(shape);

// WGSL for WebGPU
string wgsl = AliceSdf.ToWgsl(shape);
```

## Mesh Export

```csharp
AliceSdf.ExportObj(shape, "Assets/output.obj");
AliceSdf.ExportGlb(shape, "Assets/output.glb");
AliceSdf.ExportUsda(shape, "Assets/output.usda");
AliceSdf.ExportFbx(shape, "Assets/output.fbx");
```

## File I/O

```csharp
// Save SDF tree to .asdf file
AliceSdf.Save(shape, "Assets/shape.asdf");

// Load SDF tree from .asdf file
var loaded = AliceSdf.Load("Assets/shape.asdf");
```

## API Summary

| Category | Count | Key Methods |
|----------|-------|-------------|
| Primitives | 72 | Sphere, Box, Torus, Gyroid, Heart, Helix, ... |
| Operations | 24 | Union, SmoothUnion, Chamfer, Morph, Pipe, Groove, ... |
| Transforms | 7 | Translate, Rotate, Scale, ProjectiveTransform, LatticeDeform, Skinning |
| Modifiers | 23 | Twist, Bend, Mirror, IcosahedralSymmetry, IFS, HeightmapDisplacement, ... |
| Evaluation | 4 | Eval, EvalBatch, EvalCompiled, EvalCompiledBatch |
| Shader | 3 | ToWgsl, ToHlsl, ToGlsl |
| Export | 4 | ExportObj, ExportGlb, ExportUsda, ExportFbx |
| I/O | 2 | Save, Load |
| Memory | 3 | Free, FreeCompiled, Clone |
