# ALICE-SDF Unreal Engine 5 Integration Guide

Complete guide for integrating ALICE-SDF with Unreal Engine 5.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Building the Plugin](#building-the-plugin)
- [Plugin Installation](#plugin-installation)
- [C++ Integration](#c-integration)
- [HLSL Shader Integration](#hlsl-shader-integration)
- [Blueprint Integration](#blueprint-integration)
- [Material Setup](#material-setup)
- [Collision Setup](#collision-setup)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Overview

ALICE-SDF integrates with UE5 through two channels:

1. **C FFI Plugin** - Native `.dll`/`.so` library linked via `alice_sdf.h` for runtime SDF evaluation (particles, physics, procedural generation)
2. **HLSL Transpiler** - Generate HLSL code that runs directly in UE5 Custom Material Expression nodes for GPU raymarching

### Architecture

```
ALICE-SDF (Rust)
    |
    +-- cargo build --features unreal
    |       |
    |       +-- alice_sdf.dll/so/dylib  (C FFI)
    |       +-- HLSL code generation    (Shader)
    |
    v
UE5 Project
    |
    +-- Plugins/AliceSDF/
    |       +-- Source/AliceSDF/  (C++ wrapper)
    |       +-- Binaries/        (native library)
    |       +-- include/         (alice_sdf.h)
    |
    +-- Materials/
            +-- Custom Expression nodes (HLSL)
```

---

## System Requirements

| Requirement | Version |
|-------------|---------|
| Unreal Engine | 5.3+ |
| Rust | 1.75+ |
| Visual Studio | 2022 (Windows) |
| Xcode | 15+ (macOS) |
| Target | Win64, Linux, macOS |

---

## Building the Plugin

### Windows (Primary)

```bash
cd ALICE-SDF

# Build the DLL
cargo build --release --features unreal --target x86_64-pc-windows-msvc

# The DLL is at:
# target/x86_64-pc-windows-msvc/release/alice_sdf.dll
```

### macOS

```bash
cargo build --release --features unreal

# Output: target/release/libalice_sdf.dylib
```

### Linux

```bash
cargo build --release --features unreal --target x86_64-unknown-linux-gnu

# Output: target/x86_64-unknown-linux-gnu/release/libalice_sdf.so
```

### Cross-Compilation (macOS to Windows)

```bash
# Install cross-compiler
rustup target add x86_64-pc-windows-msvc

# Build (requires Windows SDK or cross-linker)
cargo build --release --features unreal --target x86_64-pc-windows-msvc
```

---

## Plugin Installation

### 1. Copy Plugin Structure

Copy `unreal-plugin/AliceSDF/` to your UE5 project:

```
YourProject/
  Plugins/
    AliceSDF/
      AliceSDF.uplugin
      Source/
        AliceSDF/
          AliceSDF.Build.cs
          Public/
            AliceSdfBlueprintLibrary.h
          Private/
            AliceSdfBlueprintLibrary.cpp
      Binaries/
        Win64/
          alice_sdf.dll
      include/
        alice_sdf.h
```

### 2. Copy the Built Library

```bash
# Windows
copy target\release\alice_sdf.dll YourProject\Plugins\AliceSDF\Binaries\Win64\

# macOS
cp target/release/libalice_sdf.dylib YourProject/Plugins/AliceSDF/Binaries/Mac/

# Linux
cp target/release/libalice_sdf.so YourProject/Plugins/AliceSDF/Binaries/Linux/
```

### 3. Regenerate Project Files

```bash
# Windows (from UE5 installation)
"C:\Program Files\Epic Games\UE_5.4\Engine\Build\BatchFiles\GenerateProjectFiles.bat" YourProject.uproject

# macOS
/Users/Shared/Epic\ Games/UE_5.4/Engine/Build/BatchFiles/Mac/GenerateProjectFiles.sh YourProject.uproject
```

### 4. Enable Plugin

In UE5 Editor: `Edit > Plugins > Search "AliceSDF" > Enable > Restart Editor`

---

## C++ Integration

### Include Header

```cpp
#include "alice_sdf.h"
```

### Basic Usage

```cpp
#include "alice_sdf.h"

void AMyActor::BeginPlay()
{
    Super::BeginPlay();

    // Create SDF shape
    SdfHandle Sphere = alice_sdf_sphere(1.0f);
    SdfHandle Box = alice_sdf_box(0.5f, 0.5f, 0.5f);
    SdfHandle Shape = alice_sdf_smooth_union(Sphere, Box, 0.2f);

    // Compile for fast evaluation
    CompiledHandle Compiled = alice_sdf_compile(Shape);

    // Evaluate at actor position
    FVector Pos = GetActorLocation() / 100.0f; // UE uses cm, SDF uses meters
    float Distance = alice_sdf_eval_compiled(Compiled, Pos.X, Pos.Y, Pos.Z);
    UE_LOG(LogTemp, Log, TEXT("SDF Distance: %f"), Distance);

    // Clean up
    alice_sdf_free_compiled(Compiled);
    alice_sdf_free(Shape);
    alice_sdf_free(Box);
    alice_sdf_free(Sphere);
}
```

### Batch Evaluation (Particles/Physics)

```cpp
void AMyActor::BatchEvaluate(const TArray<FVector>& Points, TArray<float>& OutDistances)
{
    const int32 Count = Points.Num();
    OutDistances.SetNum(Count);

    // Convert to flat array (SoA layout for maximum performance)
    TArray<float> X, Y, Z;
    X.SetNum(Count);
    Y.SetNum(Count);
    Z.SetNum(Count);

    for (int32 i = 0; i < Count; ++i)
    {
        X[i] = Points[i].X / 100.0f;
        Y[i] = Points[i].Y / 100.0f;
        Z[i] = Points[i].Z / 100.0f;
    }

    // Evaluate all points in parallel (SoA path - fastest)
    alice_sdf_eval_soa(
        CompiledSdf,
        X.GetData(), Y.GetData(), Z.GetData(),
        OutDistances.GetData(),
        Count
    );
}
```

### File I/O

```cpp
// Save SDF to file
SdfResult Result = alice_sdf_save(Shape, "Content/SDF/MyShape.asdf");

// Load SDF from file
SdfHandle Loaded = alice_sdf_load("Content/SDF/MyShape.asdf");
```

---

## HLSL Shader Integration

### Generate HLSL from Rust

```rust
use alice_sdf::prelude::*;
use alice_sdf::compiled::HlslShader;

fn main() {
    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2)
        .twist(0.5);

    let shader = HlslShader::transpile(&shape);

    // For UE5 Custom Material Expression
    let ue5_code = shader.to_ue5_custom_node();
    println!("{}", ue5_code);

    // For DirectX compute shader
    let compute_code = shader.to_compute_shader();
    std::fs::write("sdf_compute.hlsl", compute_code).unwrap();
}
```

### Generate HLSL via C FFI

```cpp
// Generate HLSL from C++
StringResult HlslResult = alice_sdf_to_hlsl(Shape);
if (HlslResult.result == SdfResult_Ok)
{
    FString HlslCode = UTF8_TO_TCHAR(HlslResult.data);
    // Use HlslCode in material...
    alice_sdf_free_string(HlslResult.data);
}
```

### Custom Material Expression

1. Create a new Material in UE5
2. Add a **Custom** expression node
3. Paste the generated HLSL code
4. Connect inputs:
   - `WorldPosition` (from World Position node, divided by 100 for cm→m)
5. Connect output to:
   - **Opacity Mask** (for cutout rendering)
   - **World Position Offset** (for deformation)
   - **Custom Data** (for multi-pass)

### Example Material Setup

```
Material: M_SdfRaymarch
  |
  +-- Custom Expression (HLSL code)
  |     Input 0: AbsoluteWorldPosition / 100.0
  |     Output:  float (distance)
  |
  +-- Compare: distance < 0 ? 1 : 0 --> Opacity Mask
  |
  +-- Material Domain: Surface
  +-- Blend Mode: Masked
  +-- Shading Model: Default Lit
```

### Raymarching Material

For full raymarching (not just distance query), use the complete fragment shader:

```hlsl
// In Custom Expression node:
// Input: CameraPosition, RayDirection, MaxSteps (default 128)

float3 ro = CameraPosition / 100.0;  // cm to meters
float3 rd = normalize(RayDirection);

float t = 0.0;
for (int i = 0; i < MaxSteps; i++)
{
    float3 p = ro + rd * t;
    float d = sdf_eval(p);  // Generated SDF function
    if (d < 0.001) break;
    t += d;
    if (t > 100.0) break;
}

return t;
```

---

## Blueprint Integration

The plugin provides Blueprint-callable functions via `UAliceSdfBlueprintLibrary`:

### Available Blueprint Functions

| Function | Description |
|----------|-------------|
| `CreateSphere(Radius)` | Create sphere SDF |
| `CreateBox(HalfExtents)` | Create box SDF |
| `CreateCylinder(Radius, HalfHeight)` | Create cylinder SDF |
| `Union(A, B)` | Boolean union |
| `SmoothUnion(A, B, K)` | Smooth union |
| `Subtract(A, B)` | Boolean subtraction |
| `Translate(Node, Offset)` | Translate SDF |
| `Rotate(Node, Rotation)` | Rotate SDF |
| `Scale(Node, Factor)` | Uniform scale |
| `Compile(Node)` | Compile to bytecode |
| `EvalCompiled(Compiled, Position)` | Evaluate distance |
| `SaveToFile(Node, Path)` | Save to .asdf |
| `LoadFromFile(Path)` | Load from .asdf |
| `GenerateHLSL(Node)` | Generate HLSL code |

### Blueprint Example

```
Event BeginPlay
    |
    +-- Create Sphere (Radius: 1.0) --> Sphere
    +-- Create Box (0.5, 0.5, 0.5)  --> Box
    +-- Smooth Union (Sphere, Box, 0.2) --> Shape
    +-- Compile (Shape) --> Compiled
    |
    +-- Store Compiled as variable for Tick

Event Tick
    |
    +-- Get Actor Location --> Position
    +-- Eval Compiled (Compiled, Position / 100.0) --> Distance
    +-- Branch: Distance < 0
          |
          True: Inside SDF (collision)
          False: Outside SDF
```

---

## Nanite Integration

ALICE-SDF can generate UE5 Nanite-compatible hierarchical cluster meshes directly from SDF mathematical descriptions. This enables a hybrid pipeline that combines the compactness of SDF with the rendering performance of Nanite.

### SDF × Nanite Hybrid Pipeline

| Process | Technology | Benefit |
|---------|-----------|---------|
| Storage & Transfer | ALICE-SDF | Data is just a few KB (mathematical formula). Downloads are instant. |
| Load Time | ALICE Engine | Generates Nanite mesh from the formula at high speed. |
| Rendering | UE5 Nanite | Renders hundreds of millions of polygons smoothly without GPU bottleneck. |

The key insight: **transfer at SDF compactness, render at polygon speed**. There is no need to run raymarching (expensive per-pixel computation) every frame — the GPU workload is the same as a conventional polygon game.

```
┌────────────────────┐     ┌─────────────────────────────┐     ┌──────────────┐
│  ALICE-SDF (.asdf) │     │  ALICE Engine (load time)    │     │  UE5 Nanite  │
│  Math formula      │ ──→ │  generate_nanite_mesh()      │ ──→ │  GPU render  │
│  ~2-4 KB           │     │  Hierarchical cluster + DAG  │     │  100M+ poly  │
└────────────────────┘     └─────────────────────────────┘     └──────────────┘
```

### Why This Is Different

Conventionally, using Nanite requires GB-scale pre-authored mesh data (e.g., from ZBrush scans) stored on disk. With ALICE-SDF:

- **Disk savings**: A massive rock formation is stored as "sphere + noise formula" — a few KB instead of hundreds of MB. No bloated asset folders.
- **Procedural generation × Nanite**: Generate infinite cave systems, terrain, or organic structures mathematically, then materialize them as high-detail Nanite meshes on the fly.
- **Runtime destruction**: Because ALICE-SDF mesh generation is fast, a wall destroyed by a spell can be **re-generated as a new Nanite mesh with the destroyed shape** in real time. This enables freeform destruction that is fundamentally different from the traditional "swap in a pre-made destroyed mesh" approach.

### Comparison with Conventional Approaches

| | Conventional Nanite | SDF × Nanite (ALICE) |
|---|---|---|
| Asset authoring | ZBrush/Photogrammetry scan | Mathematical formula (CSG, noise, procedural) |
| Asset size on disk | 100 MB - 1 GB per asset | 80 bytes - 4 KB per asset |
| Network transfer | Requires large download | Instant (formula only) |
| Runtime modification | Not possible (static mesh) | Re-generate from modified SDF |
| LOD | Baked into Nanite clusters | Generated per LOD level from SDF |
| Variety | Each variation = separate asset | Parameterized — infinite variations from one formula |

### Generate Nanite Mesh from SDF

```rust
use alice_sdf::prelude::*;
use alice_sdf::mesh::{generate_nanite_mesh, NaniteConfig, NaniteMesh};

// Define SDF shape
let shape = SdfNode::sphere(1.0)
    .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2)
    .twist(0.5);

// Generate Nanite hierarchical clusters
let config = NaniteConfig::high_detail();
let nanite: NaniteMesh = generate_nanite_mesh(
    &shape,
    Vec3::splat(-2.0),
    Vec3::splat(2.0),
    &config,
);

println!("Clusters: {}", nanite.clusters.len());
println!("LOD levels: {}", nanite.lod_levels.len());
```

### NaniteConfig Presets

| Preset | Base Resolution | LOD Levels | Use Case |
|--------|----------------|------------|----------|
| `high_detail()` | 256 | 6 | Final quality, cinematics |
| `medium_detail()` | 128 | 4 | Standard game assets |
| `preview()` | 64 | 3 | Editor preview, rapid iteration |

### Export for UE5

```rust
use alice_sdf::io::nanite::{export_nanite, export_nanite_json};

// Export binary .nanite file (cluster hierarchy)
export_nanite(&nanite, "Content/Nanite/MyAsset.nanite")?;

// Export JSON manifest (metadata, LOD statistics)
export_nanite_json(&nanite, "Content/Nanite/MyAsset.nanite.json")?;
```

### Export HLSL Material Function

```rust
use alice_sdf::io::nanite::export_nanite_hlsl_material;

// Generate UE5 material function for Nanite material assignment
let hlsl = export_nanite_hlsl_material(&nanite, "MyAssetMaterial");
std::fs::write("MyAssetMaterial.hlsl", hlsl)?;
```

### Runtime Destruction Example

```rust
// Original wall
let wall = SdfNode::box3d(5.0, 3.0, 0.3);
let wall_nanite = generate_nanite_mesh(
    &wall, Vec3::splat(-6.0), Vec3::splat(6.0),
    &NaniteConfig::medium_detail(),
);

// Player casts a spell — subtract an impact sphere
let impact = SdfNode::sphere(1.2).translate(hit_point);
let destroyed_wall = wall.subtract(impact);

// Re-generate Nanite mesh with the destroyed shape
let destroyed_nanite = generate_nanite_mesh(
    &destroyed_wall, Vec3::splat(-6.0), Vec3::splat(6.0),
    &NaniteConfig::medium_detail(),
);
// Feed destroyed_nanite to UE5 — freeform destruction without pre-baked meshes
```

### Cluster Structure

Each `NaniteCluster` contains:

| Field | Type | Description |
|-------|------|-------------|
| `vertices` | `Vec<[f32; 3]>` | Vertex positions |
| `triangles` | `Vec<[u32; 3]>` | Triangle indices (~128 triangles per cluster) |
| `bounds` | `ClusterBounds` | AABB bounding box |
| `parent` / `children` | `Option<usize>` / `Vec<usize>` | LOD DAG hierarchy |
| `geometric_error` | `f32` | Error metric for LOD selection |
| `material_id` | `u32` | Per-cluster material assignment |

### Text-to-3D: Natural Language to Nanite Mesh

ALICE-SDF includes a FastAPI server (Text-to-3D Pipeline) that eliminates the need for humans to write SDF formulas. Combined with the Nanite pipeline above, the full workflow becomes:

```
"古びた石の塔を作って"          LLM thinks            ALICE-SDF             UE5
  (natural language)   ──→   SDF formula    ──→   Nanite mesh    ──→   GPU render
  User types/speaks          Claude/Gemini         <55ms               100M+ poly
```

Users (artists, level designers) do not need to know SDF operations (Union, Intersection, SmoothMin...) at all. They describe what they want in plain language, and the pipeline produces a Nanite-ready mesh.

#### How It Works

1. **User input**: Natural language text (e.g., "A medieval castle with towers")
2. **LLM translation**: Claude or Gemini generates an SDF node tree (JSON)
3. **Automatic error repair**: If the LLM output has syntax errors (unclosed brackets, invalid parameters), the server automatically repairs the JSON and retries — feeding the error message back to the LLM for up to 2 correction attempts
4. **SDF compilation + mesh generation**: ALICE-SDF compiles the node tree and generates geometry in <55ms (excluding LLM inference time)
5. **Output**: `.glb`, `.obj`, `.nanite`, or raw SDF JSON

#### Error Self-Repair

LLM-generated code is not always perfect. The server's robustness features handle this automatically:

| Feature | Description |
|---------|-------------|
| JSON repair | Automatic brace/bracket completion for truncated LLM output |
| Structure validation | Pre-validates Boolean ops and transforms before Rust deserialization |
| Retry with feedback | Error messages are fed back to the LLM — "fix this" — up to 2 retries |
| Rate limit handling | Automatic wait-and-retry on 429 errors |

This ensures that only valid, working assets are produced — the most common failure mode ("AI generated it but it doesn't work") is eliminated by the feedback loop.

#### Runtime On-Demand Generation

The server runs as a FastAPI web service. Because the SDF-to-mesh step takes <55ms, game clients can send requests to the server during gameplay and receive new geometry on the fly:

```
Player: Casts "Ice Wall" spell (voice input or game event)
    │
    ▼
Server: Generates ice wall SDF from prompt
    │
    ▼
Client: Ice wall appears in-game with physics collision
```

This is **on-demand generation**, not pre-baked assets. Every invocation can produce unique geometry.

#### Complete Pipeline: No Blind Spots

| Concern | Solution | Technology |
|---------|----------|-----------|
| Rendering | Nanite integration — polygon-speed GPU rendering | `generate_nanite_mesh()` → UE5 Nanite |
| Physics | Deterministic fixed-point collision | ALICE-Physics (Fix128) |
| Asset creation | Natural language — no SDF expertise needed | Text-to-3D server (Claude/Gemini) |
| Delivery | Instant transfer — formulas, not polygons | ALICE-CDN + ALICE-Cache |

---

## Collision Setup

### Generate Collision Meshes

Use V-HACD convex decomposition for accurate physics:

```rust
use alice_sdf::prelude::*;

let shape = SdfNode::sphere(1.0)
    .subtract(SdfNode::box3d(0.5, 0.5, 0.5));

let mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0),
    &MarchingCubesConfig::aaa(64));

// Generate convex decomposition for UE5 physics
let decomp = convex_decomposition(&mesh, &VhacdConfig {
    max_hulls: 16,
    resolution: 100000,
    max_vertices_per_hull: 32,
    volume_error_percent: 1.0,
});

// Export each hull as separate OBJ for UE5 import
for (i, hull) in decomp.hulls.iter().enumerate() {
    // hull.vertices and hull.indices contain the convex hull
    println!("Hull {}: {} vertices", i, hull.vertices.len());
}
```

### Import Collision in UE5

1. Export collision meshes as separate OBJ files
2. Import into UE5 as Static Meshes
3. In Static Mesh Editor: `Collision > Add Custom Collision`
4. Or use `UCX_` naming convention for auto-collision

---

## Performance Tuning

### Evaluation Performance

| Method | Throughput | Use Case |
|--------|-----------|----------|
| `eval_soa` | 1B+ ops/sec | Particle systems, physics |
| `eval_compiled_batch` | 500M ops/sec | General batch |
| `eval_compiled` | 100M ops/sec | Per-frame queries |
| `eval` | 50M ops/sec | Debugging only |

### Best Practices

1. **Always compile** - Call `alice_sdf_compile()` once at initialization
2. **Use SoA layout** - `eval_soa` is 2-5x faster than AoS `eval_compiled_batch`
3. **Convert units** - UE5 uses centimeters, ALICE-SDF uses meters. Divide by 100.
4. **Batch large workloads** - Minimum 256 points for parallelization benefit
5. **Cache compiled handles** - Store in member variable, reuse across frames

### Memory Management

```cpp
// IMPORTANT: Free all handles when done
void AMyActor::EndPlay(const EEndPlayReason::Type Reason)
{
    if (CompiledSdf) alice_sdf_free_compiled(CompiledSdf);
    if (SdfShape) alice_sdf_free(SdfShape);
    Super::EndPlay(Reason);
}
```

---

## Troubleshooting

### DLL Not Found

```
LogModuleManager: Error: Unable to load module 'AliceSDF'
```

**Fix**: Ensure `alice_sdf.dll` is in `Plugins/AliceSDF/Binaries/Win64/`.

### Linker Errors

```
error LNK2019: unresolved external symbol _alice_sdf_sphere
```

**Fix**: Verify `AliceSDF.Build.cs` includes the library path:

```csharp
PublicAdditionalLibraries.Add(
    Path.Combine(PluginDir, "Binaries", Target.Platform.ToString(), "alice_sdf.lib")
);
```

### HLSL Compilation Errors

**Fix**: Ensure the generated HLSL uses `float3` (not `vec3`). The HLSL transpiler handles this automatically, but manual edits may introduce GLSL syntax.

### Crash on Shader Hot-Reload

**Fix**: HLSL code generated by ALICE-SDF is pure functions without global state. It should survive hot-reload. If crashes occur, regenerate the HLSL and re-paste.

### Performance Issues

1. **Profile first**: Use UE5 Insights or Unreal Frontend to identify bottleneck
2. **Check compilation**: Ensure `alice_sdf_compile()` is called only once
3. **Reduce point count**: For particle systems, evaluate every 2nd or 4th frame
4. **Use LOD**: Generate lower-resolution meshes for distant objects

---

## Example: Procedural Terrain

```cpp
// ProceduralTerrain.h
UCLASS()
class AProceduralTerrain : public AActor
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere)
    float TerrainScale = 100.0f;

    UPROPERTY(EditAnywhere)
    int32 Resolution = 64;

private:
    SdfHandle TerrainSdf = nullptr;
    CompiledHandle CompiledTerrain = nullptr;

    void GenerateTerrain();
    void CleanupSdf();
};

// ProceduralTerrain.cpp
void AProceduralTerrain::GenerateTerrain()
{
    CleanupSdf();

    // Create terrain SDF: flat plane with noise
    SdfHandle Plane = alice_sdf_plane(0.0f, 1.0f, 0.0f, 0.0f);
    TerrainSdf = alice_sdf_twist(Plane, 0.1f); // Add some variation

    CompiledTerrain = alice_sdf_compile(TerrainSdf);

    // Generate mesh via CLI or Rust API, then import as Static Mesh
}

void AProceduralTerrain::CleanupSdf()
{
    if (CompiledTerrain) { alice_sdf_free_compiled(CompiledTerrain); CompiledTerrain = nullptr; }
    if (TerrainSdf) { alice_sdf_free(TerrainSdf); TerrainSdf = nullptr; }
}
```

---

## Related Documentation

- [API Reference](API_REFERENCE.md) - Complete FFI function list
- [Architecture](ARCHITECTURE.md) - 13-layer design
- [QUICKSTART](QUICKSTART.md) - Getting started with other platforms

---

Author: Moroya Sakamoto
