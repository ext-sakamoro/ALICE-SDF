# ALICE-SDF — Unreal Engine 5 Plugin

72 primitives, 24 CSG operations, 7 transforms, 23 modifiers (126 total), HLSL generation, mesh export.
Drop into your UE5 project and start using SDFs immediately.

## Quick Start (No Rust Required)

### 1. Download

Download `AliceSDF-UE5-Plugin-{platform}.zip` from [GitHub Releases](https://github.com/ext-sakamoro/ALICE-SDF/releases).

### 2. Extract to Plugins

```
YourProject/
  Plugins/
    AliceSDF/          <-- Extract here
      AliceSDF.uplugin
      Source/
      ThirdParty/      <-- Pre-built native library included
```

### 3. Open UE5

Open your `.uproject`. The plugin loads automatically.
If prompted, click "Yes" to rebuild.

### 4. Use It

Add `AliceSdfComponent` to any Actor via Blueprint or C++.

---

## Blueprint Usage

### Create a Shape

1. Add `AliceSdfComponent` to your Actor
2. In the Event Graph, call **Create Sphere**, **Create Box**, **Create Torus**, etc.
3. The shape is auto-compiled for fast evaluation

### CSG Operations

```
[CreateSphere (r=1.0)] --> [SmoothSubtractFrom (Other, k=0.1)]
                                     ^
[CreateBox (0.7, 0.7, 0.7)] --------+
```

Available operations:
- **Standard**: Union, Intersection, Subtract
- **Smooth**: SmoothUnion, SmoothIntersect, SmoothSubtract
- **Chamfer**: ChamferUnion, ChamferIntersect, ChamferSubtract
- **Stairs**: StairsUnion, StairsIntersect, StairsSubtract (terraced)
- **Columns**: ColumnsUnion, ColumnsIntersect, ColumnsSubtract
- **Advanced**: XOR, Morph, Pipe, Engrave, Groove, Tongue

### Evaluate Distance

```
EvalDistance(WorldPosition) → float
  negative = inside
  zero     = surface
  positive = outside
```

### Generate HLSL for Materials

```
GenerateHlsl() → FString
```

Paste into a **Custom** node in your Material graph.
Connect `WorldPosition / 100.0` as input (UE5 uses cm, SDF uses m).

---

## C++ Usage

```cpp
#include "AliceSdfComponent.h"

void AMyActor::BeginPlay()
{
    Super::BeginPlay();

    auto* Sdf = FindComponentByClass<UAliceSdfComponent>();

    // Create shape
    Sdf->CreateSphere(1.0f);

    // Apply modifiers
    Sdf->ApplyTwist(2.0f);
    Sdf->ApplyNoise(0.1f, 5.0f, 42);

    // Evaluate
    float Distance = Sdf->EvalDistance(GetActorLocation());
    bool bInside = Sdf->IsPointInside(GetActorLocation());

    // Generate HLSL for Custom Material Expression
    FString Hlsl = Sdf->GenerateHlsl();

    // Export mesh
    Sdf->ExportObj(TEXT("C:/Output/shape.obj"), 128, 2.0f);
    Sdf->ExportGlb(TEXT("C:/Output/shape.glb"), 128, 2.0f);
    Sdf->ExportFbx(TEXT("C:/Output/shape.fbx"), 128, 2.0f);
    Sdf->ExportUsda(TEXT("C:/Output/shape.usda"), 128, 2.0f);

    // Save/Load SDF tree
    Sdf->SaveToFile(TEXT("C:/Output/shape.asdf"));
    Sdf->LoadFromFile(TEXT("C:/Output/shape.asdf"));
}
```

---

## Demo Actor

Drop `AAliceSdfDemoActor` into any level and set `DemoIndex` (0-4):

| Index | Demo | Description |
|-------|------|-------------|
| 0 | Basic Shapes | Sphere, Box, Torus, Heart, BoxFrame |
| 1 | CSG Operations | SmoothSubtract, ChamferUnion, XOR, Morph |
| 2 | TPMS Surfaces | Gyroid, SchwarzP, Diamond, Neovius, Lidinoid |
| 3 | Modifiers | Twist, Onion, Repeat, PolarRepeat, OctantMirror, Noise |
| 4 | Shader Generation | HLSL output + mesh export |

Check **Output Log** for results.

---

## GPU Showcase Actors

Five showcase actors recreate the full Unity SDF Universe demo using UE5's native rendering systems.

### Rendering Pipelines

| Pipeline | Actor Class | Description |
|----------|------------|-------------|
| **Nanite Mesh** | `AAliceSdfNaniteActor` | SDF → Marching Cubes → Nanite StaticMesh. 500K-8M tris at 60fps |
| **GPU Particles** | `AAliceSdfParticleActor` | Compute Shader SDF physics + instanced billboard render. 10M+ particles |
| **Raymarching** | `AAliceSdfRaymarchActor` | Per-pixel raymarching on bounding box. Infinite detail, no mesh |
| **Lumen GI** | `AAliceSdfLumenShowcase` | Gallery room with Nanite shapes + Lumen color bleeding |

### Showcase Scenes (5)

| Actor | Scene | Pipeline | Elements |
|-------|-------|----------|----------|
| `AAliceSdfCosmicShowcase` | Solar System | Nanite | Sun, 2 planets, ring, moon, asteroid belt |
| `AAliceSdfTerrainShowcase` | Landscape | Nanite | FBM terrain, water, 5 rocks, 3 floating islands |
| `AAliceSdfAbstractShowcase` | Generative Art | Nanite | Gyroid center, 6 metaballs, 3 rings, 4 Schwarz P, 8 cubes |
| `AAliceSdfFractalShowcase` | Menger Sponge | Nanite | High-res fractal with turntable rotation |
| `AAliceSdfLumenShowcase` | GI Gallery | Nanite + Lumen | 4 shapes on pedestals, colored lights, PostProcessVolume |

---

### Nanite Actor — High-Polygon SDF Meshes

Place `AAliceSdfNaniteActor` in a level. Select `ShapeType` from 23 presets:

| Category | Shapes |
|----------|--------|
| **Showcase** | TPMS Sphere, Organic Sculpture, Crystal, Arch Column, SAO Float |
| **Fractal** | Menger Sponge, Fractal Planet Mix |
| **Cosmic** | Cosmic System, Sun, Ringed Planet, Asteroid |
| **Terrain** | Ground, Water, Rock, Floating Island |
| **Abstract** | Bounded Gyroid, Metaball, Schwarz P, Floating Cube, Ring |
| **Lumen** | Cathedral, Coral Reef |

```
Resolution=256 → ~500K-2M tris (balanced)
Resolution=512 → ~2M-8M tris (ultra detail, Nanite handles it)
```

Click **Rebuild Mesh** in Details panel. Nanite automatically handles LOD.

---

### GPU Particle Actor — 10M Particles at 60fps

Place `AAliceSdfParticleActor` in a level. Select `SceneType`:

| Scene | Description |
|-------|-------------|
| Cosmic | Particles flowing across Sun + Planet + Ring |
| Terrain | Particles on FBM landscape + water |
| Abstract | Particles on Gyroid + Metaballs |
| Fractal | Particles on Menger Sponge with microscope zoom |

Parameters: ParticleCount (10K-10M), FlowSpeed, SurfaceAttraction, NoiseStrength, ParticleSize, Brightness, CoreGlow.

Time Slicing: Set `UpdateDivisions=3` to process 1/3 of particles per frame for higher counts.

---

### Raymarch Actor — Infinite Detail Without Meshes

Place `AAliceSdfRaymarchActor` in a level. Select `ShaderMode`:

| Mode | Description | Parameters |
|------|-------------|------------|
| **Fractal** | Menger Sponge with twist and zoom | BoxSize, HoleSize, RepeatScale, TwistAmount, DetailScale |
| **CosmicFractal** | 4 demo modes in one | CosmicMode (Normal/Fusion/Destruction/Morph) |

CosmicFractal Modes:
- **Normal**: Sun + orbiting fractal planet + ring + moon
- **Fusion**: Two spheres with dynamic smooth union (metaballs)
- **Destruction**: Box with fractal + runtime hole subtraction
- **Morph**: Sphere → Box → Torus → Menger interpolation (auto-cycling)

Quality: MaxSteps (32-512), SurfaceEpsilon (0.0001-0.01), FogDensity.

Note: Raymarched surfaces do not receive Lumen GI (no GBuffer output). Lighting is computed analytically in the shader (diffuse + fresnel + fog).

---

### Lumen Showcase — Global Illumination Demo

Place `AAliceSdfLumenShowcase` in a level. Click **Build Gallery**:

- Builds a gallery room (walls, floor, ceiling)
- Spawns 4 Nanite SDF shapes on pedestals (Cathedral, Coral, TPMS, Crystal)
- 4 colored point lights for visible color bleeding
- Auto-spawns PostProcessVolume (GI=Lumen, Reflections=Lumen)
- Auto-spawns SkyLight for environment indirect lighting

Parameters: LumenFinalGatherQuality (1-4), LumenSceneDetail (1-4), SkyLightIntensity.

---

## Demo Level Setup Guide

### Minimal Setup (Test one actor)

1. Create a new Empty Level
2. Add a DirectionalLight and SkyLight
3. Drag one showcase actor into the level
4. Press Play

### Full Showcase Level (All pipelines)

Recommended layout for a comprehensive demo:

```
+------------------------------------------+
|                                          |
|   [Cosmic Showcase]     [Terrain]        |  ← Nanite scenes (back)
|     (0, -2000, 0)       (3000, -2000, 0) |
|                                          |
|                                          |
|   [Abstract]            [Fractal]        |  ← Nanite scenes (mid)
|     (0, 0, 0)            (3000, 0, 0)    |
|                                          |
|                                          |
|   [Particle Actor]    [Raymarch Actor]   |  ← GPU scenes (front)
|     (0, 2000, 0)       (3000, 2000, 0)  |
|                                          |
|   [Lumen Showcase]                       |  ← Indoor GI scene
|     (6000, 0, 0)                         |
|                                          |
+------------------------------------------+
```

Steps:
1. **File → New Level → Empty Level**
2. **Add Lighting**: DirectionalLight (Movable), SkyLight, SkyAtmosphere, ExponentialHeightFog
3. **Place Showcases**: Drag each actor class from Place Actors panel (search "AliceSdf")
4. **Build Nanite**: For each Nanite-based showcase, click "Build" in Details
5. **Play**: GPU Particles + Raymarching activate on Play

### Project Settings for Lumen

Ensure these are set in **Project Settings**:
- **Global Illumination → Dynamic GI Method**: Lumen
- **Reflections → Reflection Method**: Lumen
- **Nanite → Enable Nanite**: True (default in UE5)
- **Shader Model**: SM5 or SM6

### Performance Tips

| Scenario | Recommendation |
|----------|---------------|
| Low-end GPU | Particle count ≤100K, Raymarch MaxSteps=64, Nanite Resolution=128 |
| Mid-range GPU | Particle count 500K, MaxSteps=128, Resolution=256 |
| High-end GPU | Particle count 5M+, MaxSteps=256, Resolution=512 |
| Lumen on | Reduce particle count, Lumen adds ~2ms GPU cost |

---

## Available Primitives (66)

### Basic (14)
Sphere, Box, Cylinder, Torus, Capsule, Plane, Cone, Ellipsoid, RoundedCone, Pyramid, Octahedron, HexPrism, Link, RoundedBox

### Advanced (12)
CappedCone, CappedTorus, RoundedCylinder, TriangularPrism, CutSphere, CutHollowSphere, DeathStar, SolidAngle, Heart, Barrel, Diamond, Egg

### 2D/Extruded (23)
Triangle, Bezier, Rhombus, Horseshoe, Vesica, SuperEllipsoid, RoundedX, Pie, Trapezoid, Parallelogram, Tunnel, UnevenCapsule, ArcShape, Moon, CrossShape, BlobbyCross, ParabolaSegment, RegularPolygon, StarPolygon, InfiniteCylinder, InfiniteCone, RegularPolygon, StarPolygon

### Platonic & Archimedean (5)
Tetrahedron, Dodecahedron, Icosahedron, TruncatedOctahedron, TruncatedIcosahedron

### TPMS (9 surfaces)
Gyroid, SchwarzP, DiamondSurface, Neovius, Lidinoid, IWP, FRD, FischerKochS, PMY

### Structural (5)
BoxFrame, Tube, ChamferedCube, Stairs, Helix

---

## Available Transforms (7)

Translate, Rotate, RotateEuler, Scale, ScaleNonUniform, ProjectiveTransform (v1.1.0), LatticeDeform (v1.1.0), Skinning (v1.1.0)

## Available Modifiers (23)

Round, Onion, Twist, Bend, Repeat, RepeatFinite, Mirror, Elongate, Revolution, Extrude, Noise, Taper, Displacement, PolarRepeat, OctantMirror, SweepBezier, Shear, MaterialId, Animated, IcosahedralSymmetry (v1.1.0), IFS (v1.1.0), HeightmapDisplacement (v1.1.0), SurfaceRoughness (v1.1.0)

---

## v1.1.0 New Features

### Advanced Transforms

```cpp
// Projective transform (perspective projection with 4x4 inverse matrix)
float inv_matrix[16] = { /* column-major */ };
SdfHandle projected = alice_sdf_projective_transform(shape, inv_matrix, 1.5f);

// Lattice Free-Form Deformation
uint32_t nx = 4, ny = 4, nz = 4;
float control_points[64 * 3]; // nx*ny*nz * 3
float bbox_min[3] = {-2.f, -2.f, -2.f};
float bbox_max[3] = { 2.f,  2.f,  2.f};
SdfHandle deformed = alice_sdf_lattice_deform(
    shape, control_points, 64, nx, ny, nz, bbox_min, bbox_max);

// Skeletal skinning (33 floats per bone: 16 inv_bind + 16 cur_pose + 1 weight)
float bones[2 * 33];
SdfHandle skinned = alice_sdf_skinning(shape, bones, 2);
```

### Advanced Modifiers

```cpp
// 120-fold icosahedral symmetry
SdfHandle gem = alice_sdf_icosahedral_symmetry(shape);

// IFS fractal (2 transform matrices, 5 iterations)
float transforms[2 * 16]; // column-major 4x4 matrices
SdfHandle fractal = alice_sdf_ifs(shape, transforms, 2, 5);

// Heightmap displacement
float heightmap[256 * 256];
SdfHandle displaced = alice_sdf_heightmap_displacement(
    shape, heightmap, 256, 256, 0.1f, 1.0f);

// FBM surface roughness
SdfHandle rough = alice_sdf_surface_roughness(shape, 4.0f, 0.05f, 4);
```

---

## Mesh Export Formats

| Format | Function | Use Case |
|--------|----------|----------|
| OBJ | `ExportObj()` | DCC tools (Blender, Maya) |
| GLB | `ExportGlb()` | Web, glTF viewers |
| USDA | `ExportUsda()` | USD pipelines, Omniverse |
| FBX | `ExportFbx()` | UE5 Static Mesh import |

---

## Supported Platforms

| Platform | Library | Status |
|----------|---------|--------|
| Windows x64 | `alice_sdf.dll` + `.lib` | Supported |
| macOS (arm64/x64) | `libalice_sdf.dylib` | Supported |
| Linux x64 | `libalice_sdf.so` | Supported |

---

## Building from Source (Optional)

Only needed if you want to modify the native library.

```bash
# Install Rust (https://rustup.rs)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build + package
cd ALICE-SDF
./scripts/build_ue5_plugin.sh --zip
```

---

## Performance

| Method | Throughput | Use Case |
|--------|-----------|----------|
| Compiled batch (SoA) | 1B+ ops/sec | Physics, particles |
| Compiled batch (AoS) | 500M ops/sec | General batch |
| Single eval (compiled) | 100M ops/sec | Per-frame queries |
| Single eval (tree) | 50M ops/sec | Debug only |

The component auto-compiles by default. Disable with `bAutoCompile = false`.

---

## Full API Reference

See [docs/UNREAL_ENGINE.md](../docs/UNREAL_ENGINE.md) for the complete integration guide.

---

Author: Moroya Sakamoto
