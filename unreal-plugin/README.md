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
