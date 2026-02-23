# ALICE-SDF

<p align="center">
  <img src="asset/logo-on-light.jpeg" alt="ALICE-SDF Logo" width="720">
</p>

**English** | [日本語](README_JP.md)

**A.L.I.C.E. - Adaptive Lightweight Implicit Compression Engine**

> "Don't send polygons. Send the law of shapes."

## Overview

ALICE-SDF is a 3D/spatial data specialist that transmits **mathematical descriptions of shapes** (Signed Distance Functions) instead of polygon meshes. This enables:

- **10-1000x compression** compared to traditional mesh formats
- **Infinite resolution** - shapes are mathematically perfect at any scale
- **CSG operations** - boolean operations on shapes without mesh overhead
- **Real-time raymarching** - GPU-accelerated rendering
- **PBR materials** - metallic-roughness workflow compatible with UE5/Unity/Godot
- **Keyframe animation** - parametric deformation with timeline tracks
- **Asset pipeline** - OBJ import/export, glTF 2.0 (.glb) export, FBX, USD, Alembic, Nanite, STL, PLY, 3MF, ABM export
- **5-layer mesh persistence** - ABM binary format, LOD chain persistence, chunked mesh cache with FIFO eviction, Unity/UE5 native export
- **Manifold mesh guarantee** - validation, repair, and quality metrics
- **Adaptive Marching Cubes** - octree-based mesh generation, detail where it matters
- **Dual Contouring** - QEF-based mesh generation that preserves sharp edges and corners
- **V-HACD convex decomposition** - automatic convex hull decomposition for physics
- **Attribute-preserving decimation** - QEM with UV/tangent/material boundary protection
- **Decimation-based LOD** - progressive LOD chain from high-res base mesh
- **72 primitives, 24 operations, 7 transforms, 23 modifiers** (126 total) - industry-leading shape vocabulary
- **Chamfer & Stairs blends** - hard-edge bevels and stepped/terraced CSG transitions
- **Interval Arithmetic** - conservative AABB evaluation for spatial pruning and Lipschitz bound tracking
- **Relaxed Sphere Tracing** - over-relaxation with Lipschitz-adaptive step sizing
- **Neural SDF** - pure-Rust MLP that approximates an SDF tree ~10-100x faster for complex scenes
- **SDF-to-SDF Collision** - grid-based contact detection with interval arithmetic AABB pruning
- **CSG Tree Optimization** - identity transform/modifier removal, nested transform merging, smooth→standard demotion
- **Analytic Gradient** - single-pass gradient via chain rules and Jacobian propagation (9 analytic + 44 numerical-fallback primitives)
- **Automatic Differentiation** - Dual Number forward-mode AD, Hessian estimation, mean curvature computation
- **2D SDF module** - pure 2D SDF primitives (circle, rect, bezier, font glyph) with bilinear sampling
- **CSG Tree Diff/Patch** - structural diff between SDF trees for undo/redo and network sync
- **Parametric Constraint Solver** - Gauss-Newton optimization for geometric constraints (fixed, distance, sum, ratio)
- **Distance Field Heatmap** - cross-section slicing with 4 color maps (coolwarm, binary, viridis, magma)
- **Shell / Offset Surface** - variable-thickness shell modifier with inner/outer offset control
- **Volume & Surface Area** - Monte Carlo estimation with deterministic PRNG and standard error
- **ALICE-Font Bridge** - font glyph → 2D/3D SDF conversion, text layout, 3D extrusion (`--features font`)
- **Auto Tight AABB** - interval arithmetic + binary search to find minimal bounding box containing the SDF surface
- **7 evaluation modes** - interpreted, compiled VM, SIMD 8-wide, BVH, SoA batch, JIT, GPU
- **3 shader targets** - GLSL, WGSL, HLSL transpilation
- **Engine integrations** - Unity, Unreal Engine 5, VRChat, Godot, WebAssembly

## Text-to-3D Pipeline (Server)

ALICE-SDF includes a FastAPI server that converts **natural language text into real 3D geometry** via LLM-generated SDF trees.

```
User: "A medieval castle"  →  LLM (Claude/Gemini)  →  SDF JSON  →  ALICE-SDF  →  GLB/OBJ
         text                   ~5-50s                  20 nodes      <55ms        mesh
```

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────┐
│  FastAPI     │     │  LLM Service │     │  SDF Service  │     │  Output  │
│  Server      │────▶│  Claude API  │────▶│  from_json()  │────▶│  GLB     │
│              │     │  Gemini API  │     │  compile()    │     │  OBJ     │
│  POST /gen   │     │  System      │     │  to_mesh()    │     │  JSON    │
│  WS /ws/gen  │     │  Prompt      │     │  export_glb() │     │  Viewer  │
│  GET /viewer │     └──────────────┘     └───────────────┘     └──────────┘
└─────────────┘
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Text → 3D mesh (GLB/OBJ/JSON) |
| `POST` | `/api/validate` | Validate SDF JSON structure |
| `POST` | `/api/mesh` | SDF JSON → mesh (GLB/OBJ) |
| `GET` | `/api/examples` | List few-shot example scenes |
| `GET` | `/api/viewer` | Three.js GLB viewer (browser) |
| `GET` | `/api/health` | Server health check |
| `WS` | `/ws/generate` | Streaming generation with progressive preview |

### Generated Scene Gallery

Scenes generated by Gemini 2.5 Flash from natural language prompts:

| Prompt | Nodes | Vertices | Triangles | LLM Time |
|--------|-------|----------|-----------|----------|
| "A medieval castle with towers" | 18 | 2,105 | 4,248 | 49.4s |
| "A robot standing on a platform" | 18 | 750 | 1,184 | 17.5s |
| "An underwater coral reef scene" | 15 | 2,666 | 5,166 | 63.3s |
| "A simple mushroom on grass" | 9 | 8,237 | 16,224 | 6.6s |
| "火山地帯に宇宙船" (Volcanic terrain with spaceship) | 22 | 10,466 | 20,618 | 20.5s |

Hand-crafted few-shot examples (used in LLM system prompt):

| Scene | Description | Nodes | Vertices | Triangles |
|-------|-------------|-------|----------|-----------|
| `sphere_on_ground` | Sphere on a flat ground (Union + Plane) | 4 | 1,270 | 2,448 |
| `snowman` | 3-sphere snowman (SmoothUnion) | 8 | 422 | 840 |
| `castle_tower` | Tower with battlements (PolarRepeat) | 11 | 1,030 | 2,244 |
| `alien_mushroom_forest` | Mushroom grid (RepeatFinite + Torus stems) | 9 | 4,167 | 7,854 |
| `twisted_pillar` | Twisted box + floating hollow sphere (Twist + Onion) | 7 | 510 | 968 |
| `mechanical_gear` | Gear with teeth and axle hole (PolarRepeat + Subtraction) | 9 | 465 | 912 |

Scene JSON files are stored in [`server/examples/scenes/`](server/examples/scenes/).

### Quick Start (Server)

```bash
# 1. Build Python bindings
cd /path/to/ALICE-SDF
python -m venv .venv && source .venv/bin/activate
maturin develop --features python

# 2. Install server dependencies
pip install -r server/requirements.txt

# 3. Set API keys
export ANTHROPIC_API_KEY="sk-..."   # for Claude
export GOOGLE_API_KEY="AI..."       # for Gemini

# 4. Start server
uvicorn server.main:app --reload

# 5. Generate 3D from text
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A snowman", "provider": "gemini", "resolution": 64}' \
  -o snowman.glb

# 6. Open browser viewer
open http://localhost:8000/api/viewer
```

### LLM Providers

| Provider | Model | Speed | Best For |
|----------|-------|-------|----------|
| Claude | Haiku 4.5 | ~2-5s | Simple scenes, fast iteration |
| Claude | Sonnet 4.5 | ~5-15s | Complex scenes, high accuracy |
| Gemini | 2.5 Flash | ~5-50s | Complex scenes (thinking model) |
| Gemini | 2.5 Pro | ~10-60s | Maximum quality |

### Performance Budget

| Step | Time | Notes |
|------|------|-------|
| LLM inference | 2-60s | Depends on model and complexity |
| JSON parse | <1ms | serde_json |
| SDF compile | ~1ms | SdfNode → CompiledSdf |
| Mesh generation (res=64) | ~45ms | Parallel Marching Cubes |
| GLB export | ~5ms | |
| **Total (excluding LLM)** | **<55ms** | Real-time capable |

### Robustness Features

- **JSON repair**: Automatic brace/bracket completion for truncated LLM output
- **Structure validation**: Pre-validates Boolean ops (a/b) and transforms (child) before Rust serde
- **Retry with feedback**: Up to 2 retries with error message fed back to LLM
- **Rate limit handling**: Automatic wait-and-retry on 429 errors
- **Complexity constraints**: System prompt limits scenes to 15-20 nodes, nesting depth ≤6

### Server Directory Structure

```
server/
├── main.py                  # FastAPI app, REST + WebSocket endpoints
├── config.py                # API keys, model config (env vars)
├── models.py                # Pydantic request/response models
├── services/
│   ├── llm_service.py       # Claude/Gemini API with retry logic
│   └── sdf_service.py       # alice_sdf wrapper (parse, mesh, export)
├── prompts/
│   ├── system_prompt.py     # 36-node-type SDF grammar for LLM
│   └── examples.py          # 6 few-shot examples
├── examples/
│   └── scenes/              # Pre-built scene JSON files
│       ├── sphere_on_ground.json
│       ├── snowman.json
│       ├── castle_tower.json
│       ├── alien_mushroom_forest.json
│       ├── twisted_pillar.json
│       └── mechanical_gear.json
├── static/
│   └── viewer.html          # Three.js GLB viewer
├── tests/
│   ├── test_api.py          # 7 API endpoint tests
│   ├── test_llm_service.py  # 17 JSON extraction/validation tests
│   └── test_sdf_service.py  # 13 SDF pipeline tests
└── requirements.txt
```

### Running Tests

```bash
source .venv/bin/activate
python -m pytest server/tests/ -v   # 37 tests, all passing
```

## ALICE-View (Real-time 3D Viewer)

**[ALICE-View](../ALICE-View)** is a native GPU raymarching viewer built with wgpu. It renders SDF trees directly on the GPU via WGSL transpilation — no mesh conversion needed.

```
SDF JSON → ALICE-SDF (WGSL transpile) → wgpu GPU Raymarching → Real-time 3D
              ~1ms                            60 FPS
```

### Features

- **GPU Raymarching** — SdfNode tree transpiled to WGSL shader, evaluated per-pixel on GPU
- **Drag & Drop** — Drop `.json` / `.asdf` / `.asdf.json` files onto the window
- **File Dialog** — File > Open (Ctrl+O) with format filters
- **Camera Controls** — Mouse orbit, scroll zoom, WASD movement
- **Live SDF Panel** — Node count, raymarching parameters (max steps, epsilon, AO)

### Supported Formats

| Extension | Format | Description |
|-----------|--------|-------------|
| `.json` | SDF JSON | Text-to-3D pipeline output, few-shot examples |
| `.asdf.json` | ALICE SDF JSON | Native ALICE-SDF JSON format |
| `.asdf` | ALICE SDF Binary | Compact binary with CRC32 |
| `.alice` / `.alz` | ALICE Legacy | Procedural content (Perlin, Fractal) |

### Quick Start

```bash
cd /path/to/ALICE-View

# Open a specific file
cargo run --bin alice-view -- path/to/scene.json

# Or launch empty and drag & drop files onto the window
cargo run --bin alice-view
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `W/A/S/D` | Camera movement |
| `Mouse drag` | Orbit camera |
| `Scroll` | Zoom in/out |
| `Ctrl+O` | Open file dialog |
| `Q` | Quit |

### Viewing Text-to-3D Results

Generated scene JSON files from the Text-to-3D pipeline can be viewed directly:

```bash
# View a generated scene
cargo run --bin alice-view -- /path/to/ALICE-SDF/server/examples/scenes/snowman.json

# Or drag any of these onto the window:
#   server/examples/scenes/castle_tower.json
#   server/examples/scenes/mechanical_gear.json
#   server/examples/scenes/alien_mushroom_forest.json
```

---

## Core Concepts

### SDF (Signed Distance Function)

An SDF returns the shortest distance from any point to the surface:
- **Negative** = inside the shape
- **Zero** = on the surface
- **Positive** = outside the shape

### SdfNode Tree Structure

```
SdfNode
  |-- Primitive (68): Sphere, Box3D, Cylinder, Torus, Plane, Capsule, Cone, Ellipsoid,
  |                    RoundedCone, Pyramid, Octahedron, HexPrism, Link, Triangle, Bezier,
  |                    RoundedBox, CappedCone, CappedTorus, InfiniteCylinder, RoundedCylinder,
  |                    TriangularPrism, CutSphere, CutHollowSphere, DeathStar, SolidAngle,
  |                    Rhombus, Horseshoe, Vesica, InfiniteCone, Heart, Gyroid,
  |                    Tube, Barrel, Diamond, ChamferedCube, SchwarzP, Superellipsoid, RoundedX,
  |                    Pie, Trapezoid, Parallelogram, Tunnel, UnevenCapsule, Egg,
  |                    ArcShape, Moon, CrossShape, BlobbyCross, ParabolaSegment,
  |                    RegularPolygon, StarPolygon, Stairs, Helix,
  |                    Tetrahedron, Dodecahedron, Icosahedron,                    ← Platonic solids (GDF)
  |                    TruncatedOctahedron, TruncatedIcosahedron,                 ← Archimedean solids
  |                    BoxFrame,                                                   ← IQ wireframe box
  |                    DiamondSurface, Neovius, Lidinoid, IWP, FRD,              ← TPMS surfaces
  |                    FischerKochS, PMY,                                          ← TPMS surfaces
  |                    Circle2D, Rect2D, Segment2D, Polygon2D,                   ← 2D primitives (extruded)
  |                    RoundedRect2D, Annular2D                                    ← 2D primitives (extruded)
  |-- Operation (24): Union, Intersection, Subtraction,
  |                    SmoothUnion, SmoothIntersection, SmoothSubtraction,
  |                    ChamferUnion, ChamferIntersection, ChamferSubtraction,
  |                    StairsUnion, StairsIntersection, StairsSubtraction,
  |                    ExpSmoothUnion, ExpSmoothIntersection, ExpSmoothSubtraction, ← IQ exponential smooth
  |                    XOR, Morph,                                                 ← Boolean / Interpolation
  |                    ColumnsUnion, ColumnsIntersection, ColumnsSubtraction,      ← hg_sdf columns
  |                    Pipe, Engrave, Groove, Tongue                               ← hg_sdf advanced
  |-- Transform (7): Translate, Rotate, Scale, ScaleNonUniform,
  |                   ProjectiveTransform,                                         ← perspective projection with inv_matrix
  |                   LatticeDeform,                                               ← Free-Form Deformation (FFD) grid
  |                   SdfSkinning                                                  ← bone-weight skeletal deformation
  |-- Modifier (23): Twist, Bend, RepeatInfinite, RepeatFinite, Noise, Round, Onion, Elongate,
  |                   Mirror, Revolution, Extrude, Taper, Displacement, PolarRepeat, SweepBezier,
  |                   Shear,                                                       ← 3-axis shear deformation
  |                   OctantMirror,                                                ← 48-fold symmetry
  |                   IcosahedralSymmetry,                                         ← 120-fold icosahedral symmetry
  |                   IFS,                                                         ← Iterated Function System fractals
  |                   HeightmapDisplacement,                                       ← heightmap-driven surface displacement
  |                   SurfaceRoughness,                                            ← FBM noise roughness
  |                   Animated,                                                    ← timeline-driven parameter animation
  |                   WithMaterial                                                 ← PBR material assignment
```

## Installation

### Rust

```bash
cargo add alice-sdf
```

### Python

```bash
pip install alice-sdf
```

## Usage

### Rust

```rust
use alice_sdf::prelude::*;

// Create a sphere with radius 1
let sphere = SdfNode::sphere(1.0);

// Subtract a box from it
let result = sphere.subtract(SdfNode::box3d(1.5, 1.5, 1.5));

// Evaluate distance at a point
let distance = eval(&result, glam::Vec3::ZERO);

// Convert to mesh
let mesh = sdf_to_mesh(
    &result,
    glam::Vec3::splat(-2.0),
    glam::Vec3::splat(2.0),
    &MarchingCubesConfig::default()
);
```

### Python

```python
import alice_sdf as sdf

# Create primitives
sphere = sdf.SdfNode.sphere(1.0)
box3d = sdf.SdfNode.box3d(2.0, 1.0, 1.0)

# CSG operations (method syntax)
result = sphere.subtract(box3d)

# Operator overloads (Pythonic syntax)
a = sdf.SdfNode.sphere(1.0)
b = sdf.SdfNode.box3d(0.5, 0.5, 0.5)
union     = a | b    # a.union(b)
intersect = a & b    # a.intersection(b)
subtract  = a - b    # a.subtract(b)

# Transform
translated = result.translate(1.0, 0.0, 0.0)

# Evaluate at points (NumPy array)
import numpy as np
points = np.array([[0.5, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
distances = sdf.eval_batch(translated, points)

# Compiled evaluation (2-5x faster for repeated use)
compiled = sdf.compile_sdf(sphere)
distances = compiled.eval_batch(points)               # compiled batch
vertices, indices = compiled.to_mesh((-2,-2,-2), (2,2,2), resolution=64)  # compiled mesh

# Convert to mesh
vertices, indices = sdf.to_mesh(translated, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))

# Export to multiple formats
sdf.export_obj(vertices, indices, "model.obj")
sdf.export_glb(vertices, indices, "model.glb")
sdf.export_fbx(vertices, indices, "model.fbx")
sdf.export_usda(vertices, indices, "model.usda")
sdf.export_alembic(vertices, indices, "model.abc")

# UV unwrap → (positions[N,3], uvs[N,2], indices[M])
positions, uvs, indices = sdf.uv_unwrap(vertices, indices)
```

## Material System

PBR metallic-roughness material system compatible with glTF 2.0, UE5, Unity HDRP, and Godot.

### Material Properties

| Property | Type | Description |
|----------|------|-------------|
| `base_color` | `[f32; 4]` | RGBA base color (linear space) |
| `metallic` | `f32` | 0.0 = dielectric, 1.0 = metal |
| `roughness` | `f32` | 0.0 = mirror, 1.0 = diffuse |
| `emission` | `[f32; 3]` | Emissive color (RGB) |
| `emission_strength` | `f32` | Emissive intensity multiplier |
| `opacity` | `f32` | 0.0 = transparent, 1.0 = opaque |
| `ior` | `f32` | Index of refraction (glass=1.5, water=1.33) |
| `normal_scale` | `f32` | Normal map strength |

### Usage

```rust
use alice_sdf::prelude::*;

// Create materials
let gold = Material::metal("Gold", 1.0, 0.766, 0.336, 0.3);
let glass = Material::glass("Glass", 1.5);
let glow = Material::emissive("Neon", 0.0, 1.0, 0.0, 10.0);

// Material library
let mut lib = MaterialLibrary::new();
let gold_id = lib.add(gold);

// Assign material to shape
let sphere = SdfNode::sphere(1.0).with_material(gold_id);

// Generate mesh with AAA vertex format (UV, tangent, color, material_id)
let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &MarchingCubesConfig::aaa(64));
```

### Vertex Format

The mesh vertex includes all attributes needed for AAA rendering:

| Attribute | Type | Description |
|-----------|------|-------------|
| `position` | `Vec3` | 3D position |
| `normal` | `Vec3` | Surface normal |
| `uv` | `Vec2` | Triplanar-projected texture coordinates |
| `tangent` | `Vec4` | Tangent (xyz=direction, w=handedness) |
| `color` | `[f32; 4]` | Vertex color (RGBA linear) |
| `material_id` | `u32` | Material library index |

## Animation System

Keyframe-based animation of SDF parameters for real-time deformation, morphing, and cinematic sequences.

### Features

- **Interpolation modes**: Linear, Cubic Bezier (Hermite), Step
- **Loop modes**: Once, Loop, PingPong
- **Timeline**: Multiple tracks with named parameters
- **AnimatedSdf**: Time-varying SDF evaluation
- **Morph**: Smooth blending between two SDF shapes

### Usage

```rust
use alice_sdf::prelude::*;

// Create a bouncing sphere animation
let sphere = SdfNode::sphere(1.0);

let mut timeline = Timeline::new("bounce");

let mut ty = Track::new("translate.y").with_loop(LoopMode::PingPong);
ty.add_keyframe(Keyframe::new(0.0, 0.0));
ty.add_keyframe(Keyframe::cubic(0.5, 3.0, 0.0, 0.0));
ty.add_keyframe(Keyframe::new(1.0, 0.0));
timeline.add_track(ty);

let animated = AnimatedSdf::new(sphere, timeline);

// Evaluate at time t=0.25
let node_at_t = animated.evaluate_at(0.25);
let distance = eval(&node_at_t, Vec3::ZERO);

// Morph between two shapes
let sphere = SdfNode::sphere(1.0);
let cube = SdfNode::box3d(1.0, 1.0, 1.0);
let morphed = morph(&sphere, &cube, 0.5); // 50% blend
```

### Supported Track Names

| Track | Description |
|-------|-------------|
| `translate.x/y/z` | Translation offset |
| `rotate.x/y/z` | Euler rotation (radians) |
| `scale` | Uniform scale factor |
| `twist` | Twist strength |
| `bend` | Bend curvature |

## File Formats

### .asdf (ALICE SDF Binary)

Compact binary format with CRC32 integrity check.

```
Header (16 bytes):
  - Magic: "ASDF" (4 bytes)
  - Version: u16 (2 bytes)
  - Flags: u16 (2 bytes)
  - Node count: u32 (4 bytes)
  - CRC32: u32 (4 bytes)

Body:
  - Bincode-serialized SdfTree
```

### .asdf.json (ALICE SDF JSON)

Human-readable JSON format for debugging, interoperability, and LLM generation.

```json
{
  "version": "0.1.0",
  "root": {
    "Subtraction": {
      "a": {"Sphere": {"radius": 1.0}},
      "b": {"Box3d": {"half_extents": [1.5, 1.5, 1.5]}}
    }
  },
  "metadata": null
}
```

This is the same format used by the Text-to-3D pipeline — LLMs generate this JSON directly.

### .obj (Wavefront OBJ)

Standard mesh format with material (.mtl) support.

```rust
use alice_sdf::prelude::*;

let mesh = sdf_to_mesh(&shape, min, max, &MarchingCubesConfig::aaa(64));
export_obj(&mesh, "model.obj", &ObjConfig::default(), Some(&mat_lib))?;
```

### .glb (glTF 2.0 Binary)

Industry-standard 3D format with PBR materials. Compatible with UE5, Unity, Blender, Godot, and web viewers.

```rust
use alice_sdf::prelude::*;

let mesh = sdf_to_mesh(&shape, min, max, &MarchingCubesConfig::aaa(64));
export_glb(&mesh, "model.glb", &GltfConfig::aaa(), Some(&mat_lib))?;
```

### Asset Pipeline Summary

| Format | Import | Export | Materials | Description |
|--------|--------|--------|-----------|-------------|
| `.asdf` | yes | yes | - | Native SDF binary (CRC32) |
| `.asdf.json` | yes | yes | - | Native SDF JSON |
| `.obj` | yes | yes | .mtl | Wavefront OBJ (universal DCC) |
| `.glb` | - | yes | PBR | glTF 2.0 binary (game engines) |
| `.fbx` | - | yes | PBR | FBX 7.4 ASCII/Binary (DCC tools) |
| `.usda` | - | yes | UsdPreviewSurface | USD ASCII (Pixar/Omniverse/Houdini/Maya/Blender) |
| `.abc` | - | yes | - | Alembic Ogawa binary (Maya/Houdini/Nuke/Blender) |
| `.nanite` | - | yes | - | UE5 Nanite hierarchical cluster binary + JSON manifest |
| `.stl` | yes | yes | - | STL ASCII/Binary (3D printing, CAD) |
| `.ply` | yes | yes | - | PLY ASCII/Binary (point clouds, scanning) |
| `.3mf` | - | yes | PBR | 3MF XML (modern 3D printing with materials) |
| `.abm` | yes | yes | - | ALICE Binary Mesh (compact, CRC32, LOD chain support) |
| `.unity_mesh` | - | yes | - | Unity native JSON/binary mesh (left-handed, Z-flip) |
| `.ue5_mesh` | - | yes | - | UE5 native JSON/binary mesh (Z-up, centimeters) |

## Architecture

ALICE-SDF uses a layered architecture. Each SDF primitive/operation is implemented across all 16 core layers, plus specialized modules:

```
Layer 1:  types.rs          -- SdfNode enum (AST definition)
Layer 2:  primitives/       -- Mathematical SDF formulas (Inigo Quilez)
Layer 3:  eval/             -- Recursive interpreter
Layer 4:  compiled/opcode   -- OpCode enum for VM
Layer 5:  compiled/instr    -- Instruction encoding (params[7] + aux_offset/aux_len)
Layer 6:  compiled/compiler -- AST -> instruction compilation (+ aux_data serialization)
Layer 7:  compiled/eval     -- Stack-based VM evaluator (with aux_data deserialization)
Layer 8:  compiled/eval_simd-- SIMD 8-wide evaluator (AVX2/NEON, per-lane aux_data dispatch)
Layer 9:  compiled/eval_bvh -- BVH-accelerated evaluator (AABB pruning)
Layer 10: compiled/glsl     -- GLSL transpiler (Unity/OpenGL/Vulkan)
Layer 11: compiled/wgsl     -- WGSL transpiler (WebGPU)
Layer 12: compiled/hlsl     -- HLSL transpiler (DirectX/Unreal)
Layer 13: compiled/jit      -- JIT native code scalar (Cranelift)
Layer 14: compiled/jit_simd -- JIT SIMD 8-wide native code (Cranelift)
Layer 15: crispy.rs         -- Hardware-native math (branchless, BitMask64, BloomFilter)
Layer 16: interval.rs       -- Interval arithmetic evaluation + Lipschitz bounds

Specialized modules:
  neural.rs     -- Neural SDF (MLP approximation, training, inference)
  collision.rs  -- SDF-to-SDF contact detection with IA pruning
  eval/gradient -- Analytic gradient (chain rules, Jacobian propagation)
  mesh/dual_contouring -- Dual Contouring (QEF vertex placement, sharp edges)
  optimize.rs   -- CSG tree optimization (identity folding, transform merging)
  tight_aabb.rs -- Auto tight AABB (interval arithmetic + binary search)

Analysis & measurement:
  autodiff.rs   -- Dual Number AD (Dual, Dual3), Hessian, mean curvature
  measure.rs    -- Monte Carlo volume/surface area/center of mass estimation
  heatmap.rs    -- Distance field cross-section slicing with color maps
  shell.rs      -- Variable-thickness shell / offset surface modifier
  constraint.rs -- Gauss-Newton parametric constraint solver
  diff.rs       -- CSG tree structural diff and patch (undo/redo, sync)
  sdf2d.rs      -- Pure 2D SDF module (circle, rect, bezier, font glyph)
  font_bridge.rs-- ALICE-Font glyph → SDF conversion (requires `font` feature)
```

### Evaluation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Interpreted** | Recursive tree walk | Debugging, prototyping |
| **Compiled VM** | Stack-based bytecode | General purpose |
| **SIMD 8-wide** | 8 points in parallel (Vec3x8) | Batch evaluation |
| **BVH-accelerated** | AABB spatial pruning | Complex scenes |
| **SoA Batch** | Structure-of-Arrays memory layout | Cache-optimal SIMD batches |
| **JIT Native** | Cranelift machine code | Maximum throughput |
| **GPU Compute** | WGSL compute shaders | Massive batches |

### Shader Transpilers

| Target | Output | Use Case |
|--------|--------|----------|
| **GLSL** | OpenGL/Vulkan shaders | Unity, custom engines |
| **WGSL** | WebGPU shaders | Browser, wgpu |
| **HLSL** | DirectX shaders | Unreal Engine, DirectX |

#### Engine-Specific Shader Exports

| Method | Output | Target Engine |
|--------|--------|---------------|
| `HlslShader::export_ue5_material_function()` | `.ush` Material Function include | Unreal Engine 5 (Custom Expression) |
| `GlslShader::export_unity_shader_graph()` | `.hlsl` Custom Function node | Unity Shader Graph (HDRP/URP) |

## Mesh Module

### Conversion

| Function | Description |
|----------|-------------|
| `sdf_to_mesh()` | SDF to mesh via parallel marching cubes (Z-slab parallelization) |
| `sdf_to_mesh_compiled()` | Compiled VM path — SIMD batch grid eval + grid finite-difference normals |
| `marching_cubes_compiled()` | Compiled MC with `eval_compiled_batch_simd_parallel` grid evaluation |
| `adaptive_marching_cubes()` | Octree-adaptive MC (interpreted) — 60-80% fewer triangles |
| `adaptive_marching_cubes_compiled()` | Octree-adaptive MC (compiled VM) — 2-5x faster |
| `mesh_to_sdf()` | Mesh to SDF via capsule approximation (edge-based) |
| `mesh_to_sdf_exact()` | Mesh to SDF via BVH exact distance (O(log n) queries) |

### Advanced Features

| Feature | Module | Description |
|---------|--------|-------------|
| **Hermite Data** | `mesh/hermite` | Position + normal extraction for Dual Contouring |
| **Primitive Fitting** | `mesh/primitive_fitting` | Detect sphere/box/cylinder in mesh data for CSG reconstruction |
| **Nanite Clusters** | `mesh/nanite` | UE5 Nanite-compatible hierarchical cluster generation with Dual Contouring, tight AABB, material ID, curvature-adaptive density |
| **LOD Generation** | `mesh/lod` | Level-of-detail chain generation for efficient rendering |
| **Decimation LOD** | `mesh/lod` | Progressive decimation-based LOD from high-res base mesh |
| **Adaptive MC** | `mesh/sdf_to_mesh` | Octree-based marching cubes with surface-adaptive subdivision |
| **Mesh Decimation** | `mesh/decimate` | QEM decimation with UV/tangent/material boundary preservation |
| **Convex Decomposition** | `mesh/collision` | V-HACD volumetric convex decomposition for physics |
| **Collision Primitives** | `mesh/collision` | AABB, bounding sphere, convex hull, simplified collision |
| **Lightmap UVs** | `mesh/lightmap` | Automatic lightmap UV generation (UV channel 1) |
| **Vertex Optimization** | `mesh/optimize` | Vertex cache optimization and deduplication |
| **Mesh BVH** | `mesh/bvh` | Bounding volume hierarchy for exact signed distance queries |
| **Manifold Validation** | `mesh/manifold` | Topology validation, repair, and quality metrics |
| **UV Unwrapping** | `mesh/uv_unwrap` | LSCM conformal UV unwrapping with seam detection and chart packing |

### Mesh Persistence (5-Layer Architecture)

| Layer | Module | Description |
|-------|--------|-------------|
| **ABM Binary Format** | `io/abm` | ALICE Binary Mesh — compact binary with CRC32, optional normals/UVs/tangents/colors, LOD chain support |
| **LOD Chain Persistence** | `mesh/lod_persist` | Save/load entire LOD chains (meshes + transition distances) with SDF hash validation |
| **Chunked Mesh Cache** | `cache/chunked` | Thread-safe LRU-like cache with FIFO eviction, dirty tracking, and persistence to disk |
| **Unity Export** | `io/unity_mesh` | JSON and binary export with left-handed coordinate conversion (Z-flip, winding flip, scale) |
| **UE5 Export** | `io/ue5_asset` | JSON and binary export with Z-up coordinate conversion (meters → centimeters) and multi-LOD support |

```rust
use alice_sdf::prelude::*;
use alice_sdf::io::{save_abm, load_abm, export_unity_mesh, export_ue5_mesh, UnityMeshConfig, Ue5MeshConfig};

// Save/load ABM binary mesh
save_abm(&mesh, "model.abm")?;
let loaded = load_abm("model.abm")?;

// Export to Unity (left-handed, Z-flip)
export_unity_mesh(&mesh, "model.unity_mesh", &UnityMeshConfig::default())?;

// Export to UE5 (Z-up, centimeters)
export_ue5_mesh(&mesh, "model.ue5_mesh", &Ue5MeshConfig::default())?;
```

## New: Platonic Solids, TPMS Surfaces, Advanced CSG (v2.0)

### Platonic & Archimedean Solids (GDF)

5 new Platonic/Archimedean solids using the Generalized Distance Function (GDF) approach from hg_sdf. Each is defined by a set of face normals — the SDF is the maximum abs-dot product minus the radius.

| Primitive | Normals | Description |
|-----------|---------|-------------|
| `Tetrahedron` | 4 | Regular tetrahedron |
| `Dodecahedron` | 6 | Regular dodecahedron (12 pentagonal faces) |
| `Icosahedron` | 10 | Regular icosahedron (20 triangular faces) |
| `TruncatedOctahedron` | 7 | Archimedean solid (8 hex + 6 square faces) |
| `TruncatedIcosahedron` | 16 | Soccer ball / C60 (12 pentagons + 20 hexagons) |

All 5 evaluate via native SIMD 8-wide (no scalar fallback) — pure FMA + max operations.

### BoxFrame

IQ's `sdBoxFrame` — a wireframe box showing only edges. Parameters: `half_extents: Vec3`, `edge: f32`.

### TPMS Surfaces (7 new + 2 existing)

Triply-Periodic Minimal Surfaces for lattice/scaffold generation. Each uses `scale` (spatial frequency) and `thickness` (shell half-thickness).

| Surface | Implicit Function |
|---------|-------------------|
| `DiamondSurface` | `sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z) + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z)` |
| `Neovius` | `3(cos(x)+cos(y)+cos(z)) + 4cos(x)cos(y)cos(z)` |
| `Lidinoid` | Complex sin/cos terms with 2x harmonics |
| `IWP` | `2(cos(x)cos(y)+cos(y)cos(z)+cos(z)cos(x)) - (cos(2x)+cos(2y)+cos(2z))` |
| `FRD` | `cos(2x)sin(y)cos(z) + cos(x)cos(2y)sin(z) + sin(x)cos(y)cos(2z)` |
| `FischerKochS` | FRD - 0.4 |
| `PMY` | `2cos(x)cos(y)cos(z) + sin(2x)sin(y) + sin(x)sin(2z) + sin(2y)sin(z)` |

All use **double-angle identities** (`cos(2x) = 2cos²(x)-1`) to eliminate redundant trig calls, and evaluate via **native SIMD** with `sin_approx`/`cos_approx`.

### Advanced CSG Operations (9 new)

| Operation | Formula | Source |
|-----------|---------|--------|
| `XOR` | `min(a,b).max(-max(a,b))` | IQ |
| `Morph` | `a*(1-t) + b*t` | libfive/Curv |
| `ColumnsUnion` | Column-shaped blending with `r` radius, `n` columns | hg_sdf |
| `ColumnsIntersection` | Column blend at intersection boundary | hg_sdf |
| `ColumnsSubtraction` | Column blend at subtraction boundary | hg_sdf |
| `Pipe` | `length(vec2(a,b)) - r` | hg_sdf |
| `Engrave` | `max(a, (a+r-abs(b))*√½)` | hg_sdf |
| `Groove` | `max(a, min(a+ra, rb-abs(b)))` | hg_sdf |
| `Tongue` | `min(a, max(a-ra, abs(b)-rb))` | hg_sdf |

### OctantMirror Modifier

48-fold symmetry: `abs()` all coordinates + sort descending (`x ≥ y ≥ z`). Creates full octahedral symmetry group. OpCode 63.

### Nanite Enhancements

| Feature | Description |
|---------|-------------|
| **Dual Contouring** | Optional DC mesh generation per LOD for sharp-edge shapes |
| **Tight AABB** | Automatic bounds shrinking via interval arithmetic |
| **Material ID** | Per-cluster material assignment via `eval_material()` at centroid |
| **Curvature Adaptive** | High-curvature regions get finer cluster density |
| **HLSL Material Export** | `export_nanite_hlsl_material()` generates UE5 `.usf` with `AliceSdfNormal()` and `AliceSdfMaterial()` |

### 2D Primitives (6 new)

Extruded 2D shapes for UI, decals, and cross-sections. Each is evaluated in the XY plane and extruded along Z.

| Primitive | Parameters | Description |
|-----------|-----------|-------------|
| `Circle2D` | `radius` | 2D circle |
| `Rect2D` | `half_width, half_height` | 2D rectangle |
| `Segment2D` | `a: Vec2, b: Vec2` | Line segment |
| `Polygon2D` | `vertices: Vec<Vec2>` | Arbitrary polygon |
| `RoundedRect2D` | `half_width, half_height, radius` | Rounded rectangle |
| `Annular2D` | `outer_radius, inner_radius` | Ring/annulus |

All 2D primitives support full SIMD evaluation and GLSL/HLSL/WGSL transpilation.

### ExpSmooth Operations (3 new)

IQ's exponential smooth min/max — C-infinity smooth with exponential falloff. Better for organic shapes than polynomial smooth.

| Operation | Formula | Description |
|-----------|---------|-------------|
| `ExpSmoothUnion` | `exp(-k*a) + exp(-k*b)` → `-ln(sum)/k` | Exponential smooth union |
| `ExpSmoothIntersection` | Negated ExpSmoothUnion | Exponential smooth intersection |
| `ExpSmoothSubtraction` | ExpSmoothIntersection(a, -b) | Exponential smooth subtraction |

### Shear Transform

3-axis shear deformation: `y' = y - shear.x * x`, `z' = z - shear.y * x - shear.z * y`. Useful for italic text, architectural slopes, and non-rigid deformation.

### Animated Modifier

Timeline-driven parameter animation modifier. Wraps a child SDF with keyframe animation tracks (translate, rotate, scale, twist, bend). Evaluated by sampling the timeline at a given time `t`.

### STL / PLY / 3MF I/O

| Format | Import | Export | Description |
|--------|--------|--------|-------------|
| STL | ASCII + Binary | ASCII + Binary | Standard 3D printing format |
| PLY | ASCII + Binary | ASCII + Binary | Point cloud and scanning format |
| 3MF | - | XML | Modern 3D printing with PBR materials |

### Performance Optimizations (v2.1)

| Optimization | Target | Effect |
|-------------|--------|--------|
| **Schraudolph fast exp** | SIMD eval (ExpSmooth) | ~25 → ~3 cycles/lane (~0.3% error) |
| **IEEE 754 fast ln** | SIMD eval (ExpSmooth) | Exponent extraction + Pade (~0.4% error) |
| **Fast reciprocal** | SIMD eval (atan2) | Bit trick + Newton-Raphson (~0.02% error) |
| **Ellipsoid division halving** | SIMD eval | 6 → 3 divisions via precomputed inverse |
| **Triplanar UV division exorcism** | Mesh generation | `1/scale` precomputed once per triangle |
| **Zero-Copy NumPy** | Python bindings | `with_numpy_as_vec3()` eliminates intermediate Vec allocation |

### smooth_min_root (IQ)

C-infinity square root smooth minimum: `0.5 * (a + b - sqrt((b-a)² + k²))`. Unlike polynomial `smooth_min`, this has a constant blend width regardless of input separation.

## Interval Arithmetic

Conservative evaluation of SDFs over axis-aligned bounding boxes (AABBs). Returns `[lo, hi]` bounds on the distance field — if `lo > 0`, the entire region is outside the surface and can be skipped.

```rust
use alice_sdf::interval::{eval_interval, eval_lipschitz, Interval, Vec3Interval};
use alice_sdf::prelude::*;

let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));

// Evaluate over a spatial region
let region = Vec3Interval::from_bounds(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
let bounds = eval_interval(&shape, region);
// bounds.lo > 0 → entire region is outside, skip it

// Lipschitz constant for adaptive step sizing
let lip = eval_lipschitz(&shape); // 1.0 for distance-preserving shapes
```

Supports all 72 primitives, 24 operations, transforms, and modifiers. Used internally by relaxed sphere tracing and SDF-to-SDF collision for spatial pruning.

## Neural SDF

Pure-Rust MLP (multi-layer perceptron) that learns to approximate an SDF tree. No external ML dependencies. Useful for accelerating evaluation of complex trees with many nodes.

```rust
use alice_sdf::neural::{NeuralSdf, NeuralSdfConfig};
use alice_sdf::prelude::*;

let complex_scene = SdfNode::sphere(1.0)
    .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5).translate(1.5, 0.0, 0.0), 0.3);

// Train (samples random points, evaluates tree, fits MLP)
let config = NeuralSdfConfig::default(); // 3 layers, 64 wide, positional encoding
let nsdf = NeuralSdf::train(&complex_scene, Vec3::splat(-3.0), Vec3::splat(3.0), &config);

// Evaluate (~10-100x faster than tree for complex scenes)
let d = nsdf.eval(Vec3::new(0.0, 0.0, 0.0));

// Save/load binary weights
let mut buf = Vec::new();
nsdf.save(&mut buf).unwrap();
let loaded = NeuralSdf::load(&mut &buf[..]).unwrap();
```

Features: Xavier initialization, Adam optimizer, positional encoding, ReLU activations, compact binary serialization (`b"NSDF"` format).

## SDF-to-SDF Collision

Grid-based contact detection between two SDF fields with interval arithmetic AABB pruning.

```rust
use alice_sdf::prelude::*;
use alice_sdf::collision::*;

let a = SdfNode::sphere(1.0);
let b = SdfNode::sphere(1.0).translate(1.5, 0.0, 0.0);
let aabb = Aabb { min: Vec3::splat(-3.0), max: Vec3::splat(3.0) };

// Fast boolean overlap test (early exit)
if sdf_overlap(&a, &b, &aabb, 16) {
    // Detailed contact points (sorted by depth, deepest first)
    let contacts = sdf_collide(&a, &b, &aabb, 32);
    for c in &contacts {
        println!("point={}, normal={}, depth={}", c.point, c.normal, c.depth);
    }
}

// Minimum separation distance (0.0 if overlapping)
let dist = sdf_distance(&a, &b, &aabb, 16);
```

| Function | Description |
|----------|-------------|
| `sdf_overlap()` | Boolean intersection test with early exit |
| `sdf_collide()` | Contact points with position, normal, and penetration depth |
| `sdf_distance()` | Minimum separation distance (upper bound) |

All functions use interval arithmetic to skip grid cells where either SDF is provably positive, typically pruning 80-95% of cells.

## Analytic Gradient

Single-pass gradient computation via chain rules and Jacobian transpose propagation. Replaces the default 6-evaluation finite-difference method with analytic derivatives where possible.

```rust
use alice_sdf::prelude::*;

let shape = SdfNode::sphere(1.0)
    .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5).translate(1.0, 0.0, 0.0), 0.3);

// Analytic gradient (single tree pass)
let grad = eval_gradient(&shape, Vec3::new(0.5, 0.0, 0.0));

// Analytic normal (normalized gradient)
let n = eval_normal(&shape, Vec3::new(0.5, 0.0, 0.0));
```

| Component | Method | Coverage |
|-----------|--------|----------|
| **9 primitives** | Closed-form analytic | Sphere, Box3d, Plane, Cylinder, Torus, Capsule, InfiniteCylinder, Gyroid, SchwarzP |
| **44 primitives** | Numerical fallback (6 leaf evals) | All remaining complex primitives |
| **12 operations** | Chain rule | Union, Intersection, Subtraction + Smooth/Chamfer/Stairs variants |
| **4 transforms** | Jacobian transpose | Translate, Rotate, Scale, ScaleNonUniform |
| **10 modifiers** | Analytic Jacobian | Round, Onion, Elongate, Mirror, Revolution, Extrude, Twist, Bend, RepeatInfinite, RepeatFinite |
| **5 modifiers** | Numerical fallback | Noise, Taper, Displacement, PolarRepeat, SweepBezier |

For a tree of depth D with N leaf nodes, the standard numerical gradient requires `6 × (full tree eval)`. The analytic gradient requires at most N leaf evaluations plus ~6 per complex leaf — significantly cheaper for deep trees.

## Dual Contouring

Dual Contouring places one vertex per cell using QEF (Quadric Error Function) minimization, producing meshes that preserve sharp edges and corners better than Marching Cubes.

```rust
use alice_sdf::prelude::*;

let shape = SdfNode::box3d(1.0, 1.0, 1.0)
    .subtract(SdfNode::cylinder(0.3, 2.0));

let config = DualContouringConfig {
    resolution: 64,
    ..Default::default()
};

let mesh = dual_contouring(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

// Compiled version for 2-5x speedup
let compiled = CompiledSdf::compile(&shape);
let mesh_fast = dual_contouring_compiled(&compiled, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
```

| Feature | Marching Cubes | Dual Contouring |
|---------|---------------|-----------------|
| Sharp edges | Rounded off | Preserved |
| Vertex placement | Edge intersections | QEF-optimal per cell |
| Topology | Triangles from lookup table | Quads from shared edges |
| Best for | Organic shapes | Hard-surface / CAD models |

## CSG Tree Optimization

Reduces SDF tree size by removing redundant nodes before evaluation or mesh generation.

```rust
use alice_sdf::prelude::*;
use alice_sdf::optimize::{optimize, optimization_stats};

let shape = SdfNode::sphere(1.0)
    .translate(0.0, 0.0, 0.0)  // identity → removed
    .scale(1.0)                 // identity → removed
    .translate(1.0, 0.0, 0.0)
    .translate(0.0, 2.0, 0.0); // merges → Translate(1, 2, 0)

let optimized = optimize(&shape);
let stats = optimization_stats(&shape, &optimized);
println!("{}", stats); // "5 → 2 nodes (3 removed, 60.0% reduction)"
```

| Optimization | Before | After |
|-------------|--------|-------|
| `Translate(0,0,0)` | 2 nodes | 1 node (child only) |
| `Scale(1.0)` / `Rotate(identity)` | 2 nodes | 1 node |
| `Translate(Translate(c,a),b)` | 3 nodes | 2 nodes (`Translate(c,a+b)`) |
| `Scale(Scale(c,a),b)` | 3 nodes | 2 nodes (`Scale(c,a*b)`) |
| `Rotate(Rotate(c,r1),r2)` | 3 nodes | 2 nodes (`Rotate(c,r2*r1)`) |
| `SmoothUnion(k=0)` | 3 nodes | 3 nodes (demoted to `Union`) |
| `Round(0.0)` / `Twist(0.0)` / `Bend(0.0)` | 2 nodes | 1 node |

## Auto Tight AABB

Computes a minimal axis-aligned bounding box for any SDF tree using interval arithmetic and binary search. Useful for automatic mesh generation bounds, spatial indexing, and culling.

```rust
use alice_sdf::prelude::*;
use alice_sdf::tight_aabb::{compute_tight_aabb, compute_tight_aabb_with_config, TightAabbConfig};

let shape = SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0);
let aabb = compute_tight_aabb(&shape);
// aabb ≈ (1.0, -1.0, -1.0) to (3.0, 1.0, 1.0)

// Custom config for larger shapes
let config = TightAabbConfig {
    initial_half_size: 50.0,   // search range [-50, 50]³
    bisection_iterations: 25,  // ~1e-7 precision
    coarse_subdivisions: 16,   // finer initial scan
};
let aabb = compute_tight_aabb_with_config(&shape, &config);
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_half_size` | 10.0 | Search cube `[-h, h]³`. Must contain the shape. |
| `bisection_iterations` | 20 | Precision ≈ `2h / 2^iters` per axis |
| `coarse_subdivisions` | 8 | Coarse scan slabs before binary search |

### Manifold Mesh Guarantee

Ensures watertight, manifold meshes suitable for physics, rendering, and 3D printing.

```rust
use alice_sdf::prelude::*;

let mesh = sdf_to_mesh(&shape, min, max, &MarchingCubesConfig::default());

// Validate
let report = validate_mesh(&mesh);
println!("{}", report); // Prints full validation report

// Repair
let repaired = MeshRepair::repair_all(&mesh, 1e-6);

// Quality metrics
let quality = compute_quality(&repaired);
println!("Avg aspect ratio: {}", quality.avg_aspect_ratio);
```

| Function | Description |
|----------|-------------|
| `validate_mesh()` | Non-manifold edge detection, boundary edges, degenerate triangles, duplicate vertices, normal consistency |
| `MeshRepair::remove_degenerate_triangles()` | Remove zero-area triangles |
| `MeshRepair::merge_duplicate_vertices()` | Spatial-hash based vertex welding |
| `MeshRepair::fix_normals()` | Fix inconsistent winding order |
| `MeshRepair::repair_all()` | Run all repairs in sequence |
| `compute_quality()` | Aspect ratio and area statistics |

## Texture Fitting (Texture-to-Formula)

Converts bitmap textures (PNG/JPG) into resolution-independent procedural noise formulas. The fitted formula can be rendered on GPU at **any resolution** without the original image.

```
texture(u,v) ≈ bias + Σᵢ aᵢ · noise(uv · fᵢ + φᵢ, seedᵢ)
```

The CPU noise implementation (`hash_noise_3d`) exactly matches the GPU version in WGSL/HLSL/GLSL, guaranteeing CPU fitting = GPU rendering.

### Pipeline

1. Load image → grayscale f32
2. DCT frequency analysis → dominant bands
3. Greedy octave-by-octave fitting (Nelder-Mead, SIMD f32x8 + rayon parallel)
4. Export as JSON parameters and/or standalone shader function

### CLI

```bash
# Basic: fit texture and print results
alice-sdf texture-fit granite.png

# Export JSON parameters + HLSL shader
alice-sdf texture-fit granite.png -o params.json --shader hlsl --shader-output granite.hlsl

# High quality: more octaves, higher PSNR target
alice-sdf texture-fit marble.png --octaves 12 --target-psnr 35.0 --iterations 2000
```

### Rust API

```rust
use alice_sdf::texture::{fit_texture, generate_shader, ShaderLanguage, TextureFitConfig};
use std::path::Path;

let config = TextureFitConfig {
    max_octaves: 8,
    target_psnr_db: 28.0,
    iterations_per_octave: 500,
    tileable: true,
};

let result = fit_texture(Path::new("granite.png"), &config).unwrap();
println!("PSNR: {:.1} dB, {} octaves", result.psnr_db, result.octaves[0].len());

// Generate standalone WGSL shader
let shader = generate_shader(&result, ShaderLanguage::Wgsl, "granite.png");
```

### Output Shader Languages

| Target | Function Signature | Use Case |
|--------|-------------------|----------|
| **WGSL** | `fn procedural_texture(uv: vec2<f32>) -> f32` | WebGPU, wgpu |
| **HLSL** | `float procedural_texture(float2 uv)` | Unity, Unreal, DirectX |
| **GLSL** | `float procedural_texture(vec2 uv)` | OpenGL, Vulkan |

## Raymarching

Sphere tracing for ray-SDF intersection with specialized optimizations:

| Function | Description |
|----------|-------------|
| `raymarch()` | Single ray intersection with sphere tracing |
| `raymarch_relaxed()` | Relaxed sphere tracing with Lipschitz-adaptive over-relaxation |
| `raymarch_batch()` | Batch ray evaluation |
| `raymarch_batch_parallel()` | Parallel batch via Rayon |
| `render_depth()` | Depth buffer rendering |
| `render_normals()` | Normal map rendering |

Features: dedicated Shadow/AO loops (skip normal computation), early exit for hard shadows, configurable iteration limits via `RaymarchConfig`.

### Relaxed Sphere Tracing

Uses `RaymarchConfig::relaxed(node)` to automatically compute the Lipschitz constant from the SDF tree and apply over-relaxation (ω ∈ [1, 2)) with safety fallback:

```rust
use alice_sdf::prelude::*;

let shape = SdfNode::sphere(1.0).twist(0.5);
let config = RaymarchConfig::relaxed(&shape); // auto Lipschitz
let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
let hit = raymarch_with_config(&shape, ray, 20.0, &config);
```

Step formula: `step = d / L × ω` with safety check — if `d < prev_step - prev_dist`, falls back to `d / L`. Typically reaches surfaces in 30-50% fewer steps compared to standard sphere tracing.

## FFI & Language Bindings

### C/C++ (`include/alice_sdf.h`)

```c
#include "alice_sdf.h"

// Evaluate SDF
AliceSdfHandle sdf = alice_sdf_sphere(1.0);
float dist = alice_sdf_eval(sdf, 0.5, 0.0, 0.0);

// Generate mesh once, export to multiple formats
MeshHandle mesh = alice_sdf_generate_mesh(sdf, 64, 2.0);
alice_sdf_export_glb(mesh, NULL, "model.glb", 0, 0);
alice_sdf_export_fbx(mesh, NULL, "model.fbx", 0, 0);
alice_sdf_export_obj(mesh, NULL, "model.obj", 0, 0);

// --- Mesh Persistence ---

// Save/load ABM binary mesh
alice_sdf_save_abm(mesh, NULL, "model.abm");
MeshHandle loaded = alice_sdf_load_abm("model.abm");

// Export for Unity (left-handed, Z-flip, scale=1.0)
alice_sdf_export_unity(mesh, NULL, "model.unity_mesh", 1/*flip_z*/, 1/*flip_winding*/, 1.0f);
alice_sdf_export_unity_binary(mesh, NULL, "model.unity_mesh_bin", 1, 1, 1.0f);

// Export for UE5 (Z-up, centimeters, scale=100.0)
alice_sdf_export_ue5(mesh, NULL, "model.ue5_mesh", 100.0f);
alice_sdf_export_ue5_binary(mesh, NULL, "model.ue5_mesh_bin", 100.0f);

// Save/load LOD chain (multiple detail levels)
MeshHandle lods[4] = { mesh_lod0, mesh_lod1, mesh_lod2, mesh_lod3 };
float distances[4] = { 0.0f, 10.0f, 30.0f, 100.0f };
alice_sdf_save_lod_chain(lods, distances, 4, "asset_lod");

MeshHandle out_lods[8];
float out_dist[8];
int count = alice_sdf_load_lod_chain("asset_lod", out_lods, out_dist, 8);

alice_sdf_free_mesh(loaded);
alice_sdf_free_mesh(mesh);
alice_sdf_free(sdf);
```

### C# / Unity (`bindings/AliceSdf.cs`)

```csharp
using AliceSdf;

// --- Basic SDF ---
var sdf = AliceSdf.Sphere(1.0f);
float dist = sdf.Eval(new Vector3(0.5f, 0f, 0f));

// --- Mesh Generation & Export ---
var mesh = sdf.GenerateMesh(resolution: 64, bounds: 2.0f);

// Save to ABM (fast binary, good for asset caching)
mesh.SaveAbm("Assets/Cache/sphere.abm");

// Load from ABM
var cached = AliceSdf.LoadAbm("Assets/Cache/sphere.abm");

// Export for Unity (auto left-handed conversion)
mesh.ExportUnity("Assets/Meshes/sphere.unity_mesh");
mesh.ExportUnity("Assets/Meshes/sphere.unity_mesh", flipZ: true, flipWinding: true, scale: 1.0f);

// Export for UE5 (auto Z-up + cm conversion)
mesh.ExportUE5("sphere.ue5_mesh", scale: 100.0f);

// --- LOD Chain ---
var lod0 = sdf.GenerateMesh(resolution: 128, bounds: 2.0f);
var lod1 = sdf.GenerateMesh(resolution: 64,  bounds: 2.0f);
var lod2 = sdf.GenerateMesh(resolution: 32,  bounds: 2.0f);

AliceSdf.SaveLodChain(
    meshes: new[] { lod0, lod1, lod2 },
    distances: new[] { 0f, 15f, 50f },
    path: "Assets/LOD/sphere_lod"
);

var (lodMeshes, lodDistances) = AliceSdf.LoadLodChain("Assets/LOD/sphere_lod");
// lodMeshes[0] = highest detail (distance >= 0)
// lodMeshes[2] = lowest detail  (distance >= 50)
```

### Python (PyO3)

```bash
pip install alice-sdf  # or: maturin develop --features python
```

```python
import alice_sdf as sdf
import numpy as np

# --- Basic SDF ---
node = sdf.sphere(1.0)
d = node.eval(0.0, 0.0, 0.0)       # -1.0

# Analytic gradient (GIL released)
gx, gy, gz = node.gradient(1.0, 0.0, 0.0)  # (1.0, 0.0, 0.0)

# Auto tight AABB (Rayon parallel, GIL released)
(min_xyz, max_xyz) = node.tight_aabb()      # ((-1.0,-1.0,-1.0), (1.0,1.0,1.0))

# CSG tree optimization (GIL released)
optimized = node.translate(0, 0, 0).optimize()  # identity removed

# Dual contouring mesh (sharp edges, GIL released, NumPy zero-copy)
verts, indices = sdf.to_mesh_dual_contouring(node, (-2,-2,-2), (2,2,2), resolution=64)
```

#### Python: Mesh Persistence & Engine Export

```python
import alice_sdf as sdf

# Create a shape and generate mesh
shape = sdf.SdfNode.sphere(1.0) - sdf.SdfNode.box3d(0.5, 0.5, 0.5)
verts, indices = sdf.to_mesh(shape, (-2,-2,-2), (2,2,2))

# --- ABM Binary Format ---
# Fast save/load for asset caching (GIL released, ~10x faster than OBJ)
sdf.save_abm(verts, indices, "model.abm")
verts2, indices2 = sdf.load_abm("model.abm")  # Zero-Copy NumPy return

# --- Unity Export ---
# Automatic left-handed coordinate conversion (Z-flip + winding)
sdf.export_unity(verts, indices, "model.unity_mesh")

# Custom parameters
sdf.export_unity(verts, indices, "model.unity_mesh",
    flip_z=True,          # Unity is left-handed (default: True)
    flip_winding=True,    # Flip triangle winding (default: True)
    scale=1.0             # Scale factor (default: 1.0)
)

# --- UE5 Export ---
# Automatic Z-up + meters-to-centimeters conversion
sdf.export_ue5(verts, indices, "model.ue5_mesh")

# Custom scale (UE5 uses centimeters, default: 100.0)
sdf.export_ue5(verts, indices, "model.ue5_mesh", scale=100.0)

# --- Chunked Mesh Cache ---
# Thread-safe cache with FIFO eviction (useful for streaming/open worlds)
cache = sdf.PyMeshCache(max_chunks=256, chunk_size=1.0)
print(f"Chunks: {len(cache)}, Empty: {cache.is_empty()}")
print(f"Memory: {cache.memory_usage()} bytes")
cache.clear()  # Evict all cached chunks
```

#### Python: Full Workflow Example

```python
import alice_sdf as sdf

# 1. Design shape via CSG
base = sdf.SdfNode.box3d(2.0, 0.2, 2.0)          # floor
pillar = sdf.SdfNode.cylinder(0.15, 1.5)           # pillar
pillars = pillar.repeat_finite(1.5, 0.0, 1.5, 2, 0, 2)  # 3x3 grid
roof = sdf.SdfNode.box3d(2.5, 0.1, 2.5).translate(0, 1.6, 0)
temple = base | pillars | roof                      # union via operator

# 2. Generate mesh
verts, indices = sdf.to_mesh(temple, (-4,-1,-4), (4,3,4))

# 3. Export to all engines simultaneously
sdf.export_obj(verts, indices, "temple.obj")       # Blender / DCC tools
sdf.export_glb(verts, indices, "temple.glb")       # Web / glTF viewers
sdf.save_abm(verts, indices, "temple.abm")         # Fast binary cache
sdf.export_unity(verts, indices, "temple.unity_mesh")   # Unity
sdf.export_ue5(verts, indices, "temple.ue5_mesh")       # Unreal Engine 5
```

### FFI Mesh Export

| Function | Format | Notes |
|----------|--------|-------|
| `alice_sdf_generate_mesh` | — | Generate mesh once (returns `MeshHandle`) |
| `alice_sdf_export_obj` | `.obj` | Wavefront OBJ |
| `alice_sdf_export_glb` | `.glb` | Binary glTF 2.0 |
| `alice_sdf_export_fbx` | `.fbx` | Autodesk FBX |
| `alice_sdf_export_usda` | `.usda` | Universal Scene Description |
| `alice_sdf_export_alembic` | `.abc` | Alembic (Ogawa) |
| `alice_sdf_save_abm` | `.abm` | ALICE Binary Mesh |
| `alice_sdf_load_abm` | `.abm` | Load ABM → MeshHandle |
| `alice_sdf_export_unity` | `.unity_mesh` | Unity JSON mesh |
| `alice_sdf_export_unity_binary` | `.unity_mesh_bin` | Unity binary mesh |
| `alice_sdf_export_ue5` | `.ue5_mesh` | UE5 JSON mesh |
| `alice_sdf_export_ue5_binary` | `.ue5_mesh_bin` | UE5 binary mesh |
| `alice_sdf_save_lod_chain` | `.abm` + `.json` | Save LOD chain |
| `alice_sdf_load_lod_chain` | `.abm` + `.json` | Load LOD chain |
| `alice_sdf_free_mesh` | — | Free mesh handle |

All export functions accept `MeshHandle` (pre-generated) or `SdfHandle` (generates on the fly).

### FFI Performance Hierarchy

| Function | Speed | Use Case |
|----------|-------|----------|
| `alice_sdf_eval_soa` | Fastest | Physics, particles, tracing |
| `alice_sdf_eval_compiled_batch` | Fast | General batch evaluation |
| `alice_sdf_eval_batch` | Medium | Convenience (auto-compile) |
| `alice_sdf_eval` | Slow | Debugging only |

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `cli` (default) | Command-line interface | clap |
| `python` | Python bindings | pyo3, numpy |
| `jit` | JIT native code compilation | cranelift |
| `gpu` | WebGPU compute shaders | wgpu, pollster, bytemuck |
| `glsl` | GLSL shader transpiler | - |
| `hlsl` | HLSL shader transpiler | - |
| `ffi` | C/C++/C# FFI bindings | lazy_static |
| `unity` | Unity integration | ffi + glsl |
| `unreal` | Unreal Engine integration | ffi + hlsl |
| `all-shaders` | All shader transpilers | gpu + hlsl + glsl |
| `texture-fit` | Texture-to-formula conversion | image, rayon, wide |
| `font` | ALICE-Font glyph → SDF bridge | alice-font |
| `physics` | ALICE-Physics integration (SdfField trait) | alice-physics |

```bash
# Examples
cargo build --features "jit,gpu"          # JIT + GPU
cargo build --features unity              # Unity (FFI + GLSL)
cargo build --features unreal             # Unreal (FFI + HLSL)
cargo build --features "all-shaders,jit"  # Everything
cargo build --features physics             # ALICE-Physics integration
cargo build --features font                # ALICE-Font glyph bridge
```

## Physics Bridge (ALICE-Physics Integration)

ALICE-SDF can serve as a collision shape backend for [ALICE-Physics](../ALICE-Physics), a deterministic 128-bit fixed-point physics engine. Instead of GJK/EPA on convex hulls, the physics engine samples the SDF distance field directly — O(1) per query with mathematically exact surfaces.

### How It Works

```
ALICE-Physics (Fix128 world)          ALICE-SDF (f32 world)
┌──────────────────────┐             ┌──────────────────────┐
│ PhysicsWorld         │             │ CompiledSdfField     │
│   bodies: Vec<Body>  │──query──▶  │   distance(x,y,z)→f32│
│   sdf_colliders: Vec │             │   normal(x,y,z)→Vec3 │
│   step(dt)           │◀─contact─  │   distance_and_normal │
│                      │             │     → (f32, Vec3)     │
│ Fix128 ←→ f32 bridge│             │ 4-eval tetrahedral    │
└──────────────────────┘             └──────────────────────┘
```

The `physics` feature gate enables the `SdfField` trait implementation. All coordinate conversion between Fix128 and f32 happens at the trait boundary.

### Enable

```toml
# Cargo.toml
[dependencies]
alice-sdf = { path = "../ALICE-SDF", features = ["physics"] }
```

### Usage (ALICE-SDF side)

```rust
use alice_sdf::prelude::*;
use alice_sdf::physics_bridge::CompiledSdfField;

// Create an SDF shape
let shape = SdfNode::sphere(1.0)
    .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);

// Wrap as physics-ready field
let field = CompiledSdfField::new(shape);

// The field implements alice_physics::SdfField trait:
//   field.distance(x, y, z) → f32       (1 eval)
//   field.normal(x, y, z)   → (f32,f32,f32)  (4 evals, tetrahedral)
//   field.distance_and_normal(x, y, z)   → (f32, (f32,f32,f32))  (4 evals combined)
```

### Performance

| Operation | Evals | Notes |
|-----------|-------|-------|
| `distance()` | 1 | Single compiled SDF evaluation |
| `normal()` | 4 | Tetrahedral gradient (4 offset samples) |
| `distance_and_normal()` | 4 | Combined: distance ≈ avg of 4, normal from differences |

The 4-eval tetrahedral method computes both distance (average of 4 samples, O(epsilon²) error) and normal from the same 4 evaluations, saving 1 eval compared to the naive 1+4 approach.

### Integration with ALICE-Physics

See the [ALICE-Physics README](../ALICE-Physics/README.md#sdf-collider-alice-sdf-integration) for how to register `CompiledSdfField` as a physics collider.

## Testing

1,079 tests across all modules (primitives, operations, transforms, modifiers, compiler, evaluators, BVH, I/O, mesh, shader transpilers, materials, animation, manifold, OBJ, glTF, FBX, USD, Alembic, Nanite, STL, PLY, 3MF, ABM, UV unwrap, mesh collision, decimation, LOD, adaptive MC, dual contouring, CSG optimization, tight AABB, crispy utilities, BloomFilter, interval arithmetic, Lipschitz bounds, relaxed sphere tracing, neural SDF, SDF-to-SDF collision, analytic gradient, 2D primitives, ExpSmooth operations, Shear transform, Animated modifier, mesh persistence, chunked cache, Unity/UE5 export, FFI bindings, autodiff, shell, measure, heatmap, diff, constraint, sdf2d, font_bridge).

```bash
cargo test                  # default features (1,069 tests)
cargo test --features font  # + ALICE-Font bridge (1,079 tests)
```

## CLI

```bash
# Display file info
alice-sdf info model.asdf

# Convert between formats (.asdf ↔ .asdf.json)
alice-sdf convert model.asdf -o model.asdf.json

# Export SDF to mesh (auto-detect format from extension)
alice-sdf export model.asdf -o model.glb
alice-sdf export model.asdf -o model.obj --resolution 128 --bounds 3.0
alice-sdf export model.asdf -o scene.usda
alice-sdf export model.asdf -o anim.abc
alice-sdf export model.asdf -o model.fbx
alice-sdf export model.asdf -o model.stl
alice-sdf export model.asdf -o model.3mf
alice-sdf export model.asdf -o model.ply

# 3D Print: SDF → printable mesh (auto-detect bounds, high resolution)
alice-sdf print model.asdf.json                     # → model.3mf (default)
alice-sdf print model.asdf.json -o basket.stl       # → STL format
alice-sdf print model.asdf.json -r 256              # higher resolution
alice-sdf print model.asdf.json -r 256 -b 150.0     # manual bounds (mm)

# Generate demo SDF
alice-sdf demo -o demo.asdf

# Benchmark evaluation
alice-sdf bench --points 1000000
```

Supported export formats: `.obj`, `.glb`, `.fbx`, `.usda`, `.abc`, `.stl`, `.ply`, `.3mf`

## 3D Print Pipeline (LLM → SDF → Slicer → Printer)

ALICE-SDF enables a direct **"describe it → print it"** workflow for FDM/SLA 3D printers.

```
User Prompt → LLM (Claude/Gemini) → ALICE-SDF JSON → alice-sdf print → .3mf/.stl → Slicer → Printer
                                         ~5s              ~58ms            ~0s
```

### How It Works

1. **Describe** — natural language prompt to LLM with `prompts/sdf_generation_system_prompt.md`
2. **Generate** — LLM outputs ALICE-SDF JSON (CSG tree with dimensions in mm)
3. **Mesh** — `alice-sdf print model.asdf.json` compiles to SIMD VM → Marching Cubes → 3MF/STL
4. **Slice** — Bambu Studio / Orca Slicer / PrusaSlicer converts mesh to G-code
5. **Print** — Send to printer via network or USB

### Print Command Performance

| Step | Time | Notes |
|------|------|-------|
| SDF compile | <0.1ms | SdfNode → CompiledSdf (SIMD 8-wide bytecode) |
| Bounds detect | ~2ms | Interval arithmetic tight AABB (500mm search) |
| Mesh generation | ~37ms | Compiled Marching Cubes, Rayon parallel, res=128 |
| 3MF export | ~20ms | ZIP compression |
| **Total** | **~58ms** | 105K triangles from 11 nodes (release build, M3 Max) |

### Example: Storage Basket

```bash
# Load the included example (200x120x80mm basket with diamond mesh pattern)
alice-sdf print examples/storage_basket.asdf.json -o basket.3mf -r 256

# Result: ~105K triangles, 8.9 MB 3MF, ready for Bambu Studio
```

### LLM System Prompt

A complete system prompt for LLM-based SDF generation is provided at `prompts/sdf_generation_system_prompt.md`. It includes:

- Full SdfNode JSON reference (primitives, operations, transforms, modifiers)
- 3D printing design rules (wall thickness, overhangs, feature size)
- Composition patterns (hollow containers, repeated patterns, organic shapes)
- Worked example (storage basket with Onion shell + RepeatInfinite mesh)

### Bambu Lab Integration (MQTT + FTPS)

For direct printer integration (Bambu Lab X1C, H2S, etc.), ALICE can control printers via:

| Protocol | Port | Purpose |
|----------|------|---------|
| MQTTS | 8883 | Status monitoring, print commands |
| FTPS | 990 | File transfer (.3mf upload) |

Auth: username `bblp`, password = LAN Access Code from printer display.

```
ALICE-SDF (Rust) → .3mf → FTPS upload → MQTT "print" command → Bambu H2S
                                                                     │
ALICE-View ← MQTT telemetry (XYZ position, temperature, progress) ──┘
```

## Performance

Benchmarked on Apple M3 Max, Rust 1.75+, `--release` build.

### Single Point Evaluation

| Primitive | Time |
|-----------|------|
| Sphere | 6.1 ns |
| Box3D | 5.0 ns |
| Cylinder | 8.0 ns |
| Torus | 9.3 ns |

| Operation | Time |
|-----------|------|
| Union (2 nodes) | 13.3 ns |
| Smooth Union | 21.4 ns |
| Complex tree (5 nodes) | 12.6 ns |
| Complex tree (10 nodes) | 51.5 ns |
| Complex tree (20 nodes) | 66.5 ns |

### Batch Evaluation Comparison (1M points)

| Mode | Throughput | ns/point | Feature |
|------|------------|----------|---------|
| CPU JIT SIMD | 977 M/s | 1.0 ns | `--features jit` |
| CPU Scalar | 307 M/s | 3.3 ns | default |
| CPU SIMD (VM) | 252 M/s | 4.0 ns | default |
| GPU Compute | 101 M/s | 9.9 ns | `--features gpu` |

### JIT Compilation

The JIT compiler generates native SIMD machine code using Cranelift, achieving the highest throughput. The original 15 primitives are fully supported in both JIT scalar and JIT SIMD (8-wide) backends. The 16 new IQ primitives are supported in interpreted, compiled VM, and shader transpiler modes.

**Deep Fried v2 optimizations:**
- **Division Exorcism** - all runtime divisions pre-computed as reciprocal multiplications at compile time
- **Branchless SIMD selection** - sign-bit extraction via `bitcast`/`sshr`/`bitselect` (zero-overhead on SSE/AVX/NEON)
- **FMA fusion** - fused multiply-add for reduced latency in complex primitives (Cone, RoundedCone, Pyramid)

### crispy.rs — Hardware-Native Math Utilities

Low-level branchless operations for hot inner loops. Trades sub-ULP precision for throughput.

| Function | Description |
|----------|-------------|
| `fast_recip(x)` | Fast `1/x` via hardware rcpss + Newton-Raphson (~0.02% error) |
| `fast_inv_sqrt(x)` | Quake III inverse sqrt + NR iteration (~0.175% error) |
| `fast_normalize_2d(gx, gz)` | Normalize 2D gradient using fast inverse sqrt |
| `select_f32(cond, a, b)` | Branchless cmov via bit manipulation |
| `branchless_min/max/clamp/abs` | Zero-branch arithmetic via `select_f32` |
| `BitMask64` | 64-element batch mask (AND/OR/NOT/popcnt via hardware) |
| `BloomFilter` | 4KB Bloom filter with FNV-1a double-hashing, O(1) membership test |
| `fnv1a_hash(data)` | FNV-1a 64-bit hash (fast, well-distributed) |

### Compiled Marching Cubes Performance

The compiled MC path (`sdf_to_mesh_compiled`) applies two key optimizations:

1. **SIMD Batch Grid Evaluation** — All grid points are collected into a `Vec<Vec3>`, then evaluated in one call to `eval_compiled_batch_simd_parallel` (8-wide SIMD + Rayon). Eliminates per-point function call overhead.

2. **Grid Finite-Difference Normals** — Vertex normals are computed from neighboring grid values (`values[x+1] - values[x-1]`) instead of 6 additional `eval_compiled` calls per vertex. Interior cells use zero eval calls for normals; boundary cells fall back to analytical normals.

```bash
# Run CLI benchmark
cargo run --features "cli,jit" --release -- bench --points 1000000
```

| Points | JIT SIMD | SIMD (VM) | Speedup |
|--------|----------|-----------|---------|
| 100K | 330 M/s | 197 M/s | 1.7x |
| 1M | 977 M/s | 252 M/s | 3.9x |

### SIMD 8-wide Evaluation

| Mode | Time (8 points) | Speedup |
|------|-----------------|---------|
| Scalar | 563 ns | 1.0x |
| SIMD | 143 ns | 3.9x |

### Marching Cubes (Sphere, bounds +/-2.0)

| Resolution | Time |
|------------|------|
| 16^3 | 140 us |
| 32^3 | 390 us |
| 64^3 | 1.64 ms |

### Raymarching

| Shape | Time per ray |
|-------|--------------|
| Sphere | 62 ns |
| Complex (smooth union + twist) | 178 ns |

### GPU Compute (WebGPU)

The GPU module provides WebGPU compute shaders with persistent buffer pooling for repeated evaluations.

**When to use GPU vs CPU:**

| Batch Size | Recommended | Reason |
|------------|-------------|--------|
| < 100K | CPU JIT SIMD | GPU transfer overhead dominates |
| 100K - 1M | CPU JIT SIMD | JIT still faster on M3 Max |
| > 1M | Test both | Depends on hardware and shape complexity |

Note: GPU performance varies significantly by hardware. On discrete GPUs, crossover point may be lower.

```rust
use alice_sdf::prelude::*;
use alice_sdf::compiled::{GpuEvaluator, WgslShader, GpuBufferPool};

let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);

// Create GPU evaluator (compiles SDF to WGSL)
let gpu = GpuEvaluator::new(&shape).unwrap();

// Single-shot batch evaluation
let points: Vec<Vec3> = (0..100000)
    .map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0))
    .collect();
let distances = gpu.eval_batch(&points).unwrap();

// Persistent buffer pool for repeated evaluations (2-5x faster)
let mut pool = gpu.create_buffer_pool(100000);
for frame in 0..60 {
    let distances = gpu.eval_batch_pooled(&points, &mut pool).unwrap();
}

// Auto-tuned batch (splits large batches into optimal chunks)
let distances = gpu.eval_batch_auto(&points, &mut pool).unwrap();
```

Enable with: `cargo build --features gpu`

| Mode | Throughput (1M pts) |
|------|---------------------|
| CPU JIT SIMD | ~977 M/s |
| CPU SIMD (VM) | ~252 M/s |
| GPU Compute | ~101 M/s |

## WebAssembly (Browser)

ALICE-SDF runs in the browser via WebAssembly with WebGPU/Canvas2D support.

### npm Package (`@alice-sdf/wasm`)

```bash
npm install @alice-sdf/wasm
```

Full TypeScript type definitions included. Supports all 72 primitives, 24 CSG operations, transforms, mesh conversion, and shader generation (WGSL/GLSL).

### Building the WASM Demo

```bash
cd examples/wasm-demo
wasm-pack build --target web
python3 -m http.server 8080
# Open http://localhost:8080
```

### Features

- **WebGPU Compute**: Hardware-accelerated evaluation (Chrome 113+, Edge 113+)
- **Canvas2D Fallback**: CPU raymarching for older browsers
- **Real-time Visualization**: Interactive shape editing and rendering

### Browser Compatibility

| Browser | WebGPU | Canvas2D |
|---------|--------|----------|
| Chrome 113+ | yes | yes |
| Edge 113+ | yes | yes |
| Firefox Nightly | yes (flag) | yes |
| Safari 18+ | yes | yes |
| Older browsers | no | yes |

## Benchmarking

Run benchmarks to compare evaluation modes:

```bash
# CPU benchmarks (interpreter, SIMD, BVH)
cargo bench --bench sdf_eval

# JIT + SoA throughput benchmarks
cargo bench --bench sdf_eval --features jit -- soa_throughput

# GPU vs CPU comparison
cargo bench --bench gpu_vs_cpu --features "jit,gpu"

# CLI quick benchmark
cargo run --features "cli,jit,gpu" --release -- bench --points 1000000

# View HTML report
open target/criterion/report/index.html
```

## Unity Demo: SDF Universe

The `unity-sdf-universe/` directory contains a full Unity demo showcasing ALICE-SDF capabilities:

**"5MB Procedural Universe"** - An entire procedural universe using only 5MB of code.

### Features

| Feature | Description |
|---------|-------------|
| **10M+ Particles** | GPU Compute Shader particle system at 60+ FPS |
| **4 Scene Types** | Cosmic, Terrain, Abstract, Fractal |
| **Infinite Resolution** | Raymarching + Procedural Texturing |
| **The Fractal Dive** | Microscope demo with x10,000+ zoom |

### Quick Start

```bash
# 1. Build Rust library
cargo build --release

# 2. Copy to Unity
cp target/release/libalice_sdf.dylib unity-sdf-universe/Assets/Plugins/  # macOS

# 3. Open in Unity 2022.3+
# 4. Open Assets/Scenes/SdfUniverse.unity
# 5. Press Play
```

### The Fractal Dive (Microscope Demo)

Demonstrates TRUE infinite resolution via raymarching:

- **SDF Formula**: `Subtract(Box, Repeat(Cross))` - single mathematical object
- **Raymarching**: Per-pixel SDF evaluation (128 steps)
- **Procedural Texturing**: Colors from FBM noise (never pixelates)
- **[R] key**: Toggle between Raymarching and Particles mode

See `unity-sdf-universe/README.md` for full documentation.

## VRChat Integration

The `vrchat-package/` directory provides a VRChat SDK package for SDF-based worlds and avatars.

- **ALICE-Shader** - HLSL raymarching kernel with dynamic LOD
- **ALICE-Udon** - UdonSharp SDF collider with pure C# math
- **ALICE-Baker v0.3** - Editor tool to generate optimized shader + Udon from `.asdf.json`
- **7 Sample Worlds** - Basic, Cosmic, Fractal, Mix, DeformableWall, Mochi, TerrainSculpt

See `vrchat-package/README.md` for full documentation.

## Unreal Engine 5 Integration

ALICE-SDF provides full UE5 support via HLSL transpiler and C FFI bindings.

```bash
# Build the plugin DLL
cargo build --release --features unreal
```

- **HLSL Transpiler** - Generate Custom Material Expression nodes
- **C++ FFI** - Native plugin with `alice_sdf.h` header
- **Blueprint-ready** - UFunction wrappers for visual scripting

See `docs/UNREAL_ENGINE.md` for detailed setup instructions.

## Godot Integration

ALICE-SDF works with Godot via glTF 2.0 import and GDExtension FFI.

- **glTF Pipeline** - Export `.glb` and import directly into Godot
- **GDNative/GDExtension** - Link `libalice_sdf` via C FFI
- **Visual Shader** - Use GLSL transpiler output in shader nodes

See `docs/GODOT_GUIDE.md` for integration guide.

## Cross-Crate Bridges

### Cache Bridge (feature: `sdf-cache`)

SDF evaluation result caching via [ALICE-Cache](../ALICE-Cache). Caches distance field evaluations keyed by quantised grid positions to avoid redundant computation during interactive editing and mesh generation.

```toml
[dependencies]
alice-sdf = { path = "../ALICE-SDF", features = ["sdf-cache"] }
```

```rust
use alice_sdf::cache_bridge::SdfEvalCache;

let cache = SdfEvalCache::new(1_000_000, 0.01); // 1M entries, 0.01 cell size
cache.put(1.0, 2.0, 3.0, 0.42);
if let Some(dist) = cache.get(1.0, 2.0, 3.0) {
    // Cache hit — skip evaluation
}
```

## Documentation

| Document | Description |
|----------|-------------|
| [ALICE-View](../ALICE-View) | Real-time GPU raymarching viewer (wgpu, drag & drop) |
| [QUICKSTART](docs/QUICKSTART.md) | 5-minute getting started guide for all platforms |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | 13-layer architecture deep dive |
| [API Reference](docs/API_REFERENCE.md) | Complete API reference |
| [Unreal Engine](docs/UNREAL_ENGINE.md) | UE5 setup and integration guide |
| [Python Guide](docs/PYTHON_GUIDE.md) | Python and Blender integration |
| [WASM Guide](docs/WASM_GUIDE.md) | WebAssembly deployment guide |
| [Godot Guide](docs/GODOT_GUIDE.md) | Godot integration guide |
| [Unity Setup](unity-sdf-universe/SETUP_GUIDE.md) | Unity project setup |
| [VRChat Package](vrchat-package/README.md) | VRChat SDK integration |

## SDF Asset Delivery Network

ALICE-SDF + [ALICE-CDN](https://github.com/ext-sakamoro/ALICE-CDN) + [ALICE-Cache](https://github.com/ext-sakamoro/ALICE-Cache) to achieve **200-800x bandwidth reduction** vs traditional glTF delivery.

> "Why transfer 2MB of triangles when 80 bytes of math will do?"

### Architecture

```
Client Request (asset_id + VivaldiCoord)
    │
    ▼
┌──────────────────────────────────────┐
│  ALICE-CDN (Vivaldi Routing)          │
│  ・VivaldiCoord → nearest node (RTT)  │
│  ・IndexedLocator: O(log n + k)       │
│  ・Rendezvous hash + distance weight  │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  ALICE-Cache (Markov Prefetch)        │
│  ・256-shard parallel cache           │
│  ・SharedOracle: lock-free prediction │
│  ・TinyLFU sampled eviction           │
└──────────┬───────────────────────────┘
           │ cache miss → origin
           ▼
┌──────────────────────────────────────┐
│  ALICE-SDF (ASDF Binary Format)       │
│  ・16-byte header + bincode body      │
│  ・CRC32 integrity validation         │
│  ・68+ SDF primitives + CSG ops       │
└──────────────────────────────────────┘
```

### Compression Ratio

| Asset Type | glTF Size | SDF Size | Ratio |
|------------|-----------|----------|-------|
| Sphere | 15-25 KB | ~80 bytes | **200-300x** |
| Box | 15-20 KB | ~90 bytes | **170-220x** |
| CSG (10 ops) | 200-500 KB | ~500 bytes | **400-1000x** |
| Complex scene (100 nodes) | 2-4 MB | 2-4 KB | **500-1000x** |

### Use Cases

1. **Game Level Streaming**: Stream SDF game worlds zone-by-zone. Markov oracle prefetches the next zone while the current one renders.
2. **Procedural Content**: Transmit CSG recipes instead of baked meshes. Client-side SDF evaluation generates geometry at any LOD.
3. **Collaborative 3D Editing**: Real-time sync of SDF edits (add/remove CSG nodes) at minimal bandwidth.
4. **IoT/Edge 3D**: Deliver 3D content to bandwidth-constrained devices (80 bytes vs 20 KB per object).
5. **CDN Cost Reduction**: 200-800x less data transferred = proportional CDN cost savings.

## SDF × Nanite Hybrid Pipeline

ALICE-SDF generates UE5 Nanite-compatible hierarchical cluster meshes directly from SDF mathematical descriptions. This hybrid pipeline combines SDF compactness with Nanite rendering performance — a fundamentally different workflow from conventional polygon mesh delivery.

```
┌────────────────────┐     ┌─────────────────────────────┐     ┌──────────────┐
│  ALICE-SDF (.asdf) │     │  ALICE Engine (load time)    │     │  UE5 Nanite  │
│  Formula = ~KB     │ ──→ │  generate_nanite_mesh()      │ ──→ │  GPU render  │
│  Storage/Transfer  │     │  Hierarchical cluster + DAG  │     │  100M+ poly  │
└────────────────────┘     └─────────────────────────────┘     └──────────────┘
```

| Process | Technology | Benefit |
|---------|-----------|---------|
| Storage & Transfer | ALICE-SDF | Data is just a few KB (mathematical formula). Downloads are instant. |
| Load Time | ALICE Engine | Generates Nanite mesh from the formula at high speed. |
| Rendering | UE5 Nanite | Renders hundreds of millions of polygons smoothly without GPU bottleneck. |

**Transfer at SDF compactness, render at polygon speed.** There is no need to run raymarching (expensive per-pixel computation) every frame — the GPU workload is the same as a conventional polygon game.

### Comparison with Conventional Nanite Workflow

| | Conventional Nanite | SDF × Nanite (ALICE) |
|---|---|---|
| Asset authoring | ZBrush / Photogrammetry scan | Mathematical formula (CSG, noise, procedural) |
| Disk space | 100 MB - 1 GB per asset | 80 bytes - 4 KB per asset |
| Network transfer | Requires large download | Instant (formula only) |
| Runtime modification | Not possible (static mesh) | Re-generate from modified SDF |
| LOD | Baked into Nanite clusters | Generated per LOD level from SDF |
| Variety | Each variation = separate asset | Parameterized — infinite variations from one formula |

### What Makes This Different

Conventionally, using Nanite requires GB-scale pre-authored mesh data (e.g., ZBrush scans) stored on disk. With ALICE-SDF:

- **Disk savings**: A massive rock formation is stored as "sphere + noise formula" — a few KB instead of hundreds of MB.
- **Procedural generation × Nanite**: Generate infinite cave systems, terrain, or organic structures mathematically, then materialize them as high-detail Nanite meshes on the fly.
- **Runtime destruction**: Because ALICE-SDF mesh generation is fast, a wall destroyed by a spell can be **re-generated as a new Nanite mesh with the destroyed shape** in real time. This enables freeform destruction that is fundamentally different from the traditional "swap in a pre-made destroyed mesh" approach.

### Usage

```rust
use alice_sdf::prelude::*;
use alice_sdf::mesh::{generate_nanite_mesh, NaniteConfig};

let shape = SdfNode::sphere(1.0)
    .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);

let nanite = generate_nanite_mesh(
    &shape,
    Vec3::splat(-2.0),
    Vec3::splat(2.0),
    &NaniteConfig::high_detail(),
);

alice_sdf::io::nanite::export_nanite(&nanite, "MyAsset.nanite")?;
```

See `docs/UNREAL_ENGINE.md` — "Nanite Integration" section for full setup instructions.

### Asset Creation: Natural Language to Nanite Mesh

ALICE-SDF includes a Text-to-3D Pipeline (FastAPI server) that eliminates the need for humans to write SDF formulas. Combined with the Nanite pipeline, the full workflow becomes:

```
"A weathered stone tower"      LLM thinks            ALICE-SDF             UE5
  (natural language)   ──→   SDF formula    ──→   Nanite mesh    ──→   GPU render
  User types/speaks          Claude/Gemini         <55ms               100M+ poly
```

Users (artists, level designers) do not need to know SDF operations (Union, Intersection, SmoothMin...) at all. They describe what they want in plain language, and the pipeline produces a Nanite-ready mesh.

#### Error Self-Repair

LLM-generated code is not always perfect. The server handles this automatically:

- **JSON repair**: Automatic brace/bracket completion for truncated LLM output
- **Retry with feedback**: Error messages are fed back to the LLM for up to 2 correction attempts — only valid, working assets are produced

#### Runtime On-Demand Generation

The server runs as a FastAPI web service. The SDF-to-mesh step takes <55ms (excluding LLM inference), enabling game clients to request new geometry during gameplay. This is on-demand generation, not pre-baked assets.

#### Complete Pipeline: No Blind Spots

| Concern | Solution | Technology |
|---------|----------|-----------|
| Rendering | Nanite integration — polygon-speed GPU rendering | `generate_nanite_mesh()` → UE5 Nanite |
| Physics | Deterministic fixed-point collision | ALICE-Physics (Fix128) |
| Asset creation | Natural language — no SDF expertise needed | Text-to-3D server (Claude/Gemini) |
| Delivery | Instant transfer — formulas, not polygons | ALICE-CDN + ALICE-Cache |

## v1.1.0 Release — Stable

### Quality Metrics

| Metric | Value |
|--------|-------|
| Unit tests | 1003 passed, 0 failed |
| Clippy pedantic | 0 warnings |
| Doc warnings | 0 warnings |
| TODO / FIXME | 0 |
| `unimplemented!()` / `todo!()` | 0 |
| SIMD placeholder operations | 0 (all 7 fully implemented) |
| Formatting | `cargo fmt --check` clean |

### Pre-built Binaries

Download from [GitHub Releases](https://github.com/ext-sakamoro/ALICE-SDF/releases/tag/v1.1.0):

| Asset | Platform | Engine |
|-------|----------|--------|
| `AliceSDF-UE5-Plugin-macOS.zip` | macOS ARM64 | Unreal Engine 5 |
| `AliceSDF-UE5-Plugin-macOS-Intel.zip` | macOS x86_64 | Unreal Engine 5 |
| `AliceSDF-UE5-Plugin-Win64.zip` | Windows x86_64 | Unreal Engine 5 |
| `AliceSDF-UE5-Plugin-Linux.zip` | Linux x86_64 | Unreal Engine 5 |
| `AliceSDF-Unity-Plugin-macOS.zip` | macOS ARM64 | Unity |
| `AliceSDF-Unity-Plugin-macOS-Intel.zip` | macOS x86_64 | Unity |
| `AliceSDF-Unity-Plugin-Win64.zip` | Windows x86_64 | Unity |
| `AliceSDF-Unity-Plugin-Linux.zip` | Linux x86_64 | Unity |
| `AliceSDF-VRChat-Package.zip` | Cross-platform | VRChat |

### v1.1.0 Key Changes

- **Auxiliary data buffer**: `Instruction` now carries `aux_offset`/`aux_len` pointing into `CompiledSdf.aux_data`, enabling variable-length data (matrices, control points, heightmaps) for complex operations
- **7 new compiled operations**: ProjectiveTransform, LatticeDeform, SdfSkinning, IcosahedralSymmetry, IFS, HeightmapDisplacement, SurfaceRoughness — fully implemented in both scalar and SIMD evaluators
- **126 total SdfNode variants**: 72 primitives, 24 operations, 7 transforms, 23 modifiers
- **220 pedantic clippy warnings fixed**: raw string hashes, implicit clone, useless format, dead code, etc.
- **5 roundtrip tests**: each new operation verified against tree-walker for correctness

## License

**Open Core Model** - Free for creators, licensed for infrastructure.

| Component | License | Use Case |
|-----------|---------|----------|
| **Core Engine** (Rust) | MIT License | Hack away! |
| **Unity Integration** | ALICE Community License | Free for Indie & Game Dev |
| **Enterprise / Cloud Infra** | Commercial License | Contact for pricing |

### Free Use (No License Required)

- Personal projects
- Indie game development (any revenue)
- AAA game studios (shipped games)
- Educational & research
- Open source projects

### Commercial License Required

- Metaverse platforms (10,000+ MAU)
- Cloud streaming services (SaaS/PaaS)
- Infrastructure providers
- Competing products

See [LICENSE](LICENSE) (MIT) and [LICENSE-COMMUNITY](LICENSE-COMMUNITY) for details.

**Content you create (.asdf files, worlds, games) is 100% yours. No royalties.**

## Related Projects

| Project | Description | Link |
|---------|-------------|------|
| **Open Source SDF Assets** | 991 free CC0 3D assets in .asdf.json format, converted via ALICE-SDF | [GitHub](https://github.com/ext-sakamoro/Open-Source-SDF-Assets) |
| **ALICE Ecosystem** | 52-component edge-to-cloud data pipeline | [GitHub](https://github.com/ext-sakamoro/ALICE-Eco-System) |
| **AI Modeler SaaS** | Browser-based 3D modeling powered by ALICE-SDF | Coming soon |

---

Copyright (c) 2025-2026 Moroya Sakamoto
