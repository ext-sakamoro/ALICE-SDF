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
- **PBR materials** - metallic-roughness workflow compatible with UE5/UE6/Unity/Godot
- **Keyframe animation** - parametric deformation with timeline tracks
- **Asset pipeline** - OBJ import/export, glTF 2.0 (.glb) export, FBX, USD, Alembic, Nanite, STL, PLY, 3MF, ABM export
- **5-layer mesh persistence** - ABM binary format, LOD chain persistence, chunked mesh cache with FIFO eviction, Unity/UE5/UE6 native export
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
- **Engine integrations** - Unity, Unreal Engine 5 / 6, VRChat, Godot, WebAssembly

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

### With ALICE-LOL DSL (Recommended)

The easiest way to create SDF scenes is [ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL) — a `lol!` proc_macro that lets you write SDF trees declaratively instead of constructing them by hand.

```toml
# Cargo.toml
[dependencies]
alice-sdf = { path = "../ALICE-SDF" }
alice-lol = { path = "../ALICE-LOL/alice-lol" }
```

**Before (manual SdfNode construction):**

```rust
use alice_sdf::prelude::*;

let scene = SdfNode::SmoothUnion {
    k: 0.3,
    children: vec![
        SdfNode::sphere(1.0),
        SdfNode::Translate {
            offset: glam::Vec3::new(2.0, 0.0, 0.0),
            child: Box::new(SdfNode::Round {
                radius: 0.05,
                child: Box::new(SdfNode::box3d(0.8, 0.8, 0.8)),
            }),
        },
    ],
};
```

**After (LOL DSL):**

```rust
use alice_lol::{lol, to_glsl, eval};

let scene = lol! {
    smooth_union(0.3,
        sphere(1.0),
        translate(2.0, 0.0, 0.0, round(0.05, box3d(0.8, 0.8, 0.8)))
    )
};
```

Same `SdfNode` tree, fraction of the code. All 76 constructs (27 primitives, 23 CSG ops, 4 transforms, 19 modifiers, 2 time controls, 3 law constraints) work as function calls.

**Transpile to GPU shaders:**

```rust
let glsl = to_glsl(&scene);                      // GLSL
let wgsl = alice_lol::to_wgsl(&scene);            // WGSL (WebGPU)
let hlsl = alice_lol::to_hlsl(&scene);            // HLSL (DirectX)
```

**CPU evaluation:**

```rust
let dist = eval(&scene, glam::Vec3::new(0.0, 1.0, 0.0));
```

**Inject Rust variables at runtime:**

```rust
let radius = 1.5_f32;
let height = compute_height();
let scene = lol! {
    smooth_union(0.2,
        sphere({radius}),
        translate(0.0, {height}, 0.0, cylinder(2.0, 0.5))
    )
};
```

**Validate shape constraints at compile time:**

```rust
use alice_lol::law::{LawSet, Law, Priority};

let laws = LawSet::new()
    .add(Law::non_overlap(&a, &b), Priority::Hard)        // shapes must not intersect
    .add(Law::min_thickness(&scene, 0.1), Priority::Soft(0.5));  // wall thickness >= 0.1
let report = laws.check();
```

For full LOL documentation, see [ALICE-LOL README](https://github.com/ext-sakamoro/ALICE-LOL).

---

### Rust (Direct SdfNode Construction)

For fine-grained control or when you need access to advanced node types not yet covered by the LOL DSL, you can construct `SdfNode` trees directly:

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


For deep technical sections (Material / Animation / Architecture / Mesh / Platonic Solids / Interval Arithmetic / Neural SDF / Collision / Analytic Gradient / Dual Contouring / CSG Tree Optimization / Auto Tight AABB / Texture Fitting / Raymarching / FFI bindings / Feature Flags / Physics Bridge / 3D Print Pipeline / Performance / Benchmarking / Unity / VRChat / Unreal / Godot / Cross-Crate Bridges / Asset Delivery Network / Nanite hybrid pipeline) see [`docs/USAGE.md`](docs/USAGE.md).

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

## LLM × 3D Creation Pipeline (SDF + LOL + View + Physics)

Four ALICE projects combine to form an end-to-end **text-to-3D** workflow — from natural language to physically simulated 3D scenes:

```
User: "A snowman with a top hat"
         │
         ▼
┌──────────────────┐  LOL DSL or JSON  ┌───────────────────┐  WGSL / GLB   ┌──────────────┐
│  LLM             │ ────────────────▶ │  ALICE-SDF        │ ────────────▶ │  ALICE-View  │
│  (Claude/Gemini) │                   │  parse → compile  │               │  GPU Preview │
│                  │                   │  → mesh / shader  │               │  60 FPS      │
└──────────────────┘                   └────────┬──────────┘               └──────────────┘
                                                │
                                                │ SdfField trait
                                                │ (feature = "physics")
                                                ▼
                                       ┌───────────────────┐
                                       │  ALICE-Physics     │
                                       │  Fix128 XPBD       │
                                       │  SDF CCD / Forces  │
                                       │  Destruction / Fluid│
                                       └───────────────────┘
```

| Component | Role |
|-----------|------|
| **[ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL)** | LLM-friendly DSL — fewer tokens, lower hallucination rate than raw JSON. `runtime_parser::parse_lol()` converts LLM text output into `SdfNode` at runtime |
| **ALICE-SDF** | Core engine — SIMD/BVH/JIT evaluation, mesh generation (Marching Cubes / Dual Contouring), GLSL/WGSL/HLSL transpilation, GLB/OBJ/STL export |
| **[ALICE-View](https://github.com/ext-sakamoro/ALICE-View)** | Real-time GPU raymarching viewer — drag & drop JSON/ASDF files, instant visual feedback |
| **[ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics)** | Deterministic 128-bit fixed-point physics — SDF shapes become collision geometry via `SdfField` trait. SDF CCD, force fields, destruction, cloth, fluid simulation |

LLM-generated shapes are not just visual — they are physics-ready. The `CompiledSdfField` wrapper exposes the SDF as an O(1) collision query surface, enabling rigid body, destruction, and fluid interactions without convex decomposition.

### Quick Start

```bash
# 1. Start the Text-to-3D server (generates LOL/JSON via LLM)
cd ALICE-SDF/server
python main.py

# 2. POST a prompt — server returns SDF JSON
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A snowman with a top hat", "format": "json"}'

# 3. View the result in real-time
cd ALICE-View
cargo run --bin alice-view -- ../ALICE-SDF/server/output/latest.json
```

### Programmatic (Rust)

```rust
use alice_lol::runtime_parser::parse_lol;
use alice_sdf::prelude::*;
use alice_sdf::physics_bridge::CompiledSdfField;

// LLM output (text) → SdfNode
let lol_text = r#"smooth_union(0.3, sphere(1.0), translate(0.0, 1.5, 0.0, sphere(0.7)))"#;
let scene = parse_lol(lol_text).unwrap();

// GPU shader for rendering
let wgsl = alice_lol::to_wgsl(&scene);

// Export mesh
let mesh = alice_sdf::mesh::sdf_to_mesh(
    &scene,
    glam::Vec3::splat(-3.0),
    glam::Vec3::splat(3.0),
    &MeshConfig::default(),
);

// Physics-ready collision shape (no convex decomposition needed)
let field = CompiledSdfField::new(scene);
// field.distance(x, y, z)            → f32        (1 eval)
// field.distance_and_normal(x, y, z) → (f32, Vec3) (4 evals, tetrahedral)
```

### Why LOL over JSON?

| Metric | JSON (SdfNode) | LOL DSL |
|--------|---------------|---------|
| Tokens per shape | ~120 | ~30 |
| LLM error rate | higher (bracket nesting) | lower (function-call style) |
| Runtime parsing | `serde_json` | `runtime_parser::parse_lol()` |
| Compile-time macro | — | `lol! { ... }` |

For complex scenes, LOL typically uses **3-4x fewer tokens**, which reduces both LLM cost and hallucination.

---

## Mobile (iOS / Android)

ALICE-SDF ships a [UniFFI](https://mozilla.github.io/uniffi-rs/)-based mobile SDK so the Rust core can be called directly from **Swift** (iOS) and **Kotlin** (Android).

### Supported targets

| Platform | Architectures | Distribution |
|----------|---------------|--------------|
| **iOS** | `aarch64-apple-ios` (device), `aarch64-apple-ios-sim`, `x86_64-apple-ios` | `AliceSDF.xcframework` (static lib + Swift bindings) |
| **Android** | `arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86` | `libuniffi_alice_sdf.so` + Kotlin bindings |

### Verified on real devices (2026-06-06)

| Platform | Hardware | Result |
|----------|----------|--------|
| iOS | iPhone 17 Pro Simulator (iOS 26.0, Xcode 26.2) | ✅ Demo app boots, renders 2D SDF slice ([screenshot](mobile/samples/ios-swiftui/screenshots/AliceSDF-demo.png)) |
| Android | Pixel 6 emulator (Android 14 / API 34, arm64-v8a) | ✅ Demo app boots, renders 2D SDF slice ([screenshot](mobile/samples/android-compose/screenshots/AliceSDF-android-demo.png)) |

Both platforms produced identical numerical output (`sphere d = 0.2806`, `smooth_union(k=0.3) = 0.2056`) — proving binary parity of the Rust core across Apple Silicon and Android ARM.

### Quick start (Swift)

```swift
import AliceSDF

let d = sdfSphere(
    point:  Vec3(x: 1, y: 0, z: 0),
    center: Vec3(x: 0, y: 0, z: 0),
    radius: 1.0
)
// d ≈ 0 (point on sphere surface)

let blended = opSmoothUnion(a: 0.5, b: 0.6, k: 0.1)
// blended < 0.5 (smooth union pulls below min)
```

### Quick start (Kotlin)

```kotlin
import uniffi.alice_sdf.*

val d = sdfSphere(
    point  = Vec3(1f, 0f, 0f),
    center = Vec3(0f, 0f, 0f),
    radius = 1.0f
)
// d ≈ 0

val blended = opSmoothUnion(a = 0.5f, b = 0.6f, k = 0.1f)
```

### Build the SDK

```bash
# iOS XCFramework (device + simulator)
cd mobile/packaging/ios && ./build-xcframework.sh

# Android .so + Kotlin bindings (4 ABI)
export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk
cd mobile/packaging/android && ./build-aar.sh
```

Sample apps and the full integration guide live in [`mobile/`](mobile/).

---

## Web (WebAssembly), VFX (OpenVDB), Bevy Engine

### `wasm` feature — WebAssembly bindings

Browser-side SDF evaluation + slice rendering via [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/).

```bash
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
```

JavaScript usage:

```js
import init, { sdf_sphere, op_smooth_union, render_sphere_slice_rgba } from './alice_sdf.js';
await init();
const d = sdf_sphere(1, 0, 0, /*center*/ 0, 0, 0, /*radius*/ 1.0);  // ≈ 0
const rgba = render_sphere_slice_rgba(256, 256, 0, 0, 0, 1.0, 2.5);  // Uint8Array for canvas
```

### `openvdb` feature — OpenVDB Float Grid I/O

Bake SDF trees into voxel grids for DCC tools (Houdini, Maya, Nuke, Blender).

```rust
use alice_sdf::io::vdb::{bake_to_vdb, load_dense_grid_from_vdb};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
let bytes = bake_to_vdb(&node, (-2.0, 2.0), 64).unwrap();
std::fs::write("sphere.vdb", &bytes).unwrap();
```

Backed by [`vdb-rs`](https://crates.io/crates/vdb-rs) 0.6 (pure Rust). Currently writes a compact custom container (`ALICEVDB1`); full OpenVDB binary parity is planned as `vdb-rs` exposes its write API.

### `alice-sdf-bevy` — Bevy 0.18 plugin

Drop-in plugin that turns `SdfShape` components into renderable `Mesh3d` assets automatically.

```rust
use bevy::prelude::*;
use alice_sdf_bevy::{AliceSdfPlugin, SdfShape};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(AliceSdfPlugin)
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(SdfShape::Sphere { radius: 1.0 });
        })
        .run();
}
```

See `bindings/bevy/alice-sdf-bevy/examples/sphere_demo.rs` for a full 3-shape demo with camera + light.

### 3D Gaussian Splatting (`.splat`)

Convert SDF surfaces into Inria 3DGS-compatible `.splat` files (32 bytes/splat: position + scale + RGBA + compressed quaternion). Drop-and-drop into any WebGL viewer (gsplat.tech / SuperSplat / antimatter15/splat).

```rust
use alice_sdf::io::splat::{sdf_to_splats, save_splat, SplatConfig};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
let cfg = SplatConfig { bounds: (-2.0, 2.0), resolution: 64, base_color: [220, 220, 240, 255] };
let splats = sdf_to_splats(&node, &cfg);
save_splat("sphere.splat", &splats).unwrap();
```

### Blender Add-on (`bindings/blender/`)

Blender 4.0+ add-on. Imports `.asdf` directly and adds an "ALICE-SDF" N-panel with sphere/box/torus generators. Requires the `alice_sdf` Python module (built via `cargo build --release --features python`).

Install: zip the `alice_sdf_blender/` folder and load via `Edit > Preferences > Add-ons > Install...`.

### Houdini Python Plugin (`bindings/houdini/`)

SideFX Houdini 20+ Python module + Python SOP code for `.asdf` loading and primitive generation. Auto-installer detects `$HSITE` / `$HOUDINI_USER_PREF_DIR`.

```python
import alice_sdf_hou
sdf = alice_sdf_hou.sphere(1.0)
alice_sdf_hou.sdf_to_hou_geo(sdf, hou.pwd().geometry(), bounds=(-2.0, 2.0), resolution=64)
```

### MagicaVoxel `.vox` IO

Voxelize an SDF tree and save as a MagicaVoxel-compatible `.vox` file (v150 RIFF) — ideal for indie / voxel art pipelines.

```rust
use alice_sdf::io::vox::{sdf_to_vox, save_vox, VoxConfig};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
let cfg = VoxConfig { size: 64, bounds: (-1.5, 1.5), color_index: 79 };
save_vox("sphere.vox", &sdf_to_vox(&node, &cfg)).unwrap();
```

### `@alice-sdf/threejs` — Three.js / React Three Fiber wrapper

TypeScript npm package built on the `wasm` feature. Exposes `AliceSDF` class, Three.js `DataTexture` helper, optional `<AliceSDFSlicePlane>` for `@react-three/fiber`, and WebXR raymarching helpers.

```ts
import { AliceSDF } from "@alice-sdf/threejs";
const sdf = await AliceSDF.load("/alice_sdf.js");
const tex = await sdf.createSliceTexture(512, 512, [0, 0, 0], 1.0, 2.5);
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: tex })));
```

### Maya Python Plugin (`bindings/maya/`)

Autodesk Maya 2024+ Python module — adds "ALICE-SDF" top-level menu, builds polygon meshes through `MFnMesh`.

```python
import alice_sdf_maya
alice_sdf_maya.register_menu()
alice_sdf_maya.add_sphere(radius=1.5, resolution=64)
```

### Nuke Python Plugin (`bindings/nuke/`)

Foundry Nuke 15 / 16+ Python module — exports `.asdf` to volume binary and 2D RGBA slices for compositing pipelines.

```python
import alice_sdf_nuke
alice_sdf_nuke.export_asdf_as_volume("/path/to/model.asdf", out_path="/tmp/model.alicevdb")
```

### Cinema 4D Python Plugin (`bindings/cinema4d/`)

Maxon Cinema 4D 2024 / 2025 / 2026+ Python module — generates `PolygonObject` from SDF primitives and imports `.asdf` directly into the C4D scene.

```python
import alice_sdf_c4d
alice_sdf_c4d.add_sphere(radius=100.0, resolution=64)   # C4D units: cm
alice_sdf_c4d.import_asdf("/path/to/model.asdf", bounds=(-300.0, 300.0), resolution=128)
```

### CAD Interchange — STEP / IGES (FEM-style mesh export)

ALICE-SDF can tessellate any SDF tree (Marching Cubes) and emit either:

- **STEP AP203** (ISO 10303-21 ASCII) — `CARTESIAN_POINT` + `POLY_LOOP` + `FACE_OUTER_BOUND` triangles. *Note:* this is a mesh-style representation, not a full `ADVANCED_FACE` / `MANIFOLD_SOLID_BREP` BREP. Tools that accept faceted STEP (FreeCAD, some Rhino plugins, mesh-aware CAD viewers) can open it; tools that strictly require `MANIFOLD_SOLID_BREP` may reject it. Full BREP wrapping is planned (see `docs/PUBLISH.md`).
- **IGES** (Entity 134 Node + Entity 136 Finite Element) — FEM mesh entities. Intended for FEM solvers and viewers that understand these entity types. Standard CAD modeling tools usually expect Entity 144 (Trimmed Surface), which is not yet emitted.

Both files round-trip in the unit tests but real-world CAD interoperability is best verified per-tool.

```rust
use alice_sdf::io::step::{export_step, StepConfig};
use alice_sdf::io::iges::{export_iges, IgesConfig};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
export_step("sphere.step", &node, &StepConfig::default()).unwrap();
export_iges("sphere.igs",  &node, &IgesConfig::default()).unwrap();
```

### `alice-sdf-openxr` — Native VR / AR Helpers (`bindings/openxr/`)

A small Rust crate on top of the [`openxr`](https://crates.io/crates/openxr) bindings that turns ALICE-SDF distance queries into VR-friendly helpers. Works with Meta Quest standalone (Android APK), PC VR (SteamVR / Oculus PC), Microsoft Mixed Reality, and Apple Vision Pro via its OpenXR backend.

```rust
use alice_sdf_openxr::{XrPose, raymarch_sphere};
use glam::Vec3;

// inside an XRFrame callback
let head_pose: XrPose = openxr_pose.into();
let hit_dist = raymarch_sphere(head_pose, Vec3::new(0.0, 1.5, -1.0), 0.3, 5.0);
if hit_dist > 0.0 {
    // controller / head is looking at the sphere
}
```

### `AliceSDFVisionOS` — Apple Vision Pro SwiftPM Package (`mobile/swift-package-visionos/`)

A visionOS / RealityKit-flavoured Swift Package that wraps the same `AliceSDF.xcframework` used by the iOS / iPadOS / macOS targets and adds factory helpers for `ModelEntity`.

```swift
import SwiftUI
import RealityKit
import AliceSDFVisionOS

struct ImmersiveView: View {
    var body: some View {
        RealityView { content in
            let sphere = AliceSDFRealityKit.makeSphereEntity(radius: 0.1)
            sphere.position = SIMD3(0, 1.5, -1.0)
            content.add(sphere)
        }
    }
}
```

### REST API Server (`server/`)

An `axum` + `tokio` HTTP server that exposes ALICE-SDF primitive evaluation and operations over JSON — intended as the backend for cloud-served SDF UIs (e.g. `alicelaw.net/sdf-metaverse`).

```bash
cd server && cargo run --release
# → ALICE-SDF server listening on http://0.0.0.0:8787
```

```http
POST /eval
Content-Type: application/json

{ "shape": "sphere", "point": [1, 0, 0], "params": { "radius": 1.0, "center": [0, 0, 0] } }
```

Response: `{ "distance": 0.0 }`

---

## Related Projects

| Project | Description | Link |
|---------|-------------|------|
| **ALICE-View** | Real-time GPU raymarching viewer for SDF files (wgpu + WGSL) | [GitHub](https://github.com/ext-sakamoro/ALICE-View) |
| **Open Source SDF Assets** | 991 free CC0 3D assets in .asdf.json format, converted via ALICE-SDF | [GitHub](https://github.com/ext-sakamoro/Open-Source-SDF-Assets) |
| **ALICE-LOL** | Law-Oriented Language — `lol!` proc_macro DSL for declarative SDF scene authoring (76 constructs, GLSL/WGSL/HLSL transpile, law constraints, spatial pruning) | [GitHub](https://github.com/ext-sakamoro/ALICE-LOL) |
| **ALICE Ecosystem** | 52-component edge-to-cloud data pipeline | [GitHub](https://github.com/ext-sakamoro/ALICE-Eco-System) |
| **AI Modeler SaaS** | Browser-based 3D modeling powered by ALICE-SDF | [GitHub](https://github.com/ext-sakamoro/AI-Modeler-SaaS) |
| **ALICE SDF Metaverse** | Live browser demo — WebGL2 raymarching world with JS-side CCD physics, runs ALICE-SDF concepts in the browser | [Demo](https://alicelaw.net/sdf-metaverse) |
| **alicelaw.net** | Personal site source — Cloudflare Pages + Pages Functions QR router (`/0x01`-`/0xFF`), hosts the SDF Metaverse demo | [GitHub](https://github.com/ext-sakamoro/alicelaw-net) |

---

Copyright (c) 2025-2026 Moroya Sakamoto — https://alicelaw.net/
