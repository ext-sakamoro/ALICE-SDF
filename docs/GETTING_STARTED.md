# Getting started

An introduction to ALICE-SDF: what the crate does, the shape vocabulary, and
the common ways to build, evaluate and export a shape. The [README](../README.md)
has the short version; this page has the long one.

[日本語](GETTING_STARTED_JP.md)

## Feature list
ALICE-SDF is a 3D/spatial data specialist that transmits **mathematical descriptions of shapes** (Signed Distance Functions) instead of polygon meshes. This enables:

- **10-1000x compression** compared to traditional mesh formats
- **Infinite resolution** - shapes are mathematically perfect at any scale
- **CSG operations** - boolean operations on shapes without mesh overhead
- **The metric as a value** (4.0.0) - `MetricBall` takes the norm itself as a parameter (`‖p‖₂ − r` is a sphere, `‖p‖∞ − r` a cube, `‖p‖₁ − r` an octahedron: the same expression, a different norm), with an *exact closed form* for its Lipschitz bound rather than an estimate; `MetricBlend` puts one field inside a bubble and another outside it, leaving the world beyond the skin untouched bit for bit. `measure_tension` reports what a field's gradient actually does over a region — the seam where two laws meet is the one place no static bound can describe
- **Cross-platform bit-exact evaluation** (3.1.0) - every transcendental in the laws and evaluators goes through [`alice-det-math`](https://crates.io/crates/alice-det-math) (the crate `alice-physics` uses) and `a * b + c` is never fused, so the tree, compiled scalar, `f32x8` SIMD, BVH and Cranelift SIMD-JIT evaluators return the *same bits* on x86_64, aarch64 and wasm32 (`tests/test_det_parity.rs`, `tests/test_det_golden.rs`); the GPU shaders stay a tolerance domain except the axis ties of `atan2`, which `alice_atan2` pins to the CPU law
- **Real-time raymarching** - GPU-accelerated rendering
- **PBR materials** - metallic-roughness workflow compatible with UE5/UE6/Unity/Godot
- **Keyframe animation** - parametric deformation with timeline tracks
- **Asset pipeline** - OBJ import/export, glTF 2.0 (.glb) export, FBX, USD, Alembic, Nanite, STL, PLY, 3MF, ABM export
- **5-layer mesh persistence** - ABM binary format, LOD chain persistence, chunked mesh cache with FIFO eviction, Unity/UE5/UE6 native export
- **Manifold mesh guarantee** - validation, repair, and quality metrics
- **Printability validation** (`validity`) - answers "can this be printed" *and states how it was decided*. A global erosion proof over `eval_interval` returns a three-valued `ErosionVerdict` (`HasThickEnoughRegion` / `EntirelyTooThin` / `Undecided`, where `Undecided` is **not** a pass), per-triangle sphere tracing measures exact local wall thickness, and overhang comes from the closed form `asin(-n · b)`. `export_step_validated` refuses to write a file that fails the requirements
- **Adaptive Marching Cubes** - octree-based mesh generation, detail where it matters (outward CCW winding, closed meshes; 1.11.0 flipped the index order — see CHANGELOG)
- **Dual Contouring** - QEF-based mesh generation that preserves sharp edges and corners
- **V-HACD convex decomposition** - automatic convex hull decomposition for physics
- **Attribute-preserving decimation** - QEM with UV/tangent/material boundary protection
- **Advanced simplifier options** - per-vertex lock mask (`lock_vertices`) for LOD seam preservation, absolute/relative error thresholds (`error_absolute`)
- **Decimation-based LOD** - progressive LOD chain from high-res base mesh
- **meshopt-compatible codec** - binary-compatible index / vertex buffer compression (indexcodec v1 + vertexcodec v0/v1), verified against reference vectors produced by the zeux/meshoptimizer C++ library
- **glTF `EXT_meshopt_compression`** - `io::meshopt_gltf` module + `GltfConfig::meshopt_compress` option; POSITION / NORMAL / TEXCOORD_0 / JOINTS_0 / WEIGHTS_0 / indices all compressible with per-attribute channel estimation (u8 / u16 / u32 XOR + 8-rotation heuristic)
- **Vertex filters** - Octahedral (normal / tangent, 50-75% smaller with <1% angular error), Quaternion (rotation, largest-component + cyclic swizzle), Exponential (float mantissa + shared exponent)
- **Triangle stripifier** - Evans-Skiena-Varshney greedy strip algorithm with 8-triangle lookahead (~48% index reduction on closed meshes), supports primitive-restart or degenerate-triangle joining
- **Nanite-style meshlet clusters** - V2 adjacency + `cone_weight` grow with `NormalCone` (basic + `cone_apex`) for Vulkan `VK_EXT_mesh_shader` / DirectX 12 mesh shader culling
- **74 primitives, 25 operations, 7 transforms, 24 modifiers** (130 `SdfNode` variants, counted by `SdfNode::category()`)
- **Chamfer & Stairs blends** - hard-edge bevels and stepped/terraced CSG transitions
- **Interval Arithmetic** - conservative AABB evaluation for spatial pruning and Lipschitz bound tracking
- **Relaxed Sphere Tracing** - over-relaxation (Keinert 2014, with overshoot retreat) and Lipschitz-adaptive step sizing; `RaymarchConfig::relaxed(&node)` is required for TPMS surfaces (Gyroid, Neovius, …), whose fields are not distance-bounded
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
- **ALICE-Font Bridge** - font glyph → 2D/3D SDF conversion, text layout, 3D extrusion (`--features font` is an inert gate on crates.io until alice-font publishes — see [Installation](../README.md#installation) note; use a `git` dep for the bridge)
- **Auto Tight AABB** - interval arithmetic + binary search to find minimal bounding box containing the SDF surface
- **7 evaluation modes** - interpreted, compiled VM, SIMD 8-wide, BVH, SoA batch, JIT, GPU
- **4 shader targets** - GLSL, WGSL, HLSL transpilation, plus MSL for Metal / Apple (derived from the WGSL emit through naga, so all four share one law source)
- **Engine integrations** - Unity, Unreal Engine 5 / 6, VRChat, Godot, WebAssembly


## Two front ends: this crate's API, or the LOL language

ALICE-SDF is the **evaluator**: it holds the laws (the distance functions), the
compiled backends (scalar / SIMD / BVH / JIT), the shader transpilers and the
mesh pipeline. It does not care how the tree was written.

[**ALICE-LOL**](https://github.com/ext-sakamoro/ALICE-LOL) is the **language for
writing those laws** — a DSL that parses to the very same `SdfNode` tree, plus a
law verifier that answers "does this shape satisfy the constraints" in three
values (satisfied / violated / *undecided*, where undecided is never silently
promoted to a pass).

The same shape, both ways:

```rust
// A: this crate's builder API
use alice_sdf::prelude::*;
let a = SdfNode::sphere(1.0).subtract(SdfNode::box3d(1.0, 1.0, 1.0));

// B: the LOL DSL, parsed at runtime (what an LLM emits)
use alice_lol::runtime_parser::parse_lol;
let b = parse_lol("subtract(sphere(1.0), box3d(0.5, 0.5, 0.5))").unwrap();

// Same field: both evaluate through alice_sdf::eval
assert_eq!(eval(&a, Vec3::new(0.7, 0.2, 0.1)), alice_lol::eval(&b, Vec3::new(0.7, 0.2, 0.1)));
```

Note the box arguments: **this crate's `SdfNode::box3d` takes full extents**
(it halves them internally), while **LOL's `box3d` takes half-extents** — the
DSL writes the variant's field directly. `SdfNode::box3d_half_extents` is the
half-extent constructor on this side if you want the two to read alike
(`tests/readme_parity.rs` in ALICE-LOL pins both examples against each other).

Reach for LOL when you want text in and geometry out (LLM authoring, GBNF
constrained decoding, print-ready STL/3MF from a prompt), or when you want the
law verifier. Reach for this crate directly when you are building the tree in
Rust and want the evaluators, meshing and shader output.

## How the core crates lock together

These four are built as one mechanism rather than as a bundle. Each owns exactly
one thing, and the seams between them are the point:

| Crate | Owns | The joint |
|-------|------|-----------|
| [ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL) | the language and the law verifier | parses to ALICE-SDF's `SdfNode`; verdicts are three-valued (satisfied / violated / *undecided*) and *undecided* is never promoted to a pass |
| [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) | the distance functions and every backend (scalar / SIMD / BVH / JIT / shader transpilers / mesh) | evaluates the tree LOL writes, and supplies colliders to ALICE-Physics |
| [ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics) | 128-bit fixed-point rigid bodies, CCD, XPBD | collides against the same field that is rendered, instead of a second approximation of it |
| [ALICE-DetMath](https://github.com/ext-sakamoro/ALICE-DetMath) | `sin` / `cos` / `atan2` … under a bit-exact contract | the joint itself — ALICE-SDF and ALICE-Physics both call it instead of platform libm |

The coupling exists for one property: **the same input has to produce the same
bits on every platform.** A field that disagrees with itself across machines
cannot be printed to spec, verified by a law, or replayed in lockstep, so the
transcendentals are shared rather than reimplemented per crate.

> Keep `alice-det-math` unified across the resolved graph. Two versions in one
> dependency tree means two implementations of the same function, and the
> guarantee is gone. Check with `cargo tree -i alice-det-math`.

## Installation notes

> **Bridge features on crates.io** — since 1.12.0 `physics` (alice-physics 2) / `codec` (alice-codec 0.1.2) / `asp` (libasp 1.0) / `sdf-cache` (alice-cache 0.2) resolve to the sibling crates on crates.io and are tested by the CI `bridges` job. `font` is still an inert gate: the `font_bridge` module needs a local `alice-font` path dep plus `RUSTFLAGS="--cfg alice_font_bridge"` until alice-font publishes. (Between 1.7.7 and 1.11.0 all five were removed from `[features]`; see the `[v1.7.7]` and `[v1.12.0]` CHANGELOG entries.)

### Claude Code / Codex skill

The `skills/implicit-cad/` directory bundles ALICE-SDF as an installable agent skill for Claude Code / Codex. It exposes SDF authoring, GLSL/WGSL/HLSL/MSL transpile, and mesh export (GLB/OBJ/STL/PLY/3MF) as thin CLI wrappers around this crate. See [`skills/implicit-cad/SKILL.md`](../skills/implicit-cad/SKILL.md). Companion `alice-lol-sdf` skill (in the [ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL) repo) provides the DSL front-end with LLM constrained-decoding support.

## Core Concepts

### SDF (Signed Distance Function)

An SDF returns the shortest distance from any point to the surface:
- **Negative** = inside the shape
- **Zero** = on the surface
- **Positive** = outside the shape

### SdfNode Tree Structure

```
SdfNode
  |-- Primitive (74): Sphere, Box3D, Cylinder, Torus, Plane, Capsule, Cone, Ellipsoid,
  |                    RoundedCone, Pyramid, Octahedron, HexPrism, Link, Triangle, Bezier,
  |                    RoundedBox, CappedCone, CappedTorus, InfiniteCylinder, RoundedCylinder,
  |                    TriangularPrism, CutSphere, CutHollowSphere, DeathStar, SolidAngle,
  |                    Rhombus, Horseshoe, Vesica, InfiniteCone, Heart, Gyroid, MetricBall,
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
  |                    RoundedRect2D, Annular2D,                                   ← 2D primitives (extruded)
  |                    Terrain                                                     ← procedural biome terrain (FBM + Voronoi erosion)
  |-- Operation (25): Union, Intersection, Subtraction,
  |                    SmoothUnion, SmoothIntersection, SmoothSubtraction,
  |                    ChamferUnion, ChamferIntersection, ChamferSubtraction,
  |                    StairsUnion, StairsIntersection, StairsSubtraction,
  |                    ExpSmoothUnion, ExpSmoothIntersection, ExpSmoothSubtraction, ← IQ exponential smooth
  |                    XOR, Morph,                                                 ← Boolean / Interpolation
  |                    ColumnsUnion, ColumnsIntersection, ColumnsSubtraction,      ← hg_sdf columns
  |                    Pipe, Engrave, Groove, Tongue,                              ← hg_sdf advanced
  |                    MetricBlend                                                 ← one field inside a bubble, another outside
  |-- Transform (7): Translate, Rotate, Scale, ScaleNonUniform,
  |                   ProjectiveTransform,                                         ← perspective projection with inv_matrix
  |                   LatticeDeform,                                               ← Free-Form Deformation (FFD) grid
  |                   SdfSkinning                                                  ← bone-weight skeletal deformation
  |-- Modifier (24): Twist, Bend, RepeatInfinite, RepeatFinite, Noise, Round, Onion, Elongate,
  |                   Mirror, Revolution, Extrude, Taper, Displacement, SineDisplacement, PolarRepeat, SweepBezier,
  |                   Shear,                                                       ← 3-axis shear deformation
  |                   OctantMirror,                                                ← 48-fold symmetry
  |                   IcosahedralSymmetry,                                         ← 120-fold icosahedral symmetry
  |                   IFS,                                                         ← Iterated Function System fractals
  |                   HeightmapDisplacement,                                       ← heightmap-driven surface displacement
  |                   SurfaceRoughness,                                            ← FBM noise roughness
  |                   Animated,                                                    ← timeline-driven parameter animation
  |                   WithMaterial                                                 ← PBR material assignment
```

## Usage

### Choose Your Path

Pick the path that matches your role. All paths share the same `SdfNode` intermediate representation, so you can mix them (e.g. build in LOL, transpile to GLSL, evaluate in Python).

| You are… | Path | Best for | Section |
|----------|------|----------|---------|
| **Rust dev, want it declarative** | [ALICE-LOL DSL](#with-alice-lol-dsl-recommended) | Scene composition, GPU shader transpile, compile-time law checks | ↓ |
| **Rust dev, need low-level control** | [Rust direct SdfNode](#rust-direct-sdfnode-construction) | Custom modifier nodes, hand-tuned code paths | ↓ |
| **Python / data-science user** | [Python bindings](#python) | NumPy batch evaluation, mesh export, notebook workflows | ↓ |
| **Unity / UE5 / Godot integrator** | C-ABI FFI | Native plugin, real-time evaluation from game engine | [docs/UNREAL_ENGINE.md](UNREAL_ENGINE.md) / [docs/GODOT_GUIDE.md](GODOT_GUIDE.md) |
| **Web / WebGPU developer** | WASM build | Browser-side SDF evaluation + WGSL shader compile | [docs/WASM_GUIDE.md](WASM_GUIDE.md) |
| **Mobile (iOS / Android)** | XCFramework / AAR | On-device evaluation in Swift / Kotlin | [Mobile section](INTEGRATIONS.md#mobile-ios--android) |
| **3D artist / VFX** | Cookbook recipes | Copy-paste procedural forms, displacement, tiling | [docs/VFX_COOKBOOK.md](VFX_COOKBOOK.md) |
| **First time here** | 30-second sample below ↓ | Sanity check the install | ↓ |

### Hello, First SDF (30 seconds)

The smallest useful sample — construct, evaluate, mesh, done:

```rust
use alice_sdf::prelude::*;

let sphere = SdfNode::sphere(1.0);
let d = eval(&sphere, glam::Vec3::new(0.5, 0.0, 0.0));
assert!((d + 0.5).abs() < 1e-6);           // point 0.5 away from center → distance -0.5 (inside)

let mesh = sdf_to_mesh(
    &sphere,
    glam::Vec3::splat(-1.5),
    glam::Vec3::splat(1.5),
    &MarchingCubesConfig::default(),
);
println!("{} vertices, {} triangles", mesh.vertices.len(), mesh.indices.len() / 3);
```

Once this runs, jump to the path that matches your role above.

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

**Mesh export from a LOL scene:**

```rust
use alice_lol::lol;
use alice_sdf::prelude::*;

let scene = lol! {
    smooth_union(0.3,
        sphere(1.0),
        translate(2.0, 0.0, 0.0, box3d(0.8, 0.8, 0.8))
    )
};

let mesh = sdf_to_mesh(
    &scene,
    glam::Vec3::splat(-3.0),
    glam::Vec3::splat(3.0),
    &MarchingCubesConfig { resolution: 128, ..Default::default() },
);

alice_sdf::export::obj::write_obj("out.obj", &mesh)?;
alice_sdf::export::glb::write_glb("out.glb", &mesh)?;
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


### Sine Displacement (v1.7.3, isotropic + anisotropic)

`SineDisplacement` adds a sin-wave perturbation to any child SDF. As of **v1.7.3** it accepts a per-axis `Vec3` frequency, so you can drive fine XYZ-independent patterns (scales, wood grain, cloth drape) from a single node.

Two ergonomic constructors:

| Method | Frequency type | Use for |
|--------|---------------|---------|
| `.sine_displacement(amplitude, freq: f32)` | isotropic (`Vec3::splat`) | uniform ripples / roughness / cellular scales |
| `.sine_displacement_aniso(amplitude, freq: Vec3)` | per-axis | wood grain, cloth drape, fish scales, wave-fronts along one axis |

```rust
use alice_sdf::prelude::*;
use glam::Vec3;

// Isotropic: sphere with fine cellular perturbation.
let scales = SdfNode::sphere(1.0).sine_displacement(0.03, 25.0);

// Anisotropic: long thin scales (high freq on X, low on Y and Z).
let fish_scales = SdfNode::sphere(1.0).sine_displacement_aniso(0.03, Vec3::new(40.0, 10.0, 10.0));

// Anisotropic: wood grain (rings dense along Y, sparse elsewhere).
let wood = SdfNode::box3d(1.0, 3.0, 1.0)
    .sine_displacement_aniso(0.01, Vec3::new(4.0, 30.0, 4.0));
```

Both variants transpile to GLSL / WGSL / HLSL through the standard pipeline (`to_glsl` / `to_wgsl` / `to_hlsl`). The transpiler emits per-axis `sin(freq.x * p.x) * sin(freq.y * p.y) * sin(freq.z * p.z)`, matching the CPU `modifier_sine_displacement` reference. **Interval-arithmetic bounds and diff (derivative) support are already wired**, so displaced SDFs remain safe for raymarching and gradient-based sampling.

### Common Recipes

Copy-paste patterns that come up over and over. All examples assume `use alice_sdf::prelude::*;`.

**1. Rounded box (chamfer)**

```rust
let rounded = SdfNode::box3d(1.0, 1.0, 1.0).round(0.15);
```

**2. Smooth blob (union with soft blend)**

```rust
let blob = SdfNode::sphere(1.0)
    .smooth_union(SdfNode::sphere(0.7).translate(1.2, 0.0, 0.0), 0.4);
// signature: smooth_union(self, other: Self, k: f32) — k is the blend radius
```

**3. Hollow shell (onion)**

```rust
let shell = SdfNode::sphere(1.0).onion(0.05);   // 5cm thick wall
```

**4. Infinite tiling (`repeat_infinite`)**

```rust
let tiles = SdfNode::box3d(0.4, 0.4, 0.4).repeat_infinite(1.0, 1.0, 1.0);
// finite variant: .repeat_finite([count_x, count_y, count_z], spacing)
```

**5. Twist along Y**

```rust
let twisted = SdfNode::box3d(0.4, 2.0, 0.4).twist(1.5);   // 1.5 rad/unit around Y
```

**6. Displacement (sine + noise)**

```rust
let rough = SdfNode::sphere(1.0).sine_displacement(0.03, 20.0);
```

**7. CSG chain (subtract holes from a plate)**

```rust
let plate = SdfNode::box3d(2.0, 0.1, 2.0);
let hole  = SdfNode::cylinder(2.0, 0.15);
let drilled = plate
    .subtract(hole.translate(-0.8, 0.0,  0.0))
    .subtract(hole.translate( 0.8, 0.0,  0.0))
    .subtract(hole.translate( 0.0, 0.0, -0.8));
```

**8. GPU transpile (WebGPU / Metal / DX12)**

```rust
use alice_lol::{lol, to_wgsl};

let scene = lol! { smooth_union(0.3, sphere(1.0), box3d(0.8, 0.8, 0.8)) };
let wgsl_source = to_wgsl(&scene);      // paste into a WGSL shader
```

For **VFX-focused patterns** (fluid simulation, mandelbulb, magic effects, ribbon FX, plasma balls, portals, force fields) see [`docs/VFX_COOKBOOK.md`](VFX_COOKBOOK.md).

For **general recipe collection** (procedural terrain, camera-relative modifiers, bounding-volume tricks, LOD strategy) see [`docs/COOKBOOK.md`](COOKBOOK.md).

### Where to go next

| I want to… | See |
|------------|-----|
| Understand every `SdfNode` variant | [`docs/API_REFERENCE.md`](API_REFERENCE.md) |
| Set up a fresh project | [`docs/QUICKSTART.md`](QUICKSTART.md) |
| Deep dive into the compiler / evaluator internals | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) |
| Wire it into Unity / Unreal / Godot | [`docs/UNREAL_ENGINE.md`](UNREAL_ENGINE.md) · [`docs/GODOT_GUIDE.md`](GODOT_GUIDE.md) |
| Use it from Python | [`docs/PYTHON_GUIDE.md`](PYTHON_GUIDE.md) |
| Ship it in a browser | [`docs/WASM_GUIDE.md`](WASM_GUIDE.md) |
| Build 3D-print-ready parts | See [ALICE-Bamboo](https://github.com/ext-sakamoro/ALICE-Bamboo) (LOL → SDF → 3MF pipeline) |

For deep technical sections (Material / Animation / Architecture / Mesh / Platonic Solids / Interval Arithmetic / Neural SDF / Collision / Analytic Gradient / Dual Contouring / CSG Tree Optimization / Auto Tight AABB / Texture Fitting / Raymarching / FFI bindings / Feature Flags / Physics Bridge / 3D Print Pipeline / Performance / Benchmarking / Unity / VRChat / Unreal / Godot / Cross-Crate Bridges / Asset Delivery Network / Nanite hybrid pipeline) see [`docs/USAGE.md`](USAGE.md).

