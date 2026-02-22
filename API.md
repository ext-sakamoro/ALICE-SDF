# ALICE-SDF API Reference

## Quick Start

```rust
use alice_sdf::prelude::*;

let shape = SdfNode::sphere(1.0)
    .subtract(SdfNode::box3d(0.5, 0.5, 0.5))
    .translate(1.0, 0.0, 0.0);

let distance = eval(&shape, Vec3::new(0.0, 0.0, 0.0));

let mesh = sdf_to_mesh(
    &shape,
    Vec3::splat(-2.0), Vec3::splat(2.0),
    &MarchingCubesConfig::default(),
);
```

## Primitives (72)

All primitives are constructors on `SdfNode`:

### Basic Shapes
| Function | Parameters | Description |
|----------|-----------|-------------|
| `SdfNode::sphere(radius)` | `f32` | Sphere |
| `SdfNode::box3d(hx, hy, hz)` | `f32, f32, f32` | Axis-aligned box |
| `SdfNode::cylinder(radius, height)` | `f32, f32` | Cylinder along Y |
| `SdfNode::torus(major, minor)` | `f32, f32` | Torus around Y |
| `SdfNode::plane(nx, ny, nz, d)` | `f32, f32, f32, f32` | Infinite plane |
| `SdfNode::capsule(a, b, radius)` | `Vec3, Vec3, f32` | Line capsule |
| `SdfNode::cone(angle, height)` | `f32, f32` | Cone |
| `SdfNode::ellipsoid(rx, ry, rz)` | `f32, f32, f32` | Ellipsoid |

### Extended Shapes
`rounded_cone`, `pyramid`, `octahedron`, `hex_prism`, `link`, `triangle_prim`, `bezier`, `rounded_box`, `capped_cone`, `capped_torus`, `rounded_cylinder`, `triangular_prism`, `cut_sphere`, `cut_hollow_sphere`, `death_star`, `solid_angle`, `rhombus`, `horseshoe`, `vesica`, `heart`, `tube`, `barrel`, `chamfered_cube`, `superellipsoid`, `rounded_x`, `pie`, `trapezoid`, `parallelogram`, `tunnel`, `uneven_capsule`, `egg`, `arc_shape`, `moon`, `cross_shape`, `blobby_cross`, `parabola_segment`, `regular_polygon`, `star_polygon`, `stairs_prim`, `helix`, `box_frame`

### Platonic / Archimedean Solids (GDF)
`tetrahedron`, `dodecahedron`, `icosahedron`, `truncated_octahedron`, `truncated_icosahedron`

### TPMS Surfaces
`gyroid`, `schwarz_p`, `diamond_surface`, `neovius`, `lidinoid`, `iwp`, `frd`, `fischer_koch_s`, `pmy`

### 2D Primitives (extruded along Z)
`circle_2d`, `rect_2d`, `segment_2d`, `polygon_2d`, `rounded_rect_2d`, `annular_2d`

### Infinite Shapes
`infinite_cylinder`, `infinite_cone`

## Operations (24)

All operations are methods on `SdfNode`:

| Method | Parameters | Description |
|--------|-----------|-------------|
| `.union(other)` | `SdfNode` | Boolean union (min) |
| `.intersection(other)` | `SdfNode` | Boolean intersection (max) |
| `.subtract(other)` | `SdfNode` | Boolean subtraction |
| `.smooth_union(other, k)` | `SdfNode, f32` | Polynomial smooth union |
| `.smooth_intersection(other, k)` | `SdfNode, f32` | Polynomial smooth intersection |
| `.smooth_subtraction(other, k)` | `SdfNode, f32` | Polynomial smooth subtraction |
| `.chamfer_union(other, k)` | `SdfNode, f32` | Hard-edge bevel union |
| `.chamfer_intersection(other, k)` | `SdfNode, f32` | Hard-edge bevel intersection |
| `.chamfer_subtraction(other, k)` | `SdfNode, f32` | Hard-edge bevel subtraction |
| `.stairs_union(other, k, n)` | `SdfNode, f32, u32` | Stepped/terraced union |
| `.stairs_intersection(other, k, n)` | `SdfNode, f32, u32` | Stepped intersection |
| `.stairs_subtraction(other, k, n)` | `SdfNode, f32, u32` | Stepped subtraction |
| `.exp_smooth_union(other, k)` | `SdfNode, f32` | Exponential smooth union |
| `.exp_smooth_intersection(other, k)` | `SdfNode, f32` | Exponential smooth intersection |
| `.exp_smooth_subtraction(other, k)` | `SdfNode, f32` | Exponential smooth subtraction |
| `.columns_union(other, r, n)` | `SdfNode, f32, u32` | Column blend union |
| `.columns_intersection(other, r, n)` | `SdfNode, f32, u32` | Column blend intersection |
| `.columns_subtraction(other, r, n)` | `SdfNode, f32, u32` | Column blend subtraction |
| `.xor(other)` | `SdfNode` | Symmetric difference |
| `.morph(other, t)` | `SdfNode, f32` | Linear blend (0.0=self, 1.0=other) |
| `.pipe(other, r)` | `SdfNode, f32` | Pipe along intersection |
| `.engrave(other, r)` | `SdfNode, f32` | Engrave pattern |
| `.groove(other, ra, rb)` | `SdfNode, f32, f32` | Groove cut |
| `.tongue(other, ra, rb)` | `SdfNode, f32, f32` | Tongue protrusion |

## Transforms (7)

| Method | Parameters | Description |
|--------|-----------|-------------|
| `.translate(x, y, z)` | `f32, f32, f32` | Translation |
| `.rotate_euler(rx, ry, rz)` | `f32, f32, f32` | Euler rotation (radians) |
| `.scale(s)` | `f32` | Uniform scale |
| `.scale_non_uniform(sx, sy, sz)` | `f32, f32, f32` | Non-uniform scale |
| `.projective_transform(inv_matrix, lipschitz)` | `[f32; 16], f32` | Perspective projection |
| `.lattice_deform(cps, nx, ny, nz, min, max)` | `Vec<Vec3>, u32, u32, u32, Vec3, Vec3` | FFD grid |
| `.sdf_skinning(bones)` | `Vec<BoneTransform>` | Skeletal deformation |

## Modifiers (23)

| Method | Parameters | Description |
|--------|-----------|-------------|
| `.twist(strength)` | `f32` | Twist around Y |
| `.bend(curvature)` | `f32` | Bend around Y |
| `.repeat_infinite(period)` | `Vec3` | Infinite repetition |
| `.repeat_finite(period, count)` | `Vec3, Vec3` | Finite repetition |
| `.noise(amplitude, frequency, seed)` | `f32, f32, u32` | Noise displacement |
| `.round(radius)` | `f32` | Round edges |
| `.onion(thickness)` | `f32` | Hollow shell |
| `.elongate(h)` | `Vec3` | Elongation |
| `.mirror(axis)` | `Vec3` | Axis mirror |
| `.revolution()` | — | Revolution around Y |
| `.extrude(height)` | `f32` | Extrude 2D→3D along Z |
| `.taper(amount)` | `f32` | Taper along Y |
| `.displacement(amplitude, frequency)` | `f32, f32` | Sine displacement |
| `.polar_repeat(count)` | `u32` | Polar array around Y |
| `.sweep_bezier(a, b, c)` | `Vec3, Vec3, Vec3` | Sweep along Bezier |
| `.shear(xy, xz, yz)` | `f32, f32, f32` | 3-axis shear |
| `.octant_mirror()` | — | 48-fold octahedral symmetry |
| `.icosahedral_symmetry()` | — | 120-fold icosahedral symmetry |
| `.ifs(transforms, iterations)` | `Vec<[f32; 16]>, u32` | Iterated Function System |
| `.heightmap_displacement(hm, w, h, amp, scale)` | `Vec<f32>, u32, u32, f32, f32` | Heightmap displacement |
| `.surface_roughness(freq, amp, octaves)` | `f32, f32, u32` | FBM noise roughness |
| `.animated(timeline)` | `Timeline` | Keyframe animation |
| `.with_material(id)` | `u32` | PBR material assignment |

## Evaluation Functions

### Single-Point
| Function | Description |
|----------|-------------|
| `eval(&node, point)` | Tree-walker evaluation |
| `eval_compiled(&compiled, point)` | Compiled VM evaluation |
| `eval_compiled_bvh(&bvh, point)` | BVH-accelerated evaluation |
| `eval_compiled_simd(&compiled, points8)` | SIMD 8-wide evaluation |

### Batch
| Function | Description |
|----------|-------------|
| `eval_batch(&node, &[Vec3])` | Interpreted batch |
| `eval_batch_parallel(&node, &[Vec3])` | Interpreted parallel batch |
| `eval_compiled_batch_simd(&compiled, &[Vec3])` | Compiled SIMD batch |
| `eval_compiled_batch_simd_parallel(&compiled, &[Vec3])` | Compiled SIMD parallel batch |
| `eval_compiled_batch_soa(&compiled, &SoAPoints)` | Compiled SoA batch |
| `eval_compiled_batch_soa_parallel(&compiled, &SoAPoints)` | Compiled SoA parallel batch |

### Gradient / Normal
| Function | Description |
|----------|-------------|
| `eval_gradient(&node, point)` | Analytic gradient (chain rules) |
| `eval_normal(&node, point)` | Normalized gradient |
| `eval_compiled_normal(&compiled, point)` | Compiled normal (finite diff) |
| `eval_compiled_distance_and_normal(&compiled, point)` | Distance + normal |
| `eval_with_gradient(&node, point)` | Dual3 AD gradient |
| `eval_hessian(&node, point)` | Hessian via Dual3 |
| `mean_curvature(&node, point)` | Mean curvature estimation |

## Compilation

```rust
let compiled = CompiledSdf::compile(&node);    // AST → bytecode + aux_data
let bvh = CompiledSdfBvh::compile(&node)?;     // AST → BVH bytecode
```

| Field | Type | Description |
|-------|------|-------------|
| `compiled.instructions` | `Vec<Instruction>` | Bytecode instructions |
| `compiled.aux_data` | `Vec<f32>` | Auxiliary data (matrices, heightmaps) |
| `compiled.node_count` | `usize` | Number of AST nodes |
| `compiled.instruction_count()` | `usize` | Number of instructions |
| `compiled.memory_size()` | `usize` | Total memory in bytes |

## Mesh Generation

```rust
let mesh = sdf_to_mesh(&node, min, max, &MarchingCubesConfig::default());
let mesh = dual_contouring(&node, min, max, &DualContouringConfig::default());
let mesh = adaptive_marching_cubes(&node, min, max, &AdaptiveConfig::default());
```

## I/O

### Export
```rust
export_obj(&mesh, "model.obj", &ObjConfig::default(), Some(&mat_lib))?;
export_glb(&mesh, "model.glb", &GltfConfig::default(), Some(&mat_lib))?;
export_fbx(&mesh, "model.fbx", &FbxConfig::default())?;
export_usda(&mesh, "model.usda", &UsdConfig::default())?;
export_alembic(&mesh, "model.abc", &AlembicConfig::default())?;
save_asdf(&tree, "model.asdf")?;
save_asdf_json(&tree, "model.asdf.json")?;
```

### Import
```rust
let mesh = import_obj("model.obj")?;
let mesh = import_glb("model.glb")?;
let tree = load_asdf("model.asdf")?;
let tree = load_asdf_json("model.asdf.json")?;
```

## Feature Flags

| Feature | Dependencies | Description |
|---------|-------------|-------------|
| `cli` (default) | clap | Command-line interface |
| `python` | pyo3, numpy | Python bindings |
| `godot` | godot | Godot 4.x GDExtension |
| `jit` | cranelift-* | JIT native code compilation |
| `gpu` | wgpu, pollster, bytemuck | GPU compute evaluation |
| `glsl` | — | GLSL shader transpiler |
| `hlsl` | — | HLSL shader transpiler |
| `ffi` | lazy_static | C/C++/C# FFI bindings |
| `svo` | — | Sparse Voxel Octree |
| `destruction` | — | Voxel destruction system |
| `terrain` | — | Terrain heightmap + erosion |
| `gi` | svo | Cone tracing global illumination |
| `volume` | gpu | 3D volume texture baking |
| `gpu-mesh` | gpu | GPU Marching Cubes |
| `unity` | ffi, glsl | Unity integration |
| `unreal` | ffi, hlsl | Unreal Engine 5 integration |
| `aaa` | volume, gpu-mesh, svo-gpu, destruction, terrain, gi | All AAA game features |
