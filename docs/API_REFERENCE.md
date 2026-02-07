# ALICE-SDF API Reference

Complete API reference for all modules.

## Table of Contents

- [Primitives](#primitives)
- [Operations](#operations)
- [Transforms](#transforms)
- [Modifiers](#modifiers)
- [Evaluation](#evaluation)
- [Mesh Generation](#mesh-generation)
- [Mesh Processing](#mesh-processing)
- [Collision & Physics](#collision--physics)
- [Material System](#material-system)
- [Animation](#animation)
- [File I/O](#file-io)
- [Compiled / VM](#compiled--vm)
- [Shader Transpilers](#shader-transpilers)
- [FFI (C/C++/C#)](#ffi-ccc)

---

## Primitives

All primitives return `SdfNode` and can be chained with operations, transforms, and modifiers.

### Core Primitives (15)

| Constructor | Parameters | Description |
|-------------|-----------|-------------|
| `SdfNode::sphere(radius)` | `f32` | Sphere centered at origin |
| `SdfNode::box3d(width, height, depth)` | `f32, f32, f32` | Axis-aligned box (full dimensions, halved internally) |
| `SdfNode::cylinder(radius, height)` | `f32, f32` | Cylinder along Y axis (full height, halved internally) |
| `SdfNode::torus(major_radius, minor_radius)` | `f32, f32` | Torus in XZ plane |
| `SdfNode::capsule(point_a, point_b, radius)` | `Vec3, Vec3, f32` | Capsule between two points |
| `SdfNode::plane(normal, distance)` | `Vec3, f32` | Infinite plane |
| `SdfNode::cone(radius, height)` | `f32, f32` | Cone along Y axis |
| `SdfNode::ellipsoid(rx, ry, rz)` | `f32, f32, f32` | Ellipsoid (non-uniform radii) |
| `SdfNode::rounded_cone(r1, r2, height)` | `f32, f32, f32` | Rounded cone |
| `SdfNode::pyramid(height)` | `f32` | Pyramid (square base) |
| `SdfNode::octahedron(size)` | `f32` | Regular octahedron |
| `SdfNode::hex_prism(hex_radius, height)` | `f32, f32` | Hexagonal prism |
| `SdfNode::link(length, r1, r2)` | `f32, f32, f32` | Chain link |
| `SdfNode::triangle(v0, v1, v2)` | `Vec3, Vec3, Vec3` | Triangle primitive |
| `SdfNode::bezier(v0, v1, v2)` | `Vec3, Vec3, Vec3` | Quadratic Bezier curve |

### Extended Geometric (16)

| Constructor | Parameters | Description |
|-------------|-----------|-------------|
| `SdfNode::rounded_box(w, h, d, r)` | `f32, f32, f32, f32` | Box with rounded edges |
| `SdfNode::capped_cone(h, r1, r2)` | `f32, f32, f32` | Capped cone (two radii) |
| `SdfNode::capped_torus(ro, ri, angle)` | `f32, f32, f32` | Partial torus arc |
| `SdfNode::rounded_cylinder(r, rr, h)` | `f32, f32, f32` | Cylinder with rounded edges |
| `SdfNode::triangular_prism(w, h)` | `f32, f32` | Triangular cross-section prism |
| `SdfNode::cut_sphere(r, h)` | `f32, f32` | Sphere with flat cut |
| `SdfNode::cut_hollow_sphere(r, h, t)` | `f32, f32, f32` | Hollow sphere with cut |
| `SdfNode::death_star(ra, rb, d)` | `f32, f32, f32` | Death Star shape |
| `SdfNode::solid_angle(angle, ra)` | `f32, f32` | Solid angle (cone sector) |
| `SdfNode::rhombus(la, lb, h, ra)` | `f32, f32, f32, f32` | Rhombus shape |
| `SdfNode::horseshoe(angle, r, w, h)` | `f32, f32, f32, f32` | Horseshoe shape |
| `SdfNode::vesica(a, b, w)` | `Vec3, Vec3, f32` | Vesica shape |
| `SdfNode::infinite_cylinder(dir, r)` | `Vec3, f32` | Infinite cylinder |
| `SdfNode::infinite_cone(angle)` | `f32` | Infinite cone |
| `SdfNode::gyroid(scale, thickness)` | `f32, f32` | Gyroid minimal surface |
| `SdfNode::heart(size)` | `f32` | Heart shape |

### 3D Native (7)

| Constructor | Parameters | Description |
|-------------|-----------|-------------|
| `SdfNode::tube(outer_r, inner_r, h)` | `f32, f32, f32` | Tube (hollow cylinder) |
| `SdfNode::barrel(r, h, bulge)` | `f32, f32, f32` | Barrel shape |
| `SdfNode::diamond(size)` | `f32` | Diamond (octahedron variant) |
| `SdfNode::chamfered_cube(size, chamfer)` | `f32, f32` | Cube with chamfered edges |
| `SdfNode::schwarz_p(scale, thickness)` | `f32, f32` | Schwarz P minimal surface |
| `SdfNode::superellipsoid(r, e1, e2)` | `f32, f32, f32` | Superellipsoid |
| `SdfNode::rounded_x(w, r)` | `f32, f32` | Rounded X shape |

### 2D→3D Prisms (13)

| Constructor | Parameters | Description |
|-------------|-----------|-------------|
| `SdfNode::pie(r, angle)` | `f32, f32` | Pie slice |
| `SdfNode::trapezoid(r1, r2, h)` | `f32, f32, f32` | Trapezoid prism |
| `SdfNode::parallelogram(w, h, skew)` | `f32, f32, f32` | Parallelogram |
| `SdfNode::tunnel(r, h)` | `f32, f32` | Tunnel shape |
| `SdfNode::uneven_capsule(r1, r2, h)` | `f32, f32, f32` | Capsule with different end radii |
| `SdfNode::egg(r)` | `f32` | Egg shape |
| `SdfNode::arc_shape(r, angle, w)` | `f32, f32, f32` | Arc shape |
| `SdfNode::moon(d, ra, rb)` | `f32, f32, f32` | Moon/crescent |
| `SdfNode::cross_shape(w, h, r)` | `f32, f32, f32` | Cross shape |
| `SdfNode::blobby_cross(size)` | `f32` | Blobby cross |
| `SdfNode::parabola_segment(w, h)` | `f32, f32` | Parabola segment |
| `SdfNode::regular_polygon(r, n)` | `f32, u32` | Regular N-gon |
| `SdfNode::star_polygon(r, n, m)` | `f32, u32, u32` | Star polygon {n/m} |

### Complex 3D (2)

| Constructor | Parameters | Description |
|-------------|-----------|-------------|
| `SdfNode::stairs(step_count, step_height, step_depth)` | `u32, f32, f32` | Staircase |
| `SdfNode::helix(r, pitch, thickness)` | `f32, f32, f32` | Helix / spring |

---

## Operations

### Boolean (Sharp)

| Method | Parameters | Description |
|--------|-----------|-------------|
| `a.union(b)` | `SdfNode` | A + B (union) |
| `a.intersection(b)` | `SdfNode` | A & B (intersection) |
| `a.subtract(b)` | `SdfNode` | A - B (subtraction) |

### Boolean (Smooth)

| Method | Parameters | Description |
|--------|-----------|-------------|
| `a.smooth_union(b, k)` | `SdfNode, f32` | Smooth union (k = blend radius) |
| `a.smooth_intersection(b, k)` | `SdfNode, f32` | Smooth intersection |
| `a.smooth_subtract(b, k)` | `SdfNode, f32` | Smooth subtraction |

### Multi-Shape (Module-Level Functions)

| Function | Parameters | Description |
|----------|-----------|-------------|
| `sdf_union_multi(distances)` | `&[f32]` | Union of pre-evaluated distances |
| `sdf_intersection_multi(distances)` | `&[f32]` | Intersection of pre-evaluated distances |

---

## Transforms

| Method | Parameters | Description |
|--------|-----------|-------------|
| `node.translate(x, y, z)` | `f32, f32, f32` | Translation |
| `node.rotate(quaternion)` | `Quat` | Rotation by quaternion |
| `node.rotate_euler(rx, ry, rz)` | `f32, f32, f32` | Rotation by Euler angles (radians) |
| `node.scale(factor)` | `f32` | Uniform scale |
| `node.scale_xyz(x, y, z)` | `f32, f32, f32` | Non-uniform scale |

---

## Modifiers

| Method | Parameters | Description |
|--------|-----------|-------------|
| `node.twist(strength)` | `f32` | Twist around Y axis (radians/unit) |
| `node.bend(curvature)` | `f32` | Bend along X axis |
| `node.repeat_infinite(sx, sy, sz)` | `f32, f32, f32` | Infinite repetition |
| `node.repeat_finite(count, spacing)` | `[u32; 3], Vec3` | Finite repetition (count per axis, spacing) |
| `node.noise(amplitude, frequency, seed)` | `f32, f32, u32` | Perlin noise displacement |
| `node.round(radius)` | `f32` | Round edges |
| `node.onion(thickness)` | `f32` | Shell / onion |
| `node.elongate(amount)` | `Vec3` | Elongate along axes |
| `node.extrude(height)` | `f32` | 2D to 3D extrusion |
| `node.revolution(offset)` | `f32` | Revolution around Y axis (radial offset) |
| `node.mirror_x()` / `mirror_y()` / `mirror_z()` | - | Mirror across axis |
| `node.with_material(id)` | `u32` | Assign material ID |

---

## Evaluation

### Single Point

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `eval(node, point)` | `&SdfNode, Vec3` | `f32` | Evaluate SDF distance |
| `eval_material(node, point)` | `&SdfNode, Vec3` | `u32` | Material ID at point |
| `normal(node, point, epsilon)` | `&SdfNode, Vec3, f32` | `Vec3` | Surface normal via gradient |
| `gradient(node, point, eps)` | `&SdfNode, Vec3, f32` | `Vec3` | Gradient vector |

### Batch

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `eval_batch(node, points)` | `&SdfNode, &[Vec3]` | `Vec<f32>` | Sequential batch |
| `eval_batch_parallel(node, points)` | `&SdfNode, &[Vec3]` | `Vec<f32>` | Parallel batch (Rayon) |
| `eval_grid(node, min, max, res)` | `&SdfNode, Vec3, Vec3, usize` | `Vec<f32>` | 3D grid evaluation |

---

## Mesh Generation

### Marching Cubes

```rust
let mesh = sdf_to_mesh(&shape, min_bounds, max_bounds, &config);
```

| Config Field | Type | Default | Description |
|-------------|------|---------|-------------|
| `resolution` | `usize` | 32 | Grid resolution per axis |
| `iso_level` | `f32` | 0.0 | Surface threshold |
| `compute_normals` | `bool` | true | Calculate vertex normals |
| `compute_uvs` | `bool` | false | Generate triplanar UVs |
| `uv_scale` | `f32` | 1.0 | UV tiling scale |
| `compute_tangents` | `bool` | false | MikkTSpace tangent generation |
| `compute_materials` | `bool` | false | Per-vertex material IDs |

**Preset**: `MarchingCubesConfig::aaa(resolution)` enables all attributes.

### Compiled Marching Cubes

Compiled VM path with SIMD batch grid evaluation and grid finite-difference normals.

```rust
let compiled = CompiledSdf::compile(&shape);

// Standard compiled MC (SIMD batch grid eval + grid FD normals)
let mesh = sdf_to_mesh_compiled(&compiled, min, max, &config);

// Low-level compiled MC (no dedup/tangent post-processing)
let mesh = marching_cubes_compiled(&compiled, min, max, &config);
```

### Adaptive Marching Cubes

```rust
// Interpreted
let mesh = adaptive_marching_cubes(&shape, min, max, &config);

// Compiled VM (2-5x faster)
let compiled = CompiledSdf::compile(&shape);
let mesh = adaptive_marching_cubes_compiled(&compiled, min, max, &config);
```

| Config Field | Type | Default | Description |
|-------------|------|---------|-------------|
| `max_depth` | `u32` | 6 | Maximum octree depth (effective resolution = 2^max_depth) |
| `min_depth` | `u32` | 2 | Minimum subdivision depth |
| `surface_threshold` | `f32` | 0.0 | Surface proximity threshold (0 = auto) |
| `iso_level` | `f32` | 0.0 | Surface level |
| `compute_normals` | `bool` | true | Calculate normals |
| `compute_uvs` | `bool` | false | Generate UVs |
| `compute_tangents` | `bool` | false | Generate tangents |

**Preset**: `AdaptiveConfig::aaa(max_depth)` enables all attributes.

### Mesh to SDF

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `mesh_to_sdf(mesh, config)` | `&Mesh, &MeshToSdfConfig` | `SdfNode` | Capsule approximation |
| `mesh_to_sdf_exact(mesh)` | `&Mesh` | `MeshSdf` | BVH exact distance (not exportable to shaders) |

---

## Mesh Processing

### Decimation

```rust
decimate(&mut mesh, &DecimateConfig { target_ratio: 0.5, ..Default::default() });
```

| Config Field | Type | Default | Description |
|-------------|------|---------|-------------|
| `target_ratio` | `f32` | 0.5 | Target triangle ratio (0.5 = halve) |
| `max_error` | `f32` | f32::MAX | Maximum QEM error threshold |
| `preserve_materials` | `bool` | true | Prevent cross-material edge collapses |
| `locked_materials` | `Vec<u32>` | `[]` | Material IDs that cannot be decimated |

### LOD Generation

```rust
// Resolution-based LOD
let lod_chain = generate_lod_chain(&shape, min, max, &LodConfig::default());

// Decimation-based LOD (recommended)
let lod_chain = generate_lod_chain_decimated(&shape, min, max, &DecimationLodConfig::default());
```

| DecimationLodConfig Field | Type | Default | Description |
|--------------------------|------|---------|-------------|
| `num_levels` | `usize` | 4 | Number of LOD levels |
| `base_resolution` | `usize` | 64 | LOD 0 resolution |
| `decimation_ratio` | `f32` | 0.5 | Triangle reduction per level |
| `preserve_materials` | `bool` | true | Material boundary preservation |

### Vertex Optimization

| Function | Description |
|----------|-------------|
| `optimize_vertex_cache(&mut mesh)` | Reorder triangles for GPU vertex cache |
| `compute_acmr(&mesh, cache_size)` | Compute Average Cache Miss Ratio |
| `deduplicate_vertices(&mut mesh, epsilon)` | Merge duplicate vertices |

### Lightmap UV Generation

| Function | Description |
|----------|-------------|
| `generate_lightmap_uvs(&mut mesh)` | Generate non-overlapping UV2 for lightmapping |
| `generate_lightmap_uvs_fast(&mut mesh)` | Fast approximate lightmap UVs |

### Manifold Validation

| Function | Description |
|----------|-------------|
| `validate_mesh(&mesh)` | Full topology validation report |
| `compute_quality(&mesh)` | Aspect ratio and area metrics |
| `MeshRepair::repair_all(&mesh, eps)` | Fix all detected issues |
| `MeshRepair::remove_degenerate_triangles(&mesh, eps)` | Remove zero-area triangles |
| `MeshRepair::merge_duplicate_vertices(&mesh, eps)` | Weld close vertices |
| `MeshRepair::fix_normals(&mesh)` | Fix inconsistent winding |

---

## Collision & Physics

### Basic Primitives

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_aabb(&mesh)` | `CollisionAabb` | Axis-aligned bounding box |
| `compute_bounding_sphere(&mesh)` | `BoundingSphere` | Minimum bounding sphere |
| `compute_convex_hull(&mesh)` | `ConvexHull` | Convex hull of mesh |
| `convex_hull_from_points(&points)` | `ConvexHull` | Convex hull from point cloud |
| `simplify_collision(&mesh, target)` | `CollisionMesh` | Simplified collision mesh |

### V-HACD Convex Decomposition

```rust
let decomposition = convex_decomposition(&mesh, &VhacdConfig::default());
```

| Config Field | Type | Default | Description |
|-------------|------|---------|-------------|
| `max_hulls` | `usize` | 16 | Maximum convex hulls |
| `resolution` | `usize` | 100000 | Voxelization resolution |
| `max_vertices_per_hull` | `usize` | 32 | Max vertices per hull |
| `volume_error_percent` | `f32` | 1.0 | Acceptable volume error |

---

## Material System

### Material Creation

```rust
Material::default()                                      // White dielectric
Material::metal(name, r, g, b, roughness)               // Metallic
Material::glass(name, ior)                                // Transparent glass
Material::emissive(name, r, g, b, strength)              // Emissive
```

### Material Properties

| Property | Type | Range | Description |
|----------|------|-------|-------------|
| `base_color` | `[f32; 4]` | 0-1 | RGBA linear color |
| `metallic` | `f32` | 0-1 | 0=dielectric, 1=metal |
| `roughness` | `f32` | 0-1 | 0=mirror, 1=diffuse |
| `emission` | `[f32; 3]` | 0+ | Emissive RGB |
| `emission_strength` | `f32` | 0+ | Emission multiplier |
| `opacity` | `f32` | 0-1 | 0=transparent, 1=opaque |
| `ior` | `f32` | 1-3 | Index of refraction |
| `normal_scale` | `f32` | 0-2 | Normal map strength |

### Extended PBR (glTF extensions)

| Extension | Properties |
|-----------|-----------|
| Clearcoat | `clearcoat_factor`, `clearcoat_roughness` |
| Sheen | `sheen_color`, `sheen_roughness` |
| Transmission | `transmission_factor` |
| Volume | `thickness_factor`, `attenuation_distance`, `attenuation_color` |
| Anisotropy | `anisotropy_strength`, `anisotropy_rotation` |
| Subsurface | `subsurface_factor`, `subsurface_color` |

### Texture Slots

```rust
material
    .with_albedo_map("textures/albedo.png")
    .with_normal_map("textures/normal.png")
    .with_metallic_roughness_map("textures/mr.png")
    .with_emission_map("textures/emissive.png")
    .with_ao_map("textures/ao.png")
```

---

## Animation

### Timeline

```rust
let mut timeline = Timeline::new("bounce");
let mut track = Track::new("translate.y").with_loop(LoopMode::PingPong);
track.add_keyframe(Keyframe::new(0.0, 0.0));
track.add_keyframe(Keyframe::cubic(0.5, 3.0, 0.0, 0.0));
track.add_keyframe(Keyframe::new(1.0, 0.0));
timeline.add_track(track);
```

### Interpolation Modes

| Mode | Description |
|------|-------------|
| `Interpolation::Linear` | Linear interpolation |
| `Interpolation::Cubic` | Cubic Hermite (Bezier tangents) |
| `Interpolation::Step` | Instant jump at keyframe |

### Loop Modes

| Mode | Description |
|------|-------------|
| `LoopMode::Once` | Play once and stop |
| `LoopMode::Loop` | Loop from end to start |
| `LoopMode::PingPong` | Bounce back and forth |

### Morph

```rust
let morphed = morph(&shape_a, &shape_b, 0.5); // 50% blend
```

---

## File I/O

### Save / Load

| Function | Parameters | Description |
|----------|-----------|-------------|
| `save(&tree, path)` | `&SdfTree, &str` | Auto-detect format by extension |
| `load(path)` | `&str` | Auto-detect format by extension |
| `save_asdf(&tree, path)` | `&SdfTree, &str` | Binary .asdf |
| `load_asdf(path)` | `&str` | Binary .asdf |
| `save_asdf_json(&tree, path)` | `&SdfTree, &str` | JSON .asdf.json |
| `load_asdf_json(path)` | `&str` | JSON .asdf.json |

### Export

| Function | Parameters | Description |
|----------|-----------|-------------|
| `export_obj(&mesh, path, config, mats)` | `&Mesh, &str, &ObjConfig, Option<&MaterialLibrary>` | Wavefront OBJ |
| `import_obj(path, config)` | `&str, &ObjConfig` | Import OBJ |
| `export_glb(&mesh, path, config, mats)` | `&Mesh, &Path, &GltfConfig, Option<&MaterialLibrary>` | glTF 2.0 binary |
| `export_gltf_json(&mesh, path, config, mats)` | Same | glTF 2.0 JSON |
| `export_fbx(&mesh, path, config, mats)` | `&Mesh, &Path, &FbxConfig, Option<&MaterialLibrary>` | FBX 7.4 |

---

## Compiled / VM

### Compilation

```rust
let compiled = CompiledSdf::compile(&shape);
let distance = eval_compiled(&compiled, point);
let normal = eval_compiled_normal(&compiled, point);
```

### SIMD Evaluation

```rust
let distances = eval_compiled_simd(&compiled, vec3x8);  // 8 points at once
let distances = eval_compiled_batch_simd(&compiled, &points);  // Auto-chunked
let distances = eval_compiled_batch_simd_parallel(&compiled, &points);  // + Rayon
```

### SoA Evaluation

```rust
let soa_points = SoAPoints::from_vec3_slice(&points);
let distances = eval_compiled_batch_soa(&compiled, &soa_points);
let distances = eval_compiled_batch_soa_parallel(&compiled, &soa_points);
```

### BVH-Accelerated

```rust
let bvh = CompiledSdfBvh::build(&shape);
let distance = eval_compiled_bvh(&bvh, point);
let aabb = get_scene_aabb(&bvh);
```

---

## Shader Transpilers

### GLSL

```rust
use alice_sdf::compiled::GlslShader;
let shader = GlslShader::transpile(&shape);
let unity_code = shader.to_unity_custom_function();
let frag_code = shader.to_fragment_shader();
let compute_code = shader.to_compute_shader();
```

### HLSL

```rust
use alice_sdf::compiled::{HlslShader, HlslTranspileMode};
let shader = HlslShader::transpile(&shape, HlslTranspileMode::Hardcoded);
let ue5_code = shader.to_ue5_custom_node();
let compute_code = shader.to_compute_shader();
```

### WGSL

```rust
use alice_sdf::compiled::WgslShader;
let shader = WgslShader::transpile(&shape);
println!("{}", shader.source);
```

---

## crispy.rs — Hardware-Native Math Utilities

Branchless operations and data structures for hot inner loops.

### Branchless Arithmetic

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `fast_recip(x)` | `f32` | `f32` | Fast `1/x` (rcpss + Newton-Raphson) |
| `fast_recip_vec3(v)` | `Vec3` | `Vec3` | Component-wise fast reciprocal |
| `fast_inv_sqrt(x)` | `f32` | `f32` | Quake III inverse sqrt (~0.175% error) |
| `fast_normalize_2d(gx, gz)` | `f32, f32` | `(f32, f32)` | Fast 2D normalize via inv_sqrt |
| `select_f32(cond, a, b)` | `bool, f32, f32` | `f32` | Branchless cmov (bit manipulation) |
| `branchless_min(a, b)` | `f32, f32` | `f32` | Min without branching |
| `branchless_max(a, b)` | `f32, f32` | `f32` | Max without branching |
| `branchless_clamp(x, lo, hi)` | `f32, f32, f32` | `f32` | Clamp without branching |
| `branchless_abs(x)` | `f32` | `f32` | Abs via sign-bit clear |

### BitMask64

64-element batch mask for branchless filtering (hardware `popcnt`).

```rust
let a = BitMask64(0b1010);
let b = BitMask64(0b1100);
assert_eq!(a.and(b), BitMask64(0b1000));
assert_eq!(a.count_ones(), 2);
assert!(a.test(1));
```

| Method | Description |
|--------|-------------|
| `and(other)` | Bitwise AND |
| `or(other)` | Bitwise OR |
| `not()` | Bitwise NOT |
| `count_ones()` | Population count (hardware popcnt) |
| `test(index)` | Test bit at index |
| `set(index)` | Set bit at index |
| `clear(index)` | Clear bit at index |
| `is_empty()` | True if no bits set |

### BloomFilter

4KB Bloom filter with FNV-1a double-hashing. O(1) membership test, ~1-2% false positive rate at 200 entries.

```rust
let mut bloom = BloomFilter::new();
bloom.insert(b"Sphere");
bloom.insert(b"Box");
assert!(bloom.test(b"Sphere"));   // true (guaranteed)
assert!(!bloom.test(b"Teapot"));  // false (probabilistic)
```

| Method | Description |
|--------|-------------|
| `BloomFilter::new()` | Create empty 4KB filter |
| `BloomFilter::from_items(iter)` | Build from byte slice iterator |
| `insert(data)` | Insert element |
| `test(data)` | O(1) membership test |
| `test_hash(filter, hash)` | Test using pre-computed hash (hot loop optimization) |
| `fnv1a_hash(data)` | FNV-1a 64-bit hash function |

---

## FFI (C/C++/C#)

### Primitives

| Function | Returns | Description |
|----------|---------|-------------|
| `alice_sdf_sphere(radius)` | `SdfHandle` | Sphere |
| `alice_sdf_box(hx, hy, hz)` | `SdfHandle` | Box |
| `alice_sdf_cylinder(r, h)` | `SdfHandle` | Cylinder |
| `alice_sdf_torus(R, r)` | `SdfHandle` | Torus |
| `alice_sdf_capsule(ax,ay,az,bx,by,bz,r)` | `SdfHandle` | Capsule |
| `alice_sdf_plane(nx,ny,nz,d)` | `SdfHandle` | Plane |

### Operations

| Function | Returns | Description |
|----------|---------|-------------|
| `alice_sdf_union(a, b)` | `SdfHandle` | Union |
| `alice_sdf_intersection(a, b)` | `SdfHandle` | Intersection |
| `alice_sdf_subtract(a, b)` | `SdfHandle` | Subtraction |
| `alice_sdf_smooth_union(a, b, k)` | `SdfHandle` | Smooth union |
| `alice_sdf_smooth_intersection(a, b, k)` | `SdfHandle` | Smooth intersection |
| `alice_sdf_smooth_subtract(a, b, k)` | `SdfHandle` | Smooth subtraction |

### Transforms & Modifiers

| Function | Returns | Description |
|----------|---------|-------------|
| `alice_sdf_translate(n, x,y,z)` | `SdfHandle` | Translate |
| `alice_sdf_rotate(n, qx,qy,qz,qw)` | `SdfHandle` | Rotate (quaternion) |
| `alice_sdf_rotate_euler(n, x,y,z)` | `SdfHandle` | Rotate (Euler) |
| `alice_sdf_scale(n, f)` | `SdfHandle` | Uniform scale |
| `alice_sdf_scale_xyz(n, x,y,z)` | `SdfHandle` | Non-uniform scale |
| `alice_sdf_twist(n, s)` | `SdfHandle` | Twist |
| `alice_sdf_bend(n, c)` | `SdfHandle` | Bend |
| `alice_sdf_repeat(n, sx,sy,sz)` | `SdfHandle` | Repeat |
| `alice_sdf_round(n, r)` | `SdfHandle` | Round |
| `alice_sdf_onion(n, t)` | `SdfHandle` | Onion |

### Evaluation

| Function | Returns | Description |
|----------|---------|-------------|
| `alice_sdf_eval(n, x,y,z)` | `float` | Single point (slow) |
| `alice_sdf_eval_compiled(c, x,y,z)` | `float` | Single compiled point |
| `alice_sdf_eval_batch(n, pts, dst, cnt)` | `BatchResult` | AoS batch |
| `alice_sdf_eval_compiled_batch(c, pts, dst, cnt)` | `BatchResult` | AoS compiled batch |
| `alice_sdf_eval_soa(c, x,y,z, dst, cnt)` | `BatchResult` | SoA batch (fastest) |
| `alice_sdf_eval_gradient_soa(c, x,y,z, nx,ny,nz, d, cnt)` | `BatchResult` | SoA gradient+distance |

### Compilation

| Function | Returns | Description |
|----------|---------|-------------|
| `alice_sdf_compile(n)` | `CompiledHandle` | Compile to bytecode |
| `alice_sdf_free_compiled(c)` | `void` | Free compiled handle |
| `alice_sdf_compiled_instruction_count(c)` | `uint32_t` | Instruction count |

### Shader Generation

| Function | Returns | Description |
|----------|---------|-------------|
| `alice_sdf_to_glsl(n)` | `StringResult` | GLSL code |
| `alice_sdf_to_hlsl(n)` | `StringResult` | HLSL code |
| `alice_sdf_to_wgsl(n)` | `StringResult` | WGSL code |

### File I/O

| Function | Returns | Description |
|----------|---------|-------------|
| `alice_sdf_save(n, path)` | `SdfResult` | Save to file |
| `alice_sdf_load(path)` | `SdfHandle` | Load from file |

### Memory Management

| Function | Description |
|----------|-------------|
| `alice_sdf_free(n)` | Free SDF handle |
| `alice_sdf_free_compiled(c)` | Free compiled handle |
| `alice_sdf_free_string(s)` | Free string from shader generation |
| `alice_sdf_clone(n)` | Clone a handle |

---

Author: Moroya Sakamoto
