# ALICE-SDF Godot 4.x GDExtension

Native Godot 4.x integration for ALICE-SDF via GDExtension. Provides SDF modeling, mesh generation, collision shapes, LOD management, and batch evaluation directly from GDScript.

Author: Moroya Sakamoto

## Building

```bash
cargo build --release --features godot
```

This produces the GDExtension dynamic library:

| Platform | Output |
|----------|--------|
| macOS (ARM64) | `target/release/libalice_sdf.dylib` |
| macOS (Intel) | `target/release/libalice_sdf.dylib` |
| Windows x64 | `target/release/alice_sdf.dll` |
| Linux x64 | `target/release/libalice_sdf.so` |

## Installation

1. Copy the dynamic library to your Godot project's `addons/alice_sdf/bin/` directory
2. Create an `alice_sdf.gdextension` file in your project:

```ini
[configuration]
entry_symbol = "gdext_rust_init"
compatibility_minimum = 4.1

[libraries]
macos.debug = "res://addons/alice_sdf/bin/libalice_sdf.dylib"
macos.release = "res://addons/alice_sdf/bin/libalice_sdf.dylib"
windows.debug.x86_64 = "res://addons/alice_sdf/bin/alice_sdf.dll"
windows.release.x86_64 = "res://addons/alice_sdf/bin/alice_sdf.dll"
linux.debug.x86_64 = "res://addons/alice_sdf/bin/libalice_sdf.so"
linux.release.x86_64 = "res://addons/alice_sdf/bin/libalice_sdf.so"
```

3. Restart Godot. The ALICE-SDF nodes appear in the Create Node dialog.

## Quick Start

```gdscript
extends Node3D

func _ready():
    # Create SDF shape
    var sdf = AliceSdfNode.new()
    sdf.shape = "sphere"
    sdf.radius = 1.0
    sdf.build()

    # Create another shape and subtract
    var box = AliceSdfNode.new()
    box.shape = "box"
    box.half_extents = Vector3(0.5, 0.5, 0.5)
    box.build()

    sdf.boolean_subtract(box)

    # Generate mesh
    var mesh_inst = AliceSdfMeshInstance.new()
    mesh_inst.resolution = 64
    mesh_inst.bounds_size = 2.0
    mesh_inst.set_sdf_from_node(sdf)
    add_child(mesh_inst)

    # Evaluate distance at a point
    var dist = sdf.eval_distance(Vector3(0.5, 0, 0))
    print("Distance: ", dist)
```

## Nodes

### AliceSdfNode (Node3D)

The core SDF builder node. Create shapes, apply boolean operations and modifiers.

**Properties:**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| shape | String | "sphere" | Shape type (see 72 available shapes) |
| radius | float | 1.0 | Radius for spherical shapes |
| half_extents | Vector3 | (0.5, 0.5, 0.5) | Half-extents for box-like shapes |
| height | float | 2.0 | Height for cylindrical shapes |
| smooth_k | float | 0.0 | Smoothing factor for boolean ops (0 = sharp) |

**Methods:**

| Method | Description |
|--------|-------------|
| `build() -> bool` | Build the SDF from current properties |
| `boolean_union(other)` | Union with another AliceSdfNode |
| `boolean_subtract(other)` | Subtract another AliceSdfNode |
| `boolean_intersect(other)` | Intersect with another AliceSdfNode |
| `eval_distance(point) -> float` | Evaluate SDF distance at a world point |
| `to_glsl() -> String` | Generate GLSL shader code |

**v1.1.0 Modifier Methods:**

| Method | Parameters | Description |
|--------|-----------|-------------|
| `apply_twist(strength)` | float | Twist around Y axis |
| `apply_bend(curvature)` | float | Bend around Y axis |
| `apply_mirror(axis)` | Vector3 | Mirror across axis (1.0 = mirror, 0.0 = no mirror) |
| `apply_onion(thickness)` | float | Create hollow shell |
| `apply_round(radius)` | float | Round all edges |
| `apply_icosahedral_symmetry()` | -- | 120-fold icosahedral symmetry |
| `apply_surface_roughness(freq, amp, octaves)` | float, float, int | FBM noise surface roughness |
| `apply_noise(amp, freq, seed)` | float, float, int | Noise displacement |
| `apply_polar_repeat(count)` | int | Polar array around Y axis |
| `apply_octant_mirror()` | -- | 48-fold octahedral symmetry |

### AliceSdfResource (Resource)

Serializable SDF resource for the Godot inspector. Stores shape type and JSON parameters.

```gdscript
var res = AliceSdfResource.new()
res.set_shape("gyroid", '{"scale": 2.0, "thickness": 0.1}')
res.build_sdf()
```

### AliceSdfMeshInstance (MeshInstance3D)

Auto-generates a mesh from an SDF using Marching Cubes.

```gdscript
var mesh_inst = AliceSdfMeshInstance.new()
mesh_inst.resolution = 64        # Marching Cubes resolution
mesh_inst.bounds_size = 2.0      # Half-extent of evaluation volume
mesh_inst.auto_rebuild = true    # Rebuild when SDF changes
mesh_inst.set_sdf_from_node(sdf_node)
add_child(mesh_inst)
```

### AliceSdfCollisionGenerator (Node3D)

Generates ConvexPolygonShape3D collision shapes from SDF meshes.

```gdscript
var collider = AliceSdfCollisionGenerator.new()
collider.collision_resolution = 16    # Lower than visual for performance
collider.max_collision_vertices = 256

var collision_shape = collider.generate_collision_shape(sdf_node)
if collision_shape:
    $StaticBody3D.add_child(collision_shape)
```

### AliceSdfLodManager (Node3D)

Camera-distance LOD switching for SDF meshes.

```gdscript
var lod = AliceSdfLodManager.new()
lod.num_lod_levels = 3
lod.base_resolution = 64      # LOD 0 = 64, LOD 1 = 32, LOD 2 = 16
lod.lod0_distance = 10.0
lod.distance_multiplier = 2.0
lod.set_sdf(sdf_node)

# Call in _process() to update LOD
func _process(_delta):
    var camera_pos = get_viewport().get_camera_3d().global_position
    var level = lod.update_lod(camera_pos)
    var mesh = lod.get_lod_mesh(level)
    if mesh:
        $MeshInstance3D.mesh = mesh
```

### AliceSdfBatchEvaluator (RefCounted)

High-performance batch SDF evaluation using compiled SIMD + multi-threading.

```gdscript
var evaluator = AliceSdfBatchEvaluator.new()
evaluator.compile(sdf_node)

# Evaluate 1000 points at once
var points = PackedVector3Array()
for i in range(1000):
    points.append(Vector3(randf() * 4 - 2, randf() * 4 - 2, randf() * 4 - 2))

var distances = evaluator.eval_batch(points)
# distances[i] = SDF distance at points[i]
```

## v1.1.0 Modifier Examples

```gdscript
# Create a sphere and apply modifiers
var sdf = AliceSdfNode.new()
sdf.shape = "sphere"
sdf.radius = 1.0
sdf.build()

# Twist it
sdf.apply_twist(3.0)

# Add surface roughness (frequency, amplitude, octaves)
sdf.apply_surface_roughness(5.0, 0.05, 4)

# Apply icosahedral symmetry for a gem-like shape
var gem = AliceSdfNode.new()
gem.shape = "sphere"
gem.radius = 1.0
gem.build()
gem.apply_icosahedral_symmetry()

# Polar repeat a cylinder 8 times around Y axis
var col = AliceSdfNode.new()
col.shape = "cylinder"
col.radius = 0.1
col.height = 2.0
col.build()
col.apply_polar_repeat(8)
```

## Available Shapes (72)

All shapes from `AliceSdfResource.get_available_shapes()`:

**Basic:** sphere, box, cylinder, torus, plane, capsule, cone, ellipsoid, rounded_cone, pyramid

**Platonic:** octahedron, hex_prism, tetrahedron, dodecahedron, icosahedron, truncated_octahedron, truncated_icosahedron

**TPMS:** gyroid, schwarz_p, diamond_surface, neovius, lidinoid, iwp, frd, fischer_koch_s, pmy

**Advanced:** heart, tube, barrel, diamond, egg, moon, cross_shape, blobby_cross, superellipsoid, star_polygon, stairs, helix, box_frame

**2D Primitives:** circle_2d, rect_2d, segment_2d, polygon_2d, rounded_rect_2d, annular_2d

See `get_available_shapes()` for the full list.
