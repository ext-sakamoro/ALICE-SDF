# ALICE-SDF Godot Integration Guide

Integrate ALICE-SDF with Godot Engine (4.x) via glTF import and GDExtension FFI.

## Table of Contents

- [Overview](#overview)
- [glTF Pipeline (Recommended)](#gltf-pipeline-recommended)
- [GDExtension FFI (Real-time)](#gdextension-ffi-real-time)
- [Visual Shader Integration](#visual-shader-integration)
- [Collision Setup](#collision-setup)
- [LOD Configuration](#lod-configuration)
- [Examples](#examples)

---

## Overview

Two integration paths:

| Approach | Use Case | Complexity | Performance |
|----------|----------|------------|-------------|
| **glTF Pipeline** | Static meshes, level geometry | Simple | N/A (offline) |
| **GDExtension FFI** | Real-time SDF evaluation, particles, procedural | Advanced | High |

---

## glTF Pipeline (Recommended)

### 1. Generate Mesh from SDF

```bash
# Using CLI
cd ALICE-SDF
cargo run --release -- to-mesh input.asdf -o model.glb --resolution 64

# Or from Rust
use alice_sdf::prelude::*;

let shape = SdfNode::sphere(1.0)
    .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);

let mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0),
    &MarchingCubesConfig::aaa(64));

let mut mat_lib = MaterialLibrary::new();
mat_lib.add(Material::metal("Chrome", 0.9, 0.9, 0.9, 0.2));

export_glb(&mesh, "model.glb", &GltfConfig::aaa(), Some(&mat_lib)).unwrap();
```

### 2. Import in Godot

1. Copy `model.glb` to your Godot project's `res://models/` directory
2. Godot auto-imports glTF 2.0 files
3. Drag the imported scene into your 3D scene

### 3. Material Mapping

ALICE-SDF's PBR materials map directly to Godot's `StandardMaterial3D`:

| ALICE-SDF Property | Godot Property |
|-------------------|----------------|
| `base_color` | `albedo_color` |
| `metallic` | `metallic` |
| `roughness` | `roughness` |
| `emission` | `emission` |
| `emission_strength` | `emission_energy_multiplier` |
| `normal_scale` | `normal_scale` |
| `opacity` | `albedo_color.a` + Transparency mode |

### 4. Batch Export with LOD

```rust
use alice_sdf::prelude::*;

let shape = SdfNode::sphere(1.0);

// Generate LOD chain (4 levels)
let lod_chain = generate_lod_chain_decimated(
    &shape,
    Vec3::splat(-2.0),
    Vec3::splat(2.0),
    &DecimationLodConfig {
        num_levels: 4,
        base_resolution: 64,
        decimation_ratio: 0.5,
        ..Default::default()
    },
);

// Export each LOD level
for (i, level) in lod_chain.levels.iter().enumerate() {
    let filename = format!("model_lod{}.glb", i);
    export_glb(&level.mesh, &filename, &GltfConfig::aaa(), None).unwrap();
}
```

In Godot, use `GeometryInstance3D.lod_bias` or `VisibilityRange` to configure LOD switching.

---

## GDExtension FFI (Real-time)

### 1. Build Shared Library

```bash
cd ALICE-SDF
cargo build --release --features ffi

# Output:
# macOS:   target/release/libalice_sdf.dylib
# Windows: target/release/alice_sdf.dll
# Linux:   target/release/libalice_sdf.so
```

### 2. Create GDExtension Configuration

Create `alice_sdf.gdextension` in your Godot project root:

```ini
[configuration]
entry_symbol = "alice_sdf_version"
compatibility_minimum = "4.2"

[libraries]
macos.release = "res://bin/libalice_sdf.dylib"
windows.release.x86_64 = "res://bin/alice_sdf.dll"
linux.release.x86_64 = "res://bin/libalice_sdf.so"
```

### 3. Copy Library

```bash
mkdir -p your_godot_project/bin/

# macOS
cp target/release/libalice_sdf.dylib your_godot_project/bin/

# Windows
copy target\release\alice_sdf.dll your_godot_project\bin\

# Linux
cp target/release/libalice_sdf.so your_godot_project/bin/
```

### 4. GDScript Wrapper

Create `res://scripts/alice_sdf.gd`:

```gdscript
class_name AliceSDF
extends RefCounted

## ALICE-SDF GDScript wrapper using OS.execute for CLI operations
## For high-performance real-time use, create a C++ GDExtension wrapper

static func eval_at_point(asdf_path: String, point: Vector3) -> float:
    """Evaluate SDF distance at a point using CLI"""
    var output := []
    var exit_code := OS.execute("alice-sdf", [
        "eval", asdf_path,
        str(point.x), str(point.y), str(point.z)
    ], output)
    if exit_code == 0 and output.size() > 0:
        return float(output[0])
    return INF

static func generate_mesh(asdf_path: String, output_path: String,
                          resolution: int = 64, bounds: float = 2.0) -> bool:
    """Generate mesh from SDF file"""
    var exit_code := OS.execute("alice-sdf", [
        "to-mesh", asdf_path,
        "-o", output_path,
        "--resolution", str(resolution),
        "--bounds", str(bounds)
    ])
    return exit_code == 0
```

### 5. C++ GDExtension Wrapper (High Performance)

For real-time evaluation, create a C++ GDExtension that links `alice_sdf`:

```cpp
// alice_sdf_node.h
#ifndef ALICE_SDF_NODE_H
#define ALICE_SDF_NODE_H

#include <godot_cpp/classes/node3d.hpp>
#include "alice_sdf.h"

namespace godot {

class AliceSdfNode : public Node3D {
    GDCLASS(AliceSdfNode, Node3D)

private:
    SdfHandle sdf_handle = nullptr;
    CompiledHandle compiled_handle = nullptr;

protected:
    static void _bind_methods();

public:
    AliceSdfNode();
    ~AliceSdfNode();

    void create_sphere(float radius);
    void create_box(Vector3 half_extents);
    void smooth_union_with(Ref<AliceSdfNode> other, float k);

    float eval_distance(Vector3 point);
    Vector3 eval_normal(Vector3 point);

    void compile();
    void load_from_file(String path);
    void save_to_file(String path);

    String generate_glsl();
};

}

#endif
```

```cpp
// alice_sdf_node.cpp
#include "alice_sdf_node.h"

using namespace godot;

void AliceSdfNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("create_sphere", "radius"), &AliceSdfNode::create_sphere);
    ClassDB::bind_method(D_METHOD("create_box", "half_extents"), &AliceSdfNode::create_box);
    ClassDB::bind_method(D_METHOD("eval_distance", "point"), &AliceSdfNode::eval_distance);
    ClassDB::bind_method(D_METHOD("eval_normal", "point"), &AliceSdfNode::eval_normal);
    ClassDB::bind_method(D_METHOD("compile"), &AliceSdfNode::compile);
    ClassDB::bind_method(D_METHOD("load_from_file", "path"), &AliceSdfNode::load_from_file);
    ClassDB::bind_method(D_METHOD("save_to_file", "path"), &AliceSdfNode::save_to_file);
    ClassDB::bind_method(D_METHOD("generate_glsl"), &AliceSdfNode::generate_glsl);
}

AliceSdfNode::AliceSdfNode() {}

AliceSdfNode::~AliceSdfNode() {
    if (compiled_handle) alice_sdf_free_compiled(compiled_handle);
    if (sdf_handle) alice_sdf_free(sdf_handle);
}

void AliceSdfNode::create_sphere(float radius) {
    if (sdf_handle) alice_sdf_free(sdf_handle);
    sdf_handle = alice_sdf_sphere(radius);
}

void AliceSdfNode::create_box(Vector3 half_extents) {
    if (sdf_handle) alice_sdf_free(sdf_handle);
    sdf_handle = alice_sdf_box(half_extents.x, half_extents.y, half_extents.z);
}

float AliceSdfNode::eval_distance(Vector3 point) {
    if (compiled_handle) {
        return alice_sdf_eval_compiled(compiled_handle, point.x, point.y, point.z);
    }
    if (sdf_handle) {
        return alice_sdf_eval(sdf_handle, point.x, point.y, point.z);
    }
    return INFINITY;
}

void AliceSdfNode::compile() {
    if (compiled_handle) alice_sdf_free_compiled(compiled_handle);
    if (sdf_handle) {
        compiled_handle = alice_sdf_compile(sdf_handle);
    }
}

void AliceSdfNode::load_from_file(String path) {
    if (sdf_handle) alice_sdf_free(sdf_handle);
    CharString utf8 = path.utf8();
    sdf_handle = alice_sdf_load(utf8.get_data());
}

String AliceSdfNode::generate_glsl() {
    if (!sdf_handle) return "";
    StringResult result = alice_sdf_to_glsl(sdf_handle);
    if (result.result == SdfResult_Ok) {
        String code = String::utf8(result.data, result.len);
        alice_sdf_free_string(result.data);
        return code;
    }
    return "";
}
```

### 6. Usage in GDScript

```gdscript
extends Node3D

var sdf_node: AliceSdfNode

func _ready():
    sdf_node = AliceSdfNode.new()
    add_child(sdf_node)

    sdf_node.create_sphere(1.0)
    sdf_node.compile()

func _physics_process(delta):
    # SDF collision detection
    var player_pos = $Player.global_position
    var distance = sdf_node.eval_distance(player_pos)

    if distance < 0.0:
        # Player is inside SDF - push out
        var normal = sdf_node.eval_normal(player_pos)
        $Player.global_position += normal * abs(distance)
```

---

## Visual Shader Integration

### Generate GLSL for Godot Shader

```rust
use alice_sdf::compiled::GlslShader;

let shape = SdfNode::sphere(1.0).twist(0.5);
let shader = GlslShader::transpile(&shape);
let glsl_code = shader.to_fragment_shader();
```

### Godot Visual Shader

1. Create a new `ShaderMaterial`
2. In the shader code, paste the generated GLSL `sdf_eval()` function
3. Use for raymarching or distance-based effects:

```glsl
shader_type spatial;

// ---- Paste generated SDF function here ----
float sdf_eval(vec3 p) {
    // ... generated code ...
}

void fragment() {
    vec3 world_pos = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
    float d = sdf_eval(world_pos);

    // Distance-based coloring
    ALBEDO = mix(vec3(0.2, 0.4, 0.8), vec3(0.8, 0.2, 0.1), step(0.0, d));
    ALPHA = smoothstep(0.1, 0.0, abs(d));
}
```

---

## Collision Setup

### Static Collision (Mesh-based)

Export mesh with collision shapes:

```rust
use alice_sdf::prelude::*;

let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));
let mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0),
    &MarchingCubesConfig::aaa(32));

// Generate convex decomposition
let decomp = convex_decomposition(&mesh, &VhacdConfig {
    max_hulls: 8,
    max_vertices_per_hull: 32,
    ..Default::default()
});

// Export main mesh
export_glb(&mesh, "model.glb", &GltfConfig::aaa(), None).unwrap();

// Export collision hulls as separate meshes
for (i, hull) in decomp.hulls.iter().enumerate() {
    export_obj(&hull_mesh, &format!("collision_{}.obj", i),
        &ObjConfig::default(), None).unwrap();
}
```

In Godot:
1. Import `model.glb` as MeshInstance3D
2. Add `StaticBody3D` with `CollisionShape3D` children
3. Use `ConvexPolygonShape3D` with imported collision mesh vertices

### Dynamic SDF Collision

```gdscript
# Real-time SDF collision (requires GDExtension)
func check_sdf_collision(body_position: Vector3) -> Dictionary:
    var distance = sdf_node.eval_distance(body_position)
    var normal = sdf_node.eval_normal(body_position)

    return {
        "colliding": distance < 0.0,
        "distance": distance,
        "normal": normal,
        "push_vector": normal * max(0.0, -distance)
    }
```

---

## LOD Configuration

### Import LOD Meshes

```gdscript
# Load LOD chain
var lod_meshes := [
    preload("res://models/model_lod0.glb"),  # Full detail
    preload("res://models/model_lod1.glb"),  # 50% triangles
    preload("res://models/model_lod2.glb"),  # 25% triangles
    preload("res://models/model_lod3.glb"),  # 12.5% triangles
]

func _ready():
    for i in range(lod_meshes.size()):
        var instance = lod_meshes[i].instantiate()
        instance.visibility_range_begin = i * 20.0
        instance.visibility_range_end = (i + 1) * 20.0 if i < 3 else 0.0
        add_child(instance)
```

### Godot 4.x LOD (Built-in)

Godot 4 supports automatic mesh LOD via `ImporterMesh`:

1. Import high-res glTF
2. In Import settings, enable "Generate LODs"
3. Set LOD bias in `GeometryInstance3D` properties

---

## Examples

### Procedural Dungeon Generator

```gdscript
extends Node3D

@export var room_count: int = 10
@export var room_size: float = 5.0

func _ready():
    var sdf = AliceSdfNode.new()
    add_child(sdf)

    # Generate rooms as SDF
    # (requires GDExtension wrapper)
    sdf.create_box(Vector3(room_size, 3.0, room_size))

    for i in range(1, room_count):
        var offset = Vector3(
            randf_range(-20, 20),
            0,
            randf_range(-20, 20)
        )
        var room = AliceSdfNode.new()
        room.create_box(Vector3(room_size, 3.0, room_size))
        # Connect rooms with corridors...

    # Export to glTF and reimport
    sdf.save_to_file("user://dungeon.asdf")
```

### SDF-Based Particle Attractor

```gdscript
extends GPUParticles3D

var sdf: AliceSdfNode

func _ready():
    sdf = AliceSdfNode.new()
    sdf.create_sphere(2.0)
    sdf.compile()

func _process(delta):
    # Generate GLSL for particle shader
    var glsl = sdf.generate_glsl()
    # Apply to particle process material
```

---

## Related Documentation

- [QUICKSTART](QUICKSTART.md) - Getting started
- [API Reference](API_REFERENCE.md) - Complete function list
- [Architecture](ARCHITECTURE.md) - 13-layer design

---

Author: Moroya Sakamoto
