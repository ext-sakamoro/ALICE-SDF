# ALICE-SDF Python Guide

Python integration for SDF authoring, batch evaluation, mesh export, and Blender workflows.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Primitives](#primitives)
- [CSG Operations](#csg-operations)
- [Transforms and Modifiers](#transforms-and-modifiers)
- [Batch Evaluation (NumPy)](#batch-evaluation-numpy)
- [Mesh Generation](#mesh-generation)
- [File I/O](#file-io)
- [Shader Export](#shader-export)
- [Compiled Evaluation](#compiled-evaluation)
- [Blender Integration](#blender-integration)
- [Examples](#examples)
- [API Reference](#api-reference)

---

## Installation

### From PyPI

```bash
pip install alice-sdf
```

### From Source (Development)

```bash
cd ALICE-SDF
pip install maturin
maturin develop --features python
```

### Verify Installation

```python
import alice_sdf as sdf
sphere = sdf.SdfNode.sphere(1.0)
print(sphere.eval(0.0, 0.0, 0.0))  # -1.0
```

---

## Basic Usage

```python
import alice_sdf as sdf

# Create a sphere
sphere = sdf.SdfNode.sphere(1.0)

# Evaluate distance at a point
d = sphere.eval(0.0, 0.0, 0.0)
print(f"Distance at origin: {d}")  # -1.0 (inside)

d = sphere.eval(1.5, 0.0, 0.0)
print(f"Distance at (1.5,0,0): {d}")  # 0.5 (outside)
```

---

## Primitives

```python
import alice_sdf as sdf

# Sphere (radius)
sphere = sdf.SdfNode.sphere(1.0)

# Box (half-extents: width, height, depth)
box = sdf.SdfNode.box3d(1.0, 0.5, 0.5)

# Cylinder (radius, half-height)
cylinder = sdf.SdfNode.cylinder(0.5, 1.0)

# Torus (major radius, minor radius)
torus = sdf.SdfNode.torus(1.0, 0.3)

# Capsule (point A, point B, radius)
capsule = sdf.SdfNode.capsule(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.3)
```

---

## CSG Operations

```python
import alice_sdf as sdf

a = sdf.SdfNode.sphere(1.0)
b = sdf.SdfNode.box3d(0.7, 0.7, 0.7)

# Boolean union (A + B)
union = a.union(b)

# Boolean intersection (A & B)
intersection = a.intersection(b)

# Boolean subtraction (A - B)
subtraction = a.subtract(b)

# Smooth variants (k = blending radius)
smooth_union = a.smooth_union(b, 0.2)
smooth_inter = a.smooth_intersection(b, 0.2)
smooth_sub = a.smooth_subtract(b, 0.2)
```

---

## Transforms and Modifiers

```python
import alice_sdf as sdf

shape = sdf.SdfNode.sphere(1.0)

# Translate (x, y, z)
moved = shape.translate(2.0, 0.0, 0.0)

# Rotate (euler angles in radians: x, y, z)
rotated = shape.rotate(0.0, 1.57, 0.0)  # 90 degrees around Y

# Uniform scale
scaled = shape.scale(2.0)

# Twist (strength: radians per unit along Y axis)
twisted = shape.twist(1.0)

# Bend (curvature)
bent = shape.bend(0.5)

# Infinite repetition (spacing x, y, z)
repeated = shape.repeat(3.0, 3.0, 3.0)

# Perlin noise (amplitude, frequency, seed)
noisy = shape.noise(0.1, 2.0, 42)

# Round edges (radius)
rounded = shape.round(0.1)

# Shell / Onion (thickness)
shell = shape.onion(0.05)
```

---

## Batch Evaluation (NumPy)

For evaluating thousands to millions of points efficiently:

```python
import alice_sdf as sdf
import numpy as np

shape = sdf.SdfNode.sphere(1.0).subtract(sdf.SdfNode.box3d(0.5, 0.5, 0.5))

# Create point grid
N = 100
x = np.linspace(-2, 2, N, dtype=np.float32)
xx, yy, zz = np.meshgrid(x, x, x)
points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
# Shape: (1000000, 3), dtype: float32

# Batch evaluate (uses parallel Rayon internally)
distances = sdf.eval_batch(shape, points)
# Shape: (1000000,), dtype: float32

# Find surface points
surface_mask = np.abs(distances) < 0.05
surface_points = points[surface_mask]
print(f"Surface points: {len(surface_points)}")
```

### Visualization with Matplotlib

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot surface points
ax.scatter(
    surface_points[::10, 0],
    surface_points[::10, 1],
    surface_points[::10, 2],
    s=1, alpha=0.5
)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('SDF Surface Visualization')
plt.show()
```

### Slice Visualization

```python
import matplotlib.pyplot as plt

# 2D slice at z=0
N = 256
x = np.linspace(-2, 2, N, dtype=np.float32)
xx, yy = np.meshgrid(x, x)
points_2d = np.stack([xx.ravel(), yy.ravel(), np.zeros(N*N, dtype=np.float32)], axis=1)

distances_2d = sdf.eval_batch(shape, points_2d).reshape(N, N)

plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, distances_2d, levels=50, cmap='RdBu')
plt.contour(xx, yy, distances_2d, levels=[0], colors='black', linewidths=2)
plt.colorbar(label='Signed Distance')
plt.title('SDF Cross-Section (Z=0)')
plt.axis('equal')
plt.show()
```

---

## Mesh Generation

```python
import alice_sdf as sdf

shape = sdf.SdfNode.sphere(1.0).smooth_union(sdf.SdfNode.box3d(0.5, 0.5, 0.5), 0.2)

# Generate mesh
vertices, indices = sdf.to_mesh(
    shape,
    bounds_min=(-2.0, -2.0, -2.0),
    bounds_max=(2.0, 2.0, 2.0),
    resolution=64
)

print(f"Vertices: {vertices.shape}")  # (N, 3)
print(f"Indices: {indices.shape}")    # (M,) - flat triangle indices

# Convert to triangle faces
faces = indices.reshape(-1, 3)
print(f"Triangles: {faces.shape}")    # (T, 3)
```

### Export to OBJ (Manual)

```python
def save_obj(vertices, indices, filename):
    faces = indices.reshape(-1, 3)
    with open(filename, 'w') as f:
        f.write("# ALICE-SDF generated mesh\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

save_obj(vertices, indices, "output.obj")
```

### Export to STL (for 3D Printing)

```python
import struct

def save_stl_binary(vertices, indices, filename):
    faces = indices.reshape(-1, 3)
    with open(filename, 'wb') as f:
        f.write(b'\x00' * 80)  # Header
        f.write(struct.pack('<I', len(faces)))
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))

save_stl_binary(vertices, indices, "output.stl")
```

---

## File I/O

```python
import alice_sdf as sdf

shape = sdf.SdfNode.sphere(1.0).twist(0.5)

# Save to .asdf (binary) or .asdf.json (human-readable)
sdf.save_sdf(shape, "my_shape.asdf")
sdf.save_sdf(shape, "my_shape.asdf.json")

# Load
loaded = sdf.load_sdf("my_shape.asdf")
d = loaded.eval(0.0, 0.0, 0.0)
```

---

## Shader Export

Generate shader code from Python for use in game engines.

> **Note**: Shader export requires building with additional feature flags:
> `maturin develop --features "python,glsl,hlsl,gpu"`

```python
import alice_sdf as sdf

shape = sdf.SdfNode.sphere(1.0).smooth_union(sdf.SdfNode.box3d(0.5, 0.5, 0.5), 0.2)

# Export GLSL (Unity, OpenGL) — requires 'glsl' feature
glsl_code = sdf.to_glsl(shape)
with open("sdf_shader.glsl", "w") as f:
    f.write(glsl_code)

# Export HLSL (Unreal Engine, DirectX) — requires 'hlsl' feature
hlsl_code = sdf.to_hlsl(shape)
with open("sdf_shader.hlsl", "w") as f:
    f.write(hlsl_code)

# Export WGSL (WebGPU) — requires 'gpu' feature
wgsl_code = sdf.to_wgsl(shape)
with open("sdf_shader.wgsl", "w") as f:
    f.write(wgsl_code)
```

---

## Compiled Evaluation

For maximum throughput in repeated evaluations:

```python
import alice_sdf as sdf
import numpy as np

shape = sdf.SdfNode.sphere(1.0)

# Compile once
compiled = sdf.compile_sdf(shape)

# Evaluate many times (reuses compiled bytecode)
for frame in range(60):
    points = np.random.randn(100000, 3).astype(np.float32)
    distances = sdf.eval_compiled_batch(compiled, points)
```

---

## Blender Integration

### As a Blender Addon

ALICE-SDF can be used inside Blender via Python scripting.

#### Prerequisites

```bash
# Install alice-sdf into Blender's Python
/path/to/blender/python/bin/python -m pip install alice-sdf
```

#### Import SDF as Blender Mesh

```python
import bpy
import alice_sdf as sdf
import numpy as np

def import_sdf_as_mesh(shape, name="SDF_Object", resolution=64, bounds=2.0):
    """Import an ALICE-SDF shape as a Blender mesh object."""

    # Generate mesh
    vertices, indices = sdf.to_mesh(
        shape,
        bounds_min=(-bounds, -bounds, -bounds),
        bounds_max=(bounds, bounds, bounds),
        resolution=resolution
    )

    faces = indices.reshape(-1, 3).tolist()
    verts = vertices.tolist()

    # Create Blender mesh
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Create object and link to scene
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return obj

# Example: Create a twisted sphere
shape = sdf.SdfNode.sphere(1.0).twist(0.5)
obj = import_sdf_as_mesh(shape, "TwistedSphere", resolution=64)
```

#### Load .asdf File in Blender

```python
import bpy
import alice_sdf as sdf

def load_asdf_file(filepath, resolution=64):
    """Load an .asdf file and create a Blender mesh."""
    shape = sdf.load_sdf(filepath)
    name = filepath.split("/")[-1].replace(".asdf", "")
    return import_sdf_as_mesh(shape, name, resolution)

# Usage from Blender's text editor
load_asdf_file("/path/to/model.asdf", resolution=64)
```

#### Blender Operator (Full Addon)

```python
bl_info = {
    "name": "ALICE-SDF Importer",
    "blender": (4, 0, 0),
    "category": "Import-Export",
    "version": (0, 1, 0),
    "author": "Moroya Sakamoto",
    "description": "Import ALICE-SDF (.asdf) files as mesh objects"
}

import bpy
from bpy_extras.io_utils import ImportHelper
import alice_sdf as sdf

class IMPORT_OT_asdf(bpy.types.Operator, ImportHelper):
    bl_idname = "import_mesh.asdf"
    bl_label = "Import ASDF"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".asdf"
    filter_glob: bpy.props.StringProperty(default="*.asdf;*.asdf.json", options={'HIDDEN'})

    resolution: bpy.props.IntProperty(
        name="Resolution",
        default=64,
        min=8,
        max=256,
        description="Marching cubes grid resolution"
    )

    bounds: bpy.props.FloatProperty(
        name="Bounds",
        default=2.0,
        min=0.1,
        max=100.0,
        description="Half-extent of bounding box"
    )

    def execute(self, context):
        shape = sdf.load_sdf(self.filepath)
        vertices, indices = sdf.to_mesh(
            shape,
            bounds_min=(-self.bounds, -self.bounds, -self.bounds),
            bounds_max=(self.bounds, self.bounds, self.bounds),
            resolution=self.resolution
        )

        name = self.filepath.split("/")[-1].replace(".asdf.json", "").replace(".asdf", "")
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices.tolist(), [], indices.reshape(-1, 3).tolist())
        mesh.update()

        obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        self.report({'INFO'}, f"Imported {name}: {len(vertices)} vertices")
        return {'FINISHED'}

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_asdf.bl_idname, text="ALICE SDF (.asdf)")

def register():
    bpy.utils.register_class(IMPORT_OT_asdf)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(IMPORT_OT_asdf)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()
```

Save this as `alice_sdf_importer.py` in Blender's addons directory, then enable via `Edit > Preferences > Add-ons`.

---

## Examples

### Parametric Shape Generator

```python
import alice_sdf as sdf
import numpy as np

def generate_gear(teeth=12, radius=1.0, tooth_height=0.3, tooth_width=0.15):
    """Generate a gear shape using CSG operations."""
    base = sdf.SdfNode.cylinder(radius, 0.2)

    # Create teeth using repeated boxes
    tooth = sdf.SdfNode.box3d(tooth_width, tooth_height, 0.2)
    tooth = tooth.translate(radius + tooth_height * 0.5, 0.0, 0.0)

    gear = base
    for i in range(teeth):
        angle = 2.0 * np.pi * i / teeth
        rotated_tooth = tooth.rotate(0.0, 0.0, angle)
        gear = gear.union(rotated_tooth)

    return gear

gear = generate_gear(teeth=16)
sdf.save_sdf(gear, "gear.asdf")
```

### SDF Volume Rendering

```python
import alice_sdf as sdf
import numpy as np
import matplotlib.pyplot as plt

shape = sdf.SdfNode.torus(1.0, 0.3).twist(2.0)

# Render slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
N = 256

for idx, (axis, label) in enumerate(zip(axes, ['XY (z=0)', 'XZ (y=0)', 'YZ (x=0)'])):
    coords = np.linspace(-2, 2, N, dtype=np.float32)
    u, v = np.meshgrid(coords, coords)

    if idx == 0:  # XY
        pts = np.stack([u.ravel(), v.ravel(), np.zeros(N*N, dtype=np.float32)], axis=1)
    elif idx == 1:  # XZ
        pts = np.stack([u.ravel(), np.zeros(N*N, dtype=np.float32), v.ravel()], axis=1)
    else:  # YZ
        pts = np.stack([np.zeros(N*N, dtype=np.float32), u.ravel(), v.ravel()], axis=1)

    d = sdf.eval_batch(shape, pts).reshape(N, N)
    axis.contourf(u, v, d, levels=50, cmap='RdBu')
    axis.contour(u, v, d, levels=[0], colors='black', linewidths=2)
    axis.set_title(label)
    axis.set_aspect('equal')

plt.tight_layout()
plt.savefig("sdf_slices.png", dpi=150)
plt.show()
```

---

## API Reference

### `sdf.SdfNode` (Class)

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `sphere(r)` | `float` | `SdfNode` | Sphere primitive |
| `box3d(w, h, d)` | `float, float, float` | `SdfNode` | Box primitive |
| `cylinder(r, h)` | `float, float` | `SdfNode` | Cylinder primitive |
| `torus(R, r)` | `float, float` | `SdfNode` | Torus primitive |
| `capsule(ax,ay,az,bx,by,bz,r)` | `7x float` | `SdfNode` | Capsule primitive |
| `union(other)` | `SdfNode` | `SdfNode` | Boolean union |
| `intersection(other)` | `SdfNode` | `SdfNode` | Boolean intersection |
| `subtract(other)` | `SdfNode` | `SdfNode` | Boolean subtraction |
| `smooth_union(other, k)` | `SdfNode, float` | `SdfNode` | Smooth union |
| `smooth_intersection(other, k)` | `SdfNode, float` | `SdfNode` | Smooth intersection |
| `smooth_subtract(other, k)` | `SdfNode, float` | `SdfNode` | Smooth subtraction |
| `translate(x, y, z)` | `float, float, float` | `SdfNode` | Translation |
| `rotate(x, y, z)` | `float, float, float` | `SdfNode` | Euler rotation (radians) |
| `scale(f)` | `float` | `SdfNode` | Uniform scale |
| `twist(s)` | `float` | `SdfNode` | Twist modifier |
| `bend(c)` | `float` | `SdfNode` | Bend modifier |
| `repeat(sx, sy, sz)` | `float, float, float` | `SdfNode` | Infinite repetition |
| `noise(amp, freq, seed)` | `float, float, int` | `SdfNode` | Perlin noise |
| `round(r)` | `float` | `SdfNode` | Round edges |
| `onion(t)` | `float` | `SdfNode` | Shell modifier |
| `eval(x, y, z)` | `float, float, float` | `float` | Evaluate distance |
| `node_count()` | - | `int` | Count nodes in tree |

### Module Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `eval_batch(node, points)` | `SdfNode, ndarray(N,3)` | `ndarray(N,)` | Parallel batch evaluation |
| `compile_sdf(node)` | `SdfNode` | `CompiledSdf` | Compile for fast evaluation |
| `eval_compiled_batch(compiled, points)` | `CompiledSdf, ndarray(N,3)` | `ndarray(N,)` | Evaluate compiled SDF in batch |
| `to_mesh(node, min, max, res)` | `SdfNode, tuple, tuple, int` | `(ndarray, ndarray)` | Marching cubes |
| `to_mesh_adaptive(node, ...)` | `SdfNode, bounds, depths, threshold` | `(ndarray, ndarray)` | Adaptive mesh generation |
| `save_sdf(node, path)` | `SdfNode, str` | `None` | Save to .asdf/.asdf.json |
| `load_sdf(path)` | `str` | `SdfNode` | Load from file |

---

Author: Moroya Sakamoto
