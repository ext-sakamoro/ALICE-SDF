# ALICE-SDF VFX Cookbook

Recipes for common VFX shapes using ALICE-SDF's existing primitives, operations, and modifiers. Each recipe shows the combination technique in Rust, GDScript (Godot), C# (Unity), and HLSL (UE5/VRChat).

Author: Moroya Sakamoto

---

## 1. Metaball Blob Cluster

Organic fluid-like blobs merging together. Chain `SmoothUnion` with a high `k` value.

**Technique:** Multiple spheres + SmoothUnion chain

### Rust

```rust
use alice_sdf::SdfNode;

let blob = SdfNode::sphere(0.4)
    .translate(0.0, 0.0, 0.0)
    .smooth_union(SdfNode::sphere(0.35).translate(0.6, 0.1, 0.0), 0.3)
    .smooth_union(SdfNode::sphere(0.3).translate(0.2, 0.5, -0.2), 0.3)
    .smooth_union(SdfNode::sphere(0.38).translate(-0.3, 0.2, 0.4), 0.3)
    .smooth_union(SdfNode::sphere(0.25).translate(0.1, -0.4, 0.3), 0.3);
```

### GDScript

```gdscript
var s1 = AliceSdfNode.new()
s1.shape = "sphere"; s1.radius = 0.4; s1.build()

var s2 = AliceSdfNode.new()
s2.shape = "sphere"; s2.radius = 0.35; s2.build()
s2.position = Vector3(0.6, 0.1, 0.0)

var s3 = AliceSdfNode.new()
s3.shape = "sphere"; s3.radius = 0.3; s3.build()
s3.position = Vector3(0.2, 0.5, -0.2)

s1.smooth_k = 0.3
s1.boolean_union(s2)  # smooth union when smooth_k > 0
s1.boolean_union(s3)
```

### C# (Unity)

```csharp
var blob = AliceSdf.SmoothUnion(
    AliceSdf.SmoothUnion(
        AliceSdf.SmoothUnion(
            AliceSdf.Translate(AliceSdf.Sphere(0.4f), 0, 0, 0),
            AliceSdf.Translate(AliceSdf.Sphere(0.35f), 0.6f, 0.1f, 0),
            0.3f),
        AliceSdf.Translate(AliceSdf.Sphere(0.3f), 0.2f, 0.5f, -0.2f),
        0.3f),
    AliceSdf.Translate(AliceSdf.Sphere(0.38f), -0.3f, 0.2f, 0.4f),
    0.3f);
```

### HLSL

```hlsl
float metaball(float3 p) {
    float d = sdSphere(p, 0.4);
    d = opSmoothUnion(d, sdSphere(p - float3(0.6, 0.1, 0.0), 0.35), 0.3);
    d = opSmoothUnion(d, sdSphere(p - float3(0.2, 0.5, -0.2), 0.30), 0.3);
    d = opSmoothUnion(d, sdSphere(p - float3(-0.3, 0.2, 0.4), 0.38), 0.3);
    return d;
}
```

**Tips:**
- `k = 0.3` gives a gooey merge. Lower (`0.05`) for subtle blending, higher (`0.5+`) for very soft blobs.
- Animate sphere positions over time for lava lamp / fluid effects.
- Add `SurfaceRoughness(2.0, 0.02, 3)` for a bubbly membrane texture.

---

## 2. Crystal Gemstone

Faceted gem with sharp reflective faces. Use icosahedral symmetry on a sphere, then intersect with a box to create a flat bottom.

**Technique:** Sphere + IcosahedralSymmetry + Intersection (flat base)

### Rust

```rust
let gem = SdfNode::sphere(1.0)
    .icosahedral_symmetry()
    .intersection(SdfNode::box3d(1.5, 0.8, 1.5).translate(0.0, 0.1, 0.0));
```

### GDScript

```gdscript
var gem = AliceSdfNode.new()
gem.shape = "sphere"; gem.radius = 1.0; gem.build()
gem.apply_icosahedral_symmetry()

var base = AliceSdfNode.new()
base.shape = "box"; base.half_extents = Vector3(1.5, 0.8, 1.5); base.build()

gem.boolean_intersect(base)
```

### C# (Unity)

```csharp
var gem = AliceSdf.Intersection(
    AliceSdf.IcosahedralSymmetry(AliceSdf.Sphere(1.0f)),
    AliceSdf.Translate(AliceSdf.Box(1.5f, 0.8f, 1.5f), 0, 0.1f, 0));
```

### HLSL

```hlsl
float gem(float3 p) {
    float d = sdSphere(p, 1.0);
    d = opIcosahedralSymmetry(d, p);
    float base = sdBox(p - float3(0, 0.1, 0), float3(1.5, 0.8, 1.5));
    return max(d, base);  // intersection
}
```

**Tips:**
- Scale Y by 0.6 before symmetry for a flatter, diamond-like cut.
- `OctantMirror` gives a cubic crystal system instead of icosahedral.
- Apply `Round(0.01)` for slightly beveled edges that catch light better.

---

## 3. Twisted Column / Pillar

Architectural twisted pillar with polar repetition.

**Technique:** Cylinder + PolarRepeat + Twist

### Rust

```rust
let column = SdfNode::cylinder(0.08, 2.0)
    .translate(0.5, 0.0, 0.0)
    .polar_repeat(8)
    .smooth_union(SdfNode::cylinder(0.15, 2.0), 0.05)
    .twist(2.0);
```

### GDScript

```gdscript
var col = AliceSdfNode.new()
col.shape = "cylinder"; col.radius = 0.08; col.height = 2.0; col.build()

col.apply_polar_repeat(8)

var core = AliceSdfNode.new()
core.shape = "cylinder"; core.radius = 0.15; core.height = 2.0; core.build()

col.smooth_k = 0.05
col.boolean_union(core)
col.apply_twist(2.0)
```

### C# (Unity)

```csharp
var arms = AliceSdf.PolarRepeat(
    AliceSdf.Translate(AliceSdf.Cylinder(0.08f, 2.0f), 0.5f, 0, 0), 8);
var core = AliceSdf.Cylinder(0.15f, 2.0f);
var column = AliceSdf.Twist(AliceSdf.SmoothUnion(arms, core, 0.05f), 2.0f);
```

### HLSL

```hlsl
float twistedColumn(float3 p) {
    float3 tp = opTwist(p, 2.0);
    float3 rp = opPolarRepeat(tp.xz, 8);  // repeat in XZ
    float arms = sdCylinder(float3(rp.x - 0.5, tp.y, rp.y), 0.08, 2.0);
    float core = sdCylinder(tp, 0.15, 2.0);
    return opSmoothUnion(arms, core, 0.05);
}
```

**Tips:**
- Increase PolarRepeat count (12, 16) for denser patterns.
- Add `Bend(0.3)` after twist for a curved archway column.
- `Onion(0.02)` makes it hollow — good for decorative lattice.

---

## 4. Eroded Rock / Asteroid

Natural-looking weathered rock surface.

**Technique:** Sphere + Noise + SurfaceRoughness + SmoothSubtraction (cavities)

### Rust

```rust
let rock = SdfNode::sphere(1.0)
    .noise(0.15, 3.0, 42)
    .surface_roughness(6.0, 0.08, 5)
    .smooth_subtraction(
        SdfNode::sphere(0.4).translate(0.5, 0.6, 0.2).noise(0.1, 4.0, 7),
        0.1,
    )
    .smooth_subtraction(
        SdfNode::sphere(0.3).translate(-0.4, -0.3, 0.5).noise(0.1, 4.0, 13),
        0.1,
    );
```

### GDScript

```gdscript
var rock = AliceSdfNode.new()
rock.shape = "sphere"; rock.radius = 1.0; rock.build()
rock.apply_noise(0.15, 3.0, 42)
rock.apply_surface_roughness(6.0, 0.08, 5)

var cavity1 = AliceSdfNode.new()
cavity1.shape = "sphere"; cavity1.radius = 0.4; cavity1.build()
cavity1.position = Vector3(0.5, 0.6, 0.2)
cavity1.apply_noise(0.1, 4.0, 7)

rock.smooth_k = 0.1
rock.boolean_subtract(cavity1)
```

### C# (Unity)

```csharp
var base = AliceSdf.SurfaceRoughness(
    AliceSdf.Noise(AliceSdf.Sphere(1.0f), 0.15f, 3.0f, 42), 6.0f, 0.08f, 5);
var cavity = AliceSdf.Noise(
    AliceSdf.Translate(AliceSdf.Sphere(0.4f), 0.5f, 0.6f, 0.2f), 0.1f, 4.0f, 7);
var rock = AliceSdf.SmoothSubtraction(base, cavity, 0.1f);
```

### HLSL

```hlsl
float rock(float3 p) {
    float d = sdSphere(p, 1.0);
    d += fbmNoise(p * 3.0) * 0.15;              // large deformation
    d += fbmNoise(p * 6.0, 5) * 0.08;           // surface roughness
    float cavity1 = sdSphere(p - float3(0.5, 0.6, 0.2), 0.4)
                   + fbmNoise(p * 4.0) * 0.1;
    d = opSmoothSubtraction(cavity1, d, 0.1);
    return d;
}
```

**Tips:**
- Different `seed` values per cavity prevent repetitive patterns.
- Low-frequency noise (`freq = 2-3`) for overall shape, high-frequency (`freq = 6-10`) for surface detail.
- For an asteroid field, compile once and instance with different transforms.

---

## 5. Organic Creature / Alien Pod

Bulbous organic form with tentacle-like protrusions.

**Technique:** Ellipsoid body + SmoothUnion capsule limbs + Bend + SurfaceRoughness

### Rust

```rust
let body = SdfNode::ellipsoid(0.6, 0.8, 0.5);

let tentacle = SdfNode::capsule(0.06, 0.8)
    .translate(0.3, -0.7, 0.0)
    .bend(1.5);

let creature = body
    .smooth_union(tentacle.clone(), 0.15)
    .smooth_union(
        SdfNode::capsule(0.06, 0.8)
            .translate(-0.3, -0.7, 0.0)
            .bend(-1.5),
        0.15,
    )
    .smooth_union(
        SdfNode::capsule(0.05, 0.6)
            .translate(0.0, -0.6, 0.3)
            .bend(1.0),
        0.15,
    )
    .surface_roughness(8.0, 0.02, 3);
```

### GDScript

```gdscript
var body = AliceSdfNode.new()
body.shape = "ellipsoid"; body.half_extents = Vector3(0.6, 0.8, 0.5); body.build()

var tent = AliceSdfNode.new()
tent.shape = "capsule"; tent.radius = 0.06; tent.height = 0.8; tent.build()
tent.position = Vector3(0.3, -0.7, 0.0)
tent.apply_bend(1.5)

body.smooth_k = 0.15
body.boolean_union(tent)
body.apply_surface_roughness(8.0, 0.02, 3)
```

### C# (Unity)

```csharp
var body = AliceSdf.Ellipsoid(0.6f, 0.8f, 0.5f);
var tent1 = AliceSdf.Bend(
    AliceSdf.Translate(AliceSdf.Capsule(0.06f, 0.8f), 0.3f, -0.7f, 0), 1.5f);
var tent2 = AliceSdf.Bend(
    AliceSdf.Translate(AliceSdf.Capsule(0.06f, 0.8f), -0.3f, -0.7f, 0), -1.5f);
var creature = AliceSdf.SurfaceRoughness(
    AliceSdf.SmoothUnion(
        AliceSdf.SmoothUnion(body, tent1, 0.15f), tent2, 0.15f),
    8.0f, 0.02f, 3);
```

### HLSL

```hlsl
float creature(float3 p) {
    float body = sdEllipsoid(p, float3(0.6, 0.8, 0.5));
    float t1 = sdCapsule(opBend(p - float3(0.3, -0.7, 0), 1.5), 0.06, 0.8);
    float t2 = sdCapsule(opBend(p - float3(-0.3, -0.7, 0), -1.5), 0.06, 0.8);
    float d = opSmoothUnion(body, t1, 0.15);
    d = opSmoothUnion(d, t2, 0.15);
    d += fbmNoise(p * 8.0, 3) * 0.02;  // skin texture
    return d;
}
```

**Tips:**
- Vary capsule radius along length (use `RoundedCone` instead) for tapered tentacles.
- `Twist` on individual tentacles adds spiral motion.
- Animate `Bend` curvature for wriggling movement.

---

## 6. Hollow Shell / Geode

A cut-open shell revealing an interior cavity with crystal-like inner surface.

**Technique:** Sphere + Onion (shell) + Subtraction (cut) + SurfaceRoughness (inner crystals)

### Rust

```rust
let geode_outer = SdfNode::sphere(1.0)
    .onion(0.08)
    .noise(0.05, 4.0, 99);

// Cut away the top half to reveal interior
let cut = SdfNode::box3d(2.0, 1.0, 2.0).translate(0.0, 1.0, 0.0);
let geode = geode_outer.subtraction(cut);

// Inner crystal layer — small noisy sphere inside
let inner = SdfNode::sphere(0.85)
    .surface_roughness(12.0, 0.1, 4)
    .noise(0.08, 6.0, 55);
let final_geode = geode.union(inner);
```

### GDScript

```gdscript
var shell = AliceSdfNode.new()
shell.shape = "sphere"; shell.radius = 1.0; shell.build()
shell.apply_onion(0.08)
shell.apply_noise(0.05, 4.0, 99)

var cut = AliceSdfNode.new()
cut.shape = "box"; cut.half_extents = Vector3(2.0, 1.0, 2.0); cut.build()
cut.position = Vector3(0, 1.0, 0)
shell.boolean_subtract(cut)

var crystals = AliceSdfNode.new()
crystals.shape = "sphere"; crystals.radius = 0.85; crystals.build()
crystals.apply_surface_roughness(12.0, 0.1, 4)

shell.boolean_union(crystals)
```

### C# (Unity)

```csharp
var shell = AliceSdf.Noise(
    AliceSdf.Onion(AliceSdf.Sphere(1.0f), 0.08f), 0.05f, 4.0f, 99);
var cut = AliceSdf.Translate(AliceSdf.Box(2.0f, 1.0f, 2.0f), 0, 1.0f, 0);
var opened = AliceSdf.Subtraction(shell, cut);
var crystals = AliceSdf.Noise(
    AliceSdf.SurfaceRoughness(AliceSdf.Sphere(0.85f), 12.0f, 0.1f, 4),
    0.08f, 6.0f, 55);
var geode = AliceSdf.Union(opened, crystals);
```

### HLSL

```hlsl
float geode(float3 p) {
    float shell = abs(sdSphere(p, 1.0)) - 0.08;  // onion
    shell += fbmNoise(p * 4.0) * 0.05;
    float cut = sdBox(p - float3(0, 1, 0), float3(2, 1, 2));
    float opened = max(shell, -cut);               // subtraction
    float crystals = sdSphere(p, 0.85) + fbmNoise(p * 12.0, 4) * 0.1;
    return min(opened, crystals);
}
```

**Tips:**
- Increase `SurfaceRoughness` frequency (15-20) for finer crystal detail.
- Use a tilted `cut` plane (rotated box) for a more natural break.
- Two `Onion` layers with different thicknesses create banded agate patterns.

---

## 7. Sci-Fi Greeble Panel

Detailed mechanical surface panel with repeating sub-geometry.

**Technique:** Box base + RepeatFinite (grid of features) + SmoothSubtraction (channels)

### Rust

```rust
// Base panel
let panel = SdfNode::box3d(2.0, 0.1, 2.0);

// Repeating raised detail
let detail = SdfNode::box3d(0.15, 0.06, 0.15)
    .round(0.01)
    .repeat_finite([4, 1, 4], [0.4, 0.0, 0.4]);
let panel = panel.smooth_union(detail, 0.02);

// Channel grooves
let groove_x = SdfNode::box3d(2.0, 0.05, 0.02)
    .repeat_finite([1, 1, 5], [0.0, 0.0, 0.4]);
let groove_z = SdfNode::box3d(0.02, 0.05, 2.0)
    .repeat_finite([5, 1, 1], [0.4, 0.0, 0.0]);
let panel = panel
    .smooth_subtraction(groove_x, 0.01)
    .smooth_subtraction(groove_z, 0.01);
```

### GDScript

```gdscript
var panel = AliceSdfNode.new()
panel.shape = "box"; panel.half_extents = Vector3(2.0, 0.1, 2.0); panel.build()

# Manually create detail grid using CSG
# (RepeatFinite is available via compiled evaluation)
var detail = AliceSdfNode.new()
detail.shape = "box"; detail.half_extents = Vector3(0.15, 0.06, 0.15); detail.build()
detail.apply_round(0.01)

panel.smooth_k = 0.02
panel.boolean_union(detail)
```

### C# (Unity)

```csharp
var panel = AliceSdf.Box(2.0f, 0.1f, 2.0f);
var detail = AliceSdf.RepeatFinite(
    AliceSdf.Round(AliceSdf.Box(0.15f, 0.06f, 0.15f), 0.01f),
    4, 1, 4, 0.4f, 0.0f, 0.4f);
panel = AliceSdf.SmoothUnion(panel, detail, 0.02f);

var grooveX = AliceSdf.RepeatFinite(
    AliceSdf.Box(2.0f, 0.05f, 0.02f), 1, 1, 5, 0.0f, 0.0f, 0.4f);
panel = AliceSdf.SmoothSubtraction(panel, grooveX, 0.01f);
```

### HLSL

```hlsl
float greeblePanel(float3 p) {
    float panel = sdBox(p, float3(2.0, 0.1, 2.0));
    float3 rp = opRepeatFinite(p, float3(0.4, 0, 0.4), int3(4, 1, 4));
    float detail = sdRoundBox(rp, float3(0.15, 0.06, 0.15), 0.01);
    float d = opSmoothUnion(panel, detail, 0.02);
    float3 gp = opRepeatFinite(p, float3(0, 0, 0.4), int3(1, 1, 5));
    float groove = sdBox(gp, float3(2.0, 0.05, 0.02));
    d = opSmoothSubtraction(groove, d, 0.01);
    return d;
}
```

**Tips:**
- Layer multiple `RepeatFinite` at different scales for fractal-like detail.
- `ChamferSubtraction` instead of `SmoothSubtraction` gives sharper mechanical edges.
- Add `Onion(0.005)` to individual details for panel inset lines.

---

## 8. TPMS Lattice Structure

Lightweight structural lattice using Triply Periodic Minimal Surfaces.

**Technique:** Gyroid + Intersection (bounding shape) + Onion (wall thickness)

### Rust

```rust
// Gyroid lattice bounded by a sphere
let lattice = SdfNode::gyroid(3.0, 0.05)
    .intersection(SdfNode::sphere(1.5));

// Or: Schwarz-P lattice in a box
let lattice_box = SdfNode::schwarz_p(4.0, 0.04)
    .intersection(SdfNode::rounded_box(1.0, 1.0, 1.0, 0.05));
```

### GDScript

```gdscript
var gyroid = AliceSdfNode.new()
gyroid.shape = "gyroid"; gyroid.build()
# Set scale/thickness via resource params

var bounds = AliceSdfNode.new()
bounds.shape = "sphere"; bounds.radius = 1.5; bounds.build()

gyroid.boolean_intersect(bounds)
```

### C# (Unity)

```csharp
var lattice = AliceSdf.Intersection(
    AliceSdf.Gyroid(3.0f, 0.05f),
    AliceSdf.Sphere(1.5f));
```

### HLSL

```hlsl
float lattice(float3 p) {
    float g = abs(sin(p.x*3.0)*cos(p.y*3.0)
              + sin(p.y*3.0)*cos(p.z*3.0)
              + sin(p.z*3.0)*cos(p.x*3.0)) - 0.05;
    float bounds = sdSphere(p, 1.5);
    return max(g, bounds);
}
```

**Tips:**
- All 9 TPMS surfaces (Gyroid, SchwarzP, Diamond, Neovius, Lidinoid, IWP, FRD, FischerKochS, PMY) work with this pattern.
- Lower thickness values (0.02-0.03) create more delicate lattices.
- `Morph(gyroid, schwarz_p, 0.5)` blends two TPMS topologies for unique structures.

---

## 9. Portal / Magic Ring

Glowing toroidal portal with energy effects.

**Technique:** Torus + Onion (ring shell) + Twist + Noise (energy distortion)

### Rust

```rust
let ring = SdfNode::torus(1.0, 0.08)
    .onion(0.02)
    .twist(4.0)
    .noise(0.03, 8.0, 77);

// Inner energy disc
let disc = SdfNode::cylinder(0.02, 1.0)
    .scale_non_uniform(1.0, 0.01, 1.0)
    .noise(0.02, 12.0, 33);

let portal = ring.union(disc);
```

### GDScript

```gdscript
var ring = AliceSdfNode.new()
ring.shape = "torus"; ring.radius = 1.0; ring.build()
ring.apply_onion(0.02)
ring.apply_twist(4.0)
ring.apply_noise(0.03, 8.0, 77)
```

### C# (Unity)

```csharp
var ring = AliceSdf.Noise(
    AliceSdf.Twist(
        AliceSdf.Onion(AliceSdf.Torus(1.0f, 0.08f), 0.02f),
        4.0f),
    0.03f, 8.0f, 77);
var disc = AliceSdf.Noise(
    AliceSdf.ScaleNonUniform(AliceSdf.Cylinder(0.02f, 1.0f), 1, 0.01f, 1),
    0.02f, 12.0f, 33);
var portal = AliceSdf.Union(ring, disc);
```

### HLSL

```hlsl
float portal(float3 p) {
    float3 tp = opTwist(p, 4.0);
    float ring = abs(sdTorus(tp, 1.0, 0.08)) - 0.02;  // onion
    ring += fbmNoise(tp * 8.0) * 0.03;
    float disc = sdCylinder(p * float3(1, 100, 1), 0.02, 1.0);
    disc += fbmNoise(p * 12.0) * 0.02;
    return min(ring, disc);
}
```

**Tips:**
- Animate the `Twist` strength over time for a spinning portal effect.
- `Animated` modifier with `speed = 2.0` adds automatic time-based distortion.
- Use the ring's SDF distance as a glow intensity in the shader.

---

## 10. Fractal Growth / IFS Tree

Recursive branching structure using Iterated Function Systems.

**Technique:** Box/Capsule + IFS (recursive transforms)

### Rust

```rust
use std::f32::consts::PI;

// Two transform matrices: scale-down + rotate left/right
let angle = PI / 6.0; // 30 degrees
let scale = 0.65;

// Transform 1: scale + rotate +30° + translate up
let t1: [f32; 16] = /* column-major 4x4: scale(0.65) * rotZ(+30°) * translate(0, 1, 0) */;
// Transform 2: scale + rotate -30° + translate up
let t2: [f32; 16] = /* column-major 4x4: scale(0.65) * rotZ(-30°) * translate(0, 1, 0) */;

let transforms = [t1, t2].concat();
let tree = SdfNode::capsule(0.05, 0.5)
    .ifs(&transforms, 2, 6);  // 2 transforms, 6 iterations
```

### C# (Unity)

```csharp
float[] transforms = new float[2 * 16]; // 2 column-major 4x4 matrices
// Fill t1: scale(0.65) * rotateZ(+30°) * translate(0, 1, 0)
// Fill t2: scale(0.65) * rotateZ(-30°) * translate(0, 1, 0)

var tree = AliceSdf.IFS(AliceSdf.Capsule(0.05f, 0.5f), transforms, 2, 6);
```

### HLSL

```hlsl
float ifsTree(float3 p) {
    float d = 1e10;
    float s = 1.0;
    for (int i = 0; i < 6; i++) {
        d = min(d, sdCapsule(p, 0.05, 0.5) * s);
        // Apply two transforms, take min
        float3 p1 = rotZ(p, 0.523) * 1.538 - float3(0, 1, 0);
        float3 p2 = rotZ(p, -0.523) * 1.538 - float3(0, 1, 0);
        p = (sdCapsule(p1, 0.05, 0.5) < sdCapsule(p2, 0.05, 0.5)) ? p1 : p2;
        s *= 0.65;
    }
    return d;
}
```

**Tips:**
- More iterations (7-8) create denser fractal detail but are slower.
- Three transforms (120° spacing) create coral-like branching.
- Add `SurfaceRoughness` at the end for organic bark texture.
- Use `Morph` between an IFS tree and a sphere for a growth animation.

---

## Modifier Combination Cheat Sheet

Common modifier stacks for VFX categories:

| Effect | Modifier Chain |
|--------|---------------|
| **Organic surface** | `Noise(0.1, 3) → SurfaceRoughness(8, 0.02, 4)` |
| **Mechanical edge** | `Round(0.02) → ChamferUnion` |
| **Crystal facets** | `IcosahedralSymmetry → Round(0.005)` |
| **Tentacle/vine** | `Bend(1.5) → Twist(2.0) → Taper(0.5)` |
| **Hollow vessel** | `Onion(0.05) → Subtraction(cut_plane)` |
| **Repeating detail** | `RepeatFinite → SmoothUnion(base, 0.02)` |
| **Radial array** | `PolarRepeat(N) → SmoothUnion(core, 0.05)` |
| **Eroded/weathered** | `Noise(0.15, 3) → SmoothSubtraction(cavities, 0.1)` |
| **Energy/plasma** | `Twist(4) → Noise(0.05, 10) → Animated(2, 0.1)` |
| **Fractal** | `IFS(transforms, 5-7) → SurfaceRoughness(6, 0.03, 3)` |
| **Mirror symmetry** | `OctantMirror` (48-fold) or `Mirror([1,1,0])` (bilateral) |
| **Lattice infill** | `Gyroid/SchwarzP → Intersection(bounding_shape)` |

---

## Performance Notes

- **Compile** (`SdfNode::compile()`) before batch evaluation or mesh generation. Compiled SDFs are 2-10x faster.
- **SurfaceRoughness** and **IFS** are the most expensive modifiers. Keep octave count low (3-5) for real-time use.
- **RepeatFinite** is much cheaper than manually creating N copies with SmoothUnion.
- For **real-time raymarching** in shaders, use `ToGlsl()` / `ToHlsl()` / `ToWgsl()` to generate optimized shader code instead of evaluating through FFI.
- **LOD strategy**: Use `AliceSdfLodManager` (Godot) or generate multiple mesh resolutions (64/32/16) and swap by camera distance.
