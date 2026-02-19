# New Transform Variants - Quick Reference

**Author:** Moroya Sakamoto

## Quick Examples

### 1. ProjectiveTransform - Perspective Warp

```rust
use alice_sdf::types::SdfNode;

// Identity matrix (no transform)
let identity = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

// Apply projective transform with Lipschitz bound 1.0
let sphere = SdfNode::sphere(1.0)
    .projective_transform(identity, 1.0);
```

**Parameters:**
- `inv_matrix: [f32; 16]` - Inverse projection matrix (column-major)
- `lipschitz_bound: f32` - Conservative Lipschitz bound for distance scaling

**Use Cases:**
- Perspective projection effects
- Camera-space warping
- Artistic distortion

### 2. LatticeDeform - Free-Form Deformation

```rust
use alice_sdf::types::SdfNode;
use glam::Vec3;

// Create 4x4x4 control point grid
let mut control_points = Vec::new();
for i in 0..4 {
    for j in 0..4 {
        for k in 0..4 {
            let x = i as f32 / 3.0;
            let y = j as f32 / 3.0;
            let z = k as f32 / 3.0;
            control_points.push(Vec3::new(x, y, z));
        }
    }
}

// Apply lattice deformation
let box_deformed = SdfNode::box3d(0.4, 0.4, 0.4)
    .lattice_deform(
        control_points,
        3, 3, 3,                    // nx, ny, nz
        Vec3::ZERO,                 // bbox_min
        Vec3::ONE,                  // bbox_max
    );
```

**Parameters:**
- `control_points: Vec<Vec3>` - Control point positions (nx+1)×(ny+1)×(nz+1)
- `nx, ny, nz: u32` - Grid resolution (0-indexed)
- `bbox_min, bbox_max: Vec3` - Lattice bounding box

**Use Cases:**
- Organic deformations
- Sculpting operations
- Surface warping

### 3. SdfSkinning - Bone Animation

```rust
use alice_sdf::types::SdfNode;
use alice_sdf::transforms::skinning::BoneTransform;
use glam::Vec3;

// Define bone transforms
let identity = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

let bone1 = BoneTransform {
    inv_bind_pose: identity,
    current_pose: identity,
    weight: 0.7,
};

let bone2 = BoneTransform {
    inv_bind_pose: identity,
    current_pose: identity,
    weight: 0.3,
};

// Apply skinning
let capsule = SdfNode::capsule(
    Vec3::new(-1.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 0.0),
    0.2
).sdf_skinning(vec![bone1, bone2]);
```

**Parameters:**
- `bones: Vec<BoneTransform>` - Bone transforms with weights

**BoneTransform Fields:**
- `inv_bind_pose: [f32; 16]` - Inverse bind pose matrix (world→bone)
- `current_pose: [f32; 16]` - Current pose matrix (bone→world)
- `weight: f32` - Vertex weight (should sum to ~1.0 across all bones)

**Use Cases:**
- Character animation
- Rigged deformations
- Skeletal systems

## Chaining Transforms

All new transforms work seamlessly with existing ones:

```rust
use alice_sdf::types::SdfNode;
use glam::Vec3;

let complex_shape = SdfNode::sphere(1.0)
    .translate(1.0, 0.0, 0.0)           // Existing
    .projective_transform(identity, 1.0) // New
    .scale(2.0)                         // Existing
    .rotate_euler(0.0, 1.57, 0.0);      // Existing
```

## CSG Operations

New transforms integrate with CSG:

```rust
let sphere1 = SdfNode::sphere(1.0)
    .projective_transform(identity, 1.0);

let sphere2 = SdfNode::sphere(0.8)
    .lattice_deform(control_points, 3, 3, 3, min, max);

let union = sphere1.union(sphere2);
let intersection = sphere1.intersection(sphere2);
let subtraction = sphere1.subtract(sphere2);
```

## Performance Notes

| Transform | Eval Cost | Gradient Cost | Best For |
|-----------|-----------|---------------|----------|
| ProjectiveTransform | O(1) | Numerical (6 evals) | Perspective effects |
| LatticeDeform | O(64) | Numerical (6 evals) | Organic deformations |
| SdfSkinning | O(B) bones | Numerical (6 evals) | Character animation |

## Common Patterns

### Perspective Camera Effect

```rust
// Compute perspective projection matrix and invert it
let inv_proj = compute_inverse_projection_matrix(fov, aspect, near, far);

let scene = build_scene()
    .projective_transform(inv_proj, estimated_lipschitz);
```

### Sculpting with Lattice

```rust
// Create deformed control points
let deformed_cp: Vec<Vec3> = base_control_points
    .iter()
    .map(|&p| p + sculpting_displacement(p))
    .collect();

let sculpted = shape.lattice_deform(deformed_cp, nx, ny, nz, min, max);
```

### Character Arm Bending

```rust
// Upper arm bone (identity)
let upper_bone = BoneTransform {
    inv_bind_pose: identity,
    current_pose: identity,
    weight: 1.0 - t, // Falloff
};

// Lower arm bone (rotated)
let lower_bone = BoneTransform {
    inv_bind_pose: identity,
    current_pose: rotated_matrix,
    weight: t, // Falloff
};

let arm = capsule.sdf_skinning(vec![upper_bone, lower_bone]);
```

## Files to Import

```rust
use alice_sdf::eval::eval;
use alice_sdf::types::SdfNode;
use alice_sdf::transforms::skinning::BoneTransform;
use glam::Vec3;
```

## Testing

```bash
# Unit tests
cargo test --lib transforms

# Integration tests
cargo test --test test_new_transforms

# Example
cargo run --example new_transforms
```

## Troubleshooting

### ProjectiveTransform

**Problem:** Distances seem incorrect
**Solution:** Check Lipschitz bound - increase if too tight

**Problem:** Division by zero
**Solution:** Ensure w-component (m[15]) is non-zero

### LatticeDeform

**Problem:** Shape disappears
**Solution:** Check control_points length = (nx+1)×(ny+1)×(nz+1)

**Problem:** Deformation too extreme
**Solution:** Keep control point offsets small relative to bbox size

### SdfSkinning

**Problem:** Shape distorted
**Solution:** Ensure bone weights sum to ~1.0

**Problem:** No deformation
**Solution:** Check that current_pose differs from inv_bind_pose

## Documentation

- Full implementation: `NEW_TRANSFORMS_SUMMARY.md`
- API docs: `cargo doc --open`
- Examples: `examples/new_transforms.rs`
