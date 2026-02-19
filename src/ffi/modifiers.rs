//! FFI functions for SDF modifiers (round, onion, twist, bend, repeat, IFS, heightmap, etc.).
//!
//! Author: Moroya Sakamoto

use super::registry::{get_node, register_node};
use super::types::*;
use std::slice;

// ============================================================================
// Modifiers
// ============================================================================

/// Apply rounding to an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_round(node: SdfHandle, radius: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().round(radius);
    register_node(new_node)
}

/// Apply onion (shell) modifier to an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_onion(node: SdfHandle, thickness: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().onion(thickness);
    register_node(new_node)
}

/// Apply twist modifier
#[no_mangle]
pub extern "C" fn alice_sdf_twist(node: SdfHandle, strength: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().twist(strength);
    register_node(new_node)
}

/// Apply bend modifier
#[no_mangle]
pub extern "C" fn alice_sdf_bend(node: SdfHandle, curvature: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().bend(curvature);
    register_node(new_node)
}

/// Apply infinite repetition
#[no_mangle]
pub extern "C" fn alice_sdf_repeat(node: SdfHandle, sx: f32, sy: f32, sz: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().repeat_infinite(sx, sy, sz);
    register_node(new_node)
}

/// Apply mirror modifier (reflects along specified axes)
#[no_mangle]
pub extern "C" fn alice_sdf_mirror(node: SdfHandle, mx: u8, my: u8, mz: u8) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().mirror(mx != 0, my != 0, mz != 0);
    register_node(new_node)
}

/// Apply elongation modifier
#[no_mangle]
pub extern "C" fn alice_sdf_elongate(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().elongate(x, y, z);
    register_node(new_node)
}

/// Apply revolution modifier (rotational symmetry around Y-axis)
#[no_mangle]
pub extern "C" fn alice_sdf_revolution(node: SdfHandle, offset: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().revolution(offset);
    register_node(new_node)
}

/// Apply extrude modifier (creates 3D from XY cross-section along Z)
#[no_mangle]
pub extern "C" fn alice_sdf_extrude(node: SdfHandle, half_height: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    // The .extrude() method takes full height and halves it internally,
    // but FFI takes half_height directly, so pass double to match the method.
    let new_node = (*sdf_node).clone().extrude(half_height * 2.0);
    register_node(new_node)
}

/// Apply Perlin noise displacement
#[no_mangle]
pub extern "C" fn alice_sdf_noise(
    node: SdfHandle,
    amplitude: f32,
    frequency: f32,
    seed: u32,
) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().noise(amplitude, frequency, seed);
    register_node(new_node)
}

/// Apply finite repetition
#[no_mangle]
pub extern "C" fn alice_sdf_repeat_finite(
    node: SdfHandle,
    cx: u32,
    cy: u32,
    cz: u32,
    sx: f32,
    sy: f32,
    sz: f32,
) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node)
        .clone()
        .repeat_finite([cx, cy, cz], glam::Vec3::new(sx, sy, sz));
    register_node(new_node)
}

/// Apply taper modifier (scale XZ by Y position)
#[no_mangle]
pub extern "C" fn alice_sdf_taper(node: SdfHandle, factor: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().taper(factor);
    register_node(new_node)
}

/// Apply sin-based displacement modifier
#[no_mangle]
pub extern "C" fn alice_sdf_displacement(node: SdfHandle, strength: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().displacement(strength);
    register_node(new_node)
}

/// Apply polar repetition around Y-axis
#[no_mangle]
pub extern "C" fn alice_sdf_polar_repeat(node: SdfHandle, count: u32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().polar_repeat(count);
    register_node(new_node)
}

/// Apply octant mirror (48-fold symmetry)
#[no_mangle]
pub extern "C" fn alice_sdf_octant_mirror(node: SdfHandle) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().octant_mirror();
    register_node(new_node)
}

/// Shear deformation
#[no_mangle]
pub extern "C" fn alice_sdf_shear(node: SdfHandle, xy: f32, xz: f32, yz: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().shear(xy, xz, yz);
    register_node(new_node)
}

/// Apply time-based animation
#[no_mangle]
pub extern "C" fn alice_sdf_animated(node: SdfHandle, speed: f32, amplitude: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().animated(speed, amplitude);
    register_node(new_node)
}

/// Assign a material ID to a subtree
#[no_mangle]
pub extern "C" fn alice_sdf_with_material(node: SdfHandle, material_id: u32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().with_material(material_id);
    register_node(new_node)
}

/// Apply sweep along a quadratic Bezier curve in XZ plane
#[no_mangle]
pub extern "C" fn alice_sdf_sweep_bezier(
    node: SdfHandle,
    p0x: f32,
    p0z: f32,
    p1x: f32,
    p1z: f32,
    p2x: f32,
    p2z: f32,
) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().sweep_bezier(
        glam::Vec2::new(p0x, p0z),
        glam::Vec2::new(p1x, p1z),
        glam::Vec2::new(p2x, p2z),
    );
    register_node(new_node)
}

/// Apply non-uniform scale modifier
#[no_mangle]
pub extern "C" fn alice_sdf_scale_non_uniform(
    node: SdfHandle,
    x: f32,
    y: f32,
    z: f32,
) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().scale_xyz(x, y, z);
    register_node(new_node)
}

// ============================================================================
// Advanced Modifiers
// ============================================================================

/// Apply icosahedral symmetry (60-fold rotational symmetry)
#[no_mangle]
pub extern "C" fn alice_sdf_icosahedral_symmetry(node: SdfHandle) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().icosahedral_symmetry();
    register_node(new_node)
}

/// Apply Iterated Function System (fractal) transformations
///
/// # Parameters
/// - `node`: SDF handle
/// - `transforms`: Pointer to transformation matrices (16 floats per matrix)
/// - `transform_count`: Number of transformation matrices
/// - `iterations`: Number of IFS iterations
///
/// # Safety
/// `transforms` must point to valid array of transform_count * 16 floats
#[no_mangle]
pub extern "C" fn alice_sdf_ifs(
    node: SdfHandle,
    transforms: *const f32,
    transform_count: u32,
    iterations: u32,
) -> SdfHandle {
    if transforms.is_null() {
        return SDF_HANDLE_NULL;
    }
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };

    // SAFETY: Caller guarantees transforms points to a valid array of at least transform_count * 16
    // f32 values (one 4x4 column-major matrix per entry). Null check performed above. The pointer
    // remains valid for the duration of this call. This is an FFI contract.
    let transform_slice =
        unsafe { slice::from_raw_parts(transforms, transform_count as usize * 16) };
    let transform_matrices: Vec<[f32; 16]> = transform_slice
        .chunks_exact(16)
        .map(|c| c.try_into().unwrap())
        .collect();

    let new_node = (*sdf_node).clone().ifs(transform_matrices, iterations);
    register_node(new_node)
}

/// Apply heightmap-based displacement
///
/// # Parameters
/// - `node`: SDF handle
/// - `heightmap`: Pointer to heightmap data (width * height floats)
/// - `width`: Heightmap width
/// - `height`: Heightmap height
/// - `amplitude`: Displacement amplitude
/// - `scale`: UV scale factor
///
/// # Safety
/// `heightmap` must point to valid array of width * height floats
#[no_mangle]
pub extern "C" fn alice_sdf_heightmap_displacement(
    node: SdfHandle,
    heightmap: *const f32,
    width: u32,
    height: u32,
    amplitude: f32,
    scale: f32,
) -> SdfHandle {
    if heightmap.is_null() {
        return SDF_HANDLE_NULL;
    }
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };

    let hm_size = (width * height) as usize;
    // SAFETY: Caller guarantees heightmap points to a valid array of at least width * height f32
    // values. Null check performed above. The pointer remains valid for the duration of this call.
    // This is an FFI contract.
    let hm_slice = unsafe { slice::from_raw_parts(heightmap, hm_size) };
    let hm_vec = hm_slice.to_vec();

    let new_node = (*sdf_node)
        .clone()
        .heightmap_displacement(hm_vec, width, height, amplitude, scale);
    register_node(new_node)
}

/// Apply surface roughness using Perlin noise
#[no_mangle]
pub extern "C" fn alice_sdf_surface_roughness(
    node: SdfHandle,
    frequency: f32,
    amplitude: f32,
    octaves: u32,
) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node)
        .clone()
        .surface_roughness(frequency, amplitude, octaves);
    register_node(new_node)
}
