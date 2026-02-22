//! Generalized Distance Function (GDF) normal vectors
//!
//! Pre-normalized plane normals used by Platonic/Archimedean solid SDFs.
//! Based on [hg_sdf](http://mercury.sexy/hg_sdf/) by Mercury.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Cube face normals [0..3]
pub const GDF_CUBE: [Vec3; 3] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
];

/// Octahedron face normals [3..7] — normalize(±1, ±1, ±1)
/// Only 4 needed since GDF uses abs(dot)
const INV_SQRT3: f32 = 0.577_350_26;
pub const GDF_OCTAHEDRON: [Vec3; 4] = [
    Vec3::new(INV_SQRT3, INV_SQRT3, INV_SQRT3),
    Vec3::new(-INV_SQRT3, INV_SQRT3, INV_SQRT3),
    Vec3::new(INV_SQRT3, -INV_SQRT3, INV_SQRT3),
    Vec3::new(INV_SQRT3, INV_SQRT3, -INV_SQRT3),
];

/// Icosahedron face normals [7..13] — normalize(0, ±1, ±φ) and permutations
/// Pre-normalized: 1/sqrt(1 + φ²) ≈ 0.52573, φ/sqrt(1 + φ²) ≈ 0.85065
const ICO_A: f32 = 0.525_731_1;
const ICO_B: f32 = 0.850_650_8;
pub const GDF_ICOSAHEDRON: [Vec3; 6] = [
    Vec3::new(0.0, ICO_A, ICO_B),
    Vec3::new(0.0, ICO_A, -ICO_B),
    Vec3::new(ICO_A, ICO_B, 0.0),
    Vec3::new(ICO_A, -ICO_B, 0.0),
    Vec3::new(ICO_B, 0.0, ICO_A),
    Vec3::new(-ICO_B, 0.0, ICO_A),
];

/// Dodecahedron face normals [13..19] — normalize(0, ±φ, ±1) and permutations
/// Pre-normalized: same magnitudes as icosahedron but swapped
pub const GDF_DODECAHEDRON: [Vec3; 6] = [
    Vec3::new(0.0, ICO_B, ICO_A),
    Vec3::new(0.0, ICO_B, -ICO_A),
    Vec3::new(ICO_B, ICO_A, 0.0),
    Vec3::new(-ICO_B, ICO_A, 0.0),
    Vec3::new(ICO_A, 0.0, ICO_B),
    Vec3::new(ICO_A, 0.0, -ICO_B),
];

/// Evaluate GDF: max of absolute dot products with normals, minus radius
#[inline(always)]
pub fn gdf_eval(p: Vec3, normals: &[Vec3], radius: f32) -> f32 {
    let mut d = f32::NEG_INFINITY;
    for n in normals {
        d = d.max(p.dot(*n).abs());
    }
    d - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdf_vectors_normalized() {
        for n in &GDF_CUBE {
            assert!((n.length() - 1.0).abs() < 1e-6);
        }
        for n in &GDF_OCTAHEDRON {
            assert!((n.length() - 1.0).abs() < 1e-6);
        }
        for n in &GDF_ICOSAHEDRON {
            assert!((n.length() - 1.0).abs() < 1e-6, "ICO norm: {}", n.length());
        }
        for n in &GDF_DODECAHEDRON {
            assert!(
                (n.length() - 1.0).abs() < 1e-6,
                "DODEC norm: {}",
                n.length()
            );
        }
    }
}
