//! Octant mirror modifier (Deep Fried Edition)
//!
//! Maps any point to the first octant (all positive) and sorts
//! coordinates so x >= y >= z. This creates 48-fold symmetry
//! (full octahedral symmetry group).
//!
//! Based on hg_sdf's pR45/octant approach.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Octant mirror: abs + sort so x >= y >= z
///
/// Creates 48-fold symmetry by:
/// 1. Taking absolute value of all coordinates (8-fold: octant mirror)
/// 2. Sorting so x >= y >= z (6-fold: sextant within octant)
///
/// Total symmetry factor: 8 * 6 = 48
///
/// # Deep Fried: branchless swap via min/max
#[inline(always)]
pub fn modifier_octant_mirror(p: Vec3) -> Vec3 {
    let mut x = p.x.abs();
    let mut y = p.y.abs();
    let mut z = p.z.abs();

    // Sort descending: x >= y >= z (3 compare-and-swap)
    if y > x { std::mem::swap(&mut x, &mut y); }
    if z > y { std::mem::swap(&mut y, &mut z); }
    if y > x { std::mem::swap(&mut x, &mut y); }

    Vec3::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octant_mirror_positive() {
        let p = Vec3::new(3.0, 1.0, 2.0);
        let m = modifier_octant_mirror(p);
        assert_eq!(m, Vec3::new(3.0, 2.0, 1.0));
    }

    #[test]
    fn test_octant_mirror_negative() {
        let p = Vec3::new(-1.0, -3.0, -2.0);
        let m = modifier_octant_mirror(p);
        assert_eq!(m, Vec3::new(3.0, 2.0, 1.0));
    }

    #[test]
    fn test_octant_mirror_sorted() {
        let p = Vec3::new(-0.5, 2.0, -1.0);
        let m = modifier_octant_mirror(p);
        assert!(m.x >= m.y && m.y >= m.z);
        assert!(m.x >= 0.0 && m.y >= 0.0 && m.z >= 0.0);
    }

    #[test]
    fn test_octant_mirror_symmetry() {
        // All octants should map to the same point
        let expected = Vec3::new(3.0, 2.0, 1.0);
        let cases = [
            Vec3::new(3.0, 2.0, 1.0),
            Vec3::new(-3.0, 2.0, 1.0),
            Vec3::new(3.0, -2.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, -2.0, -3.0),
            Vec3::new(2.0, 3.0, 1.0),
        ];
        for p in &cases {
            let m = modifier_octant_mirror(*p);
            assert!((m - expected).length() < 1e-6, "Failed for {:?}: got {:?}", p, m);
        }
    }

    #[test]
    fn test_octant_mirror_origin() {
        let m = modifier_octant_mirror(Vec3::ZERO);
        assert_eq!(m, Vec3::ZERO);
    }
}
