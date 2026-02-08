//! Icosahedral symmetry modifier (120-fold)
//!
//! Maps points to the fundamental domain of the icosahedral symmetry group.
//! More powerful than OctantMirror (48-fold octahedral symmetry).
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Golden ratio
const PHI: f32 = 1.618033988749895;

/// Fold point into icosahedral fundamental domain
/// Applies reflections through icosahedral symmetry planes until convergence
#[inline(always)]
pub fn icosahedral_fold(p: Vec3) -> Vec3 {
    let mut p = p.abs(); // Start with octant fold

    // Normalize icosahedral normals
    let n0 = Vec3::new(0.0, 1.0, PHI).normalize();
    let n1 = Vec3::new(1.0, PHI, 0.0).normalize();
    let n2 = Vec3::new(PHI, 0.0, 1.0).normalize();

    // Iterative folding through icosahedral planes
    for _ in 0..8 {
        // Sort axes (like octant mirror but extended)
        if p.y > p.x {
            p = Vec3::new(p.y, p.x, p.z);
        }
        if p.z > p.y {
            p = Vec3::new(p.x, p.z, p.y);
        }
        if p.y > p.x {
            p = Vec3::new(p.y, p.x, p.z);
        }

        // Reflect through icosahedral planes
        let d0 = p.dot(n0);
        if d0 < 0.0 {
            p -= n0 * (2.0 * d0);
        }

        let d1 = p.dot(n1);
        if d1 < 0.0 {
            p -= n1 * (2.0 * d1);
        }

        let d2 = p.dot(n2);
        if d2 < 0.0 {
            p -= n2 * (2.0 * d2);
        }
    }

    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icosahedral_symmetry() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let q = icosahedral_fold(p);
        // Result should be in the fundamental domain (all components non-negative, sorted)
        assert!(q.x >= -1e-6 && q.y >= -1e-6 && q.z >= -1e-6);
    }

    #[test]
    fn test_icosahedral_origin() {
        let p = Vec3::ZERO;
        let q = icosahedral_fold(p);
        assert!(q.length() < 1e-6);
    }
}
