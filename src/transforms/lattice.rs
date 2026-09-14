//! Free-Form Deformation (FFD) via lattice control points
//!
//! Tricubic Bernstein polynomial interpolation over a control point grid.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Cubic Bernstein basis function
#[inline(always)]
fn bernstein3(t: f32, i: usize) -> f32 {
    match i {
        0 => (1.0 - t) * (1.0 - t) * (1.0 - t),
        1 => 3.0 * t * (1.0 - t) * (1.0 - t),
        2 => 3.0 * t * t * (1.0 - t),
        3 => t * t * t,
        _ => 0.0,
    }
}

/// Evaluate lattice deformation at point p.
///
/// `control_points`: `(nx+1) * (ny+1) * (nz+1)` control points.
/// `bbox_min`/`bbox_max`: lattice bounding box.
/// Returns deformed point and approximate Jacobian magnitude for distance correction.
///
/// Points outside `[bbox_min, bbox_max]` are returned unchanged with correction `1.0`
/// (standard FFD: the lattice only deforms its own volume). Before 1.9.2 the local
/// coordinate was clamped instead, which mapped every outside point to the same
/// deformed boundary point and, via the `0.1` correction floor, inflated the
/// resulting distance 10x.
#[inline(always)]
pub fn lattice_deform(
    p: Vec3,
    control_points: &[Vec3],
    nx: u32,
    ny: u32,
    nz: u32,
    bbox_min: Vec3,
    bbox_max: Vec3,
) -> (Vec3, f32) {
    if p.cmplt(bbox_min).any() || p.cmpgt(bbox_max).any() {
        return (p, 1.0);
    }
    let size = bbox_max - bbox_min;
    let inv_size = Vec3::new(1.0 / size.x, 1.0 / size.y, 1.0 / size.z);

    // Map to [0,1] local coordinates
    let s = (p - bbox_min) * inv_size;
    let s = s.clamp(Vec3::ZERO, Vec3::ONE);

    // Evaluate tricubic Bernstein
    let mut result = Vec3::ZERO;
    let cpx = (nx + 1) as usize;
    let cpy = (ny + 1) as usize;

    for i in 0..4.min(cpx) {
        let bi = bernstein3(s.x, i);
        for j in 0..4.min(cpy) {
            let bj = bernstein3(s.y, j);
            for k in 0..4.min((nz + 1) as usize) {
                let bk = bernstein3(s.z, k);
                let idx = i * cpy * (nz + 1) as usize + j * (nz + 1) as usize + k;
                if idx < control_points.len() {
                    result += control_points[idx] * bi * bj * bk;
                }
            }
        }
    }

    // Approximate Jacobian magnitude by finite differences
    let eps = 0.001;
    let dx = lattice_deform_inner(
        p + Vec3::X * eps,
        control_points,
        nx,
        ny,
        nz,
        bbox_min,
        bbox_max,
    ) - lattice_deform_inner(
        p - Vec3::X * eps,
        control_points,
        nx,
        ny,
        nz,
        bbox_min,
        bbox_max,
    );
    let correction = (dx.length() / (2.0 * eps)).max(0.1);

    (result, correction)
}

fn lattice_deform_inner(
    p: Vec3,
    cp: &[Vec3],
    nx: u32,
    ny: u32,
    nz: u32,
    bmin: Vec3,
    bmax: Vec3,
) -> Vec3 {
    let size = bmax - bmin;
    let inv_size = Vec3::new(1.0 / size.x, 1.0 / size.y, 1.0 / size.z);
    let s = ((p - bmin) * inv_size).clamp(Vec3::ZERO, Vec3::ONE);
    let mut result = Vec3::ZERO;
    let cpx = (nx + 1) as usize;
    let cpy = (ny + 1) as usize;
    for i in 0..4.min(cpx) {
        let bi = bernstein3(s.x, i);
        for j in 0..4.min(cpy) {
            let bj = bernstein3(s.y, j);
            for k in 0..4.min((nz + 1) as usize) {
                let bk = bernstein3(s.z, k);
                let idx = i * cpy * (nz + 1) as usize + j * (nz + 1) as usize + k;
                if idx < cp.len() {
                    result += cp[idx] * bi * bj * bk;
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_outside_bbox_is_identity() {
        // FFD only deforms its own volume: outside points pass through unchanged
        // with correction 1.0 (before 1.9.2 they were clamped to the boundary and
        // the 0.1 correction floor inflated distances 10x).
        let cp: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -1.0 } else { 1.0 },
                    if i & 2 == 0 { -1.0 } else { 1.0 },
                    if i & 4 == 0 { -1.0 } else { 1.0 },
                ) * 1.3
            })
            .collect();
        for p in [
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, -1.5, 0.3),
            Vec3::new(1.0001, 0.0, 0.0),
        ] {
            let (q, c) = lattice_deform(p, &cp, 1, 1, 1, Vec3::splat(-1.0), Vec3::splat(1.0));
            assert_eq!(q, p, "outside point must be unchanged");
            assert_eq!(c, 1.0, "outside correction must be 1");
        }
        // Inside: still deformed
        let (q, _) = lattice_deform(
            Vec3::splat(0.5),
            &cp,
            1,
            1,
            1,
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
        );
        assert!(
            (q - Vec3::splat(0.5)).length() > 1e-3,
            "inside point must be deformed"
        );
    }

    #[test]
    fn test_identity_lattice() {
        // Control points at regular grid positions = identity deformation
        let mut cp = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    cp.push(Vec3::new(i as f32 / 3.0, j as f32 / 3.0, k as f32 / 3.0));
                }
            }
        }
        let p = Vec3::new(0.5, 0.5, 0.5);
        let (q, _) = lattice_deform(p, &cp, 3, 3, 3, Vec3::ZERO, Vec3::ONE);
        assert!(
            (q - p).length() < 0.05,
            "Identity lattice should preserve point: got {:?}",
            q
        );
    }
}
