//! Pyramid SDF (Deep Fried Edition)
//!
//! Exact SDF for a 4-sided pyramid centered at origin.
//! Base square (side = 1) at y = -half_height,
//! Tip at y = half_height.
//!
//! Based on Inigo Quilez's sdPyramid formula.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::Vec3;

/// Square pyramid, unit base, `half_height` (generic over [`Real`]).
#[inline(always)]
pub fn sdf_pyramid_r<R: Real>(p: Vec3R<R>, half_height: f32) -> R {
    let h = half_height * 2.0;
    let m2 = h * h + 0.25;
    let (rh, rm2, half) = (R::splat(h), R::splat(m2), R::splat(0.5));
    let py = p.y + R::splat(half_height);
    let ax = p.x.abs();
    let az = p.z.abs();
    let swap = az.gt(ax);
    let px = R::select(swap, az, ax) - half;
    let pz = R::select(swap, ax, az) - half;
    let qx = pz;
    let qy = rh * py - half * px;
    let qz = rh * px + half * py;
    let s = (-qx).max(R::zero());
    let t = ((qy - half * pz) / R::splat(m2 + 0.25)).clamp(R::zero(), R::one());
    let a = rm2 * (qx + s) * (qx + s) + qy * qy;
    let b = rm2 * (qx + half * t) * (qx + half * t) + (qy - rm2 * t) * (qy - rm2 * t);
    let inner = qy.min(-qx * rm2 - qy * half);
    let d2 = R::select(inner.gt(R::zero()), R::zero(), a.min(b));
    ((d2 + qz * qz) / rm2).sqrt() * qz.max(-py).signum()
}

/// Square pyramid, unit base, `half_height`.
#[inline(always)]
pub fn sdf_pyramid(p: Vec3, half_height: f32) -> f32 {
    sdf_pyramid_r::<f32>(p.into(), half_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pyramid_origin_inside() {
        let d = sdf_pyramid(Vec3::ZERO, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_pyramid_tip() {
        let d = sdf_pyramid(Vec3::new(0.0, 1.0, 0.0), 1.0);
        assert!(d.abs() < 0.001, "Tip should be on surface, got {}", d);
    }

    #[test]
    fn test_pyramid_outside() {
        let d = sdf_pyramid(Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}
