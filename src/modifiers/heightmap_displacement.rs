//! Heightmap-based displacement modifier
//!
//! Displaces SDF surface using a 2D heightmap image projected onto the shape.
//! Uses bilinear interpolation for smooth results.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Bilinear interpolation of heightmap at (u, v) in [0, w-1] x [0, h-1]
///
/// A degenerate map (`w == 0`, `h == 0`, or fewer samples than `w * h`) samples as `0.0` instead of
/// underflowing `w - 1` or indexing out of bounds: the dimensions come from the file / script that
/// built the node.
#[inline(always)]
pub fn bilinear_sample(heightmap: &[f32], w: u32, h: u32, u: f32, v: f32) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let u = u.clamp(0.0, (w - 1) as f32);
    let v = v.clamp(0.0, (h - 1) as f32);

    let u0 = u.floor() as u32;
    let v0 = v.floor() as u32;
    let u1 = (u0 + 1).min(w - 1);
    let v1 = (v0 + 1).min(h - 1);

    let fu = u.fract();
    let fv = v.fract();

    let at = |x: u32, y: u32| {
        heightmap
            .get(
                (y as usize)
                    .saturating_mul(w as usize)
                    .saturating_add(x as usize),
            )
            .copied()
            .unwrap_or(0.0)
    };
    let s00 = at(u0, v0);
    let s10 = at(u1, v0);
    let s01 = at(u0, v1);
    let s11 = at(u1, v1);

    let a = s00 * (1.0 - fu) + (s10 * fu);
    let b = s01 * (1.0 - fu) + (s11 * fu);
    a * (1.0 - fv) + b * fv
}

/// Compute heightmap displacement for a point
/// Uses triplanar projection to map 3D point to 2D heightmap coordinates
/// Returns the displacement amount
#[inline(always)]
pub fn heightmap_displacement(
    p: Vec3,
    heightmap: &[f32],
    w: u32,
    h: u32,
    amplitude: f32,
    scale: f32,
) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    // Triplanar-like projection: use dominant axis
    let abs_p = p.abs();
    let (u, v) = if abs_p.x > abs_p.y && abs_p.x > abs_p.z {
        (p.y * scale, p.z * scale) // YZ plane
    } else if abs_p.y > abs_p.z {
        (p.x * scale, p.z * scale) // XZ plane
    } else {
        (p.x * scale, p.y * scale) // XY plane
    };

    // Map to [0, w-1] x [0, h-1]
    let u = ((u * 0.5 + 0.5) * (w - 1) as f32).clamp(0.0, (w - 1) as f32);
    let v = ((v * 0.5 + 0.5) * (h - 1) as f32).clamp(0.0, (h - 1) as f32);

    bilinear_sample(heightmap, w, h, u, v) * amplitude
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_heightmap() {
        let heightmap = vec![0.5; 16];
        let p = Vec3::new(0.0, 0.0, 0.0);
        let d = heightmap_displacement(p, &heightmap, 4, 4, 1.0, 1.0);
        assert!((d - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_interpolation() {
        let heightmap = vec![0.0, 1.0, 0.0, 1.0];
        let d = bilinear_sample(&heightmap, 2, 2, 0.5, 0.0);
        assert!(
            (d - 0.5).abs() < 1e-6,
            "Bilinear should interpolate: got {}",
            d
        );
    }
}
