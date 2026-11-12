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

/// The fraction of the dominant component within which the other faces are blended in (1/16,
/// exact in binary so every implementation computes the same band)
pub const FACE_BLEND_BAND: f32 = 0.0625;

/// The bilinear sample of the face of `axis` (0: the YZ plane, 1: XZ, 2: XY) at `p`, mapped to
/// [0, w-1] x [0, h-1]
#[inline(always)]
fn face_sample(p: Vec3, axis: usize, heightmap: &[f32], w: u32, h: u32, scale: f32) -> f32 {
    let (u, v) = match axis {
        0 => (p.y * scale, p.z * scale),
        1 => (p.x * scale, p.z * scale),
        _ => (p.x * scale, p.y * scale),
    };
    let u = ((u * 0.5 + 0.5) * (w - 1) as f32).clamp(0.0, (w - 1) as f32);
    let v = ((v * 0.5 + 0.5) * (h - 1) as f32).clamp(0.0, (h - 1) as f32);
    bilinear_sample(heightmap, w, h, u, v)
}

/// Compute heightmap displacement for a point
///
/// The map is projected onto the face of the dominant axis (YZ for x, XZ for y, XY for z). Where
/// another axis is within [`FACE_BLEND_BAND`] of the dominant component `m`, its face is blended in:
/// each face has the weight `w_i = max(0, |p_i| - (m - m * band))` and the sample is
/// `h_dom + sum_{i != dom} (w_i / sum w) * (h_i - h_dom)` (the weighted mean of the faces). The field
/// is continuous where the dominant axis changes (a hard choice jumps by the difference of the two
/// faces' samples there), and equal, bit for bit, to the dominant face's sample wherever the other
/// axes are outside the band (their weights are exactly 0). Returns the displacement amount
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
    let a = p.abs();
    let dom = if a.x > a.y && a.x > a.z {
        0
    } else if a.y > a.z {
        1
    } else {
        2
    };
    let max = crate::compiled::real::law_max_f32;
    let m = max(max(a.x, a.y), a.z);
    let lo = m - m * FACE_BLEND_BAND;
    let wt = [max(a.x - lo, 0.0), max(a.y - lo, 0.0), max(a.z - lo, 0.0)];
    let h_dom = face_sample(p, dom, heightmap, w, h, scale);
    let total = (wt[0] + wt[1]) + wt[2];
    let mut sample = h_dom;
    for (axis, &w_axis) in wt.iter().enumerate() {
        if axis != dom && w_axis > 0.0 {
            let h_i = face_sample(p, axis, heightmap, w, h, scale);
            sample += (w_axis / total) * (h_i - h_dom);
        }
    }
    sample * amplitude
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
