//! Mirror modifier (Deep Fried Edition)
//!
//! Mirrors the evaluation point along specified axes by taking
//! the absolute value of the coordinate.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Mirror point along specified axes
///
/// Takes absolute value of coordinates where the corresponding
/// axis component is non-zero.
///
/// # Arguments
/// * `p` - Point to transform
/// * `axes` - Non-zero components indicate which axes to mirror
///
/// # Deep Fried: branchless using bitwise operations
#[inline(always)]
pub fn modifier_mirror(p: Vec3, axes: Vec3) -> Vec3 {
    Vec3::new(
        if axes.x != 0.0 { p.x.abs() } else { p.x },
        if axes.y != 0.0 { p.y.abs() } else { p.y },
        if axes.z != 0.0 { p.z.abs() } else { p.z },
    )
}

/// Mirror point along X axis only
#[inline(always)]
pub fn modifier_mirror_x(p: Vec3) -> Vec3 {
    Vec3::new(p.x.abs(), p.y, p.z)
}

/// Mirror point along Y axis only
#[inline(always)]
pub fn modifier_mirror_y(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.y.abs(), p.z)
}

/// Mirror point along Z axis only
#[inline(always)]
pub fn modifier_mirror_z(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.y, p.z.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirror_x() {
        let p = Vec3::new(-2.0, 3.0, -1.0);
        let m = modifier_mirror(p, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(m, Vec3::new(2.0, 3.0, -1.0));
    }

    #[test]
    fn test_mirror_xy() {
        let p = Vec3::new(-2.0, -3.0, -1.0);
        let m = modifier_mirror(p, Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(m, Vec3::new(2.0, 3.0, -1.0));
    }

    #[test]
    fn test_mirror_all() {
        let p = Vec3::new(-2.0, -3.0, -1.0);
        let m = modifier_mirror(p, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(m, Vec3::new(2.0, 3.0, 1.0));
    }

    #[test]
    fn test_mirror_x_shorthand() {
        let p = Vec3::new(-2.0, 3.0, -1.0);
        assert_eq!(modifier_mirror_x(p), Vec3::new(2.0, 3.0, -1.0));
    }
}
