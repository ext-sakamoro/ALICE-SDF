//! Pipe operation for SDFs (Deep Fried Edition)
//!
//! Based on hg_sdf's fOpPipe. Creates a pipe-shaped surface at the
//! intersection of two shapes.
//!
//! Author: Moroya Sakamoto

/// Pipe operation on two SDFs
///
/// Creates a cylindrical surface along the intersection of two shapes.
/// Formula: `length(vec2(a, b)) - r`
/// - `r`: pipe radius
#[inline(always)]
pub fn sdf_pipe(a: f32, b: f32, r: f32) -> f32 {
    (a * a + b * b).sqrt() - r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipe_on_surface() {
        // When a=r, b=0: length(r,0)-r = 0
        let d = sdf_pipe(0.5, 0.0, 0.5);
        assert!(d.abs() < 1e-6, "Should be on surface, got {}", d);
    }

    #[test]
    fn test_pipe_inside() {
        let d = sdf_pipe(0.1, 0.1, 0.5);
        assert!(d < 0.0, "Should be inside, got {}", d);
    }

    #[test]
    fn test_pipe_outside() {
        let d = sdf_pipe(1.0, 1.0, 0.5);
        assert!(d > 0.0, "Should be outside, got {}", d);
    }

    #[test]
    fn test_pipe_symmetry() {
        let d1 = sdf_pipe(0.3, 0.5, 0.2);
        let d2 = sdf_pipe(0.5, 0.3, 0.2);
        assert!((d1 - d2).abs() < 1e-6, "Should be symmetric");
    }
}
