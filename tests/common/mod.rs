//! Common test helpers for ALICE-SDF integration tests
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;

// ============================================================================
// Standard test shapes
// ============================================================================

/// Unit sphere at origin
pub fn test_sphere() -> SdfNode {
    SdfNode::sphere(1.0)
}

/// Unit box at origin
pub fn test_box() -> SdfNode {
    SdfNode::box3d(0.5, 0.5, 0.5)
}

/// Complex CSG shape: sphere with box subtracted
pub fn test_csg() -> SdfNode {
    SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.6, 0.6, 0.6))
}

/// Multi-operation shape for stress testing
pub fn test_complex_shape() -> SdfNode {
    let base = SdfNode::sphere(1.0);
    let cut = SdfNode::box3d(0.5, 2.0, 0.5).translate(0.5, 0.0, 0.0);
    let torus = SdfNode::torus(0.8, 0.2).translate(0.0, 1.0, 0.0);
    base.subtract(cut).smooth_union(torus, 0.1)
}

// ============================================================================
// Standard test points
// ============================================================================

/// 8 canonical test points (origin, axes, diagonal, surface, outside)
pub fn test_points() -> Vec<Vec3> {
    vec![
        Vec3::ZERO,                     // origin (inside sphere)
        Vec3::new(1.0, 0.0, 0.0),       // X-axis surface
        Vec3::new(0.0, 1.0, 0.0),       // Y-axis surface
        Vec3::new(0.0, 0.0, 1.0),       // Z-axis surface
        Vec3::new(0.577, 0.577, 0.577), // diagonal (~surface)
        Vec3::new(2.0, 0.0, 0.0),       // outside X
        Vec3::new(0.0, -1.5, 0.0),      // outside -Y
        Vec3::new(0.3, 0.3, 0.3),       // inside
    ]
}

/// Generate a grid of points in [-2, 2]^3
pub fn test_grid_points(resolution: usize) -> Vec<Vec3> {
    let mut points = Vec::with_capacity(resolution * resolution * resolution);
    let step = 4.0 / resolution as f32;
    for i in 0..resolution {
        for j in 0..resolution {
            for k in 0..resolution {
                points.push(Vec3::new(
                    -2.0 + (i as f32 + 0.5) * step,
                    -2.0 + (j as f32 + 0.5) * step,
                    -2.0 + (k as f32 + 0.5) * step,
                ));
            }
        }
    }
    points
}

// ============================================================================
// Assertion helpers
// ============================================================================

/// Assert two f32 values are close within tolerance
#[allow(dead_code)]
pub fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
    assert!(
        (a - b).abs() < tol,
        "{}: {} vs {} (diff={}, tol={})",
        msg,
        a,
        b,
        (a - b).abs(),
        tol
    );
}

/// Assert compiled evaluation matches interpreted evaluation
pub fn assert_compiled_matches_interpreted(shape: &SdfNode, tol: f32) {
    let compiled = CompiledSdf::compile(shape);
    for p in test_points() {
        let d_interp = eval(shape, p);
        let d_compiled = eval_compiled(&compiled, p);
        assert!(
            (d_interp - d_compiled).abs() < tol,
            "Compiled/Interpreted mismatch at {:?}: interp={}, compiled={} (diff={})",
            p,
            d_interp,
            d_compiled,
            (d_interp - d_compiled).abs()
        );
    }
}
