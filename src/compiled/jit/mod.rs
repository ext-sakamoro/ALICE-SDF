//! JIT Compilation for SDF Evaluation
//!
//! This module provides Just-In-Time compilation of SDF trees using Cranelift.
//! The SDF tree structure is converted directly to native machine code,
//! eliminating interpreter overhead entirely.
//!
//! # Performance Characteristics
//!
//! | Aspect | Interpreted | Compiled (VM) | JIT Scalar | JIT SIMD |
//! |--------|-------------|---------------|------------|----------|
//! | Dispatch overhead | High | Low | None | None |
//! | Code specialization | None | None | Full | Full |
//! | SIMD | Software | Software | None | Hardware |
//! | Expected speedup | 1x | ~3x | ~5x | ~15-20x |
//!
//! # Usage
//!
//! ## Scalar JIT (single point)
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::jit::JitCompiledSdf;
//!
//! let shape = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
//!
//! // JIT compile (slower initial compilation, faster execution)
//! let jit = JitCompiledSdf::compile(&shape).unwrap();
//!
//! // Evaluate (native machine code, no interpreter)
//! let d = jit.eval(Vec3::new(0.5, 0.0, 0.0));
//! ```
//!
//! ## SIMD JIT (8 points at once)
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::jit::JitSimdSdf;
//! use alice_sdf::soa::SoAPoints;
//!
//! let shape = SdfNode::sphere(1.0);
//! let compiled = CompiledSdf::compile(&shape);
//!
//! // JIT compile with native SIMD (AVX/NEON)
//! let jit = JitSimdSdf::compile(&compiled).unwrap();
//!
//! // Evaluate 8 points using hardware SIMD
//! let x = [0.0, 1.0, 2.0, 0.5, -1.0, 0.0, 0.0, 1.5];
//! let y = [0.0; 8];
//! let z = [0.0; 8];
//! let results = unsafe { jit.eval_8(&x, &y, &z) };
//!
//! // Or use SoA batch interface
//! let points = SoAPoints::from_vec3_slice(&points_vec);
//! let distances = jit.eval_soa(&points);
//! ```
//!
//! # Architecture
//!
//! The JIT compiler:
//! 1. Traverses the SDF tree
//! 2. Generates Cranelift IR for each node
//! 3. Compiles IR to native machine code
//! 4. Returns a callable function pointer
//!
//! The SIMD JIT uses F32X4 (128-bit) vectors with 2 lanes to process 8 points,
//! generating native AVX2 (x86_64) or NEON (AArch64) instructions.
//!
//! The generated code is specialized for the specific SDF shape,
//! allowing optimizations like constant propagation and dead code elimination.
//!
//! Author: Moroya Sakamoto

mod codegen;
mod runtime;
mod simd;

pub use codegen::{extract_jit_params, JitCompiler};
pub use runtime::{JitCompiledSdf, JitCompiledSdfDynamic, JitError};
pub use simd::{extract_simd_params, JitSimdSdf, JitSimdSdfDynamic};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;
    use crate::types::SdfNode;
    use glam::Vec3;

    #[test]
    fn test_jit_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let jit = JitCompiledSdf::compile(&sphere).unwrap();

        // Test at origin (inside)
        let d_jit = jit.eval(Vec3::ZERO);
        let d_interp = eval(&sphere, Vec3::ZERO);
        assert!(
            (d_jit - d_interp).abs() < 0.0001,
            "Mismatch at origin: jit={}, interp={}",
            d_jit,
            d_interp
        );

        // Test at surface
        let d_jit = jit.eval(Vec3::new(1.0, 0.0, 0.0));
        let d_interp = eval(&sphere, Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (d_jit - d_interp).abs() < 0.0001,
            "Mismatch at surface: jit={}, interp={}",
            d_jit,
            d_interp
        );

        // Test outside
        let d_jit = jit.eval(Vec3::new(2.0, 0.0, 0.0));
        let d_interp = eval(&sphere, Vec3::new(2.0, 0.0, 0.0));
        assert!(
            (d_jit - d_interp).abs() < 0.0001,
            "Mismatch outside: jit={}, interp={}",
            d_jit,
            d_interp
        );
    }

    #[test]
    fn test_jit_box() {
        let box3d = SdfNode::box3d(1.0, 0.5, 0.5);
        let jit = JitCompiledSdf::compile(&box3d).unwrap();

        let test_points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.25, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];

        for p in test_points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&box3d, p);
            assert!(
                (d_jit - d_interp).abs() < 0.0001,
                "Mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_union() {
        let shape =
            SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0));
        let jit = JitCompiledSdf::compile(&shape).unwrap();

        let test_points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
        ];

        for p in test_points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.0001,
                "Mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_smooth_union() {
        let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::cylinder(0.5, 1.0), 0.2);
        let jit = JitCompiledSdf::compile(&shape).unwrap();

        let test_points = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        for p in test_points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.001,
                "Mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_translate() {
        let shape = SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0);
        let jit = JitCompiledSdf::compile(&shape).unwrap();

        let test_points = [
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];

        for p in test_points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.0001,
                "Mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_scale() {
        let shape = SdfNode::sphere(1.0).scale(2.0);
        let jit = JitCompiledSdf::compile(&shape).unwrap();

        let test_points = [
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];

        for p in test_points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.001,
                "Mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_complex() {
        let shape = SdfNode::sphere(1.0)
            .smooth_union(
                SdfNode::cylinder(0.3, 1.5).rotate_euler(1.57, 0.0, 0.0),
                0.2,
            )
            .subtract(SdfNode::box3d(0.4, 0.4, 0.4))
            .translate(0.5, 0.0, 0.0);

        let jit = JitCompiledSdf::compile(&shape).unwrap();

        let test_points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, -1.0, -1.0),
        ];

        for p in test_points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.001,
                "Mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_batch() {
        let shape =
            SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0));
        let jit = JitCompiledSdf::compile(&shape).unwrap();

        let points: Vec<Vec3> = (0..100)
            .map(|i| Vec3::new(i as f32 * 0.05 - 2.5, 0.0, 0.0))
            .collect();

        let results = jit.eval_batch(&points);

        for (i, p) in points.iter().enumerate() {
            let d_interp = eval(&shape, *p);
            assert!(
                (results[i] - d_interp).abs() < 0.0001,
                "Mismatch at {:?}: jit={}, interp={}",
                p,
                results[i],
                d_interp
            );
        }
    }

    // ============ Dynamic Parameter Tests ============

    #[test]
    fn test_jit_dynamic_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let jit = JitCompiledSdfDynamic::compile(&sphere).unwrap();

        let test_points = [Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];
        for p in test_points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&sphere, p);
            assert!(
                (d_jit - d_interp).abs() < 0.0001,
                "Dynamic sphere mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_dynamic_update_params() {
        // Compile with radius=1.0
        let shape1 = SdfNode::sphere(1.0);
        let mut jit = JitCompiledSdfDynamic::compile(&shape1).unwrap();

        let d1 = jit.eval(Vec3::new(1.5, 0.0, 0.0));
        let expected1 = eval(&shape1, Vec3::new(1.5, 0.0, 0.0));
        assert!(
            (d1 - expected1).abs() < 0.0001,
            "Before update: jit={}, interp={}",
            d1,
            expected1
        );

        // Update to radius=2.0 (no recompilation!)
        let shape2 = SdfNode::sphere(2.0);
        jit.update_params(&shape2);

        let d2 = jit.eval(Vec3::new(1.5, 0.0, 0.0));
        let expected2 = eval(&shape2, Vec3::new(1.5, 0.0, 0.0));
        assert!(
            (d2 - expected2).abs() < 0.0001,
            "After update: jit={}, interp={}",
            d2,
            expected2
        );
    }

    #[test]
    fn test_jit_dynamic_complex() {
        let shape = SdfNode::sphere(1.0)
            .smooth_union(
                SdfNode::cylinder(0.3, 1.5).rotate_euler(1.57, 0.0, 0.0),
                0.2,
            )
            .subtract(SdfNode::box3d(0.4, 0.4, 0.4))
            .translate(0.5, 0.0, 0.0);

        let jit = JitCompiledSdfDynamic::compile(&shape).unwrap();

        let test_points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, -1.0, -1.0),
        ];

        for p in test_points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.001,
                "Dynamic complex mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_dynamic_translate_update() {
        let shape1 = SdfNode::sphere(1.0).translate(0.0, 0.0, 0.0);
        let mut jit = JitCompiledSdfDynamic::compile(&shape1).unwrap();

        // Move sphere to (2, 0, 0)
        let shape2 = SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0);
        jit.update_params(&shape2);

        let d = jit.eval(Vec3::new(2.0, 0.0, 0.0));
        let expected = eval(&shape2, Vec3::new(2.0, 0.0, 0.0));
        assert!(
            (d - expected).abs() < 0.0001,
            "Translate update: jit={}, interp={}",
            d,
            expected
        );

        // Distance from origin should be ~1.0 (2.0 - radius)
        let d_origin = jit.eval(Vec3::ZERO);
        let expected_origin = eval(&shape2, Vec3::ZERO);
        assert!(
            (d_origin - expected_origin).abs() < 0.0001,
            "Translate origin: jit={}, interp={}",
            d_origin,
            expected_origin
        );
    }

    #[test]
    fn test_jit_dynamic_extract_params() {
        let shape = SdfNode::sphere(1.5).translate(2.0, 3.0, 4.0);
        let params = extract_jit_params(&shape);

        // Translate: 3 params (offset.x/y/z), Sphere: 1 param (radius)
        assert_eq!(params.len(), 4, "Expected 4 params, got {}", params.len());
        assert!((params[0] - 2.0).abs() < 1e-6, "offset.x");
        assert!((params[1] - 3.0).abs() < 1e-6, "offset.y");
        assert!((params[2] - 4.0).abs() < 1e-6, "offset.z");
        assert!((params[3] - 1.5).abs() < 1e-6, "radius");
    }

    // ============ New Primitive Tests (Deep Fried v2) ============

    #[test]
    fn test_jit_cone() {
        let shape = SdfNode::Cone {
            radius: 1.0,
            half_height: 1.0,
        };
        let jit = JitCompiledSdf::compile(&shape).unwrap();
        let points = [
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        for p in points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.01,
                "Cone mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_rounded_cone() {
        let shape = SdfNode::RoundedCone {
            r1: 1.0,
            r2: 0.5,
            half_height: 1.0,
        };
        let jit = JitCompiledSdf::compile(&shape).unwrap();
        let points = [
            Vec3::ZERO,
            Vec3::new(0.0, -2.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
        ];
        for p in points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.01,
                "RoundedCone mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_pyramid() {
        let shape = SdfNode::Pyramid { half_height: 1.0 };
        let jit = JitCompiledSdf::compile(&shape).unwrap();
        let points = [
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
        ];
        for p in points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.01,
                "Pyramid mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_octahedron() {
        let shape = SdfNode::Octahedron { size: 1.0 };
        let jit = JitCompiledSdf::compile(&shape).unwrap();
        let points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
        ];
        for p in points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.01,
                "Octahedron mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_hex_prism() {
        let shape = SdfNode::HexPrism {
            hex_radius: 1.0,
            half_height: 1.0,
        };
        let jit = JitCompiledSdf::compile(&shape).unwrap();
        let points = [
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
        ];
        for p in points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.01,
                "HexPrism mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }

    #[test]
    fn test_jit_dynamic_cone() {
        let shape = SdfNode::Cone {
            radius: 1.0,
            half_height: 1.0,
        };
        let jit = JitCompiledSdfDynamic::compile(&shape).unwrap();
        let points = [Vec3::ZERO, Vec3::X, Vec3::new(0.0, 1.0, 0.0)];
        for p in points {
            let d_jit = jit.eval(p);
            let d_interp = eval(&shape, p);
            assert!(
                (d_jit - d_interp).abs() < 0.01,
                "Dynamic Cone mismatch at {:?}: jit={}, interp={}",
                p,
                d_jit,
                d_interp
            );
        }
    }
}
