//! Legacy JIT-SIMD entry point (thin wrapper, deprecated since 1.9.2)
//!
//! `JitSimd` used to be a second, hand-copied Cranelift SIMD compiler next to
//! [`super::jit::JitSimdSdf`]. The two drifted (opcode coverage, `Plane` sign,
//! trig approximations), so since 1.9.2 this type delegates to `JitSimdSdf`
//! and only preserves the raw-pointer `eval` / `eval_soa` API for existing
//! callers. New code should use [`super::jit::JitSimdSdf`] directly.
//!
//! Author: Moroya Sakamoto

use super::compiler::CompiledSdf;
use super::jit::JitSimdSdf;

/// Legacy JIT-SIMD compiled SDF. Delegates to [`JitSimdSdf`].
#[deprecated(since = "1.9.2", note = "use `compiled::jit::JitSimdSdf` directly")]
pub struct JitSimd {
    inner: JitSimdSdf,
}

#[allow(deprecated)]
impl JitSimd {
    /// Compile bytecode to native SIMD code. See [`JitSimdSdf::compile`].
    ///
    /// # Errors
    ///
    /// Returns `Err` for opcodes without a SIMD codegen arm (never a silent
    /// fallback distance).
    pub fn compile(sdf: &CompiledSdf) -> Result<Self, String> {
        JitSimdSdf::compile(sdf).map(|inner| Self { inner })
    }

    /// Evaluate 8 points from raw SoA pointers.
    ///
    /// # Safety
    ///
    /// `px` / `py` / `pz` must each point to at least 8 readable `f32`s and
    /// `pout` to at least 8 writable `f32`s.
    #[inline(always)]
    pub unsafe fn eval(&self, px: *const f32, py: *const f32, pz: *const f32, pout: *mut f32) {
        self.inner.eval_8_raw(px, py, pz, pout);
    }

    /// Evaluate a whole SoA point set. See [`JitSimdSdf::eval_soa`].
    pub fn eval_soa(&self, points: &crate::soa::SoAPoints) -> Vec<f32> {
        self.inner.eval_soa(points)
    }
}

#[cfg(all(test, feature = "jit"))]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::compiled::CompiledSdf;
    use crate::eval::eval;
    use crate::soa::SoAPoints;
    use crate::types::SdfNode;
    use glam::Vec3;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_jit_simd_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.5, 0.0, 0.0),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&sphere, *p);
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Sphere mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_box() {
        let box3d = SdfNode::box3d(1.0, 0.5, 0.5);
        let compiled = CompiledSdf::compile(&box3d);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.5, 0.25, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&box3d, *p);
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Box mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_union() {
        let shape =
            SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0));
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.5, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&shape, *p);
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Union mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_smooth_union() {
        let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::cylinder(0.5, 1.0), 0.2);
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.5, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 0.5),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&shape, *p);
            assert!(
                approx_eq(results[i], expected, 0.01),
                "SmoothUnion mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_translate_scale() {
        let shape = SdfNode::sphere(1.0).scale(2.0).translate(1.0, 0.0, 0.0);
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(1.0, 0.0, 2.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&shape, *p);
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Translate+Scale mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_large_batch() {
        let shape = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points: Vec<Vec3> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0 * std::f32::consts::TAU;
                Vec3::new(
                    alice_det_math::cos(t) * 2.0,
                    alice_det_math::sin(t) * 2.0,
                    0.0,
                )
            })
            .collect();

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        assert_eq!(results.len(), points.len());

        for i in [0, 100, 500, 999] {
            let expected = eval(&shape, points[i]);
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Large batch mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_raw_eval() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let jit = JitSimd::compile(&compiled).unwrap();

        let x = [0.0f32, 1.0, 2.0, 0.5, -1.0, 0.0, 0.0, 0.5];
        let y = [0.0f32; 8];
        let z = [0.0f32; 8];
        let mut out = [0.0f32; 8];

        // SAFETY: All arrays are [f32; 8], providing exactly 8 contiguous elements
        // as required by the JIT-compiled SIMD function.
        unsafe {
            jit.eval(x.as_ptr(), y.as_ptr(), z.as_ptr(), out.as_mut_ptr());
        }

        let expected = [-1.0, 0.0, 1.0, -0.5, 0.0, -1.0, -1.0, -0.5];
        for i in 0..8 {
            assert!(
                approx_eq(out[i], expected[i], 0.001),
                "Raw eval mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected[i]
            );
        }
    }
}
