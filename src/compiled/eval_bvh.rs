//! BVH-annotated compiled SDF: bytecode + per-instruction AABBs
//!
//! Since 1.9.2 there is no separate BVH compiler. [`CompiledSdfBvh::try_compile`]
//! reuses [`CompiledSdf::try_compile`] for the bytecode and runs the
//! instruction-driven AABB pass in [`super::refit`] (`refit_all` +
//! `build_parent_indices` + `build_subtree_ends`), so the compile law and the
//! AABB law each live in exactly one place and every opcode the main compiler
//! emits is accepted here.
//!
//! Author: Moroya Sakamoto

use super::aabb::AabbPacked;
use super::compiler::{CompileError, CompiledSdf};
use super::instruction::Instruction;
use super::refit::{self, RefitError};
use crate::types::SdfNode;
use glam::Vec3;

/// Compiled SDF with BVH acceleration data
///
/// Each instruction has an associated AABB for spatial pruning.
/// When the evaluation point is far from an AABB, the entire
/// subtree rooted at that instruction can be skipped.
#[derive(Clone, Debug)]
pub struct CompiledSdfBvh {
    /// The instruction bytecode
    pub instructions: Vec<Instruction>,
    /// Auxiliary data buffer (heightmaps, lattices, bones, IFS matrices, polygon
    /// vertices), indexed by `Instruction::aux_offset` / `aux_len`.
    pub aux_data: Vec<f32>,
    /// AABB for each instruction (same length as instructions)
    pub aabbs: Vec<AabbPacked>,
    /// Original node count
    pub node_count: usize,
    /// AABB encompassing the entire scene
    pub scene_aabb: AabbPacked,
    /// Parent instruction index per instruction (same length as `instructions`).
    /// `Some(idx)` when the instruction is consumed by another instruction
    /// (child of a CSG binary op, or wrapped by a transform / modifier).
    /// `None` for the root instruction and structural markers.
    ///
    /// Populated by `try_compile` and used by
    /// `crate::compiled::refit::refit_partial` to walk the ancestor chain
    /// of dirty instructions. Empty when the BVH was constructed by a
    /// path that predates parent tracking (in which case `refit_partial`
    /// will treat the whole BVH as affected).
    pub parent_indices: Vec<Option<u32>>,
    /// End index of the subtree rooted at each instruction, inclusive.
    ///
    /// - Leaves (primitives): `subtree_end[i] == i`.
    /// - Postfix binary CSG at `i`: `subtree_end[i] == i` (children come
    ///   before in bytecode order and are covered via `parent_indices`).
    /// - Prefix transforms / modifiers at `i` (with matching `PopTransform`
    ///   at `k`): `subtree_end[i] == k`, so the whole `[i, k]` range can be
    ///   skipped when no dirty index falls in it.
    /// - `PopTransform` / `End` markers: `subtree_end[i] == i`.
    ///
    /// Populated by `try_compile`. Empty when population failed (e.g.
    /// unsupported opcode) — the refit walker then falls back to the
    /// no-skip path.
    pub subtree_end: Vec<u32>,
}

impl CompiledSdfBvh {
    /// Compile an SdfNode tree with BVH data.
    ///
    /// # Panics
    ///
    /// Panics if the tree contains primitives the bytecode compiler rejects
    /// (`Triangle`, `Bezier`, `Terrain`) or the tree is too deep.
    /// Use [`try_compile`](Self::try_compile) for a non-panicking alternative.
    pub fn compile(node: &SdfNode) -> Self {
        Self::try_compile(node)
            .expect("CompiledSdfBvh::compile() failed: unsupported primitive in SDF tree")
    }

    /// Compile an SdfNode tree with BVH data, returning an error on failure.
    ///
    /// Accepts exactly the trees [`CompiledSdf::try_compile`] accepts. The AABB
    /// annotation is computed by the shared instruction-driven walker in
    /// [`super::refit`]; primitives without a tight analytic bound get a
    /// conservative box, unbounded shapes get `AabbPacked::infinite()`.
    pub fn try_compile(node: &SdfNode) -> Result<Self, CompileError> {
        let compiled = CompiledSdf::try_compile(node)?;
        let n = compiled.instructions.len();
        let mut bvh = Self {
            instructions: compiled.instructions,
            aux_data: compiled.aux_data,
            aabbs: vec![AabbPacked::empty(); n],
            node_count: compiled.node_count,
            scene_aabb: AabbPacked::empty(),
            parent_indices: Vec::new(),
            subtree_end: Vec::new(),
        };
        // Bytecode from `CompiledSdf::try_compile` is well-formed by construction, so
        // the walker can only fail on a stack-shape invariant violation — surface it
        // rather than shipping a BVH with stale AABBs.
        refit::refit_all(&mut bvh).map_err(refit_to_compile_error)?;
        bvh.parent_indices = refit::build_parent_indices(&bvh).map_err(refit_to_compile_error)?;
        bvh.subtree_end = refit::build_subtree_ends(&bvh).map_err(refit_to_compile_error)?;
        Ok(bvh)
    }

    /// Recompute every AABB from `Instruction.params[]` in place. See
    /// [`crate::compiled::refit::refit_all`].
    ///
    /// # Errors
    ///
    /// Propagates [`crate::compiled::refit::RefitError`] for malformed bytecode.
    pub fn refit_all_from_bytecode(&mut self) -> Result<usize, RefitError> {
        refit::refit_all(self)
    }

    /// Recompute AABBs only for `dirty` instructions and their ancestor
    /// chain. See [`crate::compiled::refit::refit_partial`].
    ///
    /// # Errors
    ///
    /// Propagates [`crate::compiled::refit::RefitError`].
    pub fn refit_partial_from_bytecode(&mut self, dirty: &[usize]) -> Result<usize, RefitError> {
        refit::refit_partial(self, dirty)
    }

    /// Get the number of instructions
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Get memory usage in bytes
    #[inline]
    pub fn memory_size(&self) -> usize {
        self.instructions.len() * std::mem::size_of::<Instruction>()
            + self.aabbs.len() * std::mem::size_of::<AabbPacked>()
            + self.aux_data.len() * std::mem::size_of::<f32>()
    }
}

fn refit_to_compile_error(e: RefitError) -> CompileError {
    CompileError::UnsupportedPrimitive(format!("BVH annotation failed: {e}"))
}

/// Evaluate compiled SDF with BVH annotations.
///
/// The BVH bytecode uses the same instruction set as [`super::compiler::CompiledSdf`]
/// and is executed by the shared exhaustive stack machine in
/// `eval_core`. The per-instruction AABBs (`sdf.aabbs`) are
/// retained for raymarching / refit consumers; this point evaluator does not
/// prune with them (pruning by AABB is unsafe for a single-point SDF query
/// because the distance to a culled subtree still contributes to the result).
///
#[inline]
pub fn eval_compiled_bvh(sdf: &CompiledSdfBvh, point: Vec3) -> f32 {
    super::eval_core::eval_bytecode::<f32>(&sdf.instructions, &sdf.aux_data, point.into())
}

/// Get the AABB for the entire compiled SDF
///
/// Returns the scene AABB computed during compilation, which
/// correctly handles all node types including transforms and modifiers as root.
pub const fn get_scene_aabb(sdf: &CompiledSdfBvh) -> AabbPacked {
    sdf.scene_aabb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;

    #[test]
    fn test_compile_sphere_bvh() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdfBvh::compile(&node);

        assert_eq!(compiled.node_count, 1);
        assert_eq!(compiled.instructions.len(), 2); // sphere + end
        assert_eq!(compiled.aabbs.len(), 2);

        // Check AABB
        let aabb = &compiled.aabbs[0];
        assert_eq!(aabb.min(), Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(aabb.max(), Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_compile_union_bvh() {
        let node = SdfNode::sphere(1.0).union(SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0));
        let compiled = CompiledSdfBvh::compile(&node);

        // Check that the union AABB encompasses both spheres
        let scene_aabb = get_scene_aabb(&compiled);
        assert!(scene_aabb.min_x <= -1.0);
        assert!(scene_aabb.max_x >= 4.0); // 3 + 1
    }

    #[test]
    fn test_eval_bvh_sphere() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdfBvh::compile(&node);

        let d = eval_compiled_bvh(&compiled, Vec3::ZERO);
        assert!((d + 1.0).abs() < 0.001);

        let d = eval_compiled_bvh(&compiled, Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 0.001);
    }

    #[test]
    fn test_eval_bvh_vs_interpreted() {
        let node = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
            .translate(0.5, 0.0, 0.0);
        let compiled = CompiledSdfBvh::compile(&node);

        let test_points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, -1.0, -1.0),
        ];

        for p in test_points {
            let d_interp = eval(&node, p);
            let d_bvh = eval_compiled_bvh(&compiled, p);
            assert!(
                (d_interp - d_bvh).abs() < 0.001,
                "Mismatch at {:?}: interp={}, bvh={}",
                p,
                d_interp,
                d_bvh
            );
        }
    }

    #[test]
    fn test_scene_aabb() {
        let node = SdfNode::sphere(1.0).union(SdfNode::sphere(1.0).translate(5.0, 0.0, 0.0));
        let compiled = CompiledSdfBvh::compile(&node);

        let aabb = get_scene_aabb(&compiled);
        assert!(aabb.min_x <= -1.0);
        assert!(aabb.max_x >= 6.0);
        assert!(aabb.min_y <= -1.0);
        assert!(aabb.max_y >= 1.0);
    }

    /// Helper: compare BVH evaluation against interpreted evaluation
    fn assert_bvh_matches_interp(node: &SdfNode, test_points: &[Vec3], tolerance: f32) {
        let compiled = CompiledSdfBvh::compile(node);
        for &p in test_points {
            let d_interp = eval(node, p);
            let d_bvh = eval_compiled_bvh(&compiled, p);
            assert!(
                (d_interp - d_bvh).abs() < tolerance,
                "Mismatch at {:?}: interp={}, bvh={} (diff={})",
                p,
                d_interp,
                d_bvh,
                (d_interp - d_bvh).abs()
            );
        }
    }

    const STANDARD_TEST_POINTS: [Vec3; 8] = [
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.3, -0.7, 1.2),
    ];

    #[test]
    fn test_eval_bvh_cone() {
        let node = SdfNode::cone(1.0, 1.5);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }

    #[test]
    fn test_eval_bvh_cone_translated() {
        let node = SdfNode::cone(0.8, 1.0).translate(1.0, 0.5, 0.0);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }

    #[test]
    fn test_eval_bvh_cone_aabb() {
        // cone(radius, height) → half_height = height * 0.5
        let node = SdfNode::cone(1.0, 4.0); // half_height = 2.0
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = &compiled.aabbs[0];
        assert!((aabb.min_x - (-1.0)).abs() < 0.001);
        assert!((aabb.max_x - 1.0).abs() < 0.001);
        assert!((aabb.min_y - (-2.0)).abs() < 0.001);
        assert!((aabb.max_y - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_eval_bvh_ellipsoid() {
        let node = SdfNode::ellipsoid(1.0, 0.5, 0.75);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.01);
    }

    #[test]
    fn test_eval_bvh_ellipsoid_union() {
        let node =
            SdfNode::ellipsoid(1.0, 0.5, 0.75).union(SdfNode::sphere(0.5).translate(2.0, 0.0, 0.0));
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.01);
    }

    #[test]
    fn test_eval_bvh_ellipsoid_aabb() {
        let node = SdfNode::ellipsoid(2.0, 1.0, 1.5);
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = &compiled.aabbs[0];
        assert!((aabb.min_x - (-2.0)).abs() < 0.001);
        assert!((aabb.max_x - 2.0).abs() < 0.001);
        assert!((aabb.min_y - (-1.0)).abs() < 0.001);
        assert!((aabb.max_y - 1.0).abs() < 0.001);
        assert!((aabb.min_z - (-1.5)).abs() < 0.001);
        assert!((aabb.max_z - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_eval_bvh_mirror() {
        let node = SdfNode::box3d(1.0, 0.5, 0.5)
            .translate(1.0, 0.0, 0.0)
            .mirror(true, false, false);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }

    #[test]
    fn test_eval_bvh_mirror_xyz() {
        let node = SdfNode::sphere(0.5)
            .translate(1.0, 1.0, 1.0)
            .mirror(true, true, true);
        let extra_points = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        ];
        assert_bvh_matches_interp(&node, &extra_points, 0.001);
    }

    #[test]
    fn test_eval_bvh_revolution() {
        let node = SdfNode::sphere(0.3).revolution(1.0);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.01);
    }

    #[test]
    fn test_eval_bvh_revolution_aabb() {
        // sphere(0.5).revolution(2.0)
        // Instructions: revolution, sphere, pop_transform, end
        // Revolution AABB is at index 0
        let node = SdfNode::sphere(0.5).revolution(2.0);
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = &compiled.aabbs[0]; // Revolution instruction's AABB
                                       // Child sphere AABB x: [-0.5, 0.5], max abs = 0.5
                                       // Revolution radial extent = 0.5 + |offset| = 2.5
        assert!(aabb.min_x <= -2.5, "min_x={}", aabb.min_x);
        assert!(aabb.max_x >= 2.5, "max_x={}", aabb.max_x);
        assert!(aabb.min_z <= -2.5, "min_z={}", aabb.min_z);
        assert!(aabb.max_z >= 2.5, "max_z={}", aabb.max_z);
    }

    #[test]
    fn test_eval_bvh_extrude() {
        let node = SdfNode::sphere(1.0).extrude(0.5);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.01);
    }

    #[test]
    fn test_eval_bvh_extrude_box() {
        let node = SdfNode::box3d(1.0, 0.5, 0.0).extrude(2.0);
        let test_points = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(1.5, 0.0, 0.5),
        ];
        assert_bvh_matches_interp(&node, &test_points, 0.01);
    }

    #[test]
    fn test_eval_bvh_noise() {
        // Interpreted eval uses Perlin noise, BVH uses hash noise (different algorithms)
        // So we compare BVH compiled eval vs BVH eval (same noise function)
        let node = SdfNode::sphere(1.0).noise(0.1, 2.0, 42);
        let compiled_bvh = CompiledSdfBvh::compile(&node);
        let compiled_flat = crate::compiled::CompiledSdf::compile(&node);
        for &p in &STANDARD_TEST_POINTS {
            let d_bvh = eval_compiled_bvh(&compiled_bvh, p);
            let d_flat = crate::compiled::eval::eval_compiled(&compiled_flat, p);
            assert!(
                (d_bvh - d_flat).abs() < 0.001,
                "Noise mismatch at {:?}: bvh={}, flat={} (diff={})",
                p,
                d_bvh,
                d_flat,
                (d_bvh - d_flat).abs()
            );
        }
    }

    #[test]
    fn test_eval_bvh_noise_on_union() {
        // Compare BVH vs flat compiled (both use hash noise)
        let node = SdfNode::sphere(1.0)
            .union(SdfNode::box3d(0.5, 0.5, 0.5).translate(1.5, 0.0, 0.0))
            .noise(0.05, 3.0, 7);
        let compiled_bvh = CompiledSdfBvh::compile(&node);
        let compiled_flat = crate::compiled::CompiledSdf::compile(&node);
        for &p in &STANDARD_TEST_POINTS {
            let d_bvh = eval_compiled_bvh(&compiled_bvh, p);
            let d_flat = crate::compiled::eval::eval_compiled(&compiled_flat, p);
            assert!(
                (d_bvh - d_flat).abs() < 0.001,
                "Noise mismatch at {:?}: bvh={}, flat={} (diff={})",
                p,
                d_bvh,
                d_flat,
                (d_bvh - d_flat).abs()
            );
        }
    }

    #[test]
    fn test_eval_bvh_combined_new_features() {
        // Cone + Mirror (no noise — compare against interpreted)
        let node = SdfNode::cone(0.5, 1.0).mirror(true, false, true);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }

    #[test]
    fn test_scene_aabb_transform_root() {
        // Bug fix: get_scene_aabb should return correct AABB when root is a transform
        let node = SdfNode::sphere(1.0).translate(5.0, 3.0, 0.0);
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = get_scene_aabb(&compiled);
        assert!((aabb.min_x - 4.0).abs() < 0.001, "min_x={}", aabb.min_x);
        assert!((aabb.max_x - 6.0).abs() < 0.001, "max_x={}", aabb.max_x);
        assert!((aabb.min_y - 2.0).abs() < 0.001, "min_y={}", aabb.min_y);
        assert!((aabb.max_y - 4.0).abs() < 0.001, "max_y={}", aabb.max_y);
    }

    #[test]
    fn test_scene_aabb_modifier_root() {
        // get_scene_aabb should work when root is a modifier (e.g. noise)
        let node = SdfNode::sphere(1.0).noise(0.1, 2.0, 42);
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = get_scene_aabb(&compiled);
        // Sphere AABB is [-1,1]^3, expanded by amplitude=0.1
        assert!(aabb.min_x <= -1.0);
        assert!(aabb.max_x >= 1.0);
    }

    #[test]
    fn test_noise_interpreted_vs_compiled() {
        // Bug fix: all paths now use perlin_noise_3d
        let node = SdfNode::sphere(1.0).noise(0.1, 2.0, 42);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }
}
