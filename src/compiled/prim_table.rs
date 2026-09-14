//! Per-instantiation law tables for leaf primitives and CSG binary operators.
//!
//! The generic stack machine in [`super::eval_core`] handles every transform,
//! modifier and post-processing law once (via [`super::real::Real`]); leaf
//! primitives and binary blends are one trait method each so the evaluator's
//! single `match` jumps straight into the inlined law (no second dispatch).
//! Both implementations keep separate bodies until Phase 2 of the 1.10 plan.
//!
//! Author: Moroya Sakamoto

use super::instruction::Instruction;
use super::real::{Real, Vec3R};

/// Leaf-primitive / binary-operator laws for one `Real` instantiation.
///
/// Primitive methods return the distance already multiplied by `scale_correction`.
#[allow(missing_docs)]
pub trait PrimTable: Real {
    fn sphere(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self)
        -> Self;
    fn box3d(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn cylinder(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn torus(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn plane(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn capsule(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn cone(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn ellipsoid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn rounded_cone(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn pyramid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn octahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn hex_prism(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn link(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn rounded_box(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn capped_cone(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn capped_torus(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn rounded_cylinder(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn triangular_prism(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn cut_sphere(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn cut_hollow_sphere(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn death_star(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn solid_angle(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn rhombus(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn horseshoe(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn vesica(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self)
        -> Self;
    fn infinite_cylinder(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn infinite_cone(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn gyroid(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self)
        -> Self;
    fn heart(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn tube(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn barrel(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self)
        -> Self;
    fn diamond(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn chamfered_cube(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn schwarz_p(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn superellipsoid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn rounded_x(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn pie(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn trapezoid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn parallelogram(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn tunnel(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self)
        -> Self;
    fn uneven_capsule(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn egg(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn arc_shape(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn moon(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn cross_shape(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn blobby_cross(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn parabola_segment(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn regular_polygon(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn star_polygon(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn stairs(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self)
        -> Self;
    fn helix(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn tetrahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn dodecahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn icosahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn truncated_octahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn truncated_icosahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn box_frame(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn diamond_surface(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn neovius(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn lidinoid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn iwp(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn frd(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn fischer_koch_s(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn pmy(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self;
    fn circle_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn rect_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn segment_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn polygon_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn rounded_rect_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn annular_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self;
    fn union(inst: &Instruction, a: Self, b: Self) -> Self;
    fn intersection(inst: &Instruction, a: Self, b: Self) -> Self;
    fn subtraction(inst: &Instruction, a: Self, b: Self) -> Self;
    fn smooth_union(inst: &Instruction, a: Self, b: Self) -> Self;
    fn smooth_intersection(inst: &Instruction, a: Self, b: Self) -> Self;
    fn smooth_subtraction(inst: &Instruction, a: Self, b: Self) -> Self;
    fn chamfer_union(inst: &Instruction, a: Self, b: Self) -> Self;
    fn chamfer_intersection(inst: &Instruction, a: Self, b: Self) -> Self;
    fn chamfer_subtraction(inst: &Instruction, a: Self, b: Self) -> Self;
    fn stairs_union(inst: &Instruction, a: Self, b: Self) -> Self;
    fn stairs_intersection(inst: &Instruction, a: Self, b: Self) -> Self;
    fn stairs_subtraction(inst: &Instruction, a: Self, b: Self) -> Self;
    fn xor(inst: &Instruction, a: Self, b: Self) -> Self;
    fn morph(inst: &Instruction, a: Self, b: Self) -> Self;
    fn columns_union(inst: &Instruction, a: Self, b: Self) -> Self;
    fn columns_intersection(inst: &Instruction, a: Self, b: Self) -> Self;
    fn columns_subtraction(inst: &Instruction, a: Self, b: Self) -> Self;
    fn pipe(inst: &Instruction, a: Self, b: Self) -> Self;
    fn engrave(inst: &Instruction, a: Self, b: Self) -> Self;
    fn groove(inst: &Instruction, a: Self, b: Self) -> Self;
    fn tongue(inst: &Instruction, a: Self, b: Self) -> Self;
    fn exp_smooth_union(inst: &Instruction, a: Self, b: Self) -> Self;
    fn exp_smooth_intersection(inst: &Instruction, a: Self, b: Self) -> Self;
    fn exp_smooth_subtraction(inst: &Instruction, a: Self, b: Self) -> Self;
}
