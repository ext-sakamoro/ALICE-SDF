//! Primitive / CSG-binary law table for the `f32` instantiation of the evaluator.
//!
//! Bodies are the scalar arms that used to live inline in the stack machine;
//! they call the canonical `crate::primitives` / `crate::operations` functions.
//! Phase 2 of the 1.10 plan folds these into generic `sdf_x<R: Real>` laws.
//!
//! Author: Moroya Sakamoto

use super::instruction::Instruction;
use super::prim_table::PrimTable;
use super::real::Vec3R;
use crate::operations::*;
use crate::primitives::*;
use glam::{Vec2, Vec3};

#[allow(unused_variables)]
impl PrimTable for f32 {
    #[inline(always)]
    fn sphere(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        sdf_sphere_r(p, inst.params[0]) * scale_correction
    }
    #[inline(always)]
    fn box3d(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        sdf_box3d_r(p, Vec3::new(inst.params[0], inst.params[1], inst.params[2])) * scale_correction
    }
    #[inline(always)]
    fn cylinder(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        sdf_cylinder_r(p, inst.params[0], inst.params[1]) * scale_correction
    }
    #[inline(always)]
    fn torus(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        sdf_torus_r(p, inst.params[0], inst.params[1]) * scale_correction
    }
    #[inline(always)]
    fn plane(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        sdf_plane_r(
            p,
            Vec3::new(inst.params[0], inst.params[1], inst.params[2]),
            inst.params[3],
        ) * scale_correction
    }
    #[inline(always)]
    fn capsule(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        sdf_capsule_r(
            p,
            Vec3::new(inst.params[0], inst.params[1], inst.params[2]),
            Vec3::new(inst.params[3], inst.params[4], inst.params[5]),
            inst.get_capsule_radius(),
        ) * scale_correction
    }
    #[inline(always)]
    fn cone(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        sdf_cone_r(p, inst.params[0], inst.params[1]) * scale_correction
    }
    #[inline(always)]
    fn ellipsoid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        sdf_ellipsoid_r(p, Vec3::new(inst.params[0], inst.params[1], inst.params[2]))
            * scale_correction
    }
    #[inline(always)]
    fn rounded_cone(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        sdf_rounded_cone_r(p, inst.params[0], inst.params[1], inst.params[2]) * scale_correction
    }
    #[inline(always)]
    fn pyramid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        sdf_pyramid_r(p, inst.params[0]) * scale_correction
    }
    #[inline(always)]
    fn octahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        sdf_octahedron_r(p, inst.params[0]) * scale_correction
    }
    #[inline(always)]
    fn hex_prism(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        sdf_hex_prism_r(p, inst.params[0], inst.params[1]) * scale_correction
    }
    #[inline(always)]
    fn link(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        sdf_link_r(p, inst.params[0], inst.params[1], inst.params[2]) * scale_correction
    }
    #[inline(always)]
    fn rounded_box(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let half_extents = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
        let d = sdf_rounded_box(p, half_extents, inst.params[3]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn capped_cone(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_capped_cone(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn capped_torus(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_capped_torus(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn rounded_cylinder(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_rounded_cylinder(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn triangular_prism(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_triangular_prism(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn cut_sphere(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_cut_sphere(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn cut_hollow_sphere(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_cut_hollow_sphere(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn death_star(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_death_star(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn solid_angle(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_solid_angle(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn rhombus(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_rhombus(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn horseshoe(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_horseshoe(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
            inst.params[4],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn vesica(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_vesica(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn infinite_cylinder(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_infinite_cylinder(p, inst.params[0]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn infinite_cone(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_infinite_cone(p, inst.params[0]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn gyroid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_gyroid(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn heart(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_heart(p, inst.params[0]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn tube(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_tube(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn barrel(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_barrel(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn diamond(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_diamond(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn chamfered_cube(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let half_extents = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
        let d = sdf_chamfered_cube(p, half_extents, inst.params[3]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn schwarz_p(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_schwarz_p(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn superellipsoid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let half_extents = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
        let d = sdf_superellipsoid(p, half_extents, inst.params[3], inst.params[4]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn rounded_x(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_rounded_x(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn pie(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_pie(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn trapezoid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_trapezoid(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn parallelogram(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_parallelogram(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn tunnel(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_tunnel(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn uneven_capsule(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_uneven_capsule(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn egg(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_egg(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn arc_shape(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_arc_shape(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn moon(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_moon(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn cross_shape(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_cross_shape(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn blobby_cross(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_blobby_cross(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn parabola_segment(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_parabola_segment(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn regular_polygon(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_regular_polygon(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn star_polygon(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_star_polygon(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn stairs(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_stairs(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn helix(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_helix(
            p,
            inst.params[0],
            inst.params[1],
            inst.params[2],
            inst.params[3],
        );
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn tetrahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_tetrahedron(p, inst.params[0]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn dodecahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_dodecahedron(p, inst.params[0]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn icosahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_icosahedron(p, inst.params[0]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn truncated_octahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_truncated_octahedron(p, inst.params[0]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn truncated_icosahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_truncated_icosahedron(p, inst.params[0]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn box_frame(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let he = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
        let d = sdf_box_frame(p, he, inst.params[3]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn diamond_surface(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_diamond_surface(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn neovius(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_neovius(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn lidinoid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_lidinoid(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn iwp(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_iwp(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn frd(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_frd(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn fischer_koch_s(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_fischer_koch_s(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn pmy(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_pmy(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn circle_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_circle_2d(p, inst.params[0], inst.params[1]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn rect_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let he = Vec2::new(inst.params[0], inst.params[1]);
        let d = sdf_rect_2d(p, he, inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn segment_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let a = Vec2::new(inst.params[0], inst.params[1]);
        let b = Vec2::new(inst.params[2], inst.params[3]);
        let d = sdf_segment_2d(p, a, b, inst.params[4], inst.params[5]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn polygon_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // Vertices live in aux_data as flat [x0, y0, x1, y1, ...]
        let aux_off = inst.aux_offset as usize;
        let flat = &aux_data[aux_off..aux_off + inst.aux_len as usize];
        let d = sdf_polygon_2d_flat(p, flat, inst.params[0]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn rounded_rect_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let he = Vec2::new(inst.params[0], inst.params[1]);
        let d = sdf_rounded_rect_2d(p, he, inst.params[2], inst.params[3]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn annular_2d(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = sdf_annular_2d(p, inst.params[0], inst.params[1], inst.params[2]);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn union(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_union_r(a, b)
    }
    #[inline(always)]
    fn intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_intersection_r(a, b)
    }
    #[inline(always)]
    fn subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_subtraction_r(a, b)
    }
    #[inline(always)]
    fn smooth_union(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_smooth_union_rk_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn smooth_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_smooth_intersection_rk_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn smooth_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_smooth_subtraction_rk_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn chamfer_union(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_chamfer_union_r(a, b, inst.params[0])
    }
    #[inline(always)]
    fn chamfer_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_chamfer_intersection_r(a, b, inst.params[0])
    }
    #[inline(always)]
    fn chamfer_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_chamfer_subtraction_r(a, b, inst.params[0])
    }
    #[inline(always)]
    fn stairs_union(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_stairs_union_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn stairs_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_stairs_intersection_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn stairs_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_stairs_subtraction_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn xor(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_xor_r(a, b)
    }
    #[inline(always)]
    fn morph(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_morph_r(a, b, inst.params[0])
    }
    #[inline(always)]
    fn columns_union(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_columns_union_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn columns_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_columns_intersection_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn columns_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_columns_subtraction_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn pipe(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_pipe_r(a, b, inst.params[0])
    }
    #[inline(always)]
    fn engrave(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_engrave_r(a, b, inst.params[0])
    }
    #[inline(always)]
    fn groove(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_groove_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn tongue(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_tongue_r(a, b, inst.params[0], inst.params[1])
    }
    #[inline(always)]
    fn exp_smooth_union(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_exp_smooth_union_r(a, b, inst.params[0])
    }
    #[inline(always)]
    fn exp_smooth_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_exp_smooth_intersection_r(a, b, inst.params[0])
    }
    #[inline(always)]
    fn exp_smooth_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        sdf_exp_smooth_subtraction_r(a, b, inst.params[0])
    }
}
