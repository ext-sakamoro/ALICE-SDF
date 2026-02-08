//! SDF Evaluation (Deep Fried Edition)
//!
//! Functions for evaluating SDF trees at points.
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: `eval` is marked `#[inline]` to allow recursion unrolling by LLVM.
//! - **Direct Dispatch**: Primitives are called directly without wrapper overhead.
//! - **Swizzle Normals**: Uses Vec3 swizzles for efficient gradient computation.
//!
//! Author: Moroya Sakamoto

pub mod parallel;
pub mod gradient;

pub use parallel::{eval_batch, eval_batch_parallel, eval_grid, eval_grid_with_normals};
pub use gradient::{eval_gradient, eval_normal};

use crate::modifiers::*;
use crate::operations::*;
use crate::primitives::*;
use crate::transforms::*;
use crate::types::SdfNode;
use glam::Vec3;

/// Evaluate an SDF tree at a single point (Deep Fried)
///
/// Recursively traverses the tree and computes the signed distance.
/// Marked `#[inline]` to encourage the compiler to inline small tree traversals.
///
/// # Arguments
/// * `node` - The SDF tree root
/// * `point` - Point to evaluate
///
/// # Returns
/// Signed distance to the surface
#[inline]
pub fn eval(node: &SdfNode, point: Vec3) -> f32 {
    match node {
        // === Primitives (Leaf Nodes) ===
        // These are the hot paths at the bottom of the recursion
        SdfNode::Sphere { radius } => sdf_sphere(point, *radius),
        SdfNode::Box3d { half_extents } => sdf_box3d(point, *half_extents),
        SdfNode::Cylinder {
            radius,
            half_height,
        } => sdf_cylinder(point, *radius, *half_height),
        SdfNode::Torus {
            major_radius,
            minor_radius,
        } => sdf_torus(point, *major_radius, *minor_radius),
        SdfNode::Plane { normal, distance } => sdf_plane(point, *normal, *distance),
        SdfNode::Capsule {
            point_a,
            point_b,
            radius,
        } => sdf_capsule(point, *point_a, *point_b, *radius),
        SdfNode::Cone {
            radius,
            half_height,
        } => sdf_cone(point, *radius, *half_height),
        SdfNode::Ellipsoid { radii } => sdf_ellipsoid(point, *radii),
        SdfNode::RoundedCone { r1, r2, half_height } => sdf_rounded_cone(point, *r1, *r2, *half_height),
        SdfNode::Pyramid { half_height } => sdf_pyramid(point, *half_height),
        SdfNode::Octahedron { size } => sdf_octahedron(point, *size),
        SdfNode::HexPrism { hex_radius, half_height } => sdf_hex_prism(point, *hex_radius, *half_height),
        SdfNode::Link { half_length, r1, r2 } => sdf_link(point, *half_length, *r1, *r2),
        SdfNode::Triangle { point_a, point_b, point_c } => sdf_triangle(point, *point_a, *point_b, *point_c),
        SdfNode::Bezier { point_a, point_b, point_c, radius } => sdf_bezier(point, *point_a, *point_b, *point_c, *radius),
        SdfNode::RoundedBox { half_extents, round_radius } => sdf_rounded_box(point, *half_extents, *round_radius),
        SdfNode::CappedCone { half_height, r1, r2 } => sdf_capped_cone(point, *half_height, *r1, *r2),
        SdfNode::CappedTorus { major_radius, minor_radius, cap_angle } => sdf_capped_torus(point, *major_radius, *minor_radius, *cap_angle),
        SdfNode::RoundedCylinder { radius, round_radius, half_height } => sdf_rounded_cylinder(point, *radius, *round_radius, *half_height),
        SdfNode::TriangularPrism { width, half_depth } => sdf_triangular_prism(point, *width, *half_depth),
        SdfNode::CutSphere { radius, cut_height } => sdf_cut_sphere(point, *radius, *cut_height),
        SdfNode::CutHollowSphere { radius, cut_height, thickness } => sdf_cut_hollow_sphere(point, *radius, *cut_height, *thickness),
        SdfNode::DeathStar { ra, rb, d } => sdf_death_star(point, *ra, *rb, *d),
        SdfNode::SolidAngle { angle, radius } => sdf_solid_angle(point, *angle, *radius),
        SdfNode::Rhombus { la, lb, half_height, round_radius } => sdf_rhombus(point, *la, *lb, *half_height, *round_radius),
        SdfNode::Horseshoe { angle, radius, half_length, width, thickness } => sdf_horseshoe(point, *angle, *radius, *half_length, *width, *thickness),
        SdfNode::Vesica { radius, half_dist } => sdf_vesica(point, *radius, *half_dist),
        SdfNode::InfiniteCylinder { radius } => sdf_infinite_cylinder(point, *radius),
        SdfNode::InfiniteCone { angle } => sdf_infinite_cone(point, *angle),
        SdfNode::Gyroid { scale, thickness } => sdf_gyroid(point, *scale, *thickness),
        SdfNode::Heart { size } => sdf_heart(point, *size),
        SdfNode::Tube { outer_radius, thickness, half_height } => sdf_tube(point, *outer_radius, *thickness, *half_height),
        SdfNode::Barrel { radius, half_height, bulge } => sdf_barrel(point, *radius, *half_height, *bulge),
        SdfNode::Diamond { radius, half_height } => sdf_diamond(point, *radius, *half_height),
        SdfNode::ChamferedCube { half_extents, chamfer } => sdf_chamfered_cube(point, *half_extents, *chamfer),
        SdfNode::SchwarzP { scale, thickness } => sdf_schwarz_p(point, *scale, *thickness),
        SdfNode::Superellipsoid { half_extents, e1, e2 } => sdf_superellipsoid(point, *half_extents, *e1, *e2),
        SdfNode::RoundedX { width, round_radius, half_height } => sdf_rounded_x(point, *width, *round_radius, *half_height),
        SdfNode::Pie { angle, radius, half_height } => sdf_pie(point, *angle, *radius, *half_height),
        SdfNode::Trapezoid { r1, r2, trap_height, half_depth } => sdf_trapezoid(point, *r1, *r2, *trap_height, *half_depth),
        SdfNode::Parallelogram { width, para_height, skew, half_depth } => sdf_parallelogram(point, *width, *para_height, *skew, *half_depth),
        SdfNode::Tunnel { width, height_2d, half_depth } => sdf_tunnel(point, *width, *height_2d, *half_depth),
        SdfNode::UnevenCapsule { r1, r2, cap_height, half_depth } => sdf_uneven_capsule(point, *r1, *r2, *cap_height, *half_depth),
        SdfNode::Egg { ra, rb } => sdf_egg(point, *ra, *rb),
        SdfNode::ArcShape { aperture, radius, thickness, half_height } => sdf_arc_shape(point, *aperture, *radius, *thickness, *half_height),
        SdfNode::Moon { d, ra, rb, half_height } => sdf_moon(point, *d, *ra, *rb, *half_height),
        SdfNode::CrossShape { length, thickness, round_radius, half_height } => sdf_cross_shape(point, *length, *thickness, *round_radius, *half_height),
        SdfNode::BlobbyCross { size, half_height } => sdf_blobby_cross(point, *size, *half_height),
        SdfNode::ParabolaSegment { width, para_height, half_depth } => sdf_parabola_segment(point, *width, *para_height, *half_depth),
        SdfNode::RegularPolygon { radius, n_sides, half_height } => sdf_regular_polygon(point, *radius, *n_sides, *half_height),
        SdfNode::StarPolygon { radius, n_points, m, half_height } => sdf_star_polygon(point, *radius, *n_points, *m, *half_height),
        SdfNode::Stairs { step_width, step_height, n_steps, half_depth } => sdf_stairs(point, *step_width, *step_height, *n_steps, *half_depth),
        SdfNode::Helix { major_r, minor_r, pitch, half_height } => sdf_helix(point, *major_r, *minor_r, *pitch, *half_height),
        SdfNode::Tetrahedron { size } => sdf_tetrahedron(point, *size),
        SdfNode::Dodecahedron { radius } => sdf_dodecahedron(point, *radius),
        SdfNode::Icosahedron { radius } => sdf_icosahedron(point, *radius),
        SdfNode::TruncatedOctahedron { radius } => sdf_truncated_octahedron(point, *radius),
        SdfNode::TruncatedIcosahedron { radius } => sdf_truncated_icosahedron(point, *radius),
        SdfNode::BoxFrame { half_extents, edge } => sdf_box_frame(point, *half_extents, *edge),
        SdfNode::DiamondSurface { scale, thickness } => sdf_diamond_surface(point, *scale, *thickness),
        SdfNode::Neovius { scale, thickness } => sdf_neovius(point, *scale, *thickness),
        SdfNode::Lidinoid { scale, thickness } => sdf_lidinoid(point, *scale, *thickness),
        SdfNode::IWP { scale, thickness } => sdf_iwp(point, *scale, *thickness),
        SdfNode::FRD { scale, thickness } => sdf_frd(point, *scale, *thickness),
        SdfNode::FischerKochS { scale, thickness } => sdf_fischer_koch_s(point, *scale, *thickness),
        SdfNode::PMY { scale, thickness } => sdf_pmy(point, *scale, *thickness),

        // === 2D Primitives (extruded along Z) ===
        SdfNode::Circle2D { radius, half_height } => {
            let d2d = Vec2::new(point.x, point.y).length() - radius;
            let dz = point.z.abs() - half_height;
            d2d.max(dz).min(0.0) + Vec2::new(d2d.max(0.0), dz.max(0.0)).length()
        }
        SdfNode::Rect2D { half_extents, half_height } => {
            let d = Vec2::new(point.x.abs() - half_extents.x, point.y.abs() - half_extents.y);
            let d2d = Vec2::new(d.x.max(0.0), d.y.max(0.0)).length() + d.x.max(d.y).min(0.0);
            let dz = point.z.abs() - half_height;
            d2d.max(dz).min(0.0) + Vec2::new(d2d.max(0.0), dz.max(0.0)).length()
        }
        SdfNode::Segment2D { a, b, thickness, half_height } => {
            let p2 = Vec2::new(point.x, point.y);
            let pa = p2 - *a;
            let ba = *b - *a;
            let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
            let d2d = (pa - ba * h).length() - thickness;
            let dz = point.z.abs() - half_height;
            d2d.max(dz).min(0.0) + Vec2::new(d2d.max(0.0), dz.max(0.0)).length()
        }
        SdfNode::Polygon2D { vertices, half_height } => {
            let p2 = Vec2::new(point.x, point.y);
            let n = vertices.len();
            if n < 3 { return 1e10; }
            let mut d = (p2 - vertices[0]).length_squared();
            let mut s = 1.0_f32;
            let mut j = n - 1;
            for i in 0..n {
                let e = vertices[j] - vertices[i];
                let w = p2 - vertices[i];
                let b_proj = w - e * (w.dot(e) / e.dot(e)).clamp(0.0, 1.0);
                d = d.min(b_proj.dot(b_proj));
                let c = [p2.y >= vertices[i].y, p2.y < vertices[j].y, e.x * w.y > e.y * w.x];
                if c.iter().all(|x| *x) || c.iter().all(|x| !*x) { s = -s; }
                j = i;
            }
            let d2d = s * d.sqrt();
            let dz = point.z.abs() - half_height;
            d2d.max(dz).min(0.0) + Vec2::new(d2d.max(0.0), dz.max(0.0)).length()
        }
        SdfNode::RoundedRect2D { half_extents, round_radius, half_height } => {
            let d = Vec2::new(point.x.abs() - half_extents.x + round_radius, point.y.abs() - half_extents.y + round_radius);
            let d2d = Vec2::new(d.x.max(0.0), d.y.max(0.0)).length() + d.x.max(d.y).min(0.0) - round_radius;
            let dz = point.z.abs() - half_height;
            d2d.max(dz).min(0.0) + Vec2::new(d2d.max(0.0), dz.max(0.0)).length()
        }
        SdfNode::Annular2D { outer_radius, thickness, half_height } => {
            let d2d = (Vec2::new(point.x, point.y).length() - outer_radius).abs() - thickness;
            let dz = point.z.abs() - half_height;
            d2d.max(dz).min(0.0) + Vec2::new(d2d.max(0.0), dz.max(0.0)).length()
        }

        // === Operations ===
        // Recurse first, then combine. Compiler can reorder these instruction streams.
        SdfNode::Union { a, b } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_union(d1, d2)
        }
        SdfNode::Intersection { a, b } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_intersection(d1, d2)
        }
        SdfNode::Subtraction { a, b } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_subtraction(d1, d2)
        }
        SdfNode::SmoothUnion { a, b, k } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_smooth_union(d1, d2, *k)
        }
        SdfNode::SmoothIntersection { a, b, k } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_smooth_intersection(d1, d2, *k)
        }
        SdfNode::SmoothSubtraction { a, b, k } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_smooth_subtraction(d1, d2, *k)
        }
        SdfNode::ChamferUnion { a, b, r } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_chamfer_union(d1, d2, *r)
        }
        SdfNode::ChamferIntersection { a, b, r } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_chamfer_intersection(d1, d2, *r)
        }
        SdfNode::ChamferSubtraction { a, b, r } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_chamfer_subtraction(d1, d2, *r)
        }
        SdfNode::StairsUnion { a, b, r, n } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_stairs_union(d1, d2, *r, *n)
        }
        SdfNode::StairsIntersection { a, b, r, n } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_stairs_intersection(d1, d2, *r, *n)
        }
        SdfNode::StairsSubtraction { a, b, r, n } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_stairs_subtraction(d1, d2, *r, *n)
        }
        SdfNode::XOR { a, b } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_xor(d1, d2)
        }
        SdfNode::Morph { a, b, t } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_morph(d1, d2, *t)
        }
        SdfNode::ColumnsUnion { a, b, r, n } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_columns_union(d1, d2, *r, *n)
        }
        SdfNode::ColumnsIntersection { a, b, r, n } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_columns_intersection(d1, d2, *r, *n)
        }
        SdfNode::ColumnsSubtraction { a, b, r, n } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_columns_subtraction(d1, d2, *r, *n)
        }
        SdfNode::Pipe { a, b, r } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_pipe(d1, d2, *r)
        }
        SdfNode::Engrave { a, b, r } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_engrave(d1, d2, *r)
        }
        SdfNode::Groove { a, b, ra, rb } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_groove(d1, d2, *ra, *rb)
        }
        SdfNode::Tongue { a, b, ra, rb } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_tongue(d1, d2, *ra, *rb)
        }
        SdfNode::ExpSmoothUnion { a, b, k } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            let k = k.max(1e-6);
            let res = (-d1 / k).exp() + (-d2 / k).exp();
            -res.ln() * k
        }
        SdfNode::ExpSmoothIntersection { a, b, k } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            let k = k.max(1e-6);
            let res = (d1 / k).exp() + (d2 / k).exp();
            res.ln() * k
        }
        SdfNode::ExpSmoothSubtraction { a, b, k } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            let k = k.max(1e-6);
            let res = (d1 / k).exp() + (-d2 / k).exp();
            res.ln() * k
        }

        // === Transforms ===
        // Transform point, then recurse.
        SdfNode::Translate { child, offset } => {
            // Direct inline: point - offset (no function call)
            eval(child, point - *offset)
        }
        SdfNode::Rotate { child, rotation } => {
            // Conjugate multiplication for inverse rotation (Deep Fried)
            let p = rotation.conjugate() * point;
            eval(child, p)
        }
        SdfNode::Scale { child, factor } => {
            // Division then multiply result
            eval(child, point / *factor) * factor
        }
        SdfNode::ScaleNonUniform { child, factors } => {
            let (p, mult) = transform_scale_nonuniform(point, *factors);
            eval(child, p) * mult
        }

        // === Modifiers ===
        SdfNode::Twist { child, strength } => {
            let p = modifier_twist(point, *strength);
            eval(child, p)
        }
        SdfNode::Bend { child, curvature } => {
            let p = modifier_bend(point, *curvature);
            eval(child, p)
        }
        SdfNode::RepeatInfinite { child, spacing } => {
            let p = modifier_repeat_infinite(point, *spacing);
            eval(child, p)
        }
        SdfNode::RepeatFinite {
            child,
            count,
            spacing,
        } => {
            let p = modifier_repeat_finite(point, *count, *spacing);
            eval(child, p)
        }
        SdfNode::Noise {
            child,
            amplitude,
            frequency,
            seed,
        } => {
            let d = eval(child, point);
            modifier_noise_perlin(d, point, *amplitude, *frequency, *seed)
        }

        // Direct arithmetic operations (no function call overhead)
        SdfNode::Round { child, radius } => eval(child, point) - radius,
        SdfNode::Onion { child, thickness } => eval(child, point).abs() - thickness,
        SdfNode::Elongate { child, amount } => {
            let q = point - point.clamp(-*amount, *amount);
            eval(child, q)
        }
        SdfNode::Mirror { child, axes } => {
            let p = modifier_mirror(point, *axes);
            eval(child, p)
        }
        SdfNode::OctantMirror { child } => {
            let p = modifier_octant_mirror(point);
            eval(child, p)
        }
        SdfNode::Revolution { child, offset } => {
            let p = modifier_revolution(point, *offset);
            eval(child, p)
        }
        SdfNode::Extrude { child, half_height } => {
            let p_flat = modifier_extrude_point(point);
            let d = eval(child, p_flat);
            modifier_extrude(d, point.z, *half_height)
        }
        SdfNode::Taper { child, factor } => {
            let p = modifier_taper(point, *factor);
            eval(child, p)
        }
        SdfNode::Displacement { child, strength } => {
            let d = eval(child, point);
            modifier_displacement(d, point, *strength)
        }
        SdfNode::PolarRepeat { child, count } => {
            let p = modifier_polar_repeat(point, *count);
            eval(child, p)
        }
        SdfNode::SweepBezier { child, p0, p1, p2 } => {
            let p = modifier_sweep_bezier(point, *p0, *p1, *p2);
            eval(child, p)
        }
        SdfNode::Shear { child, shear } => {
            // Inverse shear: subtract shear contributions
            let p = Vec3::new(
                point.x,
                point.y - shear.x * point.x,
                point.z - shear.y * point.x - shear.z * point.y,
            );
            eval(child, p)
        }
        SdfNode::Animated { child, .. } => {
            // Static evaluation: animation is time-dependent, evaluate child as-is
            eval(child, point)
        }

        // Material assignment is transparent for distance evaluation
        SdfNode::WithMaterial { child, .. } => eval(child, point),

        #[allow(unreachable_patterns)]
        _ => todo!("new SdfNode variant in eval"),
    }
}

/// Evaluate which material ID applies at a given point
///
/// Walks the SDF tree and returns the material_id of the closest
/// surface node that has a material assigned. Returns 0 (default) if
/// no material is assigned.
#[inline]
pub fn eval_material(node: &SdfNode, point: Vec3) -> u32 {
    match node {
        SdfNode::WithMaterial { child, material_id } => {
            // This subtree has a material; return it
            // (nested WithMaterial: inner wins if closer)
            let inner = eval_material(child, point);
            if inner != 0 { inner } else { *material_id }
        }

        // Operations: return material of the closer child
        SdfNode::Union { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::ChamferUnion { a, b, .. }
        | SdfNode::StairsUnion { a, b, .. }
        | SdfNode::XOR { a, b }
        | SdfNode::Morph { a, b, .. }
        | SdfNode::ColumnsUnion { a, b, .. }
        | SdfNode::Pipe { a, b, .. } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da <= db { eval_material(a, point) } else { eval_material(b, point) }
        }
        SdfNode::Intersection { a, b }
        | SdfNode::SmoothIntersection { a, b, .. }
        | SdfNode::ChamferIntersection { a, b, .. }
        | SdfNode::StairsIntersection { a, b, .. }
        | SdfNode::ColumnsIntersection { a, b, .. } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da >= db { eval_material(a, point) } else { eval_material(b, point) }
        }
        SdfNode::Subtraction { a, b }
        | SdfNode::SmoothSubtraction { a, b, .. }
        | SdfNode::ChamferSubtraction { a, b, .. }
        | SdfNode::StairsSubtraction { a, b, .. }
        | SdfNode::ColumnsSubtraction { a, b, .. }
        | SdfNode::Engrave { a, b, .. }
        | SdfNode::Groove { a, b, .. }
        | SdfNode::Tongue { a, b, .. } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da >= -db { eval_material(a, point) } else { eval_material(b, point) }
        }

        // Transforms: transform point, recurse
        SdfNode::Translate { child, offset } => eval_material(child, point - *offset),
        SdfNode::Rotate { child, rotation } => eval_material(child, rotation.conjugate() * point),
        SdfNode::Scale { child, factor } => eval_material(child, point / *factor),
        SdfNode::ScaleNonUniform { child, factors } => {
            let p = point / *factors;
            eval_material(child, p)
        }

        // Modifiers: transform point, recurse
        SdfNode::Twist { child, strength } => eval_material(child, modifier_twist(point, *strength)),
        SdfNode::Bend { child, curvature } => eval_material(child, modifier_bend(point, *curvature)),
        SdfNode::RepeatInfinite { child, spacing } => eval_material(child, modifier_repeat_infinite(point, *spacing)),
        SdfNode::RepeatFinite { child, count, spacing } => eval_material(child, modifier_repeat_finite(point, *count, *spacing)),
        SdfNode::Noise { child, .. } => eval_material(child, point),
        SdfNode::Round { child, .. } | SdfNode::Onion { child, .. } => eval_material(child, point),
        SdfNode::Elongate { child, amount } => eval_material(child, point - point.clamp(-*amount, *amount)),
        SdfNode::Mirror { child, axes } => eval_material(child, modifier_mirror(point, *axes)),
        SdfNode::OctantMirror { child } => eval_material(child, modifier_octant_mirror(point)),
        SdfNode::Revolution { child, offset } => eval_material(child, modifier_revolution(point, *offset)),
        SdfNode::Extrude { child, .. } => eval_material(child, modifier_extrude_point(point)),
        SdfNode::Taper { child, factor } => eval_material(child, modifier_taper(point, *factor)),
        SdfNode::Displacement { child, .. } => eval_material(child, point),
        SdfNode::PolarRepeat { child, count } => eval_material(child, modifier_polar_repeat(point, *count)),
        SdfNode::SweepBezier { child, p0, p1, p2 } => eval_material(child, modifier_sweep_bezier(point, *p0, *p1, *p2)),

        // Primitives: no material assigned
        _ => 0,
    }
}

/// Compute the surface normal at a point using finite differences (Deep Fried)
///
/// Uses swizzle operations for efficient gradient vector construction.
///
/// # Arguments
/// * `node` - The SDF tree
/// * `point` - Point on or near the surface
/// * `epsilon` - Small offset for gradient estimation
///
/// # Returns
/// Normalized surface normal
#[inline(always)]
pub fn normal(node: &SdfNode, point: Vec3, epsilon: f32) -> Vec3 {
    let ex = Vec3::new(epsilon, 0.0, 0.0);
    let ey = Vec3::new(0.0, epsilon, 0.0);
    let ez = Vec3::new(0.0, 0.0, epsilon);

    let grad = Vec3::new(
        eval(node, point + ex) - eval(node, point - ex),
        eval(node, point + ey) - eval(node, point - ey),
        eval(node, point + ez) - eval(node, point - ez),
    );

    // NaN guard: if gradient is zero/degenerate, return safe default
    let len_sq = grad.length_squared();
    if len_sq < 1e-20 {
        return Vec3::Y; // Safe fallback: up vector
    }
    grad / len_sq.sqrt()
}

/// Compute the gradient of the SDF at a point (Deep Fried)
///
/// Similar to normal but not normalized.
#[inline(always)]
pub fn gradient(node: &SdfNode, point: Vec3, epsilon: f32) -> Vec3 {
    let ex = Vec3::new(epsilon, 0.0, 0.0);
    let ey = Vec3::new(0.0, epsilon, 0.0);
    let ez = Vec3::new(0.0, 0.0, epsilon);

    let inv_2e = 1.0 / (2.0 * epsilon);

    Vec3::new(
        eval(node, point + ex) - eval(node, point - ex),
        eval(node, point + ey) - eval(node, point - ey),
        eval(node, point + ez) - eval(node, point - ez),
    ) * inv_2e
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_eval_sphere() {
        let sphere = SdfNode::sphere(1.0);
        assert!((eval(&sphere, Vec3::ZERO) + 1.0).abs() < 0.0001);
        assert!((eval(&sphere, Vec3::new(1.0, 0.0, 0.0))).abs() < 0.0001);
        assert!((eval(&sphere, Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_eval_union() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0);
        let union = a.union(b);

        // At origin, distance to union is distance to left sphere
        assert!((eval(&union, Vec3::ZERO) + 1.0).abs() < 0.0001);

        // Between spheres (at midpoint x=1.5)
        assert!(eval(&union, Vec3::new(1.5, 0.0, 0.0)) > 0.0);
    }

    #[test]
    fn test_eval_translated() {
        let sphere = SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0);
        assert!((eval(&sphere, Vec3::new(2.0, 0.0, 0.0)) + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_eval_rotated() {
        let box3d = SdfNode::box3d(2.0, 1.0, 1.0).rotate_euler(0.0, PI / 4.0, 0.0);
        // After 45° rotation, the box should extend diagonally
        let d = eval(&box3d, Vec3::ZERO);
        assert!(d < 0.0);
    }

    #[test]
    fn test_eval_scaled() {
        let sphere = SdfNode::sphere(1.0).scale(2.0);
        // Scaled by 2, so radius is now 2
        assert!((eval(&sphere, Vec3::new(2.0, 0.0, 0.0))).abs() < 0.0001);
    }

    #[test]
    fn test_normal() {
        let sphere = SdfNode::sphere(1.0);
        let n = normal(&sphere, Vec3::new(1.0, 0.0, 0.0), 0.001);
        let expected = Vec3::new(1.0, 0.0, 0.0);
        assert!((n - expected).length() < 0.01);
    }

    #[test]
    fn test_gradient() {
        let sphere = SdfNode::sphere(1.0);
        let g = gradient(&sphere, Vec3::new(1.0, 0.0, 0.0), 0.001);
        // Gradient magnitude should be ~1 for proper SDF
        assert!((g.length() - 1.0).abs() < 0.1);
    }
}
