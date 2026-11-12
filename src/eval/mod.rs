//! SDF Evaluation (Deep Fried Edition)
//!
//! Functions for evaluating SDF trees at points.
//!
//! # Deep Fried Optimizations
//! - **Small recursive frame**: `eval` keeps one small value per tree level (see [`eval`]).
//! - **Direct Dispatch**: Primitives are called directly without wrapper overhead.
//! - **Swizzle Normals**: Uses Vec3 swizzles for efficient gradient computation.
//!
//! Author: Moroya Sakamoto

pub mod gradient;
pub mod parallel;
pub mod scaled;
pub mod union_bounded;

use crate::compiled::real::LawMinMax;
use crate::compiled::real::LawVec;
pub use gradient::{eval_gradient, eval_normal};
pub use parallel::{eval_batch, eval_batch_parallel, eval_grid, eval_grid_with_normals};
pub use scaled::eval_scaled;
pub use union_bounded::{eval_union_bounded, eval_union_bounded_stats, UnionBoundStats};

use crate::modifiers::*;
use crate::operations::*;
use crate::primitives::*;
use crate::transforms::*;
use crate::types::SdfNode;
use glam::Vec3;

/// Evaluate an SDF tree at a single point
///
/// Recursively traverses the tree and computes the signed distance.
///
/// The recursion holds only what it needs across a child's evaluation: `descend` reads a node
/// (a leaf's value, or the children and the moved point), the children are evaluated, and
/// `combine` / `finish` apply the node's law to their values. Those three and the leaf
/// evaluator are never inlined here, so each level of a deep tree costs one small frame whatever
/// the number of node kinds or the size of their arms (an unoptimised build gives every arm of a
/// match its own stack slots; a smooth-union chain folded one part at a time, 2 × 24 deep, has to
/// fit a 2 MiB test thread, and `tests/test_eval_stack_budget.rs` holds each level to a byte budget)
///
/// # Arguments
/// * `node` - The SDF tree root
/// * `point` - Point to evaluate
///
/// # Returns
/// Signed distance to the surface
pub fn eval(node: &SdfNode, point: Vec3) -> f32 {
    match descend(node, point) {
        Step::Leaf(d) => d,
        Step::Two(a, b) => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            combine(node, d1, d2, point)
        }
        Step::One(child, q, extra) => finish(node, eval(child, q), point, extra),
    }
}

/// What [`eval`] does at one node
enum Step<'a> {
    /// a node without children: its value
    Leaf(f32),
    /// two operands evaluated at the same point, combined by [`combine`]
    Two(&'a SdfNode, &'a SdfNode),
    /// one child evaluated at a moved point, then [`finish`] with up to three values the move
    /// produced
    One(&'a SdfNode, Vec3, [f32; 3]),
}

/// Reads one node for [`eval`]: a leaf's value, or its children and the point they are evaluated at
#[inline(never)]
fn descend(node: &SdfNode, point: Vec3) -> Step<'_> {
    const NONE: [f32; 3] = [0.0; 3];
    match node {
        SdfNode::Union { a, b }
        | SdfNode::Intersection { a, b }
        | SdfNode::Subtraction { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::SmoothIntersection { a, b, .. }
        | SdfNode::SmoothSubtraction { a, b, .. }
        | SdfNode::ChamferUnion { a, b, .. }
        | SdfNode::ChamferIntersection { a, b, .. }
        | SdfNode::ChamferSubtraction { a, b, .. }
        | SdfNode::StairsUnion { a, b, .. }
        | SdfNode::StairsIntersection { a, b, .. }
        | SdfNode::StairsSubtraction { a, b, .. }
        | SdfNode::XOR { a, b }
        | SdfNode::Morph { a, b, .. }
        | SdfNode::ColumnsUnion { a, b, .. }
        | SdfNode::ColumnsIntersection { a, b, .. }
        | SdfNode::ColumnsSubtraction { a, b, .. }
        | SdfNode::Pipe { a, b, .. }
        | SdfNode::Engrave { a, b, .. }
        | SdfNode::Groove { a, b, .. }
        | SdfNode::Tongue { a, b, .. }
        | SdfNode::ExpSmoothUnion { a, b, .. }
        | SdfNode::ExpSmoothIntersection { a, b, .. }
        | SdfNode::ExpSmoothSubtraction { a, b, .. }
        | SdfNode::SmoothBlend { a, b, .. } => Step::Two(a, b),
        // `inner` first, as the lerp reads them
        SdfNode::MetricBlend { inner, outer, .. } => Step::Two(inner, outer),

        // === Transforms ===
        SdfNode::Translate { child, offset } => {
            // `transform_translate` is `#[inline(always)]`: the same single subtract
            Step::One(child, transform_translate(point, *offset), NONE)
        }
        SdfNode::Rotate { child, rotation } => {
            // Same formula as the compiled / SIMD paths (`Real::rotate_inverse`),
            // not `glam`: bit-identical rotation keeps sign-discontinuous laws
            // (pyramid base) on the same side on every path.
            let p = crate::compiled::real::rotate_inverse::<f32>(*rotation, point.into()).into();
            Step::One(child, p, NONE)
        }
        SdfNode::Scale { child, factor } => {
            // `p * (1/s)`, the compiled evaluators' law (`Instruction::scale`
            // precomputes the reciprocal); `p / s` differs by an ulp and the
            // tree must be bit-identical to them (`tests/test_det_parity.rs`)
            // (`transform_scale` is exactly that law and returns the multiplier `s`)
            let (p, mult) = transform_scale(point, *factor);
            Step::One(child, p, [mult, 0.0, 0.0])
        }
        SdfNode::ScaleNonUniform { child, factors } => {
            let (p, mult) = transform_scale_nonuniform(point, *factors);
            Step::One(child, p, [mult, 0.0, 0.0])
        }
        SdfNode::ProjectiveTransform {
            child,
            inv_matrix,
            lipschitz_bound,
        } => {
            let (q, correction) =
                crate::transforms::projective::projective_transform(point, inv_matrix);
            Step::One(child, q, [correction.law_min(*lipschitz_bound), 0.0, 0.0])
        }
        SdfNode::LatticeDeform {
            child,
            control_points,
            nx,
            ny,
            nz,
            bbox_min,
            bbox_max,
        } => {
            let (q, correction) = crate::transforms::lattice::lattice_deform(
                point,
                control_points,
                *nx,
                *ny,
                *nz,
                *bbox_min,
                *bbox_max,
            );
            Step::One(child, q, [correction, 0.0, 0.0])
        }
        SdfNode::SdfSkinning { child, bones } => {
            let (q, _) = crate::transforms::skinning::sdf_skinning(point, bones);
            Step::One(child, q, NONE)
        }
        SdfNode::GridWarp { child, warp } => {
            let (x, e, lam, fl) = warp.warp(point);
            Step::One(child, x, [e, lam, fl])
        }
        SdfNode::BellPull { child, pull } => Step::One(child, pull.back(point), NONE),

        // === Modifiers ===
        SdfNode::Twist { child, strength } => {
            Step::One(child, modifier_twist(point, *strength), NONE)
        }
        SdfNode::Bend { child, curvature } => {
            Step::One(child, modifier_bend(point, *curvature), NONE)
        }
        SdfNode::RepeatInfinite { child, spacing } => {
            Step::One(child, modifier_repeat_infinite(point, *spacing), NONE)
        }
        SdfNode::RepeatFinite {
            child,
            count,
            spacing,
        } => Step::One(child, modifier_repeat_finite(point, *count, *spacing), NONE),
        SdfNode::RepeatCells {
            child,
            spacing,
            count,
            mirror,
            neighbours,
            variation,
        } => {
            // a direct enum literal reaches here unchecked: the fields whose law has not
            // landed give NaN (no distance), not the single-cell value with the field ignored
            if *neighbours || variation.is_some() {
                Step::Leaf(f32::NAN)
            } else {
                Step::One(
                    child,
                    modifier_repeat_cells(point, *spacing, *count, *mirror),
                    NONE,
                )
            }
        }
        SdfNode::Elongate { child, amount } => {
            let q = point - point.law_clamp(-*amount, *amount);
            Step::One(child, q, NONE)
        }
        SdfNode::Mirror { child, axes } => Step::One(child, modifier_mirror(point, *axes), NONE),
        SdfNode::OctantMirror { child } => Step::One(child, modifier_octant_mirror(point), NONE),
        SdfNode::Revolution { child, offset } => {
            Step::One(child, modifier_revolution(point, *offset), NONE)
        }
        SdfNode::Extrude { child, .. } => Step::One(child, modifier_extrude_point(point), NONE),
        SdfNode::Taper { child, factor, .. } => {
            Step::One(child, modifier_taper(point, *factor), NONE)
        }
        SdfNode::PolarRepeat { child, count } => {
            Step::One(child, modifier_polar_repeat(point, *count), NONE)
        }
        SdfNode::SweepBezier { child, p0, p1, p2 } => {
            Step::One(child, modifier_sweep_bezier(point, *p0, *p1, *p2), NONE)
        }
        SdfNode::Shear { child, shear } => Step::One(child, modifier_shear(point, *shear), NONE),
        SdfNode::IcosahedralSymmetry { child } => {
            Step::One(child, crate::modifiers::icosahedral_fold(point), NONE)
        }
        SdfNode::IFS {
            child,
            transforms,
            iterations,
        } => {
            let (q, scale) = crate::modifiers::ifs_fold_with_scale(point, transforms, *iterations);
            Step::One(child, q, [scale, 0.0, 0.0])
        }
        // the child at the same point; the node's law reads the point in `finish`
        // (Animated: static evaluation, animation is time-dependent; WithMaterial and Pose are
        // transparent for distance evaluation)
        SdfNode::Noise { child, .. }
        | SdfNode::Round { child, .. }
        | SdfNode::Onion { child, .. }
        | SdfNode::Displacement { child, .. }
        | SdfNode::SineDisplacement { child, .. }
        | SdfNode::Animated { child, .. }
        | SdfNode::WithMaterial { child, .. }
        | SdfNode::Pose { child }
        | SdfNode::HeightmapDisplacement { child, .. }
        | SdfNode::SurfaceRoughness { child, .. } => Step::One(child, point, NONE),

        _ => Step::Leaf(eval_leaf(node, point)),
    }
}

/// The law of a two-operand node of [`descend`] on its operands' values
#[inline(never)]
fn combine(node: &SdfNode, d1: f32, d2: f32, point: Vec3) -> f32 {
    match node {
        SdfNode::Union { .. } => sdf_union(d1, d2),
        SdfNode::Intersection { .. } => sdf_intersection(d1, d2),
        SdfNode::Subtraction { .. } => sdf_subtraction(d1, d2),
        SdfNode::SmoothUnion { k, .. } => sdf_smooth_union(d1, d2, *k),
        SdfNode::SmoothIntersection { k, .. } => sdf_smooth_intersection(d1, d2, *k),
        SdfNode::SmoothSubtraction { k, .. } => sdf_smooth_subtraction(d1, d2, *k),
        SdfNode::ChamferUnion { r, .. } => sdf_chamfer_union(d1, d2, *r),
        SdfNode::ChamferIntersection { r, .. } => sdf_chamfer_intersection(d1, d2, *r),
        SdfNode::ChamferSubtraction { r, .. } => sdf_chamfer_subtraction(d1, d2, *r),
        SdfNode::StairsUnion { r, n, .. } => sdf_stairs_union(d1, d2, *r, *n),
        SdfNode::StairsIntersection { r, n, .. } => sdf_stairs_intersection(d1, d2, *r, *n),
        SdfNode::StairsSubtraction { r, n, .. } => sdf_stairs_subtraction(d1, d2, *r, *n),
        SdfNode::XOR { .. } => sdf_xor(d1, d2),
        SdfNode::Morph { t, .. } => sdf_morph(d1, d2, *t),
        SdfNode::ColumnsUnion { r, n, .. } => sdf_columns_union(d1, d2, *r, *n),
        SdfNode::ColumnsIntersection { r, n, .. } => sdf_columns_intersection(d1, d2, *r, *n),
        SdfNode::ColumnsSubtraction { r, n, .. } => sdf_columns_subtraction(d1, d2, *r, *n),
        SdfNode::Pipe { r, .. } => sdf_pipe(d1, d2, *r),
        SdfNode::Engrave { r, .. } => sdf_engrave(d1, d2, *r),
        SdfNode::Groove { ra, rb, .. } => sdf_groove(d1, d2, *ra, *rb),
        SdfNode::Tongue { ra, rb, .. } => sdf_tongue(d1, d2, *ra, *rb),
        SdfNode::ExpSmoothUnion { k, .. } => sdf_exp_smooth_union(d1, d2, *k),
        SdfNode::ExpSmoothIntersection { k, .. } => sdf_exp_smooth_intersection(d1, d2, *k),
        SdfNode::ExpSmoothSubtraction { k, .. } => sdf_exp_smooth_subtraction(d1, d2, *k),
        SdfNode::SmoothBlend {
            k, kernel, mode, ..
        } => smooth_blend(d1, d2, *k, *kernel, *mode),
        SdfNode::MetricBlend {
            center,
            radius,
            skin,
            ..
        } => {
            // exactly `outer` beyond the skin and exactly `inner` within it:
            // the weight saturates and the interpolation is endpoint-exact
            let t = alice_det_math::metric::smoothstep(
                *radius,
                *radius + *skin,
                (point - *center).length(),
            );
            alice_det_math::metric::lerp(d1, d2, t)
        }
        _ => unreachable!("combine: descend gives two operands only for the nodes above"),
    }
}

/// The law of a one-child node of [`descend`] on its child's value `d` (`point` is the node's
/// point, `extra` what `descend` computed while moving it)
#[inline(never)]
fn finish(node: &SdfNode, d: f32, point: Vec3, extra: [f32; 3]) -> f32 {
    match node {
        SdfNode::Scale { .. } | SdfNode::ScaleNonUniform { .. } => d * extra[0],
        SdfNode::ProjectiveTransform { .. } => d * extra[0],
        SdfNode::LatticeDeform { .. } => d / extra[0],
        SdfNode::GridWarp { warp, .. } => warp.finish(d, extra[0], extra[1], extra[2]),
        SdfNode::BellPull { pull, .. } => d / pull.lambda(),
        SdfNode::IFS { .. } => d / extra[0].law_max(1e-6),
        SdfNode::Noise {
            amplitude,
            frequency,
            seed,
            ..
        } => modifier_noise_perlin(d, point, *amplitude, *frequency, *seed),
        // Direct arithmetic operations (no function call overhead)
        SdfNode::Round { radius, .. } => d - radius,
        SdfNode::Onion { thickness, .. } => d.abs() - thickness,
        SdfNode::Extrude { half_height, .. } => modifier_extrude(d, point.z, *half_height),
        SdfNode::Taper { factor, reach, .. } => {
            crate::compiled::real::taper_bound::<f32>(d, point.into(), *factor, *reach)
        }
        SdfNode::Displacement { strength, .. } => modifier_displacement(d, point, *strength),
        SdfNode::SineDisplacement {
            amplitude,
            frequency,
            ..
        } => crate::modifiers::modifier_sine_displacement(d, point, *amplitude, *frequency),
        SdfNode::HeightmapDisplacement {
            heightmap,
            width,
            height,
            amplitude,
            scale,
            ..
        } => {
            let disp = crate::modifiers::heightmap_displacement(
                point, heightmap, *width, *height, *amplitude, *scale,
            );
            d - disp
        }
        SdfNode::SurfaceRoughness {
            frequency,
            amplitude,
            octaves,
            ..
        } => crate::modifiers::surface_roughness(point, d, *frequency, *amplitude, *octaves),
        // the child's value as it is (a rigid move, a fold, a repetition, or a transparent node)
        _ => d,
    }
}

/// The value of a node without children
#[inline(never)]
fn eval_leaf(node: &SdfNode, point: Vec3) -> f32 {
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
        SdfNode::RoundedCone {
            r1,
            r2,
            half_height,
        } => sdf_rounded_cone(point, *r1, *r2, *half_height),
        SdfNode::Pyramid { half_height } => sdf_pyramid(point, *half_height),
        SdfNode::Octahedron { size } => sdf_octahedron(point, *size),
        SdfNode::HexPrism {
            hex_radius,
            half_height,
        } => sdf_hex_prism(point, *hex_radius, *half_height),
        SdfNode::Link {
            half_length,
            r1,
            r2,
        } => sdf_link(point, *half_length, *r1, *r2),
        SdfNode::Triangle {
            point_a,
            point_b,
            point_c,
        } => sdf_triangle(point, *point_a, *point_b, *point_c),
        SdfNode::Bezier {
            point_a,
            point_b,
            point_c,
            radius,
        } => sdf_bezier(point, *point_a, *point_b, *point_c, *radius),
        SdfNode::RoundedBox {
            half_extents,
            round_radius,
        } => sdf_rounded_box(point, *half_extents, *round_radius),
        SdfNode::CappedCone {
            half_height,
            r1,
            r2,
        } => sdf_capped_cone(point, *half_height, *r1, *r2),
        SdfNode::CappedTorus {
            major_radius,
            minor_radius,
            cap_angle,
        } => sdf_capped_torus(point, *major_radius, *minor_radius, *cap_angle),
        SdfNode::RoundedCylinder {
            radius,
            round_radius,
            half_height,
        } => sdf_rounded_cylinder(point, *radius, *round_radius, *half_height),
        SdfNode::TriangularPrism { width, half_depth } => {
            sdf_triangular_prism(point, *width, *half_depth)
        }
        SdfNode::CutSphere { radius, cut_height } => sdf_cut_sphere(point, *radius, *cut_height),
        SdfNode::CutHollowSphere {
            radius,
            cut_height,
            thickness,
        } => sdf_cut_hollow_sphere(point, *radius, *cut_height, *thickness),
        SdfNode::DeathStar { ra, rb, d } => sdf_death_star(point, *ra, *rb, *d),
        SdfNode::SolidAngle { angle, radius } => sdf_solid_angle(point, *angle, *radius),
        SdfNode::Rhombus {
            la,
            lb,
            half_height,
            round_radius,
        } => sdf_rhombus(point, *la, *lb, *half_height, *round_radius),
        SdfNode::Horseshoe {
            angle,
            radius,
            half_length,
            width,
            thickness,
        } => sdf_horseshoe(point, *angle, *radius, *half_length, *width, *thickness),
        SdfNode::Vesica { radius, half_dist } => sdf_vesica(point, *radius, *half_dist),
        SdfNode::InfiniteCylinder { radius } => sdf_infinite_cylinder(point, *radius),
        SdfNode::InfiniteCone { angle } => sdf_infinite_cone(point, *angle),
        SdfNode::Gyroid { scale, thickness } => sdf_gyroid(point, *scale, *thickness),

        SdfNode::Loft {
            axis,
            up,
            harmonics,
            rows,
            caps,
            bounds,
        } => crate::primitives::loft::sdf_loft(
            point,
            axis,
            *up,
            crate::primitives::loft::LoftSection::Polar(*harmonics),
            rows,
            &[],
            *caps,
            bounds,
        ),
        SdfNode::LoftCurves {
            axis,
            up,
            section_points,
            points,
            knots,
            caps,
            bounds,
        } => crate::primitives::loft::sdf_loft(
            point,
            axis,
            *up,
            crate::primitives::loft::LoftSection::Curve(*section_points),
            points,
            knots,
            [caps[0][0], caps[1][0]],
            bounds,
        ),
        SdfNode::MetricBall {
            radius,
            w_l1,
            w_l2,
            w_linf,
        } => sdf_metric_ball(point, *radius, *w_l1, *w_l2, *w_linf),

        SdfNode::Heart { size } => sdf_heart(point, *size),
        SdfNode::Tube {
            outer_radius,
            thickness,
            half_height,
        } => sdf_tube(point, *outer_radius, *thickness, *half_height),
        SdfNode::Barrel {
            radius,
            half_height,
            bulge,
        } => sdf_barrel(point, *radius, *half_height, *bulge),
        SdfNode::Diamond {
            radius,
            half_height,
        } => sdf_diamond(point, *radius, *half_height),
        SdfNode::ChamferedCube {
            half_extents,
            chamfer,
        } => sdf_chamfered_cube(point, *half_extents, *chamfer),
        SdfNode::SchwarzP { scale, thickness } => sdf_schwarz_p(point, *scale, *thickness),
        SdfNode::Superellipsoid {
            half_extents,
            e1,
            e2,
        } => sdf_superellipsoid(point, *half_extents, *e1, *e2),
        SdfNode::RoundedX {
            width,
            round_radius,
            half_height,
        } => sdf_rounded_x(point, *width, *round_radius, *half_height),
        SdfNode::Pie {
            angle,
            radius,
            half_height,
        } => sdf_pie(point, *angle, *radius, *half_height),
        SdfNode::Trapezoid {
            r1,
            r2,
            trap_height,
            half_depth,
        } => sdf_trapezoid(point, *r1, *r2, *trap_height, *half_depth),
        SdfNode::Parallelogram {
            width,
            para_height,
            skew,
            half_depth,
        } => sdf_parallelogram(point, *width, *para_height, *skew, *half_depth),
        SdfNode::Tunnel {
            width,
            height_2d,
            half_depth,
        } => sdf_tunnel(point, *width, *height_2d, *half_depth),
        SdfNode::UnevenCapsule {
            r1,
            r2,
            cap_height,
            half_depth,
        } => sdf_uneven_capsule(point, *r1, *r2, *cap_height, *half_depth),
        SdfNode::Egg { ra, rb } => sdf_egg(point, *ra, *rb),
        SdfNode::ArcShape {
            aperture,
            radius,
            thickness,
            half_height,
        } => sdf_arc_shape(point, *aperture, *radius, *thickness, *half_height),
        SdfNode::Moon {
            d,
            ra,
            rb,
            half_height,
        } => sdf_moon(point, *d, *ra, *rb, *half_height),
        SdfNode::CrossShape {
            length,
            thickness,
            round_radius,
            half_height,
        } => sdf_cross_shape(point, *length, *thickness, *round_radius, *half_height),
        SdfNode::BlobbyCross { size, half_height } => sdf_blobby_cross(point, *size, *half_height),
        SdfNode::ParabolaSegment {
            width,
            para_height,
            half_depth,
        } => sdf_parabola_segment(point, *width, *para_height, *half_depth),
        SdfNode::RegularPolygon {
            radius,
            n_sides,
            half_height,
        } => sdf_regular_polygon(point, *radius, *n_sides, *half_height),
        SdfNode::StarPolygon {
            radius,
            n_points,
            m,
            half_height,
        } => sdf_star_polygon(point, *radius, *n_points, *m, *half_height),
        SdfNode::Stairs {
            step_width,
            step_height,
            n_steps,
            half_depth,
        } => sdf_stairs(point, *step_width, *step_height, *n_steps, *half_depth),
        SdfNode::Helix {
            major_r,
            minor_r,
            pitch,
            half_height,
        } => sdf_helix(point, *major_r, *minor_r, *pitch, *half_height),
        SdfNode::Tetrahedron { size } => sdf_tetrahedron(point, *size),
        SdfNode::Dodecahedron { radius } => sdf_dodecahedron(point, *radius),
        SdfNode::Icosahedron { radius } => sdf_icosahedron(point, *radius),
        SdfNode::TruncatedOctahedron { radius } => sdf_truncated_octahedron(point, *radius),
        SdfNode::TruncatedIcosahedron { radius } => sdf_truncated_icosahedron(point, *radius),
        SdfNode::BoxFrame { half_extents, edge } => sdf_box_frame(point, *half_extents, *edge),
        SdfNode::DiamondSurface { scale, thickness } => {
            sdf_diamond_surface(point, *scale, *thickness)
        }
        SdfNode::Neovius { scale, thickness } => sdf_neovius(point, *scale, *thickness),
        SdfNode::Lidinoid { scale, thickness } => sdf_lidinoid(point, *scale, *thickness),
        SdfNode::IWP { scale, thickness } => sdf_iwp(point, *scale, *thickness),
        SdfNode::FRD { scale, thickness } => sdf_frd(point, *scale, *thickness),
        SdfNode::FischerKochS { scale, thickness } => sdf_fischer_koch_s(point, *scale, *thickness),
        SdfNode::PMY { scale, thickness } => sdf_pmy(point, *scale, *thickness),

        // === 2D Primitives (extruded along Z) ===
        SdfNode::Circle2D {
            radius,
            half_height,
        } => sdf_circle_2d(point, *radius, *half_height),
        SdfNode::Rect2D {
            half_extents,
            half_height,
        } => sdf_rect_2d(point, *half_extents, *half_height),
        SdfNode::Segment2D {
            a,
            b,
            thickness,
            half_height,
        } => sdf_segment_2d(point, *a, *b, *thickness, *half_height),
        SdfNode::SectionStack(st) => st.eval(point),
        SdfNode::SmoothSectionStack(st) => st.eval(point),
        SdfNode::Polygon2D {
            vertices,
            half_height,
        } => sdf_polygon_2d(point, vertices, *half_height),
        SdfNode::RoundedRect2D {
            half_extents,
            round_radius,
            half_height,
        } => sdf_rounded_rect_2d(point, *half_extents, *round_radius, *half_height),
        SdfNode::Annular2D {
            outer_radius,
            thickness,
            half_height,
        } => sdf_annular_2d(point, *outer_radius, *thickness, *half_height),

        SdfNode::Terrain { scale, amplitude } => {
            // FBM-based terrain: y - terrainHeight(xz), 3 octaves of the
            // portable `hash_noise_3d` lattice noise sampled on the xz plane
            // (y held at 0, so the trilinear blend degenerates to bilinear —
            // see the shader emit in transpiler_common.rs for the identity).
            // Bit-exact with GLSL/WGSL/HLSL (PCG hash), unlike the previous
            // `fract(sin(·)·43758)` law the shader could not reproduce.
            let mut v = 0.0_f32;
            let mut a_fbm = 0.5_f32;
            let mut freq_x = point.x * scale;
            let mut freq_z = point.z * scale;
            for _ in 0..3 {
                let n = hash_noise_3d(Vec3::new(freq_x, 0.0, freq_z), TERRAIN_NOISE_SEED);
                v += a_fbm * n;
                let nx = 0.6f32 * freq_z + (0.8 * freq_x);
                let nz = 0.8f32 * freq_z + (-0.6 * freq_x);
                freq_x = nx * 2.1;
                freq_z = nz * 2.1;
                a_fbm *= 0.48;
            }
            point.y - v * amplitude
        }

        #[allow(unreachable_patterns)]
        _ => f32::MAX, // Unknown variant: treat as infinitely far (outside any surface)
    }
}

/// The rest point of `point`: where a colour or texture law defined on the unposed model reads its value
///
/// The point is in the frame of `node`. It follows the branch a union (the nearer child) or an
/// intersection (the farther child) selects and the first operand of a subtraction; it undoes the
/// pose nodes (the grid warp's map, and the rigid motion right below a [`SdfNode::Pose`]: a
/// translate over a rotate, the form `rigid_pose` builds; any other shape under a pose is `None`); a translate / rotate / scale elsewhere places a part built in its own frame,
/// so the rest point below it is mapped back through it into `node`'s frame. `None` when the tree
/// has a node that moves points in a way this does not follow (twist, bend, repetition, mirror, ...)
#[must_use]
pub fn eval_rest_point(node: &SdfNode, point: Vec3) -> Option<Vec3> {
    let pick = |a: &SdfNode, b: &SdfNode, nearer: bool| {
        let (da, db) = (eval(a, point), eval(b, point));
        if (da <= db) == nearer {
            eval_rest_point(a, point)
        } else {
            eval_rest_point(b, point)
        }
    };
    match node {
        SdfNode::Union { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::ChamferUnion { a, b, .. }
        | SdfNode::StairsUnion { a, b, .. }
        | SdfNode::ColumnsUnion { a, b, .. }
        | SdfNode::ExpSmoothUnion { a, b, .. } => pick(a, b, true),
        SdfNode::Intersection { a, b }
        | SdfNode::SmoothIntersection { a, b, .. }
        | SdfNode::ChamferIntersection { a, b, .. }
        | SdfNode::StairsIntersection { a, b, .. }
        | SdfNode::ColumnsIntersection { a, b, .. }
        | SdfNode::ExpSmoothIntersection { a, b, .. } => {
            // the farther child: `>=` picks a on ties, as the shader does
            let (da, db) = (eval(a, point), eval(b, point));
            if da >= db {
                eval_rest_point(a, point)
            } else {
                eval_rest_point(b, point)
            }
        }
        SdfNode::Subtraction { a, .. }
        | SdfNode::SmoothSubtraction { a, .. }
        | SdfNode::ChamferSubtraction { a, .. }
        | SdfNode::StairsSubtraction { a, .. }
        | SdfNode::ColumnsSubtraction { a, .. }
        | SdfNode::ExpSmoothSubtraction { a, .. } => eval_rest_point(a, point),
        SdfNode::SmoothBlend { a, b, mode, .. } => match mode {
            BlendMode::Union => pick(a, b, true),
            BlendMode::Intersection => {
                // the farther child, as for the other intersections
                let (da, db) = (eval(a, point), eval(b, point));
                if da >= db {
                    eval_rest_point(a, point)
                } else {
                    eval_rest_point(b, point)
                }
            }
            BlendMode::Subtraction => eval_rest_point(a, point),
        },
        SdfNode::WithMaterial { child, .. } => eval_rest_point(child, point),
        SdfNode::Translate { child, offset } => {
            eval_rest_point(child, point - *offset).map(|r| r + *offset)
        }
        SdfNode::Rotate { child, rotation } => {
            eval_rest_point(child, rotation.inverse() * point).map(|r| *rotation * r)
        }
        SdfNode::Scale { child, factor } => {
            eval_rest_point(child, point * (1.0 / factor)).map(|r| r * *factor)
        }
        SdfNode::Pose { child } => eval_rest_point_posed(child, point),
        SdfNode::GridWarp { child, warp } => eval_rest_point(child, warp.warp(point).0),
        SdfNode::BellPull { child, pull } => eval_rest_point(child, pull.back(point)),
        n if crate::diff::node_children(n).is_empty() => Some(point),
        _ => None,
    }
}

/// The rest point below a [`SdfNode::Pose`]: its rigid motion (the translate right below it and
/// the rotate right below that) is undone, not mapped back; the part under the rotate is treated
/// as everywhere else. `None` for any other shape under a pose (the form `rigid_pose` builds is
/// the only one whose motion is not ambiguous)
fn eval_rest_point_posed(node: &SdfNode, point: Vec3) -> Option<Vec3> {
    match node {
        SdfNode::Translate { child, offset } => match child.as_ref() {
            SdfNode::Rotate { child, rotation } => {
                eval_rest_point(child, rotation.inverse() * (point - *offset))
            }
            _ => None,
        },
        _ => None,
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
        SdfNode::Pose { child } => eval_material(child, point),
        SdfNode::WithMaterial { child, material_id } => {
            // This subtree has a material; return it
            // (nested WithMaterial: inner wins if closer)
            let inner = eval_material(child, point);
            if inner != 0 {
                inner
            } else {
                *material_id
            }
        }

        // Operations: return material of the closer child
        SdfNode::Union { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::ChamferUnion { a, b, .. }
        | SdfNode::StairsUnion { a, b, .. }
        | SdfNode::XOR { a, b }
        | SdfNode::Morph { a, b, .. }
        | SdfNode::ColumnsUnion { a, b, .. }
        | SdfNode::Pipe { a, b, .. }
        | SdfNode::ExpSmoothUnion { a, b, .. } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da <= db {
                eval_material(a, point)
            } else {
                eval_material(b, point)
            }
        }
        SdfNode::Intersection { a, b }
        | SdfNode::SmoothIntersection { a, b, .. }
        | SdfNode::ChamferIntersection { a, b, .. }
        | SdfNode::StairsIntersection { a, b, .. }
        | SdfNode::ColumnsIntersection { a, b, .. }
        | SdfNode::ExpSmoothIntersection { a, b, .. } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da >= db {
                eval_material(a, point)
            } else {
                eval_material(b, point)
            }
        }
        SdfNode::Subtraction { a, b }
        | SdfNode::SmoothSubtraction { a, b, .. }
        | SdfNode::ChamferSubtraction { a, b, .. }
        | SdfNode::StairsSubtraction { a, b, .. }
        | SdfNode::ColumnsSubtraction { a, b, .. }
        | SdfNode::Engrave { a, b, .. }
        | SdfNode::Groove { a, b, .. }
        | SdfNode::Tongue { a, b, .. }
        | SdfNode::ExpSmoothSubtraction { a, b, .. } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da >= -db {
                eval_material(a, point)
            } else {
                eval_material(b, point)
            }
        }
        SdfNode::SmoothBlend { a, b, mode, .. } => {
            // the same choice as the plain operator of the same mode
            let da = eval(a, point);
            let db = eval(b, point);
            let take_a = match mode {
                BlendMode::Union => da <= db,
                BlendMode::Intersection => da >= db,
                BlendMode::Subtraction => da >= -db,
            };
            if take_a {
                eval_material(a, point)
            } else {
                eval_material(b, point)
            }
        }

        // Transforms: transform point, recurse
        SdfNode::Translate { child, offset } => {
            eval_material(child, transform_translate(point, *offset))
        }
        SdfNode::Rotate { child, rotation } => eval_material(
            child,
            crate::compiled::real::rotate_inverse::<f32>(*rotation, point.into()).into(),
        ),
        SdfNode::Scale { child, factor } => eval_material(child, transform_scale(point, *factor).0),
        SdfNode::ScaleNonUniform { child, factors } => {
            let (p, _) = transform_scale_nonuniform(point, *factors);
            eval_material(child, p)
        }

        // Modifiers: transform point, recurse
        SdfNode::Twist { child, strength } => {
            eval_material(child, modifier_twist(point, *strength))
        }
        SdfNode::Bend { child, curvature } => {
            eval_material(child, modifier_bend(point, *curvature))
        }
        SdfNode::RepeatInfinite { child, spacing } => {
            eval_material(child, modifier_repeat_infinite(point, *spacing))
        }
        SdfNode::RepeatCells {
            child,
            spacing,
            count,
            mirror,
            ..
        } => eval_material(
            child,
            modifier_repeat_cells(point, *spacing, *count, *mirror),
        ),
        SdfNode::RepeatFinite {
            child,
            count,
            spacing,
        } => eval_material(child, modifier_repeat_finite(point, *count, *spacing)),
        SdfNode::Noise { child, .. } => eval_material(child, point),
        SdfNode::Round { child, .. } | SdfNode::Onion { child, .. } => eval_material(child, point),
        SdfNode::Elongate { child, amount } => {
            eval_material(child, point - point.law_clamp(-*amount, *amount))
        }
        SdfNode::Mirror { child, axes } => eval_material(child, modifier_mirror(point, *axes)),
        SdfNode::OctantMirror { child } => eval_material(child, modifier_octant_mirror(point)),
        SdfNode::Revolution { child, offset } => {
            eval_material(child, modifier_revolution(point, *offset))
        }
        SdfNode::Extrude { child, .. } => eval_material(child, modifier_extrude_point(point)),
        SdfNode::Taper { child, factor, .. } => {
            eval_material(child, modifier_taper(point, *factor))
        }
        SdfNode::Displacement { child, .. } => eval_material(child, point),
        SdfNode::SineDisplacement { child, .. } => eval_material(child, point),
        SdfNode::PolarRepeat { child, count } => {
            eval_material(child, modifier_polar_repeat(point, *count))
        }
        SdfNode::SweepBezier { child, p0, p1, p2 } => {
            eval_material(child, modifier_sweep_bezier(point, *p0, *p1, *p2))
        }
        SdfNode::Shear { child, shear } => {
            let p = Vec3::new(
                point.x,
                shear.x * -point.x + point.y,
                shear.z * -point.y + (shear.y * -point.x + point.z),
            );
            eval_material(child, p)
        }
        SdfNode::Animated { child, .. } => eval_material(child, point),
        SdfNode::IcosahedralSymmetry { child } => {
            eval_material(child, crate::modifiers::icosahedral_fold(point))
        }
        SdfNode::IFS {
            child,
            transforms,
            iterations,
        } => {
            let (q, _) = crate::modifiers::ifs_fold_with_scale(point, transforms, *iterations);
            eval_material(child, q)
        }
        SdfNode::HeightmapDisplacement { child, .. } => eval_material(child, point),
        SdfNode::SurfaceRoughness { child, .. } => eval_material(child, point),
        SdfNode::GridWarp { child, warp } => eval_material(child, warp.warp(point).0),
        SdfNode::BellPull { child, pull } => eval_material(child, pull.back(point)),

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
