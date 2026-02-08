//! Analytic gradient computation for SDF trees (Deep Fried Edition)
//!
//! Instead of 6 finite-difference evaluations of the entire tree,
//! computes the gradient in a single pass using:
//! - Chain rules for operations (Union, Smooth, Chamfer, etc.)
//! - Jacobian transpose for transforms (Translate, Rotate, Scale)
//! - Analytic derivatives for common primitives (Sphere, Box, Torus, etc.)
//! - Numerical fallback for complex primitives (leaf-level, ~6 cheap evals)
//!
//! # Performance
//!
//! For a tree of depth D with N leaf nodes, full numerical gradient requires
//! `6 * (cost of full tree eval)` evaluations. Analytic gradient requires
//! at most `N` leaf evaluations plus `~6` per complex leaf.
//!
//! Author: Moroya Sakamoto

use crate::eval::eval;
use crate::modifiers::*;
use crate::primitives::*;
use crate::types::SdfNode;
use glam::Vec3;
use std::f32::consts::FRAC_1_SQRT_2;

// ============================================================
// Main entry point
// ============================================================

/// Compute the analytic gradient of an SDF tree at a point.
///
/// Uses chain rules through operations and Jacobian transforms
/// through spatial warps, with analytic derivatives for common
/// primitives and numerical fallback for complex ones.
///
/// The gradient vector is NOT normalized. For an exact SDF,
/// `|∇f| ≈ 1` everywhere. Normalize the result if you need a
/// surface normal.
#[inline]
pub fn eval_gradient(node: &SdfNode, point: Vec3) -> Vec3 {
    match node {
        // === Primitives with analytic gradients ===
        SdfNode::Sphere { .. } => grad_sphere(point),
        SdfNode::Box3d { half_extents } => grad_box3d(point, *half_extents),
        SdfNode::Plane { normal, .. } => *normal,
        SdfNode::Cylinder { radius, half_height } => grad_cylinder(point, *radius, *half_height),
        SdfNode::Torus { major_radius, minor_radius } => grad_torus(point, *major_radius, *minor_radius),
        SdfNode::Capsule { point_a, point_b, .. } => grad_capsule(point, *point_a, *point_b),
        SdfNode::InfiniteCylinder { .. } => grad_infinite_cylinder(point),
        SdfNode::Gyroid { scale, thickness } => grad_gyroid(point, *scale, *thickness),
        SdfNode::SchwarzP { scale, thickness } => grad_schwarz_p(point, *scale, *thickness),

        // === Complex primitives: numerical fallback ===
        // 6 local eval calls on a leaf node — negligible cost
        SdfNode::Cone { .. }
        | SdfNode::Ellipsoid { .. }
        | SdfNode::RoundedCone { .. }
        | SdfNode::Pyramid { .. }
        | SdfNode::Octahedron { .. }
        | SdfNode::HexPrism { .. }
        | SdfNode::Link { .. }
        | SdfNode::Triangle { .. }
        | SdfNode::Bezier { .. }
        | SdfNode::RoundedBox { .. }
        | SdfNode::CappedCone { .. }
        | SdfNode::CappedTorus { .. }
        | SdfNode::RoundedCylinder { .. }
        | SdfNode::TriangularPrism { .. }
        | SdfNode::CutSphere { .. }
        | SdfNode::CutHollowSphere { .. }
        | SdfNode::DeathStar { .. }
        | SdfNode::SolidAngle { .. }
        | SdfNode::Rhombus { .. }
        | SdfNode::Horseshoe { .. }
        | SdfNode::Vesica { .. }
        | SdfNode::InfiniteCone { .. }
        | SdfNode::Heart { .. }
        | SdfNode::Tube { .. }
        | SdfNode::Barrel { .. }
        | SdfNode::Diamond { .. }
        | SdfNode::ChamferedCube { .. }
        | SdfNode::Superellipsoid { .. }
        | SdfNode::RoundedX { .. }
        | SdfNode::Pie { .. }
        | SdfNode::Trapezoid { .. }
        | SdfNode::Parallelogram { .. }
        | SdfNode::Tunnel { .. }
        | SdfNode::UnevenCapsule { .. }
        | SdfNode::Egg { .. }
        | SdfNode::ArcShape { .. }
        | SdfNode::Moon { .. }
        | SdfNode::CrossShape { .. }
        | SdfNode::BlobbyCross { .. }
        | SdfNode::ParabolaSegment { .. }
        | SdfNode::RegularPolygon { .. }
        | SdfNode::StarPolygon { .. }
        | SdfNode::Stairs { .. }
        | SdfNode::Helix { .. }
        | SdfNode::Tetrahedron { .. }
        | SdfNode::Dodecahedron { .. }
        | SdfNode::Icosahedron { .. }
        | SdfNode::TruncatedOctahedron { .. }
        | SdfNode::TruncatedIcosahedron { .. }
        | SdfNode::BoxFrame { .. }
        | SdfNode::DiamondSurface { .. }
        | SdfNode::Neovius { .. }
        | SdfNode::Lidinoid { .. }
        | SdfNode::IWP { .. }
        | SdfNode::FRD { .. }
        | SdfNode::FischerKochS { .. }
        | SdfNode::PMY { .. } => numerical_gradient_of(node, point),

        // === Operations: chain rule ===
        SdfNode::Union { a, b } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da <= db { eval_gradient(a, point) } else { eval_gradient(b, point) }
        }
        SdfNode::Intersection { a, b } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da >= db { eval_gradient(a, point) } else { eval_gradient(b, point) }
        }
        SdfNode::Subtraction { a, b } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da >= -db { eval_gradient(a, point) } else { -eval_gradient(b, point) }
        }
        SdfNode::SmoothUnion { a, b, k } => {
            let da = eval(a, point);
            let db = eval(b, point);
            let ga = eval_gradient(a, point);
            let gb = eval_gradient(b, point);
            let (wa, wb) = smooth_min_weights(da, db, *k);
            ga * wa + gb * wb
        }
        SdfNode::SmoothIntersection { a, b, k } => {
            let da = eval(a, point);
            let db = eval(b, point);
            let ga = eval_gradient(a, point);
            let gb = eval_gradient(b, point);
            let (wa, wb) = smooth_max_weights(da, db, *k);
            ga * wa + gb * wb
        }
        SdfNode::SmoothSubtraction { a, b, k } => {
            let da = eval(a, point);
            let db = eval(b, point);
            let ga = eval_gradient(a, point);
            let gb = eval_gradient(b, point);
            let (wa, wb) = smooth_max_weights(da, -db, *k);
            ga * wa - gb * wb
        }
        SdfNode::ChamferUnion { a, b, r } => {
            let da = eval(a, point);
            let db = eval(b, point);
            let ga = eval_gradient(a, point);
            let gb = eval_gradient(b, point);
            grad_chamfer_min(da, db, *r, ga, gb)
        }
        SdfNode::ChamferIntersection { a, b, r } => {
            let da = eval(a, point);
            let db = eval(b, point);
            let ga = eval_gradient(a, point);
            let gb = eval_gradient(b, point);
            -grad_chamfer_min(-da, -db, *r, -ga, -gb)
        }
        SdfNode::ChamferSubtraction { a, b, r } => {
            let da = eval(a, point);
            let db = eval(b, point);
            let ga = eval_gradient(a, point);
            let gb = eval_gradient(b, point);
            -grad_chamfer_min(-da, db, *r, -ga, gb)
        }
        // Stairs operations: numerical fallback (complex modular math)
        SdfNode::StairsUnion { .. }
        | SdfNode::StairsIntersection { .. }
        | SdfNode::StairsSubtraction { .. } => numerical_gradient_of(node, point),
        // XOR: piecewise — gradient of whichever term wins
        SdfNode::XOR { a, b } => {
            let da = eval(a, point);
            let db = eval(b, point);
            let min_ab = da.min(db);
            let neg_max_ab = -da.max(db);
            if min_ab > neg_max_ab {
                // min(a,b) wins
                if da <= db { eval_gradient(a, point) } else { eval_gradient(b, point) }
            } else {
                // -max(a,b) wins
                if da >= db { -eval_gradient(a, point) } else { -eval_gradient(b, point) }
            }
        }
        // Morph: linear blend
        SdfNode::Morph { a, b, t } => {
            let ga = eval_gradient(a, point);
            let gb = eval_gradient(b, point);
            ga * (1.0 - *t) + gb * *t
        }
        // Columns/Pipe/Engrave/Groove/Tongue: complex — numerical fallback
        SdfNode::ColumnsUnion { .. }
        | SdfNode::ColumnsIntersection { .. }
        | SdfNode::ColumnsSubtraction { .. }
        | SdfNode::Pipe { .. }
        | SdfNode::Engrave { .. }
        | SdfNode::Groove { .. }
        | SdfNode::Tongue { .. } => numerical_gradient_of(node, point),

        // === Transforms: Jacobian propagation ===
        SdfNode::Translate { child, offset } => {
            // Translation is isometric: gradient is unchanged
            eval_gradient(child, point - *offset)
        }
        SdfNode::Rotate { child, rotation } => {
            // f(p) = eval(child, R⁻¹p)
            // ∇f = R · ∇child(R⁻¹p)
            let p = rotation.conjugate() * point;
            let grad = eval_gradient(child, p);
            *rotation * grad
        }
        SdfNode::Scale { child, factor } => {
            // f(p) = s · eval(child, p/s)
            // ∇f = s · (1/s) · ∇child(p/s) = ∇child(p/s)
            eval_gradient(child, point / *factor)
        }
        SdfNode::ScaleNonUniform { child, factors } => {
            // f(p) = m · eval(child, p/factors), m = min(factors)
            // ∂f/∂pᵢ = m / factorsᵢ · ∂child/∂qᵢ
            let p = point / *factors;
            let grad = eval_gradient(child, p);
            let m = factors.x.min(factors.y.min(factors.z));
            Vec3::new(
                m * grad.x / factors.x,
                m * grad.y / factors.y,
                m * grad.z / factors.z,
            )
        }

        // === Modifiers with analytic Jacobians ===
        SdfNode::Round { child, .. } => {
            // f = eval(child, p) - radius → ∇f = ∇child
            eval_gradient(child, point)
        }
        SdfNode::Onion { child, .. } => {
            // f = |eval(child, p)| - thickness → ∇f = sign(d) · ∇child
            let d = eval(child, point);
            let grad = eval_gradient(child, point);
            if d >= 0.0 { grad } else { -grad }
        }
        SdfNode::Elongate { child, amount } => {
            // q = p - clamp(p, -a, a); ∂qᵢ/∂pᵢ = 1 if |pᵢ|>aᵢ, else 0
            let q = point - point.clamp(-*amount, *amount);
            let grad = eval_gradient(child, q);
            Vec3::new(
                if point.x.abs() > amount.x { grad.x } else { 0.0 },
                if point.y.abs() > amount.y { grad.y } else { 0.0 },
                if point.z.abs() > amount.z { grad.z } else { 0.0 },
            )
        }
        SdfNode::Mirror { child, axes } => {
            // abs(p) flips gradient sign for negative coordinates
            let p = modifier_mirror(point, *axes);
            let grad = eval_gradient(child, p);
            Vec3::new(
                if axes.x != 0.0 && point.x < 0.0 { -grad.x } else { grad.x },
                if axes.y != 0.0 && point.y < 0.0 { -grad.y } else { grad.y },
                if axes.z != 0.0 && point.z < 0.0 { -grad.z } else { grad.z },
            )
        }
        SdfNode::OctantMirror { child } => {
            // Octant mirror has complex Jacobian (abs + sort permutation).
            // Use numerical gradient for correctness.
            let eps = 1e-4;
            Vec3::new(
                (eval(node, point + Vec3::X * eps) - eval(node, point - Vec3::X * eps)) / (2.0 * eps),
                (eval(node, point + Vec3::Y * eps) - eval(node, point - Vec3::Y * eps)) / (2.0 * eps),
                (eval(node, point + Vec3::Z * eps) - eval(node, point - Vec3::Z * eps)) / (2.0 * eps),
            )
        }
        SdfNode::Revolution { child, offset } => {
            // q = (|p.xz| - offset, p.y, 0)
            let r_xz = (point.x * point.x + point.z * point.z).sqrt().max(1e-10);
            let q = Vec3::new(r_xz - *offset, point.y, 0.0);
            let g = eval_gradient(child, q);
            // ∂qₓ/∂pₓ = pₓ/r_xz, ∂qₓ/∂p_z = p_z/r_xz, ∂q_y/∂p_y = 1
            Vec3::new(
                g.x * point.x / r_xz,
                g.y,
                g.x * point.z / r_xz,
            )
        }
        SdfNode::Extrude { child, half_height } => {
            // f = max_min(d2d, |pz|-h) (box-like combination)
            let p_flat = Vec3::new(point.x, point.y, 0.0);
            let d2d = eval(child, p_flat);
            let w_y = point.z.abs() - *half_height;
            let grad_child = eval_gradient(child, p_flat);
            let grad_xy = Vec3::new(grad_child.x, grad_child.y, 0.0);
            let sz = if point.z >= 0.0 { 1.0 } else { -1.0 };
            let grad_z = Vec3::new(0.0, 0.0, sz);

            if d2d > 0.0 && w_y > 0.0 {
                // Outside both: blend
                let len = (d2d * d2d + w_y * w_y).sqrt().max(1e-10);
                (grad_xy * d2d + grad_z * w_y) / len
            } else if d2d > w_y {
                grad_xy
            } else {
                grad_z
            }
        }
        SdfNode::Twist { child, strength } => {
            // q = twist(p, k); J^T propagation
            let k = *strength;
            let (s, c) = (point.y * k).sin_cos();
            let q = Vec3::new(
                point.x * c - point.z * s,
                point.y,
                point.x * s + point.z * c,
            );
            let g = eval_gradient(child, q);
            // J^T = [[c, 0, s], [-k·q_z, 1, k·q_x], [-s, 0, c]]
            Vec3::new(
                c * g.x + s * g.z,
                -k * q.z * g.x + g.y + k * q.x * g.z,
                -s * g.x + c * g.z,
            )
        }
        SdfNode::Bend { child, curvature } => {
            // q = bend(p, k); J^T propagation
            let k = *curvature;
            let (s, c) = (k * point.x).sin_cos();
            let q = Vec3::new(
                c * point.x - s * point.y,
                s * point.x + c * point.y,
                point.z,
            );
            let g = eval_gradient(child, q);
            // J^T = [[c-k·q_y, s+k·q_x, 0], [-s, c, 0], [0, 0, 1]]
            Vec3::new(
                (c - k * q.y) * g.x + (s + k * q.x) * g.y,
                -s * g.x + c * g.y,
                g.z,
            )
        }
        SdfNode::RepeatInfinite { child, spacing } => {
            // Repeat is piecewise translation: gradient unchanged within each cell
            let p = modifier_repeat_infinite(point, *spacing);
            eval_gradient(child, p)
        }
        SdfNode::RepeatFinite { child, count, spacing } => {
            let p = modifier_repeat_finite(point, *count, *spacing);
            eval_gradient(child, p)
        }

        // === Complex modifiers: numerical fallback on the subtree ===
        SdfNode::Noise { .. }
        | SdfNode::Taper { .. }
        | SdfNode::Displacement { .. }
        | SdfNode::PolarRepeat { .. }
        | SdfNode::SweepBezier { .. } => numerical_gradient_of(node, point),

        // Material is transparent
        SdfNode::WithMaterial { child, .. } => eval_gradient(child, point),
    }
}

/// Compute an analytic normal from the gradient (normalized).
///
/// Equivalent to `normal()` but uses the analytic gradient path
/// instead of 6 finite-difference evaluations.
#[inline]
pub fn eval_normal(node: &SdfNode, point: Vec3) -> Vec3 {
    let g = eval_gradient(node, point);
    let len_sq = g.length_squared();
    if len_sq < 1e-20 {
        Vec3::Y // Safe fallback
    } else {
        g / len_sq.sqrt()
    }
}

// ============================================================
// Numerical fallback
// ============================================================

/// Central-difference numerical gradient on a subtree.
///
/// Uses 6 evaluations. For leaf nodes this is very cheap;
/// for subtrees it's equivalent to 6 recursive evaluations.
#[inline(always)]
fn numerical_gradient_of(node: &SdfNode, point: Vec3) -> Vec3 {
    const EPS: f32 = 0.0001;
    const INV_2E: f32 = 0.5 / EPS;
    let ex = Vec3::new(EPS, 0.0, 0.0);
    let ey = Vec3::new(0.0, EPS, 0.0);
    let ez = Vec3::new(0.0, 0.0, EPS);
    Vec3::new(
        eval(node, point + ex) - eval(node, point - ex),
        eval(node, point + ey) - eval(node, point - ey),
        eval(node, point + ez) - eval(node, point - ez),
    ) * INV_2E
}

// ============================================================
// Primitive gradient functions
// ============================================================

/// Sphere: f = |p| - r → ∇f = p / |p|
#[inline(always)]
fn grad_sphere(point: Vec3) -> Vec3 {
    let len = point.length();
    if len < 1e-10 { return Vec3::Y; }
    point / len
}

/// Box3d: piecewise gradient based on nearest face/edge/corner
#[inline(always)]
fn grad_box3d(point: Vec3, half_extents: Vec3) -> Vec3 {
    let q = point.abs() - half_extents;
    let signs = Vec3::new(
        if point.x >= 0.0 { 1.0 } else { -1.0 },
        if point.y >= 0.0 { 1.0 } else { -1.0 },
        if point.z >= 0.0 { 1.0 } else { -1.0 },
    );

    if q.x > 0.0 || q.y > 0.0 || q.z > 0.0 {
        // Outside: gradient of max(q,0).length()
        let clamped = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
        let len = clamped.length();
        if len < 1e-10 { return Vec3::Y; }
        clamped * signs / len
    } else {
        // Inside: gradient points toward nearest face
        if q.x >= q.y && q.x >= q.z {
            Vec3::new(signs.x, 0.0, 0.0)
        } else if q.y >= q.z {
            Vec3::new(0.0, signs.y, 0.0)
        } else {
            Vec3::new(0.0, 0.0, signs.z)
        }
    }
}

/// Cylinder: combines radial (XZ) and axial (Y) gradients
#[inline(always)]
fn grad_cylinder(point: Vec3, radius: f32, half_height: f32) -> Vec3 {
    let r_xz = (point.x * point.x + point.z * point.z).sqrt();
    let dr = r_xz - radius;
    let dy = point.y.abs() - half_height;

    if dr > 0.0 && dy > 0.0 {
        // Outside corner
        let len = (dr * dr + dy * dy).sqrt().max(1e-10);
        let radial = if r_xz > 1e-10 {
            Vec3::new(point.x / r_xz * dr, 0.0, point.z / r_xz * dr)
        } else {
            Vec3::ZERO
        };
        let axial = Vec3::new(0.0, if point.y >= 0.0 { dy } else { -dy }, 0.0);
        (radial + axial) / len
    } else if dr > dy {
        // Nearest to side
        if r_xz > 1e-10 {
            Vec3::new(point.x / r_xz, 0.0, point.z / r_xz)
        } else {
            Vec3::X
        }
    } else {
        // Nearest to cap
        Vec3::new(0.0, if point.y >= 0.0 { 1.0 } else { -1.0 }, 0.0)
    }
}

/// Torus: f = |q| - r where q = (|p.xz| - R, p.y)
#[inline(always)]
fn grad_torus(point: Vec3, major_radius: f32, _minor_radius: f32) -> Vec3 {
    let r_xz = (point.x * point.x + point.z * point.z).sqrt().max(1e-10);
    let qx = r_xz - major_radius;
    let q_len = (qx * qx + point.y * point.y).sqrt().max(1e-10);
    Vec3::new(
        qx * point.x / (q_len * r_xz),
        point.y / q_len,
        qx * point.z / (q_len * r_xz),
    )
}

/// Capsule: gradient is direction to closest point on segment
#[inline(always)]
fn grad_capsule(point: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let t = (point - a).dot(ab) / ab.dot(ab).max(1e-10);
    let t = t.clamp(0.0, 1.0);
    let closest = a + ab * t;
    let diff = point - closest;
    let len = diff.length();
    if len < 1e-10 { return Vec3::Y; }
    diff / len
}

/// Infinite cylinder along Y: f = |p.xz| - r → ∇f = (px,0,pz)/|p.xz|
#[inline(always)]
fn grad_infinite_cylinder(point: Vec3) -> Vec3 {
    let r = (point.x * point.x + point.z * point.z).sqrt();
    if r < 1e-10 { return Vec3::X; }
    Vec3::new(point.x / r, 0.0, point.z / r)
}

/// Gyroid: f = |g|/scale - thickness
/// g = sin(sx)cos(sy) + sin(sy)cos(sz) + sin(sz)cos(sx)
#[inline(always)]
fn grad_gyroid(point: Vec3, scale: f32, _thickness: f32) -> Vec3 {
    let sp = point * scale;
    let (sx, cx) = sp.x.sin_cos();
    let (sy, cy) = sp.y.sin_cos();
    let (sz, cz) = sp.z.sin_cos();
    let g = sx * cy + sy * cz + sz * cx;
    let sign_g = if g >= 0.0 { 1.0 } else { -1.0 };
    // ∇(g/scale)/∂p = (∂g/∂sp · scale) / scale = ∂g/∂sp
    // ∇(|g|/scale) = sign(g) · ∇(g/scale) = sign(g) · ∂g/∂sp
    Vec3::new(
        sign_g * (cx * cy - sz * sx),
        sign_g * (-sx * sy + cy * cz),
        sign_g * (-sy * sz + cz * cx),
    )
}

/// Schwarz P: f = |g|/scale - thickness
/// g = cos(sx) + cos(sy) + cos(sz)
#[inline(always)]
fn grad_schwarz_p(point: Vec3, scale: f32, _thickness: f32) -> Vec3 {
    let sp = point * scale;
    let g = sp.x.cos() + sp.y.cos() + sp.z.cos();
    let sign_g = if g >= 0.0 { 1.0 } else { -1.0 };
    // ∂g/∂sp = (-sin(sx), -sin(sy), -sin(sz))
    // ∇(|g|/scale) = sign(g) · ∂g/∂sp
    Vec3::new(
        -sign_g * sp.x.sin(),
        -sign_g * sp.y.sin(),
        -sign_g * sp.z.sin(),
    )
}

// ============================================================
// Operation gradient helpers
// ============================================================

/// Weights for smooth_min gradient: ∂smooth_min/∂a and ∂smooth_min/∂b
///
/// smooth_min(a,b,k) = min(a,b) - h²k/4  where h = max(k-|a-b|,0)/k
///
/// When |a-b| >= k: (a<b → (1,0), a>=b → (0,1))
/// When |a-b| < k:  smooth blend
#[inline(always)]
fn smooth_min_weights(da: f32, db: f32, k: f32) -> (f32, f32) {
    let k = k.max(1e-10);
    let diff = da - db;
    let h = ((k - diff.abs()) / k).max(0.0);

    if h < 1e-10 {
        // Outside blend zone: gradient of whichever is smaller
        if da <= db { (1.0, 0.0) } else { (0.0, 1.0) }
    } else if da <= db {
        (1.0 - h * 0.5, h * 0.5)
    } else {
        (h * 0.5, 1.0 - h * 0.5)
    }
}

/// Weights for smooth_max gradient
#[inline(always)]
fn smooth_max_weights(da: f32, db: f32, k: f32) -> (f32, f32) {
    let k = k.max(1e-10);
    let diff = da - db;
    let h = ((k - diff.abs()) / k).max(0.0);

    if h < 1e-10 {
        if da >= db { (1.0, 0.0) } else { (0.0, 1.0) }
    } else if da >= db {
        (1.0 - h * 0.5, h * 0.5)
    } else {
        (h * 0.5, 1.0 - h * 0.5)
    }
}

/// Gradient of chamfer_min = min(a, b, (a+b)*FRAC_1_SQRT_2 - r)
#[inline(always)]
fn grad_chamfer_min(da: f32, db: f32, r: f32, ga: Vec3, gb: Vec3) -> Vec3 {
    let chamfer_val = (da + db) * FRAC_1_SQRT_2 - r;
    let min_ab = da.min(db);

    if chamfer_val < min_ab {
        // Chamfer term wins: gradient is FRAC_1_SQRT_2 * (ga + gb)
        (ga + gb) * FRAC_1_SQRT_2
    } else if da <= db {
        ga
    } else {
        gb
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compare analytic gradient to numerical gradient
    fn check_gradient(node: &SdfNode, point: Vec3, tolerance: f32) {
        let analytic = eval_gradient(node, point);
        let numerical = numerical_gradient_of(node, point);
        let diff = (analytic - numerical).length();
        assert!(
            diff < tolerance,
            "Gradient mismatch at {:?}: analytic={:?}, numerical={:?}, diff={}",
            point, analytic, numerical, diff
        );
    }

    fn test_points() -> Vec<Vec3> {
        // Avoid exact surfaces, axes, cell boundaries, and symmetry planes
        vec![
            Vec3::new(1.2, 0.3, 0.1),
            Vec3::new(-0.4, 1.1, 0.2),
            Vec3::new(0.1, -0.3, 1.3),
            Vec3::new(0.7, 0.5, 0.3),
            Vec3::new(-0.8, 0.4, -0.6),
            Vec3::new(1.5, 0.2, 0.1),
            Vec3::new(0.3, -0.7, 0.5),
        ]
    }

    #[test]
    fn test_grad_sphere() {
        let s = SdfNode::sphere(1.0);
        for p in test_points() {
            check_gradient(&s, p, 0.01);
        }
    }

    #[test]
    fn test_grad_box() {
        let b = SdfNode::box3d(2.0, 1.0, 1.5);
        for p in test_points() {
            check_gradient(&b, p, 0.05);
        }
    }

    #[test]
    fn test_grad_cylinder() {
        let c = SdfNode::cylinder(0.5, 2.0);
        for p in test_points() {
            check_gradient(&c, p, 0.05);
        }
    }

    #[test]
    fn test_grad_torus() {
        let t = SdfNode::torus(1.5, 0.3);
        for p in test_points() {
            check_gradient(&t, p, 0.01);
        }
    }

    #[test]
    fn test_grad_plane() {
        let pl = SdfNode::plane(Vec3::new(0.0, 1.0, 0.0), 0.5);
        for p in test_points() {
            check_gradient(&pl, p, 0.001);
        }
    }

    #[test]
    fn test_grad_capsule() {
        let c = SdfNode::capsule(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.3);
        for p in test_points() {
            check_gradient(&c, p, 0.01);
        }
    }

    #[test]
    fn test_grad_infinite_cylinder() {
        let ic = SdfNode::infinite_cylinder(0.5);
        for p in test_points() {
            check_gradient(&ic, p, 0.01);
        }
    }

    #[test]
    fn test_grad_gyroid() {
        let g = SdfNode::gyroid(2.0, 0.1);
        for p in test_points() {
            check_gradient(&g, p, 0.05);
        }
    }

    #[test]
    fn test_grad_schwarz_p() {
        let s = SdfNode::schwarz_p(2.0, 0.1);
        for p in test_points() {
            check_gradient(&s, p, 0.05);
        }
    }

    #[test]
    fn test_grad_union() {
        let shape = SdfNode::sphere(1.0).union(SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0));
        for p in test_points() {
            check_gradient(&shape, p, 0.02);
        }
    }

    #[test]
    fn test_grad_smooth_union() {
        let shape = SdfNode::sphere(1.0).smooth_union(
            SdfNode::box3d(1.0, 1.0, 1.0).translate(1.0, 0.0, 0.0),
            0.3,
        );
        for p in test_points() {
            check_gradient(&shape, p, 0.1);
        }
    }

    #[test]
    fn test_grad_subtraction() {
        let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));
        for p in test_points() {
            check_gradient(&shape, p, 0.05);
        }
    }

    #[test]
    fn test_grad_translate() {
        let shape = SdfNode::sphere(1.0).translate(1.0, 2.0, 3.0);
        for p in test_points() {
            check_gradient(&shape, p, 0.01);
        }
    }

    #[test]
    fn test_grad_rotate() {
        let shape = SdfNode::box3d(2.0, 1.0, 1.0).rotate_euler(0.0, 0.785, 0.0);
        for p in test_points() {
            check_gradient(&shape, p, 0.05);
        }
    }

    #[test]
    fn test_grad_scale() {
        let shape = SdfNode::sphere(1.0).scale(2.0);
        for p in test_points() {
            check_gradient(&shape, p, 0.01);
        }
    }

    #[test]
    fn test_grad_twist() {
        let shape = SdfNode::box3d(1.0, 2.0, 1.0).twist(0.5);
        for p in test_points() {
            check_gradient(&shape, p, 0.1);
        }
    }

    #[test]
    fn test_grad_bend() {
        let shape = SdfNode::box3d(1.0, 2.0, 1.0).bend(0.3);
        for p in test_points() {
            check_gradient(&shape, p, 0.1);
        }
    }

    #[test]
    fn test_grad_mirror() {
        let shape = SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0).mirror(true, false, false);
        for p in test_points() {
            check_gradient(&shape, p, 0.02);
        }
    }

    #[test]
    fn test_grad_round() {
        let shape = SdfNode::box3d(1.0, 1.0, 1.0).round(0.1);
        for p in test_points() {
            check_gradient(&shape, p, 0.05);
        }
    }

    #[test]
    fn test_grad_onion() {
        let shape = SdfNode::sphere(1.0).onion(0.1);
        for p in test_points() {
            check_gradient(&shape, p, 0.05);
        }
    }

    #[test]
    fn test_grad_complex_tree() {
        // Complex tree: tests chain rule propagation
        let shape = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::cylinder(0.3, 1.5).rotate_euler(1.57, 0.0, 0.0), 0.2)
            .subtract(SdfNode::box3d(0.4, 0.4, 0.4))
            .translate(0.5, 0.0, 0.0);

        for p in test_points() {
            check_gradient(&shape, p, 0.15);
        }
    }

    #[test]
    fn test_eval_normal_matches() {
        let shape = SdfNode::sphere(1.0);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let n = eval_normal(&shape, p);
        let expected = Vec3::new(1.0, 0.0, 0.0);
        assert!((n - expected).length() < 0.01);
    }

    #[test]
    fn test_grad_chamfer_union() {
        let shape = SdfNode::sphere(1.0).chamfer_union(
            SdfNode::box3d(1.0, 1.0, 1.0).translate(1.0, 0.0, 0.0),
            0.2,
        );
        for p in test_points() {
            check_gradient(&shape, p, 0.1);
        }
    }

    #[test]
    fn test_grad_repeat() {
        let shape = SdfNode::sphere(0.3).repeat_infinite(2.0, 2.0, 2.0);
        for p in test_points() {
            check_gradient(&shape, p, 0.02);
        }
    }
}
