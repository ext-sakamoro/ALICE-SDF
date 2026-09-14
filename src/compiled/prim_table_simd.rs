//! Primitive / CSG-binary law table for the `wide::f32x8` instantiation.
//!
//! Bodies are the SIMD-native arms that used to live inline in `eval_simd.rs`
//! (27 primitives + 7 binary ops are per-lane scalar via the canonical laws).
//! Phase 2 of the 1.10 plan replaces these with generic `sdf_x<R: Real>` laws
//! so scalar and SIMD share one body.
//!
//! Author: Moroya Sakamoto

use super::instruction::Instruction;
use super::prim_table::PrimTable;
use super::real::Vec3R;
use super::simd::Vec3x8;
use crate::operations::{
    sdf_columns_intersection, sdf_columns_subtraction, sdf_columns_union,
    sdf_exp_smooth_intersection, sdf_exp_smooth_subtraction, sdf_exp_smooth_union,
};
use crate::primitives::*;
use glam::Vec3;
use wide::{f32x8, CmpGt, CmpLt};

#[allow(unused_variables)]
impl PrimTable for f32x8 {
    #[inline(always)]
    fn sphere(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = f32x8::splat(inst.params[0]);
        let d = p.length() - r;
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn box3d(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let hx = f32x8::splat(inst.params[0]);
        let hy = f32x8::splat(inst.params[1]);
        let hz = f32x8::splat(inst.params[2]);

        // q = abs(p) - half_extents
        let qx = p.x.abs() - hx;
        let qy = p.y.abs() - hy;
        let qz = p.z.abs() - hz;

        // length(max(q, 0)) + min(max(q.x, q.y, q.z), 0)
        let qx_pos = qx.max(f32x8::ZERO);
        let qy_pos = qy.max(f32x8::ZERO);
        let qz_pos = qz.max(f32x8::ZERO);
        let outside = (qx_pos * qx_pos + qy_pos * qy_pos + qz_pos * qz_pos).sqrt();
        let inside = qx.max(qy).max(qz).min(f32x8::ZERO);
        let d = outside + inside;

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn cylinder(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = f32x8::splat(inst.params[0]);
        let h = f32x8::splat(inst.params[1]);

        // d.x = length(p.xz) - r
        // d.y = abs(p.y) - h
        let dx = (p.x * p.x + p.z * p.z).sqrt() - r;
        let dy = p.y.abs() - h;

        let dx_pos = dx.max(f32x8::ZERO);
        let dy_pos = dy.max(f32x8::ZERO);
        let outside = (dx_pos * dx_pos + dy_pos * dy_pos).sqrt();
        let inside = dx.max(dy).min(f32x8::ZERO);
        let d = outside + inside;

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn torus(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let major = f32x8::splat(inst.params[0]);
        let minor = f32x8::splat(inst.params[1]);

        // q = vec2(length(p.xz) - major, p.y)
        let qx = (p.x * p.x + p.z * p.z).sqrt() - major;
        let qy = p.y;
        let d = (qx * qx + qy * qy).sqrt() - minor;

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn plane(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let nx = f32x8::splat(inst.params[0]);
        let ny = f32x8::splat(inst.params[1]);
        let nz = f32x8::splat(inst.params[2]);
        let dist = f32x8::splat(inst.params[3]);

        // dot(p, n) - distance: same law as `sdf_plane` (tree / scalar / BVH)
        let d = p.x * nx + p.y * ny + p.z * nz - dist;
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn capsule(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let ax = f32x8::splat(inst.params[0]);
        let ay = f32x8::splat(inst.params[1]);
        let az = f32x8::splat(inst.params[2]);
        let bx = f32x8::splat(inst.params[3]);
        let by = f32x8::splat(inst.params[4]);
        let bz = f32x8::splat(inst.params[5]);
        let radius = f32x8::splat(inst.get_capsule_radius());

        // pa = p - a, ba = b - a
        let pax = p.x - ax;
        let pay = p.y - ay;
        let paz = p.z - az;
        let bax = bx - ax;
        let bay = by - ay;
        let ba_z = bz - az;

        // h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1)
        let dot_pa_ba = pax * bax + pay * bay + paz * ba_z;
        let dot_ba_ba = bax * bax + bay * bay + ba_z * ba_z;
        // Branchless zero guard for degenerate capsule (a == b)
        let safe_dot = dot_ba_ba.max(f32x8::splat(1e-10));
        let h = (dot_pa_ba / safe_dot).max(f32x8::ZERO).min(f32x8::ONE);

        // length(pa - ba * h) - radius
        let dx = pax - bax * h;
        let dy = pay - bay * h;
        let dz = paz - ba_z * h;
        let d = (dx * dx + dy * dy + dz * dz).sqrt() - radius;

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn cone(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let radius = f32x8::splat(inst.params[0]);
        let h = f32x8::splat(inst.params[1]);

        let q_x = (p.x * p.x + p.z * p.z).sqrt();
        let q_y = p.y;

        let k2x = -radius;
        let k2y = h + h;

        // ca_r = q_y < 0 ? radius : 0
        let neg_mask = q_y.cmp_lt(f32x8::ZERO);
        let ca_r = neg_mask.blend(radius, f32x8::ZERO);

        let ca_x = q_x - q_x.min(ca_r);
        let ca_y = q_y.abs() - h;

        let diff_x = -q_x;
        let diff_y = h - q_y;
        let k2_dot = k2x * k2x + k2y * k2y;
        let t = ((diff_x * k2x + diff_y * k2y) / k2_dot)
            .max(f32x8::ZERO)
            .min(f32x8::ONE);

        let cb_x = q_x + k2x * t;
        let cb_y = q_y - h + k2y * t;

        // s = (cb_x < 0 && ca_y < 0) ? -1 : 1
        let both_neg = cb_x.cmp_lt(f32x8::ZERO) & ca_y.cmp_lt(f32x8::ZERO);
        let s = both_neg.blend(f32x8::splat(-1.0), f32x8::ONE);

        let d2 = (ca_x * ca_x + ca_y * ca_y).min(cb_x * cb_x + cb_y * cb_y);
        let d = s * d2.sqrt();

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn ellipsoid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // Division Exorcism: 3 divisions instead of 6 (inv_r² = inv_r * inv_r)
        let rx = inst.params[0].max(1e-10);
        let ry = inst.params[1].max(1e-10);
        let rz = inst.params[2].max(1e-10);
        let inv_rx = f32x8::splat(1.0 / rx);
        let inv_ry = f32x8::splat(1.0 / ry);
        let inv_rz = f32x8::splat(1.0 / rz);
        let inv_rx2 = inv_rx * inv_rx;
        let inv_ry2 = inv_ry * inv_ry;
        let inv_rz2 = inv_rz * inv_rz;

        // k0 = length(p * inv_radii)
        let px_r = p.x * inv_rx;
        let py_r = p.y * inv_ry;
        let pz_r = p.z * inv_rz;
        let k0 = (px_r * px_r + py_r * py_r + pz_r * pz_r).sqrt();

        // k1 = length(p * inv_radii²)
        let px_rr = p.x * inv_rx2;
        let py_rr = p.y * inv_ry2;
        let pz_rr = p.z * inv_rz2;
        let k1 = (px_rr * px_rr + py_rr * py_rr + pz_rr * pz_rr).sqrt();

        let d = k0 * (k0 - f32x8::ONE) / k1.max(f32x8::splat(1e-10));
        // sdf_ellipsoid law: at the centre (k1 ≈ 0) return -min(radii)
        let centre = f32x8::splat(-rx.min(ry).min(rz));
        let d = k1.cmp_lt(f32x8::splat(1e-10)).blend(centre, d);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn rounded_cone(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r1 = f32x8::splat(inst.params[0]);
        let r2 = f32x8::splat(inst.params[1]);
        let half_height = f32x8::splat(inst.params[2]);

        let h = half_height + half_height;
        let q_x = (p.x * p.x + p.z * p.z).sqrt();
        let q_y = p.y + half_height;

        let b_val = (r1 - r2) / h;
        let a_val = (f32x8::ONE - b_val * b_val).sqrt();
        let k = q_x * (-b_val) + q_y * a_val;

        // Case 1: k < 0 → bottom sphere
        let d_bottom = (q_x * q_x + q_y * q_y).sqrt() - r1;
        // Case 2: k > a*h → top sphere
        let dy_top = q_y - h;
        let d_top = (q_x * q_x + dy_top * dy_top).sqrt() - r2;
        // Case 3: mantle
        let d_mantle = q_x * a_val + q_y * b_val - r1;

        let mask_bottom = k.cmp_lt(f32x8::ZERO);
        let mask_top = k.cmp_gt(a_val * h);
        let d = mask_bottom.blend(d_bottom, mask_top.blend(d_top, d_mantle));

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn pyramid(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let half_height = f32x8::splat(inst.params[0]);
        let h = half_height + half_height;
        let m2 = h * h + f32x8::splat(0.25);
        let half = f32x8::splat(0.5);

        let py = p.y + half_height;
        let abs_px = p.x.abs();
        let abs_pz = p.z.abs();

        // if pz > px { swap(px, pz) }
        let swap_mask = abs_pz.cmp_gt(abs_px);
        let px = swap_mask.blend(abs_pz, abs_px) - half;
        let pz = swap_mask.blend(abs_px, abs_pz) - half;

        let qx = pz;
        let qy = h * py - half * px;
        let qz = h * px + half * py;

        let s = (-qx).max(f32x8::ZERO);
        let t = ((qy - half * pz) / (m2 + f32x8::splat(0.25)))
            .max(f32x8::ZERO)
            .min(f32x8::ONE);

        let a = m2 * (qx + s) * (qx + s) + qy * qy;
        let half_t = half * t;
        let b = m2 * (qx + half_t) * (qx + half_t) + (qy - m2 * t) * (qy - m2 * t);

        // d2 = if qy.min(-qx * m2 - qy * 0.5) > 0 { 0 } else { min(a, b) }
        let inner = (-qx * m2 - qy * half).min(qy);
        let zero_mask = inner.cmp_gt(f32x8::ZERO);
        let d2 = zero_mask.blend(f32x8::ZERO, a.min(b));

        // sign = signum(max(qz, -py))
        let sign_input = qz.max(-py);
        let pos = sign_input.cmp_gt(f32x8::ZERO);
        let neg = sign_input.cmp_lt(f32x8::ZERO);
        let sign_val = pos.blend(f32x8::ONE, neg.blend(f32x8::splat(-1.0), f32x8::ZERO));

        let d = ((d2 + qz * qz) / m2).sqrt() * sign_val;

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn octahedron(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let s = f32x8::splat(inst.params[0]);
        let three = f32x8::splat(3.0);
        let inv_sqrt3 = f32x8::splat(0.57735027);
        let half = f32x8::splat(0.5);

        let ax = p.x.abs();
        let ay = p.y.abs();
        let az = p.z.abs();
        let m = ax + ay + az - s;

        // d_flat: the "else" branch = m * inv_sqrt3
        let d_flat = m * inv_sqrt3;

        // Exclusive masks for the 3 permutation cases
        let mask1 = (three * ax).cmp_lt(m);
        let mask2 = (three * ay).cmp_lt(m) & !mask1;
        let mask3 = (three * az).cmp_lt(m) & !mask1 & !mask2;
        let mask_any = mask1 | mask2 | mask3;

        // Select q permutation based on which case
        let qx = mask1.blend(ax, mask2.blend(ay, mask3.blend(az, f32x8::ZERO)));
        let qy = mask1.blend(ay, mask2.blend(az, mask3.blend(ax, f32x8::ZERO)));
        let qz = mask1.blend(az, mask2.blend(ax, mask3.blend(ay, f32x8::ZERO)));

        let k = (half * (qz - qy + s)).max(f32x8::ZERO).min(s);
        let vx = qx;
        let vy = qy - s + k;
        let vz = qz - k;
        let d_edge = (vx * vx + vy * vy + vz * vz).sqrt();

        let d = mask_any.blend(d_edge, d_flat);

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn hex_prism(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let hex_radius = f32x8::splat(inst.params[0]);
        let half_height = f32x8::splat(inst.params[1]);

        let kx = f32x8::splat(-0.8660254_f32);
        let ky = f32x8::splat(0.5_f32);
        let kz_c = f32x8::splat(0.57735027_f32);
        let two = f32x8::splat(2.0);

        let mut px = p.x.abs();
        let mut py = p.y.abs();
        let pz = p.z.abs();

        // Reflect across hex symmetry
        let dot_kxy = kx * px + ky * py;
        let reflect = two * dot_kxy.min(f32x8::ZERO);
        px -= reflect * kx;
        py -= reflect * ky;

        // Clamp and compute XY distance
        let clamped_x = px.max(-kz_c * hex_radius).min(kz_c * hex_radius);
        let dx = px - clamped_x;
        let dy = py - hex_radius;
        let d_xy_len = (dx * dx + dy * dy).sqrt();

        // signum(dy)
        let pos = dy.cmp_gt(f32x8::ZERO);
        let neg = dy.cmp_lt(f32x8::ZERO);
        let dy_sign = pos.blend(f32x8::ONE, neg.blend(f32x8::splat(-1.0), f32x8::ZERO));
        let d_xy = d_xy_len * dy_sign;

        let d_z = pz - half_height;

        // max(d_xy, d_z).min(0) + sqrt(max(d_xy,0)^2 + max(d_z,0)^2)
        let d_xy_pos = d_xy.max(f32x8::ZERO);
        let d_z_pos = d_z.max(f32x8::ZERO);
        let d = d_xy.max(d_z).min(f32x8::ZERO) + (d_xy_pos * d_xy_pos + d_z_pos * d_z_pos).sqrt();

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn link(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let half_length = f32x8::splat(inst.params[0]);
        let r1 = f32x8::splat(inst.params[1]);
        let r2 = f32x8::splat(inst.params[2]);

        let qx = p.x;
        let qy = (p.y.abs() - half_length).max(f32x8::ZERO);
        let qz = p.z;

        let xy_len = (qx * qx + qy * qy).sqrt() - r1;
        let d = (xy_len * xy_len + qz * qz).sqrt() - r2;

        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn rounded_box(
        inst: &Instruction,
        aux_data: &[f32],
        p: Vec3R<Self>,
        scale_correction: Self,
    ) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // q = abs(p) - half_extents; max(q, 0).length() + min(max(q.x, q.y, q.z), 0) - r
        let hx = f32x8::splat(inst.params[0]);
        let hy = f32x8::splat(inst.params[1]);
        let hz = f32x8::splat(inst.params[2]);
        let rr = f32x8::splat(inst.params[3]);
        let qx = p.x.abs() - hx;
        let qy = p.y.abs() - hy;
        let qz = p.z.abs() - hz;
        let qx_pos = qx.max(f32x8::ZERO);
        let qy_pos = qy.max(f32x8::ZERO);
        let qz_pos = qz.max(f32x8::ZERO);
        let outer = (qx_pos * qx_pos + qy_pos * qy_pos + qz_pos * qz_pos).sqrt();
        let inner = qx.max(qy).max(qz).min(f32x8::ZERO);
        let d = outer + inner - rr;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // 2D profile in (length(p.xz), p.y) space
        let h = f32x8::splat(inst.params[0]);
        let r1 = f32x8::splat(inst.params[1]);
        let r2 = f32x8::splat(inst.params[2]);
        let qx = (p.x * p.x + p.z * p.z).sqrt();
        let qy = p.y;
        // k2 = (r2 - r1, 2*h)
        let k2x = r2 - r1;
        let k2y = h * f32x8::splat(2.0);
        let k2_dot = k2x * k2x + k2y * k2y;
        // ca = (qx - min(qx, if qy<0 {r1} else {r2}), abs(qy) - h)
        let neg_mask = qy.cmp_lt(f32x8::ZERO);
        let min_r = neg_mask.blend(r1, r2);
        let ca_x = qx - qx.min(min_r);
        let ca_y = qy.abs() - h;
        // t = clamp(dot(k1-q, k2) / dot(k2,k2), 0, 1); k1=(r2,h)
        let d_to_k1_x = r2 - qx;
        let d_to_k1_y = h - qy;
        let num = d_to_k1_x * k2x + d_to_k1_y * k2y;
        let safe_k2_dot = k2_dot.max(f32x8::splat(0.0001));
        let t = (num / safe_k2_dot).max(f32x8::ZERO).min(f32x8::ONE);
        // cb = q - k1 + k2*t
        let cb_x = qx - r2 + k2x * t;
        let cb_y = qy - h + k2y * t;
        let ca_d2 = ca_x * ca_x + ca_y * ca_y;
        let cb_d2 = cb_x * cb_x + cb_y * cb_y;
        // s = -1 if cb.x<0 && ca.y<0, else 1
        let both_neg = cb_x.cmp_lt(f32x8::ZERO) & ca_y.cmp_lt(f32x8::ZERO);
        let s = both_neg.blend(f32x8::splat(-1.0), f32x8::ONE);
        let d = s * ca_d2.min(cb_d2).sqrt();
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let major_r = f32x8::splat(inst.params[0]);
        let minor_r = f32x8::splat(inst.params[1]);
        let sc_sin = f32x8::splat(inst.params[2].sin());
        let sc_cos = f32x8::splat(inst.params[2].cos());
        let px = p.x.abs();
        // k = sc.cos*px > sc.sin*py ? sc.sin*px + sc.cos*py : sqrt(px² + py²)
        let dot_val = sc_sin * px + sc_cos * p.y;
        let len_val = (px * px + p.y * p.y).sqrt();
        let mask = (sc_cos * px).cmp_gt(sc_sin * p.y);
        let k = mask.blend(dot_val, len_val);
        // sqrt(px² + py² + pz² + R² - 2*R*k) - r
        let inner =
            px * px + p.y * p.y + p.z * p.z + major_r * major_r - f32x8::splat(2.0) * major_r * k;
        let d = inner.max(f32x8::ZERO).sqrt() - minor_r;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let radius = f32x8::splat(inst.params[0]);
        let round_r = f32x8::splat(inst.params[1]);
        let half_h = f32x8::splat(inst.params[2]);
        let dx = (p.x * p.x + p.z * p.z).sqrt() - f32x8::splat(2.0) * radius + round_r;
        let dy = p.y.abs() - half_h;
        let dx_pos = dx.max(f32x8::ZERO);
        let dy_pos = dy.max(f32x8::ZERO);
        let d = dx.max(dy).min(f32x8::ZERO) + (dx_pos * dx_pos + dy_pos * dy_pos).sqrt() - round_r;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let width = f32x8::splat(inst.params[0]);
        let half_depth = f32x8::splat(inst.params[1]);
        let qx = p.x.abs();
        let qy = p.y; // not abs for y
        let qz = p.z.abs();
        // 0.866025 = sqrt(3)/2
        let sqrt3_half = f32x8::splat(0.866025);
        let half = f32x8::splat(0.5);
        let d = (qz - half_depth).max((qx * sqrt3_half + qy * half).max(-qy) - width * half);
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let radius = f32x8::splat(inst.params[0]);
        let ch = f32x8::splat(inst.params[1]); // cut_height
        let w = (radius * radius - ch * ch).max(f32x8::ZERO).sqrt();
        let qx = (p.x * p.x + p.z * p.z).sqrt();
        let qy = p.y;
        let q_len = (qx * qx + qy * qy).sqrt();
        // Three regions via branchless blend
        let s1 = (ch - radius) * qx * qx + w * w * (ch + radius - f32x8::splat(2.0) * qy);
        let s2 = ch * qx - w * qy;
        let s = s1.max(s2);
        // d_sphere = length(q) - r
        let d_sphere = q_len - radius;
        // d_plane = h - q.y
        let d_plane = ch - qy;
        // d_edge = length(q - (w, h))
        let ex = qx - w;
        let ey = qy - ch;
        let d_edge = (ex * ex + ey * ey).sqrt();
        // if s < 0 -> d_sphere; elif qx < w -> d_plane; else -> d_edge
        let mask_s_neg = s.cmp_lt(f32x8::ZERO);
        let mask_qx_lt_w = qx.cmp_lt(w);
        let d = mask_s_neg.blend(d_sphere, mask_qx_lt_w.blend(d_plane, d_edge));
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let radius = f32x8::splat(inst.params[0]);
        let ch = f32x8::splat(inst.params[1]);
        let thickness = f32x8::splat(inst.params[2]);
        let w = (radius * radius - ch * ch).max(f32x8::ZERO).sqrt();
        let qx = (p.x * p.x + p.z * p.z).sqrt();
        let qy = p.y;
        // if h*qx < w*qy -> length(q - (w,h)) - t; else -> abs(length(q) - r) - t
        let mask = (ch * qx).cmp_lt(w * qy);
        let ex = qx - w;
        let ey = qy - ch;
        let d_cap = (ex * ex + ey * ey).sqrt() - thickness;
        let q_len = (qx * qx + qy * qy).sqrt();
        let d_shell = (q_len - radius).abs() - thickness;
        let d = mask.blend(d_cap, d_shell);
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let ra = f32x8::splat(inst.params[0]);
        let rb = f32x8::splat(inst.params[1]);
        let dd = f32x8::splat(inst.params[2]);
        let two = f32x8::splat(2.0);
        // a = (ra² - rb² + d²) / (2d)
        #[allow(clippy::suspicious_operation_groupings)]
        let a = (ra * ra - rb * rb + dd * dd) / (two * dd);
        let b = (ra * ra - a * a).max(f32x8::ZERO).sqrt();
        let p2x = p.x;
        let p2y = (p.y * p.y + p.z * p.z).sqrt();
        // Condition: p2.x*b - p2.y*a > d*max(b - p2.y, 0)
        let lhs = p2x * b - p2y * a;
        let rhs = dd * (b - p2y).max(f32x8::ZERO);
        let mask = lhs.cmp_gt(rhs);
        // d_edge = length(p2 - (a, b))
        let ex = p2x - a;
        let ey = p2y - b;
        let d_edge = (ex * ex + ey * ey).sqrt();
        // d_main = max(length(p2) - ra, -(length(p2 - (d,0)) - rb))
        let p2_len = (p2x * p2x + p2y * p2y).sqrt();
        let dx = p2x - dd;
        let d_ra = p2_len - ra;
        let d_rb = -((dx * dx + p2y * p2y).sqrt() - rb);
        let d_main = d_ra.max(d_rb);
        let d = mask.blend(d_edge, d_main);
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let c_sin = f32x8::splat(inst.params[0].sin());
        let c_cos = f32x8::splat(inst.params[0].cos());
        let radius = f32x8::splat(inst.params[1]);
        let qx = (p.x * p.x + p.z * p.z).sqrt();
        let qy = p.y;
        let q_len = (qx * qx + qy * qy).sqrt();
        let l = q_len - radius;
        // dot(q, c) clamped to [0, radius]
        let q_dot_c = (qx * c_sin + qy * c_cos).max(f32x8::ZERO).min(radius);
        // m = length(q - c * clamp(dot(q,c), 0, r))
        let proj_x = qx - c_sin * q_dot_c;
        let proj_y = qy - c_cos * q_dot_c;
        let m = (proj_x * proj_x + proj_y * proj_y).sqrt();
        // sign = c.y*q.x - c.x*q.y < 0 ? -1 : 1
        let sign_val = c_cos * qx - c_sin * qy;
        let neg_mask = sign_val.cmp_lt(f32x8::ZERO);
        let sign = neg_mask.blend(f32x8::splat(-1.0), f32x8::ONE);
        let d = l.max(m * sign);
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let la = f32x8::splat(inst.params[0]);
        let lb = f32x8::splat(inst.params[1]);
        let half_h = f32x8::splat(inst.params[2]);
        let rr = f32x8::splat(inst.params[3]);
        let ax = p.x.abs();
        let ay = p.y.abs();
        let az = p.z.abs();
        // ndot(b, b - 2*(px,pz)) = la*(la-2*px) - lb*(lb-2*pz)
        //                        = la² - 2*la*px - lb² + 2*lb*pz
        let b_dot_b = la * la + lb * lb;
        let ndot_val = la * (la - f32x8::splat(2.0) * ax) - lb * (lb - f32x8::splat(2.0) * az);
        let f = (ndot_val / b_dot_b).max(f32x8::splat(-1.0)).min(f32x8::ONE);
        // q_xz = length((px,pz) - 0.5*b*(1-f, 1+f))
        let half = f32x8::splat(0.5);
        let proj_x = ax - half * la * (f32x8::ONE - f);
        let proj_z = az - half * lb * (f32x8::ONE + f);
        let qxz_len = (proj_x * proj_x + proj_z * proj_z).sqrt();
        // sign(px*lb + pz*la - la*lb)
        let sign_input = ax * lb + az * la - la * lb;
        let pos = sign_input.cmp_gt(f32x8::ZERO);
        let neg = sign_input.cmp_lt(f32x8::ZERO);
        let sign = pos.blend(f32x8::ONE, neg.blend(f32x8::splat(-1.0), f32x8::ZERO));
        let dx = qxz_len * sign - rr;
        let dy = ay - half_h;
        let dx_pos = dx.max(f32x8::ZERO);
        let dy_pos = dy.max(f32x8::ZERO);
        let d = dx.max(dy).min(f32x8::ZERO) + (dx_pos * dx_pos + dy_pos * dy_pos).sqrt();
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_horseshoe(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
                inst.params[4],
            )
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| sdf_vesica(pt, inst.params[0], inst.params[1]));
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // Simple SIMD: length(p.xz) - r
        let r = f32x8::splat(inst.params[0]);
        let d = (p.x * p.x + p.z * p.z).sqrt() - r;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| sdf_infinite_cone(pt, inst.params[0]));
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
        let scale = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let inv_scale = f32x8::splat(1.0 / inst.params[0]);
        let spx = p.x * scale;
        let spy = p.y * scale;
        let spz = p.z * scale;
        let sx = sin_approx(spx);
        let cx = cos_approx(spx);
        let sy = sin_approx(spy);
        let cy = cos_approx(spy);
        let sz = sin_approx(spz);
        let cz = cos_approx(spz);
        let d = (sx * cy + sy * cz + sz * cx).abs() * inv_scale - thickness;
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn heart(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| sdf_heart(pt, inst.params[0]));
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn tube(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // SIMD: hollow cylinder
        let outer_r = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let half_h = f32x8::splat(inst.params[2]);
        let xz_len = (p.x * p.x + p.z * p.z).sqrt();
        let dx = (xz_len - outer_r).abs() - thickness;
        let dy = p.y.abs() - half_h;
        let dx_pos = dx.max(f32x8::ZERO);
        let dy_pos = dy.max(f32x8::ZERO);
        let d = (dx_pos * dx_pos + dy_pos * dy_pos).sqrt() + dx.max(dy).min(f32x8::ZERO);
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_barrel(pt, inst.params[0], inst.params[1], inst.params[2])
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| sdf_diamond(pt, inst.params[0], inst.params[1]));
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            let he = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
            sdf_chamfered_cube(pt, he, inst.params[3])
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: cos(x) + cos(y) + cos(z)
        let scale = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let inv_scale = f32x8::splat(1.0 / inst.params[0]);
        let d = (cos_approx(p.x * scale) + cos_approx(p.y * scale) + cos_approx(p.z * scale)).abs()
            * inv_scale
            - thickness;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            let he = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
            sdf_superellipsoid(pt, he, inst.params[3], inst.params[4])
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_rounded_x(pt, inst.params[0], inst.params[1], inst.params[2])
        });
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn pie(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_pie(pt, inst.params[0], inst.params[1], inst.params[2])
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_trapezoid(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
            )
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_parallelogram(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
            )
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_tunnel(pt, inst.params[0], inst.params[1], inst.params[2])
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_uneven_capsule(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
            )
        });
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn egg(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| sdf_egg(pt, inst.params[0], inst.params[1]));
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_arc_shape(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
            )
        });
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn moon(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_moon(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
            )
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_cross_shape(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
            )
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_blobby_cross(pt, inst.params[0], inst.params[1])
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_parabola_segment(pt, inst.params[0], inst.params[1], inst.params[2])
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_regular_polygon(pt, inst.params[0], inst.params[1], inst.params[2])
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_star_polygon(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
            )
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_stairs(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
            )
        });
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn helix(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let d = eval_per_lane(&p, |pt| {
            sdf_helix(
                pt,
                inst.params[0],
                inst.params[1],
                inst.params[2],
                inst.params[3],
            )
        });
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: 4 dot products + max (no abs — tetrahedron normals are signed)
        let radius = f32x8::splat(inst.params[0]);
        let s = f32x8::splat(0.577_350_26_f32); // 1/sqrt(3)
        let ns = f32x8::splat(-0.577_350_26_f32);
        // n0=(s,s,s) n1=(-s,-s,s) n2=(-s,s,-s) n3=(s,-s,-s)
        let d0 = p.x * s + p.y * s + p.z * s;
        let d1 = p.x * ns + p.y * ns + p.z * s;
        let d2 = p.x * ns + p.y * s + p.z * ns;
        let d3 = p.x * s + p.y * ns + p.z * ns;
        let d = d0.max(d1).max(d2).max(d3) - radius;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: 6 abs-dot products (GDF_DODECAHEDRON normals)
        let radius = f32x8::splat(inst.params[0]);
        let a = f32x8::splat(0.850_650_8_f32); // ICO_B
        let b = f32x8::splat(0.525_731_1_f32); // ICO_A
                                               // n0=(0,a,b) n1=(0,a,-b) n2=(a,b,0) n3=(-a,b,0) n4=(b,0,a) n5=(b,0,-a)
        let d0 = (p.y * a + p.z * b).abs();
        let d1 = (p.y * a - p.z * b).abs();
        let d2 = (p.x * a + p.y * b).abs();
        let d3 = (p.x * a - p.y * b).abs();
        let d4 = (p.x * b + p.z * a).abs();
        let d5 = (p.x * b - p.z * a).abs();
        let d = d0.max(d1).max(d2).max(d3).max(d4).max(d5) - radius;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: 10 abs-dot products (octahedron[4] + icosahedron[6])
        let radius = f32x8::splat(inst.params[0]);
        let s = f32x8::splat(0.577_350_26_f32); // 1/sqrt(3)
        let ia = f32x8::splat(0.525_731_1_f32); // ICO_A
        let ib = f32x8::splat(0.850_650_8_f32); // ICO_B
                                                // Octahedron normals (4): abs(dot) with (±s,±s,±s) variants
        let d0 = (p.x * s + p.y * s + p.z * s).abs();
        let d1 = (-p.x * s + p.y * s + p.z * s).abs();
        let d2 = (p.x * s - p.y * s + p.z * s).abs();
        let d3 = (p.x * s + p.y * s - p.z * s).abs();
        // Icosahedron normals (6): abs(dot) with (0,±a,±b) permutations
        let d4 = (p.y * ia + p.z * ib).abs();
        let d5 = (p.y * ia - p.z * ib).abs();
        let d6 = (p.x * ia + p.y * ib).abs();
        let d7 = (p.x * ia - p.y * ib).abs();
        let d8 = (p.x * ib + p.z * ia).abs();
        let d9 = (p.x * ib - p.z * ia).abs();
        let d = d0
            .max(d1)
            .max(d2)
            .max(d3)
            .max(d4)
            .max(d5)
            .max(d6)
            .max(d7)
            .max(d8)
            .max(d9)
            - radius;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: 7 abs-dot products (cube[3] + octahedron[4])
        let radius = f32x8::splat(inst.params[0]);
        let s = f32x8::splat(0.577_350_26_f32);
        // Cube normals (3): abs of each axis
        let d0 = p.x.abs();
        let d1 = p.y.abs();
        let d2 = p.z.abs();
        // Octahedron normals (4)
        let d3 = (p.x * s + p.y * s + p.z * s).abs();
        let d4 = (-p.x * s + p.y * s + p.z * s).abs();
        let d5 = (p.x * s - p.y * s + p.z * s).abs();
        let d6 = (p.x * s + p.y * s - p.z * s).abs();
        let d = d0.max(d1).max(d2).max(d3).max(d4).max(d5).max(d6) - radius;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: 16 abs-dot products (oct[4] + ico[6] + dodec[6])
        let radius = f32x8::splat(inst.params[0]);
        let s = f32x8::splat(0.577_350_26_f32);
        let ia = f32x8::splat(0.525_731_1_f32);
        let ib = f32x8::splat(0.850_650_8_f32);
        // Octahedron (4)
        let d0 = (p.x * s + p.y * s + p.z * s).abs();
        let d1 = (-p.x * s + p.y * s + p.z * s).abs();
        let d2 = (p.x * s - p.y * s + p.z * s).abs();
        let d3 = (p.x * s + p.y * s - p.z * s).abs();
        // Icosahedron (6)
        let d4 = (p.y * ia + p.z * ib).abs();
        let d5 = (p.y * ia - p.z * ib).abs();
        let d6 = (p.x * ia + p.y * ib).abs();
        let d7 = (p.x * ia - p.y * ib).abs();
        let d8 = (p.x * ib + p.z * ia).abs();
        let d9 = (p.x * ib - p.z * ia).abs();
        // Dodecahedron (6)
        let d10 = (p.y * ib + p.z * ia).abs();
        let d11 = (p.y * ib - p.z * ia).abs();
        let d12 = (p.x * ib + p.y * ia).abs();
        let d13 = (p.x * ib - p.y * ia).abs();
        let d14 = (p.x * ia + p.z * ib).abs();
        let d15 = (p.x * ia - p.z * ib).abs();
        let d = d0
            .max(d1)
            .max(d2)
            .max(d3)
            .max(d4)
            .max(d5)
            .max(d6)
            .max(d7)
            .max(d8)
            .max(d9)
            .max(d10)
            .max(d11)
            .max(d12)
            .max(d13)
            .max(d14)
            .max(d15)
            - radius;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: abs, max, min, sqrt — all native ops
        let bx = f32x8::splat(inst.params[0]);
        let by = f32x8::splat(inst.params[1]);
        let bz = f32x8::splat(inst.params[2]);
        let e = f32x8::splat(inst.params[3]);
        let zero = f32x8::ZERO;
        // p = abs(p) - half_extents
        let px = p.x.abs() - bx;
        let py = p.y.abs() - by;
        let pz = p.z.abs() - bz;
        // q = abs(p + e) - e
        let qx = (px + e).abs() - e;
        let qy = (py + e).abs() - e;
        let qz = (pz + e).abs() - e;
        // d1 = length(max(vec3(px,qy,qz),0)) + min(max(px,max(qy,qz)),0)
        let v1x = px.max(zero);
        let v1y = qy.max(zero);
        let v1z = qz.max(zero);
        let d1 = (v1x * v1x + v1y * v1y + v1z * v1z).sqrt() + px.max(qy.max(qz)).min(zero);
        // d2
        let v2x = qx.max(zero);
        let v2y = py.max(zero);
        let v2z = qz.max(zero);
        let d2 = (v2x * v2x + v2y * v2y + v2z * v2z).sqrt() + qx.max(py.max(qz)).min(zero);
        // d3
        let v3x = qx.max(zero);
        let v3y = qy.max(zero);
        let v3z = pz.max(zero);
        let d3 = (v3x * v3x + v3y * v3y + v3z * v3z).sqrt() + qx.max(qy.max(pz)).min(zero);
        let d = d1.min(d2).min(d3);
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z)
        let scale = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let inv_scale = f32x8::splat(1.0 / inst.params[0]);
        let spx = p.x * scale;
        let spy = p.y * scale;
        let spz = p.z * scale;
        let sx = sin_approx(spx);
        let cx = cos_approx(spx);
        let sy = sin_approx(spy);
        let cy = cos_approx(spy);
        let sz = sin_approx(spz);
        let cz = cos_approx(spz);
        let d = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz;
        let d = d.abs() * inv_scale - thickness;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD: 3*(cos(x)+cos(y)+cos(z)) + 4*cos(x)*cos(y)*cos(z)
        let scale = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let inv_scale = f32x8::splat(1.0 / inst.params[0]);
        let three = f32x8::splat(3.0);
        let four = f32x8::splat(4.0);
        let cx = cos_approx(p.x * scale);
        let cy = cos_approx(p.y * scale);
        let cz = cos_approx(p.z * scale);
        let d = three * (cx + cy + cz) + four * cx * cy * cz;
        let d = d.abs() * inv_scale - thickness;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD + double-angle identities (0 extra trig calls)
        let scale = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let inv_scale = f32x8::splat(1.0 / inst.params[0]);
        let two = f32x8::splat(2.0);
        let one = f32x8::ONE;
        let half = f32x8::splat(0.5);
        let spx = p.x * scale;
        let spy = p.y * scale;
        let spz = p.z * scale;
        let sx = sin_approx(spx);
        let cx = cos_approx(spx);
        let sy = sin_approx(spy);
        let cy = cos_approx(spy);
        let sz = sin_approx(spz);
        let cz = cos_approx(spz);
        // sin(2x) = 2*sx*cx, cos(2x) = 2*cx*cx - 1
        let s2x = two * sx * cx;
        let s2y = two * sy * cy;
        let s2z = two * sz * cz;
        let c2x = two * cx * cx - one;
        let c2y = two * cy * cy - one;
        let c2z = two * cz * cz - one;
        let term1 = half * (s2x * cy * sz + sx * s2y * cz + cx * sy * s2z);
        let term2 = half * (c2x * c2y + c2y * c2z + c2z * c2x);
        let d = (term1 - term2 + f32x8::splat(0.15)).abs() * inv_scale - thickness;
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn iwp(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD + double-angle: cos(2x) = 2*cos²(x)-1
        let scale = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let inv_scale = f32x8::splat(1.0 / inst.params[0]);
        let two = f32x8::splat(2.0);
        let one = f32x8::ONE;
        let cx = cos_approx(p.x * scale);
        let cy = cos_approx(p.y * scale);
        let cz = cos_approx(p.z * scale);
        let c2x = two * cx * cx - one;
        let c2y = two * cy * cy - one;
        let c2z = two * cz * cz - one;
        let d = two * (cx * cy + cy * cz + cz * cx) - (c2x + c2y + c2z);
        let d = d.abs() * inv_scale - thickness;
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn frd(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD + double-angle: cos(2x) = 2*cos²(x)-1
        let scale = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let inv_scale = f32x8::splat(1.0 / inst.params[0]);
        let two = f32x8::splat(2.0);
        let one = f32x8::ONE;
        let spx = p.x * scale;
        let spy = p.y * scale;
        let spz = p.z * scale;
        let sx = sin_approx(spx);
        let cx = cos_approx(spx);
        let sy = sin_approx(spy);
        let cy = cos_approx(spy);
        let sz = sin_approx(spz);
        let cz = cos_approx(spz);
        let c2x = two * cx * cx - one;
        let c2y = two * cy * cy - one;
        let c2z = two * cz * cz - one;
        let d = c2x * sy * cz + cx * c2y * sz + sx * cy * c2z;
        let d = d.abs() * inv_scale - thickness;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD + double-angle (same as FRD with -0.4 offset)
        let scale = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let inv_scale = f32x8::splat(1.0 / inst.params[0]);
        let two = f32x8::splat(2.0);
        let one = f32x8::ONE;
        let spx = p.x * scale;
        let spy = p.y * scale;
        let spz = p.z * scale;
        let sx = sin_approx(spx);
        let cx = cos_approx(spx);
        let sy = sin_approx(spy);
        let cy = cos_approx(spy);
        let sz = sin_approx(spz);
        let cz = cos_approx(spz);
        let c2x = two * cx * cx - one;
        let c2y = two * cy * cy - one;
        let c2z = two * cz * cz - one;
        let d = c2x * sy * cz + cx * c2y * sz + sx * cy * c2z - f32x8::splat(0.4);
        let d = d.abs() * inv_scale - thickness;
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn pmy(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD + double-angle: sin(2x) = 2*sin(x)*cos(x)
        let scale = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let inv_scale = f32x8::splat(1.0 / inst.params[0]);
        let two = f32x8::splat(2.0);
        let spx = p.x * scale;
        let spy = p.y * scale;
        let spz = p.z * scale;
        let sx = sin_approx(spx);
        let cx = cos_approx(spx);
        let sy = sin_approx(spy);
        let cy = cos_approx(spy);
        let sz = sin_approx(spz);
        let cz = cos_approx(spz);
        let s2x = two * sx * cx;
        let s2y = two * sy * cy;
        let s2z = two * sz * cz;
        let d = two * cx * cy * cz + s2x * sy + sx * s2z + s2y * sz;
        let d = d.abs() * inv_scale - thickness;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = f32x8::splat(inst.params[0]);
        let half_h = f32x8::splat(inst.params[1]);
        let d2d = (p.x * p.x + p.y * p.y).sqrt() - r;
        let dz = p.z.abs() - half_h;
        let wx = d2d.max(f32x8::ZERO);
        let wy = dz.max(f32x8::ZERO);
        let d = (wx * wx + wy * wy).sqrt() + d2d.max(dz).min(f32x8::ZERO);
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let hx = f32x8::splat(inst.params[0]);
        let hy = f32x8::splat(inst.params[1]);
        let half_h = f32x8::splat(inst.params[2]);
        let dx = p.x.abs() - hx;
        let dy = p.y.abs() - hy;
        let d2d = (dx.max(f32x8::ZERO) * dx.max(f32x8::ZERO)
            + dy.max(f32x8::ZERO) * dy.max(f32x8::ZERO))
        .sqrt()
            + dx.max(dy).min(f32x8::ZERO);
        let dz = p.z.abs() - half_h;
        let wx = d2d.max(f32x8::ZERO);
        let wy = dz.max(f32x8::ZERO);
        let d = (wx * wx + wy * wy).sqrt() + d2d.max(dz).min(f32x8::ZERO);
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let a = glam::Vec2::new(inst.params[0], inst.params[1]);
        let b = glam::Vec2::new(inst.params[2], inst.params[3]);
        let (thickness, half_h) = (inst.params[4], inst.params[5]);
        let d = eval_per_lane(&p, |q| sdf_segment_2d(q, a, b, thickness, half_h));
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // Vertices live in aux_data as flat [x0, y0, x1, y1, ...]
        let aux_off = inst.aux_offset as usize;
        let flat = &aux_data[aux_off..aux_off + inst.aux_len as usize];
        let half_h = inst.params[0];
        let d = eval_per_lane(&p, |q| sdf_polygon_2d_flat(q, flat, half_h));
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let hx = f32x8::splat(inst.params[0]);
        let hy = f32x8::splat(inst.params[1]);
        let round_r = f32x8::splat(inst.params[2]);
        let half_h = f32x8::splat(inst.params[3]);
        let dx = p.x.abs() - hx + round_r;
        let dy = p.y.abs() - hy + round_r;
        let d2d = (dx.max(f32x8::ZERO) * dx.max(f32x8::ZERO)
            + dy.max(f32x8::ZERO) * dy.max(f32x8::ZERO))
        .sqrt()
            + dx.max(dy).min(f32x8::ZERO)
            - round_r;
        let dz = p.z.abs() - half_h;
        let wx = d2d.max(f32x8::ZERO);
        let wy = dz.max(f32x8::ZERO);
        let d = (wx * wx + wy * wy).sqrt() + d2d.max(dz).min(f32x8::ZERO);
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let outer_r = f32x8::splat(inst.params[0]);
        let thickness = f32x8::splat(inst.params[1]);
        let half_h = f32x8::splat(inst.params[2]);
        let d2d = ((p.x * p.x + p.y * p.y).sqrt() - outer_r).abs() - thickness;
        let dz = p.z.abs() - half_h;
        let wx = d2d.max(f32x8::ZERO);
        let wy = dz.max(f32x8::ZERO);
        let d = (wx * wx + wy * wy).sqrt() + d2d.max(dz).min(f32x8::ZERO);
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn union(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        out = a.min(b);
        out
    }
    #[inline(always)]
    fn intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        out = a.max(b);
        out
    }
    #[inline(always)]
    fn subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        out = a.max(-b);
        out
    }
    #[inline(always)]
    fn smooth_union(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let k = f32x8::splat(inst.params[0]);
        let rk = f32x8::splat(inst.params[1]); // Division Exorcism: precomputed 1/k
        out = smooth_min_simd_rk(a, b, k, rk);
        out
    }
    #[inline(always)]
    fn smooth_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let k = f32x8::splat(inst.params[0]);
        let rk = f32x8::splat(inst.params[1]);
        out = -smooth_min_simd_rk(-a, -b, k, rk);
        out
    }
    #[inline(always)]
    fn smooth_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let k = f32x8::splat(inst.params[0]);
        let rk = f32x8::splat(inst.params[1]);
        out = -smooth_min_simd_rk(-a, b, k, rk);
        out
    }
    #[inline(always)]
    fn chamfer_union(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = f32x8::splat(inst.params[0]);
        out = chamfer_min_simd(a, b, r);
        out
    }
    #[inline(always)]
    fn chamfer_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = f32x8::splat(inst.params[0]);
        out = -chamfer_min_simd(-a, -b, r);
        out
    }
    #[inline(always)]
    fn chamfer_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = f32x8::splat(inst.params[0]);
        out = -chamfer_min_simd(-a, b, r);
        out
    }
    #[inline(always)]
    fn stairs_union(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        out = stairs_min_simd(a, b, inst.params[0], inst.params[1]);
        out
    }
    #[inline(always)]
    fn stairs_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        out = -stairs_min_simd(-a, -b, inst.params[0], inst.params[1]);
        out
    }
    #[inline(always)]
    fn stairs_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        out = -stairs_min_simd(-a, b, inst.params[0], inst.params[1]);
        out
    }
    #[inline(always)]
    fn xor(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        out = a.min(b).max(-a.max(b));
        out
    }
    #[inline(always)]
    fn morph(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let t = f32x8::splat(inst.params[0]);
        let one_minus_t = f32x8::splat(1.0 - inst.params[0]);
        out = a * one_minus_t + b * t;
        out
    }
    #[inline(always)]
    fn columns_union(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = inst.params[0];
        let n = inst.params[1];
        let d = eval_per_lane_binary(a, b, |av, bv| sdf_columns_union(av, bv, r, n));
        out = d;
        out
    }
    #[inline(always)]
    fn columns_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = inst.params[0];
        let n = inst.params[1];
        let d = eval_per_lane_binary(a, b, |av, bv| sdf_columns_intersection(av, bv, r, n));
        out = d;
        out
    }
    #[inline(always)]
    fn columns_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = inst.params[0];
        let n = inst.params[1];
        let d = eval_per_lane_binary(a, b, |av, bv| sdf_columns_subtraction(av, bv, r, n));
        out = d;
        out
    }
    #[inline(always)]
    fn pipe(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = f32x8::splat(inst.params[0]);
        out = (a * a + b * b).sqrt() - r;
        out
    }
    #[inline(always)]
    fn engrave(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let r = f32x8::splat(inst.params[0]);
        let s = f32x8::splat(std::f32::consts::FRAC_1_SQRT_2);
        let abs_b = b.abs();
        out = a.max((a + r - abs_b) * s);
        out
    }
    #[inline(always)]
    fn groove(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let ra = f32x8::splat(inst.params[0]);
        let rb = f32x8::splat(inst.params[1]);
        let abs_b = b.abs();
        out = a.max((a + ra).min(rb - abs_b));
        out
    }
    #[inline(always)]
    fn tongue(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let ra = f32x8::splat(inst.params[0]);
        let rb = f32x8::splat(inst.params[1]);
        let abs_b = b.abs();
        out = a.min((a - ra).max(abs_b - rb));
        out
    }
    #[inline(always)]
    fn exp_smooth_union(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let k = inst.params[0];
        out = eval_per_lane_binary(a, b, |x, y| sdf_exp_smooth_union(x, y, k));
        out
    }
    #[inline(always)]
    fn exp_smooth_intersection(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let k = inst.params[0];
        out = eval_per_lane_binary(a, b, |x, y| sdf_exp_smooth_intersection(x, y, k));
        out
    }
    #[inline(always)]
    fn exp_smooth_subtraction(inst: &Instruction, a: Self, b: Self) -> Self {
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        let k = inst.params[0];
        out = eval_per_lane_binary(a, b, |x, y| sdf_exp_smooth_subtraction(x, y, k));
        out
    }
}
/// Per-lane scalar evaluation helper for binary operations
///
/// Evaluates a scalar binary operation for each of the 8 SIMD lanes independently.
#[inline(always)]
fn eval_per_lane_binary(a: f32x8, b: f32x8, f: impl Fn(f32, f32) -> f32) -> f32x8 {
    let aa = a.as_array_ref();
    let ba = b.as_array_ref();
    let mut results = [0.0f32; 8];
    for i in 0..8 {
        results[i] = f(aa[i], ba[i]);
    }
    f32x8::new(results)
}

/// Per-lane scalar evaluation helper for extended primitives
///
/// Evaluates a scalar SDF function for each of the 8 SIMD lanes independently.
/// This is used for complex primitives where full SIMD implementation would be
/// error-prone. The SIMD benefit still comes from parallelizing tree traversal
/// (transforms, operations) across 8 points.
#[inline(always)]
fn eval_per_lane(p: &Vec3x8, f: impl Fn(Vec3) -> f32) -> f32x8 {
    let px = p.x.as_array_ref();
    let py = p.y.as_array_ref();
    let pz = p.z.as_array_ref();
    let mut results = [0.0f32; 8];
    for i in 0..8 {
        results[i] = f(Vec3::new(px[i], py[i], pz[i]));
    }
    f32x8::new(results)
}

/// Smooth min — Division Exorcism edition.
/// Takes precomputed `rk = 1/k` to eliminate SIMD division.
#[inline(always)]
fn smooth_min_simd_rk(a: f32x8, b: f32x8, k: f32x8, rk: f32x8) -> f32x8 {
    let h = (f32x8::ONE - (a - b).abs() * rk).max(f32x8::ZERO);
    a.min(b) - h * h * k * f32x8::splat(0.25)
}

/// Chamfer minimum for SIMD: 45-degree beveled blend
/// `min(min(a,b), (a + b) * FRAC_1_SQRT_2 - r)`
#[inline(always)]
fn chamfer_min_simd(a: f32x8, b: f32x8, r: f32x8) -> f32x8 {
    let s = f32x8::splat(std::f32::consts::FRAC_1_SQRT_2);
    a.min(b).min((a + b) * s - r)
}

/// Stairs minimum for SIMD: stepped/terraced blend (Mercury hg_sdf)
/// Processes 8 lanes with scalar params r and n
#[inline(always)]
fn stairs_min_simd(a: f32x8, b: f32x8, r: f32, n: f32) -> f32x8 {
    let s = f32x8::splat(std::f32::consts::FRAC_1_SQRT_2);
    let s2 = f32x8::splat(std::f32::consts::SQRT_2);
    let half = f32x8::splat(0.5);
    // Division Exorcism: precompute scalar reciprocals, then splat
    let rn = r / n; // scalar division (1 cycle amortized over 8 lanes)
    let rn_v = f32x8::splat(rn);
    let step = r * std::f32::consts::SQRT_2 / n;
    let inv_step = 1.0 / step; // scalar reciprocal
    let step_v = f32x8::splat(step);
    let inv_step_v = f32x8::splat(inv_step);
    let hs_v = step_v * half;
    let off_v = f32x8::splat((r - rn) * 0.5 * std::f32::consts::SQRT_2);
    let edge_v = f32x8::splat(0.5 * rn);

    let d = a.min(b);

    // pR45
    let mut px = (a + b) * s;
    let mut py = (b - a) * s;
    // swap
    std::mem::swap(&mut px, &mut py);

    px -= off_v;
    py -= off_v;
    px += half * s2 * rn_v;

    // pMod1: px = glsl_mod(px + hs, step) - hs — Division Exorcism: multiply by inv_step
    let t = px + hs_v;
    px = t - step_v * (t * inv_step_v).floor() - hs_v;

    let d = d.min(py);

    // Second pR45
    let npx = (px + py) * s;
    let npy = (py - px) * s;

    d.min((npx - edge_v).max(npy - edge_v))
}

/// SIMD cosine (`wide` polynomial with range reduction, ~1e-6 abs error).
///
/// Replaced the Bhaskara I approximation (1.6e-3 abs error) in 1.9.1: TPMS
/// primitives and twist/bend amplified that error to >10% vs the tree law.
#[inline(always)]
fn cos_approx(x: f32x8) -> f32x8 {
    x.cos()
}

/// SIMD sine (`wide` polynomial with range reduction, ~1e-6 abs error).
#[inline(always)]
fn sin_approx(x: f32x8) -> f32x8 {
    x.sin()
}
