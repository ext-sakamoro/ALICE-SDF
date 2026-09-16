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
use crate::operations::*;
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
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // q = abs(p) - half_extents; max(q, 0).length() + min(max(q.x, q.y, q.z), 0) - r
        let hx = Self::splat(inst.params[0]);
        let hy = Self::splat(inst.params[1]);
        let hz = Self::splat(inst.params[2]);
        let rr = Self::splat(inst.params[3]);
        let qx = p.x.abs() - hx;
        let qy = p.y.abs() - hy;
        let qz = p.z.abs() - hz;
        let qx_pos = qx.max(Self::ZERO);
        let qy_pos = qy.max(Self::ZERO);
        let qz_pos = qz.max(Self::ZERO);
        let outer = (qx_pos * qx_pos + qy_pos * qy_pos + qz_pos * qz_pos).sqrt();
        let inner = qx.max(qy).max(qz).min(Self::ZERO);
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
        let h = Self::splat(inst.params[0]);
        let r1 = Self::splat(inst.params[1]);
        let r2 = Self::splat(inst.params[2]);
        let qx = (p.x * p.x + p.z * p.z).sqrt();
        let qy = p.y;
        // k2 = (r2 - r1, 2*h)
        let k2x = r2 - r1;
        let k2y = h * Self::splat(2.0);
        let k2_dot = k2x * k2x + k2y * k2y;
        // ca = (qx - min(qx, if qy<0 {r1} else {r2}), abs(qy) - h)
        let neg_mask = qy.cmp_lt(Self::ZERO);
        let min_r = neg_mask.blend(r1, r2);
        let ca_x = qx - qx.min(min_r);
        let ca_y = qy.abs() - h;
        // t = clamp(dot(k1-q, k2) / dot(k2,k2), 0, 1); k1=(r2,h)
        let d_to_k1_x = r2 - qx;
        let d_to_k1_y = h - qy;
        let num = d_to_k1_x * k2x + d_to_k1_y * k2y;
        let safe_k2_dot = k2_dot.max(Self::splat(0.0001));
        let t = (num / safe_k2_dot).max(Self::ZERO).min(Self::ONE);
        // cb = q - k1 + k2*t
        let cb_x = qx - r2 + k2x * t;
        let cb_y = qy - h + k2y * t;
        let ca_d2 = ca_x * ca_x + ca_y * ca_y;
        let cb_d2 = cb_x * cb_x + cb_y * cb_y;
        // s = -1 if cb.x<0 && ca.y<0, else 1
        let both_neg = cb_x.cmp_lt(Self::ZERO) & ca_y.cmp_lt(Self::ZERO);
        let s = both_neg.blend(Self::splat(-1.0), Self::ONE);
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
        let major_r = Self::splat(inst.params[0]);
        let minor_r = Self::splat(inst.params[1]);
        let sc_sin = Self::splat(inst.params[2].sin());
        let sc_cos = Self::splat(inst.params[2].cos());
        let px = p.x.abs();
        // k = sc.cos*px > sc.sin*py ? sc.sin*px + sc.cos*py : sqrt(px² + py²)
        let dot_val = sc_sin * px + sc_cos * p.y;
        let len_val = (px * px + p.y * p.y).sqrt();
        let mask = (sc_cos * px).cmp_gt(sc_sin * p.y);
        let k = mask.blend(dot_val, len_val);
        // sqrt(px² + py² + pz² + R² - 2*R*k) - r
        let inner =
            px * px + p.y * p.y + p.z * p.z + major_r * major_r - Self::splat(2.0) * major_r * k;
        let d = inner.max(Self::ZERO).sqrt() - minor_r;
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
        let radius = Self::splat(inst.params[0]);
        let round_r = Self::splat(inst.params[1]);
        let half_h = Self::splat(inst.params[2]);
        let dx = (p.x * p.x + p.z * p.z).sqrt() - radius + round_r;
        let dy = p.y.abs() - half_h;
        let dx_pos = dx.max(Self::ZERO);
        let dy_pos = dy.max(Self::ZERO);
        let d = dx.max(dy).min(Self::ZERO) + (dx_pos * dx_pos + dy_pos * dy_pos).sqrt() - round_r;
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
        let width = Self::splat(inst.params[0]);
        let half_depth = Self::splat(inst.params[1]);
        let qx = p.x.abs();
        let qy = p.y; // not abs for y
        let qz = p.z.abs();
        // 0.866025 = sqrt(3)/2
        let sqrt3_half = Self::splat(0.866025);
        let half = Self::splat(0.5);
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
        let radius = Self::splat(inst.params[0]);
        let ch = Self::splat(inst.params[1]); // cut_height
        let w = (radius * radius - ch * ch).max(Self::ZERO).sqrt();
        let qx = (p.x * p.x + p.z * p.z).sqrt();
        let qy = p.y;
        let q_len = (qx * qx + qy * qy).sqrt();
        // Three regions via branchless blend
        let s1 = (ch - radius) * qx * qx + w * w * (ch + radius - Self::splat(2.0) * qy);
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
        let mask_s_neg = s.cmp_lt(Self::ZERO);
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
        let radius = Self::splat(inst.params[0]);
        let ch = Self::splat(inst.params[1]);
        let thickness = Self::splat(inst.params[2]);
        let w = (radius * radius - ch * ch).max(Self::ZERO).sqrt();
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
        let ra = Self::splat(inst.params[0]);
        let rb = Self::splat(inst.params[1]);
        let dd = Self::splat(inst.params[2]);
        let two = Self::splat(2.0);
        // a = (ra² - rb² + d²) / (2d)
        #[allow(clippy::suspicious_operation_groupings)]
        let a = (ra * ra - rb * rb + dd * dd) / (two * dd);
        let b = (ra * ra - a * a).max(Self::ZERO).sqrt();
        let p2x = p.x;
        let p2y = (p.y * p.y + p.z * p.z).sqrt();
        // Condition: p2.x*b - p2.y*a > d*max(b - p2.y, 0)
        let lhs = p2x * b - p2y * a;
        let rhs = dd * (b - p2y).max(Self::ZERO);
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
        let c_sin = Self::splat(inst.params[0].sin());
        let c_cos = Self::splat(inst.params[0].cos());
        let radius = Self::splat(inst.params[1]);
        let qx = (p.x * p.x + p.z * p.z).sqrt();
        let qy = p.y;
        let q_len = (qx * qx + qy * qy).sqrt();
        let l = q_len - radius;
        // dot(q, c) clamped to [0, radius]
        let q_dot_c = (qx * c_sin + qy * c_cos).max(Self::ZERO).min(radius);
        // m = length(q - c * clamp(dot(q,c), 0, r))
        let proj_x = qx - c_sin * q_dot_c;
        let proj_y = qy - c_cos * q_dot_c;
        let m = (proj_x * proj_x + proj_y * proj_y).sqrt();
        // sign = c.y*q.x - c.x*q.y < 0 ? -1 : 1
        let sign_val = c_cos * qx - c_sin * qy;
        let neg_mask = sign_val.cmp_lt(Self::ZERO);
        let sign = neg_mask.blend(Self::splat(-1.0), Self::ONE);
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
        let la = Self::splat(inst.params[0]);
        let lb = Self::splat(inst.params[1]);
        let half_h = Self::splat(inst.params[2]);
        let rr = Self::splat(inst.params[3]);
        let ax = p.x.abs();
        let ay = p.y.abs();
        let az = p.z.abs();
        // ndot(b, b - 2*(px,pz)) = la*(la-2*px) - lb*(lb-2*pz)
        //                        = la² - 2*la*px - lb² + 2*lb*pz
        let b_dot_b = la * la + lb * lb;
        let ndot_val = la * (la - Self::splat(2.0) * ax) - lb * (lb - Self::splat(2.0) * az);
        let f = (ndot_val / b_dot_b).max(Self::splat(-1.0)).min(Self::ONE);
        // q_xz = length((px,pz) - 0.5*b*(1-f, 1+f))
        let half = Self::splat(0.5);
        let proj_x = ax - half * la * (Self::ONE - f);
        let proj_z = az - half * lb * (Self::ONE + f);
        let qxz_len = (proj_x * proj_x + proj_z * proj_z).sqrt();
        // sign(px*lb + pz*la - la*lb)
        let sign_input = ax * lb + az * la - la * lb;
        let pos = sign_input.cmp_gt(Self::ZERO);
        let neg = sign_input.cmp_lt(Self::ZERO);
        let sign = pos.blend(Self::ONE, neg.blend(Self::splat(-1.0), Self::ZERO));
        let dx = qxz_len * sign - rr;
        let dy = ay - half_h;
        let dx_pos = dx.max(Self::ZERO);
        let dy_pos = dy.max(Self::ZERO);
        let d = dx.max(dy).min(Self::ZERO) + (dx_pos * dx_pos + dy_pos * dy_pos).sqrt();
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
        let r = Self::splat(inst.params[0]);
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
        let scale = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let inv_scale = Self::splat(1.0 / inst.params[0]);
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
        let outer_r = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let half_h = Self::splat(inst.params[2]);
        let xz_len = (p.x * p.x + p.z * p.z).sqrt();
        let dx = (xz_len - outer_r).abs() - thickness;
        let dy = p.y.abs() - half_h;
        let dx_pos = dx.max(Self::ZERO);
        let dy_pos = dy.max(Self::ZERO);
        let d = (dx_pos * dx_pos + dy_pos * dy_pos).sqrt() + dx.max(dy).min(Self::ZERO);
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
        let scale = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let inv_scale = Self::splat(1.0 / inst.params[0]);
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
        let radius = Self::splat(inst.params[0]);
        let s = Self::splat(0.577_350_26_f32); // 1/sqrt(3)
        let ns = Self::splat(-0.577_350_26_f32);
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
        let radius = Self::splat(inst.params[0]);
        let a = Self::splat(0.850_650_8_f32); // ICO_B
        let b = Self::splat(0.525_731_1_f32); // ICO_A
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
        let radius = Self::splat(inst.params[0]);
        let s = Self::splat(0.577_350_26_f32); // 1/sqrt(3)
        let ia = Self::splat(0.525_731_1_f32); // ICO_A
        let ib = Self::splat(0.850_650_8_f32); // ICO_B
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
        let radius = Self::splat(inst.params[0]);
        let s = Self::splat(0.577_350_26_f32);
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
        let radius = Self::splat(inst.params[0]);
        let s = Self::splat(0.577_350_26_f32);
        let ia = Self::splat(0.525_731_1_f32);
        let ib = Self::splat(0.850_650_8_f32);
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
        let bx = Self::splat(inst.params[0]);
        let by = Self::splat(inst.params[1]);
        let bz = Self::splat(inst.params[2]);
        let e = Self::splat(inst.params[3]);
        let zero = Self::ZERO;
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
        let scale = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let inv_scale = Self::splat(1.0 / inst.params[0]);
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
        let scale = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let inv_scale = Self::splat(1.0 / inst.params[0]);
        let three = Self::splat(3.0);
        let four = Self::splat(4.0);
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
        let scale = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let inv_scale = Self::splat(1.0 / inst.params[0]);
        let two = Self::splat(2.0);
        let one = Self::ONE;
        let half = Self::splat(0.5);
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
        let d = (term1 - term2 + Self::splat(0.15)).abs() * inv_scale - thickness;
        out = d * scale_correction;
        out
    }
    #[inline(always)]
    fn iwp(inst: &Instruction, aux_data: &[f32], p: Vec3R<Self>, scale_correction: Self) -> Self {
        let p: Vec3x8 = p.into();
        #[allow(clippy::needless_late_init, unused_variables)]
        let out;
        // ★ Native SIMD + double-angle: cos(2x) = 2*cos²(x)-1
        let scale = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let inv_scale = Self::splat(1.0 / inst.params[0]);
        let two = Self::splat(2.0);
        let one = Self::ONE;
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
        let scale = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let inv_scale = Self::splat(1.0 / inst.params[0]);
        let two = Self::splat(2.0);
        let one = Self::ONE;
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
        let scale = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let inv_scale = Self::splat(1.0 / inst.params[0]);
        let two = Self::splat(2.0);
        let one = Self::ONE;
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
        let d = c2x * sy * cz + cx * c2y * sz + sx * cy * c2z - Self::splat(0.4);
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
        let scale = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let inv_scale = Self::splat(1.0 / inst.params[0]);
        let two = Self::splat(2.0);
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
        let r = Self::splat(inst.params[0]);
        let half_h = Self::splat(inst.params[1]);
        let d2d = (p.x * p.x + p.y * p.y).sqrt() - r;
        let dz = p.z.abs() - half_h;
        let wx = d2d.max(Self::ZERO);
        let wy = dz.max(Self::ZERO);
        let d = (wx * wx + wy * wy).sqrt() + d2d.max(dz).min(Self::ZERO);
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
        let hx = Self::splat(inst.params[0]);
        let hy = Self::splat(inst.params[1]);
        let half_h = Self::splat(inst.params[2]);
        let dx = p.x.abs() - hx;
        let dy = p.y.abs() - hy;
        let d2d = (dx.max(Self::ZERO) * dx.max(Self::ZERO)
            + dy.max(Self::ZERO) * dy.max(Self::ZERO))
        .sqrt()
            + dx.max(dy).min(Self::ZERO);
        let dz = p.z.abs() - half_h;
        let wx = d2d.max(Self::ZERO);
        let wy = dz.max(Self::ZERO);
        let d = (wx * wx + wy * wy).sqrt() + d2d.max(dz).min(Self::ZERO);
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
        let hx = Self::splat(inst.params[0]);
        let hy = Self::splat(inst.params[1]);
        let round_r = Self::splat(inst.params[2]);
        let half_h = Self::splat(inst.params[3]);
        let dx = p.x.abs() - hx + round_r;
        let dy = p.y.abs() - hy + round_r;
        let d2d = (dx.max(Self::ZERO) * dx.max(Self::ZERO)
            + dy.max(Self::ZERO) * dy.max(Self::ZERO))
        .sqrt()
            + dx.max(dy).min(Self::ZERO)
            - round_r;
        let dz = p.z.abs() - half_h;
        let wx = d2d.max(Self::ZERO);
        let wy = dz.max(Self::ZERO);
        let d = (wx * wx + wy * wy).sqrt() + d2d.max(dz).min(Self::ZERO);
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
        let outer_r = Self::splat(inst.params[0]);
        let thickness = Self::splat(inst.params[1]);
        let half_h = Self::splat(inst.params[2]);
        let d2d = ((p.x * p.x + p.y * p.y).sqrt() - outer_r).abs() - thickness;
        let dz = p.z.abs() - half_h;
        let wx = d2d.max(Self::ZERO);
        let wy = dz.max(Self::ZERO);
        let d = (wx * wx + wy * wy).sqrt() + d2d.max(dz).min(Self::ZERO);
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
