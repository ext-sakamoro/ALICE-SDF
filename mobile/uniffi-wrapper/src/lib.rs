//! ALICE-SDF Mobile — UniFFI Swift/Kotlin bindings の最小実装
//!
//! ALICE-SDF v1.4.0+ の primitive / operation 関数群を、iOS (Swift) と Android (Kotlin)
//! から呼び出せる最小公開 API に整理。1077+ tests を持つ ALICE-SDF 本体の極一部を
//! mobile 向けに切り出した「Hello SDF」レベル。

// uniffi::include_scaffolding! が生成する UNIFFI_META_CONST_UDL_ALICE_SDF doc コメントは
// clippy::empty_line_after_doc_comments を引っかけるが、生成コード側で制御不可のため許容。
#![allow(clippy::empty_line_after_doc_comments)]

use alice_sdf::operations::{
    sdf_intersection, sdf_smooth_intersection, sdf_smooth_subtraction, sdf_smooth_union,
    sdf_subtraction, sdf_union,
};
use alice_sdf::primitives::{
    sdf_box3d_at, sdf_cylinder as p_sdf_cylinder, sdf_plane as p_sdf_plane, sdf_rounded_box3d,
    sdf_sphere_at, sdf_torus as p_sdf_torus,
};
use glam::Vec3 as GlamVec3;

uniffi::include_scaffolding!("alice_sdf");

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<Vec3> for GlamVec3 {
    fn from(v: Vec3) -> Self {
        GlamVec3::new(v.x, v.y, v.z)
    }
}

// === Primitives ===

pub fn sdf_sphere(point: Vec3, center: Vec3, radius: f32) -> f32 {
    sdf_sphere_at(point.into(), center.into(), radius)
}

pub fn sdf_box(point: Vec3, center: Vec3, half_extents: Vec3) -> f32 {
    sdf_box3d_at(point.into(), center.into(), half_extents.into())
}

pub fn sdf_rounded_box(point: Vec3, center: Vec3, half_extents: Vec3, round_radius: f32) -> f32 {
    let p: GlamVec3 = point.into();
    let c: GlamVec3 = center.into();
    sdf_rounded_box3d(p - c, half_extents.into(), round_radius)
}

pub fn sdf_cylinder(point: Vec3, radius: f32, half_height: f32) -> f32 {
    p_sdf_cylinder(point.into(), radius, half_height)
}

pub fn sdf_torus(point: Vec3, major_radius: f32, minor_radius: f32) -> f32 {
    p_sdf_torus(point.into(), major_radius, minor_radius)
}

pub fn sdf_plane(point: Vec3, normal: Vec3, distance: f32) -> f32 {
    p_sdf_plane(point.into(), normal.into(), distance)
}

// === Operations ===

pub fn op_union(a: f32, b: f32) -> f32 {
    sdf_union(a, b)
}

pub fn op_intersection(a: f32, b: f32) -> f32 {
    sdf_intersection(a, b)
}

pub fn op_subtraction(a: f32, b: f32) -> f32 {
    sdf_subtraction(a, b)
}

pub fn op_smooth_union(a: f32, b: f32, k: f32) -> f32 {
    sdf_smooth_union(a, b, k)
}

pub fn op_smooth_intersection(a: f32, b: f32, k: f32) -> f32 {
    sdf_smooth_intersection(a, b, k)
}

pub fn op_smooth_subtraction(a: f32, b: f32, k: f32) -> f32 {
    sdf_smooth_subtraction(a, b, k)
}

// === Batch evaluation ===

pub fn sphere_batch(points: Vec<Vec3>, center: Vec3, radius: f32) -> Vec<f32> {
    let c: GlamVec3 = center.into();
    points
        .into_iter()
        .map(|p| sdf_sphere_at(p.into(), c, radius))
        .collect()
}

// === Meta ===

pub fn alice_sdf_version() -> String {
    "1.4.0".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_surface_is_zero() {
        let p = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let c = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let d = sdf_sphere(p, c, 1.0);
        assert!(d.abs() < 1e-4, "expected ~0, got {d}");
    }

    #[test]
    fn op_union_picks_min() {
        assert_eq!(op_union(0.3, 0.5), 0.3);
    }

    #[test]
    fn smooth_union_is_below_min() {
        let s = op_smooth_union(0.5, 0.5, 0.1);
        assert!(s < 0.5);
    }

    #[test]
    fn batch_eval_matches_single() {
        let pts = vec![
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Vec3 {
                x: 0.0,
                y: 2.0,
                z: 0.0,
            },
        ];
        let c = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let r = 1.0;
        let batch = sphere_batch(pts.clone(), c, r);
        for (i, p) in pts.iter().enumerate() {
            assert!((batch[i] - sdf_sphere(*p, c, r)).abs() < 1e-6);
        }
    }
}
