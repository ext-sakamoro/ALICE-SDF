//! ALICE-SDF OpenXR helpers
//!
//! Native VR / AR アプリ向けのヘルパー集。Meta Quest standalone、PC VR (Oculus / Steam VR)、
//! Apple Vision Pro (visionOS の OpenXR backend) で動く。
//!
//! このクレートは OpenXR の Session 管理 / レンダリングそのものは扱わない。
//! 1 frame ごとに `XrPose` (head / controller / hand) を受け取り、ALICE-SDF の
//! 表面ヒットや距離クエリを返すユーティリティのみ提供。
//!
//! # 典型的なフロー
//!
//! ```ignore
//! use alice_sdf_openxr::{XrPose, raymarch_sphere};
//! use alice_sdf::prelude::*;
//!
//! // OpenXR frame callback 内で:
//! let head: XrPose = read_head_pose_from_openxr_action(&session);
//! let sphere = SdfNode::sphere(0.3);
//! let hit_dist = raymarch_sphere(head, glam::Vec3::new(0.0, 1.5, -1.0), 0.3, 5.0);
//! if hit_dist > 0.0 {
//!     // head is looking at the sphere
//! }
//! ```

use glam::{Quat, Vec3};

/// OpenXR pose (位置 + 向き)。`openxr::Posef` から変換可能。
#[derive(Clone, Copy, Debug, Default)]
pub struct XrPose {
    pub position: Vec3,
    pub orientation: Quat,
}

impl From<openxr::Posef> for XrPose {
    fn from(p: openxr::Posef) -> Self {
        Self {
            position: Vec3::new(p.position.x, p.position.y, p.position.z),
            orientation: Quat::from_xyzw(
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ),
        }
    }
}

impl XrPose {
    /// Pose の前方向ベクトル (local -Z を world に変換)
    #[must_use]
    pub fn forward(&self) -> Vec3 {
        self.orientation * Vec3::NEG_Z
    }

    /// Pose の右方向ベクトル (local +X)
    #[must_use]
    pub fn right(&self) -> Vec3 {
        self.orientation * Vec3::X
    }

    /// Pose の上方向ベクトル (local +Y)
    #[must_use]
    pub fn up(&self) -> Vec3 {
        self.orientation * Vec3::Y
    }
}

/// 単純な球 SDF への raymarching。ヒット時の距離、ヒットしなければ `-1.0`
#[must_use]
pub fn raymarch_sphere(origin_pose: XrPose, center: Vec3, radius: f32, max_dist: f32) -> f32 {
    use alice_sdf::primitives::sdf_sphere_at;
    let origin = origin_pose.position;
    let dir = origin_pose.forward();
    if dir.length_squared() < 1e-12 {
        return -1.0;
    }
    let mut t = 0.0_f32;
    for _ in 0..96 {
        let p = origin + dir * t;
        let d = sdf_sphere_at(p, center, radius);
        if d.abs() < 0.001 {
            return t;
        }
        t += d.max(0.001);
        if t > max_dist {
            return -1.0;
        }
    }
    -1.0
}

/// ヘッド/コントローラ位置から見て、与えた SDF 表面までの距離 (単一サンプル、raymarching なし)
#[must_use]
pub fn distance_to_sphere(pose: XrPose, center: Vec3, radius: f32) -> f32 {
    use alice_sdf::primitives::sdf_sphere_at;
    sdf_sphere_at(pose.position, center, radius)
}

/// 2 球 smooth-union シーンへのヒットテスト (VR デモ用 reference)
#[must_use]
pub fn raymarch_two_spheres_smooth(
    origin_pose: XrPose,
    c1: Vec3,
    r1: f32,
    c2: Vec3,
    r2: f32,
    k: f32,
    max_dist: f32,
) -> f32 {
    use alice_sdf::operations::sdf_smooth_union;
    use alice_sdf::primitives::sdf_sphere_at;
    let origin = origin_pose.position;
    let dir = origin_pose.forward();
    if dir.length_squared() < 1e-12 {
        return -1.0;
    }
    let mut t = 0.0_f32;
    for _ in 0..128 {
        let p = origin + dir * t;
        let d1 = sdf_sphere_at(p, c1, r1);
        let d2 = sdf_sphere_at(p, c2, r2);
        let d = sdf_smooth_union(d1, d2, k);
        if d.abs() < 0.001 {
            return t;
        }
        t += d.max(0.001);
        if t > max_dist {
            return -1.0;
        }
    }
    -1.0
}

/// ハンドメッシュ全頂点に対するバッチ SDF 距離評価
///
/// WebXR の `hand-tracking` / OpenXR の `XR_FB_hand_tracking_mesh` で取得した
/// 約 250 頂点の手モデルに対し、SDF シーンへの距離を一括計算。
#[must_use]
pub fn distance_batch_to_sphere(
    points: &[Vec3],
    center: Vec3,
    radius: f32,
) -> Vec<f32> {
    use alice_sdf::primitives::sdf_sphere_at;
    points
        .iter()
        .map(|&p| sdf_sphere_at(p, center, radius))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_forward_is_neg_z_at_identity() {
        let p = XrPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let f = p.forward();
        assert!((f - Vec3::NEG_Z).length() < 1e-5);
    }

    #[test]
    fn raymarch_hits_sphere_in_front() {
        let p = XrPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        // -Z 方向 2m 先に半径 0.5 球
        let hit = raymarch_sphere(p, Vec3::new(0.0, 0.0, -2.0), 0.5, 5.0);
        assert!(hit > 0.0 && hit < 5.0, "expected hit, got {hit}");
    }

    #[test]
    fn raymarch_misses_off_axis() {
        let p = XrPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let hit = raymarch_sphere(p, Vec3::new(5.0, 0.0, -2.0), 0.1, 5.0);
        assert!(hit < 0.0, "expected miss, got {hit}");
    }

    #[test]
    fn distance_batch_matches_single() {
        let pts = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0)];
        let c = Vec3::ZERO;
        let batch = distance_batch_to_sphere(&pts, c, 1.0);
        assert_eq!(batch.len(), 2);
        assert!((batch[0] - 0.0).abs() < 1e-4);
        assert!((batch[1] - 1.0).abs() < 1e-4);
    }
}
