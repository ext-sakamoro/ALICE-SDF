//! `ALICE-SDF` `OpenXR` helpers
//!
//! Native VR / AR アプリ向けのヘルパー集。Meta Quest standalone、PC VR (Oculus / Steam VR)、
//! Apple Vision Pro (`visionOS` の `OpenXR` backend) で動く。
//!
//! このクレートは `OpenXR` の `Session` 管理 / レンダリングそのものは扱わない。
//! 1 frame ごとに `XrPose` (head / controller / hand) を受け取り、`ALICE-SDF` の
//! 表面ヒットや距離クエリを返すユーティリティのみ提供。
//!
//! # 典型的なフロー
//!
//! ```ignore
//! use alice_sdf_openxr::{SceneFrame, SphereBeacon, XrPose};
//!
//! // OpenXR frame callback 内で:
//! let head: XrPose = read_head_pose_from_openxr_action(&session);
//! let left: XrPose = read_left_hand_pose(&session);
//! let right: XrPose = read_right_hand_pose(&session);
//!
//! let scene = SceneFrame::new(head)
//!     .with_left(left)
//!     .with_right(right)
//!     .with_beacon(SphereBeacon::new(glam::Vec3::new(0.0, 1.5, -1.0), 0.3));
//!
//! if let Some(hit) = scene.head_raycast() {
//!     // head looking at beacon at hit.distance, beacon index hit.beacon
//! }
//! ```

use glam::{Quat, Vec3};

/// `OpenXR` pose (位置 + 向き)。`openxr::Posef` から変換可能。
#[derive(Clone, Copy, Debug, Default)]
pub struct XrPose {
    /// world-space 位置
    pub position: Vec3,
    /// world-space 向き
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

    /// pose を `glam::Affine3A` 相当の 4x4 行列で取得
    #[must_use]
    pub fn matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.orientation, self.position)
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
/// `WebXR` の `hand-tracking` / `OpenXR` の `XR_FB_hand_tracking_mesh` で取得した
/// 約 250 頂点の手モデルに対し、SDF シーンへの距離を一括計算。
#[must_use]
pub fn distance_batch_to_sphere(points: &[Vec3], center: Vec3, radius: f32) -> Vec<f32> {
    use alice_sdf::primitives::sdf_sphere_at;
    points
        .iter()
        .map(|&p| sdf_sphere_at(p, center, radius))
        .collect()
}

// ── Scene / Beacon API (新規 v1.7.1) ────────────────────────────────

/// シーン中の球 beacon (raymarch ターゲット)
#[derive(Clone, Copy, Debug)]
pub struct SphereBeacon {
    /// world-space 中心
    pub center: Vec3,
    /// 半径
    pub radius: f32,
}

impl SphereBeacon {
    /// 新しい beacon を生成
    #[must_use]
    pub const fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }
}

/// `head_raycast` / `controller_raycast` の戻り値
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// シーン内の beacon インデックス
    pub beacon: usize,
    /// ray origin からヒット点までの距離
    pub distance: f32,
}

/// `OpenXR` frame callback 1 回分のシーン状態
///
/// 各 frame 毎にゼロから組み立てて使い捨てる想定。Alloc を避けるため
/// `beacons` は `smallvec`-like な inline buffer を使わず単純 Vec で構築するが、
/// 1 frame あたり数本〜十数本の beacon が想定なので影響は無視できる。
#[derive(Clone, Debug, Default)]
pub struct SceneFrame {
    /// ヘッドセット pose (HMD)
    pub head: XrPose,
    /// 左手 pose (None なら未トラッキング)
    pub left_hand: Option<XrPose>,
    /// 右手 pose
    pub right_hand: Option<XrPose>,
    /// シーン中の球 beacon
    pub beacons: Vec<SphereBeacon>,
    /// raymarch 最大距離 [m]
    pub max_distance: f32,
}

impl SceneFrame {
    /// 新しい frame を head pose だけで初期化
    #[must_use]
    pub const fn new(head: XrPose) -> Self {
        Self {
            head,
            left_hand: None,
            right_hand: None,
            beacons: Vec::new(),
            max_distance: 10.0,
        }
    }

    /// 左手 pose を設定
    #[must_use]
    pub const fn with_left(mut self, p: XrPose) -> Self {
        self.left_hand = Some(p);
        self
    }

    /// 右手 pose を設定
    #[must_use]
    pub const fn with_right(mut self, p: XrPose) -> Self {
        self.right_hand = Some(p);
        self
    }

    /// beacon を追加
    #[must_use]
    pub fn with_beacon(mut self, b: SphereBeacon) -> Self {
        self.beacons.push(b);
        self
    }

    /// 最大 raymarch 距離を変更
    #[must_use]
    pub const fn with_max_distance(mut self, m: f32) -> Self {
        self.max_distance = m;
        self
    }

    /// 与えた pose から forward に raymarch し、最初にヒットした beacon を返す
    #[must_use]
    pub fn raycast(&self, origin: XrPose) -> Option<RayHit> {
        let mut best: Option<RayHit> = None;
        for (i, b) in self.beacons.iter().enumerate() {
            let d = raymarch_sphere(origin, b.center, b.radius, self.max_distance);
            if d > 0.0 && best.map_or(true, |h| d < h.distance) {
                best = Some(RayHit {
                    beacon: i,
                    distance: d,
                });
            }
        }
        best
    }

    /// ヘッドから raycast
    #[must_use]
    pub fn head_raycast(&self) -> Option<RayHit> {
        self.raycast(self.head)
    }

    /// 左コントローラから raycast (左手がない場合 None)
    #[must_use]
    pub fn left_raycast(&self) -> Option<RayHit> {
        self.left_hand.and_then(|p| self.raycast(p))
    }

    /// 右コントローラから raycast (右手がない場合 None)
    #[must_use]
    pub fn right_raycast(&self) -> Option<RayHit> {
        self.right_hand.and_then(|p| self.raycast(p))
    }

    /// 手メッシュ頂点群と全 beacon の最小距離 (掴み判定向け)
    ///
    /// 各 beacon 毎に「全頂点との最短距離」を返す。返値長 = `beacons.len()`
    #[must_use]
    pub fn min_distance_to_beacons(&self, hand_points: &[Vec3]) -> Vec<f32> {
        self.beacons
            .iter()
            .map(|b| {
                hand_points
                    .iter()
                    .map(|p| alice_sdf::primitives::sdf_sphere_at(*p, b.center, b.radius))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect()
    }
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
    fn pose_right_up_orthogonal_at_identity() {
        let p = XrPose::default();
        assert!(p.right().dot(p.up()).abs() < 1e-5);
        assert!(p.right().dot(p.forward()).abs() < 1e-5);
        assert!(p.up().dot(p.forward()).abs() < 1e-5);
    }

    #[test]
    fn pose_matrix_round_trip() {
        let p = XrPose {
            position: Vec3::new(1.0, 2.0, 3.0),
            orientation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
        };
        let m = p.matrix();
        let recovered_pos = m.transform_point3(Vec3::ZERO);
        assert!((recovered_pos - p.position).length() < 1e-5);
    }

    #[test]
    fn raymarch_hits_sphere_in_front() {
        let p = XrPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
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
    fn raymarch_zero_direction_returns_miss() {
        let p = XrPose {
            position: Vec3::ZERO,
            orientation: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        };
        let hit = raymarch_sphere(p, Vec3::ZERO, 1.0, 5.0);
        assert!(hit < 0.0);
    }

    #[test]
    fn distance_batch_matches_single() {
        let pts = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0)];
        let batch = distance_batch_to_sphere(&pts, Vec3::ZERO, 1.0);
        assert_eq!(batch.len(), 2);
        assert!((batch[0] - 0.0).abs() < 1e-4);
        assert!((batch[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn scene_head_raycast_hits_nearest_beacon() {
        let head = XrPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let scene = SceneFrame::new(head)
            .with_beacon(SphereBeacon::new(Vec3::new(0.0, 0.0, -1.0), 0.3))
            .with_beacon(SphereBeacon::new(Vec3::new(0.0, 0.0, -4.0), 0.3));
        let hit = scene.head_raycast().expect("must hit nearer beacon");
        assert_eq!(hit.beacon, 0);
        assert!(hit.distance < 1.0);
    }

    #[test]
    fn scene_head_raycast_misses_when_nothing_in_front() {
        let head = XrPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let scene =
            SceneFrame::new(head).with_beacon(SphereBeacon::new(Vec3::new(0.0, 0.0, 4.0), 0.3));
        assert!(scene.head_raycast().is_none());
    }

    #[test]
    fn scene_left_raycast_none_without_left_hand() {
        let scene = SceneFrame::new(XrPose::default())
            .with_beacon(SphereBeacon::new(Vec3::new(0.0, 0.0, -1.0), 0.5));
        assert!(scene.left_raycast().is_none());
    }

    #[test]
    fn scene_min_distance_to_beacons_returns_min_per_beacon() {
        let scene = SceneFrame::new(XrPose::default())
            .with_beacon(SphereBeacon::new(Vec3::new(1.0, 0.0, 0.0), 0.1))
            .with_beacon(SphereBeacon::new(Vec3::new(-1.0, 0.0, 0.0), 0.1));
        let hand = vec![
            Vec3::new(1.05, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, 5.0, 0.0),
        ];
        let d = scene.min_distance_to_beacons(&hand);
        assert_eq!(d.len(), 2);
        // beacon0 は hand[0] (距離 -0.05 程度) が最近
        assert!(d[0] < 0.0);
        // beacon1 は hand[1] (距離 0.4 程度)
        assert!(d[1] > 0.0);
    }

    #[test]
    fn two_spheres_smooth_hits_when_aligned() {
        let p = XrPose::default();
        let hit = raymarch_two_spheres_smooth(
            p,
            Vec3::new(0.0, 0.0, -2.0),
            0.3,
            Vec3::new(0.3, 0.0, -2.0),
            0.3,
            0.2,
            5.0,
        );
        assert!(hit > 0.0, "expected hit, got {hit}");
    }
}
