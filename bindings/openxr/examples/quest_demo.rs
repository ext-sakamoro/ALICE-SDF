//! Meta Quest 3 風の `OpenXR` frame loop デモ
//!
//! 実 `OpenXR` `Session` を初期化せず、ダミー pose を流して
//! `SceneFrame` API の動作を 60 frame シミュレートする。
//! Quest / PCVR で動かす際は `read_*_pose_from_openxr_action` を
//! 各 runtime の `Action` システムに差し替える。
//!
//! Run: `cargo run -p alice-sdf-openxr --example quest_demo`

use alice_sdf_openxr::{SceneFrame, SphereBeacon, XrPose};
use glam::{Quat, Vec3};

#[allow(clippy::cast_precision_loss)]
fn read_head_pose_at(frame: u32) -> XrPose {
    let angle = (frame as f32) * 0.05;
    XrPose {
        position: Vec3::new(0.0, 1.6, 0.0),
        orientation: Quat::from_rotation_y(angle),
    }
}

#[allow(clippy::cast_precision_loss)]
fn read_left_hand_pose_at(frame: u32) -> XrPose {
    let t = (frame as f32) * 0.1;
    XrPose {
        position: Vec3::new(0.1_f32.mul_add(t.sin(), -0.2), 1.2, -0.4),
        orientation: Quat::IDENTITY,
    }
}

#[allow(clippy::cast_precision_loss)]
fn read_right_hand_pose_at(frame: u32) -> XrPose {
    let t = (frame as f32) * 0.1;
    XrPose {
        position: Vec3::new(0.1_f32.mul_add(t.cos(), 0.2), 1.2, -0.4),
        orientation: Quat::IDENTITY,
    }
}

fn main() {
    let beacons = [
        SphereBeacon::new(Vec3::new(0.5, 1.5, -1.5), 0.15),
        SphereBeacon::new(Vec3::new(-0.5, 1.5, -1.5), 0.15),
        SphereBeacon::new(Vec3::new(0.0, 1.5, -2.5), 0.20),
    ];

    let mut head_hits = 0u32;
    let mut left_hits = 0u32;
    let mut right_hits = 0u32;

    for f in 0..60 {
        let scene = SceneFrame::new(read_head_pose_at(f))
            .with_left(read_left_hand_pose_at(f))
            .with_right(read_right_hand_pose_at(f))
            .with_max_distance(8.0);
        let scene = beacons.iter().fold(scene, |s, b| s.with_beacon(*b));

        if let Some(h) = scene.head_raycast() {
            head_hits += 1;
            if f % 10 == 0 {
                println!(
                    "frame {f:>3}: head -> beacon {} @ {:.2}m",
                    h.beacon, h.distance
                );
            }
        }
        if scene.left_raycast().is_some() {
            left_hits += 1;
        }
        if scene.right_raycast().is_some() {
            right_hits += 1;
        }
    }

    println!("\n=== Summary over 60 frames ===");
    println!("head hits  : {head_hits}");
    println!("left hits  : {left_hits}");
    println!("right hits : {right_hits}");
}
