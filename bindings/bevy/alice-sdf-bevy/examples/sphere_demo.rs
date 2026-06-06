//! Bevy demo: SDF sphere + box + torus が並ぶ
//!
//! ```bash
//! cargo run --example sphere_demo
//! ```

use alice_sdf_bevy::{AliceSdfPlugin, SdfShape};
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(AliceSdfPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let red = materials.add(StandardMaterial::from_color(Color::srgb(0.9, 0.3, 0.3)));
    let green = materials.add(StandardMaterial::from_color(Color::srgb(0.3, 0.9, 0.3)));
    let blue = materials.add(StandardMaterial::from_color(Color::srgb(0.3, 0.3, 0.9)));

    commands.spawn((
        SdfShape::Sphere { radius: 1.0 },
        MeshMaterial3d(red),
        Transform::from_xyz(-2.5, 0.0, 0.0),
    ));
    commands.spawn((
        SdfShape::Box {
            half_extents: Vec3::splat(0.8),
        },
        MeshMaterial3d(green),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
    commands.spawn((
        SdfShape::Torus {
            major_radius: 0.8,
            minor_radius: 0.3,
        },
        MeshMaterial3d(blue),
        Transform::from_xyz(2.5, 0.0, 0.0),
    ));
}
