//! ALICE-SDF Bevy plugin
//!
//! `AliceSdfPlugin` を `App::add_plugins` で登録し、`SdfShape` Component を持つ
//! Entity を spawn すれば、自動的に Mesh を生成 + `MeshMaterial3d` に attach する。
//!
//! # 最小例
//!
//! ```no_run
//! use bevy::prelude::*;
//! use alice_sdf_bevy::{AliceSdfPlugin, SdfShape};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(AliceSdfPlugin)
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands) {
//!     commands.spawn(SdfShape::Sphere { radius: 1.0 });
//! }
//! ```

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// ALICE-SDF Bevy plugin。`MeshSpawner` system を登録する。
pub struct AliceSdfPlugin;

impl Plugin for AliceSdfPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, generate_sdf_meshes);
    }
}

/// SDF shape Component。最小公開セット (sphere / box / torus / cylinder)。
#[derive(Component, Clone, Copy, Debug)]
pub enum SdfShape {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: Vec3,
    },
    Torus {
        major_radius: f32,
        minor_radius: f32,
    },
    Cylinder {
        radius: f32,
        half_height: f32,
    },
}

/// Marker: `SdfShape` から Mesh 生成済 (重複防止用)。
#[derive(Component)]
pub struct SdfMeshBuilt;

/// `SdfShape` を持ち `SdfMeshBuilt` がない Entity を発見、Mesh を生成して attach。
fn generate_sdf_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(Entity, &SdfShape), Without<SdfMeshBuilt>>,
) {
    for (entity, shape) in &query {
        let mesh = build_mesh_for(*shape);
        let handle = meshes.add(mesh);
        commands
            .entity(entity)
            .insert(Mesh3d(handle))
            .insert(SdfMeshBuilt);
    }
}

/// 形状から Bevy Mesh を構築 (最小実装、本格的には ALICE-SDF の dual_contouring / marching cubes を呼ぶ)。
fn build_mesh_for(shape: SdfShape) -> Mesh {
    match shape {
        SdfShape::Sphere { radius } => uv_sphere_mesh(radius, 24, 16),
        SdfShape::Box { half_extents } => box_mesh(half_extents),
        SdfShape::Torus {
            major_radius,
            minor_radius,
        } => torus_mesh(major_radius, minor_radius, 24, 12),
        SdfShape::Cylinder {
            radius,
            half_height,
        } => cylinder_mesh(radius, half_height, 24),
    }
}

fn uv_sphere_mesh(radius: f32, sectors: u32, stacks: u32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let pi = std::f32::consts::PI;
    for i in 0..=stacks {
        let phi = (i as f32 / stacks as f32) * pi;
        for j in 0..=sectors {
            let theta = (j as f32 / sectors as f32) * 2.0 * pi;
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            positions.push([x * radius, y * radius, z * radius]);
            normals.push([x, y, z]);
            uvs.push([j as f32 / sectors as f32, i as f32 / stacks as f32]);
        }
    }
    for i in 0..stacks {
        for j in 0..sectors {
            let a = i * (sectors + 1) + j;
            let b = a + sectors + 1;
            indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn box_mesh(he: Vec3) -> Mesh {
    let positions: Vec<[f32; 3]> = vec![
        // +X
        [he.x, -he.y, -he.z],
        [he.x, he.y, -he.z],
        [he.x, he.y, he.z],
        [he.x, -he.y, he.z],
        // -X
        [-he.x, -he.y, he.z],
        [-he.x, he.y, he.z],
        [-he.x, he.y, -he.z],
        [-he.x, -he.y, -he.z],
        // +Y
        [-he.x, he.y, -he.z],
        [-he.x, he.y, he.z],
        [he.x, he.y, he.z],
        [he.x, he.y, -he.z],
        // -Y
        [-he.x, -he.y, he.z],
        [-he.x, -he.y, -he.z],
        [he.x, -he.y, -he.z],
        [he.x, -he.y, he.z],
        // +Z
        [he.x, -he.y, he.z],
        [he.x, he.y, he.z],
        [-he.x, he.y, he.z],
        [-he.x, -he.y, he.z],
        // -Z
        [-he.x, -he.y, -he.z],
        [-he.x, he.y, -he.z],
        [he.x, he.y, -he.z],
        [he.x, -he.y, -he.z],
    ];
    let normals: Vec<[f32; 3]> = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    .iter()
    .flat_map(|n| std::iter::repeat(*n).take(4))
    .collect();
    let uvs: Vec<[f32; 2]> = (0..6)
        .flat_map(|_| [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        .collect();
    let indices: Vec<u32> = (0..6u32)
        .flat_map(|face| {
            let b = face * 4;
            [b, b + 1, b + 2, b, b + 2, b + 3]
        })
        .collect();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn torus_mesh(major: f32, minor: f32, sectors: u32, sides: u32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let pi = std::f32::consts::PI;
    for i in 0..=sectors {
        let u = i as f32 / sectors as f32;
        let theta = u * 2.0 * pi;
        for j in 0..=sides {
            let v = j as f32 / sides as f32;
            let phi = v * 2.0 * pi;
            let cx = major * theta.cos();
            let cy = major * theta.sin();
            let nx = phi.cos() * theta.cos();
            let ny = phi.cos() * theta.sin();
            let nz = phi.sin();
            positions.push([cx + minor * nx, cy + minor * ny, minor * nz]);
            normals.push([nx, ny, nz]);
            uvs.push([u, v]);
        }
    }
    for i in 0..sectors {
        for j in 0..sides {
            let a = i * (sides + 1) + j;
            let b = a + sides + 1;
            indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn cylinder_mesh(radius: f32, half_height: f32, sectors: u32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let pi = std::f32::consts::PI;
    for i in 0..=sectors {
        let theta = (i as f32 / sectors as f32) * 2.0 * pi;
        let (s, c) = (theta.sin(), theta.cos());
        positions.push([c * radius, half_height, s * radius]);
        normals.push([c, 0.0, s]);
        uvs.push([i as f32 / sectors as f32, 0.0]);
        positions.push([c * radius, -half_height, s * radius]);
        normals.push([c, 0.0, s]);
        uvs.push([i as f32 / sectors as f32, 1.0]);
    }
    for i in 0..sectors {
        let a = i * 2;
        indices.extend_from_slice(&[a, a + 1, a + 2, a + 1, a + 3, a + 2]);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    fn positions_of(m: &Mesh) -> Vec<[f32; 3]> {
        match m.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("unexpected position format"),
        }
    }

    fn normals_of(m: &Mesh) -> Vec<[f32; 3]> {
        match m.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap() {
            VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("unexpected normal format"),
        }
    }

    fn index_count(m: &Mesh) -> usize {
        match m.indices().unwrap() {
            Indices::U32(v) => v.len(),
            Indices::U16(v) => v.len(),
        }
    }

    #[test]
    fn build_sphere_mesh_has_indices() {
        let m = build_mesh_for(SdfShape::Sphere { radius: 1.0 });
        assert!(m.indices().is_some());
    }

    #[test]
    fn build_box_mesh_has_24_vertices() {
        let m = build_mesh_for(SdfShape::Box {
            half_extents: Vec3::splat(1.0),
        });
        let pos = m.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        assert_eq!(pos.len(), 24);
    }

    #[test]
    fn build_torus_mesh_is_non_empty() {
        let m = build_mesh_for(SdfShape::Torus {
            major_radius: 1.0,
            minor_radius: 0.3,
        });
        assert!(m.attribute(Mesh::ATTRIBUTE_POSITION).is_some());
    }

    #[test]
    fn build_cylinder_mesh_is_non_empty() {
        let m = build_mesh_for(SdfShape::Cylinder {
            radius: 1.0,
            half_height: 2.0,
        });
        assert!(m.attribute(Mesh::ATTRIBUTE_POSITION).is_some());
    }

    #[test]
    fn sphere_normals_are_unit_length() {
        let m = build_mesh_for(SdfShape::Sphere { radius: 1.5 });
        let normals = normals_of(&m);
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "normal not unit: {len}");
        }
    }

    #[test]
    fn sphere_vertices_within_radius_bound() {
        let r = 0.7_f32;
        let m = build_mesh_for(SdfShape::Sphere { radius: r });
        let positions = positions_of(&m);
        for p in &positions {
            let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(dist <= r + 1e-4, "vertex outside radius: {dist} > {r}");
        }
    }

    #[test]
    fn box_vertices_inside_half_extents() {
        let he = Vec3::new(2.0, 3.0, 4.0);
        let m = build_mesh_for(SdfShape::Box { half_extents: he });
        let positions = positions_of(&m);
        for p in &positions {
            assert!(p[0].abs() <= he.x + 1e-4);
            assert!(p[1].abs() <= he.y + 1e-4);
            assert!(p[2].abs() <= he.z + 1e-4);
        }
    }

    #[test]
    fn all_shapes_indices_are_multiple_of_three() {
        for shape in &[
            SdfShape::Sphere { radius: 1.0 },
            SdfShape::Box {
                half_extents: Vec3::splat(0.5),
            },
            SdfShape::Torus {
                major_radius: 1.0,
                minor_radius: 0.3,
            },
            SdfShape::Cylinder {
                radius: 1.0,
                half_height: 0.5,
            },
        ] {
            let m = build_mesh_for(*shape);
            let n = index_count(&m);
            assert!(n > 0, "empty index buffer for {shape:?}");
            assert_eq!(n % 3, 0, "non-triangle list for {shape:?}: {n} indices");
        }
    }

    #[test]
    fn torus_vertices_in_expected_annulus() {
        let m = build_mesh_for(SdfShape::Torus {
            major_radius: 2.0,
            minor_radius: 0.4,
        });
        let positions = positions_of(&m);
        for p in &positions {
            let r_xy = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (1.5..=2.5).contains(&r_xy),
                "torus xy-radius {r_xy} out of band"
            );
            assert!(p[2].abs() <= 0.4 + 1e-4, "torus z {} outside minor", p[2]);
        }
    }

    #[test]
    fn cylinder_top_bottom_planes() {
        let m = build_mesh_for(SdfShape::Cylinder {
            radius: 1.0,
            half_height: 2.5,
        });
        let positions = positions_of(&m);
        for p in &positions {
            assert!(p[1].abs() - 2.5 < 1e-4, "cylinder y out of caps: {}", p[1]);
        }
    }

    #[test]
    fn plugin_builds_without_panic() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(bevy::asset::AssetPlugin::default())
            .init_asset::<Mesh>()
            .add_plugins(AliceSdfPlugin);
        app.update();
    }
}
