//! Mesh presets — AAA extraction, compiled meshers, decimation presets and
//! UV layout checks
//!
//! Meshes the unit sphere with the all-attribute presets
//! (`MarchingCubesConfig::aaa`, `AdaptiveConfig::aaa` through
//! `adaptive_marching_cubes_compiled`, `DualContouringConfig::aaa` through
//! `dual_contouring_compiled`), decimates with the conservative and
//! aggressive presets, lays out lightmap UVs (atlas and fast variants) and
//! measures UV texel density. Each step prints its numbers and checks them
//! against the sphere's closed forms with `assert!`; the oracle is
//! `tests/test_mesh_extract_uv_oracle.rs`.
//!
//! # Running
//! ```bash
//! cargo run --example mesh_presets
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::CompiledSdf;
use alice_sdf::mesh::uv_unwrap::{compute_uv_density, UvDensityReport};
use alice_sdf::mesh::{
    adaptive_marching_cubes_compiled, decimate, dual_contouring_compiled, generate_lightmap_uvs,
    generate_lightmap_uvs_fast, sdf_to_mesh, AdaptiveConfig, DecimateConfig, DualContouringConfig,
    MarchingCubesConfig, Mesh, Vertex,
};
use alice_sdf::SdfNode;
use glam::{Vec2, Vec3, Vec4};

fn radial_error(m: &Mesh) -> f32 {
    m.vertices
        .iter()
        .map(|v| (v.position.length() - 1.0).abs())
        .fold(0.0, f32::max)
}

fn volume(m: &Mesh) -> f32 {
    m.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| m.vertices[t[k] as usize].position);
            a.dot(b.cross(c)) / 6.0
        })
        .sum()
}

fn show(name: &str, m: &Mesh, cell: f32) {
    let err = radial_error(m);
    let vol = volume(m);
    let want = 4.0 / 3.0 * std::f32::consts::PI;
    println!(
        "  {name:<28} {:>6} tris  max ||p|-1| {err:.4}  volume {vol:.4} (4pi/3 = {want:.4})",
        m.triangle_count()
    );
    assert!(err <= cell * 3f32.sqrt());
    assert!((vol - want).abs() <= 0.05 * want);
}

fn main() {
    println!("ALICE-SDF — mesh presets");
    println!("========================");

    let node = SdfNode::sphere(1.0);
    let compiled = CompiledSdf::compile(&node);
    let (lo, hi) = (Vec3::splat(-1.5), Vec3::splat(1.5));
    let cell = 3.0 / 32.0;

    println!("extraction (unit sphere, 32 cells per axis):");
    let mc = sdf_to_mesh(&node, lo, hi, &MarchingCubesConfig::aaa(32));
    show("marching cubes aaa", &mc, cell);
    // the aaa preset fills tangents: unit length, orthogonal to the normal
    assert!(mc.vertices.iter().all(|v| {
        let t = v.tangent.truncate();
        (t.length() - 1.0).abs() < 1e-3 && t.dot(v.normal).abs() < 1e-3
    }));
    let ad = adaptive_marching_cubes_compiled(&compiled, lo, hi, &AdaptiveConfig::aaa(5));
    show("adaptive (compiled) aaa", &ad, cell);
    let dc = dual_contouring_compiled(&compiled, lo, hi, &DualContouringConfig::aaa(32));
    show("dual contouring (compiled) aaa", &dc, cell);

    println!("decimation:");
    let base = sdf_to_mesh(
        &node,
        lo,
        hi,
        &MarchingCubesConfig {
            resolution: 32,
            ..Default::default()
        },
    );
    let mut cons = base.clone();
    decimate(&mut cons, &DecimateConfig::conservative());
    let mut aggr = base.clone();
    decimate(&mut aggr, &DecimateConfig::aggressive());
    println!(
        "  {} -> conservative {} (max ||p|-1| {:.4}) -> aggressive {} (max ||p|-1| {:.4})",
        base.triangle_count(),
        cons.triangle_count(),
        radial_error(&cons),
        aggr.triangle_count(),
        radial_error(&aggr)
    );
    assert!(aggr.triangle_count() < cons.triangle_count());
    assert!(cons.triangle_count() < base.triangle_count());
    assert!(radial_error(&cons) <= 0.1 + radial_error(&base));

    println!("lightmap UVs:");
    let mut lm = base.clone();
    generate_lightmap_uvs(&mut lm, 1024);
    let inside = lm
        .vertices
        .iter()
        .all(|v| (0.0..=1.0).contains(&v.uv2.x) && (0.0..=1.0).contains(&v.uv2.y));
    println!(
        "  atlas: {} vertices after seam splits (was {}), all UV2 in [0,1]^2: {inside}",
        lm.vertex_count(),
        base.vertex_count()
    );
    assert!(inside);
    let mut fast = base;
    generate_lightmap_uvs_fast(&mut fast);
    let v0 = fast.vertices[0];
    println!("  fast: vertex 0 at {:?} -> UV2 {:?}", v0.position, v0.uv2);
    assert!(fast
        .vertices
        .iter()
        .all(|v| (0.0..=1.0).contains(&v.uv2.x) && (0.0..=1.0).contains(&v.uv2.y)));

    println!("UV texel density:");
    let quad = |s: f32| {
        let v = |u: f32, w: f32| {
            Vertex::with_all(
                Vec3::new(u, w, 0.0),
                Vec3::Z,
                Vec2::new(u, w) * s,
                Vec2::ZERO,
                Vec4::new(1.0, 0.0, 0.0, 1.0),
                [1.0; 4],
                0,
            )
        };
        let mut m = Mesh::new();
        m.vertices = vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)];
        m.indices = vec![0, 1, 2, 0, 2, 3];
        m
    };
    for (scale, size) in [(1.0f32, 1024u32), (0.01, 1024)] {
        let r: UvDensityReport = compute_uv_density(&quad(scale), size);
        let want = 0.5 * scale * scale * (size * size) as f32;
        println!("  UV scale {scale}, {size}^2 texture: {:.1} texels/face (closed form {want:.1}), warning {}",
            r.avg_texels_per_face, r.has_warning());
        assert!((r.avg_texels_per_face - want).abs() <= 1e-3 * want);
        assert_eq!(
            r.has_warning(),
            want < UvDensityReport::RECOMMENDED_MIN_TEXELS_PER_FACE
        );
    }
    println!("{}", compute_uv_density(&quad(0.01), 1024));
}
