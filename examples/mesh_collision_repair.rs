//! Collision proxies, mesh validation / repair, and fitting primitives to a
//! point cloud.
//!
//! The inputs are a box (volume and bounds in closed form) and a sphere mesh
//! (Euler characteristic V − E + F = 2), so each printed number is checked
//! against the value it must have.
//!
//! Run: `cargo run --example mesh_collision_repair`
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::mesh::primitive_fitting::FittedPrimitiveKind;
use alice_sdf::mesh::{
    compute_aabb, compute_bounding_sphere, compute_convex_hull, compute_quality,
    convex_decomposition, convex_hull_from_points, detect_primitive, fit_box, fit_cylinder,
    fit_plane, fit_sphere, primitives_to_csg, sdf_to_mesh, simplify_collision, validate_mesh,
    BoundingSphere, CollisionAabb, CollisionMesh, ConvexDecomposition, ConvexHull, FittedPrimitive,
    FittingConfig, FittingResult, MarchingCubesConfig, Mesh, MeshQuality, MeshRepair,
    MeshValidation, Vertex, VhacdConfig,
};
use alice_sdf::types::SdfNode;
use glam::Vec3;
use std::collections::HashSet;

fn box_mesh(center: Vec3, half: Vec3) -> Mesh {
    let c = |x: f32, y: f32, z: f32| center + half * Vec3::new(x, y, z);
    let corners = [
        c(-1., -1., -1.),
        c(1., -1., -1.),
        c(1., 1., -1.),
        c(-1., 1., -1.),
        c(-1., -1., 1.),
        c(1., -1., 1.),
        c(1., 1., 1.),
        c(-1., 1., 1.),
    ];
    Mesh {
        vertices: corners
            .iter()
            .map(|&q| Vertex::new(q, (q - center).normalize()))
            .collect(),
        indices: vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7,
            3, 1, 2, 6, 1, 6, 5,
        ],
    }
}

fn euler(indices: &[u32]) -> i64 {
    let mut v = HashSet::new();
    let mut e = HashSet::new();
    for t in indices.chunks(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            v.insert(a);
            e.insert((a.min(b), a.max(b)));
        }
    }
    v.len() as i64 - e.len() as i64 + (indices.len() / 3) as i64
}

fn main() {
    // ── collision proxies ────────────────────────────────────
    let half = Vec3::new(1.0, 2.0, 3.0);
    let bx = box_mesh(Vec3::ZERO, half);
    let aabb: CollisionAabb = compute_aabb(&bx);
    let sphere: BoundingSphere = compute_bounding_sphere(&bx);
    println!(
        "box: aabb centre {} half {} volume {} | bounding sphere r = {:.4} (|half| = {:.4})",
        aabb.center(),
        aabb.half_extents(),
        aabb.volume(),
        sphere.radius,
        half.length()
    );
    assert_eq!(aabb.volume(), 48.0);
    assert!(aabb.contains(Vec3::ZERO) && sphere.contains(half));
    assert!((sphere.radius - half.length()).abs() < 1e-5);

    let mut cloud: Vec<Vec3> = bx.vertices.iter().map(|v| v.position).collect();
    cloud.extend([Vec3::new(0.2, 0.3, -1.0), Vec3::new(-0.5, 1.0, 2.0)]);
    let hull: ConvexHull = convex_hull_from_points(&cloud);
    let hull_mesh: ConvexHull = compute_convex_hull(&bx);
    let vol: f32 = hull
        .indices
        .chunks(3)
        .map(|t| {
            let (a, b, c) = (
                hull.vertices[t[0] as usize],
                hull.vertices[t[1] as usize],
                hull.vertices[t[2] as usize],
            );
            a.dot(b.cross(c)) / 6.0
        })
        .sum();
    println!(
        "convex hull: {} triangles, volume {vol:.4} (box 48), from mesh {} triangles",
        hull.indices.len() / 3,
        hull_mesh.indices.len() / 3
    );
    assert!((vol - 48.0).abs() < 1e-3);
    assert_eq!(hull_mesh.indices.len() / 3, 12);

    let ball = sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig::default(),
    );
    let proxy: CollisionMesh = simplify_collision(&ball, 6);
    println!(
        "simplified collision mesh: {} → {} vertices, {} triangles",
        ball.vertices.len(),
        proxy.vertices.len(),
        proxy.indices.len() / 3
    );
    assert!(proxy.vertices.len() <= 216 && proxy.vertices.len() < ball.vertices.len());

    let mut two = box_mesh(Vec3::new(0.0, -3.0, -3.0), Vec3::ONE);
    let b = box_mesh(Vec3::new(0.0, 3.0, 3.0), Vec3::ONE);
    let off = two.vertices.len() as u32;
    two.vertices.extend_from_slice(&b.vertices);
    two.indices.extend(b.indices.iter().map(|i| i + off));
    for (name, cfg) in [
        ("fast", VhacdConfig::fast()),
        ("default", VhacdConfig::default()),
        ("high quality", VhacdConfig::high_quality()),
    ] {
        let dec: ConvexDecomposition = convex_decomposition(&two, &cfg);
        println!(
            "decomposition ({name}): {} parts, {} vertices, {} triangles",
            dec.parts.len(),
            dec.total_vertices(),
            dec.total_triangles()
        );
        assert_eq!(dec.parts.len(), 2);
    }

    // ── validation and repair ────────────────────────────────
    let report: MeshValidation = validate_mesh(&ball);
    println!("{report}");
    println!("euler characteristic {}", euler(&ball.indices));
    assert_eq!(euler(&ball.indices), 2);
    assert!(report.is_clean());

    let mut holed = ball.clone();
    holed.indices.drain(0..3);
    let mut scrambled = holed.clone();
    for t in scrambled.indices.chunks_mut(6) {
        t.swap(1, 2);
    }
    let oriented = MeshRepair::orient_faces(&scrambled);
    let filled = MeshRepair::fill_holes(&oriented);
    let v = validate_mesh(&filled);
    println!(
        "hole: boundary edges {} → {}, euler {} → {}",
        validate_mesh(&holed).boundary_edges,
        v.boundary_edges,
        euler(&holed.indices),
        euler(&filled.indices)
    );
    assert_eq!(v.boundary_edges, 0);
    assert_eq!(euler(&filled.indices), 2);

    let mut specked = ball.clone();
    let speck = box_mesh(Vec3::splat(5.0), Vec3::splat(0.01));
    let off = specked.vertices.len() as u32;
    specked.vertices.extend_from_slice(&speck.vertices);
    specked
        .indices
        .extend(speck.indices.iter().map(|i| i + off));
    let cleaned = MeshRepair::drop_specks(&specked, 0.01);
    println!(
        "specks: {} → {} triangles",
        specked.indices.len() / 3,
        cleaned.indices.len() / 3
    );
    assert_eq!(cleaned.indices.len(), ball.indices.len());

    let quality: MeshQuality = compute_quality(&ball);
    println!(
        "quality: aspect min {:.3} avg {:.3} (1 = equilateral), area {:.4} (4π = {:.4})",
        quality.min_aspect_ratio,
        quality.avg_aspect_ratio,
        quality.total_area,
        4.0 * std::f32::consts::PI
    );
    assert!(quality.avg_aspect_ratio > 0.5 && quality.avg_aspect_ratio <= 1.0);
    assert!((quality.total_area - 4.0 * std::f32::consts::PI).abs() < 0.1);

    // ── primitive fitting ────────────────────────────────────
    let cfg = FittingConfig::default();
    let centre = Vec3::new(0.3, -0.2, 0.5);
    let pts: Vec<Vec3> = (0..2000)
        .map(|i| {
            // Fibonacci sphere
            let k = i as f32 + 0.5;
            let y = 1.0 - 2.0 * k / 2000.0;
            let rr = (1.0 - y * y).sqrt();
            let a = k * std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
            centre + Vec3::new(rr * a.cos(), y, rr * a.sin()) * 1.7
        })
        .collect();
    let fit: FittingResult = fit_sphere(&pts, &cfg).expect("enough points");
    println!(
        "fit sphere: {:?}, mse {:.2e}, acceptable {}",
        fit.primitive,
        fit.mse,
        fit.is_acceptable(cfg.max_mse, cfg.min_inlier_ratio, pts.len())
    );
    match fit.primitive {
        FittedPrimitive::Sphere { center, radius } => {
            assert!((center - centre).length() < 1e-3 && (radius - 1.7).abs() < 1e-3);
        }
        _ => unreachable!("fit_sphere returns a sphere"),
    }
    let best = detect_primitive(&pts, &cfg).expect("fit");
    let kind: FittedPrimitiveKind = best.primitive.kind();
    println!(
        "detect: {kind:?}, error {:.2e}",
        best.primitive.compute_error(&pts)
    );
    assert_eq!(kind, FittedPrimitiveKind::Sphere);

    let box_pts: Vec<Vec3> = bx
        .vertices
        .iter()
        .map(|v| v.position)
        .chain([Vec3::new(0.0, 0.0, 3.0), Vec3::new(1.0, 0.0, 0.0)])
        .collect();
    let fb = fit_box(&box_pts, &cfg).expect("box");
    let cyl_pts: Vec<Vec3> = (0..64)
        .map(|i| {
            let a = std::f32::consts::TAU * (i % 16) as f32 / 16.0;
            Vec3::new(0.5 * a.cos(), (i / 16) as f32 * 0.5 - 0.75, 0.5 * a.sin())
        })
        .collect();
    let fc = fit_cylinder(&cyl_pts, &cfg).expect("cylinder");
    let plane_pts: Vec<Vec3> = (0..25)
        .map(|i| Vec3::new((i % 5) as f32, 2.0, (i / 5) as f32))
        .collect();
    let fp = fit_plane(&plane_pts, &cfg).expect("plane");
    println!(
        "fit box {:?}\nfit cylinder {:?}\nfit plane {:?}",
        fb.primitive, fc.primitive, fp.primitive
    );
    assert!(fb.max_error < 1e-6 && fc.max_error < 1e-5 && fp.max_error < 1e-5);
    match fp.primitive {
        FittedPrimitive::Plane { normal, distance } => {
            assert!((normal.y.abs() - 1.0).abs() < 1e-6 && (distance.abs() - 2.0).abs() < 1e-5);
        }
        _ => unreachable!("fit_plane returns a plane"),
    }

    // the fitted primitives as an editable CSG tree
    let parts = [fit.primitive, fb.primitive, fc.primitive];
    let csg = primitives_to_csg(&parts).expect("non-empty");
    for q in [Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), centre] {
        let want = parts
            .iter()
            .map(|p| p.distance(q))
            .fold(f32::INFINITY, f32::min);
        let got = eval(&csg, q);
        println!("  csg at {q}: {got:.4} (min of parts {want:.4})");
        assert!((got - want).abs() < 1e-4);
        for p in &parts {
            assert!((eval(&p.to_sdf_node(), q) - p.distance(q)).abs() < 1e-4);
        }
    }
}
