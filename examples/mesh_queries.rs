//! Mesh queries: distance through a BVH, a mesh turned into a signed distance
//! field, the winding-independent exterior sign, a point cloud SDF and Hermite
//! edge data for dual contouring.
//!
//! The mesh is a sphere of radius 1 built from a subdivided octahedron, so every
//! number printed has a closed form to compare with: an inscribed polyhedron is
//! within `h = max_t (r − sqrt(r² − R_t²))` of its sphere (`R_t` the triangle
//! circumradius), so the mesh field reads `|p| − r` to within `h`.
//!
//! Run: `cargo run --example mesh_queries`
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::mesh::{
    extract_edge_crossings, extract_hermite, mesh_to_sdf_exact, Aabb, BvhNode, BvhTriangle,
    EdgeCrossing, ExteriorField, HermiteConfig, HermiteExtractor, HermitePoint, MeshBvh, MeshSdf,
    MeshSignMode, MeshToSdfConfig, PointCloudSdf, PointCloudSdfConfig,
};
use glam::{DVec3, Vec3};
use std::collections::HashMap;

/// Octahedron subdivided `levels` times, projected onto the sphere of radius `r`.
fn sphere_mesh(levels: u32, r: f32) -> (Vec<Vec3>, Vec<u32>) {
    let mut dirs = vec![
        DVec3::X,
        DVec3::NEG_X,
        DVec3::Y,
        DVec3::NEG_Y,
        DVec3::Z,
        DVec3::NEG_Z,
    ];
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    for _ in 0..levels {
        let mut mid: HashMap<(usize, usize), usize> = HashMap::new();
        let mut midpoint = |a: usize, b: usize, dirs: &mut Vec<DVec3>| {
            *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
                dirs.push((dirs[a] + dirs[b]).normalize());
                dirs.len() - 1
            })
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let ab = midpoint(a, b, &mut dirs);
            let bc = midpoint(b, c, &mut dirs);
            let ca = midpoint(c, a, &mut dirs);
            next.extend_from_slice(&[[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
        }
        faces = next;
    }
    let verts = dirs.iter().map(|d| (*d * f64::from(r)).as_vec3()).collect();
    let idx = faces
        .iter()
        .flat_map(|f| f.iter().map(|&i| i as u32))
        .collect();
    (verts, idx)
}

fn hausdorff_bound(verts: &[Vec3], idx: &[u32], r: f32) -> f32 {
    idx.chunks(3)
        .map(|t| {
            let (a, b, c) = (
                verts[t[0] as usize],
                verts[t[1] as usize],
                verts[t[2] as usize],
            );
            let circ = (b - c).length() * (c - a).length() * (a - b).length()
                / (2.0 * (b - a).cross(c - a).length());
            r - (r * r - circ * circ).max(0.0).sqrt()
        })
        .fold(0.0, f32::max)
}

fn main() {
    let r = 1.0_f32;
    let (verts, idx) = sphere_mesh(4, r);
    let h = hausdorff_bound(&verts, &idx, r);
    println!(
        "sphere mesh: {} vertices, {} triangles, h = {h:.2e}",
        verts.len(),
        idx.len() / 3
    );

    // ── one triangle, one box ────────────────────────────────
    let tri = BvhTriangle::new(Vec3::ZERO, Vec3::X, Vec3::Y);
    let p = Vec3::new(1.0, 1.0, 0.0);
    println!(
        "triangle: closest to {p} is {} (edge midpoint), distance {:.4}, signed {:.4}",
        tri.closest_point(p),
        tri.unsigned_distance(p),
        tri.signed_distance(Vec3::new(0.2, 0.2, -1.0))
    );
    assert!((tri.closest_point(p) - Vec3::new(0.5, 0.5, 0.0)).length() < 1e-6);
    assert!((tri.signed_distance(Vec3::new(0.2, 0.2, -1.0)) + 1.0).abs() < 1e-6);

    let mut bx = Aabb::empty();
    bx.expand_point(Vec3::new(-1.0, -2.0, 0.0));
    bx.expand_aabb(&Aabb::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0)));
    println!(
        "aabb: centre {}, area {}, longest axis {}, distance from origin {}",
        bx.center(),
        bx.surface_area(),
        bx.longest_axis(),
        bx.signed_distance(Vec3::ZERO)
    );
    assert_eq!(bx.surface_area(), 2.0 * (2.0 * 4.0 + 4.0 * 3.0 + 3.0 * 2.0));
    assert_eq!(bx.longest_axis(), 1);

    // ── BVH ──────────────────────────────────────────────────
    let bvh = MeshBvh::build(&verts, &idx, 4);
    let root: &BvhNode = bvh.root.as_ref().expect("non-empty mesh");
    let bounds = bvh.bounds().expect("bounds");
    println!(
        "bvh: {} triangles, root box {:?}..{:?}",
        bvh.triangle_count(),
        root.aabb().min,
        bounds.max
    );
    let queries = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.5, 0.0, 0.0),
        Vec3::new(0.3, -0.4, 0.5),
        Vec3::new(-1.2, 1.1, 0.4),
    ];
    let signed = bvh.signed_distance_batch(&queries);
    let unsigned = bvh.unsigned_distance_batch(&queries);
    for (i, &q) in queries.iter().enumerate() {
        let want = q.length() - r;
        let c = bvh.closest_point(q).expect("closest");
        println!(
            "  {q}: signed {:+.4} unsigned {:.4} (|p|−r = {want:+.4}), closest {c}",
            signed[i], unsigned[i]
        );
        assert!((bvh.signed_distance(q) - want).abs() <= h + 1e-5);
        assert!((bvh.unsigned_distance(q) - want.abs()).abs() <= h + 1e-5);
    }

    // ── mesh → SDF ───────────────────────────────────────────
    let robust =
        MeshSdf::try_new(&verts, &idx, &MeshToSdfConfig::topology_robust()).expect("valid mesh");
    let hybrid = MeshSdf::new(&verts, &idx, &MeshToSdfConfig::hybrid()).expect("valid mesh");
    let exact = mesh_to_sdf_exact(&verts, &idx, &MeshToSdfConfig::accurate()).expect("valid");
    assert_eq!(exact.sign_mode(), MeshSignMode::ExteriorFloodFill);
    assert_eq!(hybrid.sign_mode(), MeshSignMode::NearestFaceNormal);
    let (lo, hi) = exact.bounds();
    println!(
        "mesh sdf: {} triangles ({} in its bvh), bounds {lo}..{hi}",
        exact.triangle_count(),
        exact.bvh().triangle_count()
    );
    let values = exact.eval_batch(&queries);
    let uvalues = exact.eval_unsigned_batch(&queries);
    for (i, &q) in queries.iter().enumerate() {
        let want = q.length() - r;
        assert!((values[i] - want).abs() <= h + 1e-5);
        assert_eq!(uvalues[i].to_bits(), exact.eval_unsigned(q).to_bits());
        assert!((robust.eval(q) - want).abs() <= h + 1e-5);
    }
    let g = exact.gradient(Vec3::new(1.5, 0.0, 0.0), 1e-2);
    println!("  gradient at (1.5, 0, 0): {g}");
    assert!(g.x > 0.99);

    // capsule tree for places that need an SdfNode: vertices sit on the capsule axes
    let node = exact.to_sdf_node(0.05);
    println!(
        "  capsule tree: {} nodes, value at a vertex {:.5}",
        node.node_count(),
        eval(&node, verts[0])
    );
    assert!(eval(&node, verts[0]) < 0.0);

    // ── exterior sign field ──────────────────────────────────
    let bvh_ref = exact.bvh();
    let field = ExteriorField::build(bvh_ref, 48).expect("in budget");
    println!(
        "exterior field: cell {:.4}, dims {:?} (padding {} cells, band walk ≤ {} steps, budget {} cells)",
        field.cell_size(),
        field.dims(),
        ExteriorField::PADDING_CELLS,
        ExteriorField::MAX_BAND_STEPS,
        ExteriorField::MAX_CELLS
    );
    assert!(!field.is_exterior(bvh_ref, Vec3::ZERO));
    assert!(field.is_exterior(bvh_ref, Vec3::new(1.5, 0.0, 0.0)));
    assert!(field.signed_distance(bvh_ref, Vec3::ZERO) < 0.0);

    // ── point cloud ──────────────────────────────────────────
    let normals: Vec<Vec3> = verts.iter().map(|v| v.normalize()).collect();
    let cloud = PointCloudSdf::try_new(&verts, &normals, &PointCloudSdfConfig::accurate()).unwrap();
    let fast = PointCloudSdf::try_new(&verts, &normals, &PointCloudSdfConfig::fast()).unwrap();
    let cloud_values = cloud.eval_batch(&queries);
    println!("point cloud: {} points", cloud.point_count());
    for (i, &q) in queries.iter().enumerate() {
        println!(
            "  {q}: {:+.4} (|p|−r = {:+.4})",
            cloud_values[i],
            q.length() - r
        );
        assert_eq!(cloud.eval(q).to_bits(), fast.eval(q).to_bits());
    }
    assert!(cloud.eval(Vec3::ZERO) < 0.0 && cloud.eval(Vec3::new(1.5, 0.0, 0.0)) > 0.0);

    // ── Hermite data ─────────────────────────────────────────
    let ball = move |p: Vec3| p.length() - 0.7;
    let cfg = HermiteConfig {
        resolution: 10,
        ..HermiteConfig::default()
    };
    let (blo, bhi) = (Vec3::splat(-1.0), Vec3::splat(1.0));
    let crossings: Vec<EdgeCrossing> = extract_edge_crossings(&ball, blo, bhi, &cfg);
    let points: Vec<HermitePoint> = extract_hermite(&ball, blo, bhi, &cfg);
    let extractor = HermiteExtractor::new(&ball, blo, bhi, cfg);
    let worst = crossings
        .iter()
        .map(|e| (e.intersection.length() - 0.7).abs())
        .fold(0.0, f32::max);
    println!(
        "hermite: {} edge crossings, worst |x|−r = {worst:.2e}, first t = {:.3}",
        crossings.len(),
        crossings[0].t()
    );
    assert_eq!(points.len(), crossings.len());
    assert_eq!(extractor.extract_surface_points().len(), crossings.len());
    assert_eq!(extractor.extract_edge_crossings().len(), crossings.len());
    assert!(worst < 1e-3);
    let hp = HermitePoint::new(Vec3::X, Vec3::X);
    assert_eq!(hp.position, hp.normal);
}
