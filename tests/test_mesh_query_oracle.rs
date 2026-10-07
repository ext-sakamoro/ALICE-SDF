//! Oracles for the mesh query modules: BVH, mesh → SDF, exterior sign field,
//! point cloud SDF and Hermite edge data.
//!
//! Every expected value comes from a closed form or from an independent f64
//! implementation written here, never from the code under test:
//!
//! - **BVH**: point–triangle distance by plane projection + three segments (a
//!   different construction from the Voronoi-region code in `bvh.rs`), taken as
//!   a brute-force minimum over all triangles.
//! - **mesh → SDF**: a sphere of radius `r` approximated by an inscribed
//!   polyhedron whose vertices lie on the sphere. Every triangle lies between
//!   its circumcircle plane and the sphere, so the Hausdorff distance between
//!   mesh and sphere is at most `h = max_t (r − sqrt(r² − R_t²))` (`R_t` is the
//!   triangle's circumradius), and a distance function to a set moves by at most
//!   the Hausdorff distance: `|d_mesh(p) − (|p| − r)| ≤ h`.
//! - **point cloud**: nearest sample on a latitude/longitude grid. Any point of
//!   the sphere is within `r·(Δθ + Δφ)/2` (arc, so also chord) of a grid sample,
//!   so `||p| − r| ≤ |d| ≤ ||p| − r| + r·(Δθ + Δφ)/2`.
//! - **Hermite**: a plane `z = c` (linear field, so the crossing is exact) and
//!   a sphere (crossing on `|x| = r`, normal `x/|x|`).
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::mesh::mesh_to_sdf_exact;
use alice_sdf::mesh::{
    extract_edge_crossings, extract_hermite, Aabb, BvhNode, BvhTriangle, ExteriorField,
    HermiteConfig, HermiteExtractor, HermitePoint, MeshBvh, MeshSdf, MeshSignMode, MeshToSdfConfig,
    MeshToSdfStrategy, PointCloudSdf, PointCloudSdfConfig,
};
use glam::{DVec3, Vec3};
use std::collections::HashMap;

// ───────────────────────────── fixtures ─────────────────────────────

/// Octahedron subdivided `levels` times, vertices pushed onto the sphere of
/// radius `r` around `center`, outward winding, shared (welded) vertices.
/// V = 4·4ⁿ + 2, E = 12·4ⁿ, F = 8·4ⁿ.
fn icosphere(levels: u32, r: f32, center: Vec3) -> (Vec<Vec3>, Vec<u32>) {
    let mut dirs: Vec<DVec3> = vec![
        DVec3::X,
        DVec3::NEG_X,
        DVec3::Y,
        DVec3::NEG_Y,
        DVec3::Z,
        DVec3::NEG_Z,
    ];
    // (+x,+y,+z) octant first; each face listed counter-clockwise from outside
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
        let mut midpoint = |a: usize, b: usize, dirs: &mut Vec<DVec3>| -> usize {
            let key = (a.min(b), a.max(b));
            *mid.entry(key).or_insert_with(|| {
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
    let verts = dirs
        .iter()
        .map(|d| center + (*d * f64::from(r)).as_vec3())
        .collect();
    let idx = faces
        .iter()
        .flat_map(|f| f.iter().map(|&i| i as u32))
        .collect();
    (verts, idx)
}

/// Hausdorff bound between the inscribed polyhedron and its sphere (see the
/// module docs), in f64 from the vertex positions.
fn inscribed_hausdorff(verts: &[Vec3], idx: &[u32], r: f64) -> f64 {
    idx.chunks(3)
        .map(|t| {
            let a = verts[t[0] as usize].as_dvec3();
            let b = verts[t[1] as usize].as_dvec3();
            let c = verts[t[2] as usize].as_dvec3();
            let (la, lb, lc) = ((b - c).length(), (c - a).length(), (a - b).length());
            let area2 = (b - a).cross(c - a).length(); // 2·area
            let circ = la * lb * lc / (2.0 * area2); // R = abc / (4·area)
            r - (r * r - circ * circ).max(0.0).sqrt()
        })
        .fold(0.0, f64::max)
}

/// Deterministic points in `[-ext, ext]³` (64-bit LCG, high bits).
fn lcg_points(n: usize, ext: f32, seed: u64) -> Vec<Vec3> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32 / (1u64 << 24) as f32).mul_add(2.0, -1.0) * ext
    };
    (0..n).map(|_| Vec3::new(next(), next(), next())).collect()
}

/// Closest point on segment `a..b` (f64).
fn seg_closest(p: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    let ab = b - a;
    let t = (ab.dot(p - a) / ab.length_squared()).clamp(0.0, 1.0);
    a + ab * t
}

/// Independent point–triangle distance: project onto the plane; if the
/// projection is inside (all three edge half-planes), that is the answer,
/// otherwise the best of the three segments.
fn tri_dist(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let n = (b - a).cross(c - a);
    let q = p - n * (n.dot(p - a) / n.length_squared());
    let inside = (b - a).cross(q - a).dot(n) >= 0.0
        && (c - b).cross(q - b).dot(n) >= 0.0
        && (a - c).cross(q - c).dot(n) >= 0.0;
    if inside {
        return (p - q).length();
    }
    [
        seg_closest(p, a, b),
        seg_closest(p, b, c),
        seg_closest(p, c, a),
    ]
    .iter()
    .map(|&s| (p - s).length())
    .fold(f64::INFINITY, f64::min)
}

fn brute_udf(verts: &[Vec3], idx: &[u32], p: Vec3) -> f64 {
    let p = p.as_dvec3();
    idx.chunks(3)
        .map(|t| {
            tri_dist(
                p,
                verts[t[0] as usize].as_dvec3(),
                verts[t[1] as usize].as_dvec3(),
                verts[t[2] as usize].as_dvec3(),
            )
        })
        .fold(f64::INFINITY, f64::min)
}

/// Strictly inside a closed convex outward-wound mesh: below every face plane.
fn inside_convex(verts: &[Vec3], idx: &[u32], p: Vec3) -> bool {
    let p = p.as_dvec3();
    idx.chunks(3).all(|t| {
        let a = verts[t[0] as usize].as_dvec3();
        let b = verts[t[1] as usize].as_dvec3();
        let c = verts[t[2] as usize].as_dvec3();
        (b - a).cross(c - a).dot(p - a) < 0.0
    })
}

// ───────────────────────────── BVH ─────────────────────────────

#[test]
fn triangle_queries_match_voronoi_closed_form() {
    let t = BvhTriangle::new(Vec3::ZERO, Vec3::X, Vec3::Y);
    assert_eq!(t.normal, Vec3::Z);
    assert_eq!(t.aabb.min, Vec3::ZERO);
    assert_eq!(t.aabb.max, Vec3::new(1.0, 1.0, 0.0));

    // (query, closest point) — one case per Voronoi region
    let cases = [
        (Vec3::new(-1.0, -1.0, 0.5), Vec3::ZERO), // vertex A
        (Vec3::new(2.0, -0.5, 0.0), Vec3::X),     // vertex B
        (Vec3::new(-0.5, 2.0, 0.0), Vec3::Y),     // vertex C
        (Vec3::new(0.5, -1.0, 0.0), Vec3::new(0.5, 0.0, 0.0)), // edge AB
        (Vec3::new(-1.0, 0.25, 0.0), Vec3::new(0.0, 0.25, 0.0)), // edge AC
        (Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.5, 0.5, 0.0)), // edge BC
        (Vec3::new(0.25, 0.25, 3.0), Vec3::new(0.25, 0.25, 0.0)), // face, above
        (Vec3::new(0.25, 0.5, -2.0), Vec3::new(0.25, 0.5, 0.0)), // face, below
    ];
    let mut compared = 0;
    for (p, want) in cases {
        let got = t.closest_point(p);
        assert!(
            (got - want).length() < 1e-6,
            "closest({p}) = {got}, want {want}"
        );
        let d = (p - want).length();
        assert!((t.unsigned_distance(p) - d).abs() < 1e-6);
        let sign = if p.z < 0.0 { -1.0 } else { 1.0 };
        assert!(
            (t.signed_distance(p) - sign * d).abs() < 1e-6,
            "signed({p})"
        );
        compared += 1;
    }
    assert_eq!(compared, cases.len());
}

#[test]
fn aabb_methods_match_closed_form() {
    let mut b = Aabb::empty();
    assert!(b.min.x.is_infinite() && b.max.x.is_infinite());
    b.expand_point(Vec3::new(1.0, -2.0, 0.5));
    b.expand_point(Vec3::new(-1.0, 2.0, 3.5));
    assert_eq!(b.min, Vec3::new(-1.0, -2.0, 0.5));
    assert_eq!(b.max, Vec3::new(1.0, 2.0, 3.5));
    let mut u = Aabb::new(Vec3::splat(10.0), Vec3::splat(11.0));
    u.expand_aabb(&b);
    assert_eq!(u.min, b.min);
    assert_eq!(u.max, Vec3::splat(11.0));

    // box 2 × 4 × 3
    assert_eq!(b.center(), Vec3::new(0.0, 0.0, 2.0));
    assert_eq!(b.surface_area(), 2.0 * (2.0 * 4.0 + 4.0 * 3.0 + 3.0 * 2.0));
    assert_eq!(b.longest_axis(), 1);
    assert_eq!(
        Aabb::new(Vec3::ZERO, Vec3::new(5.0, 1.0, 1.0)).longest_axis(),
        0
    );
    assert_eq!(
        Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 5.0)).longest_axis(),
        2
    );
    // outside along +x by 2, inside at the centre by the smallest half extent
    assert_eq!(b.signed_distance(Vec3::new(3.0, 0.0, 2.0)), 2.0);
    assert_eq!(b.signed_distance(b.center()), -1.0);
    // corner region: distance to the corner (1, 2, 3.5)
    let p = Vec3::new(4.0, 6.0, 3.5);
    assert!((b.signed_distance(p) - 5.0).abs() < 1e-6);
}

#[test]
fn bvh_matches_brute_force_on_sphere_mesh() {
    let (verts, idx) = icosphere(3, 1.0, Vec3::new(0.2, -0.1, 0.3));
    let bvh = MeshBvh::build(&verts, &idx, 4);
    assert_eq!(bvh.triangle_count(), 8 * 64);
    assert_eq!(bvh.triangles.len(), bvh.triangle_count());

    // bounds = componentwise min / max of the vertices, exactly
    let lo = verts.iter().fold(Vec3::INFINITY, |m, &v| m.min(v));
    let hi = verts.iter().fold(Vec3::NEG_INFINITY, |m, &v| m.max(v));
    let bounds = bvh.bounds().expect("non-empty mesh");
    assert_eq!((bounds.min, bounds.max), (lo, hi));

    // tree shape: children inside parents, every triangle in exactly one leaf
    let root = bvh.root.as_ref().expect("root");
    assert_eq!(root.aabb().min, lo);
    let mut seen = vec![0u32; bvh.triangle_count()];
    let mut stack: Vec<&BvhNode> = vec![root];
    while let Some(n) = stack.pop() {
        match n {
            BvhNode::Leaf { aabb, triangles } => {
                assert!(triangles.len() <= bvh.max_triangles_per_leaf);
                for &t in triangles {
                    seen[t] += 1;
                    let ta = bvh.triangles[t].aabb;
                    assert!(ta.min.cmpge(aabb.min).all() && ta.max.cmple(aabb.max).all());
                }
            }
            BvhNode::Internal { aabb, left, right } => {
                for c in [left.as_ref(), right.as_ref()] {
                    assert!(c.aabb().min.cmpge(aabb.min).all());
                    assert!(c.aabb().max.cmple(aabb.max).all());
                    stack.push(c);
                }
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "each triangle in exactly one leaf"
    );

    let pts = lcg_points(400, 2.0, 7);
    let mut compared = 0;
    for &p in &pts {
        let want = brute_udf(&verts, &idx, p);
        let got = bvh.unsigned_distance(p);
        assert!(
            (f64::from(got) - want).abs() <= 2e-6 * (1.0 + want),
            "udf({p}) = {got}, brute force {want}"
        );
        let q = bvh.closest_point(p).expect("closest");
        assert!((f64::from((p - q).length()) - want).abs() <= 2e-6 * (1.0 + want));
        assert!(
            brute_udf(&verts, &idx, q) < 1e-5,
            "closest point is on the mesh"
        );

        let s = bvh.signed_distance(p);
        assert!((f64::from(s.abs()) - want).abs() <= 2e-6 * (1.0 + want));
        if want > 1e-4 {
            assert_eq!(s < 0.0, inside_convex(&verts, &idx, p), "sign at {p}");
        }
        compared += 1;
    }
    assert_eq!(compared, pts.len());

    // batch = per-point, bit for bit
    let sb = bvh.signed_distance_batch(&pts);
    let ub = bvh.unsigned_distance_batch(&pts);
    for (i, &p) in pts.iter().enumerate() {
        assert_eq!(sb[i].to_bits(), bvh.signed_distance(p).to_bits());
        assert_eq!(ub[i].to_bits(), bvh.unsigned_distance(p).to_bits());
    }

    // empty mesh
    let empty = MeshBvh::build(&verts, &[], 4);
    assert!(empty.bounds().is_none() && empty.closest_point(Vec3::ZERO).is_none());
    assert_eq!(empty.unsigned_distance(Vec3::ZERO), f32::INFINITY);
}

// ───────────────────────────── mesh → SDF ─────────────────────────────

#[test]
fn mesh_sdf_of_inscribed_sphere_is_within_hausdorff_bound() {
    let r = 1.0_f32;
    let (verts, idx) = icosphere(4, r, Vec3::ZERO);
    let h = inscribed_hausdorff(&verts, &idx, f64::from(r));
    // 4 levels: circumradius ≈ 0.07, so h ≈ 2.5e-3 — small but not zero
    assert!(h > 1e-4 && h < 5e-3, "h = {h}");
    let tol = h + 1e-5;

    let configs = [
        ("accurate", MeshToSdfConfig::accurate()),
        ("topology_robust", MeshToSdfConfig::topology_robust()),
        ("hybrid", MeshToSdfConfig::hybrid()),
        ("default", MeshToSdfConfig::default()),
    ];
    assert_eq!(
        MeshToSdfConfig::accurate().strategy,
        MeshToSdfStrategy::BvhExact
    );
    assert_eq!(
        MeshToSdfConfig::hybrid().strategy,
        MeshToSdfStrategy::Hybrid
    );
    assert_eq!(
        MeshToSdfConfig::accurate().sign_mode,
        MeshSignMode::ExteriorFloodFill
    );

    let pts: Vec<Vec3> = lcg_points(300, 1.8, 11)
        .into_iter()
        .filter(|p| (p.length() - r).abs() > 0.02)
        .collect();
    assert!(pts.len() > 200);
    let mut compared = 0;
    for (name, cfg) in configs {
        let sdf = MeshSdf::new(&verts, &idx, &cfg).expect(name);
        assert_eq!(sdf.sign_mode(), cfg.sign_mode);
        assert_eq!(sdf.triangle_count(), idx.len() / 3);
        assert_eq!(sdf.bvh().triangle_count(), idx.len() / 3);
        let (lo, hi) = sdf.bounds();
        assert!((lo + Vec3::splat(r)).abs().max_element() < 1e-6);
        assert!((hi - Vec3::splat(r)).abs().max_element() < 1e-6);
        for &p in &pts {
            let want = f64::from(p.length() - r);
            let d = f64::from(sdf.eval(p));
            assert!(
                (d - want).abs() <= tol,
                "{name}: eval({p}) = {d}, |p|−r = {want}"
            );
            let u = f64::from(sdf.eval_unsigned(p));
            assert!((u - want.abs()).abs() <= tol, "{name}: eval_unsigned({p})");
            // central difference of a field within h of |p|−r points outward
            let g = sdf.gradient(p, 0.05);
            assert!(g.dot(p.normalize()) > 0.95, "{name}: gradient({p}) = {g}");
            compared += 1;
        }
        let batch = sdf.eval_batch(&pts);
        let ubatch = sdf.eval_unsigned_batch(&pts);
        for (i, &p) in pts.iter().enumerate() {
            assert_eq!(batch[i].to_bits(), sdf.eval(p).to_bits());
            assert_eq!(ubatch[i].to_bits(), sdf.eval_unsigned(p).to_bits());
        }
    }
    assert_eq!(compared, 4 * pts.len());

    // the free function is the same constructor
    let a = mesh_to_sdf_exact(&verts, &idx, &MeshToSdfConfig::accurate()).expect("some");
    let b = MeshSdf::try_new(&verts, &idx, &MeshToSdfConfig::accurate()).expect("ok");
    for &p in &pts {
        assert_eq!(a.eval(p).to_bits(), b.eval(p).to_bits());
    }
    assert!(mesh_to_sdf_exact(&verts, &[], &MeshToSdfConfig::accurate()).is_none());
    // over the 16.7M cell budget
    let mut huge = MeshToSdfConfig::accurate();
    huge.sign_flood_fill_resolution = 1_000;
    assert!(MeshSdf::try_new(&verts, &idx, &huge).is_err());
}

#[test]
fn mesh_sdf_capsule_tree_reads_minus_radius_at_vertices() {
    let (verts, idx) = icosphere(2, 1.0, Vec3::ZERO);
    let sdf = MeshSdf::new(&verts, &idx, &MeshToSdfConfig::accurate()).expect("sdf");
    // independent average over unique undirected edges
    let mut edges = std::collections::HashSet::new();
    for t in idx.chunks(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.insert((a.min(b), a.max(b)));
        }
    }
    assert_eq!(edges.len(), 12 * 16);
    let avg: f64 = edges
        .iter()
        .map(|&(a, b)| f64::from((verts[a as usize] - verts[b as usize]).length()))
        .sum::<f64>()
        / edges.len() as f64;
    let factor = 0.1_f32;
    let radius = avg * f64::from(factor);
    let node = sdf.to_sdf_node(factor);
    let mut compared = 0;
    for &v in &verts {
        // a vertex is on the axis of the capsules that meet there
        assert!((f64::from(eval(&node, v)) + radius).abs() < 1e-5);
        compared += 1;
    }
    assert_eq!(compared, verts.len());
    // edge midpoints are also on an axis; face centres are not (they are farther)
    let t = &idx[0..3];
    let c = (verts[t[0] as usize] + verts[t[1] as usize] + verts[t[2] as usize]) / 3.0;
    assert!(f64::from(eval(&node, c)) > -radius);
}

#[test]
fn exterior_field_grid_and_sign_match_closed_form() {
    let (verts, idx) = icosphere(3, 1.0, Vec3::ZERO);
    let bvh = MeshBvh::build(&verts, &idx, 4);
    let res = 32;
    let field = ExteriorField::build(&bvh, res).expect("field");

    let b = bvh.bounds().expect("bounds");
    let ext = b.max - b.min;
    let longest = ext.max_element();
    let cell = (f64::from(longest) / f64::from(res)) as f32;
    assert_eq!(field.cell_size(), cell);
    let pad = ExteriorField::PADDING_CELLS;
    assert_eq!(pad, 2);
    let want_dims =
        [ext.x, ext.y, ext.z].map(|e| (f64::from(e) / f64::from(cell)).ceil() as usize + 2 * pad);
    assert_eq!(field.dims(), want_dims);
    assert_eq!(ExteriorField::MAX_CELLS, 1 << 24);

    let pts = lcg_points(300, 1.6, 23);
    let mut compared = 0;
    for &p in &pts {
        let want = brute_udf(&verts, &idx, p);
        if want < 1e-3 {
            continue;
        }
        let inside = inside_convex(&verts, &idx, p);
        assert_eq!(field.is_exterior(&bvh, p), !inside, "is_exterior({p})");
        let s = f64::from(field.signed_distance(&bvh, p));
        assert!((s.abs() - want).abs() <= 2e-6 * (1.0 + want));
        assert_eq!(s < 0.0, inside);
        compared += 1;
    }
    assert!(compared > 250, "compared {compared}");
    // beyond the padded grid is outside by construction
    assert!(field.is_exterior(&bvh, Vec3::splat(100.0)));

    // a resolution over the cell budget is refused
    assert!(ExteriorField::build(&bvh, 1_000).is_err());
}

// ───────────────────────────── point cloud ─────────────────────────────

/// Latitude/longitude samples on the sphere of radius `r` with outward
/// normals, poles included once each.
fn lat_long_sphere(r: f32, n_theta: usize, n_phi: usize) -> (Vec<Vec3>, Vec<Vec3>, f64) {
    let mut pts = vec![Vec3::new(0.0, r, 0.0), Vec3::new(0.0, -r, 0.0)];
    let dt = std::f64::consts::PI / n_theta as f64;
    let dp = std::f64::consts::TAU / n_phi as f64;
    for i in 1..n_theta {
        let th = i as f64 * dt;
        for j in 0..n_phi {
            let ph = j as f64 * dp;
            let d = DVec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
            pts.push((d * f64::from(r)).as_vec3());
        }
    }
    let normals = pts.iter().map(|p| p.normalize()).collect();
    // any point of the sphere: ≤ Δθ/2 along the meridian to a sample latitude,
    // then ≤ sinθ·Δφ/2 ≤ Δφ/2 along it
    let cover = f64::from(r) * (dt + dp) / 2.0;
    (pts, normals, cover)
}

#[test]
fn point_cloud_sdf_of_sphere_samples_is_bracketed() {
    let r = 1.3_f32;
    let (pts, normals, cover) = lat_long_sphere(r, 40, 80);
    assert!(cover < 0.11, "cover = {cover}");

    let queries: Vec<Vec3> = lcg_points(400, 2.5, 5)
        .into_iter()
        .filter(|p| p.length() < 0.5 * r || p.length() > 1.5 * r)
        .collect();
    assert!(queries.len() > 100);
    let configs = [
        PointCloudSdfConfig::default(),
        PointCloudSdfConfig::fast(),
        PointCloudSdfConfig::accurate(),
    ];
    let mut compared = 0;
    for cfg in &configs {
        let sdf = PointCloudSdf::try_new(&pts, &normals, cfg).unwrap();
        assert_eq!(sdf.point_count(), pts.len());
        for &q in &queries {
            let d = sdf.eval(q);
            // exact nearest sample, brute force
            let nearest = pts
                .iter()
                .map(|&s| (q - s).length_squared())
                .fold(f32::INFINITY, f32::min)
                .sqrt();
            assert_eq!(
                d.abs().to_bits(),
                nearest.to_bits(),
                "|eval({q})| vs nearest"
            );
            let radial = f64::from(q.length() - r);
            assert!(f64::from(d.abs()) >= radial.abs() - 1e-5);
            assert!(f64::from(d.abs()) <= radial.abs() + cover + 1e-5);
            // well away from the surface the nearest sample's normal decides right
            assert_eq!(d < 0.0, radial < 0.0, "sign at {q}");
            compared += 1;
        }
        let batch = sdf.eval_batch(&queries);
        for (i, &q) in queries.iter().enumerate() {
            assert_eq!(batch[i].to_bits(), sdf.eval(q).to_bits());
        }
    }
    assert_eq!(compared, configs.len() * queries.len());

    // the deprecated panicking forms are the same field as `try_new`
    let t = PointCloudSdf::try_new(&pts, &normals, &PointCloudSdfConfig::default()).unwrap();
    #[allow(deprecated)]
    let f = alice_sdf::mesh::point_cloud_to_sdf(&pts, &normals, &PointCloudSdfConfig::default());
    #[allow(deprecated)]
    let g = PointCloudSdf::new(&pts, &normals, &PointCloudSdfConfig::default());
    for &q in &queries {
        assert_eq!(f.eval(q).to_bits(), t.eval(q).to_bits());
        assert_eq!(g.eval(q).to_bits(), t.eval(q).to_bits());
    }
}

// ───────────────────────────── Hermite ─────────────────────────────

#[test]
fn hermite_crossings_of_a_plane_are_exact_and_cover_the_whole_lattice() {
    let c = 0.3_f32;
    let res = 8;
    let cfg = HermiteConfig {
        resolution: res,
        ..HermiteConfig::default()
    };
    let (lo, hi) = (Vec3::splat(-1.0), Vec3::splat(1.0));
    let cell = (hi - lo) / res as f32;
    // a plane normal to each axis in turn: it crosses one edge along that axis
    // through every lattice vertex of the other two axes, including the
    // vertices on their max faces
    let mut compared = 0;
    for axis in 0..3 {
        let n = Vec3::AXES[axis];
        let plane = move |p: Vec3| p[axis] - c;
        let crossings = extract_edge_crossings(&plane, lo, hi, &cfg);
        assert_eq!(crossings.len(), (res + 1) * (res + 1), "axis {axis}");
        let mut seen = std::collections::HashSet::new();
        for e in &crossings {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            assert_eq!((e.start[u], e.start[v]), (e.end[u], e.end[v]));
            assert!(
                (e.intersection[axis] - c).abs() < 1e-6,
                "{:?}",
                e.intersection
            );
            assert_eq!(
                (e.intersection[u], e.intersection[v]),
                (e.start[u], e.start[v])
            );
            assert!((e.normal - n).length() < 1e-3, "normal {:?}", e.normal);
            // linear field: t is the exact parameter of the crossing
            let want_t = (c - e.start[axis]) / (e.end[axis] - e.start[axis]);
            assert!((e.t() - want_t).abs() < 1e-5);
            assert!((e.start_dist - plane(e.start)).abs() < 1e-7);
            assert!((e.end_dist - plane(e.end)).abs() < 1e-7);
            let iu = ((e.start[u] - lo[u]) / cell[u]).round() as i32;
            let iv = ((e.start[v] - lo[v]) / cell[v]).round() as i32;
            assert!(seen.insert((iu, iv)));
            compared += 1;
        }
        assert_eq!(seen.len(), (res + 1) * (res + 1));
    }
    assert_eq!(compared, 3 * (res + 1) * (res + 1));

    let plane = move |p: Vec3| p.z - c;
    let crossings = extract_edge_crossings(&plane, lo, hi, &cfg);
    let pts = extract_hermite(&plane, lo, hi, &cfg);
    let ex = HermiteExtractor::new(&plane, lo, hi, cfg);
    let pts2: Vec<HermitePoint> = ex.extract_surface_points();
    assert_eq!(pts.len(), crossings.len());
    assert_eq!(pts2.len(), crossings.len());
    assert_eq!(ex.extract_edge_crossings().len(), crossings.len());
    let hp = HermitePoint::new(Vec3::X, Vec3::Y);
    assert_eq!((hp.position, hp.normal), (Vec3::X, Vec3::Y));
}

#[test]
fn hermite_crossings_of_a_sphere_lie_on_it_with_radial_normals() {
    let r = 0.77_f32;
    let sphere = move |p: Vec3| p.length() - r;
    let res = 12;
    let cfg = HermiteConfig {
        resolution: res,
        refinement_iterations: 8,
        ..HermiteConfig::default()
    };
    let (lo, hi) = (Vec3::splat(-1.0), Vec3::splat(1.0));
    let cell = (hi - lo) / res as f32;
    let crossings = extract_edge_crossings(&sphere, lo, hi, &cfg);

    // independent count over every lattice edge
    let at = |x: usize, y: usize, z: usize| lo + Vec3::new(x as f32, y as f32, z as f32) * cell;
    let mut want = 0;
    for z in 0..=res {
        for y in 0..=res {
            for x in 0..=res {
                let a = sphere(at(x, y, z)) > 0.0;
                if x < res && a != (sphere(at(x + 1, y, z)) > 0.0) {
                    want += 1;
                }
                if y < res && a != (sphere(at(x, y + 1, z)) > 0.0) {
                    want += 1;
                }
                if z < res && a != (sphere(at(x, y, z + 1)) > 0.0) {
                    want += 1;
                }
            }
        }
    }
    assert!(want > 100);
    assert_eq!(crossings.len(), want);
    for e in &crossings {
        assert!(
            (e.intersection.length() - r).abs() < 1e-4,
            "{:?}",
            e.intersection
        );
        assert!(e.normal.dot(e.intersection.normalize()) > 0.9999);
        // the crossing lies on its edge
        let along = (e.intersection - e.start).length() + (e.end - e.intersection).length();
        assert!((along - (e.end - e.start).length()).abs() < 1e-5);
    }
}
