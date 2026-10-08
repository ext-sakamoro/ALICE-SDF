//! Oracles for mesh collision proxies, manifold validation / repair and
//! primitive fitting.
//!
//! Expected values are closed forms of the shapes built here (box volume,
//! sphere radius, Euler characteristic V − E + F, triangle area and aspect
//! ratio) or independent recomputations in f64, never outputs of the code under
//! test.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

use alice_sdf::eval::eval;
use alice_sdf::mesh::{
    compute_aabb, compute_bounding_sphere, compute_convex_hull, compute_quality,
    convex_decomposition, convex_hull_from_points, detect_primitive, fit_box, fit_cylinder,
    fit_plane, fit_sphere, primitives_to_csg, simplify_collision, validate_mesh, BoundingSphere,
    CollisionAabb, ConvexDecomposition, ConvexHull, FittedPrimitive, FittingConfig, FittingResult,
    Mesh, MeshRepair, Vertex, VhacdConfig,
};
use glam::{DVec3, Vec3};
use std::collections::{HashMap, HashSet};

// ───────────────────────────── fixtures ─────────────────────────────

/// Octahedron subdivided `levels` times on the sphere of radius `r`, outward
/// winding, welded. V = 4·4ⁿ + 2, E = 12·4ⁿ, F = 8·4ⁿ.
fn icosphere(levels: u32, r: f32, center: Vec3) -> Mesh {
    let mut dirs: Vec<DVec3> = vec![
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
        let mut midpoint = |a: usize, b: usize, dirs: &mut Vec<DVec3>| -> usize {
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
    Mesh {
        vertices: dirs
            .iter()
            .map(|d| Vertex::new(center + (*d * f64::from(r)).as_vec3(), d.as_vec3()))
            .collect(),
        indices: faces
            .iter()
            .flat_map(|f| f.iter().map(|&i| i as u32))
            .collect(),
    }
}

/// Axis-aligned box mesh, 8 welded corners, 12 outward triangles.
fn box_mesh(center: Vec3, half: Vec3) -> Mesh {
    let c = |x: f32, y: f32, z: f32| center + half * Vec3::new(x, y, z);
    let p = [
        c(-1., -1., -1.),
        c(1., -1., -1.),
        c(1., 1., -1.),
        c(-1., 1., -1.),
        c(-1., -1., 1.),
        c(1., -1., 1.),
        c(1., 1., 1.),
        c(-1., 1., 1.),
    ];
    let idx: Vec<u32> = vec![
        0, 2, 1, 0, 3, 2, // -z
        4, 5, 6, 4, 6, 7, // +z
        0, 1, 5, 0, 5, 4, // -y
        3, 7, 6, 3, 6, 2, // +y
        0, 4, 7, 0, 7, 3, // -x
        1, 2, 6, 1, 6, 5, // +x
    ];
    Mesh {
        vertices: p
            .iter()
            .map(|&q| Vertex::new(q, (q - center).normalize()))
            .collect(),
        indices: idx,
    }
}

/// V − E + F over the vertices that triangles actually reference.
fn euler_characteristic(indices: &[u32]) -> i64 {
    let mut verts = HashSet::new();
    let mut edges = HashSet::new();
    for t in indices.chunks(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            verts.insert(a);
            edges.insert((a.min(b), a.max(b)));
        }
    }
    verts.len() as i64 - edges.len() as i64 + (indices.len() / 3) as i64
}

/// Every directed edge appears exactly once and its reverse exactly once:
/// closed, manifold and consistently wound.
fn consistently_wound_closed(indices: &[u32]) -> bool {
    let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
    for t in indices.chunks(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *directed.entry((a, b)).or_insert(0) += 1;
        }
    }
    directed
        .iter()
        .all(|(&(a, b), &n)| n == 1 && directed.get(&(b, a)) == Some(&1))
}

/// Signed volume (f64) by the divergence theorem: Σ v0·(v1×v2) / 6.
fn signed_volume(positions: &[Vec3], indices: &[u32]) -> f64 {
    indices
        .chunks(3)
        .map(|t| {
            let a = positions[t[0] as usize].as_dvec3();
            let b = positions[t[1] as usize].as_dvec3();
            let c = positions[t[2] as usize].as_dvec3();
            a.dot(b.cross(c))
        })
        .sum::<f64>()
        / 6.0
}

fn positions(mesh: &Mesh) -> Vec<Vec3> {
    mesh.vertices.iter().map(|v| v.position).collect()
}

// ───────────────────────────── collision ─────────────────────────────

#[test]
fn aabb_and_bounding_sphere_match_closed_form() {
    let c = Vec3::new(0.5, -1.0, 2.0);
    let half = Vec3::new(1.0, 2.0, 3.0);
    let bx = box_mesh(c, half);
    let aabb: CollisionAabb = compute_aabb(&bx);
    assert_eq!((aabb.min, aabb.max), (c - half, c + half));
    assert_eq!(aabb.center(), c);
    assert_eq!(aabb.half_extents(), half);
    assert_eq!(aabb.volume(), 48.0);
    assert!(aabb.contains(c) && aabb.contains(c + half));
    assert!(!aabb.contains(c + half + Vec3::X * 1e-3));

    // Ritter on a box: corner → opposite corner → back, so the initial sphere
    // is already the circumsphere (centre c, radius |half|) and nothing grows it
    let s: BoundingSphere = compute_bounding_sphere(&bx);
    assert!((s.center - c).length() < 1e-6);
    assert!((s.radius - half.length()).abs() < 1e-6);
    assert!(bx.vertices.iter().all(|v| s.contains(v.position)));
    assert!(!s.contains(c + Vec3::splat(half.length())));

    // a sphere mesh: antipodal vertices give the exact sphere
    let sp = icosphere(3, 1.25, c);
    let s = compute_bounding_sphere(&sp);
    assert!((s.center - c).length() < 1e-5, "centre {}", s.center);
    assert!((s.radius - 1.25).abs() < 1e-5, "radius {}", s.radius);
    let mut compared = 0;
    for v in &sp.vertices {
        assert!((v.position - s.center).length() <= s.radius * (1.0 + 1e-6));
        compared += 1;
    }
    assert_eq!(compared, sp.vertices.len());

    let empty = Mesh {
        vertices: vec![],
        indices: vec![],
    };
    assert_eq!(compute_aabb(&empty).volume(), 0.0);
    assert_eq!(compute_bounding_sphere(&empty).radius, 0.0);
}

/// Checks the hull of `points` against the definition of a convex hull.
fn assert_is_convex_hull(hull: &ConvexHull, points: &[Vec3], want_volume: f64) {
    let ref_verts: HashSet<u32> = hull.indices.iter().copied().collect();
    assert!(consistently_wound_closed(&hull.indices), "closed + wound");
    assert_eq!(euler_characteristic(&hull.indices), 2);
    let vol = signed_volume(&hull.vertices, &hull.indices);
    assert!(
        (vol - want_volume).abs() < 1e-4 * want_volume,
        "volume {vol}, want {want_volume}"
    );
    // every input point is on the inner side of every face plane
    let mut compared = 0;
    for t in hull.indices.chunks(3) {
        let a = hull.vertices[t[0] as usize].as_dvec3();
        let b = hull.vertices[t[1] as usize].as_dvec3();
        let c = hull.vertices[t[2] as usize].as_dvec3();
        let n = (b - a).cross(c - a).normalize();
        for p in points {
            assert!(n.dot(p.as_dvec3() - a) < 1e-5, "point {p} outside a face");
            compared += 1;
        }
    }
    assert!(compared > 0);
    assert!(ref_verts.len() >= 4);
}

#[test]
fn convex_hull_of_box_corners_and_interior_points_is_the_box() {
    let half = Vec3::new(1.0, 2.0, 3.0);
    let bx = box_mesh(Vec3::ZERO, half);
    let mut pts = positions(&bx);
    // interior points and face centres must not change the hull
    for i in 0..40 {
        let f = i as f32 / 40.0;
        pts.push(half * Vec3::new(f - 0.5, 0.3 - f * 0.5, (f * 7.0).sin() * 0.9));
    }
    pts.push(Vec3::new(0.0, 0.0, half.z));
    pts.push(Vec3::new(half.x, 0.0, 0.0));
    let hull = convex_hull_from_points(&pts);
    assert_is_convex_hull(&hull, &pts, 48.0);
    // the hull's surface uses exactly the 8 corners
    let used: HashSet<[u32; 3]> = hull
        .indices
        .iter()
        .map(|&i| {
            let v = hull.vertices[i as usize];
            [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
        })
        .collect();
    assert_eq!(used.len(), 8);

    // from a mesh: an inscribed polyhedron is its own hull
    let sp = icosphere(2, 1.0, Vec3::ZERO);
    let want = signed_volume(&positions(&sp), &sp.indices);
    let hull = compute_convex_hull(&sp);
    assert_is_convex_hull(&hull, &positions(&sp), want);

    // fewer than four points come back as they are
    let tri = convex_hull_from_points(&[Vec3::ZERO, Vec3::X, Vec3::Y]);
    assert_eq!((tri.vertices.len(), tri.indices), (3, vec![0, 1, 2]));
    assert!(convex_hull_from_points(&[Vec3::ZERO]).indices.is_empty());
}

#[test]
fn simplify_collision_clusters_into_grid_cells() {
    let sp = icosphere(3, 1.0, Vec3::ZERO);
    let res = 4;
    let out = simplify_collision(&sp, res);
    assert!(out.vertices.len() <= (res * res * res) as usize);
    assert!(out.vertices.len() > 8);
    let aabb = compute_aabb(&sp);
    let mut compared = 0;
    for t in out.indices.chunks(3) {
        assert!(
            t[0] != t[1] && t[1] != t[2] && t[0] != t[2],
            "no collapsed triangle"
        );
        for &i in t {
            let v = out.vertices[i as usize];
            assert!(v.cmpge(aabb.min - 1e-6).all() && v.cmple(aabb.max + 1e-6).all());
            compared += 1;
        }
    }
    assert!(compared > 0);

    // a grid finer than the vertex spacing keeps every vertex and triangle
    let bx = box_mesh(Vec3::ZERO, Vec3::ONE);
    let fine = simplify_collision(&bx, 64);
    assert_eq!(fine.vertices.len(), 8);
    assert_eq!(fine.indices.len(), bx.indices.len());
    assert!((signed_volume(&fine.vertices, &fine.indices) - 8.0).abs() < 1e-5);
}

#[test]
fn convex_decomposition_separates_disjoint_parts() {
    // two unit boxes separated along y and z (each single axis:
    // tests/test_mesh_fit_hull_oracle.rs)
    let a = box_mesh(Vec3::new(0.0, -3.0, -3.0), Vec3::ONE);
    let b = box_mesh(Vec3::new(0.0, 3.0, 3.0), Vec3::ONE);
    let mut mesh = a;
    let off = mesh.vertices.len() as u32;
    mesh.vertices.extend_from_slice(&b.vertices);
    mesh.indices.extend(b.indices.iter().map(|i| i + off));

    let cfg = VhacdConfig::default();
    let dec: ConvexDecomposition = convex_decomposition(&mesh, &cfg);
    assert_eq!(dec.parts.len(), 2);
    let mut n_verts = 0;
    let mut n_tris = 0;
    for part in &dec.parts {
        assert!(part.vertices.len() <= cfg.max_vertices_per_hull as usize);
        let c = part.vertices.iter().copied().sum::<Vec3>() / part.vertices.len() as f32;
        // each part sits inside one of the boxes (voxel centres, so a cell in)
        let near_a = (c - Vec3::new(0.0, -3.0, -3.0)).abs().max_element() < 1.0;
        let near_b = (c - Vec3::new(0.0, 3.0, 3.0)).abs().max_element() < 1.0;
        assert!(near_a ^ near_b, "part centroid {c}");
        n_verts += part.vertices.len();
        n_tris += part.indices.len() / 3;
    }
    assert_eq!(dec.total_vertices(), n_verts);
    assert_eq!(dec.total_triangles(), n_tris);

    // a cap of one hull folds every component into it
    let one = VhacdConfig {
        max_hulls: 1,
        ..VhacdConfig::fast()
    };
    assert_eq!(convex_decomposition(&mesh, &one).parts.len(), 1);
    let hq = VhacdConfig::high_quality();
    assert!(hq.resolution > cfg.resolution && hq.max_hulls > cfg.max_hulls);
    let fast = VhacdConfig::fast();
    assert!(fast.resolution < cfg.resolution && fast.max_hulls < cfg.max_hulls);
}

// ───────────────────────────── manifold ─────────────────────────────

#[test]
fn closed_sphere_mesh_validates_with_euler_characteristic_two() {
    for levels in 0..4 {
        let sp = icosphere(levels, 1.0, Vec3::ZERO);
        let n = 4u32.pow(levels) as usize;
        let v = validate_mesh(&sp);
        assert_eq!(v.vertex_count, 4 * n + 2);
        assert_eq!(v.triangle_count, 8 * n);
        assert!(v.is_manifold && v.is_clean());
        assert_eq!(
            (
                v.boundary_edges,
                v.non_manifold_edges,
                v.degenerate_triangles
            ),
            (0, 0, 0)
        );
        assert_eq!((v.duplicate_vertices, v.inconsistent_normals), (0, 0));
        assert_eq!(euler_characteristic(&sp.indices), 2);
    }
}

#[test]
fn punching_and_filling_a_hole_moves_euler_characteristic() {
    let sp = icosphere(2, 1.0, Vec3::ZERO);
    let mut holed = sp.clone();
    holed.indices.drain(0..3);
    let v = validate_mesh(&holed);
    assert_eq!(v.boundary_edges, 3);
    assert!(!v.is_manifold && !v.is_clean());
    assert_eq!(euler_characteristic(&holed.indices), 1);

    let filled = MeshRepair::fill_holes(&holed);
    assert_eq!(filled.vertices.len(), sp.vertices.len() + 1);
    assert_eq!(filled.indices.len(), holed.indices.len() + 9);
    let v = validate_mesh(&filled);
    assert!(v.is_manifold);
    assert_eq!(euler_characteristic(&filled.indices), 2);
    assert!(consistently_wound_closed(&filled.indices));
    // the fan centre is the centroid of the three ring vertices
    let t = &sp.indices[0..3];
    let want = (sp.vertices[t[0] as usize].position
        + sp.vertices[t[1] as usize].position
        + sp.vertices[t[2] as usize].position)
        / 3.0;
    assert!((filled.vertices.last().expect("centroid").position - want).length() < 1e-6);
    // nothing to fill on a closed mesh
    assert_eq!(MeshRepair::fill_holes(&sp).indices, sp.indices);
}

#[test]
fn orient_faces_restores_outward_winding() {
    let sp = icosphere(2, 1.0, Vec3::ZERO);
    let want = signed_volume(&positions(&sp), &sp.indices);
    assert!(want > 0.0);
    let mut scrambled = sp;
    for (t, tri) in scrambled.indices.chunks_mut(3).enumerate() {
        if t % 3 != 1 {
            tri.swap(1, 2);
        }
    }
    assert!(!consistently_wound_closed(&scrambled.indices));
    let fixed = MeshRepair::orient_faces(&scrambled);
    assert!(consistently_wound_closed(&fixed.indices));
    assert!((signed_volume(&positions(&fixed), &fixed.indices) - want).abs() < 1e-9);

    // vertex normals are outward, so fix_normals reaches the same winding
    let by_normals = MeshRepair::fix_normals(&scrambled);
    assert!(consistently_wound_closed(&by_normals.indices));
    assert_eq!(validate_mesh(&by_normals).inconsistent_normals, 0);
    assert!(validate_mesh(&scrambled).inconsistent_normals > 0);
}

#[test]
fn repair_steps_remove_exactly_what_they_name() {
    let sp = icosphere(2, 1.0, Vec3::ZERO);
    let n_tris = sp.indices.len() / 3;

    // unweld: every triangle gets its own three vertices
    let mut soup = Mesh {
        vertices: vec![],
        indices: vec![],
    };
    for &i in &sp.indices {
        soup.indices.push(soup.vertices.len() as u32);
        soup.vertices.push(sp.vertices[i as usize]);
    }
    assert_eq!(validate_mesh(&soup).boundary_edges, 3 * n_tris);
    let welded = MeshRepair::merge_duplicate_vertices(&soup, 1e-5);
    assert_eq!(welded.vertices.len(), sp.vertices.len());
    assert!(validate_mesh(&welded).is_clean());
    assert_eq!(euler_characteristic(&welded.indices), 2);

    // a zero-area triangle and a duplicate face
    let mut dirty = sp.clone();
    dirty.indices.extend_from_slice(&[0, 0, 1]);
    dirty.indices.extend_from_slice(&sp.indices[3..6]);
    let v = validate_mesh(&dirty);
    assert_eq!(v.degenerate_triangles, 1);
    assert!(v.non_manifold_edges > 0);
    let no_degen = MeshRepair::remove_degenerate_triangles(&dirty);
    assert_eq!(no_degen.indices.len() / 3, n_tris + 1);
    let no_dup = MeshRepair::remove_duplicate_triangles(&no_degen);
    assert_eq!(no_dup.indices, sp.indices);

    // repair_all on the dirty soup gives back the clean welded sphere
    let mut dirty_soup = soup.clone();
    dirty_soup.indices.extend_from_slice(&[0, 0, 1]);
    let repaired = MeshRepair::repair_all(&dirty_soup, 1e-5);
    let v = validate_mesh(&repaired);
    assert!(v.is_clean(), "{v}");
    assert_eq!(repaired.indices.len() / 3, n_tris);

    // a far-away tetrahedron speck goes, the sphere stays
    let mut specked = sp.clone();
    let tet = box_mesh(Vec3::splat(10.0), Vec3::splat(0.01));
    let off = specked.vertices.len() as u32;
    specked.vertices.extend_from_slice(&tet.vertices);
    specked.indices.extend(tet.indices.iter().map(|i| i + off));
    let dropped = MeshRepair::drop_specks(&specked, 0.5);
    assert_eq!(dropped.indices, sp.indices);
    assert_eq!(
        MeshRepair::drop_specks(&specked, 0.0).indices,
        specked.indices
    );
}

#[test]
fn mesh_quality_matches_triangle_closed_forms() {
    // one equilateral triangle of side 2 and one right isosceles of legs 1
    let s3 = 3.0_f32.sqrt();
    let mesh = Mesh {
        vertices: [
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, s3, 0.0),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(1.0, 0.0, 5.0),
            Vec3::new(0.0, 1.0, 5.0),
        ]
        .iter()
        .map(|&p| Vertex::new(p, Vec3::Z))
        .collect(),
        indices: vec![0, 1, 2, 3, 4, 5],
    };
    let q = compute_quality(&mesh);
    let eq_area = s3; // √3/4 · 2²
    let rt_area = 0.5;
    assert!((q.max_area - eq_area).abs() < 1e-6);
    assert!((q.min_area - rt_area).abs() < 1e-6);
    assert!((q.total_area - (eq_area + rt_area)).abs() < 1e-6);
    assert!((q.avg_area - f32::midpoint(eq_area, rt_area)).abs() < 1e-6);
    // normalised so that the equilateral triangle reads 1:
    // 12·√3·A / P² — equilateral: 12·√3·(√3/4·s²) / (3s)² = 1
    let rt_p = 2.0 + 2.0_f64.sqrt();
    let rt_aspect = 12.0 * 3.0_f64.sqrt() * 0.5 / (rt_p * rt_p);
    assert!(
        (f64::from(q.min_aspect_ratio) - rt_aspect).abs() < 1e-6,
        "min aspect {} want {rt_aspect}",
        q.min_aspect_ratio
    );
    assert!(
        (f64::from(q.avg_aspect_ratio) - f64::midpoint(1.0, rt_aspect)).abs() < 1e-6,
        "avg aspect {}",
        q.avg_aspect_ratio
    );
    let empty = compute_quality(&Mesh {
        vertices: vec![],
        indices: vec![],
    });
    assert_eq!(empty.total_area, 0.0);
}

// ───────────────────────────── primitive fitting ─────────────────────────────

fn sphere_points(center: Vec3, r: f32, n_theta: usize, n_phi: usize, max_theta: f64) -> Vec<Vec3> {
    let mut pts = vec![];
    for i in 0..=n_theta {
        let th = max_theta * i as f64 / n_theta as f64;
        for j in 0..n_phi {
            let ph = std::f64::consts::TAU * j as f64 / n_phi as f64;
            let d = DVec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
            pts.push(center + (d * f64::from(r)).as_vec3());
        }
    }
    pts
}

fn box_surface_points(center: Vec3, half: Vec3, n: usize) -> Vec<Vec3> {
    let mut pts = vec![];
    let g = |i: usize| -1.0 + 2.0 * i as f32 / n as f32;
    for axis in 0..3 {
        for sign in [-1.0_f32, 1.0] {
            for i in 0..=n {
                for j in 0..=n {
                    let mut u = [0.0_f32; 3];
                    u[axis] = sign;
                    u[(axis + 1) % 3] = g(i);
                    u[(axis + 2) % 3] = g(j);
                    pts.push(center + half * Vec3::from(u));
                }
            }
        }
    }
    pts
}

#[test]
fn fitting_recovers_sphere_box_cylinder_and_plane() {
    let cfg = FittingConfig::default();

    let c = Vec3::new(0.3, -0.2, 0.5);
    let pts = sphere_points(c, 1.7, 24, 48, std::f64::consts::PI);
    let r: FittingResult = fit_sphere(&pts, &cfg).expect("sphere");
    match r.primitive {
        FittedPrimitive::Sphere { center, radius } => {
            assert!((center - c).length() < 1e-4, "centre {center}");
            assert!((radius - 1.7).abs() < 1e-4, "radius {radius}");
        }
        ref p => panic!("not a sphere: {p:?}"),
    }
    assert!(r.mse < 1e-8 && r.max_error < 1e-3);
    assert_eq!(r.inlier_count, pts.len());
    assert!(r.is_acceptable(cfg.max_mse, cfg.min_inlier_ratio, pts.len()));
    assert!(fit_sphere(&pts[..3], &cfg).is_none());

    let bc = Vec3::new(-1.0, 0.5, 2.0);
    let half = Vec3::new(0.5, 1.5, 0.8);
    let bpts = box_surface_points(bc, half, 6);
    let r = fit_box(&bpts, &cfg).expect("box");
    match r.primitive {
        FittedPrimitive::Box {
            center,
            half_extents,
        } => {
            assert!((center - bc).length() < 1e-6);
            assert!((half_extents - half).length() < 1e-6);
        }
        ref p => panic!("not a box: {p:?}"),
    }
    assert!(r.max_error < 1e-6);

    // cylinder along z: lateral surface plus both caps
    let (cc, cr, ch) = (Vec3::new(0.2, 0.4, -0.3), 0.6_f32, 0.9_f32);
    let mut cpts = vec![];
    for k in 0..=8 {
        let z = -ch + 2.0 * ch * k as f32 / 8.0;
        for j in 0..32 {
            let a = std::f32::consts::TAU * j as f32 / 32.0;
            cpts.push(cc + Vec3::new(cr * a.cos(), cr * a.sin(), z));
        }
    }
    let r = fit_cylinder(&cpts, &cfg).expect("cylinder");
    match r.primitive {
        FittedPrimitive::Cylinder {
            center,
            axis,
            radius,
            half_height,
        } => {
            assert_eq!(axis, Vec3::Z);
            assert!((center - cc).length() < 1e-5, "centre {center}");
            assert!((radius - cr).abs() < 1e-5);
            assert!((half_height - ch).abs() < 1e-5);
        }
        ref p => panic!("not a cylinder: {p:?}"),
    }

    let n = Vec3::new(1.0, 2.0, 2.0) / 3.0;
    let (u, v) = (
        Vec3::new(2.0, -1.0, 0.0) / 5.0_f32.sqrt(),
        n.cross(Vec3::new(2.0, -1.0, 0.0) / 5.0_f32.sqrt()),
    );
    let mut ppts = vec![];
    for i in 0..10 {
        for j in 0..10 {
            ppts.push(n * 0.7 + u * (i as f32 * 0.3 - 1.0) + v * (j as f32 * 0.2 - 0.7));
        }
    }
    let r = fit_plane(&ppts, &cfg).expect("plane");
    match r.primitive {
        FittedPrimitive::Plane { normal, distance } => {
            let s = normal.dot(n).signum();
            assert!((normal * s - n).length() < 1e-4, "normal {normal}");
            assert!((distance * s - 0.7).abs() < 1e-4, "distance {distance}");
        }
        ref p => panic!("not a plane: {p:?}"),
    }

    // an axis-aligned plane: the covariance is exactly singular
    let flat: Vec<Vec3> = (0..25)
        .map(|i| Vec3::new((i % 5) as f32 - 2.0, 2.0, (i / 5) as f32 * 0.5))
        .collect();
    let r = fit_plane(&flat, &cfg).expect("plane");
    match r.primitive {
        FittedPrimitive::Plane { normal, distance } => {
            let s = normal.y.signum();
            assert!((normal * s - Vec3::Y).length() < 1e-6, "normal {normal}");
            assert!((distance * s - 2.0).abs() < 1e-5, "distance {distance}");
        }
        ref p => panic!("not a plane: {p:?}"),
    }
    assert!(r.max_error < 1e-5);

    // detection picks the generating family
    let kind = |pts: &[Vec3]| detect_primitive(pts, &cfg).expect("fit").primitive.kind();
    use alice_sdf::mesh::primitive_fitting::FittedPrimitiveKind;
    assert_eq!(kind(&pts), FittedPrimitiveKind::Sphere);
    assert_eq!(kind(&bpts), FittedPrimitiveKind::Box);
    assert_eq!(kind(&ppts), FittedPrimitiveKind::Plane);
}

#[test]
fn fitted_primitive_distance_and_sdf_node_agree_with_closed_forms() {
    let prims = [
        FittedPrimitive::Sphere {
            center: Vec3::new(0.1, 0.2, 0.3),
            radius: 0.8,
        },
        FittedPrimitive::Box {
            center: Vec3::new(-0.5, 0.0, 0.25),
            half_extents: Vec3::new(0.4, 0.7, 0.3),
        },
        FittedPrimitive::Cylinder {
            center: Vec3::new(0.0, 0.3, 0.0),
            axis: Vec3::Y,
            radius: 0.5,
            half_height: 0.9,
        },
        FittedPrimitive::Cylinder {
            center: Vec3::new(0.2, 0.0, -0.1),
            axis: Vec3::X,
            radius: 0.35,
            half_height: 0.6,
        },
        FittedPrimitive::Capsule {
            point_a: Vec3::new(-0.5, 0.0, 0.0),
            point_b: Vec3::new(0.5, 0.2, 0.0),
            radius: 0.25,
        },
        FittedPrimitive::Plane {
            normal: Vec3::Y,
            distance: 0.4,
        },
    ];
    // independent closed forms
    let closed = |p: &FittedPrimitive, q: Vec3| -> f32 {
        match *p {
            FittedPrimitive::Sphere { center, radius } => (q - center).length() - radius,
            FittedPrimitive::Box {
                center,
                half_extents,
            } => {
                let d = (q - center).abs() - half_extents;
                d.max(Vec3::ZERO).length() + d.max_element().min(0.0)
            }
            FittedPrimitive::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => {
                let d = q - center;
                let h = d.dot(axis);
                let rr = (d - axis * h).length();
                let w = glam::Vec2::new(rr - radius, h.abs() - half_height);
                w.max(glam::Vec2::ZERO).length() + w.max_element().min(0.0)
            }
            FittedPrimitive::Capsule {
                point_a,
                point_b,
                radius,
            } => {
                let ab = point_b - point_a;
                let t = ((q - point_a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                (q - point_a - ab * t).length() - radius
            }
            FittedPrimitive::Plane { normal, distance } => q.dot(normal) - distance,
        }
    };
    let qs: Vec<Vec3> = (0..200)
        .map(|i| {
            let f = i as f32;
            Vec3::new((f * 0.37).sin(), (f * 0.61).cos(), (f * 0.23).sin() * 1.2) * 1.4
        })
        .collect();
    let mut compared = 0;
    for p in &prims {
        let node = p.to_sdf_node();
        let mut err = 0.0;
        for &q in &qs {
            let want = closed(p, q);
            assert!((p.distance(q) - want).abs() < 1e-5, "{p:?} distance at {q}");
            assert!(
                (eval(&node, q) - want).abs() < 1e-5,
                "{p:?} to_sdf_node at {q}: {} vs {want}",
                eval(&node, q)
            );
            err += want * want;
            compared += 1;
        }
        assert!((p.compute_error(&qs) - err).abs() <= 1e-4 * err.max(1.0));
    }
    assert_eq!(compared, prims.len() * qs.len());

    // union of the parts is the minimum of the parts
    let csg = primitives_to_csg(&prims[..3]).expect("csg");
    for &q in &qs {
        let want = prims[..3]
            .iter()
            .map(|p| closed(p, q))
            .fold(f32::INFINITY, f32::min);
        assert!((eval(&csg, q) - want).abs() < 1e-5);
    }
    assert!(primitives_to_csg(&[]).is_none());
}
