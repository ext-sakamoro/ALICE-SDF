//! Least-squares sphere fitting, the vertex list of a convex hull, and the
//! interior fill of the voxel convex decomposition.
//!
//! - `fit_sphere` on exact samples of a whole sphere, a hemisphere and a small
//!   cap (generated in f64 here) must return the sphere they came from; with
//!   radial Gaussian noise the error of each of centre x / y / z and radius
//!   must stay within 5 standard deviations of the Cramér–Rao bound
//!   `σ² (JᵀJ)⁻¹` of geometric least squares (J rows `[u, 1]`, `u` the unit
//!   direction from the true centre), inverted here by Gauss–Jordan. Points on
//!   one plane do not determine a sphere and give `None`.
//! - `convex_hull_from_points`: every vertex of the hull is used by a face, so
//!   points that were on the hull while it was being built and ended up inside
//!   it are not listed. Fixture: 40 points at radius 0.9 listed before 200
//!   Fibonacci points on the unit sphere, whose hull has inradius > 0.95, so
//!   the hull's vertices are exactly the 200 sphere points.
//! - `convex_decomposition`: two disjoint boxes give two parts whatever the
//!   axis along which they are separated (x, y or z), each part inside its box.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::{
    convex_decomposition, convex_hull_from_points, fit_sphere, FittedPrimitive, FittingConfig,
    Mesh, Vertex, VhacdConfig,
};
use glam::{DVec3, Vec3};
use std::collections::HashSet;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        // splitmix64 → [0, 1)
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Fibonacci directions on the cap of polar half-angle `theta_max` around +z
fn cap_dirs(n: usize, theta_max: f64) -> Vec<DVec3> {
    let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    let zmin = theta_max.cos();
    (0..n)
        .map(|i| {
            let z = 1.0 - (1.0 - zmin) * (i as f64 + 0.5) / n as f64;
            let s = (1.0 - z * z).sqrt();
            let phi = golden * i as f64;
            DVec3::new(s * phi.cos(), s * phi.sin(), z)
        })
        .collect()
}

fn sphere_of(r: &alice_sdf::mesh::FittingResult) -> (DVec3, f64) {
    match r.primitive {
        FittedPrimitive::Sphere { center, radius } => (center.as_dvec3(), f64::from(radius)),
        ref p => panic!("not a sphere: {p:?}"),
    }
}

/// inverse of a symmetric positive definite 4×4 by Gauss–Jordan
#[allow(clippy::needless_range_loop)]
fn inverse4(m: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut a = [[0.0; 8]; 4];
    for i in 0..4 {
        a[i][..4].copy_from_slice(&m[i]);
        a[i][4 + i] = 1.0;
    }
    for c in 0..4 {
        let p = (c..4)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, p);
        let d = a[c][c];
        for k in 0..8 {
            a[c][k] /= d;
        }
        for r in 0..4 {
            if r != c {
                let f = a[r][c];
                for k in 0..8 {
                    a[r][k] -= f * a[c][k];
                }
            }
        }
    }
    let mut inv = [[0.0; 4]; 4];
    for i in 0..4 {
        inv[i].copy_from_slice(&a[i][4..]);
    }
    inv
}

#[test]
fn sphere_fit_recovers_whole_sphere_hemisphere_and_cap_exactly() {
    let cfg = FittingConfig::default();
    let c = DVec3::new(0.3, -0.2, 0.5);
    let r = 1.7;
    let mut compared = 0;
    for (name, theta, tol) in [
        ("whole", std::f64::consts::PI, 1e-4),
        ("hemisphere", std::f64::consts::FRAC_PI_2, 1e-4),
        ("cap 30 deg", 30f64.to_radians(), 1e-3),
    ] {
        let pts: Vec<Vec3> = cap_dirs(800, theta)
            .into_iter()
            .map(|d| (c + d * r).as_vec3())
            .collect();
        let fit = fit_sphere(&pts, &cfg).expect(name);
        let (fc, fr) = sphere_of(&fit);
        assert!((fc - c).length() < tol, "{name}: centre {fc}");
        assert!((fr - r).abs() < tol, "{name}: radius {fr}");
        assert_eq!(fit.inlier_count, pts.len(), "{name}");
        compared += 1;
    }
    assert_eq!(compared, 3);
}

#[test]
fn algebraic_stage_alone_is_exact_on_a_tilted_cap() {
    // max_iterations = 0 skips the Gauss–Newton refinement, so this pins the
    // linear least-squares stage by itself. The cap is centred on a tilted
    // axis so the normal equations are not diagonal.
    let cfg = FittingConfig {
        max_iterations: 0,
        ..FittingConfig::default()
    };
    let c = DVec3::new(0.7, 1.1, -0.4);
    let r = 2.3;
    let axis = DVec3::new(1.0, 2.0, -0.5).normalize();
    let helper = axis.cross(DVec3::Z).normalize();
    let other = axis.cross(helper);
    let mut compared = 0;
    for theta in [60f64.to_radians(), 25f64.to_radians()] {
        let pts: Vec<Vec3> = cap_dirs(600, theta)
            .into_iter()
            .map(|d| (c + (helper * d.x + other * d.y + axis * d.z) * r).as_vec3())
            .collect();
        let (fc, fr) = sphere_of(&fit_sphere(&pts, &cfg).expect("cap"));
        assert!((fc - c).length() < 1e-3, "theta {theta}: centre {fc}");
        assert!((fr - r).abs() < 1e-3, "theta {theta}: radius {fr}");
        compared += 1;
    }
    assert_eq!(compared, 2);
}

#[test]
fn sphere_fit_with_noise_is_within_the_cramer_rao_bound() {
    let cfg = FittingConfig::default();
    let c = DVec3::new(-1.0, 0.4, 2.0);
    let r = 1.3;
    let sigma = 0.005;
    let mut rng = Rng(0x00C0_FFEE);
    let mut compared = 0;
    for (name, theta) in [
        ("whole", std::f64::consts::PI),
        ("hemisphere", std::f64::consts::FRAC_PI_2),
        ("cap 45 deg", 45f64.to_radians()),
    ] {
        let dirs = cap_dirs(2000, theta);
        let pts: Vec<Vec3> = dirs
            .iter()
            .map(|&d| (c + d * (r + sigma * rng.gauss())).as_vec3())
            .collect();
        let mut jtj = [[0.0f64; 4]; 4];
        for d in &dirs {
            let row = [d.x, d.y, d.z, 1.0];
            for i in 0..4 {
                for j in 0..4 {
                    jtj[i][j] += row[i] * row[j];
                }
            }
        }
        let cov = inverse4(jtj);
        let (fc, fr) = sphere_of(&fit_sphere(&pts, &cfg).expect(name));
        let err = [fc.x - c.x, fc.y - c.y, fc.z - c.z, fr - r];
        for k in 0..4 {
            let sd = sigma * cov[k][k].sqrt();
            assert!(
                err[k].abs() < 5.0 * sd + 1e-5,
                "{name}: parameter {k} error {} vs sd {sd}",
                err[k]
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 12);
}

#[test]
fn sphere_fit_rejects_points_that_do_not_determine_a_sphere() {
    let cfg = FittingConfig::default();
    let plane: Vec<Vec3> = (0..50)
        .map(|i| Vec3::new((i % 7) as f32 * 0.3, (i / 7) as f32 * 0.2, 1.5))
        .collect();
    assert!(fit_sphere(&plane, &cfg).is_none());
    let one_point = vec![Vec3::ONE; 10];
    assert!(fit_sphere(&one_point, &cfg).is_none());
    assert!(fit_sphere(&plane[..3], &cfg).is_none());
}

#[test]
fn convex_hull_lists_only_the_vertices_its_faces_use() {
    let mut rng = Rng(0xBEEF);
    let mut pts: Vec<Vec3> = (0..40)
        .map(|_| {
            let d = DVec3::new(rng.gauss(), rng.gauss(), rng.gauss()).normalize();
            (d * 0.9).as_vec3()
        })
        .collect();
    let outer: Vec<Vec3> = cap_dirs(200, std::f64::consts::PI)
        .into_iter()
        .map(|d| d.as_vec3())
        .collect();
    pts.extend_from_slice(&outer);

    let hull = convex_hull_from_points(&pts);
    let used: HashSet<u32> = hull.indices.iter().copied().collect();
    assert_eq!(
        used.len(),
        hull.vertices.len(),
        "every listed vertex is used by a face"
    );
    assert!(hull
        .indices
        .iter()
        .all(|&i| (i as usize) < hull.vertices.len()));
    for v in &hull.vertices {
        assert!(
            v.length() > 0.95,
            "interior point {v} listed as a hull vertex"
        );
    }
    let key = |v: &Vec3| v.to_array().map(f32::to_bits);
    let got: HashSet<[u32; 3]> = hull.vertices.iter().map(key).collect();
    let want: HashSet<[u32; 3]> = outer.iter().map(key).collect();
    assert_eq!(got, want);
    // closed: every directed edge has its reverse
    let edges: HashSet<(u32, u32)> = hull
        .indices
        .chunks_exact(3)
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .collect();
    assert!(edges.iter().all(|&(a, b)| edges.contains(&(b, a))));
}

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
    let quads = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 4, 7, 3],
    ];
    let mut indices = Vec::new();
    for q in quads {
        indices.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
    }
    Mesh {
        vertices: p.iter().map(|&v| Vertex::new(v, Vec3::Y)).collect(),
        indices,
    }
}

#[test]
fn convex_decomposition_separates_two_boxes_along_every_axis() {
    // the 16³ grid keeps the debug-build run short; the separating gap (4)
    // spans several cells at either resolution
    let cfg = VhacdConfig::fast();
    let mut compared = 0;
    for axis in 0..3 {
        let offset = Vec3::AXES[axis] * 3.0;
        let (ca, cb) = (-offset, offset);
        let mut mesh = box_mesh(ca, Vec3::ONE);
        let b = box_mesh(cb, Vec3::ONE);
        let off = mesh.vertices.len() as u32;
        mesh.vertices.extend_from_slice(&b.vertices);
        mesh.indices.extend(b.indices.iter().map(|i| i + off));

        let dec = convex_decomposition(&mesh, &cfg);
        assert_eq!(dec.parts.len(), 2, "boxes separated along axis {axis}");
        let mut near = [0usize; 2];
        for part in &dec.parts {
            for v in &part.vertices {
                // voxel centres: within one cell (8 × 1.02 / 16 < 0.6) of the box
                let in_a = (*v - ca).abs().max_element() <= 1.6;
                let in_b = (*v - cb).abs().max_element() <= 1.6;
                assert!(in_a ^ in_b, "axis {axis}: vertex {v} outside both boxes");
            }
            let c = part.vertices.iter().copied().sum::<Vec3>() / part.vertices.len() as f32;
            near[usize::from((c - cb).length() < (c - ca).length())] += 1;
        }
        assert_eq!(near, [1, 1], "axis {axis}: one part per box");
        compared += 1;
    }
    assert_eq!(compared, 3);
}
