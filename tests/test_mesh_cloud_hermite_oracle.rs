//! Point cloud SDF sign vote and the parameter of a refined Hermite edge
//! crossing.
//!
//! - `PointCloudSdf` decides the sign by an inverse-distance weighted vote of
//!   the K nearest points (module doc): each votes `+1` when the displacement
//!   from it to the query has a non-negative dot product with its normal and
//!   `-1` otherwise, with weight `1 / distance`. The vote is recomputed here by
//!   brute force in f64 for K = 1, 4, 8, 16 (with K = 1 it is the nearest
//!   point's normal alone); the magnitude stays the distance to the nearest
//!   point. On a sphere cloud with 20 % of the normals flipped, K = 8 must
//!   misclassify fewer queries than K = 1. `k_neighbors = 0` is rejected by
//!   `try_new`.
//! - `EdgeCrossing::t` locates `intersection` on its edge:
//!   `start + t · (end − start)` must be the intersection. The field
//!   `|p|² − 1` is quadratic along every edge, so the refined crossing differs
//!   from the linear interpolation of the end-point values.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::{extract_edge_crossings, HermiteConfig, PointCloudSdf, PointCloudSdfConfig};
use glam::{DVec3, Vec3};

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
    fn dir(&mut self) -> DVec3 {
        loop {
            let v = DVec3::new(self.unit(), self.unit(), self.unit()) * 2.0 - 1.0;
            let l = v.length();
            if l > 0.1 && l <= 1.0 {
                return v / l;
            }
        }
    }
}

/// Unit sphere samples (random directions) with outward normals, a fraction
/// `flip` of them flipped.
fn noisy_cloud(n: usize, flip: f64, seed: u64) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut rng = Rng(seed);
    let mut pts = Vec::with_capacity(n);
    let mut nrm = Vec::with_capacity(n);
    for _ in 0..n {
        let d = rng.dir();
        pts.push(d.as_vec3());
        let s = if rng.unit() < flip { -1.0 } else { 1.0 };
        nrm.push((d * s).as_vec3());
    }
    (pts, nrm)
}

fn queries(n: usize, seed: u64) -> Vec<Vec3> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|i| {
            let d = rng.dir();
            // alternately inside and outside, 0.05 .. 0.25 from the surface
            let off = 0.05 + 0.2 * rng.unit();
            let r = if i % 2 == 0 { 1.0 + off } else { 1.0 - off };
            (d * r).as_vec3()
        })
        .collect()
}

/// brute-force reference: (sign of the weighted vote, nearest distance)
fn reference(pts: &[Vec3], nrm: &[Vec3], k: usize, q: Vec3) -> (f64, f32) {
    let mut d: Vec<(f32, usize)> = pts
        .iter()
        .enumerate()
        .map(|(i, &p)| ((q - p).length_squared(), i))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut vote = 0.0f64;
    for &(dsq, i) in d.iter().take(k) {
        let side = if (q - pts[i]).as_dvec3().dot(nrm[i].as_dvec3()) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        vote += side / f64::from(dsq).sqrt();
    }
    (if vote >= 0.0 { 1.0 } else { -1.0 }, d[0].0.sqrt())
}

#[test]
fn sign_is_the_inverse_distance_weighted_vote_of_k_neighbours() {
    let (pts, nrm) = noisy_cloud(1500, 0.2, 7);
    let qs = queries(600, 11);
    let mut compared = 0;
    let mut differs_from_nearest = 0;
    for k in [1usize, 4, 8, 16] {
        let cfg = PointCloudSdfConfig {
            k_neighbors: k,
            ..PointCloudSdfConfig::default()
        };
        let sdf = PointCloudSdf::new(&pts, &nrm, &cfg);
        for &q in &qs {
            let d = sdf.eval(q);
            let (sign, nearest) = reference(&pts, &nrm, k, q);
            assert_eq!(d.abs().to_bits(), nearest.to_bits(), "k {k}: |eval({q})|");
            assert_eq!(f64::from(d.signum()), sign, "k {k}: sign at {q}");
            if k > 1 && sign != reference(&pts, &nrm, 1, q).0 {
                differs_from_nearest += 1;
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 4 * qs.len());
    // the vote is not the nearest point alone for K > 1
    assert!(differs_from_nearest > 0);
}

#[test]
fn more_neighbours_misclassify_fewer_queries_with_noisy_normals() {
    let (pts, nrm) = noisy_cloud(3000, 0.2, 21);
    let qs = queries(1000, 33);
    let errors = |k: usize| {
        let cfg = PointCloudSdfConfig {
            k_neighbors: k,
            ..PointCloudSdfConfig::default()
        };
        let sdf = PointCloudSdf::new(&pts, &nrm, &cfg);
        qs.iter()
            .filter(|&&q| (sdf.eval(q) < 0.0) != (q.length() < 1.0))
            .count()
    };
    let (e1, e8) = (errors(1), errors(8));
    println!("sign errors: k=1 {e1}, k=8 {e8} of {}", qs.len());
    // with K = 1 about the flipped fraction (20 %) is wrong
    assert!(e1 > qs.len() / 10, "fixture: {e1}");
    assert!(e8 * 2 < e1, "k=8 {e8} vs k=1 {e1}");
}

#[test]
fn zero_neighbours_is_rejected_at_construction() {
    let (pts, nrm) = noisy_cloud(50, 0.0, 3);
    let zero = PointCloudSdfConfig {
        k_neighbors: 0,
        ..PointCloudSdfConfig::default()
    };
    let err = PointCloudSdf::try_new(&pts, &nrm, &zero)
        .err()
        .expect("k = 0 must be an error");
    assert!(err.to_string().contains("k_neighbors"), "{err}");
    let mismatch = PointCloudSdf::try_new(&pts, &nrm[..49], &PointCloudSdfConfig::default());
    assert!(mismatch.is_err());
    let ok = PointCloudSdf::try_new(&pts, &nrm, &PointCloudSdfConfig::default()).unwrap();
    assert_eq!(ok.point_count(), 50);
}

#[test]
fn crossing_parameter_locates_the_refined_intersection() {
    let field = |p: Vec3| p.length_squared() - 1.0;
    let cfg = HermiteConfig {
        resolution: 8,
        ..HermiteConfig::default()
    };
    let crossings = extract_edge_crossings(&field, Vec3::splat(-1.5), Vec3::splat(1.5), &cfg);
    let mut compared = 0;
    let mut nonlinear = 0;
    for e in &crossings {
        let t = f64::from(e.t());
        assert!((0.0..=1.0).contains(&t), "t = {t}");
        let (s, en, x) = (
            e.start.as_dvec3(),
            e.end.as_dvec3(),
            e.intersection.as_dvec3(),
        );
        let at_t = s + (en - s) * t;
        let edge = (en - s).length();
        assert!(
            (at_t - x).length() < 1e-5 * edge,
            "start + t (end - start) = {at_t}, intersection {x}"
        );
        // the refined crossing is (nearly) on the unit sphere
        assert!((x.length() - 1.0).abs() < 1e-3, "|x| = {}", x.length());
        let linear = f64::from(e.start_dist / (e.start_dist - e.end_dist));
        if (linear - t).abs() > 1e-3 {
            nonlinear += 1;
        }
        compared += 1;
    }
    assert!(compared > 50, "{compared}");
    // the fixture distinguishes the refined parameter from the linear one
    assert!(nonlinear > compared / 2, "{nonlinear} of {compared}");
}
