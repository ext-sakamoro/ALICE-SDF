//! Mass properties used by `examples/bake_assets` for the physics manifest
//! (volume, centre of mass, inertia tensor about the centre of mass), checked
//! against closed forms worked out by hand — not against the implementation.
//!
//! - unit right tetrahedron (0,0,0),(1,0,0),(0,1,0),(0,0,1): V = 1/6,
//!   c = (1/4, 1/4, 1/4), ∫x² = 1/60, ∫xy = 1/120 ⇒ Ixx = 2/60 − V·2/16 = 1/80,
//!   Ixy = −(1/120 − V/16) = 1/480
//! - axis-aligned box with full sizes (w, h, d) centred at t:
//!   V = whd, c = t, Ixx = V(h² + d²)/12, products 0 (independent of t)
//! - marching-cubes sphere of radius r: V → 4/3 πr³, I → 2/5 V r² as the grid
//!   is refined (the error is the meshing error only)
//!
//! Author: Moroya Sakamoto

#[path = "../examples/bake_assets/mass.rs"]
#[allow(dead_code)]
mod mass;

use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig};
use alice_sdf::prelude::*;
use mass::mass_properties;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// 12 outward triangles of the box [t − s/2, t + s/2] (built by hand).
fn box_triangles(s: [f64; 3], t: [f64; 3]) -> Vec<[[f64; 3]; 3]> {
    let c = |i: usize| -> [f64; 3] {
        [
            t[0] + if i & 1 != 0 { 0.5 } else { -0.5 } * s[0],
            t[1] + if i & 2 != 0 { 0.5 } else { -0.5 } * s[1],
            t[2] + if i & 4 != 0 { 0.5 } else { -0.5 } * s[2],
        ]
    };
    // quads (CCW seen from outside), split into two triangles each
    let quads: [[usize; 4]; 6] = [
        [0, 2, 3, 1], // z−
        [4, 5, 7, 6], // z+
        [0, 1, 5, 4], // y−
        [2, 6, 7, 3], // y+
        [0, 4, 6, 2], // x−
        [1, 3, 7, 5], // x+
    ];
    let mut out = Vec::new();
    for q in quads {
        out.push([c(q[0]), c(q[1]), c(q[2])]);
        out.push([c(q[0]), c(q[2]), c(q[3])]);
    }
    out
}

#[test]
fn unit_right_tetrahedron_matches_hand_integrals() {
    let o = [0.0, 0.0, 0.0];
    let x = [1.0, 0.0, 0.0];
    let y = [0.0, 1.0, 0.0];
    let z = [0.0, 0.0, 1.0];
    // outward: base z=0 seen from below is (o, y, x), etc.
    let tris = vec![[o, y, x], [o, x, z], [o, z, y], [x, y, z]];
    let m = mass_properties(tris).expect("closed outward tetrahedron");
    let mut compared = 0;
    assert!(close(m.volume, 1.0 / 6.0, 1e-15), "V {}", m.volume);
    compared += 1;
    for k in 0..3 {
        assert!(close(m.centroid[k], 0.25, 1e-15), "c {:?}", m.centroid);
        assert!(
            close(m.inertia[k][k], 1.0 / 80.0, 1e-15),
            "I {:?}",
            m.inertia
        );
        compared += 2;
    }
    for (i, j) in [(0, 1), (1, 2), (0, 2)] {
        assert!(
            close(m.inertia[i][j], 1.0 / 480.0, 1e-15),
            "I {:?}",
            m.inertia
        );
        assert_eq!(m.inertia[i][j], m.inertia[j][i]);
        compared += 1;
    }
    assert_eq!(compared, 10, "every quantity compared");
}

#[test]
fn offset_box_matches_closed_form() {
    let mut compared = 0;
    for (s, t) in [
        ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        ([1.2, 0.8, 1.0], [0.1, 0.0, -0.05]),
        ([2.0, 0.25, 0.5], [-3.0, 7.5, 1.25]),
    ] {
        let m = mass_properties(box_triangles(s, t)).expect("closed box");
        let v = s[0] * s[1] * s[2];
        assert!(
            close(m.volume, v, 1e-12 * v.max(1.0)),
            "V {} vs {v}",
            m.volume
        );
        let diag = [
            v * (s[1] * s[1] + s[2] * s[2]) / 12.0,
            v * (s[2] * s[2] + s[0] * s[0]) / 12.0,
            v * (s[0] * s[0] + s[1] * s[1]) / 12.0,
        ];
        for k in 0..3 {
            assert!(
                close(m.centroid[k], t[k], 1e-12),
                "c {:?} vs {t:?}",
                m.centroid
            );
            assert!(
                close(m.inertia[k][k], diag[k], 1e-10),
                "I{k}{k} {} vs {}",
                m.inertia[k][k],
                diag[k]
            );
            compared += 2;
        }
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            assert!(close(m.inertia[i][j], 0.0, 1e-10), "I {:?}", m.inertia);
            compared += 1;
        }
    }
    assert_eq!(compared, 27);
}

#[test]
fn inside_out_or_empty_input_has_no_mass() {
    // empty: volume 0 → None (not a zero-mass body with NaN centroid)
    assert!(mass_properties(Vec::<[[f64; 3]; 3]>::new()).is_none());
    // every triangle reversed: negative volume → None
    let inverted: Vec<_> = box_triangles([1.0, 1.0, 1.0], [0.0; 3])
        .into_iter()
        .map(|[a, b, c]| [a, c, b])
        .collect();
    assert!(mass_properties(inverted).is_none());
}

#[test]
fn marching_cubes_sphere_converges_to_solid_sphere() {
    let r = 0.8f64;
    let node = SdfNode::sphere(r as f32);
    let v_true = 4.0 / 3.0 * std::f64::consts::PI * (r * r * r);
    let mut errs = Vec::new();
    for res in [32usize, 64] {
        let cfg = MarchingCubesConfig {
            resolution: res,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&node, Vec3::splat(-1.0), Vec3::splat(1.0), &cfg);
        let tris = mesh.indices.chunks_exact(3).map(|t| {
            [0, 1, 2].map(|k| mesh.vertices[t[k] as usize].position.as_dvec3().to_array())
        });
        let m = mass_properties(tris).expect("closed sphere mesh");
        let i_true = 0.4 * m.volume * r * r;
        let ev = (m.volume - v_true).abs() / v_true;
        let ei = (0..3)
            .map(|k| (m.inertia[k][k] - i_true).abs() / i_true)
            .fold(0.0f64, f64::max);
        let ec = m.centroid.iter().fold(0.0f64, |a, c| a.max(c.abs()));
        let eo = [(0, 1), (1, 2), (0, 2)]
            .iter()
            .fold(0.0f64, |a, &(i, j)| a.max(m.inertia[i][j].abs() / i_true));
        eprintln!("res {res}: dV {ev:.2e} dI {ei:.2e} |c| {ec:.2e} off-diag {eo:.2e}");
        assert!(ev < 0.02 && ei < 0.02, "res {res}: dV {ev} dI {ei}");
        assert!(ec < 1e-3 && eo < 1e-3, "res {res}: |c| {ec} off-diag {eo}");
        errs.push(ev);
    }
    assert_eq!(errs.len(), 2);
    assert!(
        errs[1] < errs[0],
        "refining the grid must reduce the volume error: {errs:?}"
    );
}
