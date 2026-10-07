//! `types::Aabb` is the one axis-aligned box of the crate (5.0.0 merged the
//! copy in `mesh::bvh` into it). The methods that came from the BVH copy are
//! checked against their closed forms on dyadic boxes (every value exact in
//! f32), and the BVH bounds are that same type.
//!
//! Every loop counts its comparisons and fails on zero.

use alice_sdf::mesh::MeshBvh;
use alice_sdf::types::Aabb;
use glam::Vec3;

/// Boxes with dyadic corners: (min, max)
const BOXES: [([f32; 3], [f32; 3]); 4] = [
    ([-1.0, -2.0, -0.5], [3.0, 2.0, 0.5]),
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    ([-0.25, 4.0, -8.0], [0.75, 4.5, 8.0]),
    ([2.0, -1.0, 0.0], [2.5, 6.0, 0.125]),
];

/// Distance from `p` to the box in f64: the exterior Euclidean distance, or
/// minus the distance to the nearest face inside
fn box_sdf(min: [f32; 3], max: [f32; 3], p: [f32; 3]) -> f64 {
    let mut out = 0.0f64;
    let mut inside = f64::NEG_INFINITY;
    for a in 0..3 {
        let (lo, hi, x) = (f64::from(min[a]), f64::from(max[a]), f64::from(p[a]));
        let d = (lo - x).max(x - hi);
        out += d.max(0.0).powi(2);
        inside = inside.max(d);
    }
    if inside > 0.0 {
        out.sqrt()
    } else {
        inside
    }
}

#[test]
fn moved_methods_match_their_closed_forms() {
    let mut compared = 0;
    for (min, max) in BOXES {
        let b = Aabb::new(Vec3::from(min), Vec3::from(max));
        let d = [0, 1, 2].map(|a| f64::from(max[a] - min[a]));
        // surface area 2(dx dy + dy dz + dz dx)
        let area = 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0]);
        assert_eq!(f64::from(b.surface_area()), area, "{min:?} {max:?}");
        // longest axis, ties to the later axis
        let longest = if d[0] > d[1] && d[0] > d[2] {
            0
        } else if d[1] > d[2] {
            1
        } else {
            2
        };
        assert_eq!(b.longest_axis(), longest);
        compared += 2;
        // signed distance on a lattice of dyadic points around the box
        for i in -4..=4 {
            for j in -4..=4 {
                for k in -4..=4 {
                    let p = [0, 1, 2].map(|a| {
                        let t = [i, j, k][a] as f32 / 4.0;
                        f32::midpoint(min[a], max[a]) + t * (max[a] - min[a])
                    });
                    let want = box_sdf(min, max, p);
                    let got = f64::from(b.signed_distance(Vec3::from(p)));
                    assert!(
                        (got - want).abs() <= 1e-6 * (1.0 + want.abs()),
                        "{min:?} {max:?} at {p:?}: {got} vs {want}"
                    );
                    compared += 1;
                }
            }
        }
        // empty + expand_point over the 8 corners gives the box back
        let mut e = Aabb::empty();
        assert!(e.min.x > e.max.x);
        for c in 0..8 {
            let corner = [0, 1, 2].map(|a| if c >> a & 1 == 1 { max[a] } else { min[a] });
            e.expand_point(Vec3::from(corner));
        }
        assert_eq!((e.min, e.max), (b.min, b.max));
        // expand_aabb is the in-place union
        let mut u = Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.25));
        let union = u.union(&b);
        u.expand_aabb(&b);
        assert_eq!((u.min, u.max), (union.min, union.max));
        compared += 2;
    }
    assert!(compared > 2000, "{compared}");
}

/// The BVH reports its bounds as `types::Aabb`: the box of the vertices
#[test]
fn bvh_bounds_are_the_shared_type() {
    let corners: Vec<Vec3> = (0..8)
        .map(|c| {
            Vec3::new(
                (c & 1) as f32 * 2.0 - 1.0,
                (c >> 1 & 1) as f32,
                (c >> 2 & 1) as f32 * 0.5,
            )
        })
        .collect();
    let indices = [0u32, 1, 2, 1, 3, 2, 4, 6, 5, 5, 6, 7];
    let bvh = MeshBvh::build(&corners, &indices, 2);
    let b: Aabb = bvh.bounds().unwrap();
    assert_eq!(b.min, Vec3::new(-1.0, 0.0, 0.0));
    assert_eq!(b.max, Vec3::new(1.0, 1.0, 0.5));
}
