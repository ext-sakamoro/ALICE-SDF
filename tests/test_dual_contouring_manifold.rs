//! Dual contouring topology: the default one-vertex-per-cell output stays bit
//! for bit what it was, and the manifold mode (one vertex per surface sheet
//! in a cell) gives a closed, manifold, uncrossed mesh where one vertex per
//! cell joins two sheets.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::dual_contouring::{
    dual_contouring_field, dual_contouring_with_topology, DcTopology,
};
use alice_sdf::mesh::{dual_contouring, validate_mesh, DualContouringConfig, Mesh};
use alice_sdf::prelude::*;
use glam::DVec3;
use std::collections::{HashMap, HashSet};

/// FNV-1a over every output bit (all vertex attributes, then the indices)
fn digest(m: &Mesh) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |w: u32| {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for v in &m.vertices {
        for f in [
            v.position.x,
            v.position.y,
            v.position.z,
            v.normal.x,
            v.normal.y,
            v.normal.z,
            v.uv.x,
            v.uv.y,
            v.uv2.x,
            v.uv2.y,
            v.tangent.x,
            v.tangent.y,
            v.tangent.z,
            v.tangent.w,
            v.color[0],
            v.color[1],
            v.color[2],
            v.color[3],
        ] {
            eat(f.to_bits());
        }
        eat(v.material_id);
    }
    eat(m.vertices.len() as u32);
    for &i in &m.indices {
        eat(i);
    }
    h
}

/// The default output, digested at 19efcf7 before the topology option
/// existed: sphere, box (sharp features), torus (the fin collapse on its
/// inner equator), a smooth union, and the AAA preset (uv / tangent)
fn golden_cases() -> Vec<(&'static str, SdfNode, DualContouringConfig, Vec3, Vec3)> {
    let cfg = |res| DualContouringConfig {
        resolution: res,
        ..Default::default()
    };
    vec![
        (
            "sphere",
            SdfNode::sphere(1.0),
            cfg(32),
            Vec3::splat(-1.37),
            Vec3::splat(1.41),
        ),
        (
            "box",
            SdfNode::box3d(1.4, 1.0, 0.7),
            cfg(32),
            Vec3::splat(-1.2),
            Vec3::splat(1.3),
        ),
        (
            "torus",
            SdfNode::torus(1.0, 0.5),
            cfg(32),
            Vec3::splat(-1.6),
            Vec3::splat(1.6),
        ),
        (
            "smooth union",
            SdfNode::sphere(0.7)
                .smooth_union(SdfNode::box3d(1.6, 0.4, 0.4).translate(0.4, 0.1, 0.0), 0.2),
            cfg(40),
            Vec3::splat(-1.3),
            Vec3::splat(1.35),
        ),
        (
            "aaa sphere",
            SdfNode::sphere(0.9),
            DualContouringConfig::aaa(24),
            Vec3::splat(-1.1),
            Vec3::splat(1.2),
        ),
    ]
}

const GOLDEN: [u64; 5] = [
    0xee20_653a_a03e_9e0f,
    0xcbb9_1d69_6cd8_c897,
    0x85bb_e6d1_5833_f6d8,
    0xf569_d7e1_dea0_78ac,
    0xfbdd_28aa_1b99_455d,
];

/// The default entry and the explicit `OneVertexPerCell` topology both
/// reproduce the 19efcf7 digests
#[test]
fn default_output_is_unchanged_bit_for_bit() {
    for ((name, node, cfg, lo, hi), want) in golden_cases().into_iter().zip(GOLDEN) {
        let m = dual_contouring(&node, lo, hi, &cfg);
        assert_eq!(digest(&m), want, "{name}: default output moved");
        let m = dual_contouring_with_topology(&node, lo, hi, &cfg, DcTopology::OneVertexPerCell);
        assert_eq!(
            digest(&m),
            want,
            "{name}: OneVertexPerCell differs from the default"
        );
    }
}

/// Edges used by more than two triangles / by one triangle
fn bad_edges(m: &Mesh) -> (usize, usize) {
    let v = validate_mesh(m);
    (v.non_manifold_edges, v.boundary_edges)
}

/// Euler characteristic V − E + F over the vertices the triangles use, and
/// the number of connected pieces
fn topology(m: &Mesh) -> (i64, usize) {
    let mut parent: HashMap<u32, u32> = HashMap::new();
    fn root(p: &HashMap<u32, u32>, mut i: u32) -> u32 {
        while let Some(&j) = p.get(&i) {
            if j == i {
                break;
            }
            i = j;
        }
        i
    }
    let mut edges = HashSet::new();
    for t in m.indices.chunks_exact(3) {
        for &v in t {
            parent.entry(v).or_insert(v);
        }
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.insert((a.min(b), a.max(b)));
            let (ra, rb) = (root(&parent, a), root(&parent, b));
            if ra != rb {
                parent.insert(ra.max(rb), ra.min(rb));
            }
        }
    }
    let v = parent.len() as i64;
    let keys: Vec<u32> = parent.keys().copied().collect();
    let pieces = keys.iter().filter(|&&k| root(&parent, k) == k).count();
    (
        v - edges.len() as i64 + (m.indices.len() / 3) as i64,
        pieces,
    )
}

/// A cylinder whose axis lies parallel to a grid plane along the plane's
/// diagonal, just poking through it (depth DIP). In local coordinates the
/// plane is z = 0 and the axis runs along (1, 1, 0) at height RADIUS − DIP.
/// On the plane the inside is a band along the diagonal narrower than the
/// face diagonal, so every face along it has alternating corners: the
/// on-axis corners (i, i, 0) are inside at −DIP, the two off-axis corners
/// outside, and the face centre is inside (by DIP). On the axis side the
/// cell holds two sheets (one around each off-axis corner), on the other
/// side one thin cap owning all four face edges. One vertex per cell joins
/// the cap's vertex to a single vertex through all four edges: four quads
/// share that vertex pair. Grid points are the integers (bounds ±9,
/// resolution 18: the shape stays more than a cell inside the bounds).
const DIP: f64 = 0.05;
const RADIUS: f64 = 4.0;

/// Local (x, y, z) → global: z along `axis` with sign `side`, x and y along
/// the next two axes
fn to_global(q: DVec3, axis: usize, side: f64) -> DVec3 {
    let mut p = [0.0; 3];
    p[axis] = side * q.z;
    p[(axis + 1) % 3] = q.x;
    p[(axis + 2) % 3] = q.y;
    DVec3::from(p)
}

fn tangent_cylinder(axis: usize, side: f64) -> impl Fn(Vec3) -> f32 + Sync {
    move |p: Vec3| {
        let p = [f64::from(p.x), f64::from(p.y), f64::from(p.z)];
        let q = DVec3::new(p[(axis + 1) % 3], p[(axis + 2) % 3], side * p[axis]);
        let z0 = RADIUS - DIP;
        let a = DVec3::new(-3.0, -3.0, z0);
        let b = DVec3::new(3.0, 3.0, z0);
        let ab = b - a;
        let u = ((q - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
        ((q - (a + ab * u)).length() - RADIUS) as f32
    }
}

/// Generalized winding number of a closed mesh at x (solid angles summed,
/// Van Oosterom and Strackee): 1 inside, 0 outside
fn winding(m: &Mesh, x: DVec3) -> f64 {
    let g = |i: u32| {
        let v = m.vertices[i as usize].position;
        DVec3::new(f64::from(v.x), f64::from(v.y), f64::from(v.z)) - x
    };
    let mut total = 0.0;
    for t in m.indices.chunks_exact(3) {
        let (a, b, c) = (g(t[0]), g(t[1]), g(t[2]));
        let (la, lb, lc) = (a.length(), b.length(), c.length());
        let num = a.dot(b.cross(c));
        let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
        total += 2.0 * alice_det_math::atan2_64(num, den);
    }
    total / (4.0 * std::f64::consts::PI)
}

/// Each vertex sits on its own sheet: |field| ≤ VERTEX_TOL cells at every
/// vertex (measured: ≤ 0.072 on the tangent cylinder, ≤ 0.036 on the limbs).
/// A vertex solved from the crossings of two sheets lands between them, on
/// the order of half a cell off.
const VERTEX_TOL: f32 = 0.1;

/// Largest |field| over the mesh vertices, in cells
fn worst_vertex_distance(m: &Mesh, f: &dyn Fn(Vec3) -> f32, cell: f32) -> f32 {
    m.vertices
        .iter()
        .map(|v| f(v.position).abs() / cell)
        .fold(0.0, f32::max)
}

/// The tangent-cylinder scene on all six sides of the three grid planes
/// (each of the six entries of the cell's face table meets the alternating
/// face on the two-sheet side once)
#[test]
fn alternating_face_case_is_manifold_only_with_per_sheet_vertices() {
    let cfg = DualContouringConfig {
        resolution: 18,
        ..Default::default()
    };
    let (lo, hi) = (Vec3::splat(-9.0), Vec3::splat(9.0));
    for axis in 0..3 {
        for side in [1.0, -1.0] {
            let tag = format!("axis {axis} side {side}");
            let f = tangent_cylinder(axis, side);
            let at = |x: f64, y: f64| {
                let p = to_global(DVec3::new(x, y, 0.0), axis, side);
                f(Vec3::new(p.x as f32, p.y as f32, p.z as f32))
            };
            // the scene does exercise the case: these faces alternate, and
            // one vertex per cell is not manifold there
            let inside = |x: i32, y: i32| at(f64::from(x), f64::from(y)) < 0.0;
            let alternating = (-3..3)
                .filter(|&i| {
                    inside(i, i) && inside(i + 1, i + 1) && !inside(i + 1, i) && !inside(i, i + 1)
                })
                .count();
            assert_eq!(alternating, 6, "{tag}");
            let per_cell = dual_contouring_field(&f, lo, hi, &cfg, DcTopology::OneVertexPerCell);
            let (nm, open) = bad_edges(&per_cell);
            assert!(
                nm > 0,
                "{tag}: one vertex per cell is manifold here: the scene does not test the case"
            );
            assert_eq!(open, 0, "{tag}");
            let m = dual_contouring_field(&f, lo, hi, &cfg, DcTopology::Manifold);
            assert_eq!(
                bad_edges(&m),
                (0, 0),
                "{tag}: manifold topology: non-manifold / open edges"
            );
            assert_eq!(crossings(&m), 0, "{tag}");
            // the capsule is one closed surface of genus 0, its cap past the
            // plane included
            assert_eq!(topology(&m), (2, 1), "{tag}: Euler characteristic / pieces");
            assert!(
                m.vertices.len() > per_cell.vertices.len(),
                "{tag}: no cell got a second vertex"
            );
            // the decider keeps the face centres inside: the on-axis corners
            // stay connected through each alternating face
            for i in -3..3 {
                let c = to_global(
                    DVec3::new(f64::from(i) + 0.5, f64::from(i) + 0.5, 0.0),
                    axis,
                    side,
                );
                assert!(at(f64::from(i) + 0.5, f64::from(i) + 0.5) < 0.0);
                let w = winding(&m, c);
                assert!(w > 0.5, "{tag}: face centre {i}: winding number {w}");
            }
            let worst = worst_vertex_distance(&m, &f, 1.0);
            assert!(
                worst <= VERTEX_TOL,
                "{tag}: a vertex is {worst} cells off its sheet"
            );
        }
    }
}

/// Two capsules joined by the polynomial smooth minimum: the two-bone limb
/// an offset layer is taken from (radii, lengths, blend k, bend angle)
struct Limb {
    r: [f64; 2],
    l: [f64; 2],
    k: f64,
    bend: f64,
}

impl Limb {
    fn sdf(&self, x: DVec3) -> f64 {
        let cap = |a: DVec3, b: DVec3, r: f64| {
            let ab = b - a;
            let u = ((x - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
            (x - (a + ab * u)).length() - r
        };
        let j = DVec3::new(0.0, self.l[0], 0.0);
        let (s, c) = alice_det_math::sin_cos64(self.bend);
        let a = cap(DVec3::ZERO, j, self.r[0]);
        let b = cap(j, j + DVec3::new(0.0, c, s) * self.l[1], self.r[1]);
        let hh = (self.k - (a - b).abs()).max(0.0) / self.k;
        a.min(b) - hh * hh * self.k * 0.25
    }
}

fn limbs() -> Vec<(&'static str, Limb)> {
    (0..=135)
        .step_by(15)
        .flat_map(|deg| {
            let bend = f64::from(deg).to_radians();
            [
                (
                    "equal",
                    Limb {
                        r: [0.05, 0.05],
                        l: [0.3, 0.3],
                        k: 0.04,
                        bend,
                    },
                ),
                (
                    "unequal",
                    Limb {
                        r: [0.05, 0.035],
                        l: [0.3, 0.22],
                        k: 0.04,
                        bend,
                    },
                ),
            ]
        })
        .collect()
}

/// The offset surface sdf = h of every limb (bends 0–135° every 15°, equal
/// and unequal bones, h = 2 / 20 mm) on a 4 mm grid: closed, manifold,
/// uncrossed, and identical whether rayon runs on one thread or many. One
/// vertex per cell fails the first property on equal bones at 45° and 135°
/// with h = 2 mm (four triangles on each bad edge, along the tubes, where
/// the thin layer cuts a grid face in a band)
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release profile: 40 meshes on a 275³ grid take about 13 min unoptimised; release-profile.yml runs this file with --release"
)]
fn limb_offset_surfaces_are_closed_manifold_and_uncrossed() {
    let cfg = DualContouringConfig {
        resolution: 275,
        ..Default::default()
    };
    let (lo, hi) = (Vec3::splat(-0.4), Vec3::splat(0.7));
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut checked = 0;
    for (name, b) in limbs() {
        for h in [0.002, 0.02] {
            let f = |p: Vec3| {
                (b.sdf(DVec3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z))) - h) as f32
            };
            let m = dual_contouring_field(&f, lo, hi, &cfg, DcTopology::Manifold);
            let tag = format!("{name} {:.0}° h {h}", b.bend.to_degrees());
            assert!(
                m.indices.len() / 3 > 1000,
                "{tag}: {} triangles",
                m.indices.len() / 3
            );
            assert_eq!(bad_edges(&m), (0, 0), "{tag}: non-manifold / open edges");
            assert_eq!(crossings(&m), 0, "{tag}: crossings");
            assert_eq!(topology(&m), (2, 1), "{tag}: Euler characteristic / pieces");
            let worst = worst_vertex_distance(&m, &f, 1.1 / 275.0);
            assert!(
                worst <= VERTEX_TOL,
                "{tag}: a vertex is {worst} cells off its sheet"
            );
            let one =
                single.install(|| dual_contouring_field(&f, lo, hi, &cfg, DcTopology::Manifold));
            assert_eq!(
                digest(&one),
                digest(&m),
                "{tag}: output depends on the thread count"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 10 * 2 * 2);
}

/// Triangle pairs sharing no vertex where an edge of one passes strictly
/// through the other (f64 on the f32 positions), grid-accelerated
fn crossings(m: &Mesh) -> usize {
    let p: Vec<DVec3> = m
        .vertices
        .iter()
        .map(|v| {
            DVec3::new(
                f64::from(v.position.x),
                f64::from(v.position.y),
                f64::from(v.position.z),
            )
        })
        .collect();
    let tris: Vec<[usize; 3]> = m
        .indices
        .chunks_exact(3)
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .collect();
    let span = p.iter().fold(0.0f64, |a, x| a.max(x.abs().max_element()));
    let cell = (span / 64.0).max(1e-9);
    let c = |x: f64| (x / cell).floor() as i64;
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, t) in tris.iter().enumerate() {
        let lo = p[t[0]].min(p[t[1]]).min(p[t[2]]);
        let hi = p[t[0]].max(p[t[1]]).max(p[t[2]]);
        for x in c(lo.x)..=c(hi.x) {
            for y in c(lo.y)..=c(hi.y) {
                for z in c(lo.z)..=c(hi.z) {
                    grid.entry([x, y, z]).or_default().extend([i]);
                }
            }
        }
    }
    let hits = |e0: DVec3, e1: DVec3, a: DVec3, b: DVec3, cc: DVec3| {
        let n = (b - a).cross(cc - a);
        let (d0, d1) = ((e0 - a).dot(n), (e1 - a).dot(n));
        if d0 * d1 >= 0.0 {
            return false;
        }
        let x = e0 + (e1 - e0) * (d0 / (d0 - d1));
        let side = |u: DVec3, v: DVec3| (v - u).cross(x - u).dot(n);
        let (s0, s1, s2) = (side(a, b), side(b, cc), side(cc, a));
        (s0 > 0.0 && s1 > 0.0 && s2 > 0.0) || (s0 < 0.0 && s1 < 0.0 && s2 < 0.0)
    };
    let mut seen = HashSet::new();
    for list in grid.values() {
        for (k, &a) in list.iter().enumerate() {
            for &b in &list[k + 1..] {
                let (a, b) = (a.min(b), a.max(b));
                let (ta, tb) = (&tris[a], &tris[b]);
                if ta.iter().any(|i| tb.contains(i)) || seen.contains(&(a, b)) {
                    continue;
                }
                let e =
                    |t: &[usize; 3]| [(p[t[0]], p[t[1]]), (p[t[1]], p[t[2]]), (p[t[2]], p[t[0]])];
                if e(ta)
                    .iter()
                    .any(|&(x, y)| hits(x, y, p[tb[0]], p[tb[1]], p[tb[2]]))
                    || e(tb)
                        .iter()
                        .any(|&(x, y)| hits(x, y, p[ta[0]], p[ta[1]], p[ta[2]]))
                {
                    seen.insert((a, b));
                }
            }
        }
    }
    seen.len()
}

/// Manifold mode on SdfNode shapes: closed, manifold, uncrossed, with the
/// shapes' own topology (sphere, box and smooth union genus 0, torus genus 1;
/// the torus's inner equator lies on a grid plane, where the fin collapse
/// runs). The torus's one crossing is the default output's too (below).
#[test]
fn manifold_mode_on_sdf_nodes_keeps_their_topology() {
    let want = [(2, 1), (2, 1), (0, 1), (2, 1), (2, 1)];
    for ((name, node, cfg, lo, hi), want) in golden_cases().into_iter().zip(want) {
        let m = dual_contouring_with_topology(&node, lo, hi, &cfg, DcTopology::Manifold);
        assert_eq!(bad_edges(&m), (0, 0), "{name}: non-manifold / open edges");
        if name != "torus" {
            assert_eq!(crossings(&m), 0, "{name}: crossings");
        }
        assert_eq!(topology(&m), want, "{name}: Euler characteristic / pieces");
    }
}

/// Known defect, in both topologies: the torus at resolution 32 (inner
/// equator on a grid plane) has one pair of triangles that cross
#[test]
#[ignore = "known defect: DC-TORUS-CROSS-001: torus(1.0, 0.5) at resolution 32 over ±1.6 has 1 crossing triangle pair in both OneVertexPerCell and Manifold output (inner equator on a grid plane, where the fin collapse in triangulate_quads runs)"]
fn torus_on_a_grid_plane_has_no_crossings() {
    let (_, node, cfg, lo, hi) = golden_cases().swap_remove(2);
    for topo in [DcTopology::OneVertexPerCell, DcTopology::Manifold] {
        let m = dual_contouring_with_topology(&node, lo, hi, &cfg, topo);
        assert_eq!(crossings(&m), 0, "{topo:?}");
    }
}

/// The saddle x·y − c, clipped by a box, along each axis. On every grid face
/// across the saddle's axis the four corners alternate, and the field is
/// its own bilinear interpolant there, so the asymptotic decider is exact:
/// c > 0 puts the face centre (value −c) inside, joining the two inside
/// quadrants into one solid (genus 0: χ 2, 1 piece); c < 0 puts it outside
/// and the inside is two solids (χ 4, 2 pieces). Grid points are the
/// half-integers (bounds −4.5..3.5, resolution 8), so the saddle's axis runs
/// through the faces with corners at (±0.5, ±0.5). With the saddle moved to
/// (0.2, 0.2) and c 0.02 the face centre's value is +0.02 but the saddle's
/// is −0.02: the inside is still one solid, and only the saddle's value (not
/// the centre's) says so.
#[test]
fn asymptotic_decider_gives_the_saddles_connectivity() {
    let cfg = DualContouringConfig {
        resolution: 8,
        ..Default::default()
    };
    let (lo, hi) = (Vec3::splat(-4.5), Vec3::splat(3.5));
    for axis in 0..3 {
        for (s0, c, want) in [
            (0.0f32, 0.05f32, (2, 1)),
            (0.0, -0.05, (4, 2)),
            (0.2, 0.02, (2, 1)),
        ] {
            let f = move |p: Vec3| {
                let p = p.to_array();
                let (x, y, z) = (p[(axis + 1) % 3], p[(axis + 2) % 3], p[axis]);
                let half = 2.2f32;
                let boxd = (x.abs() - half).max(y.abs() - half).max(z.abs() - half);
                ((x - s0) * (y - s0) - c).max(boxd)
            };
            let tag = format!("axis {axis} saddle at {s0} c {c}");
            let m = dual_contouring_field(&f, lo, hi, &cfg, DcTopology::Manifold);
            assert_eq!(bad_edges(&m), (0, 0), "{tag}: non-manifold / open edges");
            assert_eq!(crossings(&m), 0, "{tag}: crossings");
            assert_eq!(topology(&m), want, "{tag}: Euler characteristic / pieces");
        }
    }
}

/// Degenerate inputs give an empty mesh in both topologies, without
/// panicking: resolution 0 or 1 (no edge inside the grid has four cells),
/// a field that is everywhere outside, everywhere inside, or NaN (NaN < 0 is
/// false: outside), and a box flat in z
#[test]
fn degenerate_inputs_give_an_empty_mesh() {
    let sphere = |p: Vec3| p.length() - 0.5;
    let (lo, hi) = (Vec3::splat(-1.0), Vec3::splat(1.0));
    type Case<'a> = (&'a str, usize, Vec3, &'a (dyn Fn(Vec3) -> f32 + Sync));
    let cases: [Case; 6] = [
        ("resolution 0", 0, hi, &sphere),
        ("resolution 1", 1, hi, &sphere),
        ("outside everywhere", 8, hi, &|_| 1.0),
        ("inside everywhere", 8, hi, &|_| -1.0),
        ("NaN", 8, hi, &|_| f32::NAN),
        ("flat box", 8, Vec3::new(1.0, 1.0, -1.0), &sphere),
    ];
    for topo in [DcTopology::OneVertexPerCell, DcTopology::Manifold] {
        for (name, res, hi, field) in cases {
            let cfg = DualContouringConfig {
                resolution: res,
                ..Default::default()
            };
            let m = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                dual_contouring_field(field, lo, hi, &cfg, topo)
            }))
            .unwrap_or_else(|_| panic!("{topo:?} {name}: panicked"));
            assert_eq!(
                (m.vertices.len(), m.indices.len()),
                (0, 0),
                "{topo:?} {name}"
            );
        }
    }
}

/// A shape within a cell of the max bound is closed, as one within a cell of
/// the min bound is: a sphere of radius 0.75 centred at (0.2, 0.2, 0.2) over
/// [−1, 1] at resolution 8 (cell 0.25) reaches 0.95, inside the last cell layer
/// on all three axes, and its mirror image at −0.2 reaches −0.95. Both are
/// closed and manifold in both topologies, with the same vertex and triangle
/// counts (the grid treats its two ends alike). Until the quad loop took the
/// last layer's edges (DC-LAST-LAYER-001) the shape near the max bound had 48
/// open edges.
#[test]
fn shapes_near_the_max_bound_are_closed() {
    let cfg = DualContouringConfig {
        resolution: 8,
        ..Default::default()
    };
    for topo in [DcTopology::OneVertexPerCell, DcTopology::Manifold] {
        let mut counts = Vec::new();
        for c in [0.2f32, -0.2] {
            let f = move |p: Vec3| (p - Vec3::splat(c)).length() - 0.75;
            let m = dual_contouring_field(&f, Vec3::splat(-1.0), Vec3::splat(1.0), &cfg, topo);
            assert!(!m.indices.is_empty(), "{topo:?} centre {c}: empty mesh");
            assert_eq!(
                bad_edges(&m),
                (0, 0),
                "{topo:?} centre {c}: non-manifold / open edges"
            );
            assert_eq!(
                topology(&m),
                (2, 1),
                "{topo:?} centre {c}: Euler characteristic / pieces"
            );
            counts.extend([(m.vertices.len(), m.indices.len())]);
        }
        assert_eq!(
            counts[0], counts[1],
            "{topo:?}: the two ends of the grid differ"
        );
    }
}

/// Non-finite field values (NaN, ±∞) in part of the grid: no panic, the same
/// output on a second run, and the classification the comparisons give: NaN < 0
/// is false, so a NaN corner counts as outside, −∞ as inside and +∞ as outside.
/// The vertices (one per sheet of each crossed cell) depend only on the corners'
/// signs, so each field has as many vertices as the same field with its
/// non-finite region replaced by +1 (outside) or −1 (inside).
#[test]
fn non_finite_corner_values_are_deterministic() {
    let cfg = DualContouringConfig {
        resolution: 12,
        ..Default::default()
    };
    let sphere = |p: Vec3| p.length() - 0.7;
    type Case<'a> = (
        &'a str,
        &'a (dyn Fn(Vec3) -> f32 + Sync),
        &'a (dyn Fn(Vec3) -> f32 + Sync),
    );
    let cases: [Case; 3] = [
        (
            "NaN for x > 0.3",
            &|p: Vec3| if p.x > 0.3 { f32::NAN } else { sphere(p) },
            &|p: Vec3| if p.x > 0.3 { 1.0 } else { sphere(p) },
        ),
        (
            "-inf for y < -0.4",
            &|p: Vec3| {
                if p.y < -0.4 {
                    f32::NEG_INFINITY
                } else {
                    sphere(p)
                }
            },
            &|p: Vec3| if p.y < -0.4 { -1.0 } else { sphere(p) },
        ),
        (
            "+inf for z > 0.5",
            &|p: Vec3| if p.z > 0.5 { f32::INFINITY } else { sphere(p) },
            &|p: Vec3| if p.z > 0.5 { 1.0 } else { sphere(p) },
        ),
    ];
    for topo in [DcTopology::OneVertexPerCell, DcTopology::Manifold] {
        for (name, f, same_signs) in cases {
            let run = |field: &(dyn Fn(Vec3) -> f32 + Sync)| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    dual_contouring_field(field, Vec3::splat(-1.0), Vec3::splat(1.0), &cfg, topo)
                }))
                .unwrap_or_else(|_| panic!("{topo:?} {name}: panicked"))
            };
            let (a, b) = (run(f), run(f));
            assert!(!a.indices.is_empty(), "{topo:?} {name}: empty mesh");
            assert_eq!(digest(&a), digest(&b), "{topo:?} {name}: not deterministic");
            let reference = run(same_signs);
            assert_eq!(
                a.vertices.len(),
                reference.vertices.len(),
                "{topo:?} {name}: classified unlike the finite field with the same signs"
            );
        }
    }
}
