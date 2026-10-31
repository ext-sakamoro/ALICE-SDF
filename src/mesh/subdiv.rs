//! Catmull–Clark subdivision of an all-quad cage, with boundaries, creases and the limit surface
//!
//! A [`Cage`] is a closed or bounded quad mesh: positions, faces (four vertex indices, wound
//! consistently) and per-edge sharpness. [`subdivide`] applies one Catmull–Clark step:
//!
//! - face point: the average of the face's four vertices;
//! - smooth interior edge point: (v₀ + v₁ + f₀ + f₁) / 4 (f the two adjacent face points);
//! - boundary or sharp edge point: (v₀ + v₁) / 2;
//! - smooth vertex of valence n: (F̄ + 2 Ē + (n − 3) v) / n, F̄ the average of the adjacent face
//!   points and Ē of the adjacent edge midpoints;
//! - crease or boundary vertex (exactly two sharp or boundary edges): (a + 6 v + b) / 8, a and b
//!   the vertices across those two edges (the cubic B-spline curve rule);
//! - corner (more than two sharp or boundary edges, or a boundary vertex of valence 2): v.
//!
//! Sharpness is an integer per edge (semi-sharp creases act at integer steps only): an edge of
//! sharpness s ≥ 1 is sharp for this step and its two halves get s − 1; [`INFINITE`] stays.
//!
//! [`limit_positions`] gives each vertex's point on the limit surface:
//!
//! - smooth vertex of valence n: (n² v + 4 Σ eᵢ + Σ fᵢ) / (n (n + 5)), eᵢ the vertices across the
//!   n edges and fᵢ the vertices diagonally across the n faces;
//! - crease or boundary vertex: (a + 4 v + b) / 6;
//! - corner: v.
//!
//! [`limit_normals`] gives the limit surface's unit normal at each smooth interior vertex from the
//! two tangent masks (n the valence, eᵢ and fᵢ the ring in cyclic order, A = 1 + cos(2π/n) +
//! cos(π/n)·√(2 (9 + cos(2π/n)))):
//!
//! - t₁ = Σ A cos(2πi/n) eᵢ + (cos(2πi/n) + cos(2π(i+1)/n)) fᵢ;
//! - t₂ = Σ A sin(2πi/n) eᵢ + (sin(2πi/n) + sin(2π(i+1)/n)) fᵢ;
//! - n = t₁ × t₂ normalised (the ring runs in the faces' winding, so n is on that side); none at crease, corner and
//!   boundary vertices (no single tangent plane there).
//!
//! Deterministic: every sum runs in a fixed order (faces in index order, a vertex's ring in the
//! order of its faces), with no hashing and no parallel reduction, so a rebuild is bit-identical.

use alice_det_math::{cos64, sin64, sqrt64};
use glam::DVec3;
use std::collections::BTreeMap;

/// The sharpness of an edge that stays sharp at every level
pub const INFINITE: u32 = u32::MAX;

/// A quad cage
#[derive(Clone, Debug, PartialEq)]
pub struct Cage {
    /// Vertex positions
    pub positions: Vec<DVec3>,
    /// Faces, four vertex indices each, wound consistently
    pub faces: Vec<[u32; 4]>,
    /// Sharpness of the edges that have any, keyed by (smaller index, larger index)
    pub sharpness: BTreeMap<(u32, u32), u32>,
}

/// Why a cage was refused
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CageError {
    /// A face names a vertex that does not exist
    VertexOutOfRange {
        /// The face's index
        face: usize,
    },
    /// A face repeats a vertex
    DegenerateFace {
        /// The face's index
        face: usize,
    },
    /// A directed edge is used by two faces (the faces are not wound consistently)
    InconsistentWinding {
        /// The edge
        edge: (u32, u32),
    },
    /// An edge is used by more than two faces
    NonManifoldEdge {
        /// The edge
        edge: (u32, u32),
    },
    /// A sharpness entry names an edge no face has
    SharpnessOnMissingEdge {
        /// The edge
        edge: (u32, u32),
    },
    /// The cage has no face
    Empty,
}

/// What [`check`] measured on a valid cage
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    /// Vertex count
    pub vertices: usize,
    /// Edge count
    pub edges: usize,
    /// Face count
    pub faces: usize,
    /// Edges with one face
    pub boundary_edges: usize,
    /// Per vertex: its valence (the number of edges at it)
    pub valence: Vec<usize>,
    /// Per vertex: whether it lies on a boundary edge
    pub on_boundary: Vec<bool>,
}

impl Topology {
    /// V − E + F
    #[must_use]
    pub fn euler(&self) -> i64 {
        self.vertices as i64 - self.edges as i64 + self.faces as i64
    }

    /// Σ over interior vertices of (4 − valence) plus Σ over boundary vertices of (3 − valence):
    /// 4 χ for a quad mesh (the discrete Gauss–Bonnet count of its irregularity)
    #[must_use]
    pub fn valence_defect(&self) -> i64 {
        self.valence
            .iter()
            .zip(&self.on_boundary)
            .map(|(&v, &b)| if b { 3 - v as i64 } else { 4 - v as i64 })
            .sum()
    }
}

/// The cage's connectivity, built once per operation
struct Mesh {
    /// Edges as (smaller, larger) in the order of their first appearance walking the faces
    edges: Vec<(u32, u32)>,
    /// Per edge: its faces, in index order (one or two)
    edge_faces: Vec<Vec<usize>>,
    /// Per vertex: its faces in index order, and its edges in first-appearance order
    vert_faces: Vec<Vec<usize>>,
    vert_edges: Vec<Vec<usize>>,
    /// Per face: its four edges, edge i joining corner i and corner i + 1
    face_edges: Vec<[usize; 4]>,
}

/// How a vertex moves: by the smooth rule, along a crease between two vertices, or not at all
enum Kind {
    Smooth,
    Crease(u32, u32),
    Corner,
}

fn build(cage: &Cage) -> Result<Mesh, CageError> {
    if cage.faces.is_empty() {
        return Err(CageError::Empty);
    }
    let n = cage.positions.len();
    let mut directed: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut index: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut m = Mesh { edges: Vec::new(), edge_faces: Vec::new(), vert_faces: vec![Vec::new(); n], vert_edges: vec![Vec::new(); n], face_edges: Vec::with_capacity(cage.faces.len()) };
    for (fi, f) in cage.faces.iter().enumerate() {
        if f.iter().any(|&v| v as usize >= n) {
            return Err(CageError::VertexOutOfRange { face: fi });
        }
        if (0..4).any(|i| (i + 1..4).any(|j| f[i] == f[j])) {
            return Err(CageError::DegenerateFace { face: fi });
        }
        let mut fe = [0usize; 4];
        for i in 0..4 {
            let (a, b) = (f[i], f[(i + 1) % 4]);
            if directed.insert((a, b), fi).is_some() {
                return Err(CageError::InconsistentWinding { edge: (a, b) });
            }
            let key = (a.min(b), a.max(b));
            let e = match index.get(&key) {
                Some(&e) => e,
                None => {
                    let e = m.edges.len();
                    index.insert(key, e);
                    m.edges.push(key);
                    m.edge_faces.push(Vec::new());
                    m.vert_edges[key.0 as usize].push(e);
                    m.vert_edges[key.1 as usize].push(e);
                    e
                }
            };
            m.edge_faces[e].push(fi);
            if m.edge_faces[e].len() > 2 {
                return Err(CageError::NonManifoldEdge { edge: key });
            }
            fe[i] = e;
            m.vert_faces[f[i] as usize].push(fi);
        }
        m.face_edges.push(fe);
    }
    if let Some(&edge) = cage.sharpness.keys().find(|k| !index.contains_key(k)) {
        return Err(CageError::SharpnessOnMissingEdge { edge });
    }
    Ok(m)
}

impl Mesh {
    fn hard(&self, cage: &Cage, e: usize) -> bool {
        self.edge_faces[e].len() == 1 || cage.sharpness.get(&self.edges[e]).is_some_and(|&s| s >= 1)
    }

    fn other(&self, e: usize, v: usize) -> u32 {
        let (a, b) = self.edges[e];
        if a as usize == v {
            b
        } else {
            a
        }
    }

    fn kind(&self, cage: &Cage, v: usize) -> Kind {
        let hard: Vec<usize> = self.vert_edges[v].iter().copied().filter(|&e| self.hard(cage, e)).collect();
        let boundary = self.vert_edges[v].iter().any(|&e| self.edge_faces[e].len() == 1);
        match hard.len() {
            // (a dart, one sharp edge ending here, moves by the smooth rule)
            0 | 1 => Kind::Smooth,
            2 if !(boundary && self.vert_edges[v].len() == 2) => Kind::Crease(self.other(hard[0], v), self.other(hard[1], v)),
            _ => Kind::Corner,
        }
    }
}

/// Checks the cage: indices in range, no repeated vertex in a face, every edge used by at most
/// two faces and, when by two, in opposite directions, every sharpness entry on an existing edge
pub fn check(cage: &Cage) -> Result<Topology, CageError> {
    let m = build(cage)?;
    let on_boundary: Vec<bool> = m.vert_edges.iter().map(|es| es.iter().any(|&e| m.edge_faces[e].len() == 1)).collect();
    Ok(Topology {
        vertices: cage.positions.len(),
        edges: m.edges.len(),
        faces: cage.faces.len(),
        boundary_edges: m.edge_faces.iter().filter(|f| f.len() == 1).count(),
        valence: m.vert_edges.iter().map(Vec::len).collect(),
        on_boundary,
    })
}

/// One Catmull–Clark step (see the module documentation). The output's vertices are, in order,
/// the input's vertices (moved), one per edge (in the order of the edges' first appearance
/// walking the faces), then one per face; each input face becomes four faces
pub fn subdivide(cage: &Cage) -> Result<Cage, CageError> {
    let m = build(cage)?;
    let p = &cage.positions;
    let (nv, ne) = (p.len(), m.edges.len());
    let face_pt: Vec<DVec3> = cage.faces.iter().map(|f| (p[f[0] as usize] + p[f[1] as usize] + p[f[2] as usize] + p[f[3] as usize]) * 0.25).collect();
    let edge_pt: Vec<DVec3> = (0..ne)
        .map(|e| {
            let (a, b) = m.edges[e];
            let (a, b) = (p[a as usize], p[b as usize]);
            if m.hard(cage, e) {
                (a + b) * 0.5
            } else {
                (a + b + face_pt[m.edge_faces[e][0]] + face_pt[m.edge_faces[e][1]]) * 0.25
            }
        })
        .collect();
    let mut positions = Vec::with_capacity(nv + ne + cage.faces.len());
    for v in 0..nv {
        positions.push(match m.kind(cage, v) {
            Kind::Corner => p[v],
            Kind::Crease(a, b) => (p[a as usize] + p[v] * 6.0 + p[b as usize]) * 0.125,
            Kind::Smooth => {
                let n = m.vert_edges[v].len() as f64;
                let fbar = m.vert_faces[v].iter().fold(DVec3::ZERO, |s, &f| s + face_pt[f]) / m.vert_faces[v].len() as f64;
                let ebar = m.vert_edges[v].iter().fold(DVec3::ZERO, |s, &e| s + (p[v] + p[m.other(e, v) as usize]) * 0.5) / n;
                (fbar + ebar * 2.0 + p[v] * (n - 3.0)) / n
            }
        });
    }
    positions.extend(edge_pt);
    positions.extend(face_pt);
    let mut faces = Vec::with_capacity(4 * cage.faces.len());
    for (fi, f) in cage.faces.iter().enumerate() {
        let fe = m.face_edges[fi];
        let centre = (nv + ne + fi) as u32;
        for i in 0..4 {
            faces.push([f[i], (nv + fe[i]) as u32, centre, (nv + fe[(i + 3) % 4]) as u32]);
        }
    }
    let mut sharpness = BTreeMap::new();
    for (&(a, b), &s) in &cage.sharpness {
        let next = if s == INFINITE { INFINITE } else { s.saturating_sub(1) };
        if next == 0 {
            continue;
        }
        let e = (nv + m.edges.iter().position(|&k| k == (a, b)).expect("checked by build")) as u32;
        sharpness.insert((a, e), next);
        sharpness.insert((b, e), next);
    }
    Ok(Cage { positions, faces, sharpness })
}

/// Each vertex's point on the limit surface (see the module documentation)
pub fn limit_positions(cage: &Cage) -> Result<Vec<DVec3>, CageError> {
    let m = build(cage)?;
    let p = &cage.positions;
    Ok((0..p.len())
        .map(|v| match m.kind(cage, v) {
            Kind::Corner => p[v],
            Kind::Crease(a, b) => (p[a as usize] + p[v] * 4.0 + p[b as usize]) / 6.0,
            Kind::Smooth => {
                let n = m.vert_edges[v].len() as f64;
                let se = m.vert_edges[v].iter().fold(DVec3::ZERO, |s, &e| s + p[m.other(e, v) as usize]);
                let sf = m.vert_faces[v].iter().fold(DVec3::ZERO, |s, &f| {
                    let face = cage.faces[f];
                    let i = face.iter().position(|&x| x as usize == v).expect("a vertex's face holds it");
                    s + p[face[(i + 2) % 4] as usize]
                });
                (p[v] * (n * n) + se * 4.0 + sf) / (n * (n + 5.0))
            }
        })
        .collect())
}

/// Each vertex's unit normal on the limit surface by the tangent masks (see the module doc): a
/// smooth interior vertex (every edge with two faces, none sharp) gets its normal, on the side of
/// its faces' winding; a crease, corner, dart or boundary vertex gets none
pub fn limit_normals(cage: &Cage) -> Result<Vec<Option<DVec3>>, CageError> {
    let m = build(cage)?;
    let p = &cage.positions;
    let corner = |f: usize, v: usize| -> (u32, u32, u32) {
        let face = cage.faces[f];
        let i = face.iter().position(|&x| x as usize == v).expect("a vertex's face holds it");
        (face[(i + 1) % 4], face[(i + 2) % 4], face[(i + 3) % 4])
    };
    Ok((0..p.len())
        .map(|v| {
            if m.vert_edges[v].iter().any(|&e| m.edge_faces[e].len() != 2 || m.hard(cage, e)) {
                return None;
            }
            // the ring in cyclic order: face k has corners (v, next, diagonal, prev), and the face
            // after it is the one whose next is this one's prev
            let faces = &m.vert_faces[v];
            let n = faces.len();
            let mut order = vec![faces[0]];
            while order.len() < n {
                let (_, _, prev) = corner(*order.last().expect("a face"), v);
                let f = faces.iter().copied().find(|&f| corner(f, v).0 == prev)?;
                order.push(f);
            }
            let nf = n as f64;
            let w = std::f64::consts::TAU / nf;
            let a = 1.0 + cos64(w) + cos64(w / 2.0) * sqrt64(2.0 * (9.0 + cos64(w)));
            let (mut t1, mut t2) = (DVec3::ZERO, DVec3::ZERO);
            for (i, &f) in order.iter().enumerate() {
                let (e, d, _) = corner(f, v);
                let (ci, cj, si, sj) = (cos64(w * i as f64), cos64(w * (i + 1) as f64), sin64(w * i as f64), sin64(w * (i + 1) as f64));
                t1 += p[e as usize] * (a * ci) + p[d as usize] * (ci + cj);
                t2 += p[e as usize] * (a * si) + p[d as usize] * (si + sj);
            }
            // (the ring runs in the faces' winding, so t₁ × t₂ is on the winding's side)
            let nn = t1.cross(t2);
            (nn.length_squared() > 0.0).then(|| nn.normalize())
        })
        .collect())
}

/// The limit surface sampled `levels` steps down: the cage subdivided `levels` times, each
/// vertex moved to its limit point; the faces are the subdivided cage's
pub fn limit_mesh(cage: &Cage, levels: u32) -> Result<Cage, CageError> {
    let mut s = cage.clone();
    build(&s)?;
    for _ in 0..levels {
        s = subdivide(&s)?;
    }
    let positions = limit_positions(&s)?;
    Ok(Cage { positions, faces: s.faces, sharpness: s.sharpness })
}

#[cfg(test)]
mod tests {
    use super::*;

    // an n × m grid of vertices over [0, n−1] × [0, m−1] with heights z(i, j), faces wound +z
    fn grid(n: usize, m: usize, z: &dyn Fn(usize, usize) -> f64) -> Cage {
        let mut positions = Vec::new();
        for j in 0..m {
            for i in 0..n {
                positions.push(DVec3::new(i as f64, j as f64, z(i, j)));
            }
        }
        let id = |i: usize, j: usize| (j * n + i) as u32;
        let mut faces = Vec::new();
        for j in 0..m - 1 {
            for i in 0..n - 1 {
                faces.push([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
            }
        }
        Cage { positions, faces, sharpness: BTreeMap::new() }
    }

    // a fixed bumpy height field (no symmetry the rules could hide behind)
    fn bumps(i: usize, j: usize) -> f64 {
        let (x, y) = (i as f64, j as f64);
        0.3 * x - 0.2 * y + 0.05 * x * y + 0.01 * x * x * y - 0.02 * y * y + if (i * 7 + j * 3) % 5 == 0 { 0.4 } else { 0.0 }
    }

    // oracle: the cubic B-spline surface reproduces a quadratic up to a constant, so on a regular
    // grid of z = a(x² + y²) the limit normal at an interior vertex is ∝ (−2ax, −2ay, 1) exactly
    #[test]
    fn limit_normals_on_a_paraboloid_grid_are_the_closed_form() {
        let a = 0.03;
        let c = grid(9, 9, &|i, j| a * ((i as f64).powi(2) + (j as f64).powi(2)));
        let nrm = limit_normals(&c).unwrap();
        let mut checked = 0;
        for j in 1..8 {
            for i in 1..8 {
                let (x, y) = (i as f64, j as f64);
                let want = DVec3::new(-2.0 * a * x, -2.0 * a * y, 1.0).normalize();
                let got = nrm[j * 9 + i].expect("an interior vertex has a normal");
                assert!((got - want).length() < 1e-12, "({i}, {j}): {got} vs {want}");
                checked += 1;
            }
        }
        assert_eq!(checked, 49);
        assert!(nrm[0].is_none() && nrm[4].is_none(), "boundary vertices have none");
    }

    // oracle: the cube's symmetry: the limit normal at a corner (valence 3) is the corner's direction
    #[test]
    fn limit_normals_at_the_cube_corners_are_the_diagonals() {
        let c = cube();
        let nrm = limit_normals(&c).unwrap();
        for (v, q) in c.positions.iter().enumerate() {
            let got = nrm[v].expect("a closed cube's vertices are smooth");
            assert!((got - q.normalize()).length() < 1e-12, "{v}: {got} vs {}", q.normalize());
        }
    }

    // oracle: valence 5 and 6 (extraordinary vertices): the normal of the limit surface from its
    // own points six levels down (the plane through the vertex's limit neighbours) agrees with the
    // mask normal to within the sampling's own error
    #[test]
    fn limit_normals_at_extraordinary_vertices_match_the_deep_limit() {
        for n in [5usize, 6] {
            // a fan of n quads about a centre, with a bumpy outer ring (a disc, the centre interior)
            let mut positions = vec![DVec3::new(0.0, 0.0, 0.2)];
            for k in 0..n {
                let t = std::f64::consts::TAU * k as f64 / n as f64;
                // a first-harmonic tilt on each ring, different on the two (the masks weigh them)
                positions.push(DVec3::new(t.cos(), t.sin(), 0.1 * (2.0 * t).sin() + 0.3 * t.cos()));
                let t2 = std::f64::consts::TAU * (k as f64 + 0.5) / n as f64;
                positions.push(DVec3::new(1.6 * t2.cos(), 1.6 * t2.sin(), 0.15 * (3.0 * t2).cos() - 0.2 * t2.sin()));
            }
            let e = |k: usize| (1 + 2 * (k % n)) as u32;
            let f = |k: usize| (2 + 2 * (k % n)) as u32;
            let faces: Vec<[u32; 4]> = (0..n).map(|k| [0, e(k), f(k), e(k + 1)]).collect();
            let c = Cage { positions, faces, sharpness: BTreeMap::new() };
            let want = limit_normals(&c).unwrap()[0].expect("the centre is smooth and interior");
            let deep = limit_mesh(&c, 6).unwrap();
            // the centre keeps index 0; its neighbours in the deep mesh lie about it in a small ring
            let nb: Vec<DVec3> = deep.faces.iter().filter(|q| q.contains(&0)).flat_map(|q| q.iter().map(|&i| deep.positions[i as usize])).collect();
            let o = deep.positions[0];
            let mut acc = DVec3::ZERO;
            let ring: Vec<DVec3> = deep.faces.iter().filter_map(|q| q.iter().position(|&i| i == 0).map(|k| (deep.positions[q[(k + 1) % 4] as usize], deep.positions[q[(k + 3) % 4] as usize]))).map(|(a, b)| (a - o).cross(b - o)).collect();
            for r in &ring {
                acc += *r;
            }
            assert_eq!(nb.len(), 4 * n);
            let fd = acc.normalize();
            let ang = fd.dot(want).clamp(-1.0, 1.0).acos().to_degrees();
            assert!(ang < 0.05, "valence {n}: mask normal vs the deep limit's {ang}°");
        }
    }

    // the cube of half-width 1, faces wound outward
    fn cube() -> Cage {
        let p = |x: f64, y: f64, z: f64| DVec3::new(x, y, z);
        let positions = vec![p(-1.0, -1.0, -1.0), p(1.0, -1.0, -1.0), p(1.0, 1.0, -1.0), p(-1.0, 1.0, -1.0), p(-1.0, -1.0, 1.0), p(1.0, -1.0, 1.0), p(1.0, 1.0, 1.0), p(-1.0, 1.0, 1.0)];
        let faces = vec![[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]];
        Cage { positions, faces, sharpness: BTreeMap::new() }
    }

    // a closed "n-gon prism" of quads around a valence-n vertex: n quads fanned around a centre,
    // their outer ring closed by a second fan around an opposite centre (a closed surface, both
    // centres of valence n)
    fn double_fan(n: usize) -> Cage {
        let tau = std::f64::consts::TAU;
        let mut positions = vec![DVec3::new(0.0, 0.0, 1.0), DVec3::new(0.0, 0.0, -1.0)];
        for k in 0..2 * n {
            let a = tau * k as f64 / (2 * n) as f64;
            let r = if k % 2 == 0 { 1.0 } else { 1.3 };
            positions.push(DVec3::new(r * a.cos(), r * a.sin(), 0.1 * (k % 3) as f64));
        }
        let ring = |k: usize| (2 + k % (2 * n)) as u32;
        let mut faces = Vec::new();
        for k in 0..n {
            faces.push([0, ring(2 * k), ring(2 * k + 1), ring(2 * k + 2)]);
            faces.push([1, ring(2 * k + 2), ring(2 * k + 1), ring(2 * k)]);
        }
        Cage { positions, faces, sharpness: BTreeMap::new() }
    }

    fn close(a: DVec3, b: DVec3, tol: f64) -> bool {
        (a - b).abs().max_element() <= tol
    }

    // oracle (the uniform bicubic B-spline): on a regular grid the limit surface is the tensor
    // B-spline of the grid; at a knot its value is Σ wᵢ wⱼ P / 36 with w = (1, 4, 1), and at an edge
    // midpoint (one step down) the weights along that direction are (1, 23, 23, 1) / 48
    #[test]
    fn a_regular_grid_is_the_bicubic_b_spline() {
        let c = grid(9, 9, &bumps);
        let lim = limit_positions(&c).unwrap();
        let w = [1.0, 4.0, 1.0];
        for &(i, j) in &[(3usize, 3usize), (4, 5), (5, 2)] {
            let mut want = DVec3::ZERO;
            for (a, wa) in w.iter().enumerate() {
                for (b, wb) in w.iter().enumerate() {
                    want += c.positions[(j + b - 1) * 9 + i + a - 1] * (wa * wb / 36.0);
                }
            }
            assert!(close(lim[j * 9 + i], want, 1e-13), "knot ({i}, {j}): {} vs {want}", lim[j * 9 + i]);
        }
        // one step down: the point between (4, 4) and (5, 4) is the edge point of that edge; its
        // limit is the B-spline at u = 4.5 (weights (1, 23, 23, 1)/48 over i = 3..6), v = 4
        let s = subdivide(&c).unwrap();
        let ls = limit_positions(&s).unwrap();
        let (pa, pb) = (c.positions[4 * 9 + 4], c.positions[4 * 9 + 5]);
        let mid = (pa + pb) * 0.5;
        let k = s.positions.iter().enumerate().filter(|(k, _)| *k >= c.positions.len()).find(|(_, p)| close(DVec3::new(p.x, p.y, 0.0), DVec3::new(mid.x, mid.y, 0.0), 1e-12)).map(|(k, _)| k).expect("the edge point of (4, 4)–(5, 4)");
        let wu = [1.0, 23.0, 23.0, 1.0];
        let mut want = DVec3::ZERO;
        for (a, wa) in wu.iter().enumerate() {
            for (b, wb) in w.iter().enumerate() {
                want += c.positions[(3 + b) * 9 + 3 + a] * (wa / 48.0 * wb / 6.0);
            }
        }
        assert!(close(ls[k], want, 1e-13), "edge midpoint: {} vs {want}", ls[k]);
    }

    // oracle (two independent paths): at an extraordinary vertex the limit mask equals the point
    // the vertex converges to under repeated subdivision. The distance shrinks per step by at most
    // the subdivision matrix's subdominant eigenvalue, in closed form for Catmull–Clark
    // λ(n) = (5 + cos(2π/n) + cos(π/n) √(2 (9 + cos(2π/n)))) / 16 (λ(4) = 1/2, λ(3) ≈ 0.410,
    // λ(5) ≈ 0.550, λ(6) ≈ 0.580); valences 3 (the cube), 5 and 6
    #[test]
    fn the_limit_mask_is_where_subdivision_converges() {
        for (name, c, v, n) in [("cube", cube(), 6usize, 3.0f64), ("fan 5", double_fan(5), 0, 5.0), ("fan 6", double_fan(6), 0, 6.0)] {
            let (c2, c1) = ((std::f64::consts::TAU / n).cos(), (std::f64::consts::PI / n).cos());
            let lambda = (5.0 + c2 + c1 * (2.0 * (9.0 + c2)).sqrt()) / 16.0;
            let target = limit_positions(&c).unwrap()[v];
            let d0 = (c.positions[v] - target).length();
            assert!(d0 > 1e-3, "{name}: the cage vertex is off its limit point");
            let mut s = c.clone();
            let mut d = d0;
            for level in 1..=6 {
                s = subdivide(&s).unwrap();
                let next = (s.positions[v] - target).length();
                assert!(next <= d * lambda * 1.05 + 1e-14, "{name} level {level}: {next} after {d} (λ {lambda:.4})");
                d = next;
            }
            assert!(d <= d0 * (lambda * 1.05).powi(6) + 1e-14, "{name}: {d} after 6 steps from {d0}");
        }
    }

    // oracle: the cube's corner (valence 3) has the mask (9 v + 4 Σ e + Σ f) / 24 written out:
    // v = (1, 1, 1), the three edge neighbours sum to (1, 1, 1), the three face diagonals to
    // (−1, −1, −1), so the limit is (9 + 4 − 1) / 24 = 1/2 along each axis
    #[test]
    fn the_cube_corner_limit() {
        let lim = limit_positions(&cube()).unwrap();
        assert!(close(lim[6], DVec3::splat(0.5), 1e-15), "{}", lim[6]);
        assert!(close(lim[0], DVec3::splat(-0.5), 1e-15), "{}", lim[0]);
    }

    // oracle (the cubic B-spline curve): an open grid's boundary is the B-spline curve of its
    // boundary vertices, independent of the interior: limit (a + 4 v + b) / 6, one step
    // (a + 6 v + b) / 8 and midpoints (v₀ + v₁) / 2; moving an interior vertex leaves it unchanged
    #[test]
    fn a_boundary_is_the_b_spline_curve_of_its_vertices() {
        let c = grid(7, 5, &bumps);
        let lim = limit_positions(&c).unwrap();
        for i in 1..6 {
            let (a, v, b) = (c.positions[i - 1], c.positions[i], c.positions[i + 1]);
            assert!(close(lim[i], (a + 4.0 * v + b) / 6.0, 1e-14), "limit at {i}");
        }
        let s = subdivide(&c).unwrap();
        for i in 1..6 {
            let (a, v, b) = (c.positions[i - 1], c.positions[i], c.positions[i + 1]);
            assert!(close(s.positions[i], (a + 6.0 * v + b) / 8.0, 1e-14), "step at {i}");
        }
        let mut moved = c.clone();
        moved.positions[2 * 7 + 3] += DVec3::new(0.3, -0.2, 1.7);
        let lm = limit_positions(&subdivide(&moved).unwrap()).unwrap();
        let l0 = limit_positions(&s).unwrap();
        let on_edge: Vec<usize> = (0..s.positions.len()).filter(|&k| s.positions[k].y == 0.0).collect();
        assert_eq!(on_edge.len(), 13, "the subdivided boundary row: 7 vertices and 6 edge points");
        for k in on_edge {
            assert_eq!(lm[k], l0[k], "boundary point {k} of the subdivided grid moved");
        }
    }

    // oracle: an infinitely sharp crease along an interior grid line is the B-spline curve of its
    // vertices, whatever the vertices beside it (they are moved and the crease stays)
    #[test]
    fn an_infinite_crease_is_the_b_spline_of_its_edge() {
        let mut c = grid(7, 7, &bumps);
        let row = 3;
        for i in 0..6 {
            let (a, b) = ((row * 7 + i) as u32, (row * 7 + i + 1) as u32);
            c.sharpness.insert((a.min(b), a.max(b)), INFINITE);
        }
        let lim = limit_positions(&c).unwrap();
        for i in 1..6 {
            let k = row * 7 + i;
            let want = (c.positions[k - 1] + 4.0 * c.positions[k] + c.positions[k + 1]) / 6.0;
            assert!(close(lim[k], want, 1e-14), "crease limit at {i}: {} vs {want}", lim[k]);
        }
        let mut moved = c.clone();
        moved.positions[2 * 7 + 3].z += 2.0;
        moved.positions[4 * 7 + 2].z -= 1.5;
        let (a, b) = (limit_mesh(&c, 2).unwrap(), limit_mesh(&moved, 2).unwrap());
        // the crease's own vertices keep their limit points at every level
        for i in 1..6 {
            let k = row * 7 + i;
            assert_eq!(a.positions[k], b.positions[k], "crease vertex {i}");
        }
        // and its sharpness survives the steps
        assert!(subdivide(&c).unwrap().sharpness.values().all(|&s| s == INFINITE));
    }

    // a semi-sharp crease acts at integer steps: sharpness s becomes s − 1 on both halves, and an
    // edge of sharpness 1 is smooth after one step
    #[test]
    fn semi_sharpness_decrements_per_step() {
        let mut c = grid(5, 5, &bumps);
        c.sharpness.insert((11, 12), 2);
        let s1 = subdivide(&c).unwrap();
        assert_eq!(s1.sharpness.len(), 2);
        assert!(s1.sharpness.values().all(|&s| s == 1), "{:?}", s1.sharpness);
        let s2 = subdivide(&s1).unwrap();
        assert!(s2.sharpness.is_empty(), "{:?}", s2.sharpness);
    }

    // Euler and the valence defect: Σ_int (4 − val) + Σ_bdy (3 − val) = 4 χ on a quad mesh
    #[test]
    fn euler_and_valence_defect() {
        let t = check(&cube()).unwrap();
        assert_eq!((t.vertices, t.edges, t.faces, t.boundary_edges), (8, 12, 6, 0));
        assert_eq!(t.euler(), 2);
        assert_eq!(t.valence_defect(), 8);
        let g = check(&grid(4, 3, &|_, _| 0.0)).unwrap();
        assert_eq!(g.euler(), 1);
        assert_eq!(g.valence_defect(), 4);
        for n in [3usize, 5, 6] {
            let f = check(&double_fan(n)).unwrap();
            assert_eq!(f.euler(), 2, "fan {n}");
            assert_eq!(f.valence_defect(), 8, "fan {n}");
            assert_eq!((f.valence[0], f.valence[1]), (n, n));
        }
        // a subdivision step keeps χ and the defect
        let s = check(&subdivide(&double_fan(5)).unwrap()).unwrap();
        assert_eq!((s.euler(), s.valence_defect()), (2, 8));
    }

    // refused cages: a flipped face (a directed edge used twice), an edge of three faces, an index
    // out of range, a repeated vertex, a sharpness on no edge, no face
    #[test]
    fn invalid_cages_are_refused() {
        let mut c = cube();
        c.faces[1] = [4, 7, 6, 5];
        assert!(matches!(check(&c), Err(CageError::InconsistentWinding { .. })));
        let mut c = grid(3, 2, &|_, _| 0.0);
        c.positions.push(DVec3::new(0.5, -1.0, 0.0));
        c.positions.push(DVec3::new(1.5, -1.0, 0.0));
        c.faces.push([1, 0, 6, 7]);
        c.faces.push([0, 1, 7, 6]);
        assert!(check(&c).is_err());
        let mut c = cube();
        c.faces[0][2] = 99;
        assert_eq!(check(&c), Err(CageError::VertexOutOfRange { face: 0 }));
        let mut c = cube();
        c.faces[0][2] = 0;
        assert_eq!(check(&c), Err(CageError::DegenerateFace { face: 0 }));
        let mut c = cube();
        c.sharpness.insert((0, 6), 1);
        assert_eq!(check(&c), Err(CageError::SharpnessOnMissingEdge { edge: (0, 6) }));
        let c = Cage { positions: vec![DVec3::ZERO], faces: Vec::new(), sharpness: BTreeMap::new() };
        assert_eq!(check(&c), Err(CageError::Empty));
        // the operations refuse what check refuses
        let mut c = cube();
        c.faces[1] = [4, 7, 6, 5];
        assert!(subdivide(&c).is_err() && limit_positions(&c).is_err() && limit_mesh(&c, 1).is_err());
    }

    // a rebuild is bit-identical, and the counts follow V' = V + E + F, F' = 4 F
    #[test]
    fn deterministic_and_counted() {
        let c = double_fan(6);
        let t = check(&c).unwrap();
        let (a, b) = (limit_mesh(&c, 3).unwrap(), limit_mesh(&c, 3).unwrap());
        assert_eq!(a, b);
        let s = subdivide(&c).unwrap();
        assert_eq!(s.positions.len(), t.vertices + t.edges + t.faces);
        assert_eq!(s.faces.len(), 4 * t.faces);
        let bits: Vec<u64> = a.positions.iter().flat_map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).collect();
        let bits2: Vec<u64> = b.positions.iter().flat_map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).collect();
        assert_eq!(bits, bits2);
    }
}
