//! Rigid-body mass properties of a closed, outward-wound triangle mesh
//! (unit density), by the divergence theorem over the tetrahedra spanned by
//! each triangle and the origin — D. Eberly, "Polyhedral Mass Properties
//! (Revisited)", 2002. Closed form, no sampling: exact for the polyhedron the
//! mesh describes, so the only error against the SDF is the meshing error.
//!
//! Shared by `examples/bake_assets` and `tests/test_bake_mass_oracle.rs`
//! (included there with `#[path]`), so the oracle checks the code the bake
//! runs.
//!
//! Author: Moroya Sakamoto

/// Volume, centre of mass and inertia tensor about the centre of mass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassProps {
    /// Enclosed volume (= mass at unit density).
    pub volume: f64,
    /// Centre of mass.
    pub centroid: [f64; 3],
    /// Inertia tensor about the centre of mass, row-major
    /// (`[[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]`, products of
    /// inertia carry the conventional minus sign).
    pub inertia: [[f64; 3]; 3],
}

#[inline]
fn subexpressions(w0: f64, w1: f64, w2: f64) -> (f64, f64, f64, f64, f64, f64) {
    let temp0 = w0 + w1;
    let f1 = temp0 + w2;
    let temp1 = w0 * w0;
    let temp2 = temp1 + w1 * temp0;
    let f2 = temp2 + w2 * f1;
    let f3 = w0 * temp1 + w1 * temp2 + w2 * f2;
    let g0 = f2 + w0 * (f1 + w0);
    let g1 = f2 + w1 * (f1 + w1);
    let g2 = f2 + w2 * (f1 + w2);
    (f1, f2, f3, g0, g1, g2)
}

/// Mass properties of the polyhedron bounded by `triangles` (each `[p0, p1,
/// p2]` counter-clockwise seen from outside). Returns `None` for a
/// non-positive volume (empty, open-and-inverted or inside-out input).
pub fn mass_properties<I>(triangles: I) -> Option<MassProps>
where
    I: IntoIterator<Item = [[f64; 3]; 3]>,
{
    // ∫1, ∫x, ∫y, ∫z, ∫x², ∫y², ∫z², ∫xy, ∫yz, ∫zx
    let mut s = [0.0f64; 10];
    for [p0, p1, p2] in triangles {
        let (x0, y0, z0) = (p0[0], p0[1], p0[2]);
        let (x1, y1, z1) = (p1[0], p1[1], p1[2]);
        let (x2, y2, z2) = (p2[0], p2[1], p2[2]);
        let (a1, b1, c1) = (x1 - x0, y1 - y0, z1 - z0);
        let (a2, b2, c2) = (x2 - x0, y2 - y0, z2 - z0);
        let d0 = b1 * c2 - b2 * c1;
        let d1 = a2 * c1 - a1 * c2;
        let d2 = a1 * b2 - a2 * b1;

        let (f1x, f2x, f3x, g0x, g1x, g2x) = subexpressions(x0, x1, x2);
        let (_, f2y, f3y, g0y, g1y, g2y) = subexpressions(y0, y1, y2);
        let (_, f2z, f3z, g0z, g1z, g2z) = subexpressions(z0, z1, z2);

        s[0] += d0 * f1x;
        s[1] += d0 * f2x;
        s[2] += d1 * f2y;
        s[3] += d2 * f2z;
        s[4] += d0 * f3x;
        s[5] += d1 * f3y;
        s[6] += d2 * f3z;
        s[7] += d0 * (y0 * g0x + y1 * g1x + y2 * g2x);
        s[8] += d1 * (z0 * g0y + z1 * g1y + z2 * g2y);
        s[9] += d2 * (x0 * g0z + x1 * g1z + x2 * g2z);
    }
    let mult = [
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 24.0,
        1.0 / 24.0,
        1.0 / 60.0,
        1.0 / 60.0,
        1.0 / 60.0,
        1.0 / 120.0,
        1.0 / 120.0,
        1.0 / 120.0,
    ];
    for (v, m) in s.iter_mut().zip(mult) {
        *v *= m;
    }
    let mass = s[0];
    if mass.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let c = [s[1] / mass, s[2] / mass, s[3] / mass];
    let ixx = s[5] + s[6] - mass * (c[1] * c[1] + c[2] * c[2]);
    let iyy = s[4] + s[6] - mass * (c[2] * c[2] + c[0] * c[0]);
    let izz = s[4] + s[5] - mass * (c[0] * c[0] + c[1] * c[1]);
    let ixy = -(s[7] - mass * c[0] * c[1]);
    let iyz = -(s[8] - mass * c[1] * c[2]);
    let ixz = -(s[9] - mass * c[2] * c[0]);
    Some(MassProps {
        volume: mass,
        centroid: c,
        inertia: [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
    })
}
