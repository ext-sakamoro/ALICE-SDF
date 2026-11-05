//! A stack of planar closed-loop sections along an axis (local Y), any topology per level.
//!
//! Each level `i` holds one or more closed loops in the (x, z) plane at height `levels[i]`. The 2D
//! signed distance `d_i(x, z)` takes the nearest edge of all the level's loops and the sign by even-odd
//! crossing parity over them, so a level may hold several separate loops, loops inside loops, or
//! loops that merge or split from one level to the next. Between two levels the field interpolates
//! the two 2D distances linearly in y; beyond the first and last level it is combined with the
//! distance past the end like an extrusion. The result is divided by a Lipschitz bound
//! `L = sqrt(1 + G²)` computed at construction, `G` being the largest ratio of the Hausdorff distance
//! between two neighbouring levels' boundaries to their spacing, so the field never overstates the
//! distance to its own zero set.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// Sampling step (m) of the boundaries when bounding the Hausdorff distance between two levels; the
/// bound adds half of it, so it is an upper bound, never an estimate
pub const SECTION_STACK_SAMPLE: f32 = 0.0005;

/// A stack of planar closed-loop sections along local Y
///
/// Loading recomputes the bound from the levels and loops and refuses a stored `lipschitz` that
/// differs from it: a smaller stored value would overstate the distance and let sphere tracing step
/// through the surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SectionStackStored")]
pub struct SectionStack {
    /// Heights of the levels (strictly increasing, at least 2)
    pub levels: Vec<f32>,
    /// Per level, its closed loops in the (x, z) plane (each at least 3 vertices)
    pub loops: Vec<Vec<Vec<Vec2>>>,
    /// The Lipschitz bound of the interpolated field (≥ 1), computed at construction
    pub lipschitz: f32,
}

/// The stored form of a [`SectionStack`], checked on load
#[derive(Deserialize)]
struct SectionStackStored {
    levels: Vec<f32>,
    loops: Vec<Vec<Vec<Vec2>>>,
    lipschitz: f32,
}

impl TryFrom<SectionStackStored> for SectionStack {
    type Error = String;
    fn try_from(v: SectionStackStored) -> Result<Self, String> {
        let s = Self::new(v.levels, v.loops)?;
        if s.lipschitz.to_bits() != v.lipschitz.to_bits() {
            return Err(format!(
                "section stack: stored lipschitz {} differs from the bound {} recomputed from its levels",
                v.lipschitz, s.lipschitz
            ));
        }
        Ok(s)
    }
}

/// The 2D signed distance to a set of closed loops: the nearest edge of any loop, negative inside an
/// odd number of loops (even-odd rule)
#[must_use]
pub fn sdf_loops_2d(q: Vec2, loops: &[Vec<Vec2>]) -> f32 {
    let mut d = f32::INFINITY;
    let mut s = 1.0_f32;
    for v in loops {
        let n = v.len();
        let mut j = n - 1;
        for i in 0..n {
            let e = v[j] - v[i];
            let w = q - v[i];
            let ee = e.dot(e);
            let b = if ee > 0.0 {
                w - e * (w.dot(e) / ee).clamp(0.0, 1.0)
            } else {
                w
            };
            d = d.law_min(b.dot(b));
            let c = [q.y >= v[i].y, q.y < v[j].y, e.x * w.y > e.y * w.x];
            if c.iter().all(|x| *x) || c.iter().all(|x| !*x) {
                s = -s;
            }
            j = i;
        }
    }
    s * d.sqrt()
}

/// The distance from a point to the nearest edge of a set of loops (unsigned)
fn edge_distance(q: Vec2, loops: &[Vec<Vec2>]) -> f32 {
    sdf_loops_2d(q, loops).abs()
}

/// An upper bound of the Hausdorff distance between the boundaries of two loop sets
///
/// Both boundaries are sampled every [`SECTION_STACK_SAMPLE`], each sample's exact distance to the other boundary, the
/// largest plus half the step (the distance function is 1-Lipschitz along an edge)
#[must_use]
pub fn boundary_hausdorff_bound(a: &[Vec<Vec2>], b: &[Vec<Vec2>]) -> f32 {
    let one_way = |x: &[Vec<Vec2>], y: &[Vec<Vec2>]| -> f32 {
        let mut m = 0.0_f32;
        for v in x {
            let n = v.len();
            for i in 0..n {
                let (p0, p1) = (v[i], v[(i + 1) % n]);
                let k = ((p1 - p0).length() / SECTION_STACK_SAMPLE).ceil().max(1.0) as usize;
                for t in 0..k {
                    let q = p0 + (p1 - p0) * (t as f32 / k as f32);
                    m = m.law_max(edge_distance(q, y));
                }
            }
        }
        m
    };
    one_way(a, b).law_max(one_way(b, a)) + 0.5 * SECTION_STACK_SAMPLE
}

impl SectionStack {
    /// A stack from its levels and loops; Err on fewer than 2 levels, levels not strictly
    /// increasing, a level without loops, a loop of fewer than 3 vertices or a non-finite number
    pub fn new(levels: Vec<f32>, loops: Vec<Vec<Vec<Vec2>>>) -> Result<Self, String> {
        if levels.len() < 2 || loops.len() != levels.len() {
            return Err(format!(
                "section stack: {} levels and {} loop sets (at least 2, equal)",
                levels.len(),
                loops.len()
            ));
        }
        if levels.iter().any(|y| !y.is_finite()) || levels.windows(2).any(|w| w[1] <= w[0]) {
            return Err("section stack: levels must be finite and strictly increasing".into());
        }
        for (i, set) in loops.iter().enumerate() {
            if set.is_empty() {
                return Err(format!("section stack: level {i} has no loop"));
            }
            for l in set {
                if l.len() < 3 || l.iter().any(|q| !q.is_finite()) {
                    return Err(format!(
                        "section stack: level {i} has a loop of {} vertices or a non-finite vertex",
                        l.len()
                    ));
                }
                if super::simple_loop::closed_polyline_crosses(l) {
                    return Err(format!("section stack: a loop at level {i} crosses itself"));
                }
            }
        }
        let g = (0..levels.len() - 1)
            .map(|i| {
                boundary_hausdorff_bound(&loops[i], &loops[i + 1]) / (levels[i + 1] - levels[i])
            })
            .fold(0.0_f32, f32::max);
        // a constant stack has H = the sampling margin only; its bound is exactly 1
        let g = if (0..levels.len() - 1).all(|i| loops[i] == loops[i + 1]) {
            0.0
        } else {
            g
        };
        Ok(Self {
            levels,
            loops,
            lipschitz: (1.0 + g * g).sqrt(),
        })
    }

    /// The bound the field was divided by
    #[must_use]
    pub const fn lipschitz(&self) -> f32 {
        self.lipschitz
    }

    /// The field at p (local coordinates, the stack's axis along Y)
    #[must_use]
    pub fn eval(&self, p: Vec3) -> f32 {
        let q = Vec2::new(p.x, p.z);
        let n = self.levels.len();
        let (y0, yn) = (self.levels[0], self.levels[n - 1]);
        let raw = if p.y < y0 || p.y > yn {
            let (d, h) = if p.y < y0 {
                (sdf_loops_2d(q, &self.loops[0]), y0 - p.y)
            } else {
                (sdf_loops_2d(q, &self.loops[n - 1]), p.y - yn)
            };
            d.law_max(h).law_min(0.0) + Vec2::new(d.law_max(0.0), h.law_max(0.0)).length()
        } else {
            let i = self.levels.partition_point(|&y| y <= p.y).clamp(1, n - 1) - 1;
            let t = (p.y - self.levels[i]) / (self.levels[i + 1] - self.levels[i]);
            let d = (1.0 - t) * sdf_loops_2d(q, &self.loops[i])
                + t * sdf_loops_2d(q, &self.loops[i + 1]);
            // inside the height range the end caps still bound it (the nearest cap plane)
            let h = (y0 - p.y).law_max(p.y - yn);
            d.law_max(h).law_min(0.0) + Vec2::new(d.law_max(0.0), h.law_max(0.0)).length()
        };
        raw / self.lipschitz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::sdf_polygon_2d_xy;
    use glam::{Vec2, Vec3};

    fn circle(c: Vec2, r: f32, n: usize) -> Vec<Vec2> {
        (0..n)
            .map(|k| {
                let a = std::f32::consts::TAU * k as f32 / n as f32;
                c + Vec2::new(r * a.cos(), r * a.sin())
            })
            .collect()
    }

    // a tiny deterministic generator for test points
    fn pts(n: usize, scale: f32) -> Vec<Vec3> {
        let mut s = 0x9e37_79b9_u32;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..n)
            .map(|_| Vec3::new(next(), next(), next()) * scale)
            .collect()
    }

    // oracle (closed form): a stack of one constant section between two levels is the polygon
    // extruded over that height, exactly; its bound is 1
    #[test]
    fn constant_stack_is_the_extruded_polygon() {
        let c = circle(Vec2::ZERO, 0.3, 64);
        let s = SectionStack::new(
            vec![-0.5, 0.0, 0.5],
            vec![vec![c.clone()], vec![c.clone()], vec![c.clone()]],
        )
        .unwrap();
        assert_eq!(s.lipschitz(), 1.0);
        for p in pts(400, 1.0) {
            let d2 = sdf_polygon_2d_xy(Vec2::new(p.x, p.z), &c);
            let want = crate::primitives::extrude_2d(d2, p.y.abs() - 0.5);
            assert!(
                (s.eval(p) - want).abs() < 1e-5,
                "{p}: {} vs {want}",
                s.eval(p)
            );
        }
    }

    // oracle (closed form): a frustum of circles r0 → r1 over height h; beside the lateral surface
    // the interpolated distance |q| − r(y), divided by sqrt(1 + G²) with G = |r1 − r0| / h, is the
    // exact distance to the cone there (the polygon is fine enough that its sag is below the tolerance)
    #[test]
    fn frustum_stack_is_the_cone_distance_at_its_side() {
        let (r0, r1, h, n) = (0.4f32, 0.2f32, 1.0f32, 512);
        let s = SectionStack::new(
            vec![0.0, h],
            vec![
                vec![circle(Vec2::ZERO, r0, n)],
                vec![circle(Vec2::ZERO, r1, n)],
            ],
        )
        .unwrap();
        let g = (r1 - r0).abs() / h;
        let sag = r0 * (1.0 - (std::f32::consts::PI / n as f32).cos());
        assert!(
            (s.lipschitz() - (1.0 + g * g).sqrt()).abs() < 2.0 * sag / h + 1e-4,
            "L {}",
            s.lipschitz()
        );
        for k in 0..50 {
            let y = 0.1 + 0.8 * k as f32 / 49.0;
            let r = r0 + (r1 - r0) * y / h;
            for off in [-0.05f32, -0.01, 0.01, 0.05] {
                let p = Vec3::new(r + off, y, 0.0);
                let exact = off / (1.0 + g * g).sqrt();
                assert!(
                    (s.eval(p) - exact).abs() < 3e-4,
                    "y {y} off {off}: {} vs {exact}",
                    s.eval(p)
                );
            }
        }
    }

    // oracle: two loops merging into one; the sign is the sign of the interpolated 2D distance
    // (computed here independently), the field times its bound is that distance, and the field is
    // 1-Lipschitz (no point pair changes faster than their distance)
    #[test]
    fn merging_loops_keep_the_sign_and_the_bound() {
        let two = vec![
            circle(Vec2::new(-0.15, 0.0), 0.1, 96),
            circle(Vec2::new(0.15, 0.0), 0.1, 96),
        ];
        let one = vec![circle(Vec2::ZERO, 0.28, 96)];
        let s = SectionStack::new(vec![0.0, 0.2], vec![two.clone(), one.clone()]).unwrap();
        assert!(s.lipschitz() > 1.0);
        let d2 = |loops: &[Vec<Vec2>], q: Vec2| -> f32 {
            // independent: nearest edge over all loops; inside = inside an odd number of loops
            let d = loops
                .iter()
                .map(|l| sdf_polygon_2d_xy(q, l).abs())
                .fold(f32::INFINITY, f32::min);
            let inside = loops
                .iter()
                .filter(|l| sdf_polygon_2d_xy(q, l) < 0.0)
                .count()
                % 2
                == 1;
            if inside {
                -d
            } else {
                d
            }
        };
        let ps = pts(2000, 0.4);
        let mut compared = 0;
        for p in &ps {
            if p.y <= 0.0 || p.y >= 0.2 {
                continue;
            }
            let t = p.y / 0.2;
            let q = Vec2::new(p.x, p.z);
            let side = (1.0 - t) * d2(&two, q) + t * d2(&one, q);
            // the nearest end plane bounds a solid stack too (capped like an extrusion)
            let dd = crate::primitives::extrude_2d(side, (0.0 - p.y).max(p.y - 0.2));
            let f = s.eval(*p);
            assert!((f * s.lipschitz() - dd).abs() < 1e-5, "{p}");
            assert_eq!(f < 0.0, dd < 0.0, "sign at {p}");
            compared += 1;
        }
        assert!(compared > 100);
        for w in ps.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(
                (s.eval(a) - s.eval(b)).abs() <= (a - b).length() * (1.0 + 1e-5),
                "Lipschitz at {a} {b}"
            );
        }
    }

    // degenerate input is refused: fewer than 2 levels, levels not increasing, a level without
    // loops, a loop of fewer than 3 vertices, a non-finite coordinate
    #[test]
    fn degenerate_stacks_are_refused() {
        let c = circle(Vec2::ZERO, 0.3, 8);
        assert!(SectionStack::new(vec![0.0], vec![vec![c.clone()]]).is_err());
        assert!(SectionStack::new(vec![0.0, 0.0], vec![vec![c.clone()], vec![c.clone()]]).is_err());
        assert!(SectionStack::new(vec![0.0, 1.0], vec![vec![c.clone()], vec![]]).is_err());
        assert!(SectionStack::new(
            vec![0.0, 1.0],
            vec![vec![c.clone()], vec![vec![Vec2::ZERO, Vec2::X]]]
        )
        .is_err());
        let mut bad = c.clone();
        bad[2].x = f32::NAN;
        assert!(SectionStack::new(vec![0.0, 1.0], vec![vec![c], vec![bad]]).is_err());
    }

    // the stored bound is recomputed on load: a round trip keeps it, a tampered value (smaller or
    // larger) and a missing one are refused, and so is a stack that new() would refuse
    #[test]
    fn load_recomputes_and_checks_the_bound() {
        let a = circle(Vec2::ZERO, 0.1, 24);
        let b = circle(Vec2::new(0.02, 0.0), 0.06, 24);
        let s = SectionStack::new(vec![0.0, 0.05], vec![vec![a], vec![b]]).unwrap();
        assert!(s.lipschitz() > 1.0);
        let json = serde_json::to_string(&s).unwrap();
        let back: SectionStack = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        for wrong in [1.0_f32, s.lipschitz() * 0.5, s.lipschitz() * 2.0] {
            v["lipschitz"] = serde_json::json!(wrong);
            let r: Result<SectionStack, _> = serde_json::from_value(v.clone());
            assert!(
                r.is_err(),
                "stored lipschitz {wrong} accepted (bound {})",
                s.lipschitz()
            );
        }
        v.as_object_mut().unwrap().remove("lipschitz");
        assert!(serde_json::from_value::<SectionStack>(v.clone()).is_err());
        let one_level = serde_json::json!({"levels": [0.0], "loops": [[[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]], "lipschitz": 1.0});
        assert!(serde_json::from_value::<SectionStack>(one_level).is_err());
    }
}
