//! Whether a closed polyline crosses itself: the check behind the closed sections' "simple loop"
//! requirement (a crossing loop has no consistent inside, so the even-odd or tangent-side sign flips
//! across the crossing away from any surface and the field jumps)
//!
//! Proper crossings only: two non-adjacent edges whose endpoints lie strictly on opposite sides of
//! each other (orientations in f64, exact for f32 inputs up to the product's rounding). Edges that
//! merely touch are not counted; a touching loop is caught by its area or by the field oracles.
//! Edges are swept in order of their smallest x, so a loop of n edges costs O(n log n + k) for k
//! pairs that overlap in x.

use glam::Vec2;

#[inline]
fn orient(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// True when two non-adjacent edges of the closed polyline `pts` cross properly
#[must_use]
pub fn closed_polyline_crosses(pts: &[Vec2]) -> bool {
    let n = pts.len();
    if n < 4 {
        return false;
    }
    let p: Vec<(f64, f64)> = pts
        .iter()
        .map(|v| (f64::from(v.x), f64::from(v.y)))
        .collect();
    let edge = |i: usize| (p[i], p[(i + 1) % n]);
    let mut order: Vec<usize> = (0..n).collect();
    let lo = |i: usize| alice_det_math::min64(edge(i).0 .0, edge(i).1 .0);
    let hi = |i: usize| alice_det_math::max64(edge(i).0 .0, edge(i).1 .0);
    order.sort_by(|&a, &b| lo(a).total_cmp(&lo(b)).then(a.cmp(&b)));
    let mut active: Vec<usize> = Vec::new();
    for &i in &order {
        let x = lo(i);
        active.retain(|&j| hi(j) >= x);
        let (a, b) = edge(i);
        for &j in &active {
            let adjacent = (i + 1) % n == j || (j + 1) % n == i;
            if adjacent {
                continue;
            }
            let (c, d) = edge(j);
            let (d1, d2) = (orient(a, b, c), orient(a, b, d));
            let (d3, d4) = (orient(c, d, a), orient(c, d, b));
            if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
                return true;
            }
        }
        active.push(i);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_is_simple_and_a_bowtie_crosses() {
        let sq = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        assert!(!closed_polyline_crosses(&sq));
        let bow = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        ];
        assert!(closed_polyline_crosses(&bow));
    }

    #[test]
    fn a_crossing_of_later_edges_is_found() {
        // edge 2 runs back down across edge 0
        let p = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(0.5, 1.0),
        ];
        assert!(closed_polyline_crosses(&p));
    }

    #[test]
    fn fewer_than_four_points_never_cross() {
        assert!(!closed_polyline_crosses(&[Vec2::ZERO, Vec2::X, Vec2::Y]));
    }
}
