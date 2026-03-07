//! Pure 2D SDF evaluation module
//!
//! Provides a dedicated 2D SDF node type and evaluator for flat geometry,
//! text rendering, and UI elements. Avoids the overhead of 3D evaluation
//! when only 2D distances are needed.
//!
//! Author: Moroya Sakamoto

// ── 2D Node Type ─────────────────────────────────────────────

/// A 2D Signed Distance Function node.
///
/// Represents flat geometry with pure 2D evaluation (no Z component).
#[derive(Debug, Clone)]
pub enum Sdf2dNode {
    /// Circle centered at `center` with `radius`.
    Circle {
        /// Center position.
        center: [f32; 2],
        /// Circle radius.
        radius: f32,
    },

    /// Axis-aligned rectangle centered at `center`.
    Rect {
        /// Center position.
        center: [f32; 2],
        /// Half-extents (half-width, half-height).
        half_extents: [f32; 2],
    },

    /// Rounded rectangle.
    RoundedRect {
        /// Center position.
        center: [f32; 2],
        /// Half-extents (before rounding).
        half_extents: [f32; 2],
        /// Corner radius.
        corner_radius: f32,
    },

    /// Line segment with thickness.
    Line {
        /// First endpoint.
        a: [f32; 2],
        /// Second endpoint.
        b: [f32; 2],
        /// Half-thickness.
        thickness: f32,
    },

    /// Cubic Bezier curve with thickness.
    Bezier {
        /// Start point.
        p0: [f32; 2],
        /// Control point 1.
        p1: [f32; 2],
        /// Control point 2.
        p2: [f32; 2],
        /// End point.
        p3: [f32; 2],
        /// Half-thickness.
        thickness: f32,
    },

    /// Glyph SDF from a precomputed 32x32 grid (ALICE-Font integration).
    FontGlyph {
        /// Flattened 32x32 distance values.
        data: Box<[f32; 1024]>,
        /// Glyph advance width (in em units).
        advance: f32,
        /// Bounding box min corner.
        bbox_min: [f32; 2],
        /// Bounding box max corner.
        bbox_max: [f32; 2],
    },

    /// Boolean union of two 2D SDFs.
    Union(Box<Self>, Box<Self>),
    /// Boolean subtraction (a - b).
    Subtract(Box<Self>, Box<Self>),
    /// Boolean intersection.
    Intersect(Box<Self>, Box<Self>),
    /// Smooth union with blending radius k.
    SmoothUnion {
        /// First child.
        a: Box<Self>,
        /// Second child.
        b: Box<Self>,
        /// Blending radius.
        k: f32,
    },

    /// Ring (annulus): circle with a hole.
    Ring {
        /// Center position.
        center: [f32; 2],
        /// Outer radius.
        outer_radius: f32,
        /// Thickness of the ring wall.
        thickness: f32,
    },

    /// N-sided regular polygon.
    RegularPolygon {
        /// Center position.
        center: [f32; 2],
        /// Circumradius.
        radius: f32,
        /// Number of sides (must be >= 3).
        sides: u32,
    },

    /// Star shape with inner and outer radii.
    Star {
        /// Center position.
        center: [f32; 2],
        /// Outer radius (tip distance).
        outer_radius: f32,
        /// Inner radius (notch distance).
        inner_radius: f32,
        /// Number of points.
        points: u32,
    },

    /// Ellipse.
    Ellipse {
        /// Center position.
        center: [f32; 2],
        /// Semi-axes (half-width, half-height).
        semi_axes: [f32; 2],
    },

    /// Onion (shell) modifier: converts a filled shape to a ring-like outline.
    Onion {
        /// Child node.
        child: Box<Self>,
        /// Shell thickness.
        thickness: f32,
    },

    /// Translation.
    Translate {
        /// Child node.
        child: Box<Self>,
        /// Offset.
        offset: [f32; 2],
    },
    /// Rotation around origin.
    Rotate {
        /// Child node.
        child: Box<Self>,
        /// Angle in radians.
        angle: f32,
    },
    /// Uniform scale.
    Scale {
        /// Child node.
        child: Box<Self>,
        /// Scale factor.
        factor: f32,
    },
}

// ── Evaluation ───────────────────────────────────────────────

/// Evaluate a 2D SDF tree at a point.
///
/// Returns the signed distance: negative = inside, positive = outside.
#[inline]
pub fn eval_2d(node: &Sdf2dNode, point: [f32; 2]) -> f32 {
    match node {
        Sdf2dNode::Circle { center, radius } => {
            let dx = point[0] - center[0];
            let dy = point[1] - center[1];
            dx.hypot(dy) - radius
        }

        Sdf2dNode::Rect {
            center,
            half_extents,
        } => {
            let dx = (point[0] - center[0]).abs() - half_extents[0];
            let dy = (point[1] - center[1]).abs() - half_extents[1];
            let outside = dx
                .max(0.0)
                .mul_add(dx.max(0.0), dy.max(0.0) * dy.max(0.0))
                .sqrt();
            let inside = dx.max(dy).min(0.0);
            outside + inside
        }

        Sdf2dNode::RoundedRect {
            center,
            half_extents,
            corner_radius,
        } => {
            let dx = (point[0] - center[0]).abs() - half_extents[0] + corner_radius;
            let dy = (point[1] - center[1]).abs() - half_extents[1] + corner_radius;
            let outside = dx
                .max(0.0)
                .mul_add(dx.max(0.0), dy.max(0.0) * dy.max(0.0))
                .sqrt();
            let inside = dx.max(dy).min(0.0);
            outside + inside - corner_radius
        }

        Sdf2dNode::Line { a, b, thickness } => {
            let pa = [point[0] - a[0], point[1] - a[1]];
            let ba = [b[0] - a[0], b[1] - a[1]];
            let ba_sq = ba[0].mul_add(ba[0], ba[1] * ba[1]);
            let t = if ba_sq > 1e-10 {
                (pa[0].mul_add(ba[0], pa[1] * ba[1]) / ba_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let dx = ba[0].mul_add(-t, pa[0]);
            let dy = ba[1].mul_add(-t, pa[1]);
            dx.hypot(dy) - thickness
        }

        Sdf2dNode::Bezier {
            p0,
            p1,
            p2,
            p3,
            thickness,
        } => {
            // Approximate cubic bezier distance by sampling
            eval_bezier_distance(point, *p0, *p1, *p2, *p3) - thickness
        }

        Sdf2dNode::FontGlyph {
            data,
            bbox_min,
            bbox_max,
            ..
        } => {
            let w = bbox_max[0] - bbox_min[0];
            let h = bbox_max[1] - bbox_min[1];
            if w < 1e-10 || h < 1e-10 {
                return f32::MAX;
            }
            let u = (point[0] - bbox_min[0]) / w;
            let v = (point[1] - bbox_min[1]) / h;
            if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
                // Outside bbox: approximate distance
                let dx = if u < 0.0 {
                    -u * w
                } else if u > 1.0 {
                    (u - 1.0) * w
                } else {
                    0.0
                };
                let dy = if v < 0.0 {
                    -v * h
                } else if v > 1.0 {
                    (v - 1.0) * h
                } else {
                    0.0
                };
                return dx.hypot(dy);
            }
            bilinear_sample(data, u, v)
        }

        Sdf2dNode::Ring {
            center,
            outer_radius,
            thickness,
        } => {
            let dx = point[0] - center[0];
            let dy = point[1] - center[1];
            (dx.hypot(dy) - outer_radius).abs() - thickness
        }

        Sdf2dNode::RegularPolygon {
            center,
            radius,
            sides,
        } => {
            let n = (*sides).max(3) as f32;
            let px = point[0] - center[0];
            let py = point[1] - center[1];
            let angle = py.atan2(px);
            let sector = std::f32::consts::TAU / n;
            let half_sector = sector * 0.5;
            // Angle within the nearest sector
            let a = (angle % sector + sector) % sector - half_sector;
            let r = px.hypot(py);
            let cos_a = a.cos();
            let sin_a = a.sin();
            // Distance from point to nearest polygon edge
            let edge_dist = radius * half_sector.cos();
            let dx = r.mul_add(cos_a, -edge_dist);
            let dy = (r * sin_a).abs() - radius * half_sector.sin();
            if dx > 0.0 && dy > 0.0 {
                dx.hypot(dy)
            } else {
                dx.max(dy)
            }
        }

        Sdf2dNode::Star {
            center,
            outer_radius,
            inner_radius,
            points,
        } => {
            let n = (*points).max(3) as f32;
            let px = point[0] - center[0];
            let py = point[1] - center[1];
            let r = px.hypot(py);
            let angle = py.atan2(px);
            let sector = std::f32::consts::PI / n;
            let a = 2.0f32.mul_add(sector, angle % (2.0 * sector)) % (2.0 * sector);
            // Interpolate between inner and outer radius based on angle
            let t = (a / sector).min(2.0 - a / sector);
            let boundary = inner_radius + (outer_radius - inner_radius) * t;
            r - boundary
        }

        Sdf2dNode::Ellipse { center, semi_axes } => {
            let px = (point[0] - center[0]).abs();
            let py = (point[1] - center[1]).abs();
            let a = semi_axes[0];
            let b = semi_axes[1];
            // Approximate ellipse SDF using scaling
            if a < 1e-10 || b < 1e-10 {
                return f32::MAX;
            }
            let scale = a.max(b);
            let nx = px / a;
            let ny = py / b;
            let r = nx.hypot(ny);
            if r < 1e-10 {
                return -a.min(b);
            }
            (r - 1.0) * scale / r.max(1.0)
        }

        Sdf2dNode::Onion { child, thickness } => eval_2d(child, point).abs() - thickness,

        Sdf2dNode::Union(a, b) => eval_2d(a, point).min(eval_2d(b, point)),
        Sdf2dNode::Subtract(a, b) => eval_2d(a, point).max(-eval_2d(b, point)),
        Sdf2dNode::Intersect(a, b) => eval_2d(a, point).max(eval_2d(b, point)),
        Sdf2dNode::SmoothUnion { a, b, k } => {
            let da = eval_2d(a, point);
            let db = eval_2d(b, point);
            smooth_min_2d(da, db, *k)
        }

        Sdf2dNode::Translate { child, offset } => {
            eval_2d(child, [point[0] - offset[0], point[1] - offset[1]])
        }
        Sdf2dNode::Rotate { child, angle } => {
            let (s, c) = angle.sin_cos();
            let p = [
                point[0].mul_add(c, point[1] * s),
                (-point[0]).mul_add(s, point[1] * c),
            ];
            eval_2d(child, p)
        }
        Sdf2dNode::Scale { child, factor } => {
            let inv = 1.0 / factor;
            eval_2d(child, [point[0] * inv, point[1] * inv]) * factor
        }
    }
}

/// Evaluate a batch of points against a 2D SDF.
pub fn eval_2d_batch(node: &Sdf2dNode, points: &[[f32; 2]]) -> Vec<f32> {
    points.iter().map(|&p| eval_2d(node, p)).collect()
}

/// Compute the 2D gradient (normal direction) at a point via central differences.
pub fn eval_2d_normal(node: &Sdf2dNode, point: [f32; 2]) -> [f32; 2] {
    let eps = 1e-4_f32;
    let dx = eval_2d(node, [point[0] + eps, point[1]]) - eval_2d(node, [point[0] - eps, point[1]]);
    let dy = eval_2d(node, [point[0], point[1] + eps]) - eval_2d(node, [point[0], point[1] - eps]);
    let len = dx.hypot(dy);
    if len < 1e-10 {
        [0.0, 0.0]
    } else {
        [dx / len, dy / len]
    }
}

// ── Helpers ──────────────────────────────────────────────────

/// Smooth minimum (polynomial) for 2D SDF blending.
#[inline(always)]
fn smooth_min_2d(a: f32, b: f32, k: f32) -> f32 {
    if k < 1e-10 {
        return a.min(b);
    }
    let h = ((k - (a - b).abs()) / k).clamp(0.0, 1.0);
    (h * h * k).mul_add(-0.25, a.min(b))
}

/// Bilinear interpolation on a 32x32 grid.
#[inline]
fn bilinear_sample(data: &[f32; 1024], u: f32, v: f32) -> f32 {
    let fx = u * 31.0;
    let fy = v * 31.0;
    let ix = (fx as u32).min(30);
    let iy = (fy as u32).min(30);
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;

    let i00 = (iy * 32 + ix) as usize;
    let i10 = i00 + 1;
    let i01 = i00 + 32;
    let i11 = i01 + 1;

    let d00 = data[i00];
    let d10 = data[i10];
    let d01 = data[i01];
    let d11 = data[i11];

    let top = (d10 - d00).mul_add(tx, d00);
    let bottom = (d11 - d01).mul_add(tx, d01);
    (bottom - top).mul_add(ty, top)
}

/// Approximate distance to a cubic Bezier curve via uniform sampling.
fn eval_bezier_distance(
    point: [f32; 2],
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
) -> f32 {
    const SAMPLES: u32 = 16;
    let mut min_dist_sq = f32::MAX;

    for i in 0..=SAMPLES {
        let t = i as f32 / SAMPLES as f32;
        let it = 1.0 - t;
        let it2 = it * it;
        let t2 = t * t;
        let bx = (t2 * t).mul_add(
            p3[0],
            (3.0 * it * t2).mul_add(p2[0], (it2 * it).mul_add(p0[0], 3.0 * it2 * t * p1[0])),
        );
        let by = (t2 * t).mul_add(
            p3[1],
            (3.0 * it * t2).mul_add(p2[1], (it2 * it).mul_add(p0[1], 3.0 * it2 * t * p1[1])),
        );
        let dx = point[0] - bx;
        let dy = point[1] - by;
        let d2 = dx * dx + dy * dy;
        if d2 < min_dist_sq {
            min_dist_sq = d2;
        }
    }

    min_dist_sq.sqrt()
}

// ── Constructors ─────────────────────────────────────────────

impl Sdf2dNode {
    /// Create a circle at the origin.
    pub const fn circle(radius: f32) -> Self {
        Self::Circle {
            center: [0.0, 0.0],
            radius,
        }
    }

    /// Create a rectangle at the origin.
    pub const fn rect(half_w: f32, half_h: f32) -> Self {
        Self::Rect {
            center: [0.0, 0.0],
            half_extents: [half_w, half_h],
        }
    }

    /// Create a rounded rectangle at the origin.
    pub const fn rounded_rect(half_w: f32, half_h: f32, corner_radius: f32) -> Self {
        Self::RoundedRect {
            center: [0.0, 0.0],
            half_extents: [half_w, half_h],
            corner_radius,
        }
    }

    /// Create a line segment.
    pub const fn line(a: [f32; 2], b: [f32; 2], thickness: f32) -> Self {
        Self::Line { a, b, thickness }
    }

    /// Create a ring (annulus) at the origin.
    pub const fn ring(outer_radius: f32, thickness: f32) -> Self {
        Self::Ring {
            center: [0.0, 0.0],
            outer_radius,
            thickness,
        }
    }

    /// Create a regular polygon at the origin.
    pub const fn regular_polygon(radius: f32, sides: u32) -> Self {
        Self::RegularPolygon {
            center: [0.0, 0.0],
            radius,
            sides,
        }
    }

    /// Create a star shape at the origin.
    pub const fn star(outer_radius: f32, inner_radius: f32, points: u32) -> Self {
        Self::Star {
            center: [0.0, 0.0],
            outer_radius,
            inner_radius,
            points,
        }
    }

    /// Create an ellipse at the origin.
    pub const fn ellipse(half_w: f32, half_h: f32) -> Self {
        Self::Ellipse {
            center: [0.0, 0.0],
            semi_axes: [half_w, half_h],
        }
    }

    /// Onion modifier: converts filled shape to shell/outline.
    pub fn onion(self, thickness: f32) -> Self {
        Self::Onion {
            child: Box::new(self),
            thickness,
        }
    }

    /// Boolean union with another 2D SDF.
    pub fn union(self, other: Self) -> Self {
        Self::Union(Box::new(self), Box::new(other))
    }

    /// Boolean subtraction.
    pub fn subtract(self, other: Self) -> Self {
        Self::Subtract(Box::new(self), Box::new(other))
    }

    /// Boolean intersection.
    pub fn intersect(self, other: Self) -> Self {
        Self::Intersect(Box::new(self), Box::new(other))
    }

    /// Smooth union.
    pub fn smooth_union(self, other: Self, k: f32) -> Self {
        Self::SmoothUnion {
            a: Box::new(self),
            b: Box::new(other),
            k,
        }
    }

    /// Translate.
    pub fn translate(self, x: f32, y: f32) -> Self {
        Self::Translate {
            child: Box::new(self),
            offset: [x, y],
        }
    }

    /// Rotate by angle (radians).
    pub fn rotate(self, angle: f32) -> Self {
        Self::Rotate {
            child: Box::new(self),
            angle,
        }
    }

    /// Uniform scale.
    pub fn scale(self, factor: f32) -> Self {
        Self::Scale {
            child: Box::new(self),
            factor,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_center_inside() {
        let c = Sdf2dNode::circle(1.0);
        assert!(eval_2d(&c, [0.0, 0.0]) < 0.0);
    }

    #[test]
    fn circle_on_surface() {
        let c = Sdf2dNode::circle(1.0);
        let d = eval_2d(&c, [1.0, 0.0]);
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn circle_outside() {
        let c = Sdf2dNode::circle(1.0);
        let d = eval_2d(&c, [2.0, 0.0]);
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rect_center_inside() {
        let r = Sdf2dNode::rect(1.0, 0.5);
        assert!(eval_2d(&r, [0.0, 0.0]) < 0.0);
    }

    #[test]
    fn rect_outside() {
        let r = Sdf2dNode::rect(1.0, 0.5);
        let d = eval_2d(&r, [2.0, 0.0]);
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn union_2d() {
        let a = Sdf2dNode::circle(1.5).translate(-1.0, 0.0);
        let b = Sdf2dNode::circle(1.5).translate(1.0, 0.0);
        let u = a.union(b);
        // Origin should be inside both circles (distance to each center = 1.0 < 1.5)
        assert!(eval_2d(&u, [0.0, 0.0]) < 0.0);
        // Far away should be outside
        assert!(eval_2d(&u, [5.0, 0.0]) > 0.0);
    }

    #[test]
    fn subtract_2d() {
        let a = Sdf2dNode::circle(2.0);
        let b = Sdf2dNode::circle(1.0);
        let s = a.subtract(b);
        // Origin: inside b, so subtracted → outside
        assert!(eval_2d(&s, [0.0, 0.0]) > 0.0);
        // At radius 1.5: inside a, outside b → inside result
        assert!(eval_2d(&s, [1.5, 0.0]) < 0.0);
    }

    #[test]
    fn batch_eval() {
        let c = Sdf2dNode::circle(1.0);
        let points = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let results = eval_2d_batch(&c, &points);
        assert_eq!(results.len(), 3);
        assert!(results[0] < 0.0); // inside
        assert!(results[1].abs() < 1e-5); // on surface
        assert!(results[2] > 0.0); // outside
    }

    #[test]
    fn translate_2d() {
        let c = Sdf2dNode::circle(1.0).translate(3.0, 0.0);
        assert!(eval_2d(&c, [3.0, 0.0]) < 0.0);
        assert!(eval_2d(&c, [0.0, 0.0]) > 0.0);
    }

    #[test]
    fn scale_2d() {
        let c = Sdf2dNode::circle(1.0).scale(2.0);
        // Scaled circle has radius 2.0
        assert!(eval_2d(&c, [1.5, 0.0]) < 0.0);
        let d = eval_2d(&c, [2.0, 0.0]);
        assert!(d.abs() < 1e-4);
    }

    #[test]
    fn rotate_2d() {
        let r = Sdf2dNode::rect(2.0, 0.1);
        let rotated = r.rotate(std::f32::consts::FRAC_PI_2);
        // After 90° rotation, a point at (0, 1.5) should be inside
        assert!(eval_2d(&rotated, [0.0, 1.5]) < 0.0);
    }

    #[test]
    fn line_distance() {
        let l = Sdf2dNode::line([0.0, 0.0], [1.0, 0.0], 0.1);
        // On the line axis
        let d = eval_2d(&l, [0.5, 0.0]);
        assert!(d < 0.0, "Should be inside line, got {}", d);
        // Away from line
        let d = eval_2d(&l, [0.5, 1.0]);
        assert!((d - 0.9).abs() < 1e-4);
    }

    #[test]
    fn font_glyph_center() {
        let mut data = Box::new([1.0f32; 1024]);
        // Set center region to negative (inside glyph)
        for y in 12..20 {
            for x in 12..20 {
                data[y * 32 + x] = -0.5;
            }
        }
        let glyph = Sdf2dNode::FontGlyph {
            data,
            advance: 0.6,
            bbox_min: [0.0, 0.0],
            bbox_max: [1.0, 1.0],
        };
        // Center should be inside
        let d = eval_2d(&glyph, [0.5, 0.5]);
        assert!(d < 0.0, "Glyph center should be inside, got {}", d);
    }

    #[test]
    fn font_glyph_outside_bbox() {
        let data = Box::new([0.0f32; 1024]);
        let glyph = Sdf2dNode::FontGlyph {
            data,
            advance: 0.6,
            bbox_min: [0.0, 0.0],
            bbox_max: [1.0, 1.0],
        };
        let d = eval_2d(&glyph, [2.0, 2.0]);
        assert!(d > 0.0);
    }

    #[test]
    fn smooth_union_2d() {
        let a = Sdf2dNode::circle(1.0).translate(-0.5, 0.0);
        let b = Sdf2dNode::circle(1.0).translate(0.5, 0.0);
        let s = a.smooth_union(b, 0.5);
        let d = eval_2d(&s, [0.0, 0.0]);
        assert!(d < 0.0);
    }

    #[test]
    fn rounded_rect() {
        let rr = Sdf2dNode::rounded_rect(1.0, 0.5, 0.1);
        assert!(eval_2d(&rr, [0.0, 0.0]) < 0.0);
        assert!(eval_2d(&rr, [2.0, 0.0]) > 0.0);
    }

    #[test]
    fn ring_inside_wall() {
        let r = Sdf2dNode::ring(1.0, 0.1);
        // On the ring at radius 1.0: inside wall
        let d = eval_2d(&r, [1.0, 0.0]);
        assert!(d < 0.0, "Should be inside ring wall, got {}", d);
    }

    #[test]
    fn ring_center_outside() {
        let r = Sdf2dNode::ring(1.0, 0.1);
        // Center is outside the ring
        let d = eval_2d(&r, [0.0, 0.0]);
        assert!(d > 0.0, "Center should be outside ring, got {}", d);
    }

    #[test]
    fn ring_far_outside() {
        let r = Sdf2dNode::ring(1.0, 0.1);
        let d = eval_2d(&r, [3.0, 0.0]);
        assert!(d > 0.0);
    }

    #[test]
    fn regular_polygon_center_inside() {
        let hex = Sdf2dNode::regular_polygon(1.0, 6);
        assert!(eval_2d(&hex, [0.0, 0.0]) < 0.0);
    }

    #[test]
    fn regular_polygon_outside() {
        let hex = Sdf2dNode::regular_polygon(1.0, 6);
        assert!(eval_2d(&hex, [2.0, 0.0]) > 0.0);
    }

    #[test]
    fn star_center_inside() {
        let s = Sdf2dNode::star(1.0, 0.4, 5);
        assert!(eval_2d(&s, [0.0, 0.0]) < 0.0);
    }

    #[test]
    fn star_far_outside() {
        let s = Sdf2dNode::star(1.0, 0.4, 5);
        assert!(eval_2d(&s, [3.0, 0.0]) > 0.0);
    }

    #[test]
    fn ellipse_center_inside() {
        let e = Sdf2dNode::ellipse(2.0, 1.0);
        assert!(eval_2d(&e, [0.0, 0.0]) < 0.0);
    }

    #[test]
    fn ellipse_along_major_axis() {
        let e = Sdf2dNode::ellipse(2.0, 1.0);
        // Inside along major axis
        assert!(eval_2d(&e, [1.5, 0.0]) < 0.0);
        // Outside along minor axis at same distance
        assert!(eval_2d(&e, [0.0, 1.5]) > 0.0);
    }

    #[test]
    fn onion_2d() {
        let c = Sdf2dNode::circle(1.0).onion(0.1);
        // Center: circle gives -1.0, onion → |−1.0| − 0.1 = 0.9 → outside
        assert!(eval_2d(&c, [0.0, 0.0]) > 0.0);
        // On surface at radius 1.0: circle gives 0, onion → |0| − 0.1 = −0.1 → inside
        let d = eval_2d(&c, [1.0, 0.0]);
        assert!(d < 0.0, "Should be inside onion shell, got {}", d);
    }

    #[test]
    fn eval_2d_normal_circle() {
        let c = Sdf2dNode::circle(1.0);
        let n = super::eval_2d_normal(&c, [1.0, 0.0]);
        assert!((n[0] - 1.0).abs() < 0.01, "nx={}", n[0]);
        assert!(n[1].abs() < 0.01, "ny={}", n[1]);
    }

    #[test]
    fn eval_2d_normal_diagonal() {
        let c = Sdf2dNode::circle(1.0);
        let n = super::eval_2d_normal(&c, [1.0, 1.0]);
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        assert!((n[0] - inv_sqrt2).abs() < 0.02, "nx={}", n[0]);
        assert!((n[1] - inv_sqrt2).abs() < 0.02, "ny={}", n[1]);
    }

    #[test]
    fn intersect_2d() {
        let a = Sdf2dNode::circle(1.5).translate(-0.5, 0.0);
        let b = Sdf2dNode::circle(1.5).translate(0.5, 0.0);
        let i = a.intersect(b);
        // Origin: inside both
        assert!(eval_2d(&i, [0.0, 0.0]) < 0.0);
        // Far left: inside a only
        assert!(eval_2d(&i, [-1.5, 0.0]) > 0.0);
    }
}
