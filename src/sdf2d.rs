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
    Union(Box<Sdf2dNode>, Box<Sdf2dNode>),
    /// Boolean subtraction (a - b).
    Subtract(Box<Sdf2dNode>, Box<Sdf2dNode>),
    /// Boolean intersection.
    Intersect(Box<Sdf2dNode>, Box<Sdf2dNode>),
    /// Smooth union with blending radius k.
    SmoothUnion {
        /// First child.
        a: Box<Sdf2dNode>,
        /// Second child.
        b: Box<Sdf2dNode>,
        /// Blending radius.
        k: f32,
    },

    /// Translation.
    Translate {
        /// Child node.
        child: Box<Sdf2dNode>,
        /// Offset.
        offset: [f32; 2],
    },
    /// Rotation around origin.
    Rotate {
        /// Child node.
        child: Box<Sdf2dNode>,
        /// Angle in radians.
        angle: f32,
    },
    /// Uniform scale.
    Scale {
        /// Child node.
        child: Box<Sdf2dNode>,
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
            (dx * dx + dy * dy).sqrt() - radius
        }

        Sdf2dNode::Rect {
            center,
            half_extents,
        } => {
            let dx = (point[0] - center[0]).abs() - half_extents[0];
            let dy = (point[1] - center[1]).abs() - half_extents[1];
            let outside = (dx.max(0.0) * dx.max(0.0) + dy.max(0.0) * dy.max(0.0)).sqrt();
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
            let outside = (dx.max(0.0) * dx.max(0.0) + dy.max(0.0) * dy.max(0.0)).sqrt();
            let inside = dx.max(dy).min(0.0);
            outside + inside - corner_radius
        }

        Sdf2dNode::Line { a, b, thickness } => {
            let pa = [point[0] - a[0], point[1] - a[1]];
            let ba = [b[0] - a[0], b[1] - a[1]];
            let ba_sq = ba[0] * ba[0] + ba[1] * ba[1];
            let t = if ba_sq > 1e-10 {
                ((pa[0] * ba[0] + pa[1] * ba[1]) / ba_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let dx = pa[0] - ba[0] * t;
            let dy = pa[1] - ba[1] * t;
            (dx * dx + dy * dy).sqrt() - thickness
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
                return (dx * dx + dy * dy).sqrt();
            }
            bilinear_sample(data, u, v)
        }

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
            let p = [point[0] * c + point[1] * s, -point[0] * s + point[1] * c];
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

// ── Helpers ──────────────────────────────────────────────────

/// Smooth minimum (polynomial) for 2D SDF blending.
#[inline(always)]
fn smooth_min_2d(a: f32, b: f32, k: f32) -> f32 {
    if k < 1e-10 {
        return a.min(b);
    }
    let h = ((k - (a - b).abs()) / k).clamp(0.0, 1.0);
    a.min(b) - h * h * k * 0.25
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

    let top = d00 + (d10 - d00) * tx;
    let bottom = d01 + (d11 - d01) * tx;
    top + (bottom - top) * ty
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
        let bx = it2 * it * p0[0] + 3.0 * it2 * t * p1[0] + 3.0 * it * t2 * p2[0] + t2 * t * p3[0];
        let by = it2 * it * p0[1] + 3.0 * it2 * t * p1[1] + 3.0 * it * t2 * p2[1] + t2 * t * p3[1];
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
    pub fn circle(radius: f32) -> Self {
        Sdf2dNode::Circle {
            center: [0.0, 0.0],
            radius,
        }
    }

    /// Create a rectangle at the origin.
    pub fn rect(half_w: f32, half_h: f32) -> Self {
        Sdf2dNode::Rect {
            center: [0.0, 0.0],
            half_extents: [half_w, half_h],
        }
    }

    /// Create a rounded rectangle at the origin.
    pub fn rounded_rect(half_w: f32, half_h: f32, corner_radius: f32) -> Self {
        Sdf2dNode::RoundedRect {
            center: [0.0, 0.0],
            half_extents: [half_w, half_h],
            corner_radius,
        }
    }

    /// Create a line segment.
    pub fn line(a: [f32; 2], b: [f32; 2], thickness: f32) -> Self {
        Sdf2dNode::Line { a, b, thickness }
    }

    /// Boolean union with another 2D SDF.
    pub fn union(self, other: Sdf2dNode) -> Self {
        Sdf2dNode::Union(Box::new(self), Box::new(other))
    }

    /// Boolean subtraction.
    pub fn subtract(self, other: Sdf2dNode) -> Self {
        Sdf2dNode::Subtract(Box::new(self), Box::new(other))
    }

    /// Boolean intersection.
    pub fn intersect(self, other: Sdf2dNode) -> Self {
        Sdf2dNode::Intersect(Box::new(self), Box::new(other))
    }

    /// Smooth union.
    pub fn smooth_union(self, other: Sdf2dNode, k: f32) -> Self {
        Sdf2dNode::SmoothUnion {
            a: Box::new(self),
            b: Box::new(other),
            k,
        }
    }

    /// Translate.
    pub fn translate(self, x: f32, y: f32) -> Self {
        Sdf2dNode::Translate {
            child: Box::new(self),
            offset: [x, y],
        }
    }

    /// Rotate by angle (radians).
    pub fn rotate(self, angle: f32) -> Self {
        Sdf2dNode::Rotate {
            child: Box::new(self),
            angle,
        }
    }

    /// Uniform scale.
    pub fn scale(self, factor: f32) -> Self {
        Sdf2dNode::Scale {
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
}
