//! The 2D SDF ([`Sdf2dNode`]) as WGSL
//!
//! [`Wgsl2dShader::transpile`] writes the node as straight-line WGSL that mirrors
//! [`eval_2d`](crate::sdf2d::eval_2d) arm for arm: [`Wgsl2dShader::source`] is the fragment
//! `fn sdf2d_eval(p: vec2<f32>) -> f32` with the helpers it uses (no entry point, so no
//! identifier: it is pasted into a host shader), and [`Wgsl2dShader::to_compute_shader`] is a
//! whole compute shader over it with the bindings of the 3D one (points in, distances out), so
//! [`GpuEvaluator::from_wgsl`](super::GpuEvaluator::from_wgsl) runs it; it carries this crate's
//! semantics identifier (ALICE-Law G8).
//!
//! The constants the CPU law computes from the node (the sine and cosine of a rotation, the
//! sector angles of a polygon, the reciprocal of a scale) are computed here with the same
//! deterministic arithmetic and written as literals; the GPU computes the rest (`length`,
//! `atan2`, `sin`, `cos`) with its own functions, so the distances agree within a tolerance, not
//! bit for bit (tests/test_sdf2d_wgsl.rs). A parameter that is not a finite number is refused.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use crate::sdf2d::Sdf2dNode;
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// A 2D SDF as WGSL
#[derive(Debug, Clone)]
pub struct Wgsl2dShader {
    /// The fragment: the helpers it uses and `fn sdf2d_eval(p: vec2<f32>) -> f32`
    pub source: String,
}

/// Why a 2D SDF was not emitted
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Wgsl2dError {
    /// A parameter (or a constant computed from one, such as the reciprocal of a scale of 0) is
    /// not a finite number
    #[error("a 2D SDF parameter is not a finite number: {0}")]
    NonFinite(f32),
}

const BOX: &str = "fn sdf2d_box(q: vec2<f32>, h: vec2<f32>) -> f32 {
    let d = abs(q) - h;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}
";

const LINE: &str = "fn sdf2d_line(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let ba_sq = ba.x * ba.x + ba.y * ba.y;
    var t = 0.0;
    if (ba_sq > 1e-10) {
        t = clamp((pa.x * ba.x + pa.y * ba.y) / ba_sq, 0.0, 1.0);
    }
    return length(vec2<f32>(ba.x * -t + pa.x, ba.y * -t + pa.y));
}
";

const BEZIER: &str = "fn sdf2d_bezier(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> f32 {
    var best = 3.4028235e38;
    for (var i = 0u; i <= 16u; i = i + 1u) {
        let t = f32(i) / 16.0;
        let it = 1.0 - t;
        let it2 = it * it;
        let t2 = t * t;
        let b = (t2 * t) * p3 + ((3.0 * it * t2) * p2 + ((it2 * it) * p0 + (3.0 * it2 * t) * p1));
        let d = p - b;
        let d2 = d.x * d.x + d.y * d.y;
        if (d2 < best) {
            best = d2;
        }
    }
    return sqrt(best);
}
";

const POLYGON: &str = "fn sdf2d_polygon(q: vec2<f32>, sector: f32, half_sector: f32, edge: f32, half_side: f32) -> f32 {
    let angle = atan2(q.y, q.x);
    let a = (angle % sector + sector) % sector - half_sector;
    let r = length(q);
    let dx = r * cos(a) + -edge;
    let dy = abs(r * sin(a)) - half_side;
    if (dx > 0.0 && dy > 0.0) {
        return length(vec2<f32>(dx, dy));
    }
    return dx;
}
";

const STAR: &str = "fn sdf2d_star(q: vec2<f32>, sector: f32, inner: f32, outer: f32) -> f32 {
    let r = length(q);
    let angle = atan2(q.y, q.x);
    let a = (2.0 * sector + (angle % (2.0 * sector))) % (2.0 * sector);
    let t = min(a / sector, 2.0 - a / sector);
    return r - (inner + (outer - inner) * t);
}
";

const ELLIPSE: &str =
    "fn sdf2d_ellipse(q: vec2<f32>, a: f32, b: f32, scale: f32, inside: f32) -> f32 {
    let r = length(vec2<f32>(abs(q.x) / a, abs(q.y) / b));
    if (r < 1e-10) {
        return inside;
    }
    return (r - 1.0) * scale / max(r, 1.0);
}
";

const SMIN: &str = "fn sdf2d_smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp((k - abs(a - b)) / k, 0.0, 1.0);
    return (h * h * k) * -0.25 + min(a, b);
}
";

/// A WGSL literal for `x`, in parentheses when negative
fn lit(x: f32) -> Result<String, Wgsl2dError> {
    if !x.is_finite() {
        return Err(Wgsl2dError::NonFinite(x));
    }
    let s = format!("{x:?}");
    let s = if s.contains(['.', 'e']) {
        s
    } else {
        format!("{s}.0")
    };
    Ok(if x.is_sign_negative() {
        format!("({s})")
    } else {
        s
    })
}

fn vec2(v: [f32; 2]) -> Result<String, Wgsl2dError> {
    Ok(format!("vec2<f32>({}, {})", lit(v[0])?, lit(v[1])?))
}

#[derive(Default)]
struct Emitter {
    body: String,
    helpers: BTreeSet<&'static str>,
    glyphs: String,
    next: usize,
}

impl Emitter {
    fn var(&mut self, prefix: &str) -> String {
        self.next += 1;
        format!("{prefix}{}", self.next)
    }

    fn line(&mut self, name: &str, expr: &str) {
        let _ = writeln!(self.body, "    let {name} = {expr};");
    }

    /// Emits `node` at the point variable `p`; returns the distance variable
    fn node(&mut self, node: &Sdf2dNode, p: &str) -> Result<String, Wgsl2dError> {
        let d = self.var("d");
        match node {
            Sdf2dNode::Circle { center, radius } => {
                let e = format!("length({p} - {}) - {}", vec2(*center)?, lit(*radius)?);
                self.line(&d, &e);
            }
            Sdf2dNode::Rect {
                center,
                half_extents,
            } => {
                self.helpers.insert(BOX);
                let e = format!(
                    "sdf2d_box({p} - {}, {})",
                    vec2(*center)?,
                    vec2(*half_extents)?
                );
                self.line(&d, &e);
            }
            Sdf2dNode::RoundedRect {
                center,
                half_extents,
                corner_radius,
            } => {
                self.helpers.insert(BOX);
                let inner = [
                    half_extents[0] - corner_radius,
                    half_extents[1] - corner_radius,
                ];
                let e = format!(
                    "sdf2d_box({p} - {}, {}) - {}",
                    vec2(*center)?,
                    vec2(inner)?,
                    lit(*corner_radius)?
                );
                self.line(&d, &e);
            }
            Sdf2dNode::Line { a, b, thickness } => {
                self.helpers.insert(LINE);
                let e = format!(
                    "sdf2d_line({p}, {}, {}) - {}",
                    vec2(*a)?,
                    vec2(*b)?,
                    lit(*thickness)?
                );
                self.line(&d, &e);
            }
            Sdf2dNode::Bezier {
                p0,
                p1,
                p2,
                p3,
                thickness,
            } => {
                self.helpers.insert(BEZIER);
                let e = format!(
                    "sdf2d_bezier({p}, {}, {}, {}, {}) - {}",
                    vec2(*p0)?,
                    vec2(*p1)?,
                    vec2(*p2)?,
                    vec2(*p3)?,
                    lit(*thickness)?
                );
                self.line(&d, &e);
            }
            Sdf2dNode::FontGlyph {
                data,
                bbox_min,
                bbox_max,
                ..
            } => {
                let f = self.glyph(data, *bbox_min, *bbox_max)?;
                self.line(&d, &format!("{f}({p})"));
            }
            Sdf2dNode::Ring {
                center,
                outer_radius,
                thickness,
            } => {
                let e = format!(
                    "abs(length({p} - {}) - {}) - {}",
                    vec2(*center)?,
                    lit(*outer_radius)?,
                    lit(*thickness)?
                );
                self.line(&d, &e);
            }
            Sdf2dNode::RegularPolygon {
                center,
                radius,
                sides,
            } => {
                self.helpers.insert(POLYGON);
                // the constants of eval_2d, computed with its arithmetic
                let n = (*sides).max(3) as f32;
                let sector = std::f32::consts::TAU / n;
                let half_sector = sector * 0.5;
                let edge = radius * alice_det_math::cos(half_sector);
                let half_side = radius * alice_det_math::sin(half_sector);
                let e = format!(
                    "sdf2d_polygon({p} - {}, {}, {}, {}, {})",
                    vec2(*center)?,
                    lit(sector)?,
                    lit(half_sector)?,
                    lit(edge)?,
                    lit(half_side)?
                );
                self.line(&d, &e);
            }
            Sdf2dNode::Star {
                center,
                outer_radius,
                inner_radius,
                points,
            } => {
                self.helpers.insert(STAR);
                let n = (*points).max(3) as f32;
                let sector = std::f32::consts::PI / n;
                let e = format!(
                    "sdf2d_star({p} - {}, {}, {}, {})",
                    vec2(*center)?,
                    lit(sector)?,
                    lit(*inner_radius)?,
                    lit(*outer_radius)?
                );
                self.line(&d, &e);
            }
            Sdf2dNode::Ellipse { center, semi_axes } => {
                let [a, b] = *semi_axes;
                vec2(*center)?;
                if a < 1e-10 || b < 1e-10 {
                    self.line(&d, &lit(f32::MAX)?);
                } else {
                    self.helpers.insert(ELLIPSE);
                    let e = format!(
                        "sdf2d_ellipse({p} - {}, {}, {}, {}, {})",
                        vec2(*center)?,
                        lit(a)?,
                        lit(b)?,
                        lit(a.law_max(b))?,
                        lit(-a.law_min(b))?
                    );
                    self.line(&d, &e);
                }
            }
            Sdf2dNode::Onion { child, thickness } => {
                let c = self.node(child, p)?;
                self.line(&d, &format!("abs({c}) - {}", lit(*thickness)?));
            }
            Sdf2dNode::Union(a, b) => {
                let (x, y) = (self.node(a, p)?, self.node(b, p)?);
                self.line(&d, &format!("min({x}, {y})"));
            }
            Sdf2dNode::Subtract(a, b) => {
                let (x, y) = (self.node(a, p)?, self.node(b, p)?);
                self.line(&d, &format!("max({x}, -{y})"));
            }
            Sdf2dNode::Intersect(a, b) => {
                let (x, y) = (self.node(a, p)?, self.node(b, p)?);
                self.line(&d, &format!("max({x}, {y})"));
            }
            Sdf2dNode::SmoothUnion { a, b, k } => {
                let (x, y) = (self.node(a, p)?, self.node(b, p)?);
                if *k < 1e-10 {
                    lit(*k)?;
                    self.line(&d, &format!("min({x}, {y})"));
                } else {
                    self.helpers.insert(SMIN);
                    self.line(&d, &format!("sdf2d_smin({x}, {y}, {})", lit(*k)?));
                }
            }
            Sdf2dNode::Translate { child, offset } => {
                let q = self.var("p");
                let e = format!("{p} - {}", vec2(*offset)?);
                self.line(&q, &e);
                let c = self.node(child, &q)?;
                self.line(&d, &c);
            }
            Sdf2dNode::Rotate { child, angle } => {
                lit(*angle)?;
                let (s, c) = alice_det_math::sin_cos(*angle);
                let (s, c) = (lit(s)?, lit(c)?);
                let q = self.var("p");
                let e =
                    format!("vec2<f32>({p}.x * {c} + {p}.y * {s}, (-{p}.x) * {s} + {p}.y * {c})");
                self.line(&q, &e);
                let r = self.node(child, &q)?;
                self.line(&d, &r);
            }
            Sdf2dNode::Scale { child, factor } => {
                let inv = lit(1.0 / factor)?;
                let q = self.var("p");
                self.line(&q, &format!("{p} * {inv}"));
                let r = self.node(child, &q)?;
                self.line(&d, &format!("{r} * {}", lit(*factor)?));
            }
        }
        Ok(d)
    }

    /// A function for one glyph grid (its 4096 values in a private array); returns its name
    fn glyph(
        &mut self,
        data: &[f32; 4096],
        bbox_min: [f32; 2],
        bbox_max: [f32; 2],
    ) -> Result<String, Wgsl2dError> {
        let id = self.var("");
        let name = format!("sdf2d_glyph{id}");
        let w = bbox_max[0] - bbox_min[0];
        let h = bbox_max[1] - bbox_min[1];
        let (min, w_l, h_l) = (vec2(bbox_min)?, lit(w)?, lit(h)?);
        if w < 1e-10 || h < 1e-10 {
            let _ = writeln!(
                self.glyphs,
                "fn {name}(p: vec2<f32>) -> f32 {{\n    return {};\n}}\n",
                lit(f32::MAX)?
            );
            return Ok(name);
        }
        let mut values = String::with_capacity(4096 * 12);
        for (i, v) in data.iter().enumerate() {
            if i > 0 {
                values.push_str(", ");
            }
            values.push_str(&lit(*v)?);
        }
        let _ = write!(
            self.glyphs,
            "var<private> {name}_data: array<f32, 4096> = array<f32, 4096>({values});

fn {name}(p: vec2<f32>) -> f32 {{
    let u = (p.x - {min}.x) / {w_l};
    let v = (p.y - {min}.y) / {h_l};
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {{
        var dx = 0.0;
        if (u < 0.0) {{ dx = -u * {w_l}; }} else if (u > 1.0) {{ dx = (u - 1.0) * {w_l}; }}
        var dy = 0.0;
        if (v < 0.0) {{ dy = -v * {h_l}; }} else if (v > 1.0) {{ dy = (v - 1.0) * {h_l}; }}
        return length(vec2<f32>(dx, dy));
    }}
    let fx = u * 63.0;
    let fy = v * 63.0;
    let ix = min(u32(fx), 62u);
    let iy = min(u32(fy), 62u);
    let tx = fx - f32(ix);
    let ty = fy - f32(iy);
    let i00 = iy * 64u + ix;
    let d00 = {name}_data[i00];
    let d10 = {name}_data[i00 + 1u];
    let d01 = {name}_data[i00 + 64u];
    let d11 = {name}_data[i00 + 65u];
    let top = (d10 - d00) * tx + d00;
    let bottom = (d11 - d01) * tx + d01;
    return (bottom - top) * ty + top;
}}

"
        );
        Ok(name)
    }
}

impl Wgsl2dShader {
    /// Emits `node`
    ///
    /// # Errors
    ///
    /// [`Wgsl2dError::NonFinite`] when a parameter, or a constant computed from one, is not a
    /// finite number.
    pub fn transpile(node: &Sdf2dNode) -> Result<Self, Wgsl2dError> {
        let mut e = Emitter::default();
        let d = e.node(node, "p")?;
        let mut source = String::from("// ALICE-SDF Generated 2D SDF (sdf2d_eval)\n\n");
        for h in &e.helpers {
            source.push_str(h);
            source.push('\n');
        }
        source.push_str(&e.glyphs);
        let _ = write!(
            source,
            "fn sdf2d_eval(p: vec2<f32>) -> f32 {{\n{}    return {d};\n}}\n",
            e.body
        );
        Ok(Self { source })
    }

    /// A whole compute shader over [`Self::source`]: the points (x, y of each; z ignored) in at
    /// binding 0, the distances out at binding 1, the count at binding 2, as the 3D compute shader
    /// (so `GpuEvaluator::from_wgsl` runs it); it carries this crate's semantics identifier
    #[must_use]
    pub fn to_compute_shader(&self) -> String {
        crate::provenance::tagged_shader(&format!(
            "// ALICE-SDF Generated 2D Compute Shader

struct InputPoint {{
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}}

struct OutputDistance {{
    distance: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}}

@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;

{}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= point_count) {{
        return;
    }}
    let point = input_points[idx];
    output_distances[idx].distance = sdf2d_eval(vec2<f32>(point.x, point.y));
}}
",
            self.source
        ))
    }
}
