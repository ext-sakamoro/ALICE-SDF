//! One call site per node kind in the bounded union of the node data buffer form.
//!
//! A driver's compile time follows the call sites of a shader, not its text: with one copy of a
//! node's functions per node, 400 extruded polygons took 643 s to build on an Orin and 7.6 GB on
//! llvmpipe (more than a hosted CI runner holds), and a body of 17 lofts 60-313 s on real drivers.
//! Here the operands of the union at the top that share a kind (and the shape of their translate /
//! rotate / scale wrappers) are evaluated by one loop over their rows in the node data buffer: the
//! row holds the operand's bound, its wrappers and the offset of its data, and the kind's function
//! is written once and called once.
//!
//! The values are those of the per-node form: every per-node number is read from the buffer in both
//! (no literal folds differently), the arithmetic of each operand is the same sequence, and the
//! union's minimum does not depend on the order of the values it keeps (the bound skips only
//! operands whose lower bound is above the minimum so far, as before; the skip count can differ
//! with the order). Kinds not handled here, operands without a bound, wrappers other than
//! translate / rotate / uniform scale, Dynamic mode and the checked reads keep the per-node form.
//!
//! Author: Moroya Sakamoto

use super::transpiler::WgslLang;
use crate::bound::{Bound, SLACK_PER_NODE};
use crate::compiled::transpiler_common::{
    GenericTranspiler, TranspileModeLang, POLYGON_2D_UNROLL_MAX,
};
use crate::types::SdfNode;
use glam::{Quat, Vec3};
use std::fmt::Write;

/// The operands a loop evaluates and the code of the loops
#[derive(Default)]
pub(super) struct Looped {
    /// Indices into the operands
    pub members: Vec<usize>,
    /// The loops over operands without a bound, written before every other operand (as the
    /// unbounded operands are: they set the minimum the bounds are compared with)
    pub unbounded: String,
    /// The loops over operands with a bound, written where the bounded operands start
    pub code: String,
}

/// A wrapper between the union and the operand's node, outermost first
enum Wrap {
    Translate(Vec3),
    /// The inverse rotation, as the per-node form applies it
    Rotate(Quat),
    /// The factor
    Scale(f32),
}

impl Wrap {
    const fn key(&self) -> char {
        match self {
            Self::Translate(_) => 'T',
            Self::Rotate(_) => 'R',
            Self::Scale(_) => 'S',
        }
    }
}

/// A node kind with a loop form
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// `Polygon2D` with more vertices than the unrolled form takes (the per-node form's array and
    /// loop function, `ShaderLang::global_polygon_2d_fn`)
    Polygon,
}

fn peel(mut node: &SdfNode) -> (Vec<Wrap>, &SdfNode) {
    let mut wraps = Vec::new();
    loop {
        match node {
            SdfNode::Translate { child, offset } => {
                wraps.push(Wrap::Translate(*offset));
                node = child;
            }
            SdfNode::Rotate { child, rotation } => {
                wraps.push(Wrap::Rotate(rotation.inverse()));
                node = child;
            }
            SdfNode::Scale { child, factor } => {
                wraps.push(Wrap::Scale(*factor));
                node = child;
            }
            _ => return (wraps, node),
        }
    }
}

const fn kind(node: &SdfNode) -> Option<Kind> {
    match node {
        SdfNode::Polygon2D { vertices, .. } if vertices.len() > POLYGON_2D_UNROLL_MAX => {
            Some(Kind::Polygon)
        }
        _ => None,
    }
}

/// The values of one row: the bound (c, k, r, slack per unit, |c|), the data offset, the kind's own
/// values, then each wrapper's
const ROW_HEAD: usize = 7;

/// The largest integer an `f32` holds exactly: offsets in the buffer are read as `f32`
const EXACT_INT: usize = 1 << 24;

#[cfg(test)]
thread_local! {
    /// The test reference: the same rows and functions, one call site per operand (no loop)
    pub(super) static UNROLL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Group the bounded operands of `ops` by kind and wrapper shape and write a loop for each group
pub(super) fn group(
    g: &mut GenericTranspiler<WgslLang>,
    ops: &[&SdfNode],
    bounded: &[Option<Bound>],
) -> Looped {
    let mut out = Looped::default();
    if g.node_data.is_none() || g.checked_reads || g.mode != TranspileModeLang::Hardcoded {
        return out;
    }
    #[cfg(test)]
    if OFF.with(std::cell::Cell::get) {
        return out;
    }
    type Row<'a> = (usize, Vec<Wrap>, &'a SdfNode, Option<Bound>);
    let mut groups: Vec<((Kind, String, bool), Vec<Row<'_>>)> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let b = bounded[i];
        let (wraps, base) = peel(op);
        let Some(k) = kind(base) else { continue };
        let key = (
            k,
            wraps.iter().map(Wrap::key).collect::<String>(),
            b.is_some(),
        );
        match groups.iter_mut().find(|(gk, _)| *gk == key) {
            Some((_, v)) => v.push((i, wraps, base, b)),
            None => groups.push((key, vec![(i, wraps, base, b)])),
        }
    }
    for ((k, shape, has_bound), rows) in groups {
        let Some(code) = emit_group(g, k, &shape, &rows) else {
            continue;
        };
        out.members.extend(rows.iter().map(|r| r.0));
        if has_bound {
            out.code.push_str(&code);
        } else {
            out.unbounded.push_str(&code);
        }
    }
    out
}

/// The kind's own data (appended to the buffer) and its per-row values
fn kind_data(k: Kind, base: &SdfNode) -> (Vec<f32>, Vec<f32>) {
    match (k, base) {
        (
            Kind::Polygon,
            SdfNode::Polygon2D {
                vertices,
                half_height,
            },
        ) => {
            let mut d = vec![vertices.len() as f32];
            d.extend(vertices.iter().flat_map(|q| [q.x, q.y]));
            (d, vec![*half_height])
        }
        _ => unreachable!("kind() classified the node"),
    }
}

fn emit_group(
    g: &mut GenericTranspiler<WgslLang>,
    k: Kind,
    shape: &str,
    rows: &[(usize, Vec<Wrap>, &SdfNode, Option<Bound>)],
) -> Option<String> {
    let buf = g.node_data.as_mut()?;
    let start = buf.len();
    // each operand's data, then the table of rows
    let mut table: Vec<f32> = Vec::new();
    let mut own_len = 0;
    for (_, wraps, base, b) in rows {
        let (data, own) = kind_data(k, base);
        own_len = own.len();
        let off = buf.len();
        buf.extend_from_slice(&data);
        // an operand without a bound has zeros there, never read (its group has no guard)
        let (c, kk, r, sl, cl) = b.map_or((Vec3::ZERO, 0.0, 0.0, 0.0, 0.0), |b| {
            (
                b.c,
                b.k,
                b.r,
                (b.depth as f32 + 1.0) * SLACK_PER_NODE,
                b.c.length(),
            )
        });
        table.extend([c.x, c.y, c.z, kk, r, sl, cl, off as f32]);
        table.extend(own);
        for w in wraps {
            match w {
                Wrap::Translate(o) => table.extend([o.x, o.y, o.z]),
                Wrap::Rotate(q) => table.extend([q.x, q.y, q.z, q.w]),
                Wrap::Scale(f) => table.extend([1.0 / f, *f]),
            }
        }
    }
    let stride = table.len() / rows.len();
    let at = buf.len();
    buf.extend_from_slice(&table);
    if buf.len() >= EXACT_INT {
        // offsets past 2^24 are not exact in f32: this tree keeps the per-node form
        buf.truncate(start);
        return None;
    }
    if shape.contains('R') {
        g.ensure_helper("quat_rotate");
    }
    let func = kind_function(k);
    if !g.globals.contains(func.0) {
        g.globals.push_str(func.1);
    }
    // the operand's point, its value and the scale factors, as the per-node form writes them
    let nd = |o: usize| format!("alice_node_data[lr + {o}]");
    let mut body = String::from("            var q: vec3<f32> = p;\n");
    let mut o = ROW_HEAD + 1 + own_len;
    let mut factors = Vec::new();
    for c in shape.chars() {
        match c {
            'T' => {
                writeln!(
                    body,
                    "            q = q - vec3<f32>({}, {}, {});",
                    nd(o),
                    nd(o + 1),
                    nd(o + 2)
                )
                .ok()?;
                o += 3;
            }
            'R' => {
                writeln!(
                    body,
                    "            q = quat_rotate(q, vec4<f32>({}, {}, {}, {}));",
                    nd(o),
                    nd(o + 1),
                    nd(o + 2),
                    nd(o + 3)
                )
                .ok()?;
                o += 4;
            }
            _ => {
                writeln!(body, "            q = q * {};", nd(o)).ok()?;
                factors.push(nd(o + 1));
                o += 2;
            }
        }
    }
    let off = format!("i32({})", nd(ROW_HEAD));
    match k {
        Kind::Polygon => {
            let hh = nd(ROW_HEAD + 1);
            write!(
                body,
                "            let ld2d: f32 = alice_poly_nd(q.xy, {off});
            let ldz: f32 = abs(q.z) - {hh};
            let lwx: f32 = max(ld2d, 0.0);
            let lwy: f32 = max(ldz, 0.0);
            var lv: f32 = sqrt(lwx*lwx + lwy*lwy) + min(max(ld2d, ldz), 0.0);
"
            )
            .ok()?;
        }
    }
    for f in factors.iter().rev() {
        writeln!(body, "            lv = lv * {f};").ok()?;
    }
    let has_bound = rows[0].3.is_some();
    let guarded = |row: &str| {
        if !has_bound {
            return format!(
                "        let lr: i32 = {row};
        {{
{body}            d_ub = select(lv, min(d_ub, lv), d_ub_set);
            d_ub_set = true;
        }}
"
            );
        }
        format!(
            "        let lr: i32 = {row};
        let lc: vec3<f32> = vec3<f32>(alice_node_data[lr], alice_node_data[lr + 1], alice_node_data[lr + 2]);
        if (!d_ub_set || !(alice_node_data[lr + 3] * length(p - lc) - alice_node_data[lr + 4] - alice_node_data[lr + 5] * (1.0 + length(p) + alice_node_data[lr + 6] + alice_node_data[lr + 4]) > d_ub)) {{
{body}            d_ub = select(lv, min(d_ub, lv), d_ub_set);
            d_ub_set = true;
        }} else {{
            n_skip = n_skip + 1.0;
        }}
"
        )
    };
    let n = rows.len();
    #[cfg(test)]
    if UNROLL.with(std::cell::Cell::get) {
        let mut s = String::new();
        for r in 0..n {
            s.push_str("    {\n");
            s.push_str(&guarded(&lit_i(at + r * stride)));
            s.push_str("    }\n");
        }
        return Some(s);
    }
    Some(format!(
        "    for (var lk: i32 = 0; lk < {n}; lk = lk + 1) {{\n{}    }}\n",
        guarded(&format!("{at} + lk * {stride}"))
    ))
}

#[cfg(test)]
fn lit_i(v: usize) -> String {
    format!("{v}")
}

/// The kind's one function: (name, definition)
const fn kind_function(k: Kind) -> (&'static str, &'static str) {
    match k {
        // `ShaderLang::global_polygon_2d_fn` with the vertices read from the buffer at `b` (the
        // count first, then the pairs): the same edge order, clamp and even-odd sign rule
        Kind::Polygon => (
            "fn alice_poly_nd(",
            "fn alice_poly_nd(q: vec2<f32>, b: i32) -> f32 {
    let n: i32 = i32(alice_node_data[b]);
    let v0 = vec2<f32>(alice_node_data[b + 1], alice_node_data[b + 2]);
    var d = dot(q - v0, q - v0);
    var s = 1.0;
    var j = n - 1;
    for (var i = 0i; i < n; i = i + 1i) {
        let vi = vec2<f32>(alice_node_data[b + 1 + 2 * i], alice_node_data[b + 2 + 2 * i]);
        let vj = vec2<f32>(alice_node_data[b + 1 + 2 * j], alice_node_data[b + 2 + 2 * j]);
        let e = vj - vi;
        let w = q - vi;
        let bb = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        d = min(d, dot(bb, bb));
        let c1 = q.y >= vi.y;
        let c2 = q.y < vj.y;
        let c3 = e.x * w.y > e.y * w.x;
        if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) { s = -s; }
        j = i;
    }
    return s * sqrt(d);
}
",
        ),
    }
}

#[cfg(test)]
thread_local! {
    /// The test reference before this module: every operand in its per-node form
    pub(super) static OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
mod tests {
    use super::{OFF, UNROLL};
    use crate::compiled::{GpuEvaluator, TranspileMode, WgslShader};
    use crate::types::SdfNode;
    use glam::{Vec2, Vec3};

    fn star(n: usize, r: f32, dr: f32) -> Vec<Vec2> {
        (0..n)
            .map(|j| {
                let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / n as f32);
                let rr = if j % 2 == 0 { r } else { r - dr };
                Vec2::new(rr * c, rr * s)
            })
            .collect()
    }

    /// Unions of extruded polygons under each wrapper shape the loops take
    fn trees() -> Vec<(&'static str, SdfNode)> {
        let poly =
            |k: u32| SdfNode::polygon_2d(star(24, 0.05 + 0.003 * (k % 5) as f32, 0.02), 0.03);
        let mut out = Vec::new();
        let union = |v: Vec<SdfNode>| v.into_iter().reduce(SdfNode::union).expect("parts");
        out.push((
            "translated",
            union(
                (0..40)
                    .map(|k| poly(k).translate(0.12 * (k % 8) as f32, 0.12 * (k / 8) as f32, 0.0))
                    .collect(),
            ),
        ));
        out.push((
            "rotated and scaled",
            union(
                (0..24)
                    .map(|k| {
                        poly(k)
                            .scale(0.8 + 0.05 * (k % 4) as f32)
                            .rotate_euler(0.3 * k as f32, 0.1, 0.0)
                            .translate(0.15 * (k % 6) as f32, 0.15 * (k / 6) as f32, 0.02)
                    })
                    .collect(),
            ),
        ));
        out.push((
            "mixed with other nodes",
            union(vec![
                poly(1).translate(0.1, 0.0, 0.0),
                SdfNode::sphere(0.05).translate(0.3, 0.1, 0.0),
                poly(2).translate(-0.1, 0.05, 0.0),
                SdfNode::box3d(0.03, 0.03, 0.03).translate(0.0, 0.2, 0.0),
                poly(3),
            ]),
        ));
        out
    }

    fn points() -> Vec<Vec3> {
        let mut s: u64 = 0x5EED_0B1A;
        let mut r = move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32) / ((1u64 << 24) as f32)
        };
        (0..4096)
            .map(|_| Vec3::new(r() * 1.2 - 0.2, r() * 1.0 - 0.2, (r() - 0.5) * 0.2))
            .collect()
    }

    fn values(node: &SdfNode, pts: &[Vec3], unroll: bool, off: bool) -> (String, Vec<f32>) {
        UNROLL.with(|c| c.set(unroll));
        OFF.with(|c| c.set(off));
        let shader = WgslShader::try_transpile_with_node_data(node, TranspileMode::Hardcoded);
        UNROLL.with(|c| c.set(false));
        OFF.with(|c| c.set(false));
        let shader = shader.expect("transpile");
        let gpu = GpuEvaluator::from_shader(&shader).expect("evaluator");
        (shader.source, gpu.eval_batch(pts).expect("gpu"))
    }

    /// The loop and the same rows with one call site per operand give the same bits on the
    /// adapter that runs the test; the per-node form before the loops is compared and reported
    #[test]
    fn test_gpu_leaf_loop_gives_the_bits_of_one_call_site_per_operand() {
        if !crate::gpu_test_support::gpu_ready(
            "test_gpu_leaf_loop_gives_the_bits_of_one_call_site_per_operand",
        ) {
            return;
        }
        let pts = points();
        let mut compared = 0usize;
        for (name, node) in trees() {
            let (looped_src, looped) = values(&node, &pts, false, false);
            let (unrolled_src, unrolled) = values(&node, &pts, true, false);
            let (_, before) = values(&node, &pts, false, true);
            assert!(
                looped_src.contains("for (var lk"),
                "{name}: no loop in the shader"
            );
            assert!(
                !unrolled_src.contains("for (var lk"),
                "{name}: the reference has the loop"
            );
            assert_eq!(looped.len(), pts.len(), "{name}");
            assert_eq!(unrolled.len(), pts.len(), "{name}");
            let differ = looped
                .iter()
                .zip(&unrolled)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let differ_before = looped
                .iter()
                .zip(&before)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst_before = looped
                .iter()
                .zip(&before)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let cpu: Vec<f32> = pts.iter().map(|p| crate::eval::eval(&node, *p)).collect();
            let gap = |v: &[f32]| {
                v.iter()
                    .zip(&cpu)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            };
            let (gap_before, gap_after) = (gap(&before), gap(&looped));
            eprintln!(
                "{name}: {} points, loop vs one call site per operand {differ} differ, vs the per-node form {differ_before} differ (worst {worst_before:e}); max |GPU - CPU| per-node form {gap_before:e}, loop {gap_after:e}",
                pts.len()
            );
            assert!(
                gap_after <= gap_before,
                "{name}: the loop is further from the CPU ({gap_after:e}) than the per-node form ({gap_before:e})"
            );
            assert!(
                looped.iter().any(|v| *v < 0.0) && looped.iter().any(|v| *v > 0.0),
                "{name}: no point inside or none outside"
            );
            assert_eq!(differ, 0, "{name}: the loop changes bits");
            assert!(
                worst_before < 1e-5,
                "{name}: the per-node form is {worst_before} away"
            );
            compared += pts.len();
        }
        assert!(compared > 0, "compared nothing");
    }
}
