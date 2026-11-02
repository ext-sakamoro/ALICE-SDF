//! The smooth section stack's law as shader source (GLSL, HLSL, WGSL; MSL through the WGSL emit).
//!
//! Written in the loft template's WGSL-like notation and rendered per dialect by the same rewrite
//! ([`crate::compiled::loft_shader::render_template`]), so every language carries the statements of
//! [`crate::primitives::SmoothSectionStack::eval`] in the same order: the level search, the Hermite
//! weights, per level the even-odd signed distance to its loops (the shared closed-curve closest
//! point: the span scan with the control-point circle bound, Newton within one scan step, de Boor
//! on uniform knots), the bound's knots, the continuous clamp and the end extrusion.
//!
//! The data is one module-scope float array ([`flatten`]): a head `[n, lmax, r0, eps]`, the levels,
//! the knots, per level the offset of its descriptor, then per level `[count, (m, offset) × count]`
//! and the control points `(x, z)` of every loop.

use crate::compiled::loft_shader::render_template;
use crate::compiled::transpiler_common::lit;
use crate::primitives::loft::{CURVE_NEWTON, CURVE_SCAN};
use crate::primitives::SmoothSectionStack;

/// The flat block of a stack (see the module doc)
#[must_use]
pub fn flatten(st: &SmoothSectionStack) -> Vec<f32> {
    let n = st.levels.len();
    let (lmax, r0, eps) = st.shader_params();
    let mut out = vec![n as f32, lmax, r0, eps];
    out.extend_from_slice(&st.levels);
    out.extend_from_slice(st.knots());
    let desc_at = out.len();
    out.extend(std::iter::repeat_n(0.0, n));
    let mut descs = Vec::with_capacity(n);
    for lv in &st.loops {
        descs.push(out.len());
        out.push(lv.len() as f32);
        let slots = out.len();
        out.extend(std::iter::repeat_n(0.0, 2 * lv.len()));
        for (k, lp) in lv.iter().enumerate() {
            out[slots + 2 * k] = lp.len() as f32;
            out[slots + 2 * k + 1] = out.len() as f32;
            for p in lp {
                out.push(p.x);
                out.push(p.y);
            }
        }
    }
    for (i, d) in descs.into_iter().enumerate() {
        out[desc_at + i] = d as f32;
    }
    out
}

fn template(name: &str, arr: &str, flat: &[f32]) -> String {
    let n = flat[0] as usize;
    let (lv, kn, ds) = (4, 4 + n, 4 + 2 * n);
    let lfar = lit(flat[1] * (1.0 + 2.0 * flat[3]));
    let r0 = lit(flat[2]);
    let e2 = lit(1.0 + 2.0 * flat[3]);
    let (y1, yl) = (lit(flat[lv + 1]), lit(flat[lv + n - 2]));
    let scan = CURVE_SCAN;
    let scanf = lit(CURVE_SCAN as f32);
    let newton = CURVE_NEWTON;
    let nm3 = n - 3;
    format!(
        "fn {name}_pt(o: I, m: I, j: I) -> V2 {{
    var k: I = ((j % m) + m) % m;
    return V2({arr}[o + 2 * k], {arr}[o + 2 * k + 1]);
}}
fn {name}_span(o: I, m: I, j: I, t: F, which: I) -> V2 {{
    var d0: V2 = {name}_pt(o, m, j - 1);
    var d1: V2 = {name}_pt(o, m, j);
    var d2: V2 = {name}_pt(o, m, j + 1);
    var d3: V2 = {name}_pt(o, m, j + 2);
    var u0: F = F(j - 2);
    var u1: F = F(j - 1);
    var u2: F = F(j);
    var u3: F = F(j + 1);
    var u4: F = F(j + 2);
    var u5: F = F(j + 3);
    var q0: V2 = (d1 - d0) * (3.0 / (u3 - u0));
    var q1: V2 = (d2 - d1) * (3.0 / (u4 - u1));
    var q2: V2 = (d3 - d2) * (3.0 / (u5 - u2));
    var r0: V2 = (q1 - q0) * (2.0 / (u3 - u1));
    var r1: V2 = (q2 - q1) * (2.0 / (u4 - u2));
    var a: F = 0.0;
    if (which == 2) {{
        a = (t - u2) / (u3 - u2);
        return r0 * (1.0 - a) + r1 * a;
    }}
    if (which == 1) {{
        a = (t - u2) / (u4 - u2);
        q2 = q1 * (1.0 - a) + q2 * a;
        a = (t - u1) / (u3 - u1);
        q1 = q0 * (1.0 - a) + q1 * a;
        a = (t - u2) / (u3 - u2);
        return q1 * (1.0 - a) + q2 * a;
    }}
    a = (t - u2) / (u5 - u2);
    d3 = d2 * (1.0 - a) + d3 * a;
    a = (t - u1) / (u4 - u1);
    d2 = d1 * (1.0 - a) + d2 * a;
    a = (t - u0) / (u3 - u0);
    d1 = d0 * (1.0 - a) + d1 * a;
    a = (t - u2) / (u4 - u2);
    d3 = d2 * (1.0 - a) + d3 * a;
    a = (t - u1) / (u3 - u1);
    d2 = d1 * (1.0 - a) + d2 * a;
    a = (t - u2) / (u3 - u2);
    return d2 * (1.0 - a) + d3 * a;
}}
fn {name}_on(t: F, k0: F, k1: F, bj: I) -> I {{
    if (t < k0) {{ return bj - 1; }}
    if (t > k1) {{ return bj + 1; }}
    return bj;
}}
fn {name}_cd(o: I, m: I, q: V2) -> F {{
    var bj: I = 0;
    var bt: F = 0.0;
    var be: F = 3.0e38;
    for (var j: I = 0; j < m; j++) {{
        var p0: V2 = {name}_pt(o, m, j - 1);
        var p1: V2 = {name}_pt(o, m, j);
        var p2: V2 = {name}_pt(o, m, j + 1);
        var p3: V2 = {name}_pt(o, m, j + 2);
        var c: V2 = (p0 + p1 + p2 + p3) * 0.25;
        var rad: F = max(max(max(length(p0 - c), length(p1 - c)), length(p2 - c)), length(p3 - c));
        var lb: F = length(q - c) - rad;
        if (lb > 0.0 && lb * lb >= be) {{ continue; }}
        var k0: F = F(j);
        var k1: F = F(j + 1);
        for (var k: I = 0; k < {scan}; k++) {{
            var t: F = k0 + (k1 - k0) * (F(k) / {scanf});
            var r: V2 = {name}_span(o, m, j, t, 0) - q;
            var e: F = dot(r, r);
            if (e < be) {{ be = e; bt = t; bj = j; }}
        }}
    }}
    var k0: F = F(bj);
    var k1: F = F(bj + 1);
    var lo: F = max(bt - (k1 - k0) / {scanf}, F(bj - 1));
    var hi: F = min(bt + (k1 - k0) / {scanf}, F(bj + 2));
    var t: F = bt;
    for (var it: I = 0; it < {newton}; it++) {{
        var jj: I = {name}_on(t, k0, k1, bj);
        var c: V2 = {name}_span(o, m, jj, t, 0);
        var d: V2 = {name}_span(o, m, jj, t, 1);
        var dd: V2 = {name}_span(o, m, jj, t, 2);
        var r: V2 = c - q;
        var g: F = dot(r, d);
        var gp: F = dot(d, d) + dot(r, dd);
        if (gp > 0.0) {{ t = min(max(t - g / gp, lo), hi); }}
    }}
    var jf: I = {name}_on(t, k0, k1, bj);
    var cc: V2 = {name}_span(o, m, jf, t, 0);
    var dc: V2 = {name}_span(o, m, jf, t, 1);
    var rc: V2 = cc - q;
    if (dot(rc, rc) > be) {{
        cc = {name}_span(o, m, bj, bt, 0);
        dc = {name}_span(o, m, bj, bt, 1);
    }}
    var rr: V2 = q - cc;
    var dist: F = length(rr);
    if (dc.x * rr.y - dc.y * rr.x > 0.0) {{ return -dist; }}
    return dist;
}}
fn {name}_lvl(l: I, q: V2) -> F {{
    var dsc: I = I({arr}[{ds} + l]);
    var cnt: I = I({arr}[dsc]);
    var best: F = 3.0e38;
    var sg: F = 1.0;
    for (var k: I = 0; k < cnt; k++) {{
        var m: I = I({arr}[dsc + 1 + 2 * k]);
        var o: I = I({arr}[dsc + 2 + 2 * k]);
        var d: F = {name}_cd(o, m, q);
        if (d < 0.0) {{ sg = -sg; }}
        best = min(best, abs(d));
    }}
    return sg * best;
}}
fn {name}(p: V3) -> F {{
    var q: V2 = V2(p.x, p.z);
    var yc: F = min(max(p.y, {y1}), {yl});
    var i: I = 1;
    for (var k: I = 1; k < {nm3}; k++) {{
        if (yc > {arr}[{lv} + i + 1]) {{ i = i + 1; }} else {{ break; }}
    }}
    var ya: F = {arr}[{lv} + i - 1];
    var yb: F = {arr}[{lv} + i];
    var yd: F = {arr}[{lv} + i + 1];
    var ye: F = {arr}[{lv} + i + 2];
    var t: F = min(max((yc - yb) / (yd - yb), 0.0), 1.0);
    var dl: F = yd - yb;
    var a1: F = dl / (yd - ya);
    var a2: F = dl / (ye - yb);
    var t2: F = t * t;
    var t3: F = t * t * t;
    var h00: F = 2.0 * t3 - 3.0 * t2 + 1.0;
    var h10: F = t3 - 2.0 * t2 + t;
    var h01: F = -2.0 * t3 + 3.0 * t2;
    var h11: F = t3 - t2;
    var f: F = 0.0;
    f = f + (-h10 * a1) * {name}_lvl(i - 1, q);
    f = f + (h00 - h11 * a2) * {name}_lvl(i, q);
    f = f + (h01 + h10 * a1) * {name}_lvl(i + 1, q);
    f = f + (h11 * a2) * {name}_lvl(i + 2, q);
    var l: F = ((1.0 - t) * {arr}[{kn} + i] + t * {arr}[{kn} + i + 1]) * {e2};
    var d2: F = f / l;
    if (abs(f) > {r0} * l) {{ d2 = sign(f) * ({r0} + (abs(f) - {r0} * l) / {lfar}); }}
    var dw: F = max({y1} - p.y, p.y - {yl});
    var ow: V2 = V2(max(d2, 0.0), max(dw, 0.0));
    return min(max(d2, dw), 0.0) + length(ow);
}}
"
    )
}

/// The stack's module-scope functions in `dialect` over the array `arr`; the field is `{name}(p)`
#[must_use]
pub fn stack_functions(name: &str, arr: &str, flat: &[f32], dialect: &str) -> String {
    render_template(&template(name, arr, flat), dialect)
}
