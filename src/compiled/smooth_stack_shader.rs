//! The smooth section stack's law as shader source (GLSL, HLSL, WGSL; MSL through the WGSL emit).
//!
//! Written in the loft template's WGSL-like notation and rendered per dialect by the same rewrite
//! ([`crate::compiled::loft_shader::render_template`]), so every language carries the statements of
//! [`crate::primitives::SmoothSectionStack::eval`] in the same order: the level search, the Hermite
//! weights, per level the even-odd signed distance to its loops (the loft's closed-curve closest point,
//! [`crate::compiled::loft_shader::closed_curve_shader`], over this stack's control points and uniform
//! knots), the bound's knots, the continuous clamp and the end extrusion.
//!
//! The data is one module-scope float array ([`flatten`]): a head `[n, lmax, r0, eps]`, the levels,
//! the knots, per level the offset of its descriptor, then per level `[count, (m, offset) × count]`
//! and the control points `(x, z)` of every loop.

use crate::compiled::loft_shader::{closed_curve_shader, render_template};
use crate::compiled::transpiler_common::lit;
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
    // the loops' constant trip counts: the most loops on a level and the most control points in a
    // loop (the data's counts stop the loops earlier; bad data cannot run them past these)
    let (mut lmax, mut mmax) = (1usize, 1usize);
    for l in 0..n {
        let d = flat[ds + l] as usize;
        let cnt = flat[d] as usize;
        lmax = lmax.max(cnt);
        for k in 0..cnt {
            mmax = mmax.max(flat[d + 1 + 2 * k] as usize);
        }
    }
    let curve = closed_curve_shader(name, "o: I, m: I", "o, m", "m", mmax, &format!("{name}_pt"), &format!("{name}_sk"));
    let nm3 = n - 3;
    format!(
        "fn {name}_pt(o: I, m: I, j: I) -> V2 {{
    var k: I = j - m * I(floor(F(j) / F(m)));
    return V2({arr}[o + 2 * k], {arr}[o + 2 * k + 1]);
}}
fn {name}_sk(i: I) -> F {{
    return F(i);
}}
{curve}fn {name}_lvl(l: I, q: V2) -> F {{
    var dsc: I = I({arr}[{ds} + l]);
    var cnt: I = I({arr}[dsc]);
    var best: F = 3.0e38;
    var sg: F = 1.0;
    for (var k: I = 0; k < {lmax}; k++) {{
        if (k >= cnt) {{ break; }}
        var m: I = I({arr}[dsc + 1 + 2 * k]);
        var o: I = I({arr}[dsc + 2 + 2 * k]);
        var d: F = {name}_ccd(o, m, q);
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
