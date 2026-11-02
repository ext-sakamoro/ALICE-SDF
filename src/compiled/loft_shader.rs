//! The loft's law as shader source (GLSL, HLSL, WGSL; MSL through the WGSL emit).
//!
//! One template, written in WGSL-like notation, is rendered per dialect, so the three languages
//! carry the same statements as each other and as [`crate::primitives::loft`] (same operations in
//! the same order: the axis Catmull-Rom, the fixed closest-point scan and ternary steps, the frame,
//! the rotation recurrence, the B-spline weights, the extrusion and the domes). The loft's data is
//! the flat block of [`crate::primitives::loft::flatten`], a module-scope array.

use crate::compiled::transpiler_common::lit;
use crate::primitives::loft::{AXIS_NEWTON, CURVE_NEWTON, CURVE_SCAN, HEAD, PIECE_FLOATS, PIECE_PLANES, SCAN_PER_SEGMENT};

/// The statements of one loft, `fn {name}(p: V3) -> F` plus helpers, over the array `{arr}`
fn template(name: &str, arr: &str, flat: &[f32], split: bool) -> String {
    let curve = flat[15] != 0.0;
    let n = flat[0] as usize;
    let kh = flat[1] as usize;
    let kr = flat[2] as usize;
    let up = format!("V3({}, {}, {})", lit(flat[3]), lit(flat[4]), lit(flat[5]));
    let caps = [flat[6], flat[7]];
    let (l, l2, core) = (lit(flat[8]), lit(flat[9]), lit(flat[10]));
    let lq = [lit(flat[11]), lit(flat[12])];
    let last = lit((n - 1) as f32);
    let a0 = HEAD;
    let rb = HEAD + 3 * n;
    let w = if curve { 2 * kh } else { 2 * kh + 1 };
    let kb = rb + kr * w;
    let ax = |i: usize| format!("V3({arr}[{}], {arr}[{}], {arr}[{}])", a0 + 3 * i, a0 + 3 * i + 1, a0 + 3 * i + 2);
    let h = lit(1.0 / SCAN_PER_SEGMENT as f32);
    let nseg = n - 1;
    let per = SCAN_PER_SEGMENT;
    let perm1 = SCAN_PER_SEGMENT - 1;
    let mut s = String::new();
    s += &format!(
        "fn {name}_ap(i: I) -> V3 {{
    if (i < 0) {{ return 2.0 * {} - {}; }}
    if (i >= {n}) {{ return 2.0 * {} - {}; }}
    return V3({arr}[{a0} + 3 * i], {arr}[{a0} + 3 * i + 1], {arr}[{a0} + 3 * i + 2]);
}}
",
        ax(0),
        ax(1),
        ax(n - 1),
        ax(n - 2)
    );
    for (suffix, ret) in [("pos", "0.5 * (2.0 * b + k1 * t + k2 * t2 + k3 * t3)"), ("der", "0.5 * (k1 + 2.0 * k2 * t + 3.0 * k3 * t2)"), ("dd", "k2 + 3.0 * k3 * t")] {
        s += &format!(
            "fn {name}_{suffix}(s: F) -> V3 {{
    var ss: F = min(max(s, 0.0), {last});
    var i: I = min(I(ss), {nm2});
    var t: F = ss - F(i);
    var a: V3 = {name}_ap(i - 1);
    var b: V3 = {name}_ap(i);
    var c: V3 = {name}_ap(i + 1);
    var d: V3 = {name}_ap(i + 2);
    var t2: F = t * t;
    var t3: F = t2 * t;
    var k1: V3 = c - a;
    var k2: V3 = 2.0 * a - 5.0 * b + 4.0 * c - d;
    var k3: V3 = -a + 3.0 * b - 3.0 * c + d;
    return {ret};
}}
",
            nm2 = n - 2
        );
    }
    s += &format!(
        "fn {name}_sd(x: V3, b: V3, c: V3) -> F {{
    var e: V3 = c - b;
    var t: F = min(max(dot(x - b, e) / dot(e, e), 0.0), 1.0);
    return length(x - (b + e * t));
}}
fn {name}_cp(p: V3) -> F {{
    var bs: F = 0.0;
    var bd: F = 3.0e38;
    for (var i: I = 0; i < {nseg}; i++) {{
        var a: V3 = {name}_ap(i - 1);
        var b: V3 = {name}_ap(i);
        var c: V3 = {name}_ap(i + 1);
        var d: V3 = {name}_ap(i + 2);
        var r: F = max({name}_sd(b + (c - a) / 6.0, b, c), {name}_sd(c - (d - b) / 6.0, b, c));
        var margin: F = 0.00001 * (length(p - b) + length(c - b) + r);
        var lb: F = {name}_sd(p, b, c) - r - margin;
        if (lb > 0.0 && lb * lb > bd) {{ continue; }}
        var k0: I = i * {per};
        var k1: I = k0 + {perm1};
        if (i + 1 == {nseg}) {{ k1 = k0 + {per}; }}
        for (var k: I = k0; k <= k1; k++) {{
            var s: F = F(k) * {h};
            var q: V3 = {name}_pos(s) - p;
            var e: F = dot(q, q);
            if (e < bd || (e == bd && s < bs)) {{ bd = e; bs = s; }}
        }}
    }}
    var lo: F = max(bs - {h}, 0.0);
    var hi: F = min(bs + {h}, {last});
    var sn: F = bs;
    for (var it: I = 0; it < {AXIS_NEWTON}; it++) {{
        var cn: V3 = {name}_pos(sn);
        var dn: V3 = {name}_der(sn);
        var rn: V3 = cn - p;
        var g: F = dot(rn, dn);
        var gp: F = dot(dn, dn) + dot(rn, {name}_dd(sn));
        if (gp > 0.0) {{ sn = min(max(sn - g / gp, lo), hi); }}
    }}
    var best: F = sn;
    var qb: V3 = {name}_pos(best) - p;
    var be: F = dot(qb, qb);
    if (lo == 0.0 || hi == 0.0) {{
        var q0: V3 = {name}_pos(0.0) - p;
        if (dot(q0, q0) < be) {{ be = dot(q0, q0); best = 0.0; }}
    }}
    if (lo == {last} || hi == {last}) {{
        var q9: V3 = {name}_pos({last}) - p;
        if (dot(q9, q9) < be) {{ be = dot(q9, q9); best = {last}; }}
    }}
    return best;
}}
fn {name}_row(j: I, c1: F, s1: F) -> F {{
    var o: I = {rb} + j * {w};
    var r: F = {arr}[o];
    var ck: F = c1;
    var sk: F = s1;
    for (var k: I = 0; k < {kh}; k++) {{
        r = r + ({arr}[o + 1 + 2 * k] * ck + {arr}[o + 2 + 2 * k] * sk);
        var nc: F = ck * c1 - sk * s1;
        sk = sk * c1 + ck * s1;
        ck = nc;
    }}
    return r;
}}
fn {name}_rr(i: I, c1: F, s1: F) -> F {{
    if (i < 0) {{ return 2.0 * {name}_row(0, c1, s1) - {name}_row(1, c1, s1); }}
    if (i >= {kr}) {{ return 2.0 * {name}_row({krm1}, c1, s1) - {name}_row({krm2}, c1, s1); }}
    return {name}_row(i, c1, s1);
}}
fn {name}_ll(s: F) -> F {{
    var w: F = s * {kr1f} / {n1f};
    w = min(max(w, 0.0), {kr1f});
    var i: I = min(I(w), {krm2});
    var t: F = w - F(i);
    return {arr}[{kb} + i] + ({arr}[{kb} + i + 1] - {arr}[{kb} + i]) * t;
}}
fn {name}_rad(s: F, c1: F, s1: F) -> F {{
    var w: F = s * {kr1f} / {n1f};
    w = min(max(w, 0.0), {kr1f});
    var i: I = min(I(w), {krm2});
    var t: F = w - F(i);
    var t2: F = t * t;
    var t3: F = t2 * t;
    var u: F = 1.0 - t;
    var b0: F = u * u * u / 6.0;
    var b1: F = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
    var b2: F = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
    var b3: F = t3 / 6.0;
    return b0 * {name}_rr(i - 1, c1, s1) + b1 * {name}_rr(i, c1, s1) + b2 * {name}_rr(i + 1, c1, s1) + b3 * {name}_rr(i + 2, c1, s1);
}}
",
        krm1 = kr - 1,
        krm2 = kr - 2,
        kr1f = lit((kr - 1) as f32),
        n1f = lit((n - 1) as f32),
    );
    if curve {
        s += &curve_functions(name, arr, kh, kr, rb, w, kb + kr, n);
    }
    let frame = |pfx: &str| {
        format!(
            "    var t{pfx}: V3 = normalize(dr{pfx});
    var nn{pfx}: V3 = normalize({up} - t{pfx} * dot({up}, t{pfx}));
    var bb{pfx}: V3 = cross(t{pfx}, nn{pfx});
"
        )
    };
    for e in 0..2 {
        if caps[e] <= 0.0 {
            continue;
        }
        let (se, sign, row) = if e == 0 { (lit(0.0), "-", 0) } else { (last.clone(), "", kr - 1) };
        s += &format!(
            "fn {name}_dome{e}(p: V3) -> F {{
    var c: V3 = {name}_pos({se});
    var dr: V3 = {name}_der({se});
{fr}    var o: V3 = {sign}t;
    var r: V3 = p - c;
    var h: F = dot(r, o);
    var u: F = dot(r, nn);
    var v: F = dot(r, bb);
    var rho: F = sqrt(u * u + v * v);
    var c1: F = 1.0;
    var s1: F = 0.0;
    if (rho > 0.0) {{ c1 = u / rho; s1 = v / rho; }}
    var re: F = {name}_row({row}, c1, s1);
    var a: F = rho / re;
    var bq: F = h / {cap};
    var q: F = sqrt(a * a + bq * bq);
    return max((q - 1.0) / {lqe}, -h);
}}
",
            fr = frame(""),
            cap = lit(caps[e]),
            lqe = lq[e]
        );
    }
    s += &format!(
        "fn {name}(p: V3) -> F {{
    var s: F = {name}_cp(p);
    var c: V3 = {name}_pos(s);
    var dr: V3 = {name}_der(s);
{fr}    var r: V3 = p - c;
    var u: F = dot(r, nn);
    var v: F = dot(r, bb);
    var rho: F = sqrt(u * u + v * v);
    var c1: F = 1.0;
    var s1: F = 0.0;
    if (rho > 0.0) {{ c1 = u / rho; s1 = v / rho; }}
{d2}    var c0: V3 = {name}_pos(0.0);
    var e0: V3 = {name}_der(0.0);
    var ce: V3 = {name}_pos({last});
    var ee: V3 = {name}_der({last});
    var dw: F = max(dot(c0 - p, normalize(e0)), dot(p - ce, normalize(ee)));
    var o: V2 = V2(max(d2, 0.0), max(dw, 0.0));
    var bigf: F = min(max(d2, dw), 0.0) + length(o);
    var f: F = bigf / {l};
{corr}",
        fr = frame(""),
        corr = correction(name, flat),
        d2 = if curve {
            format!("    var d2: F = {name}_cd(s, V2(u, v));\n")
        } else {
            format!("    var g: F = max(rho - {name}_rad(s, c1, s1), {core});\n    var d2: F = g / {l2};\n")
        }
    );
    for e in 0..2 {
        if caps[e] > 0.0 {
            s += &format!("    f = min(f, {name}_dome{e}(p));\n");
        }
    }
    s += &far_tail(name, flat);
    // the far bound is declared before the field that calls it (GLSL / HLSL need it first)
    let at = s.find(&format!("fn {name}(p: V3) -> F {{")).expect("the field function");
    s.insert_str(at, &far_function(name, arr, flat, pieces_at(flat), split));
    s
}

/// No fade: the flat block flags it with a negative fade start (`[17] = [18] = −1`)
fn no_fade(flat: &[f32]) -> bool {
    flat[17] < 0.0
}

/// The local correction `(1/L(s) − 1/L_g)·sat(F)·ψ` (only when `m > 0`), in the CPU's order:
/// `sat = F / √(1 + (F/m)²)`, `ψ` a smoothstep from 1 at `|p − C(s)| ≤ fade₀` to 0 at `≥ fade₁`
/// (with no fade, flagged by `[17] < 0`, ψ is 1 and is resolved here)
fn correction(name: &str, flat: &[f32]) -> String {
    let (lg, m, f0, f1) = (flat[8], flat[13], flat[17], flat[18]);
    if !(m > 0.0) {
        return String::new();
    }
    let psi = if no_fade(flat) {
        "    var psi: F = 1.0;\n".to_string()
    } else {
        format!(
            "    var psi: F = 0.0;\n    if (dist <= {f0}) {{ psi = 1.0; }} else {{ if (dist >= {f1}) {{ psi = 0.0; }} else {{\n        var uu: F = ({f1} - dist) / ({f1} - {f0});\n        psi = uu * uu * (3.0 - 2.0 * uu);\n    }} }}\n",
            f0 = lit(f0),
            f1 = lit(f1)
        )
    };
    format!(
        "    var x: F = bigf / {m};\n    var sat: F = bigf / sqrt(1.0 + x * x);\n    var dist: F = length(r);\n{psi}    f = f + (1.0 / {name}_ll(s) - 1.0 / {lg}) * sat * psi;\n",
        m = lit(m),
        lg = lit(lg)
    )
}

/// The end of the field: the far bound of the plane pieces, outside only, lowered by
/// `τ = max(m/4, R_b/80)` (computed here in f32 as the CPU does)
fn far_tail(name: &str, flat: &[f32]) -> String {
    let pieces = flat[16] as usize;
    if pieces == 0 {
        return "    return f;\n}\n".to_string();
    }
    let tau = (0.25 * flat[13]).max(0.0125 * flat[14]);
    format!("    if (f <= 0.0) {{ return f; }}\n    return max(f, {name}_fb(p) - {tau});\n}}\n", tau = lit(tau))
}

/// cos and sin of 2πj/16, the CPU far bound's `DIR16` (the same literals)
const DIR16: [(f32, f32); 16] = [
    (1.0, 0.0),
    (0.923_879_5, 0.382_683_43),
    (0.707_106_77, 0.707_106_77),
    (0.382_683_43, 0.923_879_5),
    (0.0, 1.0),
    (-0.382_683_43, 0.923_879_5),
    (-0.707_106_77, 0.707_106_77),
    (-0.923_879_5, 0.382_683_43),
    (-1.0, 0.0),
    (-0.923_879_5, -0.382_683_43),
    (-0.707_106_77, -0.707_106_77),
    (-0.382_683_43, -0.923_879_5),
    (0.0, -1.0),
    (0.382_683_43, -0.923_879_5),
    (0.707_106_77, -0.707_106_77),
    (0.923_879_5, -0.382_683_43),
];

/// `plane_normal(k, t, n, b)` of the CPU, as an expression over the piece frame `t, n, b`
fn plane_normal_expr(k: usize) -> String {
    let inv = lit(0.707_106_77);
    match k {
        48 => "t".into(),
        49 => "(-t)".into(),
        _ => {
            let (c, sn) = DIR16[k % 16];
            let d = format!("(n * {} + b * {})", lit(c), lit(sn));
            match k / 16 {
                0 => d,
                1 => format!("(({d} + t) * {inv})"),
                _ => format!("(({d} - t) * {inv})"),
            }
        }
    }
}

/// Where the pieces start in the flat block (after the axis, rows, knots and, for curve sections,
/// the section knots): the array emit and the template both use this
fn pieces_at(flat: &[f32]) -> usize {
    let (n, kh, kr) = (flat[0] as usize, flat[1] as usize, flat[2] as usize);
    let curve = flat[15] != 0.0;
    let w = if curve { 2 * kh } else { 2 * kh + 1 };
    HEAD + 3 * n + kr * w + kr + if curve { kh + 1 } else { 0 }
}

/// `far_bound(p, pieces)` of the CPU: min over pieces of max over its planes of `n·p − h`, a piece
/// skipped when the subset 0, 4, 8, 12, 48, 49 already reaches the least so far. The planes are a
/// loop over the direction table appended to the data array at `dir` (normal `d = n·c + b·s`, then
/// `d`, `(d + t)/√2` or `(d − t)/√2` by `k / 16`; 48 is `t`, 49 is `−t`: the CPU's `plane_normal`)
fn far_function(name: &str, arr: &str, flat: &[f32], at: usize, split: bool) -> String {
    let pieces = flat[16] as usize;
    if pieces == 0 {
        return String::new();
    }
    if split {
        return far_function_split(name, arr, pieces);
    }
    let dir = flat.len();
    let inv = lit(0.707_106_77);
    let mut lb = String::new();
    for k in [0usize, 4, 8, 12, 48, 49] {
        lb += &format!("        lb = max(lb, dot({}, p) - {arr}[o + {}]);\n", plane_normal_expr(k), 9 + k);
    }
    format!(
        "fn {name}_pn(k: I, t: V3, n: V3, b: V3) -> V3 {{
    if (k == 48) {{ return t; }}
    if (k == 49) {{ return -t; }}
    var j: I = k - 16 * (k / 16);
    var d: V3 = n * {arr}[{dir} + 2 * j] + b * {arr}[{dir} + 2 * j + 1];
    if (k < 16) {{ return d; }}
    if (k < 32) {{ return (d + t) * {inv}; }}
    return (d - t) * {inv};
}}
fn {name}_fb(p: V3) -> F {{
    var best: F = 3.0e38;
    for (var q: I = 0; q < {pieces}; q++) {{
        var o: I = {at} + q * {pf};
        var t: V3 = V3({arr}[o], {arr}[o + 1], {arr}[o + 2]);
        var n: V3 = V3({arr}[o + 3], {arr}[o + 4], {arr}[o + 5]);
        var b: V3 = V3({arr}[o + 6], {arr}[o + 7], {arr}[o + 8]);
        var lb: F = -3.0e38;
{lb}        if (lb >= best) {{ continue; }}
        var v: F = lb;
        for (var k: I = 0; k < {planes}; k++) {{
            v = max(v, dot({name}_pn(k, t, n, b), p) - {arr}[o + 9 + k]);
        }}
        best = min(best, v);
    }}
    return best;
}}
",
        pf = PIECE_FLOATS,
        planes = PIECE_PLANES
    )
}

/// The far bound for the C-like dialects: each piece's frame and offsets are their own small array
/// `{arr}_p{q}` and the directions `{arr}_dir` (see [`loft_array`]); one block per piece in the
/// CPU's order, each indexing only its own 59 floats. A single large module array indexed at run
/// time is copied per invocation by some GLSL compilers and exceeded the compute stack
fn far_function_split(name: &str, arr: &str, pieces: usize) -> String {
    let inv = lit(0.707_106_77);
    let mut s = format!(
        "fn {name}_pn(k: I, t: V3, n: V3, b: V3) -> V3 {{
    if (k == 48) {{ return t; }}
    if (k == 49) {{ return -t; }}
    var j: I = k - 16 * (k / 16);
    var d: V3 = n * {arr}_dir[2 * j] + b * {arr}_dir[2 * j + 1];
    if (k < 16) {{ return d; }}
    if (k < 32) {{ return (d + t) * {inv}; }}
    return (d - t) * {inv};
}}
fn {name}_fb(p: V3) -> F {{
    var best: F = 3.0e38;
    var t: V3 = V3(0.0, 0.0, 0.0);
    var n: V3 = V3(0.0, 0.0, 0.0);
    var b: V3 = V3(0.0, 0.0, 0.0);
    var lb: F = 0.0;
    var v: F = 0.0;
"
    );
    for q in 0..pieces {
        let a = format!("{arr}_p{q}");
        s += &format!("    t = V3({a}[0], {a}[1], {a}[2]);\n    n = V3({a}[3], {a}[4], {a}[5]);\n    b = V3({a}[6], {a}[7], {a}[8]);\n    lb = -3.0e38;\n");
        for k in [0usize, 4, 8, 12, 48, 49] {
            s += &format!("    lb = max(lb, dot({}, p) - {a}[{}]);\n", plane_normal_expr(k), 9 + k);
        }
        s += &format!(
            "    if (lb < best) {{\n        v = lb;\n        for (var k: I = 0; k < {planes}; k++) {{\n            v = max(v, dot({name}_pn(k, t, n, b), p) - {a}[9 + k]);\n        }}\n        best = min(best, v);\n    }}\n",
            planes = PIECE_PLANES
        );
    }
    s += "    return best;\n}\n";
    s
}

/// Curve sections (mode 1): the statements of `Loft::sknot`, `row_point`, `row_weights`,
/// `section_point`, `curve_on_span` and `curve_distance`, in the same order. `sb` is the offset of
/// the section knots `K₀ … K_M` in the block
#[allow(clippy::too_many_arguments)]
fn curve_functions(name: &str, arr: &str, m: usize, kr: usize, rb: usize, w: usize, sb: usize, n: usize) -> String {
    let mut s = String::new();
    // periodic section knot: K[r] + wraps * period, r = i mod M (Euclidean); the floor of a float
    // quotient, since GLSL leaves `%` of negative operands undefined (exact: small integers, and a
    // quotient that is not an integer is at least 1/M from one)
    s += &format!(
        "fn {name}_sk(i: I) -> F {{
    var wraps: I = I(floor(F(i) / {mf}));
    var r: I = i - {m} * wraps;
    return {arr}[{sb} + r] + F(wraps) * ({arr}[{sbm}] - {arr}[{sb}]);
}}
fn {name}_rpa(r: I, j: I) -> V2 {{
    return V2({arr}[{rb} + r * {w} + 2 * j], {arr}[{rb} + r * {w} + 2 * j + 1]);
}}
fn {name}_rp(i: I, j: I) -> V2 {{
    if (i < 0) {{ return 2.0 * {name}_rpa(0, j) - {name}_rpa(1, j); }}
    if (i >= {kr}) {{ return 2.0 * {name}_rpa({krm1}, j) - {name}_rpa({krm2}, j); }}
    return {name}_rpa(i, j);
}}
fn {name}_spt(i0: I, b: V4, j: I) -> V2 {{
    var jj: I = j - {m} * I(floor(F(j) / {mf}));
    return b.x * {name}_rp(i0, jj) + b.y * {name}_rp(i0 + 1, jj) + b.z * {name}_rp(i0 + 2, jj) + b.w * {name}_rp(i0 + 3, jj);
}}
fn {name}_i0(s: F) -> I {{
    var w: F = s * {kr1f} / {n1f};
    w = min(max(w, 0.0), {kr1f});
    var i: I = min(I(w), {krm2});
    return i - 1;
}}
fn {name}_bw(s: F) -> V4 {{
    var w: F = s * {kr1f} / {n1f};
    w = min(max(w, 0.0), {kr1f});
    var i: I = min(I(w), {krm2});
    var t: F = w - F(i);
    var t2: F = t * t;
    var t3: F = t2 * t;
    var u: F = 1.0 - t;
    return V4(u * u * u / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0);
}}
",
        sbm = sb + m,
        mf = lit(m as f32),
        krm1 = kr - 1,
        krm2 = kr - 2,
        kr1f = lit((kr - 1) as f32),
        n1f = lit((n - 1) as f32),
    );
    s += &closed_curve_shader(name, "i0: I, b: V4", "i0, b", &m.to_string(), &format!("{name}_spt"), &format!("{name}_sk"));
    s += &format!(
        "fn {name}_cd(s: F, q: V2) -> F {{
    var i0: I = {name}_i0(s);
    var b: V4 = {name}_bw(s);
    return {name}_ccd(i0, b, q);
}}
"
    );
    s
}

/// The closed cubic B-spline curve's closest point and signed distance, the statements of
/// [`crate::primitives::loft::closed_curve_on_span`] and [`crate::primitives::loft::closed_curve_distance`]
/// in the same order, for any curve: `ctx_params` / `ctx_args` the context the accessors take,
/// `m` the number of spans (a literal or a variable among the context), `pt(ctx, j)` the control point
/// `j` (any integer, periodic) and `sk(i)` the knot `i` (the periodic extension). Emits `{name}_cs`
/// (point and derivative), `{name}_cdd`, `{name}_on` and `{name}_ccd(ctx, q)`, the signed distance
pub(crate) fn closed_curve_shader(name: &str, ctx_params: &str, ctx_args: &str, m: &str, pt: &str, sk: &str) -> String {
    let mut s = String::new();
    // de Boor on span j: point and first derivative (V4: xy point, zw derivative), and the second
    // derivative; unrolled exactly as the CPU loops run (r = 1..=3, i = 3 down to r)
    let mut deboor = String::new();
    deboor += "    var d0: V2 = PT(CTX, j - 1);\n    var d1: V2 = PT(CTX, j);\n    var d2: V2 = PT(CTX, j + 1);\n    var d3: V2 = PT(CTX, j + 2);\n";
    for k in 0..6 {
        deboor += &format!("    var u{k}: F = SK(j - 2 + {k});\n");
    }
    for k in 0..3 {
        deboor += &format!("    var q{k}: V2 = (d{k1} - d{k}) * (3.0 / (u{k3} - u{k}));\n", k1 = k + 1, k3 = k + 3);
    }
    deboor += "    var r0: V2 = (q1 - q0) * (2.0 / (u3 - u1));\n    var r1: V2 = (q2 - q1) * (2.0 / (u4 - u2));\n";
    deboor += "    var p0: V2 = d0;\n    var p1: V2 = d1;\n    var p2: V2 = d2;\n    var p3: V2 = d3;\n    var a: F = 0.0;\n";
    for r in 1..=3usize {
        for i in (r..=3).rev() {
            deboor += &format!(
                "    a = (t - u{im1}) / (u{hi} - u{im1});\n    p{i} = p{im1} * (1.0 - a) + p{i} * a;\n",
                im1 = i - 1,
                hi = i + 3 - r
            );
        }
    }
    for r in 1..=2usize {
        for i in (r..=2).rev() {
            deboor += &format!(
                "    a = (t - u{i}) / (u{hi} - u{i});\n    q{i} = q{im1} * (1.0 - a) + q{i} * a;\n",
                im1 = i - 1,
                hi = i + 3 - r
            );
        }
    }
    deboor += "    a = (t - u2) / (u3 - u2);\n";
    let deboor = deboor.replace("PT", pt).replace("SK", sk).replace("CTX", ctx_args);
    s += &format!("fn {name}_cs({ctx_params}, j: I, t: F) -> V4 {{\n{deboor}    return V4(p3.x, p3.y, q2.x, q2.y);\n}}\n");
    s += &format!("fn {name}_cdd({ctx_params}, j: I, t: F) -> V2 {{\n{deboor}    return r0 * (1.0 - a) + r1 * a;\n}}\n");
    let scan = CURVE_SCAN;
    let scanf = lit(CURVE_SCAN as f32);
    s += &format!(
        "fn {name}_on(t: F, k0: F, k1: F, bj: I) -> I {{
    if (t < k0) {{ return bj - 1; }}
    if (t > k1) {{ return bj + 1; }}
    return bj;
}}
fn {name}_ccd({ctx_params}, q: V2) -> F {{
    var bj: I = 0;
    var bt: F = {sk}(0);
    var be: F = 3.0e38;
    for (var j: I = 0; j < {m}; j++) {{
        var p0: V2 = {pt}({ctx_args}, j - 1);
        var p1: V2 = {pt}({ctx_args}, j);
        var p2: V2 = {pt}({ctx_args}, j + 1);
        var p3: V2 = {pt}({ctx_args}, j + 2);
        var c: V2 = (p0 + p1 + p2 + p3) * 0.25;
        var rad: F = max(max(max(length(p0 - c), length(p1 - c)), length(p2 - c)), length(p3 - c));
        var lb: F = length(q - c) - rad;
        if (lb > 0.0 && lb * lb >= be) {{ continue; }}
        var k0: F = {sk}(j);
        var k1: F = {sk}(j + 1);
        for (var k: I = 0; k < {scan}; k++) {{
            var t: F = k0 + (k1 - k0) * (F(k) / {scanf});
            var cs: V4 = {name}_cs({ctx_args}, j, t);
            var dq: V2 = V2(cs.x, cs.y) - q;
            var e: F = dot(dq, dq);
            if (e < be) {{ be = e; bt = t; bj = j; }}
        }}
    }}
    var k0: F = {sk}(bj);
    var k1: F = {sk}(bj + 1);
    var lo: F = bt - (k1 - k0) / {scanf};
    var hi: F = bt + (k1 - k0) / {scanf};
    lo = max(lo, {sk}(bj - 1));
    hi = min(hi, {sk}(bj + 2));
    var t: F = bt;
    for (var it: I = 0; it < {newton}; it++) {{
        var sj: I = {name}_on(t, k0, k1, bj);
        var cs: V4 = {name}_cs({ctx_args}, sj, t);
        var dd: V2 = {name}_cdd({ctx_args}, sj, t);
        var r: V2 = V2(cs.x, cs.y) - q;
        var d: V2 = V2(cs.z, cs.w);
        var g: F = dot(r, d);
        var gp: F = dot(d, d) + dot(r, dd);
        if (gp > 0.0) {{ t = min(max(t - g / gp, lo), hi); }}
    }}
    var cf: V4 = {name}_cs({ctx_args}, {name}_on(t, k0, k1, bj), t);
    var cc: V2 = V2(cf.x, cf.y);
    var dc: V2 = V2(cf.z, cf.w);
    var dqf: V2 = cc - q;
    if (dot(dqf, dqf) > be) {{
        var cb: V4 = {name}_cs({ctx_args}, bj, bt);
        cc = V2(cb.x, cb.y);
        dc = V2(cb.z, cb.w);
    }}
    var rr: V2 = q - cc;
    var dist: F = length(rr);
    if (dc.x * rr.y - dc.y * rr.x > 0.0) {{ return -dist; }}
    return dist;
}}
",
        newton = CURVE_NEWTON
    );
    s
}

/// Rewrites the WGSL-like template into one dialect (`"wgsl"`, `"glsl"`, `"hlsl"`)
fn render(src: &str, dialect: &str) -> String {
    let (f, i, v2, v3, v4) = match dialect {
        "wgsl" => ("f32", "i32", "vec2<f32>", "vec3<f32>", "vec4<f32>"),
        "hlsl" => ("float", "int", "float2", "float3", "float4"),
        _ => ("float", "int", "vec2", "vec3", "vec4"),
    };
    let ty = |t: &str| -> String {
        match t {
            "F" => f.to_string(),
            "I" => i.to_string(),
            "V2" => v2.to_string(),
            "V3" => v3.to_string(),
            "V4" => v4.to_string(),
            other => other.to_string(),
        }
    };
    let mut out = String::new();
    for line in src.lines() {
        let mut l = line.to_string();
        if dialect != "wgsl" {
            // fn name(a: T, b: T) -> R {   →   R name(T a, T b) {
            if let Some(rest) = l.strip_prefix("fn ") {
                let open = rest.find('(').unwrap_or(0);
                let close = rest.find(')').unwrap_or(0);
                let fname = &rest[..open];
                let params: Vec<String> = rest[open + 1..close]
                    .split(',')
                    .filter(|p| !p.trim().is_empty())
                    .map(|p| {
                        let mut it = p.split(':');
                        let pn = it.next().unwrap_or("").trim();
                        let pt = it.next().unwrap_or("").trim();
                        format!("{} {pn}", ty(pt))
                    })
                    .collect();
                let ret = rest[close..].trim_start_matches(')').trim().trim_start_matches("->").trim().trim_end_matches('{').trim();
                l = format!("{} {fname}({}) {{", ty(ret), params.join(", "));
            }
            // var x: T = e;   →   T x = e;
            while let Some(at) = l.find("var ") {
                let rest = &l[at + 4..];
                let colon = rest.find(':').unwrap_or(0);
                let eq = rest.find(" =").unwrap_or(colon);
                let vn = rest[..colon].to_string();
                let vt = rest[colon + 1..eq].trim().to_string();
                l = format!("{}{} {}{}", &l[..at], ty(&vt), vn, &rest[eq..]);
            }
        }
        out += &replace_types(&l, f, i, v2, v3, v4);
        out.push('\n');
    }
    out
}

/// The template rewrite for other law templates in the same notation
pub(crate) fn render_template(src: &str, dialect: &str) -> String {
    render(src, dialect)
}

/// Replaces the whole-word type tokens `F`, `I`, `V2`, `V3` (types and casts)
fn replace_types(l: &str, f: &str, i: &str, v2: &str, v3: &str, v4: &str) -> String {
    let mut out = String::new();
    let b = l.as_bytes();
    let mut k = 0;
    while k < b.len() {
        let c = b[k] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let start = k;
            while k < b.len() && ((b[k] as char).is_ascii_alphanumeric() || b[k] == b'_') {
                k += 1;
            }
            let word = &l[start..k];
            out += match word {
                "F" => f,
                "I" => i,
                "V2" => v2,
                "V3" => v3,
                "V4" => v4,
                w => w,
            };
        } else {
            out.push(c);
            k += 1;
        }
    }
    out
}

/// The loft's module-scope source for one node in `dialect`: the data array declaration is the
/// caller's (`ShaderLang::global_float_array`), this is the functions; the field is `{name}(p)`
pub fn loft_functions(name: &str, arr: &str, flat: &[f32], dialect: &str) -> String {
    render(&template(name, arr, flat, dialect != "wgsl"), dialect)
}

/// The loft's data array: in WGSL a module-scope `const` (a `var<private>` copy of a large block
/// exhausts the per-invocation stack in some implementations), elsewhere the language's own
/// module-scope float array
pub fn loft_array(arr: &str, flat: &[f32], dialect: &str, default: fn(&str, &[f32]) -> String) -> String {
    let mut finite = flat.to_vec();
    // the far bound's 16 in-plane directions (cos, sin), after the block: the shader loops over a
    // piece's planes instead of unrolling them (an unrolled piece exceeded a compute stack)
    if flat[16] > 0.0 {
        for (c, s) in DIR16 {
            finite.push(c);
            finite.push(s);
        }
    }
    let flat = &finite[..];
    if dialect != "wgsl" && flat[16] > 0.0 {
        // the C-like dialects: the block without its pieces, each piece on its own, the directions
        let at = pieces_at(flat);
        let pieces = flat[16] as usize;
        let mut out = default(arr, &flat[..at]);
        for q in 0..pieces {
            out += &default(&format!("{arr}_p{q}"), &flat[at + q * PIECE_FLOATS..at + (q + 1) * PIECE_FLOATS]);
        }
        let dir: Vec<f32> = DIR16.iter().flat_map(|(c, s)| [*c, *s]).collect();
        out += &default(&format!("{arr}_dir"), &dir);
        return out;
    }
    const_array(arr, flat, dialect, default)
}

/// A module-scope float array for any law's flat block, with no knowledge of its layout: in WGSL a
/// `const` (a `var<private>` copy of a large block exhausts the per-invocation stack in some
/// implementations), elsewhere the language's own module-scope float array
pub fn const_array(arr: &str, flat: &[f32], dialect: &str, default: fn(&str, &[f32]) -> String) -> String {
    if dialect == "wgsl" {
        let n = flat.len();
        let items: Vec<String> = flat.iter().map(|v| lit(*v)).collect();
        format!("const {arr}: array<f32, {n}> = array<f32, {n}>({});\n", items.join(", "))
    } else {
        default(arr, flat)
    }
}
