//! The march of [`SheetField`] as a WebGL2 fragment program (GLSL ES 3.00): one fragment = one ray, in image order.
//!
//! The same arithmetic as the WGSL compute march in `gpu.rs` (its `sample_sheet`, `crossing_inside`, `eval_union`,
//! `live_sheets` and `march_main`, line for line), for pages that have WebGL2 and no compute: the storage buffers
//! become `R32UI` textures [`TEX_WIDTH`] texels wide (a table entry `i` is the texel `(i % TEX_WIDTH, i / TEX_WIDTH)`),
//! floats are read with `uintBitsToFloat`, and `countOneBits` (not in GLSL ES 3.00) is written out. The loops whose
//! bounds are constants in the WGSL take them from uniforms here ([`LOOP_BOUNDS`]), so a driver cannot unroll them. Nothing here
//! needs `wgpu`: [`pack`] lays out the tables, [`march_fragment_shader`] gives the program text, and the page
//! uploads and draws them.
//!
//! The output is 4 `RGBA32F` targets per ray, the same 4 `vec4` as the compute march writes: `(t, kind, sheet,
//! band_miss)`, `(normal, work)`, `(uv, has_uv, 0)`, `(author normal, has)`. `kind` is 1 hit, 0 miss, 3 the live list
//! overflowed, 4 over the work budget. The page clears the targets to the bit pattern [`NOT_WRITTEN`] before the draw
//! and reads them back: a texel still holding it was not written (a cut-short draw), which is an error, never a miss.
//!
//! Units: the field's (millimetres for every constant here, as in `mod.rs`).

use super::{SheetField, SheetFieldError, MIN_STEP};

/// The values the page sets for the uniforms `n_corners`, `n_bisect` and `n_taps` (8 trilinear corners, 30 bisection
/// steps, 6 normal taps, as the compute march).
pub const LOOP_BOUNDS: (u32, i32, u32) = (8, 30, 6);
/// Texels per row of every table texture.
pub const TEX_WIDTH: u32 = 4096;
/// The clear value of the output targets (a NaN no march writes), the same pattern as the compute march's.
pub const NOT_WRITTEN: u32 = 0x7FA5_A5A5;
/// `u32` words per sheet record in the sheets table.
pub const SHEET_WORDS: u32 = 12;
/// `u32` words per level record in the levels table.
pub const LEVEL_WORDS: u32 = 52;
/// The largest per-ray live list this program compiles (fragment programs keep it in registers; the compute march
/// allows 1024, a fragment program with that list is what stalls a mobile GPU).
pub const MAX_LIVE_FRAGMENT: u32 = 64;

/// The tables of a field, laid out for `R32UI` textures, and what the program needs to address them.
#[derive(Debug, Clone)]
pub struct WebglPack {
    /// Sheet records ([`SHEET_WORDS`] each): lo xyz, hc, hi xyz, level_start, level_count, 3 words of padding.
    pub sheets: Vec<u32>,
    /// Level records ([`LEVEL_WORDS`] each): lo xyz, h, nb xyz, index_off, nch, band_off, 2 pad, then 8 words each of
    /// rows_off, q_off, sc_off, nodes, st_off (the compute march's `LevelInfo`).
    pub levels: Vec<u32>,
    /// The merged table: brick indices, channel rows, BVH nodes, leaf entries, parent links, band masks and starts.
    pub ix: Vec<u32>,
    /// The channel bytes, four to a word (byte `i` is bits `8 (i % 4)` of word `i / 4`).
    pub qw: Vec<u32>,
    /// Scale-offset pairs, as `f32` bits.
    pub sc: Vec<u32>,
    /// Bases in `ix`: index, rows, BVH nodes, leaf entries, parents, BVH node words, band.
    pub index_base: u32,
    /// See [`WebglPack::index_base`].
    pub rows_base: u32,
    /// See [`WebglPack::index_base`].
    pub bn_base: u32,
    /// See [`WebglPack::index_base`].
    pub bl_base: u32,
    /// See [`WebglPack::index_base`].
    pub bp_base: u32,
    /// See [`WebglPack::index_base`].
    pub bn_len: u32,
    /// See [`WebglPack::index_base`].
    pub band_base: u32,
    /// Sheets.
    pub n_sheets: u32,
    /// 1 = the cell rule, 0 = none.
    pub rule: u32,
    /// The most sheet boxes holding one point ([`live_bound`]).
    pub live_bound: u32,
}

impl WebglPack {
    /// Bytes the five tables upload (rows padded to [`TEX_WIDTH`]).
    #[must_use]
    pub fn upload_bytes(&self) -> u64 {
        [&self.sheets, &self.levels, &self.ix, &self.qw, &self.sc]
            .iter()
            .map(|t| u64::from(rows_for(t.len())) * u64::from(TEX_WIDTH) * 4)
            .sum()
    }
}

/// Texture rows for a table of `len` words (at least 1).
#[must_use]
pub fn rows_for(len: usize) -> u32 {
    (len.div_ceil(TEX_WIDTH as usize)).max(1) as u32
}

/// The most sheet boxes that hold one point: over the sheets, the count of boxes meeting that sheet's box
/// (inclusive, itself included), the same bound as the compute march's `live_bound`.
#[must_use]
pub fn live_bound(field: &SheetField) -> u32 {
    let s = &field.sheets;
    let mut best = 0u32;
    for a in s {
        let n = s
            .iter()
            .filter(|b| a.lo.cmple(b.hi).all() && b.lo.cmple(a.hi).all())
            .count() as u32;
        best = best.max(n);
    }
    best
}

/// Lay out `field` for the fragment march. Refuses a field whose live bound is over [`MAX_LIVE_FRAGMENT`] (the
/// list is a fixed array of a fragment program) and levels with other than 2, 4 or 7 channels.
pub fn pack(field: &SheetField) -> Result<WebglPack, SheetFieldError> {
    let bound = live_bound(field);
    if bound > MAX_LIVE_FRAGMENT {
        return Err(SheetFieldError::Shape(format!(
            "{bound} sheet boxes can hold one point, over the fragment march's list of {MAX_LIVE_FRAGMENT}"
        )));
    }
    let (mut sheets, mut levels) = (Vec::new(), Vec::new());
    let (mut index, mut rows, mut qbytes, mut scoff, mut bands) = (
        Vec::<i32>::new(),
        Vec::<i32>::new(),
        Vec::<u8>::new(),
        Vec::<f32>::new(),
        Vec::<u32>::new(),
    );
    let mut level_count = 0u32;
    for sh in &field.sheets {
        sheets.extend_from_slice(&[
            sh.lo.x.to_bits(),
            sh.lo.y.to_bits(),
            sh.lo.z.to_bits(),
            sh.hc.to_bits(),
            sh.hi.x.to_bits(),
            sh.hi.y.to_bits(),
            sh.hi.z.to_bits(),
            level_count,
            sh.levels.len() as u32,
            0,
            0,
            0,
        ]);
        for lv in &sh.levels {
            if !matches!(lv.channels.len(), 2 | 4 | 7) {
                return Err(SheetFieldError::Shape(format!(
                    "a level with {} channels (2, 4 or 7)",
                    lv.channels.len()
                )));
            }
            let mut rec = vec![0u32; LEVEL_WORDS as usize];
            rec[0] = lv.lo.x.to_bits();
            rec[1] = lv.lo.y.to_bits();
            rec[2] = lv.lo.z.to_bits();
            rec[3] = lv.h.to_bits();
            rec[4..7].copy_from_slice(&lv.nb);
            rec[7] = index.len() as u32;
            rec[8] = lv.channels.len() as u32;
            rec[9] = bands.len() as u32;
            bands.extend_from_slice(&lv.band);
            index.extend_from_slice(&lv.index);
            for (c, ch) in lv.channels.iter().enumerate() {
                rec[12 + c] = rows.len() as u32;
                rows.extend_from_slice(&ch.rows);
                rec[20 + c] = qbytes.len() as u32;
                rec[36 + c] = ch.nodes;
                rec[44 + c] = bands.len() as u32;
                bands.extend_from_slice(&ch.starts);
                qbytes.extend_from_slice(&ch.q);
                rec[28 + c] = (scoff.len() / 2) as u32;
                for (s, o) in ch.scale.iter().zip(&ch.offset) {
                    scoff.push(*s);
                    scoff.push(*o);
                }
            }
            levels.extend_from_slice(&rec);
            level_count += 1;
        }
    }
    while qbytes.len() % 4 != 0 {
        qbytes.push(0);
    }
    let bvh = field.index();
    let mut ix: Vec<u32> = Vec::new();
    let index_base = ix.len() as u32;
    ix.extend(index.iter().map(|&x| x as u32));
    let rows_base = ix.len() as u32;
    ix.extend(rows.iter().map(|&x| x as u32));
    let bn_base = ix.len() as u32;
    ix.extend_from_slice(&bvh.nodes);
    let bl_base = ix.len() as u32;
    ix.extend_from_slice(&bvh.leaf_entries);
    let bp_base = ix.len() as u32;
    ix.extend_from_slice(&bvh.parents);
    let band_base = ix.len() as u32;
    ix.extend_from_slice(&bands);
    let qw = qbytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(WebglPack {
        sheets,
        levels,
        ix,
        qw,
        sc: scoff.iter().map(|x| x.to_bits()).collect(),
        index_base,
        rows_base,
        bn_base,
        bl_base,
        bp_base,
        bn_len: bvh.nodes.len() as u32,
        band_base,
        n_sheets: field.sheets.len() as u32,
        rule: u32::from(matches!(field.rule, super::ValidRule::CellV1)),
        live_bound: bound,
    })
}

/// The vertex program: one triangle over the viewport.
pub const VERTEX_SHADER: &str = "#version 300 es
in vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }
";

/// The fragment march for a live list of `max_live` (at least the field's [`live_bound`], at most
/// [`MAX_LIVE_FRAGMENT`]).
#[must_use]
pub fn march_fragment_shader(max_live: u32) -> String {
    FRAGMENT
        .replace("__MAX_LIVE__", &max_live.max(1).to_string())
        .replace("__MIN_STEP__", &format!("{:?}", MIN_STEP))
        .replace("__TEX_WIDTH__", &TEX_WIDTH.to_string())
}

const FRAGMENT: &str = r"#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
precision highp sampler2D;
uniform usampler2D SHEETS;
uniform usampler2D LEVELS;
uniform usampler2D IX;
uniform usampler2D QW;
uniform usampler2D SC;
uniform sampler2D INPUT;
uniform uint n_sheets; uniform uint rule; uniform uint steps; uniform vec3 dir; uniform float tmax;
uniform uint index_base; uniform uint rows_base; uniform uint bn_base; uniform uint bl_base; uniform uint bp_base;
uniform uint bn_len; uniform uint band_base; uniform uint work_budget; uniform uint width;
// loop bounds the page sets to 8, 30 and 6: uniforms, so the compiler keeps each loop (and the sample_sheet it calls) once
// instead of unrolling it (unrolled, the program took 55.9 s to link on a Tegra Orin)
uniform uint n_corners; uniform int n_bisect; uniform uint n_taps;
layout(location = 0) out vec4 out0;
layout(location = 1) out vec4 out1;
layout(location = 2) out vec4 out2;
layout(location = 3) out vec4 out3;

const uint TW = __TEX_WIDTH__u;
const uint LEAF = 0x80000000u;
const uint NONE = 0xffffffffu;
const uint BRICK = 8u;
const uint NODES = 9u;
const uint BRICK_NODES = 729u;
const uint BAND_WORDS = 23u;
const float HALF_DIAG = 0.8660254;
const float JUMP_S = 0.02;
const uint MAX_LIVE = __MAX_LIVE__u;
const float MIN_STEP = __MIN_STEP__;

uint at_(usampler2D t, uint i) { return texelFetch(t, ivec2(int(i % TW), int(i / TW)), 0).r; }
uint ixw(uint i) { return at_(IX, i); }
float scf(uint i) { return uintBitsToFloat(at_(SC, i)); }
uint lvw(uint level, uint w) { return at_(LEVELS, level * 52u + w); }
float lvf(uint level, uint w) { return uintBitsToFloat(lvw(level, w)); }
uint shw(uint k, uint w) { return at_(SHEETS, k * 12u + w); }
vec3 sh_lo(uint k) { return vec3(uintBitsToFloat(shw(k, 0u)), uintBitsToFloat(shw(k, 1u)), uintBitsToFloat(shw(k, 2u))); }
vec3 sh_hi(uint k) { return vec3(uintBitsToFloat(shw(k, 4u)), uintBitsToFloat(shw(k, 5u)), uintBitsToFloat(shw(k, 6u))); }
float sh_hc(uint k) { return uintBitsToFloat(shw(k, 3u)); }
// countOneBits is GLSL ES 3.10; the classic SWAR count (exact on 32-bit unsigned arithmetic)
uint popc(uint v) { v = v - ((v >> 1u) & 0x55555555u); v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u); v = (v + (v >> 4u)) & 0x0f0f0f0fu; return (v * 0x01010101u) >> 24u; }

struct Sample { float f; float g; bool inside; bool masked; vec2 uv; bool has_uv; float s; float db; float db_step; vec3 an; bool has_an; bool band_miss; };

float qbyte(uint byte_index) {
    uint w = at_(QW, byte_index >> 2u);
    return float((w >> ((byte_index & 3u) * 8u)) & 255u);
}
float box_distance(vec3 p, vec3 lo, vec3 hi) { return length(max(max(lo - p, p - hi), vec3(0.0))); }
float outside_bound(vec3 p, uint k) { return max(box_distance(p, sh_lo(k), sh_hi(k)), 3.0 * sh_hc(k)); }

// sheet k at p, interpolating the first nmax channels only (2 while marching; attributes once, at the hit)
Sample sample_sheet(uint k, vec3 p, uint nmax) {
    Sample o;
    uint ls = shw(k, 7u); uint lc = shw(k, 8u);
    for (uint li = 0u; li < lc; li++) {
        uint L = ls + li;
        vec3 lo = vec3(lvf(L, 0u), lvf(L, 1u), lvf(L, 2u)); float h = lvf(L, 3u);
        uvec3 nb = uvec3(lvw(L, 4u), lvw(L, 5u), lvw(L, 6u));
        vec3 u = (p - lo) / h;
        float b = float(BRICK);
        vec3 bi = floor(u / b);
        if (any(lessThan(bi, vec3(0.0))) || any(greaterThanEqual(bi, vec3(nb)))) { continue; }
        uvec3 bu = uvec3(bi);
        int k_ = int(ixw(index_base + lvw(L, 7u) + (bu.x * nb.y + bu.y) * nb.z + bu.z));
        if (k_ < 0) { continue; }
        vec3 w = u - bi * b;
        uvec3 c = min(uvec3(floor(w)), uvec3(BRICK - 1u));
        vec3 t = w - vec3(c);
        float vals[8]; for (int z = 0; z < 8; z++) { vals[z] = 0.0; }
        float umin = 1e30; bool band_miss = false; float q0 = 0.0; float q1 = 0.0;
        uint nch = min(lvw(L, 8u), nmax);
        for (uint ci = 0u; ci < nch; ci++) {
            int row = int(ixw(rows_base + lvw(L, 12u + ci) + uint(k_)));
            float acc = 0.0;
            uint nn = lvw(L, 36u + ci);
            uint qo = lvw(L, 20u + ci); uint so = lvw(L, 28u + ci);
            if (row >= 0) {
                // one corner loop for the three layouts (band, coarse grid, full grid), corners in the order dx, dy, dz
                float sc = scf(2u * (so + uint(row))); float off = scf(2u * (so + uint(row)) + 1u);
                bool band = nn == 0u; bool coarse = !band && nn != NODES;
                uint mw = band_base + lvw(L, 9u) + uint(k_) * BAND_WORDS;
                uint start = band ? ixw(band_base + lvw(L, 44u + ci) + uint(row)) : 0u;
                vec3 tt = t; uvec3 cb = c; uint nd = NODES; uint base = qo + uint(row) * BRICK_NODES;
                if (coarse) { vec3 wc = w / float(BRICK / (nn - 1u)); cb = min(uvec3(max(floor(wc), vec3(0.0))), uvec3(nn - 2u)); tt = wc - vec3(cb); nd = nn; base = qo + uint(row) * nn * nn * nn; }
                for (uint corner = 0u; corner < n_corners; corner++) {
                    uint dx = corner >> 2u; uint dy = (corner >> 1u) & 1u; uint dz = corner & 1u;
                    float wx = dx == 1u ? tt.x : 1.0 - tt.x; float wy = dy == 1u ? tt.y : 1.0 - tt.y; float wz = dz == 1u ? tt.z : 1.0 - tt.z;
                    uint node = ((cb.x + dx) * nd + (cb.y + dy)) * nd + (cb.z + dz);
                    float v;
                    if (band) {
                        uint word = ixw(mw + node / 32u); uint bit = node % 32u;
                        if (((word >> bit) & 1u) == 0u) { band_miss = true; continue; }
                        uint rank = popc(word & ((1u << bit) - 1u));
                        for (uint j = 0u; j < node / 32u; j++) { rank += popc(ixw(mw + j)); }
                        v = qbyte(qo + start + rank) * sc + off;
                    } else {
                        v = qbyte(base + node) * sc + off;
                    }
                    acc = acc + (wx * wy * wz) * v;
                    if (!band && !coarse && ci == 0u) { umin = min(umin, abs(v)); }
                }
                if (!band && !coarse && ci == 0u) { q0 = sc; }
                if (!band && !coarse && ci == 1u) { q1 = sc; }
            }
            vals[ci] = acc;
        }
        float s = vals[0];
        float db = max(vals[1], 0.0);
        float f = sqrt(s * s + db * db);
        float g = s < 0.0 ? -f : f;
        o.masked = false;
        if (rule == 1u && umin > HALF_DIAG * h + q0) { f = max(umin - q0 - HALF_DIAG * h, 0.05); g = f; o.masked = true; }
        o.f = f; o.g = g; o.inside = true;
        o.s = s; o.db = vals[1]; o.db_step = q1;
        o.has_uv = nch >= 4u; o.band_miss = band_miss; o.has_an = nch >= 7u;
        o.an = vec3(vals[4], vals[5], vals[6]); o.uv = vec2(vals[2], vals[3]);
        return o;
    }
    float v = outside_bound(p, k);
    o.f = v; o.g = v; o.inside = false; o.masked = false; o.has_uv = false; o.has_an = false; o.an = vec3(0.0);
    o.band_miss = false; o.s = 0.0; o.db = 0.0; o.db_step = 0.0; o.uv = vec2(0.0);
    return o;
}

bool in_box(uint k, vec3 p) { return all(greaterThanEqual(p, sh_lo(k))) && all(lessThanEqual(p, sh_hi(k))); }
vec3 nlo(uint i) { return vec3(uintBitsToFloat(ixw(bn_base + 8u * i)), uintBitsToFloat(ixw(bn_base + 8u * i + 1u)), uintBitsToFloat(ixw(bn_base + 8u * i + 2u))); }
vec3 nhi(uint i) { return vec3(uintBitsToFloat(ixw(bn_base + 8u * i + 4u)), uintBitsToFloat(ixw(bn_base + 8u * i + 5u)), uintBitsToFloat(ixw(bn_base + 8u * i + 6u))); }

// min over the sheets whose box does not hold p (their outside bound); the march takes the min of this and the live
// sheets' f, which is the union's f of the compute march
float eval_outside_f(vec3 p) {
    float rf = 1e30; uint rk = NONE;
    if (bn_len < 8u || n_sheets == 0u) { return rf; }
    uint at = 0u; uint prev = NONE;
    for (;;) {
        uint a = ixw(bn_base + 8u * at + 3u); uint b = ixw(bn_base + 8u * at + 7u);
        bool up = true;
        if ((a & LEAF) != 0u) {
            uint first = a & ~LEAF;
            for (uint j = first; j < first + b; j++) {
                uint k = ixw(bl_base + j);
                float v;
                // a sheet whose box holds p is a live sheet: its f comes from the march's own sample at p (the min of
                // the same values in another order is the same number), so here it is skipped
                if (box_distance(p, sh_lo(k), sh_hi(k)) == 0.0) { continue; } else { v = outside_bound(p, k); }
                if (v < rf || (v == rf && k < rk)) { rf = v; rk = k; }
            }
        } else {
            float dl = box_distance(p, nlo(a), nhi(a)); float dr = box_distance(p, nlo(b), nhi(b));
            uint first = a; float df = dl; uint second = b; float ds = dr;
            if (dr < dl) { first = b; df = dr; second = a; ds = dl; }
            if (prev == NONE || prev == ixw(bp_base + at)) {
                if (df <= rf) { prev = at; at = first; up = false; }
                else if (ds <= rf) { prev = at; at = second; up = false; }
            } else if (prev == first && ds <= rf) { prev = at; at = second; up = false; }
        }
        if (up) { if (at == 0u) { break; } prev = at; at = ixw(bp_base + at); }
    }
    return rf;
}

uint LV[MAX_LIVE];
// the sheets whose box holds p, ascending, into LV (count returned; MAX_LIVE + 1 means overflow)
uint live_sheets(vec3 p) {
    uint n = 0u;
    if (bn_len < 8u || n_sheets == 0u) { return 0u; }
    if (!(all(greaterThanEqual(p, nlo(0u))) && all(lessThanEqual(p, nhi(0u))))) { return 0u; }
    uint at = 0u; uint prev = NONE;
    for (;;) {
        uint a = ixw(bn_base + 8u * at + 3u); uint b = ixw(bn_base + 8u * at + 7u);
        bool up = true;
        if ((a & LEAF) != 0u) {
            uint first = a & ~LEAF;
            for (uint j = first; j < first + b; j++) {
                uint k = ixw(bl_base + j);
                if (in_box(k, p)) { if (n == MAX_LIVE) { return MAX_LIVE + 1u; } LV[n] = k; n++; }
            }
        } else {
            bool in_l = all(greaterThanEqual(p, nlo(a))) && all(lessThanEqual(p, nhi(a)));
            bool in_r = all(greaterThanEqual(p, nlo(b))) && all(lessThanEqual(p, nhi(b)));
            if (prev == NONE || prev == ixw(bp_base + at)) {
                if (in_l) { prev = at; at = a; up = false; }
                else if (in_r) { prev = at; at = b; up = false; }
            } else if (prev == a && in_r) { prev = at; at = b; up = false; }
        }
        if (up) { if (at == 0u) { break; } prev = at; at = ixw(bp_base + at); }
    }
    for (uint i = 1u; i < n; i++) {
        uint x = LV[i]; uint j = i;
        for (;;) { if (j == 0u || LV[j - 1u] <= x) { break; } LV[j] = LV[j - 1u]; j = j - 1u; }
        LV[j] = x;
    }
    return n;
}

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    // image order: ray i = row * width + column of the pixel; the page lays the origins out the same way
    vec3 o = texelFetch(INPUT, q, 0).xyz;
    vec3 d = dir;
    uint prev_k[MAX_LIVE]; float prev_g[MAX_LIVE]; uint n_prev = 0u; bool overflow = false;
    // the compute march fills prev from o before the loop; step 0 (t = 0) samples o again and replaces prev before any
    // comparison (`t <= 0` skips them), and checks the same overflow, so that pass is left out here (a call site less)
    float t = 0.0; float tprev = 0.0; int kind = 2; uint own = 0u; uint work = 0u;
    for (uint step = 0u; step < steps && !overflow; step++) {
        vec3 p = o + d * t;
        float dd = eval_outside_f(p);
        float thit = 1e30; uint khit = 0u;
        uint cur_k[MAX_LIVE]; float cur_g[MAX_LIVE]; uint n_cur = 0u;
        uint nl = live_sheets(p);
        if (nl > MAX_LIVE) { overflow = true; break; }
        uint w = nl * (n_prev + 1u);
        if (work > work_budget - min(w, work_budget)) { work = work_budget; kind = 4; break; }
        work = work + w;
        for (uint li = 0u; li < nl; li++) {
            uint k = LV[li];
            // one loop, one call site: iteration -1 samples the sheet at p (its g for this step); if that is a sign
            // change against the previous step, iterations 0 .. n_bisect - 1 halve [tprev, t] and iteration n_bisect
            // tests the crossing at the final midpoint (crossing_inside of the compute march)
            float gc = 0.0; bool found = false; float gp = 0.0; float a = tprev; float b = t; float ga = 0.0;
            for (int it = -1; it <= n_bisect; it++) {
                float mid = it < 0 ? t : 0.5 * (a + b);
                Sample sm = sample_sheet(k, o + d * mid, 2u);
                if (it < 0) {
                    gc = sm.g; dd = min(dd, sm.f);
                    cur_k[n_cur] = k; cur_g[n_cur] = gc; n_cur++;
                    if (t <= 0.0) { break; }
                    for (uint j = 0u; j < n_prev; j++) { if (prev_k[j] == k) { found = true; gp = prev_g[j]; } }
                    if (!found) { break; }
                    if ((gc < 0.0) == (gp < 0.0)) { break; }
                    ga = gp;
                    continue;
                }
                if (it == n_bisect) {
                    if (mid < thit && sm.inside && !sm.masked && abs(sm.s) <= JUMP_S && sm.db <= 0.5 * sm.db_step) { thit = mid; khit = k; }
                    break;
                }
                if ((sm.g < 0.0) == (ga < 0.0)) { a = mid; ga = sm.g; } else { b = mid; }
            }
        }
        if (overflow) { break; }
        if (thit < 1e29) { t = thit; own = khit; kind = 1; break; }
        tprev = t;
        for (uint j = 0u; j < n_cur; j++) { prev_k[j] = cur_k[j]; prev_g[j] = cur_g[j]; }
        n_prev = n_cur;
        t = t + max(dd, MIN_STEP);
        if (t >= tmax) { kind = 0; break; }
    }
    if (overflow) { out0 = vec4(0.0, 3.0, 0.0, 0.0); out1 = vec4(0.0); out2 = vec4(0.0); out3 = vec4(0.0); return; }
    vec3 nrm = vec3(0.0, 0.0, 1.0); vec3 uv = vec3(0.0); vec4 an = vec4(0.0); float miss = 0.0;
    if (kind == 1) {
        vec3 p = o + d * t; float e = 0.05;
        // the 6 normal taps (+x, -x, +y, -y, +z, -z, channels s and d_b) and then the attributes at p (all channels),
        // through one call site
        float tap[6]; for (int z = 0; z < 6; z++) { tap[z] = 0.0; }
        Sample s;
        for (uint a = 0u; a <= n_taps; a++) {
            vec3 off = vec3(0.0); if (a < n_taps) { off[a / 2u] = (a % 2u == 0u) ? e : -e; }
            Sample sa = sample_sheet(own, p + off, a < n_taps ? 2u : 8u);
            if (a < n_taps) { tap[a] = sa.s; } else { s = sa; }
        }
        vec3 n = vec3(tap[0] - tap[1], tap[2] - tap[3], tap[4] - tap[5]);
        if (length(n) > 1e-12) { nrm = normalize(n); }
        if (s.band_miss) { miss = 1.0; }
        if (s.has_uv && s.inside) { uv = vec3(s.uv, 1.0); }
        if (s.has_an && s.inside && length(s.an) > 1e-12) { an = vec4(normalize(s.an), 1.0); }
    }
    out0 = vec4(t, float(kind), float(own), miss);
    out1 = vec4(nrm, float(work));
    out2 = vec4(uv, 0.0);
    out3 = an;
}
";

#[cfg(test)]
mod tests {
    use super::{march_fragment_shader, pack, popc_ref, MAX_LIVE_FRAGMENT};
    use crate::sheet_field::build::{level_from_fn, sheet};
    use crate::sheet_field::{SheetField, ValidRule};
    use glam::Vec3;

    fn plane_field() -> SheetField {
        // a disc of radius 20 in z = 0 (s = z, d_b = radial distance past the rim), h 1
        let lv = level_from_fn(Vec3::splat(-24.0), 1.0, [6, 6, 6], |c, e| c.z.abs() <= e, |p| {
            vec![p.z, (Vec3::new(p.x, p.y, 0.0).length() - 20.0).max(0.0) - 0.0, p.x / 40.0 + 0.5, p.y / 40.0 + 0.5]
        })
        .unwrap();
        SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
    }

    #[test]
    fn pack_lays_out_every_table() {
        let f = plane_field();
        let p = pack(&f).unwrap();
        assert_eq!(p.sheets.len(), 12);
        assert_eq!(p.levels.len(), 52);
        assert_eq!(p.live_bound, 1);
        assert!(!p.qw.is_empty() && !p.sc.is_empty() && p.bn_len >= 8);
        assert!(p.upload_bytes() >= (p.ix.len() * 4) as u64);
    }

    #[test]
    fn program_text_has_no_placeholder_left() {
        let s = march_fragment_shader(MAX_LIVE_FRAGMENT);
        assert!(!s.contains("__"), "a placeholder is left in the program");
        assert!(s.contains("const float MIN_STEP = 0.01;"));
    }

    #[test]
    fn popcount_as_written_in_the_program_is_exact() {
        // the program's SWAR count, transcribed, against count_ones on edge and spread values
        for v in [0u32, 1, 0x8000_0000, u32::MAX, 0x5555_5555, 0xAAAA_AAAA, 0x0F0F_F0F0, 0x1234_5678] {
            assert_eq!(popc_ref(v), v.count_ones(), "{v:#x}");
        }
    }
}

/// The program's popcount, transcribed for its test.
#[cfg(test)]
fn popc_ref(mut v: u32) -> u32 {
    v = v.wrapping_sub((v >> 1) & 0x5555_5555);
    v = (v & 0x3333_3333) + ((v >> 2) & 0x3333_3333);
    v = (v.wrapping_add(v >> 4)) & 0x0f0f_0f0f;
    v.wrapping_mul(0x0101_0101) >> 24
}
