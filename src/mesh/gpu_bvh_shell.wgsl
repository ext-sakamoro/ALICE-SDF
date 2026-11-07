// First hit of each ray on a mesh thickened by a radius r: the operations of
// MeshBvh::shell_first_hit (src/mesh/bvh_shell.rs) in the same order, over a flattened MeshBvh
// walked without a stack (parent links). Follows gpu_bvh_closest.wgsl.
struct Params { first: u32, count: u32, r: f32, pad: u32, }
struct Node { min_x: f32, min_y: f32, min_z: f32, a: u32, max_x: f32, max_y: f32, max_z: f32, b: u32, }
struct Out { t_base: f32, t_local: f32, tri: u32, feature: u32, }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rays: array<f32>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> leaf_tris: array<u32>;
@group(0) @binding(4) var<storage, read> tris: array<f32>;
@group(0) @binding(5) var<storage, read> parents: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<Out>;
// the triangle frame: per triangle, its box centre (4 floats) and its vertices as v - c (9 floats)
@group(0) @binding(7) var<storage, read> centres: array<f32>;
@group(0) @binding(8) var<storage, read> local_tris: array<f32>;

// 0: the world frame (the crate's unframed reference), 1: the triangle's box (set when the kernel
// is built)
const FRAME: u32 = 0u;

const LEAF: u32 = 0x80000000u;
const NONE: u32 = 0xffffffffu;
const F_FACE: u32 = 1u;
const F_EDGE: u32 = 2u;
const F_VERTEX: u32 = 3u;
const F_INSIDE: u32 = 4u;
const MISS: f32 = -1.0;

// Rust's f32::min / f32::max: a NaN operand gives the other one
fn fmin(a: f32, b: f32) -> f32 { if a != a { return b; } if b != b { return a; } return min(a, b); }
fn fmax(a: f32, b: f32) -> f32 { if a != a { return b; } if b != b { return a; } return max(a, b); }

fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y);
}

// smallest root t >= 0 of |w0 + t g|^2 = rr about the closest approach; MISS when none
fn shifted_root(w0: vec3<f32>, g: vec3<f32>, rr: f32) -> f32 {
    let a = d(g, g);
    if a == 0.0 { return MISS; }
    let t0 = -d(w0, g) / a;
    let w = vadd_scaled(w0, g, t0);
    let h = rr - d(w, w);
    if h < 0.0 { return MISS; }
    let dt = sqrt(h / a);
    let lo = t0 - dt;
    let hi = t0 + dt;
    if lo >= 0.0 { return lo; }
    if hi >= 0.0 { return hi; }
    return MISS;
}

// (x, y, z) bits of p before those of q, lexicographically, as the CPU's tuple order
fn key_lt(p: vec3<f32>, q: vec3<f32>) -> bool {
    let a = bitcast<vec3<u32>>(p);
    let b = bitcast<vec3<u32>>(q);
    if a.x != b.x { return a.x < b.x; }
    if a.y != b.y { return a.y < b.y; }
    return a.z < b.z;
}

fn cylinder_t(o: vec3<f32>, dir: vec3<f32>, u0: vec3<f32>, v0: vec3<f32>, r: f32) -> f32 {
    // one order for an edge, whichever triangle names it
    var u = u0; var v = v0;
    if key_lt(v0, u0) { u = v0; v = u0; }
    let e = v - u;
    let l2 = d(e, e);
    if l2 == 0.0 { return MISS; }
    let m = o - u;
    let t = shifted_root(cross3(m, e), cross3(dir, e), r * r * l2);
    if t == MISS { return MISS; }
    let ax = d(vadd_scaled(m, dir, t), e);
    if !(ax >= 0.0 && ax <= l2) { return MISS; }
    return t;
}

// Error-free transforms, as src/mesh/bvh_shell.rs: Dekker's split, no fused multiply-add, so the
// same + - * (correctly rounded, kept unfused by the shader path) give the CPU's bits
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = s - a;
    return vec2<f32>(s, (a - (s - bb)) + (b - bb));
}
fn two_diff(a: f32, b: f32) -> vec2<f32> {
    let s = a - b;
    let bb = s - a;
    return vec2<f32>(s, (a - (s - bb)) + (-b - bb));
}
fn split(a: f32) -> vec2<f32> {
    let c = 4097.0 * a;
    let hi = c - (c - a);
    return vec2<f32>(hi, a - hi);
}
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let x = split(a);
    let y = split(b);
    return vec2<f32>(p, ((x.x * y.x - p) + x.x * y.y + x.y * y.x) + x.y * y.y);
}
// below these a difference component (or its remainder) and a normal component are exactly 0, as on
// the CPU: every intermediate of the error-free transforms stays a normal f32
const FLUSH_W: f32 = 0x1p-40f;
const FLUSH_N: f32 = 0x1p-38f;
fn flush(x: f32, below: f32) -> f32 { return select(x, 0.0, abs(x) < below); }
fn flush2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(flush(v.x, FLUSH_W), flush(v.y, FLUSH_W)); }

// (a - o) . n for an unnormalised n: exact differences, Dot2, plus the remainders against n
fn plane_distance(a: vec3<f32>, o: vec3<f32>, n0: vec3<f32>) -> f32 {
    let wx = flush2(two_diff(a.x, o.x)); let wy = flush2(two_diff(a.y, o.y)); let wz = flush2(two_diff(a.z, o.z));
    let n = vec3<f32>(flush(n0.x, FLUSH_N), flush(n0.y, FLUSH_N), flush(n0.z, FLUSH_N));
    let q0 = two_prod(wx.x, n.x);
    let r1 = two_prod(wy.x, n.y);
    let s1 = two_sum(q0.x, r1.x);
    let r2 = two_prod(wz.x, n.z);
    let s2 = two_sum(s1.x, r2.x);
    return s2.x + (q0.y + r1.y + s1.y + r2.y + s2.y + (wx.y * n.x + wy.y * n.y + wz.y * n.z));
}

struct Frame { anchor: vec3<f32>, nu: vec3<f32>, }

// the cross of the two shortest edges, at the vertex opposite the longest; a tie goes to the lower
// bits of the opposite vertex
fn face_frame(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Frame {
    let l0 = d(b - a, b - a); let l1 = d(c - b, c - b); let l2 = d(a - c, a - c);
    // edge k is opposite vertex (c, a, b)[k]
    var m = 0u; var ml = l0; var mk = c;
    if l1 > ml || (l1 == ml && key_lt(a, mk)) { m = 1u; ml = l1; mk = a; }
    if l2 > ml || (l2 == ml && key_lt(b, mk)) { m = 2u; ml = l2; mk = b; }
    if m == 0u { return Frame(c, cross3(a - c, b - c)); }
    if m == 1u { return Frame(a, cross3(b - a, c - a)); }
    return Frame(b, cross3(c - b, a - b));
}

// the edge function of u -> v at q, one direction per edge (lower bits first, sign turned back)
fn edge_function(u: vec3<f32>, v: vec3<f32>, q: vec3<f32>, nu: vec3<f32>) -> f32 {
    if !key_lt(v, u) { return d(cross3(v - u, q - u), nu); }
    return -d(cross3(u - v, q - v), nu);
}

// both planes offset by r from the shortest-edge plane; over the triangle by the edge functions
fn face_t(o: vec3<f32>, dir: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    let fr = face_frame(a, b, c);
    let nu = fr.nu;
    let nl = sqrt(d(nu, nu));
    if nl == 0.0 || !(abs(nl) <= 3.40282347e38) { return MISS; }
    // t = (P +- r |nu|) / D, P and D from + - * alone: one square root and one division in t
    let dd = d(dir, nu);
    if dd == 0.0 { return MISS; }
    let p = plane_distance(fr.anchor, o, nu);
    let rn = r * nl;
    let off = r / nl;
    var best = MISS;
    for (var k = 0u; k < 2u; k += 1u) {
        let sg = select(-1.0, 1.0, k == 0u);
        let t = (p + sg * rn) / dd;
        if t < 0.0 { continue; }
        let q = vadd_scaled(vadd_scaled(o, dir, t), nu, -(sg * off));
        let over = edge_function(a, b, q, nu) >= 0.0
            && edge_function(b, c, q, nu) >= 0.0
            && edge_function(c, a, q, nu) >= 0.0;
        if over {
            if best == MISS { best = t; } else { best = fmin(best, t); }
        }
    }
    return best;
}

struct Hit { t: f32, feature: u32, }

fn put(best: Hit, t: f32, f: u32) -> Hit {
    if t != MISS && (best.feature == 0u || t < best.t) { return Hit(t, f); }
    return best;
}

fn triangle_hit(o: vec3<f32>, dir: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, r: f32) -> Hit {
    let e = o - closest(o, a, b, c);
    if sqrt(d(e, e)) <= r { return Hit(0.0, F_INSIDE); }
    var best = Hit(0.0, 0u);
    best = put(best, face_t(o, dir, a, b, c, r), F_FACE);
    best = put(best, cylinder_t(o, dir, a, b, r), F_EDGE);
    best = put(best, cylinder_t(o, dir, b, c, r), F_EDGE);
    best = put(best, cylinder_t(o, dir, c, a, r), F_EDGE);
    best = put(best, shifted_root(o - a, dir, r * r), F_VERTEX);
    best = put(best, shifted_root(o - b, dir, r * r), F_VERTEX);
    best = put(best, shifted_root(o - c, dir, r * r), F_VERTEX);
    return best;
}

// entry parameter into the node's box inflated by r, clamped to 0; MISS when the ray misses it
// entry parameter into the box [lo - r, hi + r], clamped to 0; MISS when the ray misses it
fn slab_box(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, r: f32) -> f32 {
    let ax = (lo.x - r - o.x) * inv.x; let bx = (hi.x + r - o.x) * inv.x;
    let ay = (lo.y - r - o.y) * inv.y; let by = (hi.y + r - o.y) * inv.y;
    let az = (lo.z - r - o.z) * inv.z; let bz = (hi.z + r - o.z) * inv.z;
    let t0 = fmax(fmax(fmin(ax, bx), fmax(fmin(ay, by), fmin(az, bz))), 0.0);
    let t1 = fmin(fmax(ax, bx), fmin(fmax(ay, by), fmax(az, bz)));
    if t1 >= t0 { return t0; }
    return MISS;
}

fn slab_t(n: Node, o: vec3<f32>, inv: vec3<f32>, r: f32) -> f32 {
    return slab_box(vec3<f32>(n.min_x, n.min_y, n.min_z), vec3<f32>(n.max_x, n.max_y, n.max_z), o, inv, r);
}

// o + d t - c with every rounding error carried, as frame_origin in src/mesh/bvh_shell.rs
fn frame_comp(o: f32, c: f32, d: f32, t: f32) -> f32 {
    let s = flush2(two_diff(o, c));
    let p = flush2(two_prod(flush(d, FLUSH_W), t));
    let x = two_sum(p.x, s.x);
    return x.x + (x.y + p.y + s.y);
}
fn frame_origin(o: vec3<f32>, dir: vec3<f32>, c: vec3<f32>, t: f32) -> vec3<f32> {
    return vec3<f32>(frame_comp(o.x, c.x, dir.x, t), frame_comp(o.y, c.y, dir.y, t), frame_comp(o.z, c.z, dir.z, t));
}

fn vec_at(buf_index: u32, which: u32) -> vec3<f32> {
    // which 0: centres (4 floats an entry), 1..3: local_tris vertex which-1 (9 floats an entry)
    if which == 0u { let q = buf_index * 4u; return vec3<f32>(centres[q], centres[q + 1u], centres[q + 2u]); }
    let q = buf_index * 9u + (which - 1u) * 3u;
    return vec3<f32>(local_tris[q], local_tris[q + 1u], local_tris[q + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count { return; }
    // 10 floats a ray: origin, direction, tmax, and 1 / direction from the CPU (correctly rounded,
    // so the boxes' entries, the walk's order and every frame's base are the CPU's bits)
    let k = (params.first + i) * 10u;
    let o = vec3<f32>(rays[k], rays[k + 1u], rays[k + 2u]);
    let dir = vec3<f32>(rays[k + 3u], rays[k + 4u], rays[k + 5u]);
    let tmax = rays[k + 6u];
    let r = params.r;
    let inv = vec3<f32>(rays[k + 7u], rays[k + 8u], rays[k + 9u]);
    var best_t = tmax;
    var best_base = 0.0;
    var best_local = 0.0;
    var best_tri = NONE;
    var best_f = 0u;
    if arrayLength(&nodes) > 0u && slab_t(nodes[0], o, inv, r) != MISS {
        var at = 0u;
        var prev = NONE;
        loop {
            let n = nodes[at];
            if (n.a & LEAF) != 0u {
                let first = n.a & ~LEAF;
                for (var j = first; j < first + n.b; j += 1u) {
                    let t = leaf_tris[j];
                    let q = t * 9u;
                    let a = vec3<f32>(tris[q], tris[q + 1u], tris[q + 2u]);
                    let b = vec3<f32>(tris[q + 3u], tris[q + 4u], tris[q + 5u]);
                    let c = vec3<f32>(tris[q + 6u], tris[q + 7u], tris[q + 8u]);
                    var base = 0.0;
                    var h = Hit(0.0, 0u);
                    if FRAME == 0u {
                        h = triangle_hit(o, dir, a, b, c, r);
                    } else {
                        base = slab_box(min(min(a, b), c), max(max(a, b), c), o, inv, r);
                        if base == MISS { continue; }
                        let oa = frame_origin(o, dir, vec_at(t, 0u), base);
                        h = triangle_hit(oa, dir, vec_at(t, 1u), vec_at(t, 2u), vec_at(t, 3u), r);
                    }
                    let key = base + h.t;
                    if h.feature != 0u && key <= tmax {
                        if best_tri == NONE || key < best_t || (key == best_t && t < best_tri) {
                            best_t = key; best_base = base; best_local = h.t; best_tri = t; best_f = h.feature;
                        }
                    }
                }
                if at == 0u { break; }
                prev = at;
                at = parents[at];
                continue;
            }
            let tl = slab_t(nodes[n.a], o, inv, r);
            let tr = slab_t(nodes[n.b], o, inv, r);
            var first = n.a; var first_t = tl; var second = n.b; var second_t = tr;
            // nearer box first; a box the ray misses goes last (as the CPU walk orders them)
            if (tl != MISS && tr != MISS && tr < tl) || tl == MISS {
                first = n.b; first_t = tr; second = n.a; second_t = tl;
            }
            // a box entered at the best t can still hold a lower index at that t: `<=`
            if prev == NONE || prev == parents[at] {
                if first_t != MISS && first_t <= best_t { prev = at; at = first; continue; }
                if second_t != MISS && second_t <= best_t { prev = at; at = second; continue; }
            } else if prev == first {
                if second_t != MISS && second_t <= best_t { prev = at; at = second; continue; }
            }
            if at == 0u { break; }
            prev = at;
            at = parents[at];
        }
    }
    let slot = params.first + i;
    if best_tri == NONE { out[slot] = Out(0.0, 0.0, NONE, 0u); } else { out[slot] = Out(best_base, best_local, best_tri, best_f); }
}
