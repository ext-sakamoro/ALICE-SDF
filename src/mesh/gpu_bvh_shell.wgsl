// First hit of each ray on a mesh thickened by a radius r: the operations of
// MeshBvh::shell_first_hit (src/mesh/bvh_shell.rs) in the same order, over a flattened MeshBvh
// walked without a stack (parent links). Follows gpu_bvh_closest.wgsl.
struct Params { first: u32, count: u32, r: f32, pad: u32, }
struct Node { min_x: f32, min_y: f32, min_z: f32, a: u32, max_x: f32, max_y: f32, max_z: f32, b: u32, }
struct Out { t: f32, tri: u32, feature: u32, pad: u32, }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rays: array<f32>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> leaf_tris: array<u32>;
@group(0) @binding(4) var<storage, read> tris: array<f32>;
@group(0) @binding(5) var<storage, read> parents: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<Out>;

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

// over the triangle: the three edge functions against the unnormalised normal (no division)
fn face_t(o: vec3<f32>, dir: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    let nn = cross3(b - a, c - a);
    let nl = sqrt(d(nn, nn));
    if nl == 0.0 || !(abs(nl) <= 3.40282347e38) { return MISS; }
    let n = vec3<f32>(nn.x / nl, nn.y / nl, nn.z / nl);
    let dn = d(dir, n);
    if dn == 0.0 { return MISS; }
    var best = MISS;
    for (var k = 0u; k < 2u; k += 1u) {
        let sg = select(-1.0, 1.0, k == 0u);
        let off = sg * r;
        let t = d(vadd_scaled(a, n, off) - o, n) / dn;
        if t < 0.0 { continue; }
        let q = vadd_scaled(vadd_scaled(o, dir, t), n, -off);
        let over = d(cross3(b - a, q - a), nn) >= 0.0
            && d(cross3(c - b, q - b), nn) >= 0.0
            && d(cross3(a - c, q - c), nn) >= 0.0;
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
fn slab_t(n: Node, o: vec3<f32>, inv: vec3<f32>, r: f32) -> f32 {
    let ax = (n.min_x - r - o.x) * inv.x; let bx = (n.max_x + r - o.x) * inv.x;
    let ay = (n.min_y - r - o.y) * inv.y; let by = (n.max_y + r - o.y) * inv.y;
    let az = (n.min_z - r - o.z) * inv.z; let bz = (n.max_z + r - o.z) * inv.z;
    let t0 = fmax(fmax(fmin(ax, bx), fmax(fmin(ay, by), fmin(az, bz))), 0.0);
    let t1 = fmin(fmax(ax, bx), fmin(fmax(ay, by), fmax(az, bz)));
    if t1 >= t0 { return t0; }
    return MISS;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count { return; }
    let k = (params.first + i) * 7u;
    let o = vec3<f32>(rays[k], rays[k + 1u], rays[k + 2u]);
    let dir = vec3<f32>(rays[k + 3u], rays[k + 4u], rays[k + 5u]);
    let tmax = rays[k + 6u];
    let r = params.r;
    let inv = vec3<f32>(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    var best_t = tmax;
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
                    let h = triangle_hit(o, dir, a, b, c, r);
                    if h.feature != 0u && h.t <= tmax {
                        if best_tri == NONE || h.t < best_t || (h.t == best_t && t < best_tri) {
                            best_t = h.t; best_tri = t; best_f = h.feature;
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
    if best_tri == NONE { out[slot] = Out(0.0, NONE, 0u, 0u); } else { out[slot] = Out(best_t, best_tri, best_f, 0u); }
}
