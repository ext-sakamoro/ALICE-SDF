// One compute kernel: sphere traces rays against |distance to a triangle mesh| - shell. Walks a
// flattened MeshBvh with an explicit stack, in the same order and with the same arithmetic as
// MeshBvh::closest_triangle. Every vector operation is written out so a sum has one known order.
struct Params { first: u32, count: u32, shell: f32, hit: f32, max_evals: u32, pad0: u32, pad1: u32, pad2: u32, }
struct Node { min_x: f32, min_y: f32, min_z: f32, a: u32, max_x: f32, max_y: f32, max_z: f32, b: u32, }
struct Out { s: f32, tri: u32, evals: u32, hit: u32, }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rays: array<f32>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> leaf_tris: array<u32>;
@group(0) @binding(4) var<storage, read> tris: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<Out>;
@group(0) @binding(6) var<storage, read> parents: array<u32>;

const LEAF: u32 = 0x80000000u;
const REGION_DET_ERR: f32 = 1.9073486328125e-6;
const EPS_F32: f32 = 1.1920928955078125e-7;
const STACK: u32 = 64u;

fn d(a: vec3<f32>, b: vec3<f32>) -> f32 { return a.x * b.x + a.y * b.y + a.z * b.z; }
fn l1(x: vec3<f32>) -> f32 { return abs(x.x) + abs(x.y) + abs(x.z); }
fn vabs(x: vec3<f32>) -> vec3<f32> { return vec3<f32>(abs(x.x), abs(x.y), abs(x.z)); }
fn vadd_scaled(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return vec3<f32>(a.x + b.x * t, a.y + b.y * t, a.z + b.z * t);
}

fn on_segment(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let len_sq = d(ab, ab);
    if len_sq == 0.0 { return a; }
    let t = min(max(d(ab, p - a) / len_sq, 0.0), 1.0);
    return vadd_scaled(a, ab, t);
}

fn degenerate(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let q_ab = on_segment(p, a, b);
    let q_bc = on_segment(p, b, c);
    let q_ca = on_segment(p, c, a);
    var best = a;
    var best_d = 0x1.fffffep+127f;
    let e_ab = p - q_ab; let d_ab = d(e_ab, e_ab);
    if d_ab < best_d { best_d = d_ab; best = q_ab; }
    let e_bc = p - q_bc; let d_bc = d(e_bc, e_bc);
    if d_bc < best_d { best_d = d_bc; best = q_bc; }
    let e_ca = p - q_ca; let d_ca = d(e_ca, e_ca);
    if d_ca < best_d { best_d = d_ca; best = q_ca; }
    return best;
}

fn face_point(a: vec3<f32>, ab: vec3<f32>, ac: vec3<f32>, v: f32, w: f32) -> vec3<f32> {
    return vec3<f32>(a.x + ab.x * v + ac.x * w, a.y + ab.y * v + ac.y * w, a.z + ab.z * v + ac.z * w);
}

fn ill_conditioned(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let edge = degenerate(p, a, b, c);
    let ab = b - a; let ac = c - a; let ap = p - a; let bp = p - b; let cp = p - c;
    let d1 = d(ab, ap); let d2 = d(ac, ap); let d3 = d(ab, bp); let d4 = d(ac, bp); let d5 = d(ab, cp); let d6 = d(ac, cp);
    let va = d3 * d6 - d5 * d4; let vb = d5 * d2 - d1 * d6; let vc = d1 * d4 - d3 * d2;
    let denom = va + vb + vc;
    if va >= 0.0 && vb >= 0.0 && vc >= 0.0 && denom > 0.0 {
        let inv = 1.0 / denom;
        let q = face_point(a, ab, ac, vb * inv, vc * inv);
        let eq = p - q; let ee = p - edge;
        if d(eq, eq) <= d(ee, ee) { return q; }
    }
    return edge;
}

fn closest(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a; let ac = c - a;
    let nx = ab.y * ac.z - ab.z * ac.y; let ny = ab.z * ac.x - ab.x * ac.z; let nz = ab.x * ac.y - ab.y * ac.x;
    if nx == 0.0 && ny == 0.0 && nz == 0.0 { return degenerate(p, a, b, c); }
    let ap = p - a;
    let d1 = d(ab, ap); let d2 = d(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return a; }
    let bp = p - b;
    let d3 = d(ab, bp); let d4 = d(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return b; }
    let abs_ab = vabs(ab); let abs_ac = vabs(ac); let abs_ap = vabs(ap); let abs_bp = vabs(bp);
    let s1 = d(abs_ab, abs_ap); let s2 = d(abs_ac, abs_ap); let s3 = d(abs_ab, abs_bp); let s4 = d(abs_ac, abs_bp);
    let vc = d1 * d4 - d3 * d2;
    let err_c = REGION_DET_ERR * (s1 * s4 + s3 * s2);
    if abs(vc) <= err_c { return ill_conditioned(p, a, b, c); }
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        if denom != 0.0 { return vadd_scaled(a, ab, d1 / denom); }
        return a;
    }
    let cp = p - c;
    let d5 = d(ab, cp); let d6 = d(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return c; }
    let abs_cp = vabs(cp);
    let s5 = d(abs_ab, abs_cp); let s6 = d(abs_ac, abs_cp);
    let vb = d5 * d2 - d1 * d6;
    let err_b = REGION_DET_ERR * (s5 * s2 + s1 * s6);
    if abs(vb) <= err_b { return ill_conditioned(p, a, b, c); }
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        if denom != 0.0 { return vadd_scaled(a, ac, d2 / denom); }
        return a;
    }
    let va = d3 * d6 - d5 * d4;
    let err_a = REGION_DET_ERR * (s3 * s6 + s5 * s4);
    if abs(va) <= err_a { return ill_conditioned(p, a, b, c); }
    let bc_num = d4 - d3; let bc_den = d5 - d6;
    if va <= 0.0 && bc_num >= 0.0 && bc_den >= 0.0 {
        let denom = bc_num + bc_den;
        if denom != 0.0 { return vadd_scaled(b, c - b, bc_num / denom); }
        return b;
    }
    let denom = va + vb + vc;
    if denom == 0.0 { return degenerate(p, a, b, c); }
    let face_err = 2.0 * (l1(ab) + l1(ac)) * (err_a + err_b + err_c);
    if face_err > denom * REGION_DET_ERR * (l1(ap) + l1(bp) + l1(cp)) { return ill_conditioned(p, a, b, c); }
    let inv = 1.0 / denom;
    return face_point(a, ab, ac, vb * inv, vc * inv);
}

fn box_bound(n: Node, p: vec3<f32>) -> f32 {
    let cx = (n.min_x + n.max_x) * 0.5; let cy = (n.min_y + n.max_y) * 0.5; let cz = (n.min_z + n.max_z) * 0.5;
    let qx = abs(p.x - cx) - (n.max_x - n.min_x) * 0.5;
    let qy = abs(p.y - cy) - (n.max_y - n.min_y) * 0.5;
    let qz = abs(p.z - cz) - (n.max_z - n.min_z) * 0.5;
    let ox = max(qx, 0.0); let oy = max(qy, 0.0); let oz = max(qz, 0.0);
    let v = sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0);
    return max(v, 0.0);
}

fn may_hold(n: Node, bound: f32, p: vec3<f32>, best_sq: f32) -> bool {
    let sp = max(abs(p.x), max(abs(p.y), abs(p.z)));
    let smin = max(abs(n.min_x), max(abs(n.min_y), abs(n.min_z)));
    let smax = max(abs(n.max_x), max(abs(n.max_y), abs(n.max_z)));
    let scale = max(max(sp, smin), smax);
    return bound <= sqrt(best_sq) + 16.0 * EPS_F32 * scale;
}

struct Nearest { d_sq: f32, tri: u32, }

fn nearest(p: vec3<f32>) -> Nearest {
    var best = Nearest(0x1.fffffep+127f, 0xffffffffu);
    var stack_node: array<u32, 64>;
    var stack_bound: array<f32, 64>;
    var top = 0u;
    var at = 0u;
    var walking = true;
    loop {
        if !walking {
            if top == 0u { break; }
            top -= 1u;
            let n = nodes[stack_node[top]];
            if !may_hold(n, stack_bound[top], p, best.d_sq) { continue; }
            at = stack_node[top];
            walking = true;
        }
        let n = nodes[at];
        if (n.a & LEAF) != 0u {
            let first = n.a & ~LEAF;
            for (var i = first; i < first + n.b; i += 1u) {
                let t = leaf_tris[i];
                let o = t * 9u;
                let a = vec3<f32>(tris[o], tris[o + 1u], tris[o + 2u]);
                let b = vec3<f32>(tris[o + 3u], tris[o + 4u], tris[o + 5u]);
                let c = vec3<f32>(tris[o + 6u], tris[o + 7u], tris[o + 8u]);
                let e = p - closest(p, a, b, c);
                let d_sq = d(e, e);
                if d_sq < best.d_sq || (d_sq == best.d_sq && t < best.tri) {
                    best = Nearest(d_sq, t);
                }
            }
            walking = false;
            continue;
        }
        let bl = box_bound(nodes[n.a], p);
        let br = box_bound(nodes[n.b], p);
        var first = n.a; var first_bound = bl; var second = n.b; var second_bound = br;
        if !(bl <= br) { first = n.b; first_bound = br; second = n.a; second_bound = bl; }
        if top < STACK {
            stack_node[top] = second;
            stack_bound[top] = second_bound;
            top += 1u;
        }
        if may_hold(nodes[first], first_bound, p, best.d_sq) {
            at = first;
        } else {
            walking = false;
        }
    }
    return best;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count { return; }
    let r = (params.first + i) * 7u;
    let n = nearest(vec3<f32>(rays[r], rays[r + 1u], rays[r + 2u]));
    out[i] = Out(sqrt(n.d_sq), n.tri, 0u, 0u);
}
