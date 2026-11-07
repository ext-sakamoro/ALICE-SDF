// Closest point on a triangle, with the same arithmetic as BvhTriangle::closest_point (src/mesh/bvh.rs):
// every vector operation is written out so a sum has one known order. Shared by the mesh kernels.

const REGION_DET_ERR: f32 = 1.9073486328125e-6;

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
