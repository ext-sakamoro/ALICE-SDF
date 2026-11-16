// probe (not committed): compare two ways of finding the narrowest gap between any parts
// method A: gradient-pair bisection onto the medial surface, then minimise d along it
// method B: shrinking ball from surface samples, then minimise the radius over the surface
use alice_sdf::eval::eval;
use alice_sdf::SdfNode;
use glam::Vec3;
use std::cell::Cell;
use std::time::Instant;

struct F<'a> {
    s: &'a SdfNode,
    n: Cell<usize>,
}
impl F<'_> {
    fn d(&self, p: Vec3) -> f32 {
        self.n.set(self.n.get() + 1);
        eval(self.s, p)
    }
    fn g(&self, p: Vec3) -> Vec3 {
        let h = 1e-3;
        Vec3::new(
            self.d(p + Vec3::X * h) - self.d(p - Vec3::X * h),
            self.d(p + Vec3::Y * h) - self.d(p - Vec3::Y * h),
            self.d(p + Vec3::Z * h) - self.d(p - Vec3::Z * h),
        )
        .normalize_or_zero()
    }
}

fn tangents(u: Vec3) -> (Vec3, Vec3) {
    let t1 = u.any_orthonormal_vector();
    (t1, u.cross(t1))
}

fn ternary(mut lo: f32, mut hi: f32, f: &mut dyn FnMut(f32) -> f32) -> f32 {
    for _ in 0..30 {
        let a = lo + (hi - lo) / 3.0;
        let b = hi - (hi - lo) / 3.0;
        if f(a) < f(b) { hi = b } else { lo = a }
    }
    0.5 * (lo + hi)
}

// ---------- method A ----------
fn bisect_medial(f: &F, a: Vec3, b: Vec3, na: Vec3, nb: Vec3) -> Vec3 {
    let (mut lo, mut hi) = (a, b);
    for _ in 0..24 {
        let m = 0.5 * (lo + hi);
        let n = f.g(m);
        if n.dot(na) >= n.dot(nb) { lo = m } else { hi = m }
    }
    0.5 * (lo + hi)
}

/// medial point near `m` between the sides of directions na / nb, along u = normalize(na - nb)
fn on_medial(f: &F, x: Vec3, u: Vec3, h: f32, na: Vec3, nb: Vec3) -> Option<Vec3> {
    let (p, q) = (x + u * h, x - u * h);
    if f.d(p) <= 0.0 || f.d(q) <= 0.0 {
        return None;
    }
    let (np, nq) = (f.g(p), f.g(q));
    if np.dot(na) < np.dot(nb) || nq.dot(nb) < nq.dot(na) {
        return None;
    }
    Some(bisect_medial(f, p, q, na, nb))
}

#[derive(Debug, Clone, Copy)]
enum Found {
    Width(f32, Vec3),
    Crevice(f32, Vec3),
}

fn method_a(f: &F, lo: Vec3, hi: Vec3, cell: f32) -> (Vec<Found>, usize) {
    let n = ((hi - lo) / cell).ceil().as_uvec3() + 1;
    let idx = |i: u32, j: u32, k: u32| ((k * n.y + j) * n.x + i) as usize;
    let mut dv = vec![0.0f32; (n.x * n.y * n.z) as usize];
    let mut gv = vec![Vec3::ZERO; dv.len()];
    for k in 0..n.z { for j in 0..n.y { for i in 0..n.x {
        let p = lo + Vec3::new(i as f32, j as f32, k as f32) * cell;
        let d = f.d(p);
        dv[idx(i, j, k)] = d;
        if d > 0.0 { gv[idx(i, j, k)] = f.g(p); }
    }}}
    let mut out = Vec::new();
    let mut pairs = 0usize;
    for k in 0..n.z { for j in 0..n.y { for i in 0..n.x {
        let a = idx(i, j, k);
        if dv[a] <= 0.0 { continue; }
        for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
            let (ii, jj, kk) = (i + di, j + dj, k + dk);
            if ii >= n.x || jj >= n.y || kk >= n.z { continue; }
            let b = idx(ii, jj, kk);
            if dv[b] <= 0.0 { continue; }
            let (na, nb) = (gv[a], gv[b]);
            if na.dot(nb) >= 0.0 { continue; }
            pairs += 1;
            let pa = lo + Vec3::new(i as f32, j as f32, k as f32) * cell;
            let pb = lo + Vec3::new(ii as f32, jj as f32, kk as f32) * cell;
            let mut m = bisect_medial(f, pa, pb, na, nb);
            let u = (na - nb).normalize();
            let (t1, t2) = tangents(u);
            let h = 0.25 * cell;
            for _ in 0..3 {
                for t in [t1, t2] {
                    let base = m;
                    let s = ternary(-cell, cell, &mut |s| {
                        on_medial(f, base + t * s, u, h, na, nb).map_or(f32::INFINITY, |q| f.d(q))
                    });
                    if let Some(q) = on_medial(f, base + t * s, u, h, na, nb) {
                        if f.d(q) <= f.d(m) { m = q; }
                    }
                }
            }
            let e = 1e-3_f32.max(0.01 * f.d(m));
            let (xa, xb) = (m + u * e, m - u * e);
            let (ga, gb) = (f.g(xa), f.g(xb));
            let fa = xa - ga * f.d(xa);
            let fb = xb - gb * f.d(xb);
            let dm = f.d(m);
            if dm <= 4e-4 {
                let air = 180.0 - ga.dot(gb).clamp(-1.0, 1.0).acos().to_degrees();
                if air < 90.0 { out.push(Found::Crevice(air, m)); }
            } else {
                out.push(Found::Width(fa.distance(fb), m));
            }
        }
    }}}
    (out, pairs)
}

// ---------- method B ----------
fn shrink_ball(f: &F, foot: Vec3, n: Vec3, rmax: f32) -> Option<(f32, Vec3)> {
    let mut r = rmax;
    for _ in 0..64 {
        let c = foot + n * r;
        let dc = f.d(c);
        if dc >= r - 1e-5 { return if r >= rmax * 0.999 { None } else { Some((r, c)) }; }
        let q = c - f.g(c) * dc; // nearest surface point to c
        let v = foot - q;
        let den = 2.0 * n.dot(-v);
        if den <= 1e-9 { return None; }
        let rn = v.length_squared() / den;
        if !(rn.is_finite() && rn > 0.0) { return None; }
        if (r - rn).abs() < 1e-6 { return Some((rn, foot + n * rn)); }
        r = rn.min(rmax);
    }
    Some((r, foot + n * r))
}

fn method_b(f: &F, lo: Vec3, hi: Vec3, cell: f32) -> (Vec<Found>, usize) {
    let n = ((hi - lo) / cell).ceil().as_uvec3() + 1;
    let rmax = (hi - lo).length();
    let res = 3.0_f32.sqrt() * cell;
    let mut seeds: Vec<(f32, Vec3, Vec3)> = Vec::new();
    for k in 0..n.z { for j in 0..n.y { for i in 0..n.x {
        let p = lo + Vec3::new(i as f32, j as f32, k as f32) * cell;
        let d = f.d(p);
        if !(d > 0.0 && d <= res) { continue; }
        let g = f.g(p);
        let foot = p - g * d;
        if let Some((r, _)) = shrink_ball(f, foot, g, rmax) { seeds.push((r, foot, g)); }
    }}}
    let balls = seeds.len();
    seeds.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::new();
    let best = seeds.first().map(|s| s.0);
    for &(r0, foot, g) in seeds.iter().take(32) {
        if best.is_some_and(|b| r0 > b + cell) { break; }
        // minimise r over the surface around foot
        let (t1, t2) = tangents(g);
        let mut a = foot;
        let mut na = g;
        let mut r = r0;
        let proj = |x: Vec3, nn: Vec3| -> Option<(Vec3, Vec3)> {
            let px = x + nn * 0.05 * cell;
            let gx = f.g(px);
            let ft = px - gx * f.d(px);
            Some((ft, gx))
        };
        for _ in 0..3 {
            for t in [t1, t2] {
                let base = a;
                let s = ternary(-cell, cell, &mut |s| {
                    proj(base + t * s, na).and_then(|(ft, gx)| shrink_ball(f, ft, gx, rmax)).map_or(f32::INFINITY, |b| b.0)
                });
                if let Some((ft, gx)) = proj(base + t * s, na) {
                    if let Some((rr, _)) = shrink_ball(f, ft, gx, rmax) {
                        if rr <= r { r = rr; a = ft; na = gx; }
                    }
                }
            }
        }
        if r <= 2e-4 { out.push(Found::Crevice(f32::NAN, a)); } else { out.push(Found::Width(2.0 * r, a)); }
    }
    (out, balls)
}

fn scenes() -> Vec<(&'static str, SdfNode, Vec3, Vec3, f32)> {
    let floor = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let pin = |x: f32| SdfNode::cylinder(0.3, 6.0).translate(x, 3.0, 0.0);
    let pins = floor.clone().union(pin(-0.8)).union(pin(0.8));
    let pins_tight = floor.clone().union(pin(-0.45)).union(pin(0.45));
    let fin = |x: f32| SdfNode::box3d(6.0, 0.2, 8.0).translate(x, 0.0, 0.0);
    let fins = fin(-3.5).union(fin(3.5));
    let wall = SdfNode::box3d(4.0, 20.0, 20.0).translate(2.0, 0.0, 0.0);
    let hd = 2.0 * 2.0_f32.sqrt();
    let edge = wall.union(
        SdfNode::box3d(4.0, 4.0, 10.0).rotate_euler(0.0, 0.0, std::f32::consts::FRAC_PI_4).translate(-1.0 - hd, 0.0, 0.0),
    );
    let slot = SdfNode::box3d(10.0, 6.0, 10.0).translate(-5.5 + 0.37, 3.0, 0.0)
        .union(SdfNode::box3d(10.0, 6.0, 10.0).translate(5.5 + 0.37, 3.0, 0.0));
    vec![
        ("pins_1.0", pins, Vec3::new(-3.0, 0.5, -2.0), Vec3::new(3.0, 5.0, 2.0), 1.0),
        ("fins_1.0", fins, Vec3::new(-3.0, -2.0, -2.0), Vec3::new(3.0, 2.0, 2.0), 1.0),
        ("edge_wall_1.0", edge, Vec3::new(-8.0, -5.07, -3.0), Vec3::new(0.5, 5.0, 3.0), 1.0),
        ("slot_1.0", slot, Vec3::new(-6.0, 0.5, -3.0), Vec3::new(6.0, 5.5, 3.0), 1.0),
        ("pins_0.3", pins_tight, Vec3::new(-3.0, 0.5, -2.0), Vec3::new(3.0, 5.0, 2.0), 0.3),
        // two spheres r 0.3, surfaces 1.0 apart, and 1.8 apart (just above 2·√3·0.5 = 1.732)
        ("spheres_1.0", SdfNode::sphere(0.3).translate(-0.8, 0.0, 0.0).union(SdfNode::sphere(0.3).translate(0.8, 0.0, 0.0)),
            Vec3::new(-2.0, -1.5, -1.5), Vec3::new(2.0, 1.5, 1.5), 1.0),
        ("spheres_1.8", SdfNode::sphere(0.3).translate(-1.2, 0.0, 0.0).union(SdfNode::sphere(0.3).translate(1.2, 0.0, 0.0)),
            Vec3::new(-2.5, -1.5, -1.5), Vec3::new(2.5, 1.5, 1.5), 1.8),
    ]
}

#[test]
#[ignore = "probe"]
fn zz_medial_compare() {
    eprintln!("ZZ scene cell off | A: min found err evals ms pairs | B: min found err evals ms balls");
    for (name, s, lo0, hi, truth) in scenes() {
        for cell in [0.25_f32, 0.4, 0.5] {
            for off in [0.0_f32, 0.07, 0.13, 0.21] {
                let lo = lo0 + Vec3::splat(off);
                let f = F { s: &s, n: Cell::new(0) };
                let t = Instant::now();
                let (ra, pairs) = method_a(&f, lo, hi, cell);
                let (ea, ta) = (f.n.get(), t.elapsed().as_millis());
                let ma = ra.iter().filter_map(|x| if let Found::Width(w, _) = x { Some(*w) } else { None }).fold(f32::INFINITY, f32::min);
                let f = F { s: &s, n: Cell::new(0) };
                let t = Instant::now();
                let (rb, balls) = method_b(&f, lo, hi, cell);
                let (eb, tb) = (f.n.get(), t.elapsed().as_millis());
                let mb = rb.iter().filter_map(|x| if let Found::Width(w, _) = x { Some(*w) } else { None }).fold(f32::INFINITY, f32::min);
                eprintln!(
                    "ZZ {name} {cell} {off} | A: {ma:.4} {} {:.4} {ea} {ta} {pairs} | B: {mb:.4} {} {:.4} {eb} {tb} {balls}",
                    ma.is_finite(), (ma - truth).abs(), mb.is_finite(), (mb - truth).abs()
                );
            }
        }
    }
}
