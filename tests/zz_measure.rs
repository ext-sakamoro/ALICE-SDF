use alice_sdf::prelude::*;
use glam::Vec2;
const REACH: f32 = 0.03;
fn circle(r: f32, c: Vec2, m: usize) -> Vec<Vec2> {
    (0..m).map(|j| { let (s, co) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32); c + Vec2::new(co, s) * r }).collect()
}
#[test]
fn m() {
    for (dy, n) in [(0.002f32, 128usize), (0.004, 64), (0.008, 32), (0.016, 24)] {
        let levels: Vec<f32> = (0..n).map(|i| dy * i as f32).collect();
        let loops: Vec<_> = (0..n).map(|i| { let x = if i < n / 2 { 0.0 } else { 0.12 }; vec![circle(0.2, Vec2::new(x, 0.0), 10)] }).collect();
        let node = SdfNode::smooth_section_stack_local(levels.clone(), loops, REACH).unwrap();
        let SdfNode::SmoothSectionStack(st) = &node else { panic!() };
        let k = st.knots();
        let r = dy / (REACH / 3.0);
        let (mut worst, mut best, mut over) = (0.0f32, f32::INFINITY, 0);
        for i in 0..k.len() - 1 {
            let lim = k[i].min(k[i + 1]) * r;
            let g = (k[i + 1] - k[i]).abs();
            if g > 0.0 { worst = worst.max(g / lim); best = best.min(g / lim); }
            if g > lim * (1.0 + 1e-5) { over += 1; }
        }
        println!("dy {dy} n {n} d/s {r:.3} predicted {:.3} max gap/limit {worst:.4} min nonzero {best:.4} pairs over limit {over}", 1.0 / (1.0 + r));
        let lo = k.iter().copied().fold(f32::INFINITY, f32::min); let hi = k.iter().copied().fold(0f32, f32::max);
        let top = k.iter().position(|&x| x == hi).unwrap();
        let falling = (0..top).filter(|&i| k[i + 1] > k[i]).count();
        println!("   H {hi:.3} L {lo:.3} levels predicted {:.2} measured rising pairs before the top {falling}", (hi / lo).ln() / (1.0 + r).ln());
        let s: Vec<String> = k.iter().step_by((n / 16).max(1)).map(|x| format!("{x:.2}")).collect();
        println!("   knots {}", s.join(" "));
    }
}
