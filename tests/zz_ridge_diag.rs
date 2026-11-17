use alice_sdf::eval::{eval, eval_normal};
use alice_sdf::interval::{eval_interval, Interval, Vec3Interval};
use alice_sdf::interval_gradient::eval_interval_gradient_in_band;
use alice_sdf::validity::prove_gaps;
use alice_sdf::SdfNode;
use glam::Vec3;

fn show(name: &str, s: &SdfNode, limit: f32, lo: Vec3, hi: Vec3) {
    let p = prove_gaps(s, limit, lo, hi, 9).unwrap();
    println!(
        "DIAG {name} L {limit}: boxes {} undecided {} regions {} witnesses {}",
        p.boxes,
        p.undecided,
        p.undecided_at.len(),
        p.witnesses.len()
    );
    for c in p.undecided_at.iter().map(|r| r.at).take(6) {
        let c = &c;
        let h = Vec3::splat(0.006);
        let b = Vec3Interval::from_bounds(*c - h, *c + h);
        let v = eval_interval(s, b);
        let g = eval_interval_gradient_in_band(
            s,
            b,
            Interval {
                lo: 0.0,
                hi: f32::INFINITY,
            },
        );
        println!(
            "DIAG   at {c} d {:.4} n {} val [{:.4},{:.4}] g {:?}",
            eval(s, *c),
            eval_normal(s, *c).normalize_or_zero(),
            v.lo,
            v.hi,
            g.map(|g| (g.x.lo, g.x.hi, g.y.lo, g.y.hi, g.z.lo, g.z.hi))
        );
    }
}

#[test]
fn diag() {
    let spheres = SdfNode::sphere(0.3)
        .translate(-1.2, 0.0, 0.0)
        .union(SdfNode::sphere(0.3).translate(1.2, 0.0, 0.0));
    show(
        "spheres",
        &spheres,
        2.0,
        Vec3::new(-2.5, -1.5, -1.5),
        Vec3::new(2.5, 1.5, 1.5),
    );
    let floor = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let block = SdfNode::box3d(4.0, 4.0, 4.0).translate(0.0, 2.0, 0.0);
    show(
        "box on floor",
        &floor.union(block),
        2.0,
        Vec3::new(-5.0, 0.2, -5.0),
        Vec3::new(5.0, 6.0, 5.0),
    );
    let half = 30.0_f32.to_radians();
    let depth = 5.0 * 0.5 / alice_det_math::tan(half);
    let blk = SdfNode::box3d(30.0, 20.0, 10.0).translate(0.0, -10.0, 0.0);
    let wall = |sg: f32| {
        SdfNode::plane(
            Vec3::new(
                sg * alice_det_math::cos(half),
                -alice_det_math::sin(half),
                0.0,
            ),
            0.0,
        )
        .translate(0.0, -depth, 0.0)
    };
    let v = blk.subtract(wall(1.0).intersection(wall(-1.0)));
    let (lo, hi) = (Vec3::new(-6.0, -6.0, -3.0), Vec3::new(6.0, 1.0, 3.0));
    let g = alice_sdf::validity::measure_gaps(&v, lo, hi, 0.2).unwrap();
    println!(
        "DIAG V crevices {} first {:?}",
        g.crevices.len(),
        g.crevices
            .iter()
            .take(3)
            .map(|c| (c.location, c.angle_deg, c.mouth))
            .collect::<Vec<_>>()
    );
    let p = prove_gaps(&v, 0.8, lo, hi, 9).unwrap();
    for r in p.undecided_at.iter().take(4) {
        let w = r
            .walls
            .map(|(a, b)| (a, b, alice_det_math::acos(-a.dot(b)).to_degrees()));
        println!("DIAG V undecided at {} walls {:?}", r.at, w);
    }
    let blockonly = SdfNode::box3d(2.0, 2.0, 2.0);
    show(
        "lone block",
        &blockonly,
        0.5,
        Vec3::new(-2.0, -2.0, -2.0),
        Vec3::new(2.0, 2.0, 2.0),
    );
}
