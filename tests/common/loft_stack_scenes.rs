//! The loft and stack scenes the shader data oracles share (`test_shader_data_bounds.rs`,
//! `test_shader_read_ranges.rs`), and the contraction budget measured on them
#![allow(dead_code, reason = "each test file uses part of the shared scenes")]

use alice_sdf::prelude::*;
use glam::Vec2;

pub fn circle(r: f32, c: Vec2, m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, co) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            c + Vec2::new(co, s) * r
        })
        .collect()
}

/// A smooth stack whose data block exceeds the array limit: 70 levels of two loops of 120 control
/// points
pub fn big_smooth_stack() -> SdfNode {
    let levels: Vec<f32> = (0..70).map(|i| -0.35 + 0.01 * i as f32).collect();
    let loops = levels
        .iter()
        .map(|&y| vec![circle(0.12 + 0.05 * y, Vec2::new(-0.15, 0.0), 120), circle(0.1, Vec2::new(0.15, 0.02 * y), 120)])
        .collect();
    SdfNode::smooth_section_stack(levels, loops, 0.01, 0.1).expect("a valid stack")
}

pub fn section_stack() -> SdfNode {
    let levels = vec![0.0, 0.2, 0.4, 0.6];
    let loops = levels.iter().map(|&y| vec![circle(0.2 + 0.1 * y, Vec2::ZERO, 24), circle(0.05, Vec2::new(0.05, 0.0), 8)]).collect();
    SdfNode::section_stack(levels, loops).expect("a valid section stack")
}

pub fn lofts() -> SdfNode {
    let ellipse = |a: f32, b: f32| -> Vec<Vec2> { circle(1.0, Vec2::ZERO, 12).iter().map(|p| Vec2::new(a * p.x, b * p.y)).collect() };
    let curves = SdfNode::loft_curves_capped(
        vec![Vec3::ZERO, Vec3::new(0.03, 0.3, 0.0), Vec3::new(0.05, 0.6, 0.02)],
        Vec3::X,
        vec![ellipse(0.2, 0.12), ellipse(0.18, 0.13), ellipse(0.15, 0.1)],
        [[0.1, 0.0, 0.0], [0.15, 0.06, -0.03]],
    )
    .expect("capped curve loft");
    let polar = SdfNode::loft_capped(
        vec![Vec3::ZERO, Vec3::new(0.02, 0.3, 0.0), Vec3::new(0.0, 0.6, 0.03)],
        Vec3::Z,
        1,
        vec![vec![0.15, 0.01, 0.0], vec![0.17, 0.0, 0.01], vec![0.12, 0.0, 0.0]],
        [0.08, 0.12],
    )
    .expect("capped polar loft");
    curves.union(polar.translate(0.5, 0.0, 0.0))
}

pub fn trees() -> Vec<(&'static str, SdfNode)> {
    vec![
        ("big_smooth_stack", big_smooth_stack()),
        ("section_stack", section_stack()),
        ("lofts", lofts()),
        ("all", big_smooth_stack().union(section_stack().translate(0.8, 0.0, 0.0)).union(lofts().translate(0.0, 0.0, 0.8))),
    ]
}


/// The largest gap between the scene's HLSL text compiled as C++ with `-ffp-contract=fast` and
/// with `-ffp-contract=off`, relative to `max(1, |cpu|)`, over the dense points of
/// `test_shader_read_ranges.rs::the_contraction_gap_is_inside_its_budget`, times 4. MEASURED, not
/// derived (no forward error analysis of these laws exists): it stands for what a compiler that
/// fuses `a * b + c` may change, and is the bound the wgpu arm of
/// `test_shader_data_bounds.rs::no_checked_read_leaves_its_proven_range_on_the_gpu` tolerates
pub const CONTRACTION_BUDGET: &[(&str, f32)] = &[("big_smooth_stack", 0.0), ("section_stack", 0.0), ("lofts", 0.0), ("all", 0.0)];

/// The measured contraction budget of a scene
#[must_use]
pub fn contraction_budget(scene: &str) -> f32 {
    CONTRACTION_BUDGET.iter().find(|(n, _)| *n == scene).map_or_else(|| panic!("no budget for {scene}"), |(_, b)| *b)
}

/// Whether two GPU values of the same point differ by no more than the scene's budget
#[must_use]
pub fn within_contraction_budget(x: f32, y: f32, cpu: f32, budget: f32) -> bool {
    x.is_finite() && y.is_finite() && (x - y).abs() <= budget * cpu.abs().max(1.0)
}
