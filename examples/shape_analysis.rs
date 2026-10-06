//! Shape analysis: measure a part, slice it, hollow it, check a fit, and
//! export it only if it is printable.
//!
//! Every step prints its numbers and checks them against the closed form of
//! the shape, so the example fails loudly if an API stops meaning what it says.
//!
//! Run: `cargo run --example shape_analysis`
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::CompiledSdf;
use alice_sdf::eval::eval;
use alice_sdf::fidelity::{distance_fidelity, Fidelity};
use alice_sdf::heatmap::{generate_heatmap, heatmap_to_rgba, ColorMap, HeatmapConfig, SlicePlane};
use alice_sdf::io::step::StepConfig;
use alice_sdf::measure::{
    estimate_center_of_mass, estimate_surface_area, estimate_volume, measure_tension,
};
use alice_sdf::morphology::{
    eval_offset, eval_offset_batch, eval_offset_batch_parallel, tolerance_fits,
    tolerance_max_violation,
};
use alice_sdf::shell::{
    eval_shell, eval_shell_batch, eval_shell_batch_parallel, eval_shell_compiled,
    eval_shell_compiled_batch_parallel, eval_shell_gradient, shell_node, ShellConfig,
};
use alice_sdf::types::{Aabb, SdfNode};
use alice_sdf::validity::{export_step_validated, PrintRequirements, ValidatedExportError};
use glam::Vec3;
use std::f64::consts::PI;
use std::sync::Arc;

const fn cube3() -> Aabb {
    Aabb {
        min: Vec3::splat(-3.0),
        max: Vec3::splat(3.0),
    }
}

fn main() {
    let r = 1.0_f32;
    let ball = SdfNode::sphere(r);
    let region = Aabb {
        min: Vec3::splat(-1.2),
        max: Vec3::splat(1.2),
    };

    // ── measure ─────────────────────────────────────────────
    let vol = estimate_volume(&ball, region, 200_000, 1);
    let area = estimate_surface_area(&ball, region, 400_000, 0.02, 2);
    let com = estimate_center_of_mass(&ball.clone().translate(0.2, 0.0, 0.0), region, 200_000, 3);
    let exact_v = 4.0 / 3.0 * PI;
    let exact_a = 4.0 * PI;
    println!(
        "volume  {:.4} ± {:.4} (4/3·π = {exact_v:.4}, fill {:.3})",
        vol.volume, vol.std_error, vol.fill_ratio
    );
    println!(
        "area    {:.4} ± {:.4} (4·π = {exact_a:.4})",
        area.area, area.std_error
    );
    println!(
        "center  {:?} from {} interior samples (expected x = 0.2)",
        com.center, com.interior_count
    );
    assert!((vol.volume - exact_v).abs() < 4.0 * vol.std_error.max(1e-3));
    assert!((area.area - exact_a).abs() / exact_a < 0.04);
    assert!((com.center.x - 0.2).abs() < 0.02);

    // ── fidelity: what a distance value is worth ────────────
    let tension = measure_tension(&ball, region, 20_000, 4, 1e-3);
    let fidelity = distance_fidelity(&ball);
    println!(
        "tension {:.4} (tears: {}), fidelity {fidelity:?}, can overshoot: {}",
        tension.tension(),
        tension.tears(),
        fidelity.can_overshoot()
    );
    // an exact field reads 1 up to f32 rounding of the difference quotient
    assert!(tension.tension() <= 1.0 + 1e-3);
    assert!(matches!(fidelity, Fidelity::NeverOverReports { .. }));
    let stretched = ball.clone().scale_xyz(1.0, 2.0, 1.0);
    let stretched_fidelity = distance_fidelity(&stretched);
    println!(
        "stretched (1, 2, 1): {stretched_fidelity:?}, step divisor {:?}",
        stretched_fidelity.safe_step_scale()
    );
    assert!(stretched_fidelity.can_overshoot());
    assert_eq!(stretched_fidelity.safe_step_scale(), Some(2.0));
    // the bound is only a bound: a non-uniform scale is slack, a gyroid tears
    let stretched_tension = measure_tension(&stretched, region, 20_000, 4, 1e-3);
    let gyroid = SdfNode::gyroid(1.0, 0.1);
    let gyroid_tension = measure_tension(&gyroid, cube3(), 30_000, 5, 1e-3);
    println!(
        "measured tension: stretched {:.4}, gyroid {:.4} (bound {:?})",
        stretched_tension.tension(),
        gyroid_tension.tension(),
        distance_fidelity(&gyroid).safe_step_scale()
    );
    assert!(stretched_tension.tension() <= 2.0 + 1e-3);
    assert!(gyroid_tension.tears() && distance_fidelity(&gyroid).can_overshoot());

    // ── heatmap: a cross-section through the centre ────────
    let config = HeatmapConfig {
        resolution: 64,
        range: 1.5,
        plane: SlicePlane::XY(0.0),
    };
    let hm = generate_heatmap(&ball, &config);
    let rgba = heatmap_to_rgba(&hm, ColorMap::Coolwarm);
    let pixel_area = (2.0 * config.range / config.resolution as f32).powi(2);
    let inside_area = hm.inside_pixel_count() as f32 * pixel_area;
    println!(
        "slice   {}×{}, range [{:.3}, {:.3}], inside area {inside_area:.4} (π = {:.4}), \
         surface pixels {}",
        hm.width,
        hm.height,
        hm.min_val,
        hm.max_val,
        std::f32::consts::PI,
        hm.surface_pixel_count(0.05)
    );
    assert!((inside_area - std::f32::consts::PI).abs() < 0.05);
    assert_eq!(rgba.len(), (hm.width * hm.height) as usize);
    // the image centre is inside the ball: blue side of the map
    let centre = hm.sample(32, 32);
    assert!(centre < 0.0 && rgba[(32 * hm.width + 32) as usize][2] == 255);

    // ── shell: a 0.1 inner / 0.3 outer wall ────────────────
    let wall = ShellConfig::new(0.1, 0.3);
    let compiled = CompiledSdf::compile(&ball);
    let probes: Vec<Vec3> = [0.8_f32, 0.9, 1.1, 1.3, 1.5]
        .iter()
        .map(|&x| Vec3::new(x, 0.0, 0.0))
        .collect();
    let seq = eval_shell_batch(&ball, &probes, &wall);
    let par = eval_shell_batch_parallel(&ball, &probes, &wall);
    let cpar = eval_shell_compiled_batch_parallel(&compiled, &probes, &wall);
    println!(
        "shell   thickness {} (uniform 0.2: {})",
        wall.wall_thickness(),
        ShellConfig::uniform(0.2).wall_thickness()
    );
    for (i, p) in probes.iter().enumerate() {
        // exact distance to the annulus 0.9 ≤ |p| ≤ 1.3
        let exact = (p.x - 1.3).max(0.9 - p.x);
        println!(
            "  x = {:.1}: shell {:+.3} (annulus {exact:+.3})",
            p.x, seq[i]
        );
        for got in [
            seq[i],
            par[i],
            cpar[i],
            eval_shell(&ball, *p, &wall),
            eval_shell_compiled(&compiled, *p, &wall),
        ] {
            assert!((got - exact).abs() < 1e-5);
        }
    }
    // the same wall as a node, so it composes with the rest of a tree
    let wall_node = shell_node(Arc::new(ball.clone()), wall);
    for p in &probes {
        let exact = (p.x - 1.3).max(0.9 - p.x);
        assert!((eval(&wall_node, *p) - exact).abs() < 1e-5);
    }
    let n_out = eval_shell_gradient(&ball, Vec3::new(1.5, 0.0, 0.0), &wall);
    let n_hole = eval_shell_gradient(&ball, Vec3::new(0.5, 0.0, 0.0), &wall);
    assert!(n_out.x > 0.99 && n_hole.x < -0.99);

    // ── morphology: offset and fit tolerance ───────────────
    let pts = [Vec3::new(1.2, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0)];
    let dilated = eval_offset_batch(&ball, &pts, 0.25);
    let eroded = eval_offset_batch_parallel(&ball, &pts, -0.25);
    println!("offset  +0.25: {dilated:?}, −0.25: {eroded:?}");
    assert!((eval_offset(&ball, pts[0], 0.25) - (1.2 - 1.25)).abs() < 1e-5);
    assert!((eroded[1] - (0.5 - 0.75)).abs() < 1e-5);
    let peg = SdfNode::sphere(1.2);
    let fits = tolerance_fits(&peg, &ball, 0.25, 4096, 1.3);
    let gap = tolerance_max_violation(&peg, &ball, 0.1, 25 * 25 * 25, 1.2);
    println!("fit     peg 1.2 in hole 1.0: +0.25 fits = {fits}, +0.1 violation = {gap:?}");
    assert!(fits);
    assert!((gap.expect("does not fit") - 0.1).abs() < 1e-5);

    // ── validity: export only what can be printed ──────────
    let dir = std::env::temp_dir().join(format!("alice_sdf_shape_analysis_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let cfg = StepConfig {
        bounds: (-2.0, 2.0),
        resolution: 24,
        ..StepConfig::default()
    };
    // a 0.3 sheet against a 0.8 wall requirement: refused, nothing written
    let sheet = SdfNode::box3d(3.0, 3.0, 0.3);
    let sheet_path = dir.join("sheet.step");
    match export_step_validated(
        &sheet_path,
        &sheet,
        &cfg,
        PrintRequirements::fdm_0_4_nozzle(),
    ) {
        Err(ValidatedExportError::NotPrintable(report)) => {
            println!(
                "sheet   refused: {}",
                ValidatedExportError::NotPrintable(report)
            );
        }
        other => panic!("a 0.3 sheet must be refused, got {other:?}"),
    }
    assert!(!sheet_path.exists());
    // a 2-unit block with overhang allowed up to 90°: written
    let block = SdfNode::box3d(2.0, 2.0, 2.0);
    let block_path = dir.join("block.step");
    let req = PrintRequirements {
        max_overhang: std::f32::consts::FRAC_PI_2,
        ..PrintRequirements::fdm_0_4_nozzle()
    };
    let report = export_step_validated(&block_path, &block, &cfg, req).expect("printable");
    println!(
        "block   written: min local thickness {:?}, erosion {:?}",
        report.min_local_thickness, report.erosion
    );
    assert!(report.is_printable() && block_path.exists());
    std::fs::remove_dir_all(&dir).ok();

    println!("shape_analysis: all checks passed");
}
