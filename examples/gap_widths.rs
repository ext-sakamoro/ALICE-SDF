//! Gap widths — measure the narrowest hole / slot of a shape and compare it to a printing limit
//!
//! A post standing 1.5 from a wall leaves an open-topped slot. `measure_gaps` samples the empty
//! space near the surface and reports the narrowest width wall to wall, the sharp crevices where two
//! walls meet (which have no width), samples skipped on edges, and zero-thickness sheets (a defect
//! of the field, never a width). `local_gap` measures the width from one surface point.
//!
//! # Running
//! ```bash
//! cargo run --example gap_widths
//! ```

use alice_sdf::prelude::*;
use alice_sdf::validity::{
    local_gap, measure_gaps, Crevice, GapError, GapReport, GAP_TOLERANCE, MAX_CREVICES,
    MAX_GAP_GRID_POINTS,
};
use glam::Vec3;

/// The narrowest gap a 0.4 mm nozzle prints without the walls fusing (mm)
const LIMIT: f32 = 2.0;

fn describe(name: &str, r: &GapReport) {
    match r.min_gap {
        // a width within the measure's own error of the limit is not failed by that error
        Some(w) if w + GAP_TOLERANCE < LIMIT => {
            println!(
                "{name}: narrowest gap {w:.3} at {:?} is below {LIMIT}",
                r.location
            );
        }
        Some(w) => println!("{name}: narrowest gap {w:.3} (limit {LIMIT})"),
        None => println!("{name}: no gap (only open air around the shape)"),
    }
    let sharpest = r
        .crevices
        .iter()
        .map(|c: &Crevice| c.angle_deg)
        .reduce(f32::min);
    let at_cap = if r.crevices.len() >= MAX_CREVICES {
        "≥ "
    } else {
        ""
    };
    println!(
        "  {} samples, {} gap samples, resolution {:.3}; {at_cap}{} crevice(s), sharpest {sharpest:?}°; \
         {} edge samples; {} zero-thickness sheet samples",
        r.samples,
        r.gaps,
        r.resolution,
        r.crevices.len(),
        r.edge_samples,
        r.sheet_samples
    );
}

fn main() {
    // floor (top at y = 0), a wall whose inner face is x = 10, a post of radius 3 whose surface is 1.5 from it
    let floor = SdfNode::box3d(40.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let wall = SdfNode::box3d(4.0, 10.0, 20.0).translate(12.0, 5.0, 0.0);
    let post = SdfNode::cylinder(3.0, 10.0).translate(5.5, 5.0, 0.0);
    let part = floor.clone().union(wall).union(post);

    // a cell finer than the limit: every slot wider than √3·cell is found in any orientation
    let cell = LIMIT / (2.0 * 3.0_f32.sqrt());
    let (lo, hi) = (Vec3::new(-21.0, -3.0, -11.0), Vec3::new(21.0, 11.0, 11.0));
    let points = ((hi - lo) / cell).ceil() + Vec3::ONE;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // a few hundred per axis
    let n = points.x as usize * points.y as usize * points.z as usize;
    println!("grid {n} points (largest allowed {MAX_GAP_GRID_POINTS})");
    match measure_gaps(&part, lo, hi, cell) {
        Ok(r) => describe("post and wall", &r),
        Err(GapError::TooManyPoints) => println!("grid too fine: use a coarser cell"),
        Err(e) => println!("nothing measured: {e:?}"),
    }

    // from one point on the post's surface, straight at the wall
    let on_post = Vec3::new(8.5, 5.0, 0.0);
    println!(
        "from the post's surface toward the wall: {:?}",
        local_gap(&part, on_post, Vec3::X, 100.0)
    );

    // a plate leaning on the floor at 45 degrees: a crevice (no width), not a narrow gap
    let plate = SdfNode::box3d(40.0, 4.0, 20.0)
        .translate(20.0, 2.0, 0.0)
        .rotate_euler(0.0, 0.0, 45.0_f32.to_radians());
    if let Ok(r) = measure_gaps(
        &floor.union(plate),
        Vec3::new(0.5, -1.0, -8.0),
        Vec3::new(25.0, 25.0, 8.0),
        0.25,
    ) {
        describe("leaning plate", &r);
    }
}
