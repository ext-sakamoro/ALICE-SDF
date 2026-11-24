//! Gap widths — measure the narrowest hole / slot of a shape and compare it to a printing limit
//!
//! A post standing 1.5 from a wall leaves an open-topped slot. `measure_gaps` samples the empty
//! space near the surface and reports the narrowest width wall to wall, the sharp crevices where two
//! walls meet (which have no width), samples skipped on edges, and zero-thickness sheets (a defect
//! of the field, never a width). `local_gap` measures the width from one surface point.
//! `prove_gaps_in_layers` with `verdict_in_layers` reads the same slot for a part printed in layers:
//! narrow air open in its own layer is a neck between separate walls, not a hole.
//!
//! # Running
//! ```bash
//! cargo run --example gap_widths
//! ```

use alice_sdf::prelude::*;
use alice_sdf::validity::{
    local_gap, measure_gaps, prove_gaps, prove_gaps_in_layers, Crevice, GapError, GapFailure,
    GapProof, GapReport, GapUndetermined, GapVerdict, GAP_PROOF_WORK, GAP_TOLERANCE, MAX_CREVICES,
    MAX_GAP_GRID_POINTS, MAX_GAP_WITNESSES,
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

fn describe_proof(name: &str, proof: &GapProof) {
    let kept = if proof.truncated {
        format!(" (first {MAX_GAP_WITNESSES} kept)")
    } else {
        String::new()
    };
    println!(
        "{name}: {} witnesses{kept}, {} undecided, {} open boxes, {} boxes",
        proof.witnesses.len(),
        proof.undecided,
        proof.open_boxes,
        proof.boxes
    );
}

fn print_verdict(name: &str, v: GapVerdict) {
    match v {
        GapVerdict::Pass(c) => println!("{name}: pass from {}", c.proven_from),
        GapVerdict::Fail(GapFailure::ProvenGap { width, at }) => {
            println!("{name}: fail, a gap of at most {width:.3} at {at}");
        }
        GapVerdict::Fail(f) => println!("{name}: fail, {f:?}"),
        GapVerdict::Undetermined(GapUndetermined::ProofIncomplete { .. }) => {
            println!("{name}: undetermined, the proof did not finish");
        }
        GapVerdict::Undetermined(u) => println!("{name}: undetermined, {u:?}"),
    }
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

    // the verdict: the grid's reading together with the air-side proof over the slot between them
    let (slo, shi) = (Vec3::new(6.0, 0.5, -4.0), Vec3::new(11.0, 9.0, 4.0));
    match (
        measure_gaps(&part, slo, shi, cell),
        prove_gaps(&part, LIMIT, slo, shi, 8, GAP_PROOF_WORK),
    ) {
        (Ok(r), Ok(proof)) => {
            describe_proof("slot proof", &proof);
            print_verdict("verdict", r.verdict(LIMIT, &proof));
        }
        (r, p) => println!("no verdict: {:?} / {:?}", r.err(), p.err()),
    }

    // printed in layers stacked along y: the slot is open at the top in every layer, so it is a neck
    // between separate walls (the nozzle never bridges it), not a hole
    let up = Vec3::Y;
    match (
        measure_gaps(&part, slo, shi, cell),
        prove_gaps_in_layers(&part, LIMIT, slo, shi, 8, GAP_PROOF_WORK, up),
    ) {
        (Ok(r), Ok(proof)) => {
            describe_proof("slot proof in layers along y", &proof);
            print_verdict(
                "verdict in layers",
                r.verdict_in_layers(&part, LIMIT, &proof, up),
            );
        }
        (r, p) => println!("no layered verdict: {:?} / {:?}", r.err(), p.err()),
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
