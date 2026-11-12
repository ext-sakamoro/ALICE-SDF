//! Mirrored cell repetition — `SdfNode::repeat_cells`
//!
//! Repeats an off-centre box over cells of spacing (2, 1.5) on x / y, once
//! infinitely and once over 3 × 2 cells, with neighbouring copies mirrored
//! across each cell boundary. Each distance is printed next to the minimum
//! over every copy (the copies are placed here, not by the node), which it
//! equals for a child inside its cell; `modifier_repeat_cells` shows the local
//! point. Invalid parameters are refused with `RepeatCellsError`. The full
//! oracle is `tests/test_repeat_cells_oracle.rs`.
//!
//! # Running
//! ```bash
//! cargo run --example repeat_cells
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::modifiers::{modifier_repeat_cells, RepeatCellsError};
use alice_sdf::prelude::*;

const SPACING: Vec3 = Vec3::new(2.0, 1.5, 0.0);

fn child() -> SdfNode {
    SdfNode::box3d_half_extents(0.3, 0.2, 0.25).translate(0.45, 0.2, 0.0)
}

/// The minimum over the copies in cells `xs` × `ys` (cell index, centre in
/// units of spacing), each mirrored on the axes whose index is odd.
fn every_copy(p: Vec3, xs: &[(i32, f32)], ys: &[(i32, f32)]) -> f32 {
    let c = child();
    let mut best = f32::INFINITY;
    for &(cx, ox) in xs {
        for &(cy, oy) in ys {
            let mut q = p - Vec3::new(ox * SPACING.x, oy * SPACING.y, 0.0);
            if cx.rem_euclid(2) == 1 {
                q.x = -q.x;
            }
            if cy.rem_euclid(2) == 1 {
                q.y = -q.y;
            }
            best = best.min(eval(&c, q));
        }
    }
    best
}

fn main() -> Result<(), RepeatCellsError> {
    let infinite = child().repeat_cells(SPACING, None, true)?;
    let finite = child().repeat_cells(SPACING, Some([3, 2, 1]), true)?;

    let cells = |n: i32| (-n..=n).map(|c| (c, c as f32)).collect::<Vec<_>>();
    let centred = |n: i32| {
        (0..n)
            .map(|c| (c, c as f32 - (n as f32 - 1.0) * 0.5))
            .collect::<Vec<_>>()
    };

    println!("mirrored repetition, spacing {SPACING:?}");
    for p in [
        Vec3::new(0.4, 0.1, 0.0),
        Vec3::new(1.7, -0.6, 0.2),
        Vec3::new(-3.1, 2.4, -0.1),
        Vec3::new(5.5, 0.9, 0.0),
    ] {
        let got = eval(&infinite, p);
        let want = every_copy(p, &cells(6), &cells(6));
        let got_n = eval(&finite, p);
        let want_n = every_copy(p, &centred(3), &centred(2));
        println!(
            "  p {:>22}  infinite {got:>9.5} (every copy {want:>9.5})  3 x 2 {got_n:>9.5} (every copy {want_n:>9.5})  local {:?}",
            format!("{p:?}"),
            modifier_repeat_cells(p, SPACING, None, true)
        );
        assert!((got - want).abs() <= 2e-6 && (got_n - want_n).abs() <= 2e-6);
    }

    let refused = child().repeat_cells(Vec3::new(-1.0, 1.0, 0.0), None, true);
    assert!(matches!(refused, Err(RepeatCellsError::Spacing(_))));
    if let Err(e) = refused {
        println!("refused: {e}");
    }
    Ok(())
}
