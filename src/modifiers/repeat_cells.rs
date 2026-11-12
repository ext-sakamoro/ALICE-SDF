//! Cell repetition with mirroring (`SdfNode::RepeatCells`)
//!
//! On each axis with spacing `s > 0` the point is mapped into one cell and the
//! child is evaluated there:
//!
//! * infinite: cell `c = round_half_up(p·(1/s))`, local `q = p − c·s` (the
//!   same operations, in the same order, as `RepeatInfinite`, so without
//!   mirroring the two agree to the bit);
//! * `n` cells centred on the origin: `c = clamp(round_half_up(p·(1/s) + h),
//!   0, n − 1)` with `h = (n − 1)/2`, local `q = p − (c − h)·s` (points past the
//!   ends use the end cell);
//! * `mirror`: in odd cells the local coordinate is negated, so neighbouring
//!   copies are reflections of each other across the cell boundary.
//!
//! An axis with spacing 0 is not repeated.
//!
//! **Exactness.** Mirrored repetition of a child that lies inside its cell is
//! exact: the point and the child are on the same side of every mirror plane
//! between cells, so `|R(p) − q| ≥ |p − q|` for each reflection `R` and each
//! child point `q` — no reflected copy is closer than the copy in `p`'s own
//! cell. The fold is continuous, so the field keeps the child's Lipschitz
//! bound. A child wider than its cell is **approximate**: the value can exceed
//! the true distance (a neighbouring copy is closer); `tests/test_repeat_cells_oracle.rs`
//! measures that error. Without mirroring the field jumps at cell boundaries
//! unless the child is symmetric in its cell, as with `RepeatInfinite`.
//!
//! **Not yet supported** (fields present so that the stored form does not
//! change when they arrive): `neighbours` (minimum over the neighbouring cells)
//! and `variation` (per-cell variation from an integer hash of the cell index).
//! Every entry refuses them with [`RepeatCellsError::NotYetSupported`].
//!
//! The same domain-repetition methods are also known from Inigo Quilez's
//! articles (iquilezles.org/articles/sdfrepetition).
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Per-cell variation of a [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells)
/// (not yet supported: refused by every entry until the neighbour minimum lands).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CellVariation {
    /// Seed of the integer hash of the cell index
    pub seed: u32,
    /// Amplitude of the per-cell scale (`1 ± scale`)
    pub scale: f32,
}

/// Why a [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells) was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum RepeatCellsError {
    /// A spacing component is negative or not finite, or every component is 0
    Spacing(String),
    /// A cell count is 0
    Count(String),
    /// A field whose law has not landed yet (`neighbours`, `variation`)
    NotYetSupported(&'static str),
}

impl std::fmt::Display for RepeatCellsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spacing(s) => write!(f, "RepeatCells spacing: {s}"),
            Self::Count(s) => write!(f, "RepeatCells count: {s}"),
            Self::NotYetSupported(what) => {
                write!(f, "RepeatCells {what} is not supported yet")
            }
        }
    }
}

impl std::error::Error for RepeatCellsError {}

/// Check a [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells)'s parameters.
///
/// # Errors
///
/// [`RepeatCellsError::Spacing`] for a negative or non-finite spacing component
/// or an all-zero spacing; [`RepeatCellsError::Count`] for a zero cell count;
/// [`RepeatCellsError::NotYetSupported`] for `neighbours` or `variation`.
pub fn validate_repeat_cells(
    spacing: Vec3,
    count: Option<[u32; 3]>,
    neighbours: bool,
    variation: Option<CellVariation>,
) -> Result<(), RepeatCellsError> {
    for (axis, s) in ["x", "y", "z"].iter().zip(spacing.to_array()) {
        if !s.is_finite() || s < 0.0 {
            return Err(RepeatCellsError::Spacing(format!("{axis} = {s}")));
        }
    }
    if spacing.cmpeq(Vec3::ZERO).all() {
        return Err(RepeatCellsError::Spacing(
            "every axis is 0 (nothing repeats)".into(),
        ));
    }
    if let Some(c) = count {
        if c.contains(&0) {
            return Err(RepeatCellsError::Count(format!("{c:?} has a 0")));
        }
    }
    if neighbours {
        return Err(RepeatCellsError::NotYetSupported("neighbours"));
    }
    if variation.is_some() {
        return Err(RepeatCellsError::NotYetSupported("variation"));
    }
    Ok(())
}

/// The cell law on one axis (generic over [`Real`]): `(cell index, local
/// coordinate before mirroring)`. `count == 0` is the infinite axis.
#[inline(always)]
fn axis_cell_r<R: Real>(p: R, s: f32, count: u32) -> (R, R) {
    let inv = 1.0 / s;
    if count == 0 {
        let c = (p * R::splat(inv)).round_half_up();
        (c, p - c * R::splat(s))
    } else {
        let h = (count as f32 - 1.0) * 0.5;
        let c = (p * R::splat(inv) + R::splat(h))
            .round_half_up()
            .max(R::zero())
            .min(R::splat(count as f32 - 1.0));
        (c, p - (c - R::splat(h)) * R::splat(s))
    }
}

/// `1 − 2·(c mod 2)`: exactly ±1 for an integer-valued `c`.
#[inline(always)]
fn parity_sign_r<R: Real>(c: R) -> R {
    let odd = c - (c * R::splat(0.5)).floor() * R::splat(2.0);
    R::one() - odd * R::splat(2.0)
}

/// The cell and mirror law on one axis (generic over [`Real`]): the local
/// coordinate; `s <= 0` leaves the coordinate unchanged.
#[inline(always)]
fn axis_r<R: Real>(p: R, s: f32, count: u32, mirror: bool) -> R {
    if !(s > 0.0) {
        return p;
    }
    let (c, q) = axis_cell_r(p, s, count);
    if mirror {
        q * parity_sign_r(c)
    } else {
        q
    }
}

/// The local point of [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells)
/// (generic over [`Real`]); `count` 0 on an axis means infinite.
#[inline(always)]
pub fn repeat_cells_r<R: Real>(p: Vec3R<R>, spacing: Vec3, count: [u32; 3], mirror: bool) -> Vec3R<R> {
    Vec3R::new(
        axis_r(p.x, spacing.x, count[0], mirror),
        axis_r(p.y, spacing.y, count[1], mirror),
        axis_r(p.z, spacing.z, count[2], mirror),
    )
}

/// The local point of [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells).
#[inline(always)]
#[must_use]
pub fn modifier_repeat_cells(point: Vec3, spacing: Vec3, count: Option<[u32; 3]>, mirror: bool) -> Vec3 {
    repeat_cells_r::<f32>(point.into(), spacing, count.unwrap_or([0; 3]), mirror).into()
}

/// The sign (±1 per axis) the mirror applied at `point`, for the analytic
/// gradient (the gradient of a reflected axis is negated).
#[inline]
#[must_use]
pub fn repeat_cells_mirror_sign(point: Vec3, spacing: Vec3, count: Option<[u32; 3]>, mirror: bool) -> Vec3 {
    if !mirror {
        return Vec3::ONE;
    }
    let c = count.unwrap_or([0; 3]);
    let sign = |p: f32, s: f32, n: u32| -> f32 {
        if s > 0.0 {
            parity_sign_r::<f32>(axis_cell_r::<f32>(p, s, n).0)
        } else {
            1.0
        }
    };
    Vec3::new(
        sign(point.x, spacing.x, c[0]),
        sign(point.y, spacing.y, c[1]),
        sign(point.z, spacing.z, c[2]),
    )
}

/// Refuse a decoded tree holding a [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells)
/// whose parameters [`validate_repeat_cells`] rejects (the readers call this on
/// every decoded tree, so a stored `neighbours` / `variation` never reaches an evaluator).
pub(crate) fn check_repeat_cells_tree(root: &crate::SdfNode) -> Result<(), RepeatCellsError> {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if let crate::SdfNode::RepeatCells {
            spacing,
            count,
            neighbours,
            variation,
            ..
        } = node
        {
            validate_repeat_cells(*spacing, *count, *neighbours, *variation)?;
        }
        stack.extend(crate::diff::node_children(node));
    }
    Ok(())
}
