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
//! * `mirror`: the local coordinate is negated in every other cell, so
//!   neighbouring copies are reflections of each other across the cell
//!   boundary. The parity is that of `c − ⌊(n − 1)/2⌋` (of `c` when infinite),
//!   so the cell holding the origin is never reflected: with an odd count the
//!   centre cell keeps the child as it is, as the origin cell of the infinite
//!   repetition does.
//!
//! An axis with spacing 0 is not repeated. So `mirror = false` agrees with
//! `RepeatInfinite` only on axes with spacing `> 0`: there spacing 0 is not a
//! valid repetition (it divides by 0), here it leaves the axis as it is. A
//! finite count is not `RepeatFinite`'s law either (`n` cells centred on the
//! origin here, a clamped cell index there).
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
//! Their types hold only the off state: [`CellNeighbours`] has the one value
//! [`CellNeighbours::OFF`] and [`CellVariation`] has no value, so a node with
//! either field set cannot be built; every reader (JSON, `.asdf`, the I-packet)
//! refuses a stored one when it decodes it, wherever the node sits in the tree.
//!
//! The same domain-repetition methods are also known from Inigo Quilez's
//! articles (iquilezles.org/articles/sdfrepetition).
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Whether a [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells) takes the
/// minimum over the neighbouring cells (not yet supported).
///
/// Only [`CellNeighbours::OFF`] can be built; stored, it is the `bool` `false`
/// (the field keeps its stored form for when the law lands), and decoding a
/// stored `true` fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize)]
#[serde(transparent)]
pub struct CellNeighbours(bool);

impl CellNeighbours {
    /// The one value: no neighbour minimum
    pub const OFF: Self = Self(false);

    /// Whether the neighbour minimum is on (always `false` until its law lands)
    #[must_use]
    pub const fn is_on(self) -> bool {
        self.0
    }
}

impl<'de> Deserialize<'de> for CellNeighbours {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if bool::deserialize(d)? {
            Err(serde::de::Error::custom(RepeatCellsError::NotYetSupported(
                "neighbours",
            )))
        } else {
            Ok(Self::OFF)
        }
    }
}

/// Per-cell variation of a [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells)
/// (not yet supported).
///
/// It has no value yet, so the node's `variation` is always `None`; decoding a
/// stored one fails. Its stored form (seed `u32`, scale `f32`) is fixed so that
/// the field does not change when the law lands.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CellVariation {
    /// Seed of the integer hash of the cell index
    seed: u32,
    /// Amplitude of the per-cell scale (`1 ± scale`)
    scale: f32,
}

impl<'de> Deserialize<'de> for CellVariation {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[allow(dead_code, reason = "read only to consume the stored form")]
        struct Stored {
            seed: u32,
            scale: f32,
        }
        let _ = Stored::deserialize(d)?;
        Err(serde::de::Error::custom(RepeatCellsError::NotYetSupported(
            "variation",
        )))
    }
}

/// Why a [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells) was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepeatCellsError {
    /// A spacing component is negative or not finite, or every component is 0
    Spacing(String),
    /// A cell count is 0
    Count(String),
    /// A stored field whose law has not landed yet (`neighbours`, `variation`),
    /// refused when a reader decodes it
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
/// or an all-zero spacing; [`RepeatCellsError::Count`] for a zero cell count or
/// one above [`MAX_CELL_COUNT`].
pub fn validate_repeat_cells(
    spacing: Vec3,
    count: Option<[u32; 3]>,
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
        if c.iter().any(|&n| n > MAX_CELL_COUNT) {
            return Err(RepeatCellsError::Count(format!(
                "{c:?} has a count above {MAX_CELL_COUNT}"
            )));
        }
    }
    Ok(())
}

/// The largest cell count per axis (`2^16`)
///
/// With `n` cells the cell index reaches `(n − 1)/2` and the cell boundaries
/// `±(k + 1/2)·s`; up to `2^16` cells every boundary and index is exact in `f32`
/// (24-bit significand with room for the half), so the fold stays continuous
/// and the field keeps its Lipschitz bound.
pub const MAX_CELL_COUNT: u32 = 1 << 16;

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

/// The cell index whose parity decides the mirror: `⌊(n − 1)/2⌋` cells before
/// the cell holding the origin (none when infinite), so that cell is never
/// reflected
#[inline(always)]
pub const fn mirror_base(count: u32) -> f32 {
    if count == 0 {
        0.0
    } else {
        ((count - 1) / 2) as f32
    }
}

/// The cell and mirror law on one axis (generic over [`Real`]): the local
/// coordinate; `s <= 0` leaves the coordinate unchanged.
#[inline(always)]
fn axis_r<R: Real>(p: R, s: f32, count: u32, mirror: bool) -> R {
    if s.is_nan() || s <= 0.0 {
        return p;
    }
    let (c, q) = axis_cell_r(p, s, count);
    if mirror {
        q * parity_sign_r(c - R::splat(mirror_base(count)))
    } else {
        q
    }
}

/// The local point of [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells)
/// (generic over [`Real`]); `count` 0 on an axis means infinite.
#[inline(always)]
pub fn repeat_cells_r<R: Real>(
    p: Vec3R<R>,
    spacing: Vec3,
    count: [u32; 3],
    mirror: bool,
) -> Vec3R<R> {
    Vec3R::new(
        axis_r(p.x, spacing.x, count[0], mirror),
        axis_r(p.y, spacing.y, count[1], mirror),
        axis_r(p.z, spacing.z, count[2], mirror),
    )
}

/// The local point of [`SdfNode::RepeatCells`](crate::SdfNode::RepeatCells).
#[inline(always)]
#[must_use]
pub fn modifier_repeat_cells(
    point: Vec3,
    spacing: Vec3,
    count: Option<[u32; 3]>,
    mirror: bool,
) -> Vec3 {
    repeat_cells_r::<f32>(point.into(), spacing, count.unwrap_or([0; 3]), mirror).into()
}

/// The sign (±1 per axis) the mirror applied at `point`, for the analytic
/// gradient (the gradient of a reflected axis is negated).
#[inline]
#[must_use]
pub fn repeat_cells_mirror_sign(
    point: Vec3,
    spacing: Vec3,
    count: Option<[u32; 3]>,
    mirror: bool,
) -> Vec3 {
    if !mirror {
        return Vec3::ONE;
    }
    let c = count.unwrap_or([0; 3]);
    let sign = |p: f32, s: f32, n: u32| -> f32 {
        if s > 0.0 {
            parity_sign_r::<f32>(axis_cell_r::<f32>(p, s, n).0 - mirror_base(n))
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
pub fn check_repeat_cells_tree(root: &crate::SdfNode) -> Result<(), RepeatCellsError> {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if let crate::SdfNode::RepeatCells { spacing, count, .. } = node {
            validate_repeat_cells(*spacing, *count)?;
        }
        stack.extend(crate::diff::node_children(node));
    }
    Ok(())
}
