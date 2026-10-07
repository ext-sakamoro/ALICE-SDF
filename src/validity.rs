//! Printability validation — `ValidityReport`.
//!
//! The question this module answers is *"can this shape be printed"*, and it
//! answers the two failure modes that actually lose geometry:
//!
//! * a wall thinner than the nozzle is not printed at all — it silently
//!   disappears from the part,
//! * an unsupported overhang droops, so the printed surface is not the exported
//!   surface.
//!
//! Each answer states **how** it was obtained, because "sampled and found
//! nothing" and "proved there is nothing" are different claims and only the
//! second one is a guarantee.
//!
//! ## Two independent thickness checks (and why one is not enough)
//!
//! **A — erosion, proved by interval arithmetic** ([`ErosionVerdict`])
//!
//! Erosion is exact for an SDF: `Round { radius: -t/2 }` evaluates to
//! `d(p) + t/2`, so only material deeper than `t/2` from the surface survives.
//! If the eroded shape is *empty*, every part of the shape is thinner than `t`.
//! The octree either finds a witness point inside the eroded shape
//! ([`ErosionVerdict::HasThickEnoughRegion`]), proves emptiness over the whole
//! bound ([`ErosionVerdict::EntirelyTooThin`]), or runs out of depth
//! ([`ErosionVerdict::Undecided`] — **not a pass**).
//!
//! This is a *global* statement. It catches "the whole part is a 0.3 mm sheet"
//! with a proof, and it says nothing about a thin fin on an otherwise solid
//! block.
//!
//! **A rejected design, recorded so it is not retried**: the first attempt was
//! "a box proven inside the shape *and* proven outside the eroded shape is a
//! wall thinner than `t`". That is wrong — it holds for **every** shape, because
//! the material within `t/2` of any surface always erodes away, however thick
//! the body behind it is. Local thickness is a medial-axis quantity and deciding
//! it needs connectivity analysis (flood fill), which is exactly what
//! `alice_lol::law` declined to make sound for `Continuity` because it is
//! grid-resolution dependent.
//!
//! **B — local thickness, measured exactly per triangle** ([`local_thickness`])
//!
//! From each triangle centroid, march inward along `-n` by sphere tracing: while
//! inside, `-d(p)` is the exact distance to the nearest surface, so stepping by
//! it never overshoots, and the distance at which the ray leaves the solid is
//! the local thickness at that point. Every individual measurement is exact
//! (it reads the SDF, it does not estimate); what is sampled is *where* the
//! measurements are taken, and that is the mesh tessellation — a density the
//! caller chose and can state. This is the approach mesh-repair tools take.
//!
//! Together: A proves the global case, B localises the thin regions with exact
//! per-point numbers, and [`ValidityReport::is_printable`] requires A to be a
//! proof (not `Undecided`) *and* B to find nothing below the requirement.
//!
//! ## Overhang is a closed form
//!
//! The overhang angle of a triangle is `asin(-n · b)` for unit face normal `n`
//! and unit build direction `b`: a vertical wall gives `0`, a downward-facing
//! horizontal ceiling gives `π/2`. Upward-facing triangles report `0` rather
//! than a negative angle. No sampling is involved.
//!
//! **What this does not decide**: whether an overhang is *supported*. A box on
//! the build plate has a horizontal downward face (overhang `π/2`) needing no
//! support. That requires the plate position and the slicer's support policy,
//! so [`overhang_stats`] reports the geometric angle and leaves the policy out.

use crate::eval::eval;
use crate::interval::{eval_interval, Vec3Interval};
use crate::mesh::{validate_mesh, Mesh, MeshValidation};
use crate::types::SdfNode;
use glam::Vec3;

/// Smallest march step, so a ray starting exactly on a surface still advances.
const MARCH_EPS: f32 = 1e-4;

/// What a part must satisfy to be printable by a given process.
#[derive(Debug, Clone, Copy)]
pub struct PrintRequirements {
    /// Minimum wall thickness, in the same unit as the SDF.
    ///
    /// For FDM this is usually two extrusion widths (a 0.4 mm nozzle → 0.8 mm).
    pub min_wall: f32,
    /// Largest overhang angle that prints without support \[rad\].
    ///
    /// The common FDM figure is 45° (`std::f32::consts::FRAC_PI_4`).
    pub max_overhang: f32,
    /// Unit build direction (the direction layers stack towards).
    pub build_direction: Vec3,
    /// Octree depth used by the erosion proof.
    ///
    /// Each level halves the box edge; 6 levels resolve a bound to 1/64 of its
    /// size. Raising it makes [`ErosionVerdict::Undecided`] rarer at a cost of
    /// up to `8^n` boxes.
    pub erosion_depth: u32,
}

impl PrintRequirements {
    /// FDM defaults: 0.8 mm walls, 45° overhang, +Z build direction, depth 6.
    ///
    /// The unit is millimetres, matching the 3 MF / STEP exporters.
    #[must_use]
    pub const fn fdm_0_4_nozzle() -> Self {
        Self {
            min_wall: 0.8,
            max_overhang: std::f32::consts::FRAC_PI_4,
            build_direction: Vec3::Z,
            erosion_depth: 6,
        }
    }
}

/// Global thickness verdict from the erosion proof (three-valued).
///
/// `Undecided` must never be treated as a pass — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErosionVerdict {
    /// A point was **proven** to lie inside the shape eroded by `min_wall / 2`,
    /// so some region is at least `min_wall` thick.
    HasThickEnoughRegion {
        /// Centre of the box that was proven to be inside the eroded shape.
        witness: Vec3,
    },
    /// The octree closed over the whole bound without finding surviving
    /// material: **every** part of the shape is thinner than `min_wall`.
    EntirelyTooThin,
    /// The depth limit was reached with boxes still straddling the eroded
    /// surface.
    Undecided,
}

impl ErosionVerdict {
    /// `true` only when a thick-enough region was proven to exist.
    #[must_use]
    pub const fn has_thick_region(&self) -> bool {
        matches!(self, Self::HasThickEnoughRegion { .. })
    }
}

/// Outcome of validating a shape for printing.
#[derive(Debug, Clone)]
pub struct ValidityReport {
    /// Mesh topology, from [`validate_mesh`].
    pub mesh: MeshValidation,
    /// A — global erosion proof (three-valued).
    pub erosion: ErosionVerdict,
    /// B — smallest local thickness measured over the triangles \[same unit as
    /// the SDF\], or `None` when no triangle could be measured.
    pub min_local_thickness: Option<f32>,
    /// B — number of triangles whose local thickness is below `min_wall`.
    pub thin_triangles: usize,
    /// Largest overhang angle over all triangles \[rad\].
    pub max_overhang: f32,
    /// Number of triangles whose overhang exceeds `max_overhang`.
    pub overhang_triangles: usize,
    /// The requirements this report was produced against.
    pub requirements: PrintRequirements,
}

impl ValidityReport {
    /// `true` only when every check passed **and** the global check is a proof.
    ///
    /// An [`ErosionVerdict::Undecided`] fails this deliberately: "the octree
    /// could not decide" is not "the part is fine".
    #[must_use]
    pub const fn is_printable(&self) -> bool {
        self.mesh.is_clean()
            && self.erosion.has_thick_region()
            && self.thin_triangles == 0
            && self.overhang_triangles == 0
    }
}

/// Largest overhang angle and the number of triangles over the threshold.
///
/// Returns `(max_angle_rad, over_threshold_count)`. Degenerate triangles (zero
/// area) are skipped because they have no meaningful normal. Upward-facing
/// triangles contribute `0`, not a negative angle.
#[must_use]
pub fn overhang_stats(mesh: &Mesh, build_direction: Vec3, threshold: f32) -> (f32, usize) {
    let dir = build_direction.normalize_or_zero();
    let mut max_angle = 0.0_f32;
    let mut over = 0_usize;
    for idx in mesh.indices.chunks_exact(3) {
        let Some(normal) = face_normal(mesh, idx) else {
            continue; // 退化三角形は法線を持たない
        };
        // 下向き成分 −n·b を [0, 1] に落として asin: 垂直壁 0、水平天井 π/2
        let downward = (-normal.dot(dir)).clamp(-1.0, 1.0);
        let angle = if downward <= 0.0 {
            0.0
        } else {
            downward.asin()
        };
        if angle > max_angle {
            max_angle = angle;
        }
        if angle > threshold {
            over += 1;
        }
    }
    (max_angle, over)
}

/// Unit geometric normal of the triangle, or `None` when it is degenerate.
fn face_normal(mesh: &Mesh, idx: &[u32]) -> Option<Vec3> {
    let p = |i: u32| mesh.vertices.get(i as usize).map(|v| v.position);
    let (a, b, c) = (p(idx[0])?, p(idx[1])?, p(idx[2])?);
    let n = (b - a).cross(c - a);
    let len = n.length();
    if len <= f32::EPSILON {
        return None;
    }
    Some(n / len)
}

/// Exact local thickness at `origin` when marching along `inward`.
///
/// `origin` is expected to sit on (or just outside) the surface and `inward`
/// to be a unit vector pointing into the solid. Returns the distance at which
/// the ray leaves the solid, or `None` when the ray does not enter the solid
/// or does not leave it within `max_march`.
///
/// Each step advances by `-d(p)`, the exact distance to the nearest surface
/// while inside, so the march never overshoots the far wall.
#[must_use]
pub fn local_thickness(node: &SdfNode, origin: Vec3, inward: Vec3, max_march: f32) -> Option<f32> {
    let dir = inward.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    // 1. 固体に入るまで進む (origin が表面ちょうど or 少し外にある前提)
    //    外では d(p) が最近表面までの距離なので、それだけ進んでも通り越さない
    let mut t = MARCH_EPS;
    loop {
        let d = eval(node, origin + dir * t);
        if d < 0.0 {
            break; // 固体に入った
        }
        t += d.max(MARCH_EPS);
        if t > max_march {
            return None; // 固体に入らなかった
        }
    }

    // 2. 固体を抜けるまで進む 内側では −d(p) が最近表面までの距離
    loop {
        let d = eval(node, origin + dir * t);
        if d >= 0.0 {
            return Some(t);
        }
        t += (-d).max(MARCH_EPS);
        if t > max_march {
            return None; // 抜けなかった (bound が足りない)
        }
    }
}

/// Decides whether **any** region of the shape survives erosion by
/// `min_wall / 2` (see [`ErosionVerdict`]).
///
/// `bmin` / `bmax` must enclose the shape; material outside the bound is not
/// examined.
#[must_use]
pub fn prove_erosion(
    node: &SdfNode,
    min_wall: f32,
    bmin: Vec3,
    bmax: Vec3,
    max_depth: u32,
) -> ErosionVerdict {
    // erode: `Round { radius: -t/2 }` は d(p) + t/2 を返すので、表面から t/2 より
    // 深い材料だけが負に残る (SDF の offset なので厳密)
    let eroded = node.clone().round(-min_wall * 0.5);

    let mut undecided = false;
    let mut stack: Vec<(Vec3, Vec3, u32)> = vec![(bmin, bmax, 0)];
    while let Some((lo, hi, depth)) = stack.pop() {
        let iv = eval_interval(&eroded, Vec3Interval::from_bounds(lo, hi));
        if iv.is_positive() {
            continue; // 箱は erode 後の外と証明された = 残った材料はない
        }
        if iv.is_negative() {
            // 箱が丸ごと erode 後の内部 = min_wall 以上の肉厚を持つ領域の証明
            return ErosionVerdict::HasThickEnoughRegion {
                witness: (lo + hi) * 0.5,
            };
        }
        if depth >= max_depth {
            undecided = true; // 表面を跨いだまま深さが尽きた
            continue;
        }
        let mid = (lo + hi) * 0.5;
        for i in 0..8_usize {
            let (x0, x1) = if i & 1 == 0 {
                (lo.x, mid.x)
            } else {
                (mid.x, hi.x)
            };
            let (y0, y1) = if i & 2 == 0 {
                (lo.y, mid.y)
            } else {
                (mid.y, hi.y)
            };
            let (z0, z1) = if i & 4 == 0 {
                (lo.z, mid.z)
            } else {
                (mid.z, hi.z)
            };
            stack.push((Vec3::new(x0, y0, z0), Vec3::new(x1, y1, z1), depth + 1));
        }
    }

    if undecided {
        ErosionVerdict::Undecided
    } else {
        // 全 box が「erode 後の外」と証明された = どこも min_wall より薄い
        ErosionVerdict::EntirelyTooThin
    }
}

/// Validates a shape and its tessellation against `req`.
#[must_use]
pub fn validate_for_printing(
    node: &SdfNode,
    mesh: &Mesh,
    bmin: Vec3,
    bmax: Vec3,
    req: PrintRequirements,
) -> ValidityReport {
    let erosion = prove_erosion(node, req.min_wall, bmin, bmax, req.erosion_depth);
    let (max_overhang, overhang_triangles) =
        overhang_stats(mesh, req.build_direction, req.max_overhang);

    // B — 三角形ごとに重心から内向き (−n) に march して局所厚さを測る
    // march 上限は bound の対角長 (それを超えるなら bound の外)
    let span = (bmax - bmin).length();
    let mut min_local: Option<f32> = None;
    let mut thin_triangles = 0_usize;
    for idx in mesh.indices.chunks_exact(3) {
        let Some(normal) = face_normal(mesh, idx) else {
            continue;
        };
        let p = |i: u32| mesh.vertices[i as usize].position;
        let centroid = (p(idx[0]) + p(idx[1]) + p(idx[2])) / 3.0;
        if let Some(thickness) = local_thickness(node, centroid, -normal, span) {
            min_local = Some(min_local.map_or(thickness, |m| m.min(thickness)));
            if thickness < req.min_wall {
                thin_triangles += 1;
            }
        }
    }

    ValidityReport {
        mesh: validate_mesh(mesh),
        erosion,
        min_local_thickness: min_local,
        thin_triangles,
        max_overhang,
        overhang_triangles,
        requirements: req,
    }
}

/// Failure from [`export_step_validated`].
#[derive(Debug)]
pub enum ValidatedExportError {
    /// Writing the file failed.
    Io(std::io::Error),
    /// The shape did not satisfy the requirements, so **nothing was written**.
    NotPrintable(Box<ValidityReport>),
}

impl std::fmt::Display for ValidatedExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "STEP export failed: {e}"),
            Self::NotPrintable(r) => write!(
                f,
                "not printable, nothing written (erosion {:?}, thin triangles {}, \
                 overhang triangles {}, min local thickness {:?})",
                r.erosion, r.thin_triangles, r.overhang_triangles, r.min_local_thickness
            ),
        }
    }
}

impl std::error::Error for ValidatedExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::NotPrintable(_) => None,
        }
    }
}

impl From<std::io::Error> for ValidatedExportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Validates the shape and writes STEP **only when it passes**.
///
/// [`crate::io::step::export_step`] does not validate anything, so a part headed
/// for a printer should go through here: when `req` is not met the file is *not*
/// written and the report comes back inside
/// [`ValidatedExportError::NotPrintable`], which is the difference between
/// "exported a shape that cannot be printed" and "refused with a reason".
///
/// The mesh is tessellated once for validation and `export_step` tessellates
/// again internally. Both use the same `bounds` / `resolution`, so the triangles
/// are identical (the evaluator is deterministic); the cost is one extra
/// marching-cubes pass. Call [`validate_for_printing`] and `export_step`
/// separately if that matters.
///
/// # Errors
///
/// - [`ValidatedExportError::NotPrintable`] when the report fails
///   [`ValidityReport::is_printable`] (no file is written)
/// - [`ValidatedExportError::Io`] when writing fails
pub fn export_step_validated(
    path: impl AsRef<std::path::Path>,
    node: &SdfNode,
    cfg: &crate::io::step::StepConfig,
    req: PrintRequirements,
) -> Result<ValidityReport, ValidatedExportError> {
    let (lo, hi) = cfg.bounds;
    let (bmin, bmax) = (Vec3::splat(lo), Vec3::splat(hi));
    let mc = crate::mesh::MarchingCubesConfig {
        resolution: usize::try_from(cfg.resolution).unwrap_or(usize::MAX),
        ..Default::default()
    };
    let mesh = crate::mesh::sdf_to_mesh(node, bmin, bmax, &mc);
    let report = validate_for_printing(node, &mesh, bmin, bmax, req);
    if !report.is_printable() {
        return Err(ValidatedExportError::NotPrintable(Box::new(report)));
    }
    crate::io::step::export_step(path, node, cfg)?;
    Ok(report)
}
