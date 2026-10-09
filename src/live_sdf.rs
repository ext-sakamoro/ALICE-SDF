//! One shape shared by the physics world and the renderer.
//!
//! [`LiveSdf`] is a handle to a single distance field: a base [`SdfNode`],
//! a list of shape edits (spherical craters, kept as `SdfNode` subtractions)
//! and a chain of time-dependent [`PhysicsModifier`]s (erosion, fracture,
//! ...). Every clone of the handle points at the same shape, so the same
//! object can be
//!
//! * a collider: `world.add_sdf_collider(SdfCollider::new_static(Box::new(live.clone()), ..))`
//!   ([`SdfField`] is implemented on the handle),
//! * a world participant: `world.add_participant(Box::new(live.clone()))`
//!   ([`Participant`] is implemented on the handle; each substep advances
//!   the modifiers, so stepping the world changes the collider's shape),
//! * the source of a chunked render mesh: [`LiveMesh::sync`] re-meshes only
//!   the chunks a change can reach, from the same distance function the
//!   collider answers with.
//!
//! ```text
//!              ┌──────────── LiveSdf (Arc<RwLock<..>>) ─────────────┐
//!  collider ──▶│ base SdfNode − craters (compiled) → modifier chain │◀── LiveMesh::sync
//!  participant▶│ substep: modifier.update(h)  → generation, dirty   │    (dirty chunks only)
//!              └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Distance
//!
//! `distance(p) = m_k(..m_1(p, eval_compiled(base − c_1 − .. − c_n, p))..)`:
//! the compiled base with every crater subtracted (`max(a, -sphere)`), then
//! each active modifier in registration order, as in
//! [`alice_physics::sim_modifier::ModifiedSdf`]. The normal is the central
//! difference of that distance with the step `max(1e-3, 1e-4·max|p_i|)`,
//! the formula `ModifiedSdf` uses.
//!
//! # Changes
//!
//! Each change of the shape raises the [`generation`](LiveSdf::generation)
//! by one and records the region it can touch ([`DirtyRegion`]).
//!
//! A region is a box with this meaning: a value the change altered, whose
//! old or new value lies in `[−δ, δ]`, is at a point within `δ` of the box,
//! for every `δ ≥ 0`. Far from the surface a change may alter values
//! anywhere (a crater raises `max(a, r − |p − c|)` deep inside the base,
//! far from the sphere), but a mesher only reads values near zero:
//!
//! * a crater records the sphere's bounding box: `r − |p − c| > a` with
//!   `a ≥ −δ` or a result `≤ δ` needs `|p − c| < r + δ`;
//! * a modifier records [`LiveModifier::influence`] before and after the
//!   update, or the whole space when the modifier cannot bound it. A
//!   modifier changes the shape when the bytes of
//!   [`LiveModifier::shape_state`] change; an update that leaves them equal
//!   records nothing.
//!
//! [`LiveMesh::sync`] turns regions into chunks under the assumption that
//! the shape has Lipschitz constant ≤ 1 near its surface (true for exact
//! SDFs and their CSG); for a base that stretches distances, use a smaller
//! cell or re-mesh everything.
//!
//! # Determinism
//!
//! Edits and modifier updates are plain f32 / Fix128 arithmetic in a fixed
//! order; meshing reads the shape under one read lock and each chunk is
//! meshed independently, so the same edit list and the same steps give
//! bit-identical distances and meshes on every run.
//!
//! # Snapshots
//!
//! As a participant the shape writes (payload version 1, little endian):
//! `version: u32`, the craters (`count: u64`, then `cx, cy, cz, r` as `f32`
//! bits), the modifiers (`count: u64`, then per modifier `kind: u32`,
//! `len: u64` and its own participant payload). The base node is not in the
//! payload: a snapshot restores into a `LiveSdf` built from the same base
//! with modifiers of the same kinds in the same order. A restore raises the
//! generation and records the whole space as changed.
//!
//! Author: Moroya Sakamoto

use crate::compiled::{eval_compiled, CompiledSdf};
use crate::mesh::optimize::remove_degenerate_triangles;
use crate::mesh::sdf_to_mesh::{
    interpolate_edge, CORNER_OFFSETS, EDGE_CONNECTIONS, EDGE_TABLE, TRI_TABLE,
};
use crate::mesh::{Mesh, Vertex};
use crate::types::SdfNode;
use alice_physics::collider::Contact;
use alice_physics::erosion::ErosionModifier;
use alice_physics::fracture::FractureModifier;
use alice_physics::phase_change::PhaseChangeModifier;
use alice_physics::pressure::PressureModifier;
use alice_physics::sdf_collider::SdfField;
use alice_physics::sdf_destruction::{destruction_from_impact, DestructionType};
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::thermal::ThermalModifier;
use alice_physics::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};
use alice_physics::{Fix128, Vec3Fix};
use glam::Vec3;
use rayon::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Payload version written by [`LiveSdf`]'s `write_state`.
const STATE_VERSION: u32 = 1;

/// Number of change records kept; a consumer further behind is told
/// [`DirtyRegion::Everywhere`].
const CHANGE_LOG_CAPACITY: usize = 4096;

/// Base step of the normal's central difference (the `ModifiedSdf` value).
const NORMAL_BASE_EPS: f32 = 1.0e-3;

// ============================================================================
// Regions and errors
// ============================================================================

/// The part of space a change of the shape can touch near its surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DirtyRegion {
    /// An axis-aligned box (inclusive corners): a value the change altered,
    /// whose old or new value lies in `[−δ, δ]`, is within `δ` of the box
    /// (module documentation, "Changes").
    Aabb {
        /// Lower corner.
        min: Vec3,
        /// Upper corner.
        max: Vec3,
    },
    /// The whole space.
    Everywhere,
}

/// Where a modifier can change the distance it is given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Influence {
    /// `modify_distance` returns its input everywhere.
    Nowhere,
    /// Near the surface the modifier acts only around this box: a value it
    /// changes, whose input or output lies in `[−δ, δ]`, is at a point
    /// within `δ` of the box, for every `δ ≥ 0`. (Deep inside or far
    /// outside it may change values anywhere; a mesher only reads values
    /// near zero, see [`LiveMesh::sync`].)
    Within {
        /// Lower corner.
        min: Vec3,
        /// Upper corner.
        max: Vec3,
    },
    /// No bound is known.
    Everywhere,
}

impl Influence {
    /// Smallest region covering both influences, `None` when neither changes
    /// anything.
    fn union(self, other: Self) -> Option<DirtyRegion> {
        match (self, other) {
            (Self::Everywhere, _) | (_, Self::Everywhere) => Some(DirtyRegion::Everywhere),
            (Self::Nowhere, Self::Nowhere) => None,
            (Self::Within { min, max }, Self::Nowhere)
            | (Self::Nowhere, Self::Within { min, max }) => Some(DirtyRegion::Aabb { min, max }),
            (Self::Within { min: a0, max: a1 }, Self::Within { min: b0, max: b1 }) => {
                Some(DirtyRegion::Aabb {
                    min: a0.min(b0),
                    max: a1.max(b1),
                })
            }
        }
    }
}

/// The changes of a shape after a given generation ([`LiveSdf::changes_since`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Changes {
    /// Generation of the shape when the changes were read.
    pub generation: u64,
    /// Regions changed after the asked generation, in order. Empty when the
    /// shape did not change.
    pub regions: Vec<DirtyRegion>,
}

/// Error of a [`LiveSdf`] / [`LiveMesh`] call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LiveSdfError {
    /// A crater with a non-finite centre or a radius that is not finite and
    /// positive.
    InvalidCrater,
    /// A mesh layout with a cell size that is not finite and positive, a
    /// chunk of 0 cells or an axis of 0 chunks, or a non-finite origin.
    InvalidMeshConfig,
    /// The shape has modifiers, which have no `SdfNode` form, so the GPU
    /// mesher cannot evaluate it.
    HasModifiers,
    /// The GPU mesher failed (message of its error).
    Gpu(String),
}

impl fmt::Display for LiveSdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCrater => write!(f, "crater centre must be finite, radius finite and > 0"),
            Self::InvalidMeshConfig => write!(
                f,
                "mesh layout needs a finite origin, a finite cell size > 0 and at least one cell and chunk per axis"
            ),
            Self::HasModifiers => write!(f, "the shape has modifiers, which have no SdfNode form"),
            Self::Gpu(e) => write!(f, "GPU marching cubes failed: {e}"),
        }
    }
}

impl std::error::Error for LiveSdfError {}

// ============================================================================
// LiveModifier
// ============================================================================

/// A [`PhysicsModifier`] that can live inside a [`LiveSdf`].
///
/// It is also a [`Participant`] (its payload is carried in the shape's
/// snapshot) and says which of its state decides the shape and where it
/// acts.
///
/// Implemented for the five modifiers of `alice-physics` that are world
/// participants: thermal, phase change, pressure, erosion and fracture.
pub trait LiveModifier: PhysicsModifier + Participant {
    /// Append the bytes that decide [`PhysicsModifier::modify_distance`].
    /// The shape counts an update as a change exactly when these bytes
    /// change. Default: the whole participant payload.
    fn shape_state(&self, out: &mut Vec<u8>) {
        self.write_state(out);
    }

    /// Where `modify_distance` can change values near the surface (see
    /// [`Influence::Within`]). Default: [`Influence::Everywhere`].
    fn influence(&self) -> Influence {
        Influence::Everywhere
    }
}

impl LiveModifier for ThermalModifier {}
impl LiveModifier for PhaseChangeModifier {}
impl LiveModifier for PressureModifier {}

impl LiveModifier for ErosionModifier {
    /// `enabled` and the erosion depth field: `modify_distance` adds the
    /// sampled depth, the exposure only drives `update`.
    fn shape_state(&self, out: &mut Vec<u8>) {
        out.push(u8::from(self.enabled));
        let f = &self.erosion_depth;
        for v in [f.nx, f.ny, f.nz] {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
        for v in [f.min.0, f.min.1, f.min.2, f.max.0, f.max.1, f.max.2] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for v in &f.data {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
}

impl LiveModifier for FractureModifier {
    /// `enabled`, the crack width and each crack's segment and length: the
    /// stress field only seeds and grows cracks.
    fn shape_state(&self, out: &mut Vec<u8>) {
        out.push(u8::from(self.enabled));
        out.extend_from_slice(&self.config.crack_width.to_bits().to_le_bytes());
        out.extend_from_slice(&(self.cracks.len() as u64).to_le_bytes());
        for c in &self.cracks {
            for v in [
                c.start.0, c.start.1, c.start.2, c.end.0, c.end.1, c.end.2, c.length,
            ] {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }

    /// The union of the crack segments' boxes widened by the crack width.
    /// A crack replaces `d` by `max(d, width − |p − segment|)`; a changed
    /// value with `d ≥ −δ` or a result `≤ δ` needs
    /// `|p − segment| < width + δ`, i.e. a point within `δ` of the widened
    /// box. Cracks shorter than `1e-5` are skipped by `modify_distance` and
    /// are skipped here.
    fn influence(&self) -> Influence {
        if !self.enabled {
            return Influence::Nowhere;
        }
        let w = self.config.crack_width;
        let mut bound: Option<(Vec3, Vec3)> = None;
        for c in &self.cracks {
            if c.length < 1e-5 {
                continue;
            }
            let a = Vec3::new(c.start.0, c.start.1, c.start.2);
            let b = Vec3::new(c.end.0, c.end.1, c.end.2);
            let lo = a.min(b) - Vec3::splat(w);
            let hi = a.max(b) + Vec3::splat(w);
            bound = Some(match bound {
                Some((l, h)) => (l.min(lo), h.max(hi)),
                None => (lo, hi),
            });
        }
        match bound {
            Some((min, max)) => Influence::Within { min, max },
            None => Influence::Nowhere,
        }
    }
}

/// A [`LiveModifier`] that can be downcast to its concrete type.
trait ErasedModifier: LiveModifier {
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: LiveModifier + 'static> ErasedModifier for T {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ============================================================================
// Shared state
// ============================================================================

/// A spherical crater subtracted from the base.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Crater {
    center: Vec3,
    radius: f32,
}

impl Crater {
    fn node(self) -> SdfNode {
        SdfNode::sphere(self.radius).translate(self.center.x, self.center.y, self.center.z)
    }

    fn is_valid(self) -> bool {
        self.center.is_finite() && self.radius.is_finite() && self.radius > 0.0
    }
}

struct LiveState {
    base: SdfNode,
    craters: Vec<Crater>,
    /// `base − craters`, compiled.
    compiled: CompiledSdf,
    modifiers: Vec<Box<dyn ErasedModifier>>,
    generation: u64,
    /// `(generation, region)` of the most recent changes, oldest first.
    log: Vec<(u64, DirtyRegion)>,
    /// Highest generation whose record was dropped from `log` (0: none).
    dropped_through: u64,
}

impl LiveState {
    fn edited_node(&self) -> SdfNode {
        let mut node = self.base.clone();
        for c in &self.craters {
            node = node.subtract(c.node());
        }
        node
    }

    fn recompile(&mut self) {
        self.compiled = CompiledSdf::compile(&self.edited_node());
    }

    fn record(&mut self, region: DirtyRegion) {
        self.generation += 1;
        if self.log.len() == CHANGE_LOG_CAPACITY {
            let (g, _) = self.log.remove(0);
            self.dropped_through = g;
        }
        self.log.push((self.generation, region));
    }

    #[inline]
    fn distance(&self, p: Vec3) -> f32 {
        let mut d = eval_compiled(&self.compiled, p);
        for m in &self.modifiers {
            if m.is_active() {
                d = m.modify_distance(p.x, p.y, p.z, d);
            }
        }
        d
    }

    fn normal(&self, p: Vec3) -> Vec3 {
        let scale = p.x.abs().max(p.y.abs()).max(p.z.abs());
        let e = NORMAL_BASE_EPS.max(1.0e-4 * scale);
        let ex = Vec3::new(e, 0.0, 0.0);
        let ey = Vec3::new(0.0, e, 0.0);
        let ez = Vec3::new(0.0, 0.0, e);
        let dx = self.distance(p + ex) - self.distance(p - ex);
        let dy = self.distance(p + ey) - self.distance(p - ey);
        let dz = self.distance(p + ez) - self.distance(p - ez);
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 1e-10 {
            Vec3::Y
        } else {
            Vec3::new(dx / len, dy / len, dz / len)
        }
    }

    /// Update one modifier with `f` and record the change it made.
    fn update_modifier(&mut self, index: usize, f: impl FnOnce(&mut dyn ErasedModifier)) {
        let m = &mut self.modifiers[index];
        let mut before = Vec::new();
        m.shape_state(&mut before);
        let influence_before = m.influence();
        f(m.as_mut());
        let mut after = Vec::new();
        m.shape_state(&mut after);
        if before != after {
            if let Some(region) = influence_before.union(m.influence()) {
                self.record(region);
            }
        }
    }
}

// ============================================================================
// LiveSdf
// ============================================================================

/// A shared, editable distance field: clone the handle to give the same
/// shape to a physics collider, a world participant and a [`LiveMesh`].
/// See the [module documentation](self).
#[derive(Clone)]
pub struct LiveSdf {
    inner: Arc<RwLock<LiveState>>,
}

impl fmt::Debug for LiveSdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.read();
        f.debug_struct("LiveSdf")
            .field("craters", &s.craters.len())
            .field("modifiers", &s.modifiers.len())
            .field("generation", &s.generation)
            .finish()
    }
}

impl LiveSdf {
    /// Snapshot tag of the shape as a world participant (`"\0LSD"` read big
    /// endian, below the `0x0100_0000` reserved for user code). Never changes.
    pub const PARTICIPANT_KIND: ParticipantKind =
        ParticipantKind::new(u32::from_be_bytes(*b"\0LSD"));

    /// A shape with `base` and no edits, at generation 0.
    pub fn new(base: SdfNode) -> Self {
        let compiled = CompiledSdf::compile(&base);
        Self {
            inner: Arc::new(RwLock::new(LiveState {
                base,
                craters: Vec::new(),
                compiled,
                modifiers: Vec::new(),
                generation: 0,
                log: Vec::new(),
                dropped_through: 0,
            })),
        }
    }

    fn read(&self) -> RwLockReadGuard<'_, LiveState> {
        // A panic while the lock was held leaves the last fully assigned
        // state: every writer builds the new compiled field before storing it.
        self.inner.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, LiveState> {
        self.inner.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether two handles point at the same shape.
    pub fn same_shape(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Signed distance at `p` (the value the collider answers with).
    pub fn eval(&self, p: Vec3) -> f32 {
        self.read().distance(p)
    }

    /// Outward normal at `p` (central difference of [`Self::eval`]).
    pub fn eval_normal(&self, p: Vec3) -> Vec3 {
        self.read().normal(p)
    }

    /// Subtract a sphere (a crater) and return the new generation.
    ///
    /// The crater is kept as an `SdfNode` subtraction, so a shape without
    /// modifiers stays expressible as one node ([`Self::to_sdf_node`]).
    pub fn subtract_sphere(&self, center: Vec3, radius: f32) -> Result<u64, LiveSdfError> {
        let crater = Crater { center, radius };
        if !crater.is_valid() {
            return Err(LiveSdfError::InvalidCrater);
        }
        let mut s = self.write();
        let mut craters = s.craters.clone();
        craters.push(crater);
        let node = s.base.clone();
        let mut edited = node;
        for c in &craters {
            edited = edited.subtract(c.node());
        }
        let compiled = CompiledSdf::compile(&edited);
        s.craters = craters;
        s.compiled = compiled;
        let r = Vec3::splat(radius);
        s.record(DirtyRegion::Aabb {
            min: center - r,
            max: center + r,
        });
        Ok(s.generation)
    }

    /// Append a modifier to the chain and return its index. Records the
    /// modifier's influence as changed.
    pub fn add_modifier<M: LiveModifier + 'static>(&self, modifier: M) -> usize {
        let mut s = self.write();
        let region = Influence::Nowhere.union(modifier.influence());
        s.modifiers.push(Box::new(modifier));
        if let Some(region) = region {
            s.record(region);
        }
        s.modifiers.len() - 1
    }

    /// Run `f` on modifier `index` as its concrete type `M` (for example
    /// `FractureModifier::apply_stress_at` or
    /// `ErosionModifier::set_exposure_at`) and record the change it made to
    /// the shape. `None` when there is no such modifier or it is not an `M`.
    pub fn with_modifier_mut<M: LiveModifier + 'static, R>(
        &self,
        index: usize,
        f: impl FnOnce(&mut M) -> R,
    ) -> Option<R> {
        let mut s = self.write();
        let is_m = s
            .modifiers
            .get_mut(index)
            .is_some_and(|m| m.as_any_mut().is::<M>());
        if !is_m {
            return None;
        }
        let mut out = None;
        s.update_modifier(index, |m| {
            out = m.as_any_mut().downcast_mut::<M>().map(f);
        });
        drop(s);
        out
    }

    /// Advance every modifier by `dt` seconds outside a world (the same
    /// update a world substep of width `dt` makes), recording the changes.
    pub fn update(&self, dt: f32) {
        let mut s = self.write();
        for i in 0..s.modifiers.len() {
            s.update_modifier(i, |m| m.update(dt));
        }
    }

    /// Current generation: the number of shape changes so far (plus one per
    /// snapshot restore).
    pub fn generation(&self) -> u64 {
        self.read().generation
    }

    /// Regions changed after `generation`. A consumer whose generation is
    /// older than the kept records gets [`DirtyRegion::Everywhere`].
    pub fn changes_since(&self, generation: u64) -> Changes {
        changes_since(&self.read(), generation)
    }

    /// Number of craters.
    pub fn crater_count(&self) -> usize {
        self.read().craters.len()
    }

    /// Number of modifiers.
    pub fn modifier_count(&self) -> usize {
        self.read().modifiers.len()
    }

    /// The shape as one `SdfNode` (base minus every crater), or `None` when
    /// it has modifiers, which have no node form.
    pub fn to_sdf_node(&self) -> Option<SdfNode> {
        let s = self.read();
        if s.modifiers.is_empty() {
            Some(s.edited_node())
        } else {
            None
        }
    }

    /// Mesh the shape on the GPU ([`crate::mesh::gpu_marching_cubes()`] of
    /// [`Self::to_sdf_node`]). Fails with [`LiveSdfError::HasModifiers`] when
    /// the shape has modifiers.
    #[cfg(feature = "gpu")]
    pub fn gpu_mesh(
        &self,
        min: Vec3,
        max: Vec3,
        config: &crate::mesh::GpuMarchingCubesConfig,
    ) -> Result<Mesh, LiveSdfError> {
        let node = self.to_sdf_node().ok_or(LiveSdfError::HasModifiers)?;
        crate::mesh::gpu_marching_cubes(&node, min, max, config)
            .map_err(|e| LiveSdfError::Gpu(e.to_string()))
    }

    /// Carve a crater for every contact `policy` accepts and return how many
    /// were carved. See [`FracturePolicy`].
    pub fn apply_impacts(&self, policy: &FracturePolicy, contacts: &[ImpactContact]) -> usize {
        let mut n = 0;
        for c in contacts {
            if let Some((center, radius)) = policy.crater_for(c) {
                if self.subtract_sphere(center, radius).is_ok() {
                    n += 1;
                }
            }
        }
        n
    }

    /// Carve a crater for every body whose fastest contact in the last world
    /// step against the SDF collider `collider_index` is accepted by `policy`,
    /// and return how many were carved (at most one per body per step: the
    /// records of several substeps of one touch are one impact)
    ///
    /// The records of [`alice_physics::PhysicsWorld::last_step_sdf_contacts`]
    /// are in world space; each is converted into this shape's frame with the
    /// collider's pose ([`ImpactContact::from_world`]) before the policy sees
    /// it, so the policy's speeds and radii are in the shape's units. The
    /// world wakes resting bodies on the next step because the shape's
    /// [`SdfField::generation`] changed. Returns 0 when `collider_index` is out
    /// of range.
    pub fn apply_world_contacts(
        &self,
        policy: &FracturePolicy,
        world: &alice_physics::PhysicsWorld,
        collider_index: usize,
    ) -> usize {
        let Some(collider) = world.sdf_colliders.get(collider_index) else {
            return 0;
        };
        // One impact per body per step: a body touching for several substeps
        // leaves a record per substep, and they are one hit, so only the
        // fastest record of each body is used (ties: the earliest).
        let mut fastest: Vec<&alice_physics::sdf_collider::SdfContact> = Vec::new();
        for c in world
            .last_step_sdf_contacts()
            .iter()
            .filter(|c| c.collider_index == collider_index)
        {
            match fastest.iter_mut().find(|f| f.body_index == c.body_index) {
                Some(f) => {
                    if c.approach_speed > f.approach_speed {
                        *f = c;
                    }
                }
                None => fastest.push(c),
            }
        }
        let contacts: Vec<ImpactContact> = fastest
            .into_iter()
            .map(|c| ImpactContact::from_world(c, collider))
            .collect();
        self.apply_impacts(policy, &contacts)
    }
}

fn changes_since(s: &LiveState, generation: u64) -> Changes {
    let regions = if generation >= s.generation {
        Vec::new()
    } else if generation < s.dropped_through {
        vec![DirtyRegion::Everywhere]
    } else {
        s.log
            .iter()
            .filter(|(g, _)| *g > generation)
            .map(|(_, r)| *r)
            .collect()
    };
    Changes {
        generation: s.generation,
        regions,
    }
}

impl SdfField for LiveSdf {
    #[inline]
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.eval(Vec3::new(x, y, z))
    }

    /// The shape's [`LiveSdf::generation`], so the world wakes resting bodies
    /// when a crater or a modifier changes the shape (alice-physics 2.1)
    #[inline]
    fn generation(&self) -> u64 {
        Self::generation(self)
    }

    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let n = self.eval_normal(Vec3::new(x, y, z));
        (n.x, n.y, n.z)
    }

    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) {
        let s = self.read();
        let p = Vec3::new(x, y, z);
        let n = s.normal(p);
        (s.distance(p), (n.x, n.y, n.z))
    }
}

// ============================================================================
// Participant
// ============================================================================

/// Little-endian reader of the snapshot payload.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], StateError> {
        let end = self.pos.checked_add(n).ok_or(StateError::InvalidValue)?;
        if end > self.bytes.len() {
            return Err(StateError::Length {
                expected: end,
                found: self.bytes.len(),
            });
        }
        let head = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(head)
    }

    const fn finish(&self) -> Result<(), StateError> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(StateError::Length {
                expected: self.pos,
                found: self.bytes.len(),
            })
        }
    }

    fn u32(&mut self) -> Result<u32, StateError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, StateError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn f32(&mut self) -> Result<f32, StateError> {
        Ok(f32::from_bits(self.u32()?))
    }
}

/// The decoded payload: craters and each modifier's `(kind, bytes)`.
type DecodedState<'a> = (Vec<Crater>, Vec<(u32, &'a [u8])>);

fn decode_state(bytes: &[u8]) -> Result<DecodedState<'_>, StateError> {
    let mut r = Reader::new(bytes);
    if r.u32()? != STATE_VERSION {
        return Err(StateError::InvalidValue);
    }
    let n = r.u64()?;
    let mut craters = Vec::new();
    for _ in 0..n {
        let c = Crater {
            center: Vec3::new(r.f32()?, r.f32()?, r.f32()?),
            radius: r.f32()?,
        };
        if !c.is_valid() {
            return Err(StateError::InvalidValue);
        }
        craters.push(c);
    }
    let m = r.u64()?;
    let mut mods = Vec::new();
    for _ in 0..m {
        let kind = r.u32()?;
        let len = usize::try_from(r.u64()?).map_err(|_| StateError::InvalidValue)?;
        mods.push((kind, r.take(len)?));
    }
    r.finish()?;
    Ok((craters, mods))
}

/// One world substep advances every modifier by `update(h)` (`h` converted
/// with [`Fix128::to_f32`], the rule of the modifiers' own participant
/// impls), so the shape a collider sees follows world time.
///
/// Observations: channel 0 the number of craters, channel 1 the number of
/// modifiers. The participant stages no force.
impl Participant for LiveSdf {
    fn kind(&self) -> ParticipantKind {
        Self::PARTICIPANT_KIND
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        let dt = h.to_f32();
        let mut s = self.write();
        for i in 0..s.modifiers.len() {
            s.update_modifier(i, |m| PhysicsModifier::update(m, dt));
        }
        drop(s);
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        let s = self.read();
        out.push(0, Fix128::from_int(s.craters.len() as i64));
        out.push(1, Fix128::from_int(s.modifiers.len() as i64));
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        let s = self.read();
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&(s.craters.len() as u64).to_le_bytes());
        for c in &s.craters {
            for v in [c.center.x, c.center.y, c.center.z, c.radius] {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        out.extend_from_slice(&(s.modifiers.len() as u64).to_le_bytes());
        for m in &s.modifiers {
            let mut payload = Vec::new();
            m.write_state(&mut payload);
            out.extend_from_slice(&m.kind().get().to_le_bytes());
            out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            out.extend_from_slice(&payload);
        }
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        let (_, mods) = decode_state(bytes)?;
        let s = self.read();
        if mods.len() != s.modifiers.len() {
            return Err(StateError::InvalidValue);
        }
        for (m, (kind, payload)) in s.modifiers.iter().zip(&mods) {
            if m.kind().get() != *kind {
                return Err(StateError::InvalidValue);
            }
            m.check_state(payload)?;
        }
        drop(s);
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        let (craters, mods) = match decode_state(bytes) {
            Ok(d) => d,
            Err(e) => panic!("read_state called with a payload check_state refuses: {e:?}"),
        };
        let mut s = self.write();
        s.craters = craters;
        s.recompile();
        for (m, (_, payload)) in s.modifiers.iter_mut().zip(&mods) {
            m.read_state(payload);
        }
        s.record(DirtyRegion::Everywhere);
    }
}

/// Wake every sleeping body of `world` whose position lies in one of
/// `regions` widened by `margin` (a body's collision radius, say), and
/// return how many were woken.
///
/// Not needed for a [`LiveSdf`] collider since alice-physics 2.1: the shape
/// reports its [`SdfField::generation`], and the world wakes its resting
/// bodies on the step after a change. Kept for callers that wake only the
/// bodies near a change. The regions are in the shape's own frame, the
/// world frame for a collider placed at the origin without rotation or
/// scale.
#[deprecated(
    since = "5.1.0",
    note = "the world wakes resting bodies when a LiveSdf collider changes (alice-physics 2.1)"
)]
pub fn wake_bodies_in(
    world: &mut alice_physics::PhysicsWorld,
    regions: &[DirtyRegion],
    margin: f32,
) -> usize {
    let m = Vec3::splat(margin);
    let mut woken = 0;
    for i in 0..world.bodies.len() {
        if !world.is_sleeping(i) {
            continue;
        }
        let (x, y, z) = world.bodies[i].position.to_f32();
        let p = Vec3::new(x, y, z);
        let inside = regions.iter().any(|r| match *r {
            DirtyRegion::Everywhere => true,
            DirtyRegion::Aabb { min, max } => (min - m).cmple(p).all() && p.cmple(max + m).all(),
        });
        if inside {
            world.wake_body(i);
            woken += 1;
        }
    }
    woken
}

// ============================================================================
// Impacts
// ============================================================================

/// One contact between a body and an SDF collider, the input of
/// [`FracturePolicy`].
///
/// The fields are those of the per-step SDF contact
/// record planned for `alice-physics` 2.1 (`body_index`, `collider_index`,
/// `point`, `normal`, `depth`, `approach_speed`, `substep`), so a record of
/// that type maps onto this one field by field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImpactContact {
    /// Index of the body in the world.
    pub body_index: usize,
    /// Index of the SDF collider in the world.
    pub collider_index: usize,
    /// Contact point on the SDF surface, in the shape's own frame (the world
    /// frame for a collider placed at the origin without rotation or scale).
    pub point: Vec3Fix,
    /// Contact normal (out of the SDF).
    pub normal: Vec3Fix,
    /// Penetration depth.
    pub depth: Fix128,
    /// Speed of approach along the normal (positive when closing).
    pub approach_speed: Fix128,
    /// Substep of the step in which the contact was recorded.
    pub substep: u32,
}

impl ImpactContact {
    /// A world contact record converted into the frame of the shape that
    /// `collider` places in the world
    ///
    /// The point is moved by the inverse of the collider's pose (translation,
    /// rotation, uniform scale, the transform the collider evaluates its field
    /// through), the normal by the inverse rotation, and the depth and the
    /// approach speed are divided by the scale so they are lengths and speeds
    /// in the shape's units.
    pub fn from_world(
        contact: &alice_physics::sdf_collider::SdfContact,
        collider: &alice_physics::sdf_collider::SdfCollider,
    ) -> Self {
        let inv_rotation = collider.rotation.conjugate();
        let local = inv_rotation.rotate_vec(contact.point - collider.position);
        let normal = inv_rotation.rotate_vec(contact.normal);
        let scale = collider.scale;
        let (point, depth, approach_speed) = if scale == Fix128::ONE || scale.is_zero() {
            (local, contact.depth, contact.approach_speed)
        } else {
            (
                Vec3Fix::new(local.x / scale, local.y / scale, local.z / scale),
                contact.depth / scale,
                contact.approach_speed / scale,
            )
        };
        Self {
            body_index: contact.body_index,
            collider_index: contact.collider_index,
            point,
            normal,
            depth,
            approach_speed,
            substep: u32::try_from(contact.substep).unwrap_or(u32::MAX),
        }
    }
}

/// Which contacts carve a crater, and how large.
///
/// A contact carves when `approach_speed > speed_threshold` (and, when
/// `collider_index` is set, it touches that collider). The crater is the
/// sphere `alice_physics::sdf_destruction::destruction_from_impact` gives
/// for the contact: centre the contact point, radius
/// `clamp(|approach_speed| · velocity_to_radius_scale, min_radius, max_radius)`
/// (the bounds in either order). The function is called, not copied, so the
/// rule is the physics crate's by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FracturePolicy {
    /// Only contacts with this collider carve (`None`: every contact).
    pub collider_index: Option<usize>,
    /// Contacts approaching at this speed or slower do not carve.
    pub speed_threshold: Fix128,
    /// Crater radius per unit of approach speed.
    pub velocity_to_radius_scale: f32,
    /// Smallest crater radius.
    pub min_radius: f32,
    /// Largest crater radius.
    pub max_radius: f32,
}

impl FracturePolicy {
    /// A policy for every collider.
    pub const fn new(
        speed_threshold: Fix128,
        velocity_to_radius_scale: f32,
        min_radius: f32,
        max_radius: f32,
    ) -> Self {
        Self {
            collider_index: None,
            speed_threshold,
            velocity_to_radius_scale,
            min_radius,
            max_radius,
        }
    }

    /// The same policy restricted to one collider.
    #[must_use]
    pub const fn for_collider(mut self, collider_index: usize) -> Self {
        self.collider_index = Some(collider_index);
        self
    }

    /// The crater `(centre, radius)` this policy carves for `contact`, or
    /// `None` when the contact does not carve.
    pub fn crater_for(&self, contact: &ImpactContact) -> Option<(Vec3, f32)> {
        if let Some(i) = self.collider_index {
            if contact.collider_index != i {
                return None;
            }
        }
        if contact.approach_speed <= self.speed_threshold {
            return None;
        }
        let physics_contact = Contact {
            depth: contact.depth,
            normal: contact.normal,
            point_a: contact.point,
            point_b: contact.point,
        };
        let shape = destruction_from_impact(
            &physics_contact,
            contact.approach_speed,
            self.velocity_to_radius_scale,
            self.min_radius,
            self.max_radius,
        );
        let DestructionType::Sphere { radius } = shape.shape else {
            return None;
        };
        let (x, y, z) = shape.center.to_f32();
        Some((Vec3::new(x, y, z), radius))
    }
}

// ============================================================================
// LiveMesh
// ============================================================================

/// Layout of a [`LiveMesh`]: a lattice of cubic cells split into chunks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveMeshConfig {
    /// Lower corner of chunk `(0, 0, 0)` (a lattice point).
    pub origin: Vec3,
    /// Edge length of a cell.
    pub cell_size: f32,
    /// Cells per chunk along each axis.
    pub chunk_cells: usize,
    /// Chunks along x, y, z.
    pub chunks: [usize; 3],
}

impl LiveMeshConfig {
    fn is_valid(&self) -> bool {
        self.origin.is_finite()
            && self.cell_size.is_finite()
            && self.cell_size > 0.0
            && self.chunk_cells > 0
            && self.chunks.iter().all(|&n| n > 0)
    }

    /// Position of the global lattice point `g`: `origin + g · cell_size`
    /// per axis. Every chunk computes a shared point with this one formula,
    /// so the point (and the values sampled there) are bit-identical on
    /// both sides of a chunk border.
    #[inline]
    fn lattice_point(&self, g: [i64; 3]) -> Vec3 {
        Vec3::new(
            self.origin.x + g[0] as f32 * self.cell_size,
            self.origin.y + g[1] as f32 * self.cell_size,
            self.origin.z + g[2] as f32 * self.cell_size,
        )
    }

    /// Bounds of chunk `c`.
    pub fn chunk_bounds(&self, c: [usize; 3]) -> (Vec3, Vec3) {
        let n = self.chunk_cells as i64;
        let lo = [c[0] as i64 * n, c[1] as i64 * n, c[2] as i64 * n];
        let hi = [lo[0] + n, lo[1] + n, lo[2] + n];
        (self.lattice_point(lo), self.lattice_point(hi))
    }

    /// Lower and upper corner of the whole lattice.
    pub fn domain(&self) -> (Vec3, Vec3) {
        let n = self.chunk_cells as i64;
        let hi = [
            self.chunks[0] as i64 * n,
            self.chunks[1] as i64 * n,
            self.chunks[2] as i64 * n,
        ];
        (self.origin, self.lattice_point(hi))
    }

    const fn chunk_index(&self, c: [usize; 3]) -> usize {
        (c[2] * self.chunks[1] + c[1]) * self.chunks[0] + c[0]
    }

    fn all_chunks(&self) -> Vec<[usize; 3]> {
        let mut out = Vec::with_capacity(self.chunks.iter().product());
        for z in 0..self.chunks[2] {
            for y in 0..self.chunks[1] {
                for x in 0..self.chunks[0] {
                    out.push([x, y, z]);
                }
            }
        }
        out
    }

    /// Chunks whose bounds meet `[min, max]` widened by `margin`.
    fn chunks_touching(&self, min: Vec3, max: Vec3, margin: f32) -> Vec<[usize; 3]> {
        let len = self.chunk_cells as f32 * self.cell_size;
        let lo = (min - Vec3::splat(margin) - self.origin) / len;
        let hi = (max + Vec3::splat(margin) - self.origin) / len;
        let mut range = [(0usize, 0usize); 3];
        for (axis, r) in range.iter_mut().enumerate() {
            let (l, h) = (lo[axis].floor(), hi[axis].floor());
            let count = self.chunks[axis] as f32;
            if h < 0.0 || l >= count || !(l.is_finite() && h.is_finite()) {
                return Vec::new();
            }
            *r = (l.max(0.0) as usize, h.min(count - 1.0) as usize);
        }
        let mut out = Vec::new();
        for z in range[2].0..=range[2].1 {
            for y in range[1].0..=range[1].1 {
                for x in range[0].0..=range[0].1 {
                    out.push([x, y, z]);
                }
            }
        }
        out
    }
}

/// What one [`LiveMesh::sync`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncReport {
    /// Generation of the shape the mesh now shows.
    pub generation: u64,
    /// Chunks that were re-meshed, in `(z, y, x)` order.
    pub remeshed: Vec<[usize; 3]>,
}

/// A render mesh of a [`LiveSdf`] in chunks, kept up to date by
/// [`Self::sync`].
///
/// Each chunk is a marching-cubes mesh of the shape's own distance (base,
/// craters and modifiers: the value the collider answers with) on the
/// global lattice `origin + g · cell_size`. Vertices sit on lattice edges at
/// the crate's canonical interpolation (lexicographically smaller endpoint
/// first), so a vertex on a chunk border is the same bits in both chunks
/// and [`Self::merged`] closes the seams exactly. Normals are the
/// central-difference gradients at the two edge endpoints blended with the
/// same edge parameter, read from a one-cell padding around the chunk.
#[derive(Debug)]
pub struct LiveMesh {
    sdf: LiveSdf,
    config: LiveMeshConfig,
    meshes: Vec<Arc<Mesh>>,
    generation: u64,
}

impl LiveMesh {
    /// Mesh every chunk of `sdf` on the layout `config`.
    pub fn new(sdf: &LiveSdf, config: LiveMeshConfig) -> Result<Self, LiveSdfError> {
        if !config.is_valid() {
            return Err(LiveSdfError::InvalidMeshConfig);
        }
        let all = config.all_chunks();
        let state = sdf.read();
        let generation = state.generation;
        let meshes = all
            .par_iter()
            .map(|&c| Arc::new(mesh_chunk(&state, &config, c)))
            .collect();
        drop(state);
        Ok(Self {
            sdf: sdf.clone(),
            config,
            meshes,
            generation,
        })
    }

    /// Re-mesh the chunks the shape's changes since the last sync can reach
    /// and keep every other chunk's mesh as it was.
    ///
    /// A chunk's mesh reads the values at the endpoints of its sign-change
    /// edges (`|d| ≤ h` for a field with Lipschitz constant ≤ 1 on a lattice
    /// of spacing `h`) and at their finite-difference neighbours
    /// (`|d| ≤ 2h`), all within one cell of the chunk. With `δ = 2h` in the
    /// meaning of [`DirtyRegion`], every changed value it reads lies within
    /// `2h` of a region, so a chunk is re-meshed when its bounds meet a
    /// region widened by `3h`.
    pub fn sync(&mut self) -> SyncReport {
        let state = self.sdf.read();
        let changes = changes_since(&state, self.generation);
        let margin = 3.0 * self.config.cell_size;
        let mut touched = vec![false; self.meshes.len()];
        for region in &changes.regions {
            let list = match *region {
                DirtyRegion::Everywhere => self.config.all_chunks(),
                DirtyRegion::Aabb { min, max } => self.config.chunks_touching(min, max, margin),
            };
            for c in list {
                touched[self.config.chunk_index(c)] = true;
            }
        }
        let remeshed: Vec<[usize; 3]> = self
            .config
            .all_chunks()
            .into_iter()
            .filter(|&c| touched[self.config.chunk_index(c)])
            .collect();
        let config = self.config;
        let fresh: Vec<Mesh> = remeshed
            .par_iter()
            .map(|&c| mesh_chunk(&state, &config, c))
            .collect();
        drop(state);
        for (c, mesh) in remeshed.iter().zip(fresh) {
            let i = self.config.chunk_index(*c);
            self.meshes[i] = Arc::new(mesh);
        }
        self.generation = changes.generation;
        SyncReport {
            generation: changes.generation,
            remeshed,
        }
    }

    /// Generation of the shape the mesh shows.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// The layout.
    pub const fn config(&self) -> &LiveMeshConfig {
        &self.config
    }

    /// Mesh of chunk `c`, `None` outside the layout.
    pub fn chunk(&self, c: [usize; 3]) -> Option<&Arc<Mesh>> {
        if c.iter().zip(&self.config.chunks).any(|(&a, &n)| a >= n) {
            return None;
        }
        self.meshes.get(self.config.chunk_index(c))
    }

    /// Every chunk mesh joined into one, vertices with identical position
    /// bits merged (the chunk borders' shared vertices), degenerate
    /// triangles dropped.
    pub fn merged(&self) -> Mesh {
        let mut out = Mesh::new();
        let mut index: HashMap<[u32; 3], u32> = HashMap::new();
        for m in &self.meshes {
            let mut remap = Vec::with_capacity(m.vertices.len());
            for v in &m.vertices {
                let key = [
                    v.position.x.to_bits(),
                    v.position.y.to_bits(),
                    v.position.z.to_bits(),
                ];
                let i = *index.entry(key).or_insert_with(|| {
                    out.vertices.push(*v);
                    (out.vertices.len() - 1) as u32
                });
                remap.push(i);
            }
            out.indices
                .extend(m.indices.iter().map(|&i| remap[i as usize]));
        }
        remove_degenerate_triangles(&mut out);
        out
    }
}

/// Marching cubes of chunk `c` on the global lattice.
fn mesh_chunk(state: &LiveState, config: &LiveMeshConfig, c: [usize; 3]) -> Mesh {
    let n = config.chunk_cells;
    // Samples cover lattice indices -1 ..= n + 1 around the chunk (one cell
    // of padding for the endpoint gradients).
    let side = n + 3;
    let base = [(c[0] * n) as i64, (c[1] * n) as i64, (c[2] * n) as i64];
    let mut values = vec![0.0f32; side * side * side];
    for k in 0..side {
        for j in 0..side {
            for i in 0..side {
                let g = [
                    base[0] + i as i64 - 1,
                    base[1] + j as i64 - 1,
                    base[2] + k as i64 - 1,
                ];
                values[(k * side + j) * side + i] = state.distance(config.lattice_point(g));
            }
        }
    }
    // Local lattice index (0 ..= n) -> padded sample.
    let at = |x: usize, y: usize, z: usize| values[((z + 1) * side + (y + 1)) * side + (x + 1)];
    let gradient = |x: usize, y: usize, z: usize| {
        let s = |dx: isize, dy: isize, dz: isize| {
            values[(((z as isize + 1 + dz) as usize) * side + (y as isize + 1 + dy) as usize)
                * side
                + (x as isize + 1 + dx) as usize]
        };
        Vec3::new(
            s(1, 0, 0) - s(-1, 0, 0),
            s(0, 1, 0) - s(0, -1, 0),
            s(0, 0, 1) - s(0, 0, -1),
        )
    };

    let mut mesh = Mesh::new();
    // Global edge key (lower endpoint, axis) -> vertex index.
    let mut edge_vertex: HashMap<([i64; 3], u8), u32> = HashMap::new();
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let mut cube_index = 0usize;
                let mut corner = [[0usize; 3]; 8];
                let mut value = [0.0f32; 8];
                for (i, off) in CORNER_OFFSETS.iter().enumerate() {
                    corner[i] = [x + off[0], y + off[1], z + off[2]];
                    value[i] = at(corner[i][0], corner[i][1], corner[i][2]);
                    if value[i] < 0.0 {
                        cube_index |= 1 << i;
                    }
                }
                let edges = EDGE_TABLE[cube_index];
                if edges == 0 {
                    continue;
                }
                let mut vid = [0u32; 12];
                for (e, ends) in EDGE_CONNECTIONS.iter().enumerate() {
                    if edges & (1 << e) == 0 {
                        continue;
                    }
                    let (a, b) = (corner[ends[0]], corner[ends[1]]);
                    let lo = if (a[2], a[1], a[0]) <= (b[2], b[1], b[0]) {
                        a
                    } else {
                        b
                    };
                    let axis = (0..3).find(|&k| a[k] != b[k]).unwrap_or(0) as u8;
                    let key = (
                        [
                            base[0] + lo[0] as i64,
                            base[1] + lo[1] as i64,
                            base[2] + lo[2] as i64,
                        ],
                        axis,
                    );
                    if let Some(&id) = edge_vertex.get(&key) {
                        vid[e] = id;
                        continue;
                    }
                    let ga = [
                        base[0] + a[0] as i64,
                        base[1] + a[1] as i64,
                        base[2] + a[2] as i64,
                    ];
                    let gb = [
                        base[0] + b[0] as i64,
                        base[1] + b[1] as i64,
                        base[2] + b[2] as i64,
                    ];
                    let (pa, pb) = (config.lattice_point(ga), config.lattice_point(gb));
                    let (va, vb) = (value[ends[0]], value[ends[1]]);
                    let (pos, t, swapped) = interpolate_edge(pa, pb, va, vb, 0.0);
                    let (na, nb) = (gradient(a[0], a[1], a[2]), gradient(b[0], b[1], b[2]));
                    let (n0, n1) = if swapped { (nb, na) } else { (na, nb) };
                    let blend = n0 + (n1 - n0) * t;
                    let normal = if blend.length_squared() < 1e-30 {
                        Vec3::Y
                    } else {
                        blend.normalize()
                    };
                    let id = mesh.vertices.len() as u32;
                    mesh.vertices.push(Vertex::new(pos, normal));
                    edge_vertex.insert(key, id);
                    vid[e] = id;
                }
                let tri = &TRI_TABLE[cube_index];
                let mut i = 0;
                while tri[i] != -1 {
                    mesh.indices.push(vid[tri[i] as usize]);
                    mesh.indices.push(vid[tri[i + 1] as usize]);
                    mesh.indices.push(vid[tri[i + 2] as usize]);
                    i += 3;
                }
            }
        }
    }
    remove_degenerate_triangles(&mut mesh);
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn influence_union_covers_both_sides() {
        let a = Influence::Within {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        };
        let b = Influence::Within {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(0.5),
        };
        assert_eq!(
            a.union(b),
            Some(DirtyRegion::Aabb {
                min: Vec3::splat(-1.0),
                max: Vec3::ONE
            })
        );
        assert_eq!(Influence::Nowhere.union(Influence::Nowhere), None);
        assert_eq!(
            Influence::Nowhere.union(Influence::Everywhere),
            Some(DirtyRegion::Everywhere)
        );
    }

    #[test]
    fn change_log_drops_old_records_into_everywhere() {
        let live = LiveSdf::new(SdfNode::sphere(1.0));
        for i in 0..(CHANGE_LOG_CAPACITY + 2) {
            let mut s = live.write();
            s.record(DirtyRegion::Aabb {
                min: Vec3::splat(i as f32),
                max: Vec3::splat(i as f32),
            });
        }
        assert_eq!(live.changes_since(0).regions, vec![DirtyRegion::Everywhere]);
        let g = live.generation();
        assert_eq!(live.changes_since(g - 1).regions.len(), 1);
        assert!(live.changes_since(g).regions.is_empty());
    }
}
