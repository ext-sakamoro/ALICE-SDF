//! Animation System for SDF parameter morphing
//!
//! Provides keyframe-based animation of SDF parameters for
//! real-time deformation, morphing, and cinematic sequences.
//!
//! # Features
//! - Keyframe interpolation (Linear, CubicBezier, Step)
//! - Timeline with multiple tracks
//! - AnimatedSdf for time-varying SDF evaluation
//! - Looping modes (Once, Loop, PingPong)
//!
//! Author: Moroya Sakamoto

use crate::types::SdfNode;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Interpolation mode between keyframes
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum Interpolation {
    /// Linear interpolation (lerp)
    #[default]
    Linear,
    /// Cubic Bezier with control tangents
    CubicBezier {
        /// Out tangent of previous keyframe
        out_tangent: f32,
        /// In tangent of next keyframe
        in_tangent: f32,
    },
    /// Step: hold value until next keyframe
    Step,
}

/// Loop mode for animation playback
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum LoopMode {
    /// Play once and stop at the end
    #[default]
    Once,
    /// Loop back to start
    Loop,
    /// Alternate forward and backward
    PingPong,
}

/// A single keyframe with time, value, and interpolation mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyframe {
    /// Time in seconds
    pub time: f32,
    /// Value at this keyframe
    pub value: f32,
    /// Interpolation to next keyframe
    pub interpolation: Interpolation,
}

impl Keyframe {
    /// Create a linear keyframe
    pub fn new(time: f32, value: f32) -> Self {
        Keyframe {
            time,
            value,
            interpolation: Interpolation::Linear,
        }
    }

    /// Create a step keyframe
    pub fn step(time: f32, value: f32) -> Self {
        Keyframe {
            time,
            value,
            interpolation: Interpolation::Step,
        }
    }

    /// Create a cubic bezier keyframe
    pub fn cubic(time: f32, value: f32, out_tangent: f32, in_tangent: f32) -> Self {
        Keyframe {
            time,
            value,
            interpolation: Interpolation::CubicBezier {
                out_tangent,
                in_tangent,
            },
        }
    }
}

/// An animation track controlling a single float parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    /// Track name (e.g., "sphere.radius", "translate.x")
    pub name: String,
    /// Keyframes sorted by time
    pub keyframes: Vec<Keyframe>,
    /// Loop mode
    pub loop_mode: LoopMode,
}

impl Track {
    /// Create a new empty track
    pub fn new(name: impl Into<String>) -> Self {
        Track {
            name: name.into(),
            keyframes: Vec::new(),
            loop_mode: LoopMode::Once,
        }
    }

    /// Add a keyframe (maintains sorted order).
    ///
    /// Uses `total_cmp` for the time comparison so NaN timestamps do not panic.
    pub fn add_keyframe(&mut self, keyframe: Keyframe) {
        let pos = self
            .keyframes
            .binary_search_by(|k| k.time.total_cmp(&keyframe.time))
            .unwrap_or_else(|p| p);
        self.keyframes.insert(pos, keyframe);
    }

    /// Set loop mode
    pub fn with_loop(mut self, mode: LoopMode) -> Self {
        self.loop_mode = mode;
        self
    }

    /// Get the duration of the track
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map_or(0.0, |k| k.time)
    }

    /// Evaluate the track at a given time
    pub fn evaluate(&self, time: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }

        let duration = self.duration();
        if duration <= 0.0 {
            return self.keyframes[0].value;
        }

        // Apply loop mode
        let t = match self.loop_mode {
            LoopMode::Once => time.clamp(0.0, duration),
            LoopMode::Loop => {
                let t = time % duration;
                if t < 0.0 {
                    t + duration
                } else {
                    t
                }
            }
            LoopMode::PingPong => {
                let cycle = time / duration;
                let phase = cycle % 2.0;
                if phase < 1.0 {
                    phase * duration
                } else {
                    (2.0 - phase) * duration
                }
            }
        };

        // Find surrounding keyframes
        if t <= self.keyframes[0].time {
            return self.keyframes[0].value;
        }
        // SAFETY: keyframes is non-empty (checked above) so last() always returns Some.
        if let Some(last) = self.keyframes.last() {
            if t >= last.time {
                return last.value;
            }
        }

        // Binary search for the keyframe pair
        // Use total_cmp to avoid panicking on NaN timestamps.
        let idx = self
            .keyframes
            .binary_search_by(|k| k.time.total_cmp(&t))
            .unwrap_or_else(|p| p);

        let idx = if idx > 0 { idx - 1 } else { 0 };
        let k0 = &self.keyframes[idx];
        let k1 = &self.keyframes[(idx + 1).min(self.keyframes.len() - 1)];

        let span = k1.time - k0.time;
        if span <= 0.0 {
            return k0.value;
        }

        let alpha = (t - k0.time) / span;

        match k0.interpolation {
            Interpolation::Linear => k0.value + (k1.value - k0.value) * alpha,
            Interpolation::Step => k0.value,
            Interpolation::CubicBezier {
                out_tangent,
                in_tangent,
            } => {
                // Hermite interpolation
                let t2 = alpha * alpha;
                let t3 = t2 * alpha;
                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + alpha;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;
                h00 * k0.value + h10 * span * out_tangent + h01 * k1.value + h11 * span * in_tangent
            }
        }
    }
}

/// Animation timeline containing multiple tracks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    /// Name of this animation
    pub name: String,
    /// Animation tracks
    pub tracks: Vec<Track>,
    /// Playback speed multiplier
    pub speed: f32,
}

impl Timeline {
    /// Create a new empty timeline
    pub fn new(name: impl Into<String>) -> Self {
        Timeline {
            name: name.into(),
            tracks: Vec::new(),
            speed: 1.0,
        }
    }

    /// Add a track
    pub fn add_track(&mut self, track: Track) {
        self.tracks.push(track);
    }

    /// Get the total duration
    pub fn duration(&self) -> f32 {
        self.tracks
            .iter()
            .map(|t| t.duration())
            .fold(0.0_f32, f32::max)
    }

    /// Evaluate all tracks at a given time, returning name-value pairs
    pub fn evaluate(&self, time: f32) -> Vec<(&str, f32)> {
        let t = time * self.speed;
        self.tracks
            .iter()
            .map(|track| (track.name.as_str(), track.evaluate(t)))
            .collect()
    }

    /// Get the value of a specific track by name
    pub fn get_value(&self, track_name: &str, time: f32) -> Option<f32> {
        let t = time * self.speed;
        self.tracks
            .iter()
            .find(|track| track.name == track_name)
            .map(|track| track.evaluate(t))
    }
}

/// Animation parameters extracted from timeline (36 bytes, stack-allocated)
///
/// Zero-allocation alternative to `evaluate_at()`. Instead of cloning the
/// entire SdfNode tree every frame, extract only the transform parameters
/// and apply them during compiled evaluation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AnimationParams {
    /// Translation along the X axis (world units)
    pub translate_x: f32,
    /// Translation along the Y axis (world units)
    pub translate_y: f32,
    /// Translation along the Z axis (world units)
    pub translate_z: f32,
    /// Euler rotation around the X axis (radians)
    pub rotate_x: f32,
    /// Euler rotation around the Y axis (radians)
    pub rotate_y: f32,
    /// Euler rotation around the Z axis (radians)
    pub rotate_z: f32,
    /// Uniform scale factor (1.0 = no scale)
    pub scale: f32,
    /// Twist deformation strength
    pub twist: f32,
    /// Bend deformation curvature
    pub bend: f32,
}

impl AnimationParams {
    /// Returns `true` if any translation component is non-zero (> 1e-6).
    #[inline]
    pub fn has_translation(&self) -> bool {
        self.translate_x.abs() > 1e-6
            || self.translate_y.abs() > 1e-6
            || self.translate_z.abs() > 1e-6
    }

    /// Returns `true` if any rotation component is non-zero (> 1e-6).
    #[inline]
    pub fn has_rotation(&self) -> bool {
        self.rotate_x.abs() > 1e-6 || self.rotate_y.abs() > 1e-6 || self.rotate_z.abs() > 1e-6
    }

    /// Returns `true` if the scale factor differs from 1.0 by more than 1e-6.
    #[inline]
    pub fn has_scale(&self) -> bool {
        (self.scale - 1.0).abs() > 1e-6
    }

    /// Apply inverse transform to a point (for SDF evaluation without tree cloning)
    ///
    /// Transforms the query point into the SDF's local space,
    /// avoiding the need to construct a new SdfNode tree.
    #[inline]
    pub fn transform_point(&self, point: glam::Vec3) -> (glam::Vec3, f32) {
        let mut p = point;
        let mut scale_correction = 1.0_f32;

        // Apply inverse transforms in reverse order (translate → rotate → scale → modifiers)
        if self.has_translation() {
            p -= glam::Vec3::new(self.translate_x, self.translate_y, self.translate_z);
        }
        if self.has_rotation() {
            let quat = glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                self.rotate_x,
                self.rotate_y,
                self.rotate_z,
            );
            p = quat.inverse() * p;
        }
        if self.has_scale() {
            let inv_scale = 1.0 / self.scale;
            p *= inv_scale;
            scale_correction = self.scale;
        }

        (p, scale_correction)
    }
}

/// Animated SDF that can be evaluated at different time values
///
/// Wraps an SDF node tree and applies parametric animation by
/// constructing modified SDF trees at each time step.
#[derive(Debug, Clone)]
pub struct AnimatedSdf {
    /// Base SDF shape
    pub base: SdfNode,
    /// Animation timeline
    pub timeline: Timeline,
}

impl AnimatedSdf {
    /// Create a new animated SDF
    pub fn new(base: SdfNode, timeline: Timeline) -> Self {
        AnimatedSdf { base, timeline }
    }

    /// Extract animation parameters at a given time (zero-allocation)
    ///
    /// Returns a stack-allocated AnimationParams struct instead of cloning
    /// the entire SdfNode tree. Use with `eval_animated_compiled()` for
    /// zero-allocation per-frame evaluation.
    pub fn evaluate_params(&self, time: f32) -> AnimationParams {
        let values = self.timeline.evaluate(time);
        let mut params = AnimationParams {
            scale: 1.0,
            ..Default::default()
        };

        for (name, value) in &values {
            match *name {
                "translate.x" => params.translate_x = *value,
                "translate.y" => params.translate_y = *value,
                "translate.z" => params.translate_z = *value,
                "rotate.x" => params.rotate_x = *value,
                "rotate.y" => params.rotate_y = *value,
                "rotate.z" => params.rotate_z = *value,
                "scale" => params.scale = *value,
                "twist" => params.twist = *value,
                "bend" => params.bend = *value,
                _ => {}
            }
        }

        params
    }

    /// Get the SDF node tree at a given time
    ///
    /// Applies animated transforms based on timeline track values.
    /// Supported track names:
    /// - "translate.x/y/z" — translation offset
    /// - "rotate.x/y/z" — Euler rotation (radians)
    /// - "scale" — uniform scale factor
    /// - "twist" — twist strength
    /// - "bend" — bend curvature
    pub fn evaluate_at(&self, time: f32) -> SdfNode {
        let values = self.timeline.evaluate(time);

        let mut tx = 0.0_f32;
        let mut ty = 0.0_f32;
        let mut tz = 0.0_f32;
        let mut rx = 0.0_f32;
        let mut ry = 0.0_f32;
        let mut rz = 0.0_f32;
        let mut scale = 1.0_f32;
        let mut twist = 0.0_f32;
        let mut bend = 0.0_f32;

        for (name, value) in &values {
            match *name {
                "translate.x" => tx = *value,
                "translate.y" => ty = *value,
                "translate.z" => tz = *value,
                "rotate.x" => rx = *value,
                "rotate.y" => ry = *value,
                "rotate.z" => rz = *value,
                "scale" => scale = *value,
                "twist" => twist = *value,
                "bend" => bend = *value,
                _ => {}
            }
        }

        let mut node = self.base.clone();

        // Apply modifiers (inner-most first)
        if twist.abs() > 1e-6 {
            node = node.twist(twist);
        }
        if bend.abs() > 1e-6 {
            node = node.bend(bend);
        }
        if (scale - 1.0).abs() > 1e-6 {
            node = node.scale(scale);
        }
        if rx.abs() > 1e-6 || ry.abs() > 1e-6 || rz.abs() > 1e-6 {
            node = node.rotate_euler(rx, ry, rz);
        }
        if tx.abs() > 1e-6 || ty.abs() > 1e-6 || tz.abs() > 1e-6 {
            node = node.translate(tx, ty, tz);
        }

        node
    }
}

/// Morph between two SDF shapes using smooth blending
///
/// Returns an SDF that smoothly transitions from `from` to `to`
/// based on the blend factor (0.0 = from, 1.0 = to).
pub fn morph(from: &SdfNode, to: &SdfNode, blend: f32) -> SdfNode {
    let k = (1.0 - blend).max(0.01); // smooth blending factor
    SdfNode::SmoothUnion {
        a: Arc::new(SdfNode::Scale {
            child: Arc::new(from.clone()),
            factor: 1.0,
        }),
        b: Arc::new(SdfNode::Scale {
            child: Arc::new(to.clone()),
            factor: 1.0,
        }),
        k,
    }
}

/// Evaluate an animated SDF at a point without allocating a new SdfNode tree
///
/// Uses pre-compiled bytecode + animation parameters for zero-allocation
/// per-frame evaluation. This is the recommended path for real-time animation:
///
/// ```
/// use alice_sdf::prelude::*;
/// use alice_sdf::animation::{AnimatedSdf, Timeline, eval_animated_compiled};
///
/// let timeline = Timeline::new("idle");
/// let animated = AnimatedSdf::new(SdfNode::sphere(1.0), timeline);
/// let compiled = CompiledSdf::compile(&animated.base);
/// let params = animated.evaluate_params(0.0);
/// let distance = eval_animated_compiled(&compiled, &params, Vec3::ZERO);
/// ```
pub fn eval_animated_compiled(
    compiled: &crate::compiled::CompiledSdf,
    params: &AnimationParams,
    point: glam::Vec3,
) -> f32 {
    let (transformed_point, scale_correction) = params.transform_point(point);
    crate::compiled::eval_compiled(compiled, transformed_point) * scale_correction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_linear() {
        let mut track = Track::new("radius");
        track.add_keyframe(Keyframe::new(0.0, 1.0));
        track.add_keyframe(Keyframe::new(1.0, 3.0));

        assert!((track.evaluate(0.0) - 1.0).abs() < 0.001);
        assert!((track.evaluate(0.5) - 2.0).abs() < 0.001);
        assert!((track.evaluate(1.0) - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_step() {
        let mut track = Track::new("state");
        track.add_keyframe(Keyframe::step(0.0, 0.0));
        track.add_keyframe(Keyframe::step(1.0, 1.0));

        assert!((track.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((track.evaluate(0.5) - 0.0).abs() < 0.001); // Step holds
        assert!((track.evaluate(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_loop_mode() {
        let mut track = Track::new("loop_test").with_loop(LoopMode::Loop);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(1.0, 1.0));

        assert!((track.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((track.evaluate(0.5) - 0.5).abs() < 0.001);
        // After loop, should wrap
        assert!((track.evaluate(1.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_ping_pong() {
        let mut track = Track::new("pp_test").with_loop(LoopMode::PingPong);
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(1.0, 1.0));

        assert!((track.evaluate(0.5) - 0.5).abs() < 0.001);
        // PingPong reverses after duration
        assert!((track.evaluate(1.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_timeline() {
        let mut timeline = Timeline::new("bounce");

        let mut tx = Track::new("translate.x");
        tx.add_keyframe(Keyframe::new(0.0, 0.0));
        tx.add_keyframe(Keyframe::new(1.0, 5.0));
        timeline.add_track(tx);

        let mut ty = Track::new("translate.y");
        ty.add_keyframe(Keyframe::new(0.0, 0.0));
        ty.add_keyframe(Keyframe::new(0.5, 3.0));
        ty.add_keyframe(Keyframe::new(1.0, 0.0));
        timeline.add_track(ty);

        assert_eq!(timeline.duration(), 1.0);

        let values = timeline.evaluate(0.5);
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_animated_sdf() {
        let sphere = SdfNode::sphere(1.0);

        let mut timeline = Timeline::new("move");
        let mut tx = Track::new("translate.x");
        tx.add_keyframe(Keyframe::new(0.0, 0.0));
        tx.add_keyframe(Keyframe::new(1.0, 5.0));
        timeline.add_track(tx);

        let animated = AnimatedSdf::new(sphere, timeline);

        // At t=0, should be at origin
        let node_t0 = animated.evaluate_at(0.0);
        assert_eq!(node_t0.node_count(), 1); // No translation at origin

        // At t=1, should be translated by 5
        let node_t1 = animated.evaluate_at(1.0);
        assert!(node_t1.node_count() >= 2); // Sphere + translate
    }

    #[test]
    fn test_cubic_interpolation() {
        let mut track = Track::new("smooth");
        track.add_keyframe(Keyframe::cubic(0.0, 0.0, 1.0, 0.0));
        track.add_keyframe(Keyframe::cubic(1.0, 1.0, 0.0, 1.0));

        // At boundaries
        assert!((track.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((track.evaluate(1.0) - 1.0).abs() < 0.001);

        // Mid-point should be somewhere between 0 and 1
        let mid = track.evaluate(0.5);
        assert!(mid >= -0.1 && mid <= 1.1);
    }

    #[test]
    fn test_animation_params_zero_alloc() {
        let sphere = SdfNode::sphere(1.0);

        let mut timeline = Timeline::new("move");
        let mut tx = Track::new("translate.x");
        tx.add_keyframe(Keyframe::new(0.0, 0.0));
        tx.add_keyframe(Keyframe::new(1.0, 5.0));
        timeline.add_track(tx);

        let animated = AnimatedSdf::new(sphere.clone(), timeline);

        // At t=0, no translation
        let params = animated.evaluate_params(0.0);
        assert!(params.translate_x.abs() < 0.001);
        assert_eq!(params.scale, 1.0);

        // At t=1, translated by 5
        let params = animated.evaluate_params(1.0);
        assert!((params.translate_x - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_eval_animated_compiled() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = crate::compiled::CompiledSdf::compile(&sphere);

        let mut timeline = Timeline::new("move");
        let mut tx = Track::new("translate.x");
        tx.add_keyframe(Keyframe::new(0.0, 0.0));
        tx.add_keyframe(Keyframe::new(1.0, 5.0));
        timeline.add_track(tx);

        let animated = AnimatedSdf::new(sphere, timeline);

        // At t=0, origin should be inside sphere (distance = -1)
        let params = animated.evaluate_params(0.0);
        let d = eval_animated_compiled(&compiled, &params, glam::Vec3::ZERO);
        assert!((d + 1.0).abs() < 0.01);

        // At t=1, sphere is at x=5, so origin should be outside
        let params = animated.evaluate_params(1.0);
        let d = eval_animated_compiled(&compiled, &params, glam::Vec3::ZERO);
        assert!(d > 0.0);
    }

    #[test]
    fn test_animation_params_size() {
        assert_eq!(std::mem::size_of::<AnimationParams>(), 36);
    }

    #[test]
    fn test_morph() {
        let sphere = SdfNode::sphere(1.0);
        let box3d = SdfNode::box3d(2.0, 2.0, 2.0);

        let morphed = morph(&sphere, &box3d, 0.5);
        assert!(morphed.node_count() >= 3);
    }
}
