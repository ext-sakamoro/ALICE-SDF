//! Shape modifier methods for SdfNode
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};
use std::sync::Arc;

use super::SdfNode;

impl SdfNode {
    // === Modifier methods ===

    /// Twist around Y-axis
    #[must_use]
    #[inline]
    pub fn twist(self, strength: f32) -> Self {
        SdfNode::Twist {
            child: Arc::new(self),
            strength,
        }
    }

    /// Bend around Y-axis
    #[must_use]
    #[inline]
    pub fn bend(self, curvature: f32) -> Self {
        SdfNode::Bend {
            child: Arc::new(self),
            curvature,
        }
    }

    /// Infinite repetition
    #[must_use]
    #[inline]
    pub fn repeat_infinite(self, spacing_x: f32, spacing_y: f32, spacing_z: f32) -> Self {
        SdfNode::RepeatInfinite {
            child: Arc::new(self),
            spacing: Vec3::new(spacing_x, spacing_y, spacing_z),
        }
    }

    /// Finite repetition
    #[must_use]
    #[inline]
    pub fn repeat_finite(self, count: [u32; 3], spacing: Vec3) -> Self {
        SdfNode::RepeatFinite {
            child: Arc::new(self),
            count,
            spacing,
        }
    }

    /// Perlin noise displacement
    #[must_use]
    #[inline]
    pub fn noise(self, amplitude: f32, frequency: f32, seed: u32) -> Self {
        SdfNode::Noise {
            child: Arc::new(self),
            amplitude,
            frequency,
            seed,
        }
    }

    /// Round edges
    #[must_use]
    #[inline]
    pub fn round(self, radius: f32) -> Self {
        SdfNode::Round {
            child: Arc::new(self),
            radius,
        }
    }

    /// Create a shell (onion)
    #[must_use]
    #[inline]
    pub fn onion(self, thickness: f32) -> Self {
        SdfNode::Onion {
            child: Arc::new(self),
            thickness,
        }
    }

    /// Elongate along an axis
    #[must_use]
    #[inline]
    pub fn elongate(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::Elongate {
            child: Arc::new(self),
            amount: Vec3::new(x, y, z),
        }
    }

    /// Mirror along specified axes
    #[must_use]
    #[inline]
    pub fn mirror(self, x: bool, y: bool, z: bool) -> Self {
        SdfNode::Mirror {
            child: Arc::new(self),
            axes: Vec3::new(
                if x { 1.0 } else { 0.0 },
                if y { 1.0 } else { 0.0 },
                if z { 1.0 } else { 0.0 },
            ),
        }
    }

    /// Revolution around Y-axis
    #[must_use]
    #[inline]
    pub fn revolution(self, offset: f32) -> Self {
        SdfNode::Revolution {
            child: Arc::new(self),
            offset,
        }
    }

    /// Extrude along Z-axis
    #[must_use]
    #[inline]
    pub fn extrude(self, height: f32) -> Self {
        SdfNode::Extrude {
            child: Arc::new(self),
            half_height: height * 0.5,
        }
    }

    /// Sweep along a quadratic Bezier curve in the XZ plane.
    /// Control points are (x, z) coordinates.
    #[must_use]
    #[inline]
    pub fn sweep_bezier(self, p0: Vec2, p1: Vec2, p2: Vec2) -> Self {
        SdfNode::SweepBezier {
            child: Arc::new(self),
            p0,
            p1,
            p2,
        }
    }

    /// Taper along Y-axis
    #[must_use]
    #[inline]
    pub fn taper(self, factor: f32) -> Self {
        SdfNode::Taper {
            child: Arc::new(self),
            factor,
        }
    }

    /// Sin-based displacement
    #[must_use]
    #[inline]
    pub fn displacement(self, strength: f32) -> Self {
        SdfNode::Displacement {
            child: Arc::new(self),
            strength,
        }
    }

    /// Polar repetition around Y-axis
    #[must_use]
    #[inline]
    pub fn polar_repeat(self, count: u32) -> Self {
        SdfNode::PolarRepeat {
            child: Arc::new(self),
            count,
        }
    }

    /// Octant mirror (48-fold symmetry)
    #[must_use]
    #[inline]
    pub fn octant_mirror(self) -> Self {
        SdfNode::OctantMirror {
            child: Arc::new(self),
        }
    }

    /// Apply shear deformation
    #[must_use]
    #[inline]
    pub fn shear(self, xy: f32, xz: f32, yz: f32) -> Self {
        SdfNode::Shear {
            child: Arc::new(self),
            shear: Vec3::new(xy, xz, yz),
        }
    }

    /// Apply time-based animation
    #[must_use]
    #[inline]
    pub fn animated(self, speed: f32, amplitude: f32) -> Self {
        SdfNode::Animated {
            child: Arc::new(self),
            speed,
            amplitude,
        }
    }

    /// Assign a material ID to this subtree
    #[must_use]
    #[inline]
    pub fn with_material(self, material_id: u32) -> Self {
        SdfNode::WithMaterial {
            child: Arc::new(self),
            material_id,
        }
    }

    /// Icosahedral symmetry (120-fold)
    #[must_use]
    #[inline]
    pub fn icosahedral_symmetry(self) -> Self {
        SdfNode::IcosahedralSymmetry {
            child: Arc::new(self),
        }
    }

    /// Iterated Function System
    #[must_use]
    #[inline]
    pub fn ifs(self, transforms: Vec<[f32; 16]>, iterations: u32) -> Self {
        SdfNode::IFS {
            child: Arc::new(self),
            transforms,
            iterations,
        }
    }

    /// Heightmap displacement
    #[must_use]
    #[inline]
    pub fn heightmap_displacement(
        self,
        heightmap: Vec<f32>,
        width: u32,
        height: u32,
        amplitude: f32,
        scale: f32,
    ) -> Self {
        SdfNode::HeightmapDisplacement {
            child: Arc::new(self),
            heightmap,
            width,
            height,
            amplitude,
            scale,
        }
    }

    /// Surface roughness (FBM noise)
    #[must_use]
    #[inline]
    pub fn surface_roughness(self, frequency: f32, amplitude: f32, octaves: u32) -> Self {
        SdfNode::SurfaceRoughness {
            child: Arc::new(self),
            frequency,
            amplitude,
            octaves,
        }
    }
}
