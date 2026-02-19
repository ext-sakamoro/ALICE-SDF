//! Transform and modifier operations for PySdfNode.

use glam::{Vec2, Vec3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::node::PySdfNode;

#[pymethods]
impl PySdfNode {
    /// Translate
    fn translate(&self, x: f32, y: f32, z: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().translate(x, y, z),
        }
    }

    /// Rotate by Euler angles (radians)
    fn rotate(&self, x: f32, y: f32, z: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().rotate_euler(x, y, z),
        }
    }

    /// Uniform scale
    fn scale(&self, factor: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().scale(factor),
        }
    }

    /// Non-uniform scale (different factor per axis)
    fn scale_xyz(&self, x: f32, y: f32, z: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().scale_xyz(x, y, z),
        }
    }

    /// Twist around Y-axis
    fn twist(&self, strength: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().twist(strength),
        }
    }

    /// Bend
    fn bend(&self, curvature: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().bend(curvature),
        }
    }

    /// Infinite repetition
    fn repeat(&self, spacing_x: f32, spacing_y: f32, spacing_z: f32) -> Self {
        PySdfNode {
            inner: self
                .inner
                .clone()
                .repeat_infinite(spacing_x, spacing_y, spacing_z),
        }
    }

    /// Perlin noise
    fn noise(&self, amplitude: f32, frequency: f32, seed: u32) -> Self {
        PySdfNode {
            inner: self.inner.clone().noise(amplitude, frequency, seed),
        }
    }

    /// Round edges
    fn round(&self, radius: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().round(radius),
        }
    }

    /// Shell (onion)
    fn onion(&self, thickness: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().onion(thickness),
        }
    }

    /// Mirror along specified axes
    fn mirror(&self, x: bool, y: bool, z: bool) -> Self {
        PySdfNode {
            inner: self.inner.clone().mirror(x, y, z),
        }
    }

    /// Octant mirror (48-fold symmetry: abs + sort x >= y >= z)
    fn octant_mirror(&self) -> Self {
        PySdfNode {
            inner: self.inner.clone().octant_mirror(),
        }
    }

    /// Elongate along axes
    fn elongate(&self, x: f32, y: f32, z: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().elongate(x, y, z),
        }
    }

    /// Finite repetition
    fn repeat_finite(
        &self,
        count_x: u32,
        count_y: u32,
        count_z: u32,
        spacing_x: f32,
        spacing_y: f32,
        spacing_z: f32,
    ) -> Self {
        PySdfNode {
            inner: self.inner.clone().repeat_finite(
                [count_x, count_y, count_z],
                Vec3::new(spacing_x, spacing_y, spacing_z),
            ),
        }
    }

    /// Revolution around Y-axis
    fn revolution(&self, offset: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().revolution(offset),
        }
    }

    /// Extrude along Z-axis
    fn extrude(&self, height: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().extrude(height),
        }
    }

    /// Taper along Y-axis
    fn taper(&self, factor: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().taper(factor),
        }
    }

    /// Sin-based displacement
    fn displacement(&self, strength: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().displacement(strength),
        }
    }

    /// Polar repetition around Y-axis
    fn polar_repeat(&self, count: u32) -> Self {
        PySdfNode {
            inner: self.inner.clone().polar_repeat(count),
        }
    }

    /// Assign material ID
    fn with_material(&self, material_id: u32) -> Self {
        PySdfNode {
            inner: self.inner.clone().with_material(material_id),
        }
    }

    /// Sweep along a quadratic Bezier curve in XZ plane
    fn sweep_bezier(&self, p0x: f32, p0y: f32, p1x: f32, p1y: f32, p2x: f32, p2y: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().sweep_bezier(
                Vec2::new(p0x, p0y),
                Vec2::new(p1x, p1y),
                Vec2::new(p2x, p2y),
            ),
        }
    }

    /// Shear deformation
    fn shear(&self, xy: f32, xz: f32, yz: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().shear(xy, xz, yz),
        }
    }

    /// Apply time-based animation
    fn animated(&self, speed: f32, amplitude: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().animated(speed, amplitude),
        }
    }

    // --- New Transform Variants ---

    /// Apply a projective transformation (4x4 matrix)
    #[pyo3(text_signature = "(inv_matrix, lipschitz_bound)")]
    fn projective_transform(&self, inv_matrix: Vec<f32>, lipschitz_bound: f32) -> PyResult<Self> {
        if inv_matrix.len() != 16 {
            return Err(PyValueError::new_err("Matrix must have 16 elements"));
        }
        let matrix: [f32; 16] = inv_matrix.try_into().unwrap();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .projective_transform(matrix, lipschitz_bound),
        })
    }

    /// Apply lattice-based Free-Form Deformation (FFD)
    #[pyo3(text_signature = "(control_points, nx, ny, nz, bbox_min, bbox_max)")]
    fn lattice_deform(
        &self,
        control_points: Vec<[f32; 3]>,
        nx: u32,
        ny: u32,
        nz: u32,
        bbox_min: [f32; 3],
        bbox_max: [f32; 3],
    ) -> Self {
        let cp: Vec<Vec3> = control_points
            .iter()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();
        Self {
            inner: self.inner.clone().lattice_deform(
                cp,
                nx,
                ny,
                nz,
                Vec3::from(bbox_min),
                Vec3::from(bbox_max),
            ),
        }
    }

    /// Apply skeletal skinning deformation with bone weights
    #[pyo3(text_signature = "(bones)")]
    fn sdf_skinning(&self, bones: Vec<(Vec<f32>, Vec<f32>, f32)>) -> PyResult<Self> {
        use crate::transforms::skinning::BoneTransform;

        let mut bone_data = Vec::new();
        for (ibp, cp, w) in bones {
            if ibp.len() != 16 || cp.len() != 16 {
                return Err(PyValueError::new_err(
                    "Each bone matrix must have 16 elements (inv_bind_pose, current_pose, weight)",
                ));
            }
            bone_data.push(BoneTransform {
                inv_bind_pose: ibp.try_into().unwrap(),
                current_pose: cp.try_into().unwrap(),
                weight: w,
            });
        }
        Ok(Self {
            inner: self.inner.clone().sdf_skinning(bone_data),
        })
    }

    // --- New Modifier Variants ---

    /// Apply icosahedral symmetry (60-fold rotational symmetry)
    fn icosahedral_symmetry(&self) -> Self {
        Self {
            inner: self.inner.clone().icosahedral_symmetry(),
        }
    }

    /// Apply Iterated Function System (fractal) transformations
    #[pyo3(text_signature = "(transforms, iterations)")]
    fn ifs(&self, transforms: Vec<Vec<f32>>, iterations: u32) -> PyResult<Self> {
        let ts: Result<Vec<[f32; 16]>, _> = transforms
            .iter()
            .map(|t| {
                if t.len() != 16 {
                    Err(PyValueError::new_err(
                        "Each transform must have 16 elements",
                    ))
                } else {
                    Ok(t.as_slice().try_into().unwrap())
                }
            })
            .collect();
        Ok(Self {
            inner: self.inner.clone().ifs(ts?, iterations),
        })
    }

    /// Apply heightmap-based displacement
    #[pyo3(text_signature = "(heightmap, width, height, amplitude, scale)")]
    fn heightmap_displacement(
        &self,
        heightmap: Vec<f32>,
        width: u32,
        height: u32,
        amplitude: f32,
        scale: f32,
    ) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .heightmap_displacement(heightmap, width, height, amplitude, scale),
        }
    }

    /// Apply surface roughness using Perlin noise
    #[pyo3(text_signature = "(frequency, amplitude, octaves)")]
    fn surface_roughness(&self, frequency: f32, amplitude: f32, octaves: u32) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .surface_roughness(frequency, amplitude, octaves),
        }
    }
}
