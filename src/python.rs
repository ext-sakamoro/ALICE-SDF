//! Python bindings for ALICE-SDF
//!
//! Provides Python API via PyO3.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, IntoPyArray};
use glam::Vec3;

use crate::types::{SdfNode, SdfTree};
use crate::eval::{eval, eval_batch_parallel};
use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
use crate::io::{save, load};

/// Python-visible SdfNode wrapper
#[pyclass(name = "SdfNode")]
#[derive(Clone)]
pub struct PySdfNode {
    inner: SdfNode,
}

#[pymethods]
impl PySdfNode {
    /// Create a sphere
    #[staticmethod]
    fn sphere(radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::sphere(radius),
        }
    }

    /// Create a box
    #[staticmethod]
    fn box3d(width: f32, height: f32, depth: f32) -> Self {
        PySdfNode {
            inner: SdfNode::box3d(width, height, depth),
        }
    }

    /// Create a cylinder
    #[staticmethod]
    fn cylinder(radius: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::cylinder(radius, height),
        }
    }

    /// Create a torus
    #[staticmethod]
    fn torus(major_radius: f32, minor_radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::torus(major_radius, minor_radius),
        }
    }

    /// Create a capsule
    #[staticmethod]
    fn capsule(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::capsule(Vec3::new(ax, ay, az), Vec3::new(bx, by, bz), radius),
        }
    }

    /// Union with another shape
    fn union(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().union(other.inner.clone()),
        }
    }

    /// Intersection with another shape
    fn intersection(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().intersection(other.inner.clone()),
        }
    }

    /// Subtract another shape
    fn subtract(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().subtract(other.inner.clone()),
        }
    }

    /// Smooth union with another shape
    fn smooth_union(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().smooth_union(other.inner.clone(), k),
        }
    }

    /// Smooth intersection with another shape
    fn smooth_intersection(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().smooth_intersection(other.inner.clone(), k),
        }
    }

    /// Smooth subtraction
    fn smooth_subtract(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().smooth_subtract(other.inner.clone(), k),
        }
    }

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
            inner: self.inner.clone().repeat_infinite(spacing_x, spacing_y, spacing_z),
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

    /// Evaluate at a single point
    fn eval(&self, x: f32, y: f32, z: f32) -> f32 {
        eval(&self.inner, Vec3::new(x, y, z))
    }

    /// Get node count
    fn node_count(&self) -> u32 {
        self.inner.node_count()
    }

    fn __repr__(&self) -> String {
        format!("SdfNode(nodes={})", self.inner.node_count())
    }
}

/// Evaluate SDF at multiple points (NumPy array)
#[pyfunction]
fn eval_batch<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let points = unsafe { points.as_array() };

    if points.shape()[1] != 3 {
        return Err(PyValueError::new_err("Points must have shape (N, 3)"));
    }

    let vec_points: Vec<Vec3> = points
        .rows()
        .into_iter()
        .map(|row| Vec3::new(row[0], row[1], row[2]))
        .collect();

    let distances = eval_batch_parallel(&node.inner, &vec_points);
    Ok(distances.into_pyarray_bound(py))
}

/// Convert SDF to mesh
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64))]
fn to_mesh<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    let config = MarchingCubesConfig {
        resolution,
        iso_level: 0.0,
        compute_normals: true,
    };

    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

    let mesh = sdf_to_mesh(&node.inner, min, max, &config);

    // Convert to NumPy arrays
    let vertices: Vec<f32> = mesh
        .vertices
        .iter()
        .flat_map(|v| [v.position.x, v.position.y, v.position.z])
        .collect();

    let vertex_array = numpy::PyArray1::from_vec_bound(py, vertices)
        .reshape([mesh.vertices.len(), 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    let indices = mesh.indices.into_pyarray_bound(py);

    Ok((vertex_array, indices))
}

/// Save SDF to file
#[pyfunction]
fn save_sdf(node: &PySdfNode, path: &str) -> PyResult<()> {
    let tree = SdfTree::new(node.inner.clone());
    save(&tree, path).map_err(|e| PyValueError::new_err(format!("Save error: {}", e)))
}

/// Load SDF from file
#[pyfunction]
fn load_sdf(path: &str) -> PyResult<PySdfNode> {
    let tree = load(path).map_err(|e| PyValueError::new_err(format!("Load error: {}", e)))?;
    Ok(PySdfNode { inner: tree.root })
}

/// Get library version
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module
#[pymodule]
pub fn alice_sdf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySdfNode>()?;
    m.add_function(wrap_pyfunction!(eval_batch, m)?)?;
    m.add_function(wrap_pyfunction!(to_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(save_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(load_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
