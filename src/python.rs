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

    /// Create a plane
    #[staticmethod]
    fn plane(nx: f32, ny: f32, nz: f32, distance: f32) -> Self {
        PySdfNode {
            inner: SdfNode::Plane {
                normal: Vec3::new(nx, ny, nz).normalize(),
                distance,
            },
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

    /// Generate GLSL shader code
    #[cfg(feature = "glsl")]
    fn to_glsl(&self) -> String {
        let shader = crate::compiled::GlslShader::transpile(&self.inner, crate::compiled::GlslTranspileMode::Hardcoded);
        shader.source
    }

    /// Generate HLSL shader code
    #[cfg(feature = "hlsl")]
    fn to_hlsl(&self) -> String {
        let shader = crate::compiled::HlslShader::transpile(&self.inner, crate::compiled::HlslTranspileMode::Hardcoded);
        shader.source
    }

    /// Generate WGSL shader code
    #[cfg(feature = "gpu")]
    fn to_wgsl(&self) -> String {
        let shader = crate::compiled::WgslShader::transpile(&self.inner, crate::compiled::TranspileMode::Hardcoded);
        shader.source
    }

    fn __repr__(&self) -> String {
        format!("SdfNode(nodes={})", self.inner.node_count())
    }
}

/// Compiled SDF for fast evaluation
#[pyclass(name = "CompiledSdf")]
pub struct PyCompiledSdf {
    compiled: crate::compiled::CompiledSdf,
}

#[pymethods]
impl PyCompiledSdf {
    /// Evaluate at a single point
    fn eval(&self, x: f32, y: f32, z: f32) -> f32 {
        crate::compiled::eval_compiled(&self.compiled, Vec3::new(x, y, z))
    }

    /// Get instruction count
    fn instruction_count(&self) -> usize {
        self.compiled.instruction_count()
    }

    fn __repr__(&self) -> String {
        format!("CompiledSdf(instructions={})", self.compiled.instruction_count())
    }
}

/// Compile an SDF for fast evaluation
#[pyfunction]
fn compile_sdf(node: &PySdfNode) -> PyCompiledSdf {
    let compiled = crate::compiled::CompiledSdf::compile(&node.inner);
    PyCompiledSdf { compiled }
}

/// Evaluate compiled SDF at multiple points (NumPy array, GIL released)
#[pyfunction]
fn eval_compiled_batch<'py>(
    py: Python<'py>,
    compiled: &PyCompiledSdf,
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

    // Release GIL during heavy computation to allow Python threads to run
    let compiled_ref = &compiled.compiled;
    let distances = py.allow_threads(|| {
        crate::compiled::eval_compiled_batch_parallel(compiled_ref, &vec_points)
    });
    Ok(distances.into_pyarray_bound(py))
}

/// Evaluate SDF at multiple points (NumPy array, GIL released)
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

    // Release GIL during heavy computation to allow Python threads to run
    let node_ref = &node.inner;
    let distances = py.allow_threads(|| {
        eval_batch_parallel(node_ref, &vec_points)
    });
    Ok(distances.into_pyarray_bound(py))
}

/// Convert SDF to mesh (standard marching cubes)
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
        ..Default::default()
    };

    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

    // Release GIL during mesh generation
    let node_ref = &node.inner;
    let mesh = py.allow_threads(|| {
        sdf_to_mesh(node_ref, min, max, &config)
    });
    mesh_to_numpy(py, &mesh)
}

/// Convert SDF to mesh using adaptive marching cubes
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, max_depth=6, min_depth=2, surface_threshold=1.0))]
fn to_mesh_adaptive<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    max_depth: u32,
    min_depth: u32,
    surface_threshold: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::AdaptiveConfig;

    let config = AdaptiveConfig {
        max_depth,
        min_depth,
        surface_threshold,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };

    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

    // Release GIL during mesh generation
    let node_ref = &node.inner;
    let mesh = py.allow_threads(|| {
        crate::mesh::adaptive_marching_cubes(node_ref, min, max, &config)
    });
    mesh_to_numpy(py, &mesh)
}

/// Decimate a mesh to reduce triangle count
#[pyfunction]
#[pyo3(signature = (vertices, indices, target_ratio=0.5, preserve_boundary=true))]
fn decimate_mesh<'py>(
    py: Python<'py>,
    vertices: &Bound<'py, PyArray2<f32>>,
    indices: &Bound<'py, PyArray1<u32>>,
    target_ratio: f32,
    preserve_boundary: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::{Mesh, Vertex, DecimateConfig, decimate};

    let verts = unsafe { vertices.as_array() };
    let idx = unsafe { indices.as_array() };

    if verts.shape()[1] != 3 {
        return Err(PyValueError::new_err("Vertices must have shape (N, 3)"));
    }

    let mesh_verts: Vec<Vertex> = verts
        .rows()
        .into_iter()
        .map(|row| Vertex {
            position: Vec3::new(row[0], row[1], row[2]),
            ..Default::default()
        })
        .collect();

    let mesh_indices: Vec<u32> = idx.to_vec().unwrap_or_default();

    let mut mesh = Mesh {
        vertices: mesh_verts,
        indices: mesh_indices,
    };

    let config = DecimateConfig {
        target_ratio,
        preserve_boundary,
        ..Default::default()
    };

    decimate(&mut mesh, &config);
    mesh_to_numpy(py, &mesh)
}

/// Export mesh to OBJ file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_obj(
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::mesh::{Mesh, Vertex};
    use crate::io::{export_obj as io_export_obj, ObjConfig};

    let verts = unsafe { vertices.as_array() };
    let idx = unsafe { indices.as_array() };

    if verts.shape()[1] != 3 {
        return Err(PyValueError::new_err("Vertices must have shape (N, 3)"));
    }

    let mesh_verts: Vec<Vertex> = verts
        .rows()
        .into_iter()
        .map(|row| Vertex {
            position: Vec3::new(row[0], row[1], row[2]),
            ..Default::default()
        })
        .collect();

    let mesh = Mesh {
        vertices: mesh_verts,
        indices: idx.to_vec().unwrap_or_default(),
    };

    io_export_obj(&mesh, path, &ObjConfig::default(), None)
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Export mesh to GLB file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_glb(
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::mesh::{Mesh, Vertex};
    use crate::io::{export_glb as io_export_glb, GltfConfig};

    let verts = unsafe { vertices.as_array() };
    let idx = unsafe { indices.as_array() };

    if verts.shape()[1] != 3 {
        return Err(PyValueError::new_err("Vertices must have shape (N, 3)"));
    }

    let mesh_verts: Vec<Vertex> = verts
        .rows()
        .into_iter()
        .map(|row| Vertex {
            position: Vec3::new(row[0], row[1], row[2]),
            ..Default::default()
        })
        .collect();

    let mesh = Mesh {
        vertices: mesh_verts,
        indices: idx.to_vec().unwrap_or_default(),
    };

    io_export_glb(&mesh, path, &GltfConfig::default())
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
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

/// Helper: convert Mesh to NumPy arrays
fn mesh_to_numpy<'py>(
    py: Python<'py>,
    mesh: &crate::mesh::Mesh,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    let vertices: Vec<f32> = mesh
        .vertices
        .iter()
        .flat_map(|v| [v.position.x, v.position.y, v.position.z])
        .collect();

    let vertex_array = numpy::PyArray1::from_vec_bound(py, vertices)
        .reshape([mesh.vertices.len(), 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    let indices = mesh.indices.clone().into_pyarray_bound(py);

    Ok((vertex_array, indices))
}

/// Python module
#[pymodule]
pub fn alice_sdf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySdfNode>()?;
    m.add_class::<PyCompiledSdf>()?;
    m.add_function(wrap_pyfunction!(compile_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(eval_batch, m)?)?;
    m.add_function(wrap_pyfunction!(eval_compiled_batch, m)?)?;
    m.add_function(wrap_pyfunction!(to_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(to_mesh_adaptive, m)?)?;
    m.add_function(wrap_pyfunction!(decimate_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(export_obj, m)?)?;
    m.add_function(wrap_pyfunction!(export_glb, m)?)?;
    m.add_function(wrap_pyfunction!(save_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(load_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
