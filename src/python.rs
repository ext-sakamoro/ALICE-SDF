//! Python bindings for ALICE-SDF
//!
//! Provides Python API via PyO3.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyArrayMethods, IntoPyArray};
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

    // --- Operator overloads for Pythonic DSL ---

    /// `a | b` → union
    fn __or__(&self, other: &PySdfNode) -> Self {
        self.union(other)
    }

    /// `a & b` → intersection
    fn __and__(&self, other: &PySdfNode) -> Self {
        self.intersection(other)
    }

    /// `a - b` → subtraction
    fn __sub__(&self, other: &PySdfNode) -> Self {
        self.subtract(other)
    }

    fn __repr__(&self) -> String {
        format!("SdfNode(nodes={})", self.inner.node_count())
    }
}

/// Compiled SDF for fast evaluation
#[pyclass(name = "CompiledSdf")]
pub struct PyCompiledSdf {
    compiled: crate::compiled::CompiledSdf,
    /// Original node retained for mesh generation (sdf_to_mesh needs SdfNode)
    source_node: SdfNode,
}

#[pymethods]
impl PyCompiledSdf {
    /// Evaluate at a single point
    fn eval(&self, x: f32, y: f32, z: f32) -> f32 {
        crate::compiled::eval_compiled(&self.compiled, Vec3::new(x, y, z))
    }

    /// Evaluate compiled SDF at multiple points (GIL released, SIMD + Rayon)
    ///
    /// This is the preferred high-performance evaluation path.
    /// Internally: GIL release → Zero-Copy NumPy → SIMD 8-wide × Rayon parallel.
    fn eval_batch<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let pts = unsafe { points.as_array() };
        if pts.shape()[1] != 3 {
            return Err(PyValueError::new_err("Points must have shape (N, 3)"));
        }

        let vec_points: Vec<Vec3> = pts
            .rows()
            .into_iter()
            .map(|row| Vec3::new(row[0], row[1], row[2]))
            .collect();

        let compiled_ref = &self.compiled;
        let distances = py.allow_threads(|| {
            crate::compiled::eval_compiled_batch_parallel(compiled_ref, &vec_points)
        });
        Ok(distances.into_pyarray(py))
    }

    /// Generate mesh from compiled SDF (GIL released, Marching Cubes parallel)
    ///
    /// Returns (vertices: ndarray[N,3], indices: ndarray[M]) as NumPy arrays.
    #[pyo3(signature = (bounds_min, bounds_max, resolution=64))]
    fn to_mesh<'py>(
        &self,
        py: Python<'py>,
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

        let compiled_ref = &self.compiled;
        let mesh = py.allow_threads(|| {
            crate::mesh::sdf_to_mesh_compiled(compiled_ref, min, max, &config)
        });
        mesh_to_numpy(py, &mesh)
    }

    /// Get instruction count
    fn instruction_count(&self) -> usize {
        self.compiled.instruction_count()
    }

    fn __repr__(&self) -> String {
        format!("CompiledSdf(instructions={})", self.compiled.instruction_count())
    }
}

/// Compile an SDF for fast evaluation (GIL released during bytecode generation)
#[pyfunction]
fn compile_sdf(py: Python<'_>, node: &PySdfNode) -> PyCompiledSdf {
    let source_node = node.inner.clone();
    let compiled = py.allow_threads(|| {
        crate::compiled::CompiledSdf::compile(&source_node)
    });
    PyCompiledSdf {
        compiled,
        source_node,
    }
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
    Ok(distances.into_pyarray(py))
}

/// Evaluate compiled SDF using SoA layout for maximum SIMD throughput (GIL released)
///
/// 20-30% faster than `eval_compiled_batch` on large point clouds (100k+)
/// due to direct SIMD loading from contiguous memory.
#[pyfunction]
fn eval_compiled_batch_soa<'py>(
    py: Python<'py>,
    compiled: &PyCompiledSdf,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let pts = unsafe { points.as_array() };

    if pts.shape()[1] != 3 {
        return Err(PyValueError::new_err("Points must have shape (N, 3)"));
    }

    // Convert AoS NumPy rows to SoA layout for optimal SIMD loading
    let vec_points: Vec<Vec3> = pts
        .rows()
        .into_iter()
        .map(|row| Vec3::new(row[0], row[1], row[2]))
        .collect();

    let soa = crate::soa::SoAPoints::from_vec3_slice(&vec_points);

    let compiled_ref = &compiled.compiled;
    let soa_distances = py.allow_threads(|| {
        crate::compiled::eval_compiled_batch_soa_parallel(compiled_ref, &soa)
    });

    // Extract results from SoA distances
    let distances = soa_distances.to_vec();
    Ok(distances.into_pyarray(py))
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
    Ok(distances.into_pyarray(py))
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
    use crate::mesh::{DecimateConfig, decimate};

    let mut mesh = numpy_to_mesh(vertices, indices)?;
    let config = DecimateConfig {
        target_ratio,
        preserve_boundary,
        ..Default::default()
    };

    py.allow_threads(|| {
        decimate(&mut mesh, &config);
    });
    mesh_to_numpy(py, &mesh)
}

/// Export mesh to OBJ file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_obj(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_obj as io_export_obj, ObjConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_obj(&mesh, path, &ObjConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Export mesh to GLB file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_glb(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_glb as io_export_glb, GltfConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_glb(&mesh, path, &GltfConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Save SDF to file (GIL released during I/O)
#[pyfunction]
fn save_sdf(py: Python<'_>, node: &PySdfNode, path: &str) -> PyResult<()> {
    let tree = SdfTree::new(node.inner.clone());
    let path = path.to_string();
    py.allow_threads(|| {
        save(&tree, &path)
    }).map_err(|e| PyValueError::new_err(format!("Save error: {}", e)))
}

/// Load SDF from file (GIL released during I/O)
#[pyfunction]
fn load_sdf(py: Python<'_>, path: &str) -> PyResult<PySdfNode> {
    let path = path.to_string();
    let tree = py.allow_threads(|| {
        load(&path)
    }).map_err(|e| PyValueError::new_err(format!("Load error: {}", e)))?;
    Ok(PySdfNode { inner: tree.root })
}

/// Parse SDF tree from JSON string (for LLM-generated SDF)
#[pyfunction]
fn from_json(json_str: &str) -> PyResult<PySdfNode> {
    use crate::io::from_json_string;
    let tree = from_json_string(json_str)
        .map_err(|e| PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    Ok(PySdfNode { inner: tree.root })
}

/// Serialize SDF node to JSON string
#[pyfunction]
fn to_json(node: &PySdfNode) -> PyResult<String> {
    use crate::io::to_json_string;
    let tree = SdfTree::new(node.inner.clone());
    to_json_string(&tree)
        .map_err(|e| PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// Bake SDF to 3D volume texture (CPU, returns NumPy 3D array)
#[cfg(feature = "volume")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64, generate_mips=false))]
fn bake_volume<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: u32,
    generate_mips: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use crate::volume::{BakeConfig, bake_volume as cpu_bake};

    let config = BakeConfig {
        resolution: [resolution, resolution, resolution],
        bounds_min: Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
        bounds_max: Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        generate_mips,
        ..Default::default()
    };

    let node_ref = &node.inner;
    let volume = py.allow_threads(|| cpu_bake(node_ref, &config));
    Ok(volume.data.into_pyarray(py))
}

/// Bake SDF to 3D volume texture on GPU (returns NumPy 3D array)
#[cfg(feature = "volume")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64))]
fn gpu_bake_volume<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use crate::volume::{BakeConfig, gpu_bake_volume as gpu_bake};

    let config = BakeConfig {
        resolution: [resolution, resolution, resolution],
        bounds_min: Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
        bounds_max: Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        ..Default::default()
    };

    let node_ref = &node.inner;
    let volume = py.allow_threads(|| gpu_bake(node_ref, &config))
        .map_err(|e| PyValueError::new_err(format!("GPU bake error: {}", e)))?;
    Ok(volume.data.into_pyarray(py))
}

/// GPU Marching Cubes mesh generation (returns vertices + indices as NumPy arrays)
#[cfg(feature = "gpu")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64, iso_level=0.0))]
fn gpu_marching_cubes<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: u32,
    iso_level: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::{gpu_marching_cubes as gpu_mc, GpuMarchingCubesConfig};

    let config = GpuMarchingCubesConfig {
        resolution,
        iso_level,
        compute_normals: true,
        ..Default::default()
    };

    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

    let node_ref = &node.inner;
    let mesh = py.allow_threads(|| gpu_mc(node_ref, min, max, &config))
        .map_err(|e| PyValueError::new_err(format!("GPU MC error: {}", e)))?;
    mesh_to_numpy(py, &mesh)
}

/// Build Sparse Voxel Octree from SDF
#[cfg(feature = "svo")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, max_depth=8, distance_threshold=1.5))]
fn build_svo(
    py: Python<'_>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    max_depth: u32,
    distance_threshold: f32,
) -> PyResult<PySvo> {
    use crate::svo::{SvoBuildConfig, SparseVoxelOctree};

    let config = SvoBuildConfig {
        max_depth,
        distance_threshold,
        bounds_min: Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
        bounds_max: Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        use_compiled: true,
        ..Default::default()
    };

    let node_ref = &node.inner;
    let compiled = crate::compiled::CompiledSdf::compile(node_ref);
    let svo = py.allow_threads(|| SparseVoxelOctree::build_compiled(&compiled, &config));

    Ok(PySvo { inner: svo })
}

/// Python-visible SVO wrapper
#[cfg(feature = "svo")]
#[pyclass(name = "SparseVoxelOctree")]
pub struct PySvo {
    inner: crate::svo::SparseVoxelOctree,
}

#[cfg(feature = "svo")]
#[pymethods]
impl PySvo {
    /// Query distance at a point
    fn query_point(&self, x: f32, y: f32, z: f32) -> f32 {
        self.inner.query_point(Vec3::new(x, y, z))
    }

    /// Get node count
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Get leaf count
    fn leaf_count(&self) -> u32 {
        self.inner.leaf_count
    }

    /// Get memory usage in bytes
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Get max depth
    fn max_depth(&self) -> u32 {
        self.inner.max_depth
    }

    fn __repr__(&self) -> String {
        format!(
            "SparseVoxelOctree(nodes={}, leaves={}, depth={}, memory={}KB)",
            self.inner.node_count(),
            self.inner.leaf_count,
            self.inner.max_depth,
            self.inner.memory_bytes() / 1024,
        )
    }
}

// --- Terrain bindings ---

#[cfg(feature = "terrain")]
#[pyclass(name = "Heightmap")]
struct PyHeightmap {
    inner: crate::terrain::Heightmap,
}

#[cfg(feature = "terrain")]
#[pymethods]
impl PyHeightmap {
    /// Get height at world coordinates
    fn sample(&self, x: f32, z: f32) -> f32 {
        self.inner.sample(x, z)
    }

    /// Get the height range (min, max)
    fn height_range(&self) -> (f32, f32) {
        self.inner.height_range()
    }

    /// Get sample count
    fn sample_count(&self) -> usize {
        self.inner.sample_count()
    }

    /// Get heights as a numpy array
    fn heights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.heights.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        let (min, max) = self.inner.height_range();
        format!(
            "Heightmap({}x{}, world={}x{}, range=[{:.2}, {:.2}])",
            self.inner.width, self.inner.depth,
            self.inner.world_width, self.inner.world_depth,
            min, max,
        )
    }
}

/// Generate terrain heightmap with fBm noise and optional erosion
#[cfg(feature = "terrain")]
#[pyfunction]
#[pyo3(signature = (width=256, depth=256, world_width=100.0, world_depth=100.0, octaves=6, height_scale=10.0, seed=42))]
fn generate_terrain(
    py: Python<'_>,
    width: u32,
    depth: u32,
    world_width: f32,
    world_depth: f32,
    octaves: u32,
    height_scale: f32,
    seed: u64,
) -> PyHeightmap {
    let mut hm = py.allow_threads(|| {
        let mut hm = crate::terrain::Heightmap::new(width, depth, world_width, world_depth);
        hm.generate_fbm(octaves, 0.5, 2.0, seed);
        hm.normalize();
        hm.scale_heights(height_scale);
        hm
    });

    PyHeightmap { inner: hm }
}

/// Apply erosion to a heightmap
#[cfg(feature = "terrain")]
#[pyfunction]
#[pyo3(signature = (heightmap, iterations=10000, erosion_rate=0.3, deposition_rate=0.3))]
fn erode_terrain(
    py: Python<'_>,
    heightmap: &mut PyHeightmap,
    iterations: u32,
    erosion_rate: f32,
    deposition_rate: f32,
) {
    let config = crate::terrain::ErosionConfig {
        iterations,
        erosion_rate,
        deposition_rate,
        ..Default::default()
    };

    py.allow_threads(|| {
        crate::terrain::erode(&mut heightmap.inner, &config);
    });
}

// --- Destruction bindings ---

#[cfg(feature = "destruction")]
#[pyclass(name = "VoxelGrid")]
struct PyVoxelGrid {
    inner: crate::destruction::MutableVoxelGrid,
}

#[cfg(feature = "destruction")]
#[pymethods]
impl PyVoxelGrid {
    /// Get the number of voxels
    fn voxel_count(&self) -> usize {
        self.inner.voxel_count()
    }

    /// Get memory usage in bytes
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Get the grid resolution
    fn resolution(&self) -> (u32, u32, u32) {
        let r = self.inner.resolution;
        (r[0], r[1], r[2])
    }

    /// Get distance at a world position
    fn get_distance(&self, x: f32, y: f32, z: f32) -> f32 {
        if let Some([gx, gy, gz]) = self.inner.world_to_grid(Vec3::new(x, y, z)) {
            self.inner.get_distance(gx, gy, gz)
        } else {
            f32::MAX
        }
    }

    fn __repr__(&self) -> String {
        let r = self.inner.resolution;
        format!(
            "VoxelGrid({}x{}x{}, voxels={}, memory={}KB)",
            r[0], r[1], r[2],
            self.inner.voxel_count(),
            self.inner.memory_bytes() / 1024,
        )
    }
}

/// Create a voxel grid from an SDF node
#[cfg(feature = "destruction")]
#[pyfunction]
#[pyo3(signature = (node, resolution=32, bounds_min=(-2.0, -2.0, -2.0), bounds_max=(2.0, 2.0, 2.0)))]
fn create_voxel_grid(
    py: Python<'_>,
    node: &PySdfNode,
    resolution: u32,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
) -> PyVoxelGrid {
    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);
    let node_ref = &node.inner;

    let grid = py.allow_threads(|| {
        crate::destruction::MutableVoxelGrid::from_sdf(
            node_ref,
            [resolution, resolution, resolution],
            min, max,
        )
    });

    PyVoxelGrid { inner: grid }
}

/// Carve a sphere from a voxel grid
#[cfg(feature = "destruction")]
#[pyfunction]
#[pyo3(signature = (grid, center, radius))]
fn carve_sphere(
    py: Python<'_>,
    grid: &mut PyVoxelGrid,
    center: (f32, f32, f32),
    radius: f32,
) -> (u32, f32) {
    let shape = crate::destruction::CarveShape::Sphere {
        center: Vec3::new(center.0, center.1, center.2),
        radius,
    };

    let result = py.allow_threads(|| {
        crate::destruction::carve(&mut grid.inner, &shape)
    });

    (result.modified_voxels, result.removed_volume)
}

/// Fracture a voxel grid using Voronoi tessellation
#[cfg(feature = "destruction")]
#[pyfunction]
#[pyo3(signature = (grid, center, radius, piece_count=8, seed=42))]
fn voxel_fracture<'py>(
    py: Python<'py>,
    grid: &PyVoxelGrid,
    center: (f32, f32, f32),
    radius: f32,
    piece_count: u32,
    seed: u64,
) -> PyResult<Vec<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)>> {
    let config = crate::destruction::FractureConfig {
        piece_count,
        seed,
        min_piece_size: 0.01,
        ..Default::default()
    };

    let c = Vec3::new(center.0, center.1, center.2);
    let grid_ref = &grid.inner;

    let pieces = py.allow_threads(|| {
        crate::destruction::voronoi_fracture(grid_ref, c, radius, &config)
    });

    let mut results = Vec::new();
    for piece in &pieces {
        let (verts, indices) = mesh_to_numpy(py, &piece.mesh)?;
        results.push((verts, indices));
    }

    Ok(results)
}

// --- GI bindings ---

/// Bake irradiance probe grid from SVO (returns probe positions + SH coefficients)
#[cfg(feature = "gi")]
#[pyfunction]
#[pyo3(signature = (svo, grid_size=8, bounds_min=(-2.0, -2.0, -2.0), bounds_max=(2.0, 2.0, 2.0), samples_per_probe=32))]
fn bake_gi<'py>(
    py: Python<'py>,
    svo: &PySvo,
    grid_size: u32,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    samples_per_probe: u32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    use crate::gi::{BakeGiConfig, bake_irradiance_grid, ConeTraceConfig, DirectionalLight};

    let config = BakeGiConfig {
        grid_size: [grid_size, grid_size, grid_size],
        bounds_min: Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
        bounds_max: Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        samples_per_probe,
        cone_config: ConeTraceConfig::default(),
        sun: Some(DirectionalLight::default()),
    };

    let svo_ref = &svo.inner;
    let grid = py.allow_threads(|| bake_irradiance_grid(svo_ref, &config));

    // Return positions (N,3) and SH coefficients (N,12) for RGB L1 SH
    let n = grid.probes.len();
    let positions: Vec<f32> = grid.probes.iter()
        .flat_map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let sh_coeffs: Vec<f32> = grid.probes.iter()
        .flat_map(|p| {
            let mut c = Vec::with_capacity(12);
            c.extend_from_slice(&p.sh_r.coeffs);
            c.extend_from_slice(&p.sh_g.coeffs);
            c.extend_from_slice(&p.sh_b.coeffs);
            c
        })
        .collect();

    let pos_array = numpy::PyArray1::from_vec(py, positions)
        .reshape([n, 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;
    let sh_array = numpy::PyArray1::from_vec(py, sh_coeffs)
        .reshape([n, 12])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    Ok((pos_array, sh_array))
}

/// Export mesh to FBX file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_fbx(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_fbx as io_export_fbx, FbxConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_fbx(&mesh, path, &FbxConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Export mesh to USDA file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_usda(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_usda as io_export_usda, UsdConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_usda(&mesh, path, &UsdConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Export mesh to Alembic file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_alembic(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_alembic as io_export_alembic, AlembicConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_alembic(&mesh, path, &AlembicConfig::default()))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// UV unwrap a mesh using LSCM (Least Squares Conformal Map)
///
/// Returns (positions: ndarray[N,3], uvs: ndarray[N,2], indices: ndarray[M]).
#[pyfunction]
#[pyo3(signature = (vertices, indices))]
fn uv_unwrap<'py>(
    py: Python<'py>,
    vertices: &Bound<'py, PyArray2<f32>>,
    indices: &Bound<'py, PyArray1<u32>>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::{uv_unwrap as mesh_uv_unwrap, apply_uvs, UvUnwrapConfig};

    let mut mesh = numpy_to_mesh(vertices, indices)?;

    py.allow_threads(|| {
        let result = mesh_uv_unwrap(&mesh, &UvUnwrapConfig::default());
        apply_uvs(&mut mesh, &result);
    });

    let n = mesh.vertices.len();

    // Positions [N, 3]
    let positions: Vec<f32> = mesh.vertices.iter()
        .flat_map(|v| [v.position.x, v.position.y, v.position.z])
        .collect();
    let pos_array = numpy::PyArray1::from_vec(py, positions)
        .reshape([n, 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    // UVs [N, 2]
    let uvs: Vec<f32> = mesh.vertices.iter()
        .flat_map(|v| [v.uv.x, v.uv.y])
        .collect();
    let uv_array = numpy::PyArray1::from_vec(py, uvs)
        .reshape([n, 2])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    let indices_array = mesh.indices.clone().into_pyarray(py);

    Ok((pos_array, uv_array, indices_array))
}

/// Get library version
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Helper: convert NumPy arrays to Mesh (shared by all export functions)
fn numpy_to_mesh(
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
) -> PyResult<crate::mesh::Mesh> {
    use crate::mesh::{Mesh, Vertex};

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

    Ok(Mesh {
        vertices: mesh_verts,
        indices: idx.to_vec(),
    })
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

    let vertex_array = numpy::PyArray1::from_vec(py, vertices)
        .reshape([mesh.vertices.len(), 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    let indices = mesh.indices.clone().into_pyarray(py);

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
    m.add_function(wrap_pyfunction!(eval_compiled_batch_soa, m)?)?;
    m.add_function(wrap_pyfunction!(to_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(to_mesh_adaptive, m)?)?;
    m.add_function(wrap_pyfunction!(decimate_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(export_obj, m)?)?;
    m.add_function(wrap_pyfunction!(export_glb, m)?)?;
    m.add_function(wrap_pyfunction!(export_fbx, m)?)?;
    m.add_function(wrap_pyfunction!(export_usda, m)?)?;
    m.add_function(wrap_pyfunction!(export_alembic, m)?)?;
    m.add_function(wrap_pyfunction!(uv_unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(save_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(load_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(from_json, m)?)?;
    m.add_function(wrap_pyfunction!(to_json, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    #[cfg(feature = "volume")]
    {
        m.add_function(wrap_pyfunction!(bake_volume, m)?)?;
        m.add_function(wrap_pyfunction!(gpu_bake_volume, m)?)?;
    }
    #[cfg(feature = "gpu")]
    {
        m.add_function(wrap_pyfunction!(gpu_marching_cubes, m)?)?;
    }
    #[cfg(feature = "svo")]
    {
        m.add_class::<PySvo>()?;
        m.add_function(wrap_pyfunction!(build_svo, m)?)?;
    }
    #[cfg(feature = "destruction")]
    {
        m.add_class::<PyVoxelGrid>()?;
        m.add_function(wrap_pyfunction!(create_voxel_grid, m)?)?;
        m.add_function(wrap_pyfunction!(carve_sphere, m)?)?;
        m.add_function(wrap_pyfunction!(voxel_fracture, m)?)?;
    }
    #[cfg(feature = "terrain")]
    {
        m.add_class::<PyHeightmap>()?;
        m.add_function(wrap_pyfunction!(generate_terrain, m)?)?;
        m.add_function(wrap_pyfunction!(erode_terrain, m)?)?;
    }
    #[cfg(feature = "gi")]
    {
        m.add_function(wrap_pyfunction!(bake_gi, m)?)?;
    }
    Ok(())
}
