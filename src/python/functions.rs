//! Module-level #[pyfunction]s: compile, eval, mesh generation, volume, SVO, terrain, GI, etc.

use glam::Vec3;
#[allow(unused_imports)]
use numpy::PyArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyArray2};
#[allow(unused_imports)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eval::eval_batch_parallel;
use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};

use super::compiled::PyCompiledSdf;
use super::helpers::{mesh_to_numpy, numpy_to_mesh, with_numpy_as_vec3};
use super::node::PySdfNode;

/// Compile an SDF for fast evaluation (GIL released during bytecode generation)
#[pyfunction]
pub fn compile_sdf(py: Python<'_>, node: &PySdfNode) -> PyCompiledSdf {
    let source_node = node.inner.clone();
    let compiled = py.allow_threads(|| crate::compiled::CompiledSdf::compile(&source_node));
    PyCompiledSdf {
        compiled,
        source_node,
    }
}

/// Evaluate compiled SDF at multiple points (NumPy array, GIL released)
#[pyfunction]
pub fn eval_compiled_batch<'py>(
    py: Python<'py>,
    compiled: &PyCompiledSdf,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let compiled_ref = &compiled.compiled;
    let distances = with_numpy_as_vec3(points, |pts| {
        py.allow_threads(|| crate::compiled::eval_compiled_batch_parallel(compiled_ref, pts))
    })?;
    Ok(distances.into_pyarray(py))
}

/// Evaluate compiled SDF using SoA layout for maximum SIMD throughput (GIL released)
///
/// 20-30% faster than `eval_compiled_batch` on large point clouds (100k+)
/// due to direct SIMD loading from contiguous memory.
#[pyfunction]
pub fn eval_compiled_batch_soa<'py>(
    py: Python<'py>,
    compiled: &PyCompiledSdf,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let compiled_ref = &compiled.compiled;
    let distances = with_numpy_as_vec3(points, |pts| {
        let soa = crate::soa::SoAPoints::from_vec3_slice(pts);
        py.allow_threads(|| {
            crate::compiled::eval_compiled_batch_soa_parallel(compiled_ref, &soa).to_vec()
        })
    })?;
    Ok(distances.into_pyarray(py))
}

/// Evaluate SDF at multiple points (NumPy array, GIL released)
#[pyfunction]
pub fn eval_batch<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let node_ref = &node.inner;
    let distances = with_numpy_as_vec3(points, |pts| {
        py.allow_threads(|| eval_batch_parallel(node_ref, pts))
    })?;
    Ok(distances.into_pyarray(py))
}

/// Convert SDF to mesh (standard marching cubes)
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64))]
pub fn to_mesh<'py>(
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
    let mesh = py.allow_threads(|| sdf_to_mesh(node_ref, min, max, &config));
    mesh_to_numpy(py, &mesh)
}

/// Convert SDF to mesh using adaptive marching cubes
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, max_depth=6, min_depth=2, surface_threshold=1.0))]
pub fn to_mesh_adaptive<'py>(
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
    let mesh =
        py.allow_threads(|| crate::mesh::adaptive_marching_cubes(node_ref, min, max, &config));
    mesh_to_numpy(py, &mesh)
}

/// Convert SDF to mesh using Dual Contouring (sharp edges, QEF vertex placement)
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64))]
pub fn to_mesh_dual_contouring<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::{dual_contouring, DualContouringConfig};

    let config = DualContouringConfig {
        resolution,
        compute_normals: true,
        ..Default::default()
    };

    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

    let node_ref = &node.inner;
    let mesh = py.allow_threads(|| dual_contouring(node_ref, min, max, &config));
    mesh_to_numpy(py, &mesh)
}

/// Decimate a mesh to reduce triangle count
#[pyfunction]
#[pyo3(signature = (vertices, indices, target_ratio=0.5, preserve_boundary=true))]
pub fn decimate_mesh<'py>(
    py: Python<'py>,
    vertices: &Bound<'py, PyArray2<f32>>,
    indices: &Bound<'py, PyArray1<u32>>,
    target_ratio: f32,
    preserve_boundary: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::{decimate, DecimateConfig};

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

/// Get library version
#[pyfunction]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Bake SDF to 3D volume texture (CPU, returns NumPy 3D array)
#[cfg(feature = "volume")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64, generate_mips=false))]
pub fn bake_volume<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: u32,
    generate_mips: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use crate::volume::{bake_volume as cpu_bake, BakeConfig};

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
pub fn gpu_bake_volume<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use crate::volume::{gpu_bake_volume as gpu_bake, BakeConfig};

    let config = BakeConfig {
        resolution: [resolution, resolution, resolution],
        bounds_min: Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
        bounds_max: Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        ..Default::default()
    };

    let node_ref = &node.inner;
    let volume = py
        .allow_threads(|| gpu_bake(node_ref, &config))
        .map_err(|e| PyValueError::new_err(format!("GPU bake error: {}", e)))?;
    Ok(volume.data.into_pyarray(py))
}

/// GPU Marching Cubes mesh generation (returns vertices + indices as NumPy arrays)
#[cfg(feature = "gpu")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64, iso_level=0.0))]
pub fn gpu_marching_cubes<'py>(
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
    let mesh = py
        .allow_threads(|| gpu_mc(node_ref, min, max, &config))
        .map_err(|e| PyValueError::new_err(format!("GPU MC error: {}", e)))?;
    mesh_to_numpy(py, &mesh)
}

/// Build Sparse Voxel Octree from SDF
#[cfg(feature = "svo")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, max_depth=8, distance_threshold=1.5))]
pub fn build_svo(
    py: Python<'_>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    max_depth: u32,
    distance_threshold: f32,
) -> PyResult<PySvo> {
    use crate::svo::{SparseVoxelOctree, SvoBuildConfig};

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
    pub(crate) inner: crate::svo::SparseVoxelOctree,
}

#[cfg(feature = "svo")]
#[pymethods]
impl PySvo {
    /// Query distance at a point (GIL released)
    fn query_point(&self, py: Python<'_>, x: f32, y: f32, z: f32) -> f32 {
        let inner = &self.inner;
        py.allow_threads(|| inner.query_point(Vec3::new(x, y, z)))
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

/// Query SVO distance at multiple points (GIL released, Rayon parallel)
#[cfg(feature = "svo")]
#[pyfunction]
#[pyo3(signature = (svo, points))]
pub fn svo_query_batch<'py>(
    py: Python<'py>,
    svo: &PySvo,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let svo_ref = &svo.inner;
    let results = with_numpy_as_vec3(points, |pts| {
        py.allow_threads(|| {
            use rayon::prelude::*;
            pts.par_iter()
                .map(|p| svo_ref.query_point(*p))
                .collect::<Vec<f32>>()
        })
    })?;
    Ok(results.into_pyarray(py))
}

// --- Terrain bindings ---

#[cfg(feature = "terrain")]
#[pyclass(name = "Heightmap")]
pub struct PyHeightmap {
    pub(crate) inner: crate::terrain::Heightmap,
}

#[cfg(feature = "terrain")]
#[pymethods]
impl PyHeightmap {
    /// Get height at world coordinates (GIL released)
    fn sample(&self, py: Python<'_>, x: f32, z: f32) -> f32 {
        let inner = &self.inner;
        py.allow_threads(|| inner.sample(x, z))
    }

    /// Get the height range (min, max)
    fn height_range(&self) -> (f32, f32) {
        self.inner.height_range()
    }

    /// Get sample count
    fn sample_count(&self) -> usize {
        self.inner.sample_count()
    }

    /// Get heights as a numpy array (avoids intermediate Vec clone)
    fn heights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        numpy::PyArray1::from_slice(py, &self.inner.heights)
    }

    fn __repr__(&self) -> String {
        let (min, max) = self.inner.height_range();
        format!(
            "Heightmap({}x{}, world={}x{}, range=[{:.2}, {:.2}])",
            self.inner.width,
            self.inner.depth,
            self.inner.world_width,
            self.inner.world_depth,
            min,
            max,
        )
    }
}

/// Generate terrain heightmap with fBm noise and optional erosion
#[cfg(feature = "terrain")]
#[pyfunction]
#[pyo3(signature = (width=256, depth=256, world_width=100.0, world_depth=100.0, octaves=6, height_scale=10.0, seed=42))]
pub fn generate_terrain(
    py: Python<'_>,
    width: u32,
    depth: u32,
    world_width: f32,
    world_depth: f32,
    octaves: u32,
    height_scale: f32,
    seed: u64,
) -> PyHeightmap {
    let hm = py.allow_threads(|| {
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
pub fn erode_terrain(
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

/// Sample heightmap at multiple world positions (GIL released, Rayon parallel)
#[cfg(feature = "terrain")]
#[pyfunction]
#[pyo3(signature = (heightmap, points))]
pub fn heightmap_sample_batch<'py>(
    py: Python<'py>,
    heightmap: &PyHeightmap,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    // SAFETY: PyO3 guarantees the NumPy array object is valid for the GIL lifetime.
    // The returned ndarray view borrows from the Python object and is valid while
    // the GIL is held (which it is, as we have `py: Python<'_>`).
    let pts = unsafe { points.as_array() };
    if pts.shape()[1] != 2 {
        return Err(PyValueError::new_err(
            "points must have shape (N, 2) for (x, z) pairs",
        ));
    }
    let vec_points: Vec<(f32, f32)> = pts.rows().into_iter().map(|row| (row[0], row[1])).collect();

    let hm_ref = &heightmap.inner;
    let results = py.allow_threads(|| {
        use rayon::prelude::*;
        vec_points
            .par_iter()
            .map(|&(x, z)| hm_ref.sample(x, z))
            .collect::<Vec<f32>>()
    });
    Ok(results.into_pyarray(py))
}

// --- Destruction bindings ---

#[cfg(feature = "destruction")]
#[pyclass(name = "VoxelGrid")]
pub struct PyVoxelGrid {
    pub(crate) inner: crate::destruction::MutableVoxelGrid,
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

    /// Get distance at a world position (GIL released)
    fn get_distance(&self, py: Python<'_>, x: f32, y: f32, z: f32) -> f32 {
        let inner = &self.inner;
        py.allow_threads(|| {
            if let Some([gx, gy, gz]) = inner.world_to_grid(Vec3::new(x, y, z)) {
                inner.get_distance(gx, gy, gz)
            } else {
                f32::MAX
            }
        })
    }

    fn __repr__(&self) -> String {
        let r = self.inner.resolution;
        format!(
            "VoxelGrid({}x{}x{}, voxels={}, memory={}KB)",
            r[0],
            r[1],
            r[2],
            self.inner.voxel_count(),
            self.inner.memory_bytes() / 1024,
        )
    }
}

/// Create a voxel grid from an SDF node
#[cfg(feature = "destruction")]
#[pyfunction]
#[pyo3(signature = (node, resolution=32, bounds_min=(-2.0, -2.0, -2.0), bounds_max=(2.0, 2.0, 2.0)))]
pub fn create_voxel_grid(
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
            min,
            max,
        )
    });

    PyVoxelGrid { inner: grid }
}

/// Carve a sphere from a voxel grid
#[cfg(feature = "destruction")]
#[pyfunction]
#[pyo3(signature = (grid, center, radius))]
pub fn carve_sphere(
    py: Python<'_>,
    grid: &mut PyVoxelGrid,
    center: (f32, f32, f32),
    radius: f32,
) -> (u32, f32) {
    let shape = crate::destruction::CarveShape::Sphere {
        center: Vec3::new(center.0, center.1, center.2),
        radius,
    };

    let result = py.allow_threads(|| crate::destruction::carve(&mut grid.inner, &shape));

    (result.modified_voxels, result.removed_volume)
}

/// Fracture a voxel grid using Voronoi tessellation
#[cfg(feature = "destruction")]
#[pyfunction]
#[pyo3(signature = (grid, center, radius, piece_count=8, seed=42))]
pub fn voxel_fracture<'py>(
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

    let pieces =
        py.allow_threads(|| crate::destruction::voronoi_fracture(grid_ref, c, radius, &config));

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
pub fn bake_gi<'py>(
    py: Python<'py>,
    svo: &PySvo,
    grid_size: u32,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    samples_per_probe: u32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    use crate::gi::{bake_irradiance_grid, BakeGiConfig, ConeTraceConfig, DirectionalLight};

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
    let positions: Vec<f32> = grid
        .probes
        .iter()
        .flat_map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let sh_coeffs: Vec<f32> = grid
        .probes
        .iter()
        .flat_map(|p| {
            let mut c = [0.0f32; 12];
            c[0..4].copy_from_slice(&p.sh_r.coeffs);
            c[4..8].copy_from_slice(&p.sh_g.coeffs);
            c[8..12].copy_from_slice(&p.sh_b.coeffs);
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

/// Chunked mesh cache with FIFO eviction (thread-safe, GIL-friendly)
#[pyclass]
pub struct PyMeshCache {
    inner: std::sync::Arc<crate::cache::ChunkedMeshCache>,
}

#[pymethods]
impl PyMeshCache {
    #[new]
    #[pyo3(signature = (max_chunks=256, chunk_size=1.0))]
    fn new(max_chunks: usize, chunk_size: f32) -> Self {
        use crate::cache::{ChunkedCacheConfig, ChunkedMeshCache};
        let config = ChunkedCacheConfig {
            max_cached_chunks: max_chunks,
            chunk_size,
            ..Default::default()
        };
        Self {
            inner: std::sync::Arc::new(ChunkedMeshCache::new(config)),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.chunk_count()
    }

    fn is_empty(&self) -> bool {
        self.inner.chunk_count() == 0
    }

    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    fn clear(&self) {
        self.inner.clear();
    }
}
