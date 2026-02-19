//! PyCompiledSdf class and methods.

use glam::Vec3;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::mesh::MarchingCubesConfig;
use crate::types::SdfNode;

use super::helpers::{mesh_to_numpy, with_numpy_as_vec3};

/// Compiled SDF for fast evaluation
#[pyclass(name = "CompiledSdf")]
pub struct PyCompiledSdf {
    pub(crate) compiled: crate::compiled::CompiledSdf,
    /// Original node retained for mesh generation (sdf_to_mesh needs SdfNode)
    pub(crate) source_node: SdfNode,
}

#[pymethods]
impl PyCompiledSdf {
    /// Evaluate at a single point (GIL released)
    fn eval(&self, py: Python<'_>, x: f32, y: f32, z: f32) -> f32 {
        let compiled = &self.compiled;
        py.allow_threads(|| crate::compiled::eval_compiled(compiled, Vec3::new(x, y, z)))
    }

    /// Evaluate compiled SDF at multiple points (GIL released, SIMD + Rayon)
    ///
    /// This is the preferred high-performance evaluation path.
    /// Internally: GIL release -> Zero-Copy NumPy -> SIMD 8-wide x Rayon parallel.
    fn eval_batch<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let compiled_ref = &self.compiled;
        let distances = with_numpy_as_vec3(points, |pts| {
            py.allow_threads(|| crate::compiled::eval_compiled_batch_parallel(compiled_ref, pts))
        })?;
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
        let mesh =
            py.allow_threads(|| crate::mesh::sdf_to_mesh_compiled(compiled_ref, min, max, &config));
        mesh_to_numpy(py, &mesh)
    }

    /// Get instruction count
    fn instruction_count(&self) -> usize {
        self.compiled.instruction_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompiledSdf(instructions={})",
            self.compiled.instruction_count()
        )
    }
}
