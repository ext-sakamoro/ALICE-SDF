//! Shared helper functions for NumPy <-> Rust data conversion.

use glam::Vec3;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Helper: convert NumPy arrays to Mesh (shared by all export functions)
#[inline]
pub fn numpy_to_mesh(
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
) -> PyResult<crate::mesh::Mesh> {
    use crate::mesh::{Mesh, Vertex};

    // SAFETY: PyO3 guarantees the NumPy array object is valid for the GIL lifetime.
    // The returned ndarray view borrows from the Python object and is valid while
    // the GIL is held (which it is, as we have `py: Python<'_>` in scope via the Bound reference).
    let verts = unsafe { vertices.as_array() };
    // SAFETY: PyO3 guarantees the NumPy array object is valid for the GIL lifetime.
    // The returned ndarray view borrows from the Python object and is valid while
    // the GIL is held (which it is, as we have `py: Python<'_>` in scope via the Bound reference).
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
#[inline]
pub fn mesh_to_numpy<'py>(
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

    let indices = numpy::PyArray1::from_slice(py, &mesh.indices);

    Ok((vertex_array, indices))
}

/// Helper: fast conversion from C-contiguous NumPy (N,3) f32 to Vec<Vec3>.
///
/// Uses raw pointer reinterpret when contiguous (zero-copy slice),
/// falls back to row iteration otherwise.
#[inline]
#[allow(dead_code)]
pub fn numpy_to_vec3_fast(points: &Bound<'_, PyArray2<f32>>) -> PyResult<Vec<Vec3>> {
    // SAFETY: PyO3 guarantees the NumPy array object is valid for the GIL lifetime.
    // The returned ndarray view borrows from the Python object and is valid while
    // the GIL is held (which it is, as we have `py: Python<'_>` in scope via the Bound reference).
    let arr = unsafe { points.as_array() };
    if arr.shape()[1] != 3 {
        return Err(PyValueError::new_err("Points must have shape (N, 3)"));
    }
    let n = arr.shape()[0];
    if let Some(slice) = arr.as_slice() {
        // SAFETY: We verified the array is contiguous C-order via as_slice().
        // glam::Vec3 is repr(C) with layout [f32; 3], matching NumPy's f32 stride.
        // The slice length `n` equals arr.shape()[0], which is within the array bounds.
        let vec3_slice = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Vec3, n) };
        Ok(vec3_slice.to_vec())
    } else {
        // Non-contiguous fallback: row iteration
        Ok(arr
            .rows()
            .into_iter()
            .map(|row| Vec3::new(row[0], row[1], row[2]))
            .collect())
    }
}

/// Zero-copy callback on NumPy data, eliminating Vec allocation.
///
/// Reinterprets contiguous NumPy (N,3) f32 as &[Vec3] slice and passes
/// it directly to the closure. No allocation in the contiguous path.
#[inline]
pub fn with_numpy_as_vec3<R>(
    points: &Bound<'_, PyArray2<f32>>,
    f: impl FnOnce(&[Vec3]) -> R,
) -> PyResult<R> {
    // SAFETY: PyO3 guarantees the NumPy array object is valid for the GIL lifetime.
    // The returned ndarray view borrows from the Python object and is valid while
    // the GIL is held (which it is, as we have `py: Python<'_>` in scope via the Bound reference).
    let arr = unsafe { points.as_array() };
    if arr.shape()[1] != 3 {
        return Err(PyValueError::new_err("Points must have shape (N, 3)"));
    }
    let n = arr.shape()[0];
    if let Some(slice) = arr.as_slice() {
        // SAFETY: We verified the array is contiguous C-order via as_slice().
        // glam::Vec3 is repr(C) with layout [f32; 3], matching NumPy's f32 stride.
        // The slice length `n` equals arr.shape()[0], which is within the array bounds.
        let vec3_slice = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Vec3, n) };
        Ok(f(vec3_slice))
    } else {
        // Non-contiguous fallback: copy then process
        let vec: Vec<Vec3> = arr
            .rows()
            .into_iter()
            .map(|row| Vec3::new(row[0], row[1], row[2]))
            .collect();
        Ok(f(&vec))
    }
}
