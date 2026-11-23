//! Sheet builder helpers: the exact nearest face ([`crate::sheet_field::nearest`]) for NumPy arrays.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn rows3<T: Copy + numpy::Element>(
    a: &Bound<'_, PyArray2<T>>,
    what: &str,
) -> PyResult<Vec<[T; 3]>> {
    let ro = a.readonly();
    let v = ro.as_array();
    if v.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{what} must have shape (N, 3), got {:?}",
            v.shape()
        )));
    }
    Ok(v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

/// `sheet_nearest_faces(V, F, P) -> (dist, face)`: for every row of `P` (float64, (n, 3)) the nearest face of the open mesh
/// `V` (float64, (m, 3)), `F` (int64, (k, 3)) and its distance, by the mesh's formula (plane-or-edge below 400 faces, Ericson from
/// 400), ties to the lowest face index. Raises ValueError on bad shapes, out-of-range or negative indices, or non-finite input.
#[pyfunction]
pub fn sheet_nearest_faces<'py>(
    py: Python<'py>,
    v: &Bound<'py, PyArray2<f64>>,
    f: &Bound<'py, PyArray2<i64>>,
    p: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let vv = rows3(v, "V")?;
    let qq = rows3(p, "P")?;
    let ff = rows3(f, "F")?
        .into_iter()
        .map(|t| {
            let c = |x: i64| {
                u32::try_from(x)
                    .map_err(|_| PyValueError::new_err(format!("face index {x} not in 0..2^32")))
            };
            Ok([c(t[0])?, c(t[1])?, c(t[2])?])
        })
        .collect::<PyResult<Vec<[u32; 3]>>>()?;
    let r = py
        .detach(|| crate::sheet_field::nearest::nearest_faces(&vv, &ff, &qq))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d: Vec<f64> = r.iter().map(|x| x.dist).collect();
    let k: Vec<i64> = r.iter().map(|x| i64::from(x.face)).collect();
    Ok((PyArray1::from_vec(py, d), PyArray1::from_vec(py, k)))
}
