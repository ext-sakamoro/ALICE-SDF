//! I/O #[pyfunction]s: save/load SDF, JSON, export to OBJ/GLB/FBX/USDA/Alembic/ABM/Unity/UE5, UV unwrap.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::io::{load, save};
use crate::types::SdfTree;

use super::helpers::{mesh_to_numpy, numpy_to_mesh};
use super::node::PySdfNode;

/// Save SDF to file (GIL released during I/O)
#[pyfunction]
pub fn save_sdf(py: Python<'_>, node: &PySdfNode, path: &str) -> PyResult<()> {
    let tree = SdfTree::new(node.inner.clone());
    let path = path.to_string();
    py.allow_threads(|| save(&tree, &path))
        .map_err(|e| PyValueError::new_err(format!("Save error: {}", e)))
}

/// Load SDF from file (GIL released during I/O)
#[pyfunction]
pub fn load_sdf(py: Python<'_>, path: &str) -> PyResult<PySdfNode> {
    let path = path.to_string();
    let tree = py
        .allow_threads(|| load(&path))
        .map_err(|e| PyValueError::new_err(format!("Load error: {}", e)))?;
    Ok(PySdfNode { inner: tree.root })
}

/// Parse SDF tree from JSON string (for LLM-generated SDF)
#[pyfunction]
pub fn from_json(json_str: &str) -> PyResult<PySdfNode> {
    use crate::io::from_json_string;
    let tree = from_json_string(json_str)
        .map_err(|e| PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    Ok(PySdfNode { inner: tree.root })
}

/// Serialize SDF node to JSON string
#[pyfunction]
pub fn to_json(node: &PySdfNode) -> PyResult<String> {
    use crate::io::to_json_string;
    let tree = SdfTree::new(node.inner.clone());
    to_json_string(&tree).map_err(|e| PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// Export mesh to OBJ file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
pub fn export_obj(
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
pub fn export_glb(
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

/// Export mesh to GLB bytes in memory (no temp file I/O)
#[pyfunction]
#[pyo3(signature = (vertices, indices))]
pub fn export_glb_bytes<'py>(
    py: Python<'py>,
    vertices: &Bound<'py, PyArray2<f32>>,
    indices: &Bound<'py, PyArray1<u32>>,
) -> PyResult<Vec<u8>> {
    use crate::io::{export_glb_bytes as io_export_glb_bytes, GltfConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_glb_bytes(&mesh, &GltfConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Export mesh to FBX file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
pub fn export_fbx(
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
pub fn export_usda(
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
pub fn export_alembic(
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
pub fn uv_unwrap<'py>(
    py: Python<'py>,
    vertices: &Bound<'py, PyArray2<f32>>,
    indices: &Bound<'py, PyArray1<u32>>,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    use crate::mesh::{apply_uvs, uv_unwrap as mesh_uv_unwrap, UvUnwrapConfig};

    let mut mesh = numpy_to_mesh(vertices, indices)?;

    py.allow_threads(|| {
        let result = mesh_uv_unwrap(&mesh, &UvUnwrapConfig::default());
        apply_uvs(&mut mesh, &result);
    });

    let n = mesh.vertices.len();

    // Positions [N, 3]
    let positions: Vec<f32> = mesh
        .vertices
        .iter()
        .flat_map(|v| [v.position.x, v.position.y, v.position.z])
        .collect();
    let pos_array = numpy::PyArray1::from_vec(py, positions)
        .reshape([n, 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    // UVs [N, 2]
    let uvs: Vec<f32> = mesh
        .vertices
        .iter()
        .flat_map(|v| [v.uv.x, v.uv.y])
        .collect();
    let uv_array = numpy::PyArray1::from_vec(py, uvs)
        .reshape([n, 2])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    let indices_array = numpy::PyArray1::from_slice(py, &mesh.indices);

    Ok((pos_array, uv_array, indices_array))
}

/// Save mesh to ALICE Binary Mesh format (GIL released during I/O)
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
pub fn save_abm(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::save_abm as io_save_abm;
    let mesh = numpy_to_mesh(vertices, indices)?;
    let path = path.to_string();
    py.allow_threads(move || io_save_abm(&mesh, &path))
        .map_err(|e| PyValueError::new_err(format!("ABM save error: {}", e)))
}

/// Load mesh from ALICE Binary Mesh format (GIL released, Zero-Copy return)
#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_abm<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::io::load_abm as io_load_abm;
    let path = path.to_string();
    let mesh = py
        .allow_threads(move || io_load_abm(&path))
        .map_err(|e| PyValueError::new_err(format!("ABM load error: {}", e)))?;
    mesh_to_numpy(py, &mesh)
}

/// Export mesh to Unity JSON format (GIL released during I/O)
#[pyfunction]
#[pyo3(signature = (vertices, indices, path, flip_z=true, flip_winding=true, scale=1.0))]
pub fn export_unity(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
    flip_z: bool,
    flip_winding: bool,
    scale: f32,
) -> PyResult<()> {
    use crate::io::{export_unity_mesh as io_export_unity, UnityMeshConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    let path = path.to_string();
    let config = UnityMeshConfig {
        scale,
        flip_z,
        flip_winding,
        name: "AliceSdfMesh".to_string(),
    };
    py.allow_threads(move || io_export_unity(&mesh, &path, &config))
        .map_err(|e| PyValueError::new_err(format!("Unity export error: {}", e)))
}

/// Export mesh to UE5 JSON format (GIL released during I/O)
#[pyfunction]
#[pyo3(signature = (vertices, indices, path, scale=100.0))]
pub fn export_ue5(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
    scale: f32,
) -> PyResult<()> {
    use crate::io::{export_ue5_mesh as io_export_ue5, Ue5MeshConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    let path = path.to_string();
    let config = Ue5MeshConfig {
        scale,
        name: "SM_AliceSdf".to_string(),
        lod_index: 0,
    };
    py.allow_threads(move || io_export_ue5(&mesh, &path, &config))
        .map_err(|e| PyValueError::new_err(format!("UE5 export error: {}", e)))
}
