//! Module-level Python helpers for DCC bindings (Maya / Cinema 4D / Nuke).
//!
//! These functions match the calling convention that
//! `bindings/{maya,cinema4d,nuke}/*/__init__.py` expect, so that the DCC
//! plugin scripts work against the crates.io Python wheel without any manual
//! shim code.
//!
//! # Exposed API (module-level)
//!
//! - [`sphere`], [`make_box`] (Python name `box`), [`torus`] — primitive helpers
//! - [`sdf_to_mesh`] — marching-cubes mesh as Python `(list, list)` pairs
//! - [`load_asdf`] — `.asdf` / `.asdf.json` loader (thin alias over
//!   [`crate::io::load`]) for DCC binding compatibility
//! - [`bake_to_vdb`] — dense voxel bake returning `bytes` in `ALICEVDB1` format
//! - [`render_slice_2d`] — 2D `(H, W, 4)` RGBA uint8 numpy slice preview
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "volume")]
use pyo3::types::PyBytes;
use pyo3::types::{PyList, PyTuple};

use super::node::PySdfNode;
use crate::eval::eval_batch_parallel;
use crate::io::load;
use crate::mesh::{sdf_to_mesh as core_sdf_to_mesh, MarchingCubesConfig};
use crate::types::SdfNode;

/// Module-level sphere constructor (`backend.sphere(radius)`).
#[pyfunction]
pub fn sphere(radius: f32) -> PySdfNode {
    PySdfNode {
        inner: SdfNode::sphere(radius),
    }
}

/// Module-level box constructor (`backend.box(sx, sy, sz)`).
///
/// Exposed to Python under the name `box`; the Rust identifier is
/// [`make_box`] because `box` is a reserved keyword in Rust 2024.
#[pyfunction]
#[pyo3(name = "box")]
pub fn make_box(sx: f32, sy: f32, sz: f32) -> PySdfNode {
    PySdfNode {
        inner: SdfNode::box3d(sx, sy, sz),
    }
}

/// Module-level torus constructor (`backend.torus(R, r)`).
#[pyfunction]
pub fn torus(major_radius: f32, minor_radius: f32) -> PySdfNode {
    PySdfNode {
        inner: SdfNode::torus(major_radius, minor_radius),
    }
}

/// Convert an SDF tree to a Python-list mesh usable by Maya `MFnMesh`,
/// C4D `PolygonObject`, Blender `bmesh`, etc.
///
/// * `bounds` — `(min, max)` cubic bounding box in world units.
/// * `resolution` — marching-cubes grid resolution per axis.
///
/// Returns `(verts, faces)` where
/// - `verts` is a Python `list` of `(x, y, z)` `float` tuples
/// - `faces` is a Python `list` of `[i, j, k]` `int` triangle lists
///
/// GIL is released during meshing.
#[pyfunction]
#[pyo3(signature = (node, bounds = (-2.0, 2.0), resolution = 48))]
pub fn sdf_to_mesh<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds: (f32, f32),
    resolution: usize,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    if resolution == 0 {
        return Err(PyValueError::new_err("resolution must be > 0"));
    }
    if bounds.1 <= bounds.0 {
        return Err(PyValueError::new_err(
            "bounds.1 (max) must be > bounds.0 (min)",
        ));
    }

    let config = MarchingCubesConfig {
        resolution,
        iso_level: 0.0,
        compute_normals: false,
        ..Default::default()
    };
    let min = Vec3::splat(bounds.0);
    let max = Vec3::splat(bounds.1);
    let node_ref = &node.inner;
    let mesh = py.detach(|| core_sdf_to_mesh(node_ref, min, max, &config));

    let verts = PyList::empty(py);
    for v in &mesh.vertices {
        let p = v.position;
        let t = PyTuple::new(py, [p.x, p.y, p.z].iter().copied())?;
        verts.append(t)?;
    }
    let faces = PyList::empty(py);
    for tri in mesh.indices.chunks_exact(3) {
        let f = PyList::new(py, tri.iter().copied().map(|i| i as usize))?;
        faces.append(f)?;
    }
    Ok((verts, faces))
}

/// Load an `.asdf` / `.asdf.json` file — DCC-binding-friendly alias for
/// [`super::io::load_sdf`].
///
/// The DCC plugins call `backend.load_asdf(filepath)`; this function provides
/// that exact spelling so the plugins do not need a shim.
#[pyfunction]
pub fn load_asdf(py: Python<'_>, path: &str) -> PyResult<PySdfNode> {
    let path = path.to_string();
    let tree = py
        .detach(|| load(&path))
        .map_err(|e| PyValueError::new_err(format!("Load error: {}", e)))?;
    Ok(PySdfNode { inner: tree.root })
}

/// Bake an SDF to a dense voxel volume and return the raw `bytes` in the
/// `ALICEVDB1` container format.
///
/// Requires the `volume` cargo feature.
///
/// # `ALICEVDB1` binary layout (all little-endian)
///
/// | Offset | Size | Field |
/// |-------:|-----:|-------|
/// |     0  |   9  | magic bytes `b"ALICEVDB1"` |
/// |     9  |   3  | zero padding (align to 12) |
/// |    12  |   4  | `u32` resolution (cubic: `res * res * res` samples) |
/// |    16  |  12  | `[f32; 3]` `bounds_min` (x, y, z) |
/// |    28  |  12  | `[f32; 3]` `bounds_max` (x, y, z) |
/// |    40  | 4·N  | `f32` distance samples, Z-major order (`N = res^3`) |
///
/// The DCC binding side (Nuke `export_asdf_as_volume`) writes these bytes
/// verbatim to disk; downstream tools reading `.alicevdb` should parse the
/// header first and then take the `res^3 * 4` byte payload.
///
/// Requires the `volume` cargo feature.
#[cfg(feature = "volume")]
#[pyfunction]
#[pyo3(signature = (node, bounds = (-2.0, 2.0), resolution = 64))]
pub fn bake_to_vdb<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds: (f32, f32),
    resolution: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    use crate::volume::{bake_volume as cpu_bake, BakeConfig};

    if resolution == 0 {
        return Err(PyValueError::new_err("resolution must be > 0"));
    }
    if bounds.1 <= bounds.0 {
        return Err(PyValueError::new_err(
            "bounds.1 (max) must be > bounds.0 (min)",
        ));
    }

    let min = Vec3::splat(bounds.0);
    let max = Vec3::splat(bounds.1);
    let cfg = BakeConfig {
        resolution: [resolution, resolution, resolution],
        bounds_min: min,
        bounds_max: max,
        generate_mips: false,
        ..Default::default()
    };
    let node_ref = &node.inner;
    let volume = py.detach(|| cpu_bake(node_ref, &cfg));

    let payload_bytes = volume.data.len() * 4;
    let mut buf: Vec<u8> = Vec::with_capacity(40 + payload_bytes);
    buf.extend_from_slice(b"ALICEVDB1");
    buf.extend_from_slice(&[0u8; 3]);
    buf.extend_from_slice(&resolution.to_le_bytes());
    for c in [min.x, min.y, min.z, max.x, max.y, max.z] {
        buf.extend_from_slice(&c.to_le_bytes());
    }
    for v in &volume.data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    Ok(PyBytes::new(py, &buf))
}

/// Render a 2D SDF slice at fixed world `z` into an RGBA uint8 numpy image.
///
/// * `bounds` — `(min, max)` cubic bounding box; the returned image spans
///   `[min, max]` in both x and y at the requested `z`.
/// * `width` / `height` — output image resolution (columns × rows).
///
/// # Pixel encoding
///
/// - Contour (`|d| < span/max(w,h)`) → white `(255, 255, 255, 255)`
/// - Inside (`d ≤ 0`) → blue, darkening with depth
/// - Outside (`d > 0`) → red-to-green gradient by distance
///
/// This is intended as a debug visualization directly consumable by Nuke's
/// image nodes (Python plugin path) — pixel layout is `(H, W, 4)` in the
/// canonical NumPy image convention.
#[pyfunction]
#[pyo3(signature = (node, z, width = 512, height = 512, bounds = (-2.0, 2.0)))]
pub fn render_slice_2d<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    z: f32,
    width: usize,
    height: usize,
    bounds: (f32, f32),
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("width and height must be > 0"));
    }
    let (min_b, max_b) = bounds;
    let span = max_b - min_b;
    if span <= 0.0 {
        return Err(PyValueError::new_err(
            "bounds.1 (max) must be > bounds.0 (min)",
        ));
    }

    let node_ref = &node.inner;

    let mut points: Vec<Vec3> = Vec::with_capacity(width * height);
    for j in 0..height {
        let ty = (j as f32 + 0.5) / height as f32;
        let y = min_b + ty * span;
        for i in 0..width {
            let tx = (i as f32 + 0.5) / width as f32;
            let x = min_b + tx * span;
            points.push(Vec3::new(x, y, z));
        }
    }
    let distances = py.detach(|| eval_batch_parallel(node_ref, &points));

    let contour_eps = span / (width.max(height) as f32);
    let mut rgba: Vec<u8> = Vec::with_capacity(width * height * 4);
    for &d in &distances {
        let (r, g, b, a) = if d.abs() < contour_eps {
            (255u8, 255u8, 255u8, 255u8)
        } else if d <= 0.0 {
            let t = (-d / span).clamp(0.0, 1.0);
            let v = (255.0 * (1.0 - 0.7 * t)) as u8;
            (0, 0, v, 255)
        } else {
            let t = (d / span).clamp(0.0, 1.0);
            let r_ch = (255.0 * (1.0 - t)) as u8;
            let g_ch = (255.0 * t) as u8;
            (r_ch, g_ch, 0, 255)
        };
        rgba.push(r);
        rgba.push(g);
        rgba.push(b);
        rgba.push(a);
    }

    let arr = Array3::<u8>::from_shape_vec((height, width, 4), rgba)
        .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;
    Ok(arr.into_pyarray(py))
}

// Note: no `#[cfg(test)] mod tests` here because the `src/python/` tree is
// only compiled under `--features python`, which enables pyo3's
// `extension-module` feature — that flag deliberately unresolves the Python
// symbols, so `cargo test --features python` cannot link a test binary.
// The pure-logic invariants (ALICEVDB1 header layout, primitive delegation)
// are covered by the top-level integration tests under `tests/`.
