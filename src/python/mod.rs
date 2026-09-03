//! Python bindings for ALICE-SDF
//!
//! Provides Python API via PyO3.
//!
//! Author: Moroya Sakamoto

// pyo3 0.29+: lib.rs 側で `#[cfg(feature = "python")] pub mod python;` 済のため
// module 内での `#![cfg(feature = "python")]` は clippy::duplicated_attribute で block される
// pyo3 boilerplate は type_complexity / too_many_arguments が pyo3 側 API structure 起因で頻発するため
// 本 module 全体で allow (production path でなく Python 引き剥がし境界)
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

mod compiled;
mod dcc;
mod functions;
mod helpers;
mod io;
mod node;
mod operations;
mod transforms;

pub use compiled::PyCompiledSdf;
pub use node::PySdfNode;

use pyo3::prelude::*;

/// Python module
#[pymodule]
pub fn alice_sdf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<node::PySdfNode>()?;
    m.add_class::<compiled::PyCompiledSdf>()?;

    // Core functions
    m.add_function(wrap_pyfunction!(functions::compile_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(functions::eval_batch, m)?)?;
    m.add_function(wrap_pyfunction!(functions::eval_compiled_batch, m)?)?;
    m.add_function(wrap_pyfunction!(functions::eval_compiled_batch_soa, m)?)?;
    m.add_function(wrap_pyfunction!(functions::to_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(functions::to_mesh_adaptive, m)?)?;
    m.add_function(wrap_pyfunction!(functions::to_mesh_dual_contouring, m)?)?;
    m.add_function(wrap_pyfunction!(functions::decimate_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(functions::version, m)?)?;
    m.add_class::<functions::PyMeshCache>()?;

    // DCC-binding helpers (Maya / Cinema 4D / Nuke): module-level primitive
    // constructors + mesh/volume/slice convenience wrappers, so that the
    // DCC plugin scripts under `bindings/{maya,cinema4d,nuke}/` work
    // directly against the crates.io wheel with no shim code.
    m.add_function(wrap_pyfunction!(dcc::sphere, m)?)?;
    m.add_function(wrap_pyfunction!(dcc::make_box, m)?)?;
    m.add_function(wrap_pyfunction!(dcc::torus, m)?)?;
    m.add_function(wrap_pyfunction!(dcc::sdf_to_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(dcc::load_asdf, m)?)?;
    #[cfg(feature = "volume")]
    m.add_function(wrap_pyfunction!(dcc::bake_to_vdb, m)?)?;
    m.add_function(wrap_pyfunction!(dcc::render_slice_2d, m)?)?;

    // I/O functions
    m.add_function(wrap_pyfunction!(io::export_obj, m)?)?;
    m.add_function(wrap_pyfunction!(io::export_glb, m)?)?;
    m.add_function(wrap_pyfunction!(io::export_glb_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(io::export_fbx, m)?)?;
    m.add_function(wrap_pyfunction!(io::export_usda, m)?)?;
    m.add_function(wrap_pyfunction!(io::export_alembic, m)?)?;
    m.add_function(wrap_pyfunction!(io::uv_unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(io::save_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(io::load_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(io::save_abm, m)?)?;
    m.add_function(wrap_pyfunction!(io::load_abm, m)?)?;
    m.add_function(wrap_pyfunction!(io::export_unity, m)?)?;
    m.add_function(wrap_pyfunction!(io::export_ue5, m)?)?;
    m.add_function(wrap_pyfunction!(io::from_json, m)?)?;
    m.add_function(wrap_pyfunction!(io::to_json, m)?)?;
    m.add_function(wrap_pyfunction!(io::import_glb_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(io::import_glb, m)?)?;
    m.add_function(wrap_pyfunction!(io::mesh_to_sdf, m)?)?;

    // Feature-gated modules
    #[cfg(feature = "volume")]
    {
        m.add_function(wrap_pyfunction!(functions::bake_volume, m)?)?;
        m.add_function(wrap_pyfunction!(functions::gpu_bake_volume, m)?)?;
    }
    #[cfg(feature = "gpu")]
    {
        m.add_function(wrap_pyfunction!(functions::gpu_marching_cubes, m)?)?;
    }
    #[cfg(feature = "svo")]
    {
        m.add_class::<functions::PySvo>()?;
        m.add_function(wrap_pyfunction!(functions::build_svo, m)?)?;
        m.add_function(wrap_pyfunction!(functions::svo_query_batch, m)?)?;
    }
    #[cfg(feature = "destruction")]
    {
        m.add_class::<functions::PyVoxelGrid>()?;
        m.add_function(wrap_pyfunction!(functions::create_voxel_grid, m)?)?;
        m.add_function(wrap_pyfunction!(functions::carve_sphere, m)?)?;
        m.add_function(wrap_pyfunction!(functions::voxel_fracture, m)?)?;
    }
    #[cfg(feature = "terrain")]
    {
        m.add_class::<functions::PyHeightmap>()?;
        m.add_function(wrap_pyfunction!(functions::generate_terrain, m)?)?;
        m.add_function(wrap_pyfunction!(functions::erode_terrain, m)?)?;
        m.add_function(wrap_pyfunction!(functions::heightmap_sample_batch, m)?)?;
    }
    #[cfg(feature = "gi")]
    {
        m.add_function(wrap_pyfunction!(functions::bake_gi, m)?)?;
    }
    Ok(())
}
