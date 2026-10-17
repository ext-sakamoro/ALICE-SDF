//! Provenance — write a mesh, read its semantics identifier back and check it
//!
//! Exports a sphere as binary STL, 3MF, STEP and meshopt-compressed glTF, reads the identifier
//! each file carries, and checks it against this crate's identifier. A file written with another
//! identifier is refused, and the error names both.
//!
//! # Running
//! ```bash
//! cargo run --example provenance_check
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::io;
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig};
use alice_sdf::prelude::*;
use alice_sdf::provenance::{self, Provenance, StoredSemantics};
use glam::Vec3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig::default(),
    );
    let dir = std::env::temp_dir().join(format!(
        "alice_sdf_provenance_example_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)?;

    let stl = dir.join("sphere.stl");
    let threemf = dir.join("sphere.3mf");
    io::export_stl(&mesh, &stl)?;
    io::export_3mf(&mesh, &threemf)?;
    for path in [&stl, &threemf] {
        let stored = provenance::read(path)?;
        let semantics = provenance::check_sdf(path)?;
        assert_eq!(semantics, StoredSemantics::Current);
        println!("{}: {stored} ({semantics:?})", path.display());
    }

    // every format stamps the same identifier: STEP from the tree, meshopt-compressed glTF
    // written to a file and to bytes
    use alice_sdf::io::meshopt_gltf as meshopt;
    let mo = meshopt::MeshoptGltfConfig::default();
    let step = dir.join("sphere.step");
    io::step::export_step(
        &step,
        &SdfNode::sphere(1.0),
        &io::step::StepConfig::default(),
    )?;
    let glb = dir.join("sphere_meshopt.glb");
    meshopt::export_glb_meshopt(&mesh, &glb, &mo)?;
    let glb_skinned = dir.join("sphere_meshopt_skinned.glb");
    meshopt::export_glb_meshopt_skinned(&mesh, None, &glb_skinned, &mo)?;
    let bytes = dir.join("sphere_meshopt_bytes.glb");
    std::fs::write(&bytes, meshopt::export_glb_meshopt_bytes(&mesh, &mo)?)?;
    let bytes_skinned = dir.join("sphere_meshopt_bytes_skinned.glb");
    std::fs::write(
        &bytes_skinned,
        meshopt::export_glb_meshopt_bytes_skinned(&mesh, None, &mo)?,
    )?;
    for path in [&step, &glb, &glb_skinned, &bytes, &bytes_skinned] {
        assert_eq!(provenance::check_sdf(path)?, StoredSemantics::Current);
        println!("{}: {}", path.display(), provenance::read(path)?);
    }

    let other = dir.join("other.stl");
    io::export_stl_with_provenance(&mesh, &other, &Provenance::new([7; 32]))?;
    match provenance::check_sdf(&other) {
        Err(e) => println!("{}: refused: {e}", other.display()),
        Ok(a) => return Err(format!("another identifier was semantics: {a:?}").into()),
    }

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
