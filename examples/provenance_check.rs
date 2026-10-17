//! Provenance — write a mesh, read its semantics identifier back and check it
//!
//! Exports a sphere as binary STL, 3MF, STEP and meshopt-compressed glTF, reads the identifier
//! each file carries, and checks it against this crate's identifier. A stamped STL also carries a
//! producer identifier, read back from its own slot. A file written with another
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

    // a program that turns its own source into meshes stores its identifier as the producer,
    // next to this crate's: both read back from their own slots
    let producer = Provenance::new([0x5a; 32]);
    let stamp = provenance::Stamp::sdf_with_producer(producer);
    let stamped = dir.join("sphere_stamped.stl");
    io::export_stl_with_stamp(&mesh, &stamped, &stamp)?;
    let (primary, stored_producer) = provenance::read_with_producer(&stamped)?;
    assert_eq!(
        (primary, stored_producer),
        (Provenance::sdf(), Some(producer))
    );
    assert_eq!(
        provenance::from_bytes_with_producer(&std::fs::read(&stamped)?)?,
        (primary, stored_producer)
    );
    provenance::check_producer(&stamped, &producer)?;
    println!("{}: {primary}, producer {producer}", stamped.display());

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

    // this crate's own format: the .asdf header carries the identifier, load_asdf checks it, and
    // the explicit load_asdf_unverified returns the tree next to what is known about it
    let tree = SdfTree::new(SdfNode::sphere(1.0));
    let asdf = dir.join("sphere.asdf");
    io::save_asdf(&tree, &asdf)?;
    let header = io::AsdfHeader::from_bytes(&std::fs::read(&asdf)?)?;
    assert_eq!(
        header.semantics_id,
        io::AsdfHeader::new(&tree, header.crc32).semantics_id
    );
    let loaded = io::load_asdf_unverified(&asdf)?;
    println!("{}: {:?}", asdf.display(), loaded.semantics());
    let (back, _) = loaded.into_parts();
    assert_eq!(back.node_count(), tree.node_count());

    let other = dir.join("other.stl");
    io::export_stl_with_provenance(&mesh, &other, &Provenance::new([7; 32]))?;
    match provenance::check_sdf(&other) {
        Err(e) => println!("{}: refused: {e}", other.display()),
        Ok(a) => return Err(format!("another identifier was semantics: {a:?}").into()),
    }

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
