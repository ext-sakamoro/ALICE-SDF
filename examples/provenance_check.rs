//! Provenance — write a mesh, read its semantics identifier back and check it
//!
//! Exports a sphere as binary STL, 3MF, STEP and meshopt-compressed glTF, reads the identifier
//! each file carries, and checks it against this crate's identifier. The binary mesh format (.abm)
//! and the tree format (.asdf) carry it in their headers and their loaders check it. A stamped STL also carries a
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

    // the JSON form of the tree carries the identifier as a top-level field
    let asdf_json = dir.join("sphere.asdf.json");
    io::save_asdf_json(&tree, &asdf_json)?;
    let json_loaded = io::load_asdf_json_unverified(&asdf_json)?;
    println!("{}: {:?}", asdf_json.display(), json_loaded.semantics());
    io::save_asdf_json_with_provenance(&tree, &asdf_json, &Provenance::sdf())?;
    let text = io::to_json_string_with_provenance(&tree, &Provenance::sdf())?;
    assert_eq!(
        io::from_json_string_unverified(&text)?.semantics(),
        StoredSemantics::Current
    );
    // the default reader accepts it because it carries this build's identifier
    assert_eq!(io::from_json_string(&text)?.node_count(), tree.node_count());
    assert_eq!(back.node_count(), tree.node_count());

    // the binary mesh format: a version 2 .abm header carries the identifier, load_abm checks it,
    // the unverified loaders say what is known, and one written under another identifier is refused
    let abm = dir.join("sphere.abm");
    io::save_abm(&mesh, &abm)?;
    assert_eq!(
        io::read_abm_header(&abm)?.semantics_id,
        Some(alice_sdf::SDF_SEMANTICS_ID)
    );
    let checked = io::load_abm_unverified(&abm)?;
    assert_eq!(checked.semantics(), StoredSemantics::Current);
    assert_eq!(io::load_abm(&abm)?.vertices.len(), mesh.vertices.len());
    let abm_lods = dir.join("sphere_lods.abm");
    io::save_abm_with_lods(&[mesh.clone(), mesh.clone()], &[10.0], &abm_lods)?;
    let (lods, _) = io::load_abm_with_lods_unverified(&abm_lods)?.into_parts();
    assert_eq!(lods.0.len(), 2);
    println!("{}: {:?}", abm.display(), checked.semantics());
    let foreign = Provenance::new([7; 32]);
    let abm_other = dir.join("other.abm");
    io::save_abm_with_provenance(&mesh, &abm_other, &foreign)?;
    io::save_abm_with_lods_with_provenance(std::slice::from_ref(&mesh), &[], &abm_lods, &foreign)?;
    for path in [&abm_other, &abm_lods] {
        match io::load_abm_with_lods(path) {
            Err(e) => println!("{}: refused: {e}", path.display()),
            Ok(_) => return Err("an .abm under another identifier was read".into()),
        }
    }

    // a LOD chain: the .abm and its .abm.meta.json sidecar both carry the identifier, and the
    // unverified loader reports the weaker of the two parts
    use alice_sdf::mesh::lod_persist;
    let chain = lod_persist::LodChainPersist::new(
        vec![mesh.clone(), mesh.clone()],
        vec![0.0, 10.0],
        1,
        lod_persist::LodChainConfig::default(),
    );
    let chain_path = dir.join("sphere_chain.abm");
    lod_persist::save_lod_chain(&chain, &chain_path)?;
    let (back, semantics) = lod_persist::load_lod_chain_unverified(&chain_path)?.into_parts();
    assert_eq!(
        (back.meshes.len(), semantics),
        (2, StoredSemantics::Current)
    );
    lod_persist::save_lod_chain_metadata_with_provenance(
        &chain,
        &chain_path,
        &Provenance::new([7; 32]),
    )?;
    match lod_persist::load_lod_chain(&chain_path) {
        Err(e) => println!("{}: refused: {e}", chain_path.display()),
        Ok(_) => return Err("a LOD chain whose sidecar has another identifier was read".into()),
    }
    lod_persist::save_lod_chain_with_provenance(&chain, &chain_path, &Provenance::sdf())?;
    println!("{}: {semantics:?}", chain_path.display());
    // the sidecar alone: rewriting it stamps this crate's identifier, and reading only the
    // settings back returns them with what is known about the sidecar
    lod_persist::save_lod_chain_metadata(&chain, &chain_path)?;
    let meta_path = dir.join("sphere_chain.abm.meta.json");
    let (config, meta_semantics) = lod_persist::load_lod_chain_metadata(&meta_path)?.into_parts();
    assert_eq!(meta_semantics, StoredSemantics::Current);
    assert_eq!(config.lod_count, chain.config.lod_count);
    println!("{}: {meta_semantics:?}", meta_path.display());

    let other = dir.join("other.stl");
    io::export_stl_with_provenance(&mesh, &other, &Provenance::new([7; 32]))?;
    match provenance::check_sdf(&other) {
        Err(e) => println!("{}: refused: {e}", other.display()),
        Ok(a) => return Err(format!("another identifier was semantics: {a:?}").into()),
    }

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
