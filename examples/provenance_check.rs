//! Provenance — write a mesh, read its semantics identifier back and check it
//!
//! Exports a sphere as binary STL, 3MF, STEP and meshopt-compressed glTF, reads the identifier
//! each file carries, and checks it against this crate's identifier. The binary mesh format (.abm)
//! and the tree format (.asdf) carry it in their headers and their loaders check it, as do the
//! neural weights (NSDF). A stamped STL also carries a
//! producer identifier, read back from its own slot. A file written with another
//! identifier is refused, and the error names both. It first lists every stored form with the
//! meaning identifier this build's writers store in it.
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
use alice_sdf::provenance::{self, Provenance, StoredForm, StoredSemantics};
use glam::Vec3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // what each stored form carries under this build's identifier: one meaning identifier per
    // form, all distinct
    let mut seen = std::collections::BTreeSet::new();
    for form in StoredForm::ALL {
        let id = Provenance::new(provenance::meaning_id(form, &Provenance::sdf()));
        println!("{:>16} v{}: {id}", form.name(), form.version());
        assert!(
            seen.insert(id.hex()),
            "{} shares a meaning identifier",
            form.name()
        );
    }
    assert_eq!(seen.len(), StoredForm::ALL.len());

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
    // the primary slot holds this crate's meaning identifier for binary STL, the producer its raw one
    assert_eq!(
        primary.judge(&Provenance::sdf(), &[])?,
        StoredSemantics::Current
    );
    assert_eq!(stored_producer, Some(producer));
    // what this build's writers store for binary STL under this crate's identifier
    assert!(primary.is_written_under(&Provenance::sdf()));
    assert_eq!(
        provenance::from_bytes_with_producer(&std::fs::read(&stamped)?)?,
        (primary, stored_producer)
    );
    provenance::check_producer(&stamped, &producer)?;
    println!("{}: {primary}, producer {producer}", stamped.display());
    // the same in a text format: an OBJ keeps both on its header comment line
    let stamped_obj = dir.join("sphere_stamped.obj");
    io::export_obj_with_stamp(&mesh, &stamped_obj, &io::ObjConfig::default(), None, &stamp)?;
    let (obj_primary, obj_producer) = provenance::read_with_producer(&stamped_obj)?;
    assert_eq!(
        obj_primary.judge(&Provenance::sdf(), &[])?,
        StoredSemantics::Current
    );
    assert_eq!(obj_producer, Some(producer));
    println!(
        "{}: {obj_primary}, producer {producer}",
        stamped_obj.display()
    );

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

    // the UE5 asset: the JSON carries the identifier as a top-level field next to the mesh, for
    // one mesh or a LOD list; written under another identifier, the check names it
    let ue5 = dir.join("sphere.ue5_mesh");
    let ue5_config = io::Ue5MeshConfig::default();
    io::export_ue5_mesh_with_lods(
        &[mesh.clone(), mesh.clone()],
        &[1.0, 0.5],
        &ue5,
        &ue5_config,
    )?;
    assert_eq!(provenance::check_sdf(&ue5)?, StoredSemantics::Current);
    io::export_ue5_mesh_with_provenance(&mesh, &ue5, &ue5_config, &foreign)?;
    assert_eq!(provenance::read(&ue5)?.id, foreign.semantics_id);
    println!(
        "{}: written under {}",
        ue5.display(),
        provenance::read(&ue5)?
    );

    let other = dir.join("other.stl");
    io::export_stl_with_provenance(&mesh, &other, &Provenance::new([7; 32]))?;
    match provenance::check_sdf(&other) {
        Err(e) => println!("{}: refused: {e}", other.display()),
        Ok(a) => return Err(format!("another identifier was semantics: {a:?}").into()),
    }

    // neural weights: a version 2 NSDF file carries the identifier after its version, load checks
    // it, load_unverified says what is known, and weights saved under another identifier are refused
    use alice_sdf::neural::{NeuralSdf, NeuralSdfConfig};
    let net = NeuralSdf::new(&NeuralSdfConfig {
        hidden_layers: 1,
        hidden_width: 4,
        pos_encoding_freqs: 1,
        ..Default::default()
    });
    let mut weights = Vec::new();
    net.save(&mut weights)?;
    let checked = NeuralSdf::load_unverified(&mut weights.as_slice())?;
    println!("NSDF weights: {:?}", checked.semantics());
    assert_eq!(checked.semantics(), StoredSemantics::Current);
    let mut other = Vec::new();
    net.save_with_provenance(&mut other, &foreign)?;
    match NeuralSdf::load(&mut other.as_slice()) {
        Err(e) => println!("NSDF weights under another identifier: refused: {e}"),
        Ok(_) => return Err("NSDF weights under another identifier were read".into()),
    }

    // the ALICE-VDB dense grid (feature openvdb): `ALICEVDB2` carries the identifier after its magic
    #[cfg(feature = "openvdb")]
    {
        use alice_sdf::io::vdb;
        let grid = vdb::bake_to_vdb(&SdfNode::sphere(1.0), (-1.5, 1.5), 8)?;
        let checked = vdb::load_dense_grid_from_vdb_unverified(&grid)?;
        println!("ALICE-VDB grid: {:?}", checked.semantics());
        assert_eq!(checked.semantics(), StoredSemantics::Current);
        let other =
            vdb::bake_to_vdb_with_provenance(&SdfNode::sphere(1.0), (-1.5, 1.5), 8, &foreign)?;
        match vdb::load_dense_grid_from_vdb(&other) {
            Err(e) => println!("ALICE-VDB grid under another identifier: refused: {e}"),
            Ok(_) => return Err("an ALICE-VDB grid under another identifier was read".into()),
        }
    }

    // a MagicaVoxel .vox: the identifier is a private ALSD chunk among MAIN's children, which
    // MagicaVoxel skips; a file without it (any other program's) reads as unpinned, the pinned
    // loader refuses it
    {
        use alice_sdf::io::vox::{
            load_vox, load_vox_pinned, save_vox, save_vox_with_provenance, VoxModel, Voxel,
        };
        let model = VoxModel {
            size: (2, 2, 2),
            voxels: vec![Voxel {
                x: 1,
                y: 0,
                z: 1,
                color: 9,
            }],
        };
        let vox = dir.join("model.vox");
        save_vox(&vox, &model)?;
        let checked = load_vox(&vox)?;
        println!("{}: {:?}", vox.display(), checked.semantics());
        assert_eq!(checked.semantics(), StoredSemantics::Current);
        assert_eq!(load_vox_pinned(&vox)?.voxels.len(), 1);
        let vox_other = dir.join("other.vox");
        save_vox_with_provenance(&vox_other, &model, &foreign)?;
        match load_vox(&vox_other) {
            Err(e) => println!("{}: refused: {e}", vox_other.display()),
            Ok(_) => return Err("a .vox under another identifier was read".into()),
        }
    }

    // Gaussian splats: .splat has no place for the identifier, so save_splat writes the sidecar
    // <path>.semantics with the identifier and the file's SHA-256; a .splat without a sidecar (any
    // viewer's) reads as unpinned, the pinned loader refuses it
    {
        use alice_sdf::io::splat::{
            load_splat, load_splat_pinned, save_splat, save_splat_with_provenance, Splat,
        };
        let splats = [Splat {
            position: [0.0, 0.5, 0.0],
            scale: [0.1; 3],
            color: [200, 200, 220, 255],
            rotation: [128, 128, 128, 255],
        }];
        let splat = dir.join("cloud.splat");
        save_splat(&splat, &splats)?;
        let checked = load_splat(&splat)?;
        println!("{}: {:?}", splat.display(), checked.semantics());
        assert_eq!(checked.semantics(), StoredSemantics::Current);
        assert_eq!(load_splat_pinned(&splat)?.len(), 1);
        let splat_other = dir.join("other.splat");
        save_splat_with_provenance(&splat_other, &splats, &foreign)?;
        match load_splat(&splat_other) {
            Err(e) => println!("{}: refused: {e}", splat_other.display()),
            Ok(_) => return Err("a .splat under another identifier was read".into()),
        }
    }

    // an SVO chunk (feature svo): the `ASVC` version 2 wire form carries the identifier after its
    // version; bytes without the magic (version 1) read only through the unverified reader
    #[cfg(feature = "svo")]
    {
        use alice_sdf::svo::streaming::split_into_chunks;
        use alice_sdf::svo::{SparseVoxelOctree, SvoBuildConfig, SvoChunk};
        let svo = SparseVoxelOctree::build(
            &SdfNode::sphere(1.0),
            &SvoBuildConfig {
                max_depth: 3,
                ..Default::default()
            },
        );
        let chunk = &split_into_chunks(&svo, 1)[0];
        let checked = SvoChunk::from_bytes_unverified(&chunk.to_bytes())?;
        println!("SVO chunk: {:?}", checked.semantics());
        assert_eq!(checked.semantics(), StoredSemantics::Current);
        match SvoChunk::from_bytes(&chunk.to_bytes_with_provenance(&foreign)) {
            Err(e) => println!("SVO chunk under another identifier: refused: {e}"),
            Ok(_) => return Err("an SVO chunk under another identifier was read".into()),
        }
    }

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
