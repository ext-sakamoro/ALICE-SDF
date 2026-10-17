//! Provenance — write a mesh, read its semantics identifier back and check it
//!
//! Exports a sphere as binary STL and 3MF, reads the identifier each file carries, and checks it
//! against this crate's identifier. A file written with another identifier is refused, and the
//! error names both.
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
use alice_sdf::provenance::{self, Accepted, Provenance};
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
        let accepted = provenance::check_sdf(path)?;
        assert_eq!(accepted, Accepted::Current);
        println!("{}: {stored} ({accepted:?})", path.display());
    }

    let other = dir.join("other.stl");
    io::export_stl_with_provenance(&mesh, &other, &Provenance::new([7; 32]))?;
    match provenance::check_sdf(&other) {
        Err(e) => println!("{}: refused: {e}", other.display()),
        Ok(a) => return Err(format!("another identifier was accepted: {a:?}").into()),
    }

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
