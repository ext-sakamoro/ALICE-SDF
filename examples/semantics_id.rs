//! Record and check the identifier of what alice-sdf computes
//!
//! A program that stores distances (a baked field, a collision trace, a content hash of a
//! tree and its samples) records `SDF_SEMANTICS_ID` next to them; reading them back, it
//! asks `check_sdf_semantics` whether this build computes the same bits.
//!
//! Run: `cargo run --example semantics_id`

use alice_sdf::semantics::{sdf_semantics_id_for, MAX_MIN_RULE, MUL_ADD_RULE, SDF_SEMANTICS_PINS};
use alice_sdf::{
    check_sdf_semantics, PREVIOUS_SDF_SEMANTICS_IDS, SDF_SEMANTICS_BY_DET_MATH, SDF_SEMANTICS_ID,
};

fn hex(b: &[u8; 32]) -> String {
    b.iter().fold(String::new(), |mut s, x| {
        use std::fmt::Write;
        write!(s, "{x:02x}").expect("writing to a String");
        s
    })
}

fn main() {
    println!("SDF_SEMANTICS_ID {}", hex(&SDF_SEMANTICS_ID));
    println!("folded from {} entries:", SDF_SEMANTICS_PINS.len());
    for (name, digest) in SDF_SEMANTICS_PINS {
        println!("  {name}: {}", hex(digest));
    }
    let rules = [MAX_MIN_RULE, MUL_ADD_RULE];
    for rule in rules {
        println!("rule: {rule}");
    }
    for (det_math, id) in SDF_SEMANTICS_BY_DET_MATH {
        assert_eq!(sdf_semantics_id_for(det_math), *id);
        println!("  with alice-det-math {} -> {}", hex(det_math), hex(id));
    }

    // a stored identifier, as a reader would find it next to the data
    for stored in [SDF_SEMANTICS_ID, PREVIOUS_SDF_SEMANTICS_IDS[0], [0u8; 32]] {
        let check = check_sdf_semantics(&stored);
        println!(
            "{} -> {check:?} (reproducible: {})",
            hex(&stored),
            check.is_reproducible()
        );
    }
}
