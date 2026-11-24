//! The device speed table a field's device is admitted with says where each row came from: the crate's own table,
//! or the file `ALICE_SDF_ADAPTER_SPEEDS` names (with its path), so an override is visible in the march stats.
//!
//! Oracle: a file written here with one row, named by the variable; its row reads back with that file as its source,
//! and with the variable unset no row has a file source. This test sets a process variable, so it has a test binary
//! of its own.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::sheet_field::gpu::{adapter_speeds, SpeedSource, ADAPTER_SPEEDS_VAR};

#[test]
fn a_row_from_the_named_file_reads_back_with_that_file_as_its_source() {
    let dir = std::env::temp_dir().join(format!("alice-sdf-speed-source-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("speeds.tsv");
    std::fs::write(
        &path,
        "# a test table\n0x1234\t0x5678\tVulkan\tTest adapter\tdrv 1\t321.5\t2026-11-25\tc\n",
    )
    .unwrap();
    std::env::remove_var(ADAPTER_SPEEDS_VAR);
    let plain = adapter_speeds().unwrap();
    assert!(
        plain.iter().all(|r| r.source == SpeedSource::CrateTable),
        "a row has a file source with the variable unset: {plain:?}"
    );
    std::env::set_var(ADAPTER_SPEEDS_VAR, &path);
    let rows = adapter_speeds().unwrap();
    std::env::remove_var(ADAPTER_SPEEDS_VAR);
    let from_file: Vec<_> = rows
        .iter()
        .filter(|r| r.vendor == 0x1234 && r.device == 0x5678)
        .collect();
    assert_eq!(from_file.len(), 1, "{rows:?}");
    assert_eq!(
        from_file[0].source,
        SpeedSource::File(path.to_string_lossy().into_owned()),
        "the file's row does not name the file"
    );
    assert_eq!(
        rows.len(),
        plain.len() + 1,
        "the crate's rows changed when a file was named"
    );
    std::fs::remove_dir_all(&dir).ok();
}
