//! Fuzz target: the ASKC (kink cell sidecar) reader on arbitrary bytes.
//!
//! `KinkFile::read` takes files from disk: any input must end in `Ok` or `Err`, never a panic, an abort or an
//! allocation the input cannot back (a count is never used to size memory). The input is also read behind a valid
//! header (magic, version and this crate's semantics identifier), so the exploration reaches the sheet records instead
//! of stopping at the identifier check. A file that reads is attached to an empty sheet list (the range check).

#![no_main]

use alice_sdf::sheet_field::kink::KinkFile;
use alice_sdf::sheet_field::kink_build::write_askc;
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

/// magic, version and identifier of a sidecar this crate writes
fn header() -> &'static [u8] {
    static H: OnceLock<Vec<u8>> = OnceLock::new();
    H.get_or_init(|| {
        let mut own = Vec::new();
        write_askc(&mut own, &[(None, 0.01, 0)]).expect("an empty sidecar writes");
        own[..40].to_vec()
    })
}

fuzz_target!(|bytes: &[u8]| {
    let _ = KinkFile::read(&mut &bytes[..]);
    let mut framed = header().to_vec();
    framed.extend_from_slice(bytes);
    if let Ok(kf) = KinkFile::read(&mut framed.as_slice()) {
        let _ = kf.attach(&mut [], 0);
        let _ = kf.len();
    }
});
