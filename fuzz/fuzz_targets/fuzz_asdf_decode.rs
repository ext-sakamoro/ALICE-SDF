//! Fuzz target: ASDF binary decode で panic / 無限ループを炙り出す
//!
//! 任意のバイト列を `load_asdf` (streaming reader 経由) に食わせて、
//! パースエラーで Err を返すのは OK、panic / infinite loop / OOM は NG
//!
//! 起こり得る危険:
//! - CRC 検証前に bincode がハングする (v1 で実際に発生した)
//! - 極端な node_count で allocation → OOM
//! - malformed magic で信頼して読み進める → panic

#![no_main]

// prelude export 経由 (io::asdf module は private)
use alice_sdf::prelude::load_asdf;
use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|bytes: &[u8]| {
    // 一時 file 経由 (load_asdf は Path を受け取る API)
    let mut path = std::env::temp_dir();
    path.push(format!(
        "alice_sdf_fuzz_asdf_{}.asdf",
        std::process::id()
    ));

    // 書き込み失敗時は skip (test harness の問題であって fuzz target の問題ではない)
    let Ok(mut file) = std::fs::File::create(&path) else {
        return;
    };
    if file.write_all(bytes).is_err() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    drop(file);

    // load_asdf の Result はどちらでも OK、panic しないことのみ検証
    let _ = load_asdf(&path);

    let _ = std::fs::remove_file(&path);
});
