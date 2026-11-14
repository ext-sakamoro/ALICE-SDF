//! The bytecode compiler keeps its output: a recorded fingerprint per scene
//!
//! `tests/fixtures/compile_bytecode_golden.tsv` holds, for every scene of the shared corpus (every
//! variant the bytecode carries) and for deep folded chains, the number of instructions and an
//! FNV-1a hash over every instruction (each field, through `Debug`, which prints an `f32` so that
//! it reads back to the same bits) and every aux float's bits (a scene the compiler refuses: 0
//! instructions and the hash of the refusal's text). The table was recorded from the
//! compiler before its walk was changed (the recursive `match` over every node kind); the test
//! compares every row, fails on a scene the table does not have and on a row no scene has, and
//! fails when nothing was compared. Record a new table (only for an intended change of the
//! bytecode) with `ALICE_SDF_RECORD_BYTECODE=1`.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::compiled::CompiledSdf;
use alice_sdf::SdfNode;
use common::corpus::corpus;

const TABLE: &str = "tests/fixtures/compile_bytecode_golden.tsv";

/// A chain of `depth` smooth unions of spheres along x, folded one at a time
fn sphere_chain(depth: usize) -> SdfNode {
    let mut result = SdfNode::sphere(1.0);
    for i in 1..=depth {
        result = result.smooth_union(
            SdfNode::sphere(1.0).translate(i as f32 * 1.5, 0.0, 0.0),
            0.5,
        );
    }
    result
}

fn scenes() -> Vec<(String, SdfNode)> {
    let mut v = corpus();
    v.push(("sphere_chain_48", sphere_chain(48)));
    v.push(("sphere_chain_1024", sphere_chain(1024)));
    v.into_iter()
        .enumerate()
        .map(|(i, (name, node))| (format!("{i}:{name}"), node))
        .collect()
}

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// (instruction count, FNV-1a over the instructions' text and the aux floats' bits)
fn fingerprint(compiled: &CompiledSdf) -> (usize, u64) {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for inst in compiled.instructions() {
        fnv1a(&mut h, format!("{inst:?}").as_bytes());
        fnv1a(&mut h, b"\n");
    }
    for f in compiled.aux_data() {
        fnv1a(&mut h, &f.to_bits().to_le_bytes());
    }
    (compiled.instructions().len(), h)
}

/// Large enough for the deep chains whatever the frame of the compiler's walk
const THREAD_BYTES: usize = 256 * 1024 * 1024;

fn measured() -> Vec<(String, usize, u64)> {
    std::thread::Builder::new()
        .stack_size(THREAD_BYTES)
        .spawn(|| {
            scenes()
                .into_iter()
                .map(|(name, node)| match CompiledSdf::try_compile(&node) {
                    Ok(compiled) => {
                        let (n, h) = fingerprint(&compiled);
                        (name, n, h)
                    }
                    // a refusal is part of what is pinned: its text, 0 instructions
                    Err(e) => {
                        let mut h = 0xcbf2_9ce4_8422_2325_u64;
                        fnv1a(&mut h, format!("refused: {e}").as_bytes());
                        (name, 0, h)
                    }
                })
                .collect()
        })
        .expect("spawn a thread")
        .join()
        .expect("the thread finished")
}

fn recorded() -> Vec<(String, usize, u64)> {
    std::fs::read_to_string(TABLE)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut f = l.split('\t');
            let name = f.next().expect("scene").to_owned();
            let n = f.next().expect("count").parse().expect("a count");
            let h =
                u64::from_str_radix(f.next().expect("hash").strip_prefix("0x").expect("hex"), 16)
                    .expect("a hash");
            (name, n, h)
        })
        .collect()
}

#[test]
fn every_scene_keeps_its_bytecode() {
    let now = measured();
    if std::env::var_os("ALICE_SDF_RECORD_BYTECODE").is_some() {
        let mut text = String::from(
            "# scene<TAB>instructions<TAB>FNV-1a of the bytecode: see tests/test_compile_bytecode_golden.rs\n",
        );
        for (name, n, h) in &now {
            text.push_str(&format!("{name}\t{n}\t0x{h:016x}\n"));
        }
        std::fs::write(TABLE, text).expect("write the table");
    }
    let table = recorded();
    let mut compared = 0;
    let mut differ = Vec::new();
    for (name, n, h) in &now {
        match table.iter().find(|(t, _, _)| t == name) {
            Some((_, tn, th)) => {
                compared += 1;
                if (tn, th) != (n, h) {
                    differ.push(format!(
                        "{name}: {n} instructions 0x{h:016x} vs recorded {tn} 0x{th:016x}"
                    ));
                }
            }
            None => differ.push(format!("{name}: not in {TABLE}")),
        }
    }
    for (name, _, _) in &table {
        if !now.iter().any(|(t, _, _)| t == name) {
            differ.push(format!("{name}: in {TABLE} but no scene has it"));
        }
    }
    assert!(compared > 0, "nothing compared (empty {TABLE}?)");
    assert!(
        differ.is_empty(),
        "{compared} compared:\n{}",
        differ.join("\n")
    );
    // not vacuous: the deepest chain compiles to one instruction per node and more
    let deep = now
        .iter()
        .find(|(t, _, _)| t.ends_with("sphere_chain_1024"));
    assert!(deep.is_some_and(|&(_, n, _)| n > 3 * 1024), "{deep:?}");
}
