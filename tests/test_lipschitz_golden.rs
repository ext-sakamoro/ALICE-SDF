//! `interval::eval_lipschitz` keeps its values: a recorded bit pattern per scene
//!
//! `tests/fixtures/lipschitz_golden.tsv` holds the bound `eval_lipschitz` returns for every scene
//! of the shared corpus (every `SdfNode` variant at the root at least once, with `Triangle` and
//! `Bezier`, which the corpus leaves out because the compiler refuses them), plus a twisted and a
//! bent scene with a bounded child (their bound reads the child's box) and the deep chains the
//! stack budget tests build. The bound is read by the bytecode compiler, the JIT, the fidelity
//! report and the ray marcher (`with_bound`), so a change of any bit changes what they do.
//!
//! The table was recorded before the recursion of `eval_lipschitz` was split into a thin
//! recursive step and a non-recursive law (to keep its stack frame small); the test compares
//! every value bitwise, fails on a scene the table does not have and on a table row no scene
//! has, and fails when nothing was compared. Record a new table (only for an intended change of a
//! bound) with `ALICE_SDF_RECORD_LIPSCHITZ=1`.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::interval::eval_lipschitz;
use alice_sdf::SdfNode;
use common::corpus::corpus;
use glam::Vec3;

const TABLE: &str = "tests/fixtures/lipschitz_golden.tsv";

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

/// Every scene, named `<index>:<name>` (corpus names are not unique)
fn scenes() -> Vec<(String, SdfNode)> {
    let mut v = corpus();
    v.push((
        "triangle",
        SdfNode::triangle(
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.6, 0.2),
        ),
    ));
    v.push((
        "bezier",
        SdfNode::bezier(
            Vec3::new(-0.6, 0.0, 0.0),
            Vec3::new(0.0, 0.8, 0.0),
            Vec3::new(0.6, 0.0, 0.0),
            0.1,
        ),
    ));
    v.push((
        "projective_transform_nonidentity",
        common::corpus::projective_nonidentity(),
    ));
    v.push(("twist_of_box", SdfNode::box3d(0.4, 0.8, 0.3).twist(1.5)));
    v.push(("bend_of_box", SdfNode::box3d(0.8, 0.2, 0.3).bend(0.7)));
    // the one two-child law that is not symmetric in its children: children with different
    // bounds and a weight other than 1/2, so a swap of the children's bounds changes the value
    v.push((
        "morph_quarter_twist_sphere",
        SdfNode::box3d(0.4, 0.8, 0.3)
            .twist(1.5)
            .morph(SdfNode::sphere(1.0), 0.25),
    ));
    // a non-uniform scale with factors whose ratio is not exact, over a child whose bound is not 1:
    // the order of `bound · max / min` decides the last bit
    v.push((
        "scale_non_uniform_of_noise",
        SdfNode::sphere(1.0)
            .noise(0.37, 2.3, 7)
            .scale_xyz(0.3, 0.7, 1.1),
    ));
    v.push((
        "noise_over_chamfer",
        SdfNode::sphere(1.0)
            .chamfer_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.1)
            .noise(0.05, 3.0, 7),
    ));
    v.push(("sphere_chain_48", sphere_chain(48)));
    v.push(("sphere_chain_2048", sphere_chain(2048)));
    v.into_iter()
        .enumerate()
        .map(|(i, (name, node))| (format!("{i}:{name}"), node))
        .collect()
}

/// Large enough for the deep chain whatever the frame of the recursion
const THREAD_BYTES: usize = 256 * 1024 * 1024;

fn measured() -> Vec<(String, u32)> {
    std::thread::Builder::new()
        .stack_size(THREAD_BYTES)
        .spawn(|| {
            scenes()
                .into_iter()
                .map(|(name, node)| (name, eval_lipschitz(&node).to_bits()))
                .collect()
        })
        .expect("spawn a thread")
        .join()
        .expect("the thread finished")
}

fn recorded() -> Vec<(String, u32)> {
    std::fs::read_to_string(TABLE)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (name, bits) = l.split_once('\t').expect("scene<TAB>bits");
            let bits = bits.strip_prefix("0x").expect("hex bits");
            (
                name.to_owned(),
                u32::from_str_radix(bits, 16).expect("hex bits"),
            )
        })
        .collect()
}

#[test]
fn every_bound_keeps_its_bits() {
    let now = measured();
    if std::env::var_os("ALICE_SDF_RECORD_LIPSCHITZ").is_some() {
        let mut text = String::from(
            "# scene<TAB>eval_lipschitz bits (f32): see tests/test_lipschitz_golden.rs\n",
        );
        for (name, bits) in &now {
            text.push_str(&format!("{name}\t0x{bits:08x}\n"));
        }
        std::fs::write(TABLE, text).expect("write the table");
    }
    let table = recorded();
    let mut compared = 0;
    let mut differ = Vec::new();
    for (name, bits) in &now {
        match table.iter().find(|(n, _)| n == name) {
            Some((_, want)) => {
                compared += 1;
                if want != bits {
                    differ.push(format!(
                        "{name}: {} (0x{bits:08x}) vs recorded {} (0x{want:08x})",
                        f32::from_bits(*bits),
                        f32::from_bits(*want)
                    ));
                }
            }
            None => differ.push(format!("{name}: not in {TABLE}")),
        }
    }
    for (name, _) in &table {
        if !now.iter().any(|(n, _)| n == name) {
            differ.push(format!("{name}: in {TABLE} but no scene has it"));
        }
    }
    assert!(compared > 0, "nothing compared (empty {TABLE}?)");
    assert!(
        differ.is_empty(),
        "{compared} compared:\n{}",
        differ.join("\n")
    );
    // the table is not vacuous: finite and infinite bounds, and bounds above 1
    let finite = now
        .iter()
        .filter(|(_, b)| f32::from_bits(*b).is_finite())
        .count();
    let above_one = now.iter().filter(|(_, b)| f32::from_bits(*b) > 1.0).count();
    assert!(
        finite > 0 && finite < now.len() && above_one > 0,
        "{finite} finite / {} scenes, {above_one} above 1",
        now.len()
    );
}
