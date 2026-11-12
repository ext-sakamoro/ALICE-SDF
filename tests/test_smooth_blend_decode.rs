//! `smooth_blend_decode` is total: every code pair the compiler writes decodes
//! to its kernel and mode, every other pair is `None` (no panic), and the
//! bytecode refit refuses an undecodable `SmoothBlend` instruction with a typed
//! error (no bound is guessed for it).

use alice_sdf::compiled::{CompiledSdfBvh, OpCode, RefitError};
use alice_sdf::operations::{smooth_blend_decode, BlendMode, SmoothKernel};
use alice_sdf::prelude::*;

#[test]
fn every_written_code_decodes_and_every_other_code_is_none() {
    let mut n = 0;
    for kernel in SmoothKernel::ALL {
        for mode in [
            BlendMode::Union,
            BlendMode::Intersection,
            BlendMode::Subtraction,
        ] {
            assert_eq!(
                smooth_blend_decode(kernel.code() as f32, mode.code() as f32),
                Some((kernel, mode))
            );
            n += 1;
        }
    }
    assert_eq!(n, 9);
    for (k, m) in [
        (3.0, 0.0),
        (0.0, 3.0),
        (0.5, 0.0),
        (0.0, 1.5),
        (-1.0, 0.0),
        (f32::NAN, 0.0),
        (0.0, f32::INFINITY),
        (4_294_967_296.0, 0.0),
    ] {
        assert_eq!(smooth_blend_decode(k, m), None, "codes {k} / {m}");
    }
}

#[test]
fn refit_refuses_an_undecodable_instruction() {
    let a = SdfNode::sphere(0.5);
    let b = SdfNode::sphere(0.5).translate(2.0, 0.0, 0.0);
    let mut refused = 0;
    for mode in [
        BlendMode::Union,
        BlendMode::Intersection,
        BlendMode::Subtraction,
    ] {
        let node = SdfNode::smooth_blend(a.clone(), b.clone(), 0.3, SmoothKernel::Circular, mode);
        let mut bvh = CompiledSdfBvh::compile(&node);
        let before = bvh.scene_aabb;
        let i = bvh
            .instructions
            .iter()
            .position(|x| x.opcode == OpCode::SmoothBlend)
            .expect("a SmoothBlend instruction");
        // a valid program refits
        bvh.refit_all_from_bytecode()
            .expect("refit of the compiled program");
        for (slot, code) in [(1, 3.0), (2, 7.0), (1, 0.5)] {
            let mut bad = CompiledSdfBvh::compile(&node);
            bad.instructions[i].params[slot] = code;
            match bad.refit_all_from_bytecode() {
                Err(RefitError::UnsupportedOpcode {
                    opcode: OpCode::SmoothBlend,
                    instruction_index,
                }) => assert_eq!(instruction_index, i),
                other => panic!("{mode:?} slot {slot} code {code}: {other:?}"),
            }
            assert_eq!(
                bad.scene_aabb.min(),
                before.min(),
                "bounds changed on a refused refit"
            );
            refused += 1;
        }
    }
    assert_eq!(refused, 9);
}
