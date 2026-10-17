//! The Law max / min as Cranelift IR (ALICE-Law G2 / G4)
//!
//! Cranelift's `fmax` / `fmin` order the zeros as the Law does (`-0 < +0`) but return
//! NaN when either operand is NaN. The Law (alice-det-math `max` / `min`, the tree and
//! bytecode evaluators) gives the other operand when one is NaN, and the canonical quiet
//! NaN (`0x7fc0_0000`) when both are. These emit `fmax` / `fmin` and then pick the
//! other operand and canonicalise, for `f32` and for `f32x4` lanes, so the JIT gives the
//! evaluators' bits on NaN operands too (tests/test_law_nan_operands.rs).

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{types, InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;

/// The canonical quiet NaN of the Law's outputs
const CANONICAL_NAN_BITS: u32 = 0x7fc0_0000;

/// Law `max(a, b)`
pub fn law_fmax(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
    let m = builder.ins().fmax(a, b);
    nan_rule(builder, a, b, m)
}

/// Law `min(a, b)`
pub fn law_fmin(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
    let m = builder.ins().fmin(a, b);
    nan_rule(builder, a, b, m)
}

/// `m` where neither operand is NaN, the other operand where one is, the canonical NaN
/// where both are
fn nan_rule(builder: &mut FunctionBuilder, a: Value, b: Value, m: Value) -> Value {
    let ty = builder.func.dfg.value_type(a);
    let a_nan = builder.ins().fcmp(FloatCC::Unordered, a, a);
    let b_nan = builder.ins().fcmp(FloatCC::Unordered, b, b);
    if ty.is_vector() {
        assert_eq!(
            ty,
            types::F32X4,
            "law max / min is emitted for f32 and f32x4 only"
        );
        let bits = |builder: &mut FunctionBuilder, v: Value| {
            builder.ins().bitcast(types::I32X4, MemFlags::new(), v)
        };
        let (ai, bi, mi) = (bits(builder, a), bits(builder, b), bits(builder, m));
        let r1 = builder.ins().bitselect(b_nan, ai, mi);
        let r2 = builder.ins().bitselect(a_nan, bi, r1);
        let r2f = builder.ins().bitcast(types::F32X4, MemFlags::new(), r2);
        let r_nan = builder.ins().fcmp(FloatCC::Unordered, r2f, r2f);
        let canon_scalar = builder
            .ins()
            .iconst(types::I32, i64::from(CANONICAL_NAN_BITS));
        let canon = builder.ins().splat(types::I32X4, canon_scalar);
        let out = builder.ins().bitselect(r_nan, canon, r2);
        builder.ins().bitcast(types::F32X4, MemFlags::new(), out)
    } else {
        assert_eq!(
            ty,
            types::F32,
            "law max / min is emitted for f32 and f32x4 only"
        );
        let r1 = builder.ins().select(b_nan, a, m);
        let r2 = builder.ins().select(a_nan, b, r1);
        let r_nan = builder.ins().fcmp(FloatCC::Unordered, r2, r2);
        let canon = builder.ins().f32const(f32::from_bits(CANONICAL_NAN_BITS));
        builder.ins().select(r_nan, canon, r2)
    }
}
