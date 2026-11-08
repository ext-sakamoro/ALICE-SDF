//! One WGSL constructor for the bytecode evaluator's context, derived from the crate's field
//! table (`WGSL_BYTECODE_CTX_FIELDS`): every host shader in the tests builds the context
//! through this, so a member added to the table reaches each of them.
//!
//! Author: Moroya Sakamoto
#![allow(dead_code, reason = "shared by several test files through #[path]")]

use alice_sdf::npr::compiled_color::WGSL_BYTECODE_CTX_FIELDS;

/// `AliceNprBytecodeCtx(…)` from one scalar expression per component, in table order:
/// a width-1 member takes its scalar as is, a wider one is wrapped in `vecN<f32>(…)`.
/// Panics when the number of scalars is not the table's (the test is then wrong).
pub fn ctx_constructor(scalars: &[&str]) -> String {
    let expected: usize = WGSL_BYTECODE_CTX_FIELDS
        .iter()
        .map(|&(_, w)| usize::from(w))
        .sum();
    assert_eq!(
        scalars.len(),
        expected,
        "the context has {expected} scalars, {} were given",
        scalars.len()
    );
    let mut members = Vec::with_capacity(WGSL_BYTECODE_CTX_FIELDS.len());
    let mut next = 0;
    for &(_, width) in WGSL_BYTECODE_CTX_FIELDS {
        let w = usize::from(width);
        let part = &scalars[next..next + w];
        next += w;
        members.push(if w == 1 {
            part[0].to_string()
        } else {
            format!("vec{w}<f32>({})", part.join(", "))
        });
    }
    format!("AliceNprBytecodeCtx({})", members.join(", "))
}
