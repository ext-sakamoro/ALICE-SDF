//! Marks every floating-point arithmetic result of a SPIR-V module `NoContraction`.
//!
//! ALICE-Law G3 needs `a * b + c` to round twice. naga's SPIR-V carries no
//! `NoContraction`, so a Vulkan driver may fuse the two operations, and it may do so
//! differently for two pipelines that hold the same expression. This pass adds the
//! decoration to the module naga wrote, which is then read through SPIR-V passthrough.
//!
//! The list is by opcode, so no type tracking is needed: each opcode in
//! [`ARITHMETIC`] takes floating-point operands by definition. `OpExtInst` is not on
//! the list, since the decoration applies to arithmetic instructions: what a driver
//! does inside `length` or `normalize` is not reached by this pass.
//!
//! Author: Moroya Sakamoto

/// First word of every SPIR-V module.
const MAGIC: u32 = 0x0723_0203;
/// Words before the first instruction.
const HEADER_WORDS: usize = 5;
const OP_DECORATE: u32 = 71;
const DECORATION_NO_CONTRACTION: u32 = 42;

/// `OpFNegate`, `OpFAdd`, `OpFSub`, `OpFMul`, `OpFDiv`, `OpFRem`, `OpFMod`,
/// `OpVectorTimesScalar`, `OpMatrixTimesScalar`, `OpVectorTimesMatrix`,
/// `OpMatrixTimesVector`, `OpMatrixTimesMatrix`, `OpOuterProduct`, `OpDot`.
const ARITHMETIC: [u32; 14] = [
    127, 129, 131, 133, 136, 140, 141, 142, 143, 144, 145, 146, 147, 148,
];

/// Opcodes that come before the types in the module's required order: capabilities,
/// extensions, the memory model, entry points, debug names and annotations. A new
/// `OpDecorate` belongs right after them.
const BEFORE_TYPES: [u32; 22] = [
    2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 71, 72, 73, 74, 75, 330, 331, 332, 5632, 5633,
];

/// Why a word stream could not be read as a SPIR-V module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpirvError {
    /// Shorter than the header, or the first word is not the magic number.
    NotSpirv,
    /// An instruction at this word offset has length 0, runs past the end, or is an
    /// arithmetic opcode too short to hold a result id.
    Malformed(usize),
}

/// Returns `module` with one `OpDecorate <result> NoContraction` per arithmetic result.
///
/// # Errors
/// [`SpirvError`] when `module` is not a well-formed instruction stream.
pub(crate) fn mark_no_contraction(module: &[u32]) -> Result<Vec<u32>, SpirvError> {
    if module.len() < HEADER_WORDS || module[0] != MAGIC {
        return Err(SpirvError::NotSpirv);
    }

    let mut results = Vec::new();
    let mut insert_at = None;
    let mut at = HEADER_WORDS;
    while at < module.len() {
        let length = (module[at] >> 16) as usize;
        let opcode = module[at] & 0xffff;
        if length == 0 || at + length > module.len() {
            return Err(SpirvError::Malformed(at));
        }
        if insert_at.is_none() && !BEFORE_TYPES.contains(&opcode) {
            insert_at = Some(at);
        }
        if ARITHMETIC.contains(&opcode) {
            // Result type, then result id.
            if length < 3 {
                return Err(SpirvError::Malformed(at));
            }
            results.push(module[at + 2]);
        }
        at += length;
    }

    let insert_at = insert_at.unwrap_or(module.len());
    let mut marked = Vec::with_capacity(module.len() + results.len() * 3);
    marked.extend_from_slice(&module[..insert_at]);
    for result in results {
        marked.extend_from_slice(&[(3 << 16) | OP_DECORATE, result, DECORATION_NO_CONTRACTION]);
    }
    marked.extend_from_slice(&module[insert_at..]);
    Ok(marked)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: [u32; 5] = [MAGIC, 0x0001_0300, 0, 100, 0];

    fn op(opcode: u32, operands: &[u32]) -> Vec<u32> {
        let mut words = vec![((operands.len() as u32 + 1) << 16) | opcode];
        words.extend_from_slice(operands);
        words
    }

    fn module(instructions: &[Vec<u32>]) -> Vec<u32> {
        let mut words = HEADER.to_vec();
        for instruction in instructions {
            words.extend_from_slice(instruction);
        }
        words
    }

    fn decorate(id: u32) -> Vec<u32> {
        op(OP_DECORATE, &[id, DECORATION_NO_CONTRACTION])
    }

    /// Capability, a name, one existing decoration, a float type, an integer add and
    /// `%10 = a * b`, `%11 = %10 + c`.
    fn sample() -> [Vec<u32>; 6] {
        [
            op(17, &[1]),
            op(5, &[10, 0x0061]),
            op(OP_DECORATE, &[3, 2]),
            op(22, &[1, 32]),
            op(128, &[2, 9, 4, 5]),
            op(129, &[1, 11, 10, 6]),
        ]
    }

    #[test]
    fn a_decoration_goes_after_the_annotations_and_before_the_types() {
        let [capability, name, existing, float, iadd, fadd] = sample();
        let fmul = op(133, &[1, 10, 4, 5]);
        let input = module(&[
            capability.clone(),
            name.clone(),
            existing.clone(),
            float.clone(),
            iadd.clone(),
            fmul.clone(),
            fadd.clone(),
        ]);
        let expected = module(&[
            capability,
            name,
            existing,
            decorate(10),
            decorate(11),
            float,
            iadd,
            fmul,
            fadd,
        ]);
        assert_eq!(mark_no_contraction(&input), Ok(expected));
    }

    #[test]
    fn every_arithmetic_opcode_is_marked_and_nothing_else() {
        for opcode in 0..=400 {
            let input = module(&[op(19, &[1]), op(opcode, &[1, 50, 2, 3])]);
            let output = mark_no_contraction(&input).unwrap();
            let decorated = output
                .windows(3)
                .any(|w| w == [(3 << 16) | OP_DECORATE, 50, DECORATION_NO_CONTRACTION]);
            assert_eq!(decorated, ARITHMETIC.contains(&opcode), "opcode {opcode}");
        }
    }

    #[test]
    fn a_module_without_arithmetic_comes_back_unchanged() {
        let input = module(&[op(17, &[1]), op(19, &[1])]);
        assert_eq!(mark_no_contraction(&input).as_deref(), Ok(&input[..]));
    }

    #[test]
    fn a_module_of_only_the_opening_section_comes_back_unchanged() {
        let input = module(&[op(17, &[1])]);
        assert_eq!(mark_no_contraction(&input).as_deref(), Ok(&input[..]));
    }

    #[test]
    fn what_is_not_a_module_is_refused() {
        assert_eq!(mark_no_contraction(&[]), Err(SpirvError::NotSpirv));
        assert_eq!(
            mark_no_contraction(&[0, 0, 0, 0, 0]),
            Err(SpirvError::NotSpirv)
        );
        let mut zero_length = HEADER.to_vec();
        zero_length.push(17);
        assert_eq!(
            mark_no_contraction(&zero_length),
            Err(SpirvError::Malformed(5))
        );
        let mut overrun = HEADER.to_vec();
        overrun.push((4 << 16) | 17);
        assert_eq!(mark_no_contraction(&overrun), Err(SpirvError::Malformed(5)));
        let short_fadd = module(&[op(129, &[1])]);
        assert_eq!(
            mark_no_contraction(&short_fadd),
            Err(SpirvError::Malformed(5))
        );
    }
}
