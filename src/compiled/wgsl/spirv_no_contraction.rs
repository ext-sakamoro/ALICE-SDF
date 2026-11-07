//! Marks every floating-point arithmetic result of a SPIR-V module `NoContraction`.
//!
//! ALICE-Law G3 needs `a * b + c` to round twice. naga's SPIR-V carries no
//! `NoContraction`, so a Vulkan driver may fuse the two operations, and it may do so
//! differently for two pipelines that hold the same expression. This pass adds the
//! decoration to the module naga wrote, which is then read through SPIR-V passthrough.
//!
//! The decoration only reaches an arithmetic instruction the module spells out. A driver
//! still fuses the sum inside `OpDot` or `length`, decorated or not, and whether it does
//! depends on what else the shader computes: `length(v.xz)` next to `length(v)` shares `x * x`
//! and rounds differently from `length(v.xz)` alone. So the builtins with a sum inside are first
//! written out as marked component arithmetic ([`expand_builtins`]), and those with no expansion
//! here are refused, not passed through.
//!
//! The marking list is by opcode, so it needs no type tracking: each opcode in [`ARITHMETIC`]
//! takes floating-point operands by definition.
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

const OP_EXT_INST_IMPORT: u32 = 11;
const OP_EXT_INST: u32 = 12;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_COMPOSITE_CONSTRUCT: u32 = 80;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_FADD: u32 = 129;
const OP_FSUB: u32 = 131;
const OP_FMUL: u32 = 133;
const OP_FDIV: u32 = 136;
const OP_DOT: u32 = 148;
/// `OpVectorTimesMatrix`, `OpMatrixTimesVector`, `OpMatrixTimesMatrix`: a sum inside.
const MATRIX_SUMS: [u32; 3] = [144, 145, 146];

const GLSL_STD_450: &[u32] = &[0x4c53_4c47, 0x6474_732e, 0x3035_342e, 0];
const GLSL_FMIX: u32 = 46;
const GLSL_SQRT: u32 = 31;
const GLSL_LENGTH: u32 = 66;
const GLSL_DISTANCE: u32 = 67;
const GLSL_CROSS: u32 = 68;
const GLSL_NORMALIZE: u32 = 69;
/// `SmoothStep`, `FaceForward`, `Reflect`, `Refract`, `Determinant`, `MatrixInverse`: a sum inside
/// and no expansion here.
const GLSL_REFUSED: [u32; 6] = [49, 70, 71, 72, 33, 34];

/// Why a word stream could not be read as a SPIR-V module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpirvError {
    /// Shorter than the header, or the first word is not the magic number.
    NotSpirv,
    /// An instruction at this word offset has length 0, runs past the end, is an
    /// arithmetic opcode too short to hold a result id, or names an operand of unknown type.
    Malformed(usize),
    /// A core opcode with a sum inside and no expansion here.
    UnsupportedOpcode(u32),
    /// A `GLSL.std.450` instruction with a sum inside and no expansion here.
    UnsupportedBuiltin(u32),
}

/// Rewrites a module's instructions, taking ids past its bound.
struct Expander {
    out: Vec<u32>,
    bound: u32,
}

impl Expander {
    const fn id(&mut self) -> u32 {
        let id = self.bound;
        self.bound += 1;
        id
    }

    fn op(&mut self, opcode: u32, operands: &[u32]) {
        self.out.push(((operands.len() as u32 + 1) << 16) | opcode);
        self.out.extend_from_slice(operands);
    }

    fn binary(&mut self, opcode: u32, ty: u32, a: u32, b: u32) -> u32 {
        let id = self.id();
        self.op(opcode, &[ty, id, a, b]);
        id
    }

    fn components(&mut self, ty: u32, vector: u32, size: u32) -> Vec<u32> {
        (0..size)
            .map(|i| {
                let id = self.id();
                self.op(OP_COMPOSITE_EXTRACT, &[ty, id, vector, i]);
                id
            })
            .collect()
    }

    /// `a0 * b0 + a1 * b1 + ...`, left to right, into `result` when given.
    fn sum_of_products(&mut self, ty: u32, a: &[u32], b: &[u32], result: Option<u32>) -> u32 {
        let products: Vec<u32> = a
            .iter()
            .zip(b)
            .map(|(x, y)| self.binary(OP_FMUL, ty, *x, *y))
            .collect();
        let (last, firsts) = products.split_last().expect("a vector has components");
        let mut sum = None;
        for product in firsts {
            sum = Some(match sum {
                None => *product,
                Some(s) => self.binary(OP_FADD, ty, s, *product),
            });
        }
        let id = result.unwrap_or_else(|| self.id());
        self.op(OP_FADD, &[ty, id, sum.unwrap_or(*last), *last]);
        id
    }

    fn sqrt(&mut self, ty: u32, set: u32, value: u32, result: Option<u32>) -> u32 {
        let id = result.unwrap_or_else(|| self.id());
        self.op(OP_EXT_INST, &[ty, id, set, GLSL_SQRT, value]);
        id
    }
}

/// An opcode whose first two operands are a result type and a result id. Only what comes before a
/// use of a vector value in a module this crate compiles: an operand of any other origin has no
/// known type, and the expansion refuses it.
const fn has_type_and_result(opcode: u32) -> bool {
    matches!(
        opcode,
        1 | 12 | 41..=46 | 55 | 57 | 59 | 61 | 65 | 66 | 79..=83 | 109..=215 | 245
    )
}

/// Writes the builtins with a sum inside as component arithmetic: `OpDot`, `length`, `distance`,
/// `normalize`, `cross` and `mix`, each keeping its result id. A scalar `length`, `distance` or
/// `normalize` has no sum and stays.
///
/// # Errors
/// [`SpirvError`] when `module` is not well-formed or holds a builtin with a sum inside that has
/// no expansion here.
fn expand_builtins(module: &[u32]) -> Result<Vec<u32>, SpirvError> {
    let mut floats = Vec::new();
    let mut vectors = std::collections::BTreeMap::new();
    let mut types = std::collections::BTreeMap::new();
    let mut glsl = None;
    let mut first_function = None;
    let mut at = HEADER_WORDS;
    while at < module.len() {
        let length = (module[at] >> 16) as usize;
        let opcode = module[at] & 0xffff;
        if length == 0 || at + length > module.len() {
            return Err(SpirvError::Malformed(at));
        }
        let operands = &module[at + 1..at + length];
        match opcode {
            OP_EXT_INST_IMPORT if operands.len() > 1 && operands[1..] == *GLSL_STD_450 => {
                glsl = Some(operands[0]);
            }
            OP_TYPE_FLOAT if !operands.is_empty() => floats.push(operands[0]),
            OP_TYPE_VECTOR if operands.len() == 3 => {
                vectors.insert(operands[0], (operands[1], operands[2]));
            }
            OP_FUNCTION if first_function.is_none() => first_function = Some(at),
            _ => {}
        }
        if has_type_and_result(opcode) && operands.len() >= 2 {
            types.insert(operands[1], operands[0]);
        }
        at += length;
    }

    let mut e = Expander {
        out: Vec::with_capacity(module.len() * 2),
        bound: module[3],
    };
    e.out.extend_from_slice(&module[..HEADER_WORDS]);
    // `mix` needs 1.0, and a constant has to come before the functions
    let mut ones = std::collections::BTreeMap::new();
    let mut at = HEADER_WORDS;
    while at < module.len() {
        let length = (module[at] >> 16) as usize;
        let opcode = module[at] & 0xffff;
        let operands = &module[at + 1..at + length];
        if Some(at) == first_function {
            for float in &floats {
                let one = e.id();
                e.op(OP_CONSTANT, &[*float, one, 1.0f32.to_bits()]);
                ones.insert(*float, one);
            }
        }
        let vector_of = |id: u32| types.get(&id).and_then(|ty| vectors.get(ty)).copied();
        let unknown = SpirvError::Malformed(at);
        if MATRIX_SUMS.contains(&opcode) {
            return Err(SpirvError::UnsupportedOpcode(opcode));
        }
        if opcode == OP_DOT {
            let [ty, result, a, b] = *operands else {
                return Err(unknown);
            };
            let (_, size) = vector_of(a).ok_or(unknown)?;
            let (xs, ys) = (e.components(ty, a, size), e.components(ty, b, size));
            e.sum_of_products(ty, &xs, &ys, Some(result));
            at += length;
            continue;
        }
        if opcode == OP_EXT_INST && operands.len() >= 4 && Some(operands[2]) == glsl {
            let (ty, result, set, builtin) = (operands[0], operands[1], operands[2], operands[3]);
            let args = &operands[4..];
            if GLSL_REFUSED.contains(&builtin) {
                return Err(SpirvError::UnsupportedBuiltin(builtin));
            }
            let expanded = match (builtin, args) {
                (GLSL_LENGTH, [v]) => vector_of(*v).map(|(c, size)| {
                    let xs = e.components(c, *v, size);
                    let sum = e.sum_of_products(c, &xs, &xs, None);
                    e.sqrt(c, set, sum, Some(result));
                }),
                (GLSL_DISTANCE, [a, b]) => vector_of(*a).map(|(c, size)| {
                    let (xs, ys) = (e.components(c, *a, size), e.components(c, *b, size));
                    let ds: Vec<u32> = xs
                        .iter()
                        .zip(&ys)
                        .map(|(x, y)| e.binary(OP_FSUB, c, *x, *y))
                        .collect();
                    let sum = e.sum_of_products(c, &ds, &ds, None);
                    e.sqrt(c, set, sum, Some(result));
                }),
                (GLSL_NORMALIZE, [v]) => vector_of(*v).map(|(c, size)| {
                    let xs = e.components(c, *v, size);
                    let sum = e.sum_of_products(c, &xs, &xs, None);
                    let len = e.sqrt(c, set, sum, None);
                    let mut construct = vec![ty, result];
                    for x in &xs {
                        construct.push(e.binary(OP_FDIV, c, *x, len));
                    }
                    e.op(OP_COMPOSITE_CONSTRUCT, &construct);
                }),
                (GLSL_CROSS, [a, b]) => {
                    let (c, size) = vector_of(*a).ok_or_else(|| unknown.clone())?;
                    let (x, y) = (e.components(c, *a, size), e.components(c, *b, size));
                    let mut construct = vec![ty, result];
                    for (i, j) in [(1, 2), (2, 0), (0, 1)] {
                        let p = e.binary(OP_FMUL, c, x[i], y[j]);
                        let q = e.binary(OP_FMUL, c, x[j], y[i]);
                        construct.push(e.binary(OP_FSUB, c, p, q));
                    }
                    e.op(OP_COMPOSITE_CONSTRUCT, &construct);
                    Some(())
                }
                (GLSL_FMIX, [x, y, a]) => {
                    let mix =
                        |e: &mut Expander, c: u32, x: u32, y: u32, a: u32, into: Option<u32>| {
                            let one = ones[&c];
                            let rest = e.binary(OP_FSUB, c, one, a);
                            let p = e.binary(OP_FMUL, c, x, rest);
                            let q = e.binary(OP_FMUL, c, y, a);
                            let id = into.unwrap_or_else(|| e.id());
                            e.op(OP_FADD, &[c, id, p, q]);
                            id
                        };
                    match vector_of(*x) {
                        Some((c, size)) => {
                            let xs = e.components(c, *x, size);
                            let ys = e.components(c, *y, size);
                            let aa = e.components(c, *a, size);
                            let mut construct = vec![ty, result];
                            for i in 0..size as usize {
                                construct.push(mix(&mut e, c, xs[i], ys[i], aa[i], None));
                            }
                            e.op(OP_COMPOSITE_CONSTRUCT, &construct);
                        }
                        None => {
                            mix(&mut e, ty, *x, *y, *a, Some(result));
                        }
                    }
                    Some(())
                }
                _ => None,
            };
            if expanded.is_some() {
                at += length;
                continue;
            }
        }
        e.out.extend_from_slice(&module[at..at + length]);
        at += length;
    }
    e.out[3] = e.bound;
    Ok(e.out)
}

/// Returns `module` with its builtins expanded ([`expand_builtins`]) and one
/// `OpDecorate <result> NoContraction` per arithmetic result.
///
/// # Errors
/// [`SpirvError`] when `module` is not a well-formed instruction stream, or holds a builtin with
/// no expansion here.
pub fn mark_no_contraction(module: &[u32]) -> Result<Vec<u32>, SpirvError> {
    if module.len() < HEADER_WORDS || module[0] != MAGIC {
        return Err(SpirvError::NotSpirv);
    }
    let expanded = expand_builtins(module)?;
    let module = &expanded[..];

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
        for opcode in (0..=400).filter(|o| *o != OP_DOT && !MATRIX_SUMS.contains(o)) {
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

    /// ids: 1 the import, 2 float, 3 vec2, 4 vec3, 5 and 6 a vec2 and a vec3 parameter, 7 a scalar
    fn with_body(body: Vec<u32>) -> Vec<u32> {
        let mut import = vec![1];
        import.extend_from_slice(GLSL_STD_450);
        module(&[
            op(OP_EXT_INST_IMPORT, &import),
            op(OP_TYPE_FLOAT, &[2, 32]),
            op(OP_TYPE_VECTOR, &[3, 2, 2]),
            op(OP_TYPE_VECTOR, &[4, 2, 3]),
            op(OP_FUNCTION, &[2, 40, 0, 41]),
            op(55, &[3, 5]),
            op(55, &[4, 6]),
            op(55, &[2, 7]),
            body,
        ])
    }

    /// The opcodes after the parameters, without the decorations, and the marked result ids.
    fn body_of(output: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let (mut ops, mut marked, mut at, mut params) = (Vec::new(), Vec::new(), 5, 0);
        while at < output.len() {
            let (length, opcode) = ((output[at] >> 16) as usize, output[at] & 0xffff);
            if opcode == OP_DECORATE && output[at + 2] == DECORATION_NO_CONTRACTION {
                marked.push(output[at + 1]);
            }
            if params == 3 {
                ops.push(opcode);
            }
            if opcode == 55 {
                params += 1;
            }
            at += length;
        }
        (ops, marked)
    }

    #[test]
    fn a_vector_length_becomes_marked_arithmetic_under_the_same_result() {
        // bound in HEADER is 100, so new ids start there
        let input = with_body(op(OP_EXT_INST, &[2, 50, 1, GLSL_LENGTH, 6]));
        let output = mark_no_contraction(&input).unwrap();
        let (ops, marked) = body_of(&output);
        assert_eq!(
            ops,
            [
                OP_COMPOSITE_EXTRACT,
                OP_COMPOSITE_EXTRACT,
                OP_COMPOSITE_EXTRACT,
                OP_FMUL,
                OP_FMUL,
                OP_FMUL,
                OP_FADD,
                OP_FADD,
                OP_EXT_INST
            ]
        );
        assert_eq!(marked.len(), 5, "three products and two sums: {marked:?}");
        assert!(output.windows(5).any(|w| w[0] == (6 << 16) | OP_EXT_INST
            && w[1] == 2
            && w[2] == 50
            && w[3] == 1
            && w[4] == GLSL_SQRT));
        assert!(output[3] > 100, "the bound grows past the new ids");
    }

    #[test]
    fn a_dot_keeps_its_result_and_sums_left_to_right() {
        let input = with_body(op(OP_DOT, &[2, 50, 5, 5]));
        let output = mark_no_contraction(&input).unwrap();
        let (ops, marked) = body_of(&output);
        assert_eq!(ops.iter().filter(|o| **o == OP_FMUL).count(), 2);
        assert_eq!(ops.iter().filter(|o| **o == OP_FADD).count(), 1);
        assert!(!ops.contains(&OP_DOT));
        assert!(marked.contains(&50));
    }

    #[test]
    fn normalize_cross_mix_and_distance_leave_no_builtin() {
        for (builtin, args) in [
            (GLSL_NORMALIZE, vec![6]),
            (GLSL_CROSS, vec![6, 6]),
            (GLSL_FMIX, vec![6, 6, 6]),
            (GLSL_FMIX, vec![7, 7, 7]),
            (GLSL_DISTANCE, vec![6, 6]),
        ] {
            let ty = if args[0] == 7 || builtin == GLSL_DISTANCE {
                2
            } else {
                4
            };
            let mut words = vec![ty, 50, 1, builtin];
            words.extend_from_slice(&args);
            let output = mark_no_contraction(&with_body(op(OP_EXT_INST, &words))).unwrap();
            let (ops, marked) = body_of(&output);
            assert!(ops.contains(&OP_FMUL), "builtin {builtin}");
            assert!(!marked.is_empty(), "builtin {builtin}");
            let left = output.windows(5).any(|w| {
                w[0] & 0xffff == OP_EXT_INST && w[1] == ty && w[2] == 50 && w[4] == builtin
            });
            assert!(!left, "builtin {builtin} is still there");
        }
    }

    #[test]
    fn a_scalar_length_has_no_sum_and_stays() {
        let body = op(OP_EXT_INST, &[2, 50, 1, GLSL_LENGTH, 7]);
        let output = mark_no_contraction(&with_body(body.clone())).unwrap();
        assert!(output.windows(body.len()).any(|w| w == body));
    }

    #[test]
    fn a_builtin_with_no_expansion_is_refused_not_passed_through() {
        for builtin in GLSL_REFUSED {
            let input = with_body(op(OP_EXT_INST, &[2, 50, 1, builtin, 7, 7, 7]));
            assert_eq!(
                mark_no_contraction(&input),
                Err(SpirvError::UnsupportedBuiltin(builtin))
            );
        }
        for opcode in MATRIX_SUMS {
            let input = with_body(op(opcode, &[2, 50, 5, 5]));
            assert_eq!(
                mark_no_contraction(&input),
                Err(SpirvError::UnsupportedOpcode(opcode))
            );
        }
    }

    #[test]
    fn a_dot_of_an_operand_of_unknown_type_is_refused() {
        let input = with_body(op(OP_DOT, &[2, 50, 77, 77]));
        assert!(matches!(
            mark_no_contraction(&input),
            Err(SpirvError::Malformed(_))
        ));
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
        overrun.push((0x4 << 16) | 0x11);
        assert_eq!(mark_no_contraction(&overrun), Err(SpirvError::Malformed(5)));
        let short_fadd = module(&[op(129, &[1])]);
        assert_eq!(
            mark_no_contraction(&short_fadd),
            Err(SpirvError::Malformed(5))
        );
    }
}
