//! Every shader a GPU test builds is checked as SPIR-V before a driver sees it: naga turns the text
//! into SPIR-V and the word stream is walked. An instruction with a word count of 0 (a constant
//! composite of more than 65532 elements overflows the 16-bit count), or a last instruction that does
//! not end at the module's end, fails the test here instead of losing the device on Vulkan.
//!
//! Author: Moroya Sakamoto
#![allow(dead_code)]

/// The instructions of a SPIR-V word stream as (opcode, word count), or why it does not parse
pub fn instructions(words: &[u32]) -> Result<Vec<(u16, u16)>, String> {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err("not a SPIR-V module".into());
    }
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (count, op) = ((words[i] >> 16) as u16, (words[i] & 0xffff) as u16);
        if count == 0 {
            return Err(format!(
                "instruction {} (word {i}, opcode {op}) has a word count of 0",
                out.len()
            ));
        }
        out.push((op, count));
        i += count as usize;
    }
    if i != words.len() {
        return Err(format!(
            "the last instruction ends at word {i}, the module has {}",
            words.len()
        ));
    }
    Ok(out)
}

/// The SPIR-V naga writes for `module`, after naga's validation
pub fn spirv(module: &naga::Module) -> Vec<u32> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .expect("naga validates the module");
    naga::back::spv::write_vec(module, &info, &naga::back::spv::Options::default(), None)
        .expect("SPIR-V")
}

/// Panics unless the WGSL compute shader `src` becomes SPIR-V whose every instruction fits; returns
/// the instruction count (a module of 0 instructions also panics)
pub fn assert_wgsl_encodes(src: &str) -> usize {
    let module = naga::front::wgsl::parse_str(src).expect("naga parses the WGSL");
    let n = instructions(&spirv(&module))
        .unwrap_or_else(|e| panic!("WGSL to SPIR-V: {e}"))
        .len();
    assert!(n > 0, "an empty SPIR-V module");
    n
}

/// [`assert_wgsl_encodes`] for a GLSL compute shader
pub fn assert_glsl_encodes(src: &str) -> usize {
    let mut fe = naga::front::glsl::Frontend::default();
    let module = fe
        .parse(
            &naga::front::glsl::Options::from(naga::ShaderStage::Compute),
            src,
        )
        .expect("naga parses the GLSL");
    let n = instructions(&spirv(&module))
        .unwrap_or_else(|e| panic!("GLSL to SPIR-V: {e}"))
        .len();
    assert!(n > 0, "an empty SPIR-V module");
    n
}

/// `src` after [`assert_glsl_encodes`], for a GLSL evaluator's construction
pub fn checked_glsl(src: String) -> String {
    assert_glsl_encodes(&src);
    src
}
