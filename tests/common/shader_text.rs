//! The shader rule of ALICE-Law G8, shared by the tests that check emitted text: whether a text
//! holds a shader entry point (a whole shader, which carries the semantics identifier) or not (a
//! fragment pasted into a host file, which carries none). One copy, used by
//! tests/test_shader_text_provenance.rs and tests/test_g8_example_text.rs
//!
//! Author: Moroya Sakamoto

/// The text of `text` outside `//` comments
pub fn code_of(text: &str) -> String {
    text.lines()
        .map(|l| l.find("//").map_or(l, |i| &l[..i]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether `text` holds a shader entry point, the mark of a whole shader (one an API or compiler
/// takes as is) as opposed to a fragment pasted into a host file. The markers are those the
/// transpilers write: WGSL stage attributes, GLSL `void main(`, HLSL `[numthreads(` and the
/// `SV_Target` / `SV_Position` stage semantics (case-insensitive in HLSL), MSL `kernel` / `fragment` / `vertex` functions (naga
/// output), and the BlinkScript `ImageComputationKernel` container
pub fn has_entry_point(text: &str) -> bool {
    let code = code_of(text);
    let line_starts = |w: &str| code.lines().any(|l| l.trim_start().starts_with(w));
    ["@compute", "@fragment", "@vertex"]
        .iter()
        .any(|m| code.contains(m))
        || code.contains("void main(")
        || code.contains("[numthreads(")
        || code.to_ascii_uppercase().contains("SV_TARGET")
        || code.to_ascii_uppercase().contains("SV_POSITION")
        || line_starts("kernel void ")
        || line_starts("fragment ")
        || line_starts("vertex ")
        || code.contains(": ImageComputationKernel")
}
