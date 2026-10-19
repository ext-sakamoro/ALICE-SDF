//! ALICE-Law G3 on the shader text the transpilers emit: no fused multiply-add
//! intrinsic (`fma(`, `fmaf(`, `mad(`) in any of it.
//!
//! The CPU paths round `a * b + c` twice (clippy bans `f32::mul_add`, wide's
//! fused methods and cranelift's `fma`), so a fused intrinsic in the GPU text
//! would give the GPU path other bits, and a negated one is what G3 forbids.
//! `scripts/shader_fma_guard.py` reads the string literals of `src/`; this test
//! reads what the transpilers actually return, so text composed from pieces at
//! run time is seen too. Every entry of the shared node corpus goes through
//! every backend whose feature is on, in both modes:
//!
//! | backend     | feature       | text scanned                                    |
//! |-------------|---------------|-------------------------------------------------|
//! | GLSL        | `glsl`        | `GlslShader::transpile(..).source`              |
//! | WGSL        | `gpu`         | `WgslShader::transpile(..).source`              |
//! | HLSL        | `hlsl`        | `HlslShader::transpile(..).source`              |
//! | BlinkScript | `blinkscript` | `BlinkScriptShader::transpile(..).to_kernel(..)` |
//! | MSL         | `msl`         | `MslShader::transpile(..)`: `source` (naga's MSL) and `wgsl_source` |
//!
//! Fails on a hit, and when a backend whose feature is on compared no text
//! (an empty corpus or a transpiler that failed on every entry). CI runs it
//! with `msl,glsl,hlsl,blinkscript` so every backend is compared.
//!
//! Not covered (a stated residual): a GPU driver may contract `a * b + c` into
//! an fma by itself; no emitted text shows that.
//!
//! Author: Moroya Sakamoto

#[cfg(any(
    feature = "glsl",
    feature = "gpu",
    feature = "hlsl",
    feature = "blinkscript",
    feature = "msl"
))]
mod common;

/// The fused multiply-add intrinsics of the shader languages this crate emits,
/// as `name(` with any spaces before the parenthesis and not part of a longer
/// identifier (`nomad(` and `fmadd(` do not count; `metal::fma(` does).
const INTRINSICS: [&str; 3] = ["fmaf", "fma", "mad"];

fn fused_calls(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut found = Vec::new();
    for name in INTRINSICS {
        let mut from = 0;
        while let Some(off) = text[from..].find(name) {
            let at = from + off;
            let end = at + name.len();
            let before_ok = at == 0 || !ident(bytes[at - 1]);
            let mut k = end;
            while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                k += 1;
            }
            if before_ok
                && k < bytes.len()
                && bytes[k] == b'('
                && (end == bytes.len() || !ident(bytes[end]))
            {
                let line = text[..at].matches('\n').count() + 1;
                found.push(format!("line {line}: `{}`", &text[at..=k]));
            }
            from = end;
        }
    }
    found
}

#[test]
fn the_scanner_flags_each_language_spelling() {
    // a canary per backend, written the way that language spells it
    for text in [
        "let d = -fma(a, b, c);",                   // WGSL
        "float d = -fma(a, b, c);",                 // GLSL
        "float d = -mad(a, b, c);",                 // HLSL / BlinkScript
        "float d = -metal::precise::fma(a, b, c);", // MSL
        "float d = -fmaf (a, b, c);",
    ] {
        assert_eq!(fused_calls(text).len(), 1, "{text}");
    }
    for text in [
        "nomad(x)",
        "my_fma(a)",
        "fmadd(b)",
        "madness(c)",
        "a * b + c",
        "fma",
    ] {
        assert!(fused_calls(text).is_empty(), "{text}");
    }
}

#[cfg(any(
    feature = "glsl",
    feature = "gpu",
    feature = "hlsl",
    feature = "blinkscript",
    feature = "msl"
))]
struct Tally {
    backend: &'static str,
    compared: usize,
    hits: Vec<String>,
}

#[cfg(any(
    feature = "glsl",
    feature = "gpu",
    feature = "hlsl",
    feature = "blinkscript",
    feature = "msl"
))]
impl Tally {
    const fn new(backend: &'static str) -> Self {
        Self {
            backend,
            compared: 0,
            hits: Vec::new(),
        }
    }

    fn scan(&mut self, entry: &str, mode: &str, text: &str) {
        self.compared += 1;
        for hit in fused_calls(text) {
            self.hits
                .push(format!("{} {entry} ({mode}) {hit}", self.backend));
        }
    }

    fn check(&self) {
        assert!(
            self.compared > 0,
            "{}: no emitted text was compared",
            self.backend
        );
        assert!(
            self.hits.is_empty(),
            "fused multiply-add in emitted shader text:\n{}",
            self.hits.join("\n")
        );
        eprintln!(
            "{}: {} emitted texts, 0 fused multiply-add intrinsics",
            self.backend, self.compared
        );
    }
}

#[cfg(feature = "glsl")]
#[test]
fn glsl_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
    let mut t = Tally::new("GLSL");
    for (name, node) in common::corpus::corpus() {
        t.scan(
            name,
            "hardcoded",
            &GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).source,
        );
        t.scan(
            name,
            "dynamic",
            &GlslShader::transpile(&node, GlslTranspileMode::Dynamic).source,
        );
    }
    t.check();
}

#[cfg(feature = "gpu")]
#[test]
fn wgsl_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::{TranspileMode, WgslShader};
    let mut t = Tally::new("WGSL");
    for (name, node) in common::corpus::corpus() {
        t.scan(
            name,
            "hardcoded",
            &WgslShader::transpile(&node, TranspileMode::Hardcoded).source,
        );
        t.scan(
            name,
            "dynamic",
            &WgslShader::transpile(&node, TranspileMode::Dynamic).source,
        );
    }
    t.check();
}

#[cfg(feature = "hlsl")]
#[test]
fn hlsl_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::hlsl::{HlslShader, HlslTranspileMode};
    let mut t = Tally::new("HLSL");
    for (name, node) in common::corpus::corpus() {
        t.scan(
            name,
            "hardcoded",
            &HlslShader::transpile(&node, HlslTranspileMode::Hardcoded).source,
        );
        t.scan(
            name,
            "dynamic",
            &HlslShader::transpile(&node, HlslTranspileMode::Dynamic).source,
        );
    }
    t.check();
}

#[cfg(feature = "blinkscript")]
#[test]
fn blinkscript_kernel_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::blinkscript::{BlinkScriptShader, BlinkScriptTranspileMode};
    let mut t = Tally::new("BlinkScript");
    for (name, node) in common::corpus::corpus() {
        for (mode, m) in [
            ("hardcoded", BlinkScriptTranspileMode::Hardcoded),
            ("dynamic", BlinkScriptTranspileMode::Dynamic),
        ] {
            t.scan(
                name,
                mode,
                &BlinkScriptShader::transpile(&node, m).to_kernel(0.0, (-2.0, 2.0)),
            );
        }
    }
    t.check();
}

#[cfg(feature = "msl")]
#[test]
fn msl_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::msl::MslShader;
    use alice_sdf::compiled::TranspileMode;
    let mut msl = Tally::new("MSL");
    let mut wgsl = Tally::new("MSL's WGSL input");
    for (name, node) in common::corpus::corpus() {
        for (mode, m) in [
            ("hardcoded", TranspileMode::Hardcoded),
            ("dynamic", TranspileMode::Dynamic),
        ] {
            // a node MSL refuses is the matrix test's business (docs/node-support.md); here only emitted text counts
            if let Ok(shader) = MslShader::transpile(&node, m) {
                msl.scan(name, mode, &shader.source);
                wgsl.scan(name, mode, &shader.wgsl_source);
            }
        }
    }
    msl.check();
    wgsl.check();
}
