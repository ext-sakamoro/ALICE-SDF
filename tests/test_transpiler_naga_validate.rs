//! Every corpus node's transpiled shader must parse **and validate** with
//! naga: WGSL through `naga::front::wgsl`, GLSL through `naga::front::glsl`,
//! then `naga::valid::Validator` (type checking, expression validity). HLSL
//! has no naga frontend; it is emitted from the same `transpiler_common`
//! walker as WGSL / GLSL and is covered by the GPU parity tests through the
//! WGSL twin. Until 1.10.3 only the NPR / noise / round-tie shaders were
//! parsed, so a mirrored taper survived in all three transpilers until the
//! GPU parity test caught it (external review, 2026-09-15).
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

mod common;

use alice_sdf::compiled::{TranspileMode, WgslShader};
use common::corpus::corpus;
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate(module: &naga::Module) -> Result<(), String> {
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(module)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

#[test]
fn every_corpus_node_transpiles_to_valid_wgsl() {
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for (name, node) in corpus() {
        let shader = WgslShader::transpile(&node, TranspileMode::Hardcoded);
        let module = match naga::front::wgsl::parse_str(&shader.source) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!(
                    "{name}: WGSL parse failed:\n{}",
                    e.emit_to_string(&shader.source)
                ));
                continue;
            }
        };
        if let Err(e) = validate(&module) {
            failures.push(format!("{name}: WGSL validation failed: {e}"));
            continue;
        }
        checked += 1;
    }
    assert!(checked >= 100, "only {checked} nodes validated");
    assert!(
        failures.is_empty(),
        "{} invalid WGSL shaders:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The NaN a shader returns for a node it has no law for (a `RepeatCells` with
/// the not-yet-supported `neighbours`) is valid WGSL and GLSL, not only the
/// supported corpus.
#[test]
fn the_no_law_nan_is_valid_wgsl_and_glsl() {
    use alice_sdf::prelude::{SdfNode, Vec3};
    let node = SdfNode::RepeatCells {
        child: std::sync::Arc::new(SdfNode::sphere(0.3)),
        spacing: Vec3::new(2.0, 0.0, 0.0),
        count: None,
        mirror: true,
        neighbours: true,
        variation: None,
    };
    let wgsl = WgslShader::transpile(&node, TranspileMode::Hardcoded).source;
    assert!(wgsl.contains("0x7fc00000u"));
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&wgsl)));
    validate(&module).expect("WGSL");
    #[cfg(feature = "glsl")]
    {
        use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
        let glsl = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).source;
        assert!(glsl.contains("0x7fc00000u"));
        let source = format!(
            "{glsl}\nout vec4 alice_frag;\nvoid main() {{ alice_frag = vec4(sdf_eval(vec3(0.1, 0.2, 0.3))); }}\n"
        );
        let options = naga::front::glsl::Options {
            stage: naga::ShaderStage::Fragment,
            defines: naga::FastHashMap::default(),
        };
        let module = naga::front::glsl::Frontend::default()
            .parse(&options, &source)
            .unwrap_or_else(|e| panic!("{e:?}"));
        validate(&module).expect("GLSL");
    }
}

#[cfg(feature = "glsl")]
#[test]
fn every_corpus_node_transpiles_to_valid_glsl() {
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for (name, node) in corpus() {
        let shader = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
        let mut frontend = naga::front::glsl::Frontend::default();
        let options = naga::front::glsl::Options {
            stage: naga::ShaderStage::Fragment,
            defines: naga::FastHashMap::default(),
        };
        // The transpiler emits an `sdf_eval` function library; wrap it in a
        // minimal fragment entry point so naga has something to validate.
        let source = format!(
            "{}\nout vec4 alice_frag;\nvoid main() {{ alice_frag = vec4(sdf_eval(vec3(0.1, 0.2, 0.3))); }}\n",
            shader.source
        );
        let module = match frontend.parse(&options, &source) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{name}: GLSL parse failed: {e:?}"));
                continue;
            }
        };
        if let Err(e) = validate(&module) {
            failures.push(format!("{name}: GLSL validation failed: {e}"));
            continue;
        }
        checked += 1;
    }
    assert!(checked >= 100, "only {checked} nodes validated");
    assert!(
        failures.is_empty(),
        "{} invalid GLSL shaders:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
