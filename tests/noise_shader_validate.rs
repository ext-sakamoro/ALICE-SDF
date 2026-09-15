//! The transpiled `Noise` (Perlin) and `SurfaceRoughness` (value-noise fbm)
//! helpers must at least parse: WGSL through `naga::front::wgsl`, GLSL through
//! `naga::front::glsl`. (HLSL has no naga frontend; its helper is the same
//! text with `asuint` / `(int)` casts.)

#![cfg(all(feature = "glsl", feature = "gpu"))]

use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
use alice_sdf::compiled::{TranspileMode, WgslShader};
use alice_sdf::prelude::*;

fn nodes() -> Vec<(&'static str, SdfNode)> {
    vec![
        ("noise", SdfNode::sphere(1.0).noise(0.1, 2.0, 7)),
        (
            "surface_roughness",
            SdfNode::box3d(1.0, 1.0, 1.0).surface_roughness(3.0, 0.05, 3),
        ),
        (
            "both",
            SdfNode::torus(0.9, 0.25)
                .noise(0.05, 4.0, 1)
                .surface_roughness(2.0, 0.02, 2),
        ),
    ]
}

#[test]
fn wgsl_noise_helpers_parse() {
    for (name, node) in nodes() {
        let shader = WgslShader::transpile(&node, TranspileMode::Hardcoded);
        let result = naga::front::wgsl::parse_str(&shader.source);
        assert!(
            result.is_ok(),
            "{name}: WGSL parse failed:\n{}\n---source---\n{}",
            result
                .err()
                .map(|e| e.emit_to_string(&shader.source))
                .unwrap_or_default(),
            shader.source
        );
        assert!(shader.source.contains("perlin_noise_3d") || name == "surface_roughness");
    }
}

#[test]
fn glsl_noise_helpers_parse() {
    for (name, node) in nodes() {
        let shader = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
        let mut frontend = naga::front::glsl::Frontend::default();
        let options = naga::front::glsl::Options {
            stage: naga::ShaderStage::Fragment,
            defines: Default::default(),
        };
        // The transpiler emits an `sdf_eval` function library; wrap it in a
        // minimal fragment entry point so naga has something to validate.
        let source = format!(
            "{}\nout vec4 alice_frag;\nvoid main() {{ alice_frag = vec4(sdf_eval(vec3(0.1, 0.2, 0.3))); }}\n",
            shader.source
        );
        let result = frontend.parse(&options, &source);
        assert!(
            result.is_ok(),
            "{name}: GLSL parse failed: {:?}\n---source---\n{source}",
            result.err()
        );
    }
}
