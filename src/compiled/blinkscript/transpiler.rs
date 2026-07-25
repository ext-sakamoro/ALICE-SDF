//! BlinkScript transpiler — thin wrapper over the HLSL transpiler.
//!
//! The generated `float sdf_eval(float3 p) { ... }` function body is
//! byte-identical to HLSL for the operator set ALICE-SDF exposes, so we
//! reuse [`crate::compiled::hlsl::HlslShader`] verbatim and only add the
//! BlinkScript `ImageComputationKernel` container in [`BlinkScriptShader::to_kernel`].
//!
//! Author: Moroya Sakamoto

use super::super::hlsl::{HlslShader, HlslTranspileMode};
use crate::types::SdfNode;

/// Transpilation mode for BlinkScript (mirrors [`HlslTranspileMode`]).
///
/// - [`BlinkScriptTranspileMode::Hardcoded`] inlines constants directly (fastest).
/// - [`BlinkScriptTranspileMode::Dynamic`] emits parameter references so the
///   caller can update SDF constants without recompiling the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlinkScriptTranspileMode {
    /// Constants baked into the kernel source.
    Hardcoded,
    /// Constants read from a `float[]` parameter array.
    Dynamic,
}

impl From<BlinkScriptTranspileMode> for HlslTranspileMode {
    fn from(m: BlinkScriptTranspileMode) -> Self {
        match m {
            BlinkScriptTranspileMode::Hardcoded => HlslTranspileMode::Hardcoded,
            BlinkScriptTranspileMode::Dynamic => HlslTranspileMode::Dynamic,
        }
    }
}

/// A generated BlinkScript kernel for SDF evaluation.
#[derive(Debug, Clone)]
pub struct BlinkScriptShader {
    /// `float sdf_eval(float3 p) { ... }` — the raw eval function body.
    /// This is HLSL-compatible source and can be embedded verbatim into a
    /// caller-authored Blink kernel, or wrapped via [`Self::to_kernel`].
    pub source: String,
    /// Number of helper functions inherited from the HLSL transpiler.
    pub helper_count: usize,
    /// Parameter layout for [`BlinkScriptTranspileMode::Dynamic`]. Empty for `Hardcoded`.
    pub param_layout: Vec<f32>,
    /// Mode used to produce [`Self::source`].
    pub mode: BlinkScriptTranspileMode,
}

impl BlinkScriptShader {
    /// Transpile an SDF tree to a BlinkScript-ready `sdf_eval` function body.
    ///
    /// Under the hood this delegates to [`HlslShader::transpile`] with the
    /// matching mode; the resulting body is valid BlinkScript verbatim.
    pub fn transpile(node: &SdfNode, mode: BlinkScriptTranspileMode) -> Self {
        let hlsl = HlslShader::transpile(node, mode.into());
        Self {
            source: hlsl.source,
            helper_count: hlsl.helper_count,
            param_layout: hlsl.param_layout,
            mode,
        }
    }

    /// Extract the parameter float array for
    /// [`BlinkScriptTranspileMode::Dynamic`] without re-generating the source.
    pub fn extract_params(node: &SdfNode) -> Vec<f32> {
        HlslShader::extract_params(node)
    }

    /// Wrap the [`Self::source`] body in a complete Nuke Blink
    /// `ImageComputationKernel` that samples the SDF at each pixel of the
    /// output image (image `x/y` → world `x/y`, world `z` fixed by uniform).
    ///
    /// The generated kernel writes a single-channel signed-distance float
    /// image to `dst`; downstream Nuke nodes can normalize or ramp it into
    /// a display-friendly greyscale / colour ramp.
    ///
    /// * `z_uniform` — the world-space `z` slice sampled at every pixel.
    /// * `bounds` — `(min, max)` cubic world-space range that maps to the
    ///   image's [0, width) × [0, height) range.
    pub fn to_kernel(&self, z_uniform: f32, bounds: (f32, f32)) -> String {
        let (bmin, bmax) = bounds;
        let dynamic_note = if self.mode == BlinkScriptTranspileMode::Dynamic {
            "// NOTE: Dynamic mode — populate the `params[]` float[] buffer\n// on the BlinkScript node's Parameters panel before recompiling.\n"
        } else {
            ""
        };
        format!(
            r#"// ALICE-SDF Generated BlinkScript Kernel
// Function body reused verbatim from the HLSL transpiler.
// Drop into the "Kernel" text area of a BlinkScript node in Nuke 12+.
{dynamic_note}
kernel AliceSdfSliceKernel : ImageComputationKernel<eComponentWise>
{{
    Image<eWrite, eAccessPoint> dst;

    param:
        float z_slice;
        float bounds_min;
        float bounds_max;

    void define() {{
        defineParam(z_slice, "z_slice", {z_uniform});
        defineParam(bounds_min, "bounds_min", {bmin});
        defineParam(bounds_max, "bounds_max", {bmax});
    }}

    // ---- SDF eval function (transpiled from ALICE-SDF tree) ----
{source}

    void process(int2 pos) {{
        float span = bounds_max - bounds_min;
        float w = float(dst.bounds.width());
        float h = float(dst.bounds.height());
        float u = (float(pos.x) + 0.5f) / w;
        float v = (float(pos.y) + 0.5f) / h;
        float3 p = float3(
            bounds_min + u * span,
            bounds_min + v * span,
            z_slice
        );
        dst() = sdf_eval(p);
    }}
}}
"#,
            dynamic_note = dynamic_note,
            source = self.source,
            z_uniform = z_uniform,
            bmin = bmin,
            bmax = bmax,
        )
    }

    /// Return only the raw `sdf_eval` function body, for callers that want
    /// to author their own Blink kernel container.
    pub fn get_eval_function(&self) -> &str {
        &self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    fn transpile_sphere_produces_sdf_eval_function() {
        let node = SdfNode::sphere(1.0);
        let shader = BlinkScriptShader::transpile(&node, BlinkScriptTranspileMode::Hardcoded);
        // Sanity: the HLSL-compatible function signature must appear so that
        // downstream kernels can call `sdf_eval(p)` unambiguously.
        assert!(
            shader.source.contains("float sdf_eval(float3 p)"),
            "missing sdf_eval signature: {}",
            shader.source
        );
        assert_eq!(shader.mode, BlinkScriptTranspileMode::Hardcoded);
        // Hardcoded mode has no runtime parameter layout.
        assert!(shader.param_layout.is_empty());
    }

    #[test]
    fn to_kernel_embeds_source_and_bounds() {
        let node = SdfNode::sphere(1.0);
        let shader = BlinkScriptShader::transpile(&node, BlinkScriptTranspileMode::Hardcoded);
        let kernel = shader.to_kernel(0.0, (-2.0, 2.0));
        assert!(kernel.contains("kernel AliceSdfSliceKernel"));
        assert!(kernel.contains("ImageComputationKernel<eComponentWise>"));
        assert!(kernel.contains("defineParam(z_slice"));
        // The generated bounds must be baked into the defineParam calls so
        // the node opens with the requested defaults on first recompile.
        assert!(kernel.contains("-2"));
        assert!(kernel.contains("2"));
        assert!(kernel.contains("sdf_eval(p)"));
    }

    #[test]
    fn dynamic_mode_matches_hlsl_param_extract() {
        // Dynamic mode should surface the same parameter layout as the HLSL
        // path; the two are the same underlying pipeline.
        let node = SdfNode::sphere(1.5).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
        let dyn_shader = BlinkScriptShader::transpile(&node, BlinkScriptTranspileMode::Dynamic);
        let params = BlinkScriptShader::extract_params(&node);
        assert_eq!(dyn_shader.param_layout, params);
    }
}
