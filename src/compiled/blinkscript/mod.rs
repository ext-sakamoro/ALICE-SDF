//! BlinkScript Transpiler — SDF to Foundry Nuke Blink Framework
//!
//! Foundry Nuke's Blink Framework (`BlinkScript` node) is a C++-flavoured GPU
//! kernel language for VFX compositing. Its scalar-and-`float{2,3,4}` syntax
//! is a near-superset of HLSL for the function-body subset we need for SDF
//! evaluation, so this transpiler delegates function-body generation to the
//! existing HLSL pipeline (`HlslLang` / [`super::hlsl::HlslShader`]) and only
//! wraps the result in the BlinkScript-specific `kernel` container.
//!
//! # Output shapes
//!
//! - [`BlinkScriptShader::source`] — plain HLSL-compatible `float sdf_eval(float3 p) { ... }`
//!   function body only, suitable for embedding into a caller-authored kernel.
//! - [`BlinkScriptShader::to_kernel`] — a complete Nuke Blink `ImageComputationKernel`
//!   that samples the SDF over the image plane at a fixed `z` and outputs
//!   a signed-distance greyscale image (drop straight into a BlinkScript node).
//!
//! # Usage inside Nuke
//!
//! ```text
//! 1. Add a BlinkScript node in Nuke's node graph
//! 2. Paste the string returned by `BlinkScriptShader::to_kernel()` into
//!    the "Kernel" text area of the BlinkScript node
//! 3. Click "Recompile" — you'll get a distance-field greyscale image
//! ```
//!
//! # Why not a separate `ShaderLang` impl?
//!
//! BlinkScript and HLSL share the identical scalar / `float3` / ternary /
//! `fmod` surface used by [`crate::compiled::transpiler_common::ShaderLang`].
//! Duplicating a full 40-method trait impl would drift silently over time;
//! delegation keeps a single source of truth in `HlslLang` (see
//! [`super::hlsl::HlslShader`]) and this module contributes only the
//! BlinkScript kernel container.
//!
//! Author: Moroya Sakamoto

mod transpiler;

pub use transpiler::{BlinkScriptShader, BlinkScriptTranspileMode};
