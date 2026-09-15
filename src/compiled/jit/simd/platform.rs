//! Platform-specific SIMD configuration for Cranelift JIT
//!
//! Isolates architecture-dependent settings to enable future
//! x86_64 (AVX2) and aarch64 (NEON) specialization.
//!
//! Author: Moroya Sakamoto

use cranelift_codegen::settings::Builder as FlagBuilder;
#[cfg(target_arch = "x86_64")]
use cranelift_codegen::settings::Configurable;

/// Configure Cranelift flags for the host platform's SIMD capabilities.
///
/// On x86_64, enables the `enable_simd` flag for AVX/SSE instruction generation.
/// On aarch64, NEON is available by default via Cranelift's native ISA.
// Not `const`: the body is arch-gated — on x86_64 it calls the non-const
// `Configurable::set`; clippy on aarch64 (empty body) suggests `const fn`.
#[allow(clippy::needless_pass_by_ref_mut, clippy::missing_const_for_fn)]
pub fn configure_simd_flags(flag_builder: &mut FlagBuilder) {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = flag_builder.set("enable_simd", "true");
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is enabled by default on AArch64 via cranelift_native::builder().
        // Future: add SVE/SVE2 flags when Cranelift supports them.
        let _ = flag_builder;
    }
}
