//! WGSL → MSL translation through naga.
//!
//! The WGSL emit is the source of law (see the module docs for why there is no
//! hand-written `MslLang`); this file only does the translation and the binding
//! bookkeeping Metal needs.
//!
//! Author: Moroya Sakamoto

use crate::compiled::{TranspileMode, WgslShader};
use crate::SdfNode;
use naga::back::msl;

/// Metal Shading Language version requested from naga.
///
/// 2.0 is the floor for `metal::MTLLanguageVersion::V2_0` era devices and is
/// what Apple Silicon ships; naga's default of 1.0 rejects some constructs the
/// SDF shaders use.
const MSL_LANG_VERSION: (u8, u8) = (2, 0);

/// Put first in every emitted MSL source: the Metal compiler contracts `a * b + c`
/// into one fused multiply-add by default, and still does so inside one
/// expression with fast math off (`mathMode = .safe`), while the CPU paths round
/// twice (ALICE-Law G3). The pragma holds whatever compile options the caller
/// passes; measured on Apple silicon with MSL 2.0 and the default language
/// version (`tests/test_msl_metal_oracle.rs`, `metal_does_not_fuse_a_multiply_add`).
const MSL_FP_CONTRACT_OFF: &str = "#pragma METAL fp contract(off)\n";

/// Failure while turning a WGSL shader into MSL.
///
/// Every variant carries the full diagnostic text: a silent fallback would ship
/// invalid MSL, and `naga`'s default `fake_missing_bindings: true` does exactly
/// that, so this module turns it off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MslError {
    /// The WGSL produced by [`WgslShader`] did not parse (a transpiler bug).
    WgslParse(String),
    /// The WGSL parsed but failed naga validation (a transpiler bug).
    Validation(String),
    /// naga could not write MSL for the validated module.
    MslWrite(String),
    /// No compute entry point was found in the WGSL.
    NoEntryPoint,
    /// The WGSL carries another semantics identifier in its leading `//` block (see
    /// [`crate::provenance`]): it was transpiled under another arithmetic
    SemanticsMismatch {
        /// The identifier in the text
        stored: crate::provenance::Provenance,
        /// This crate's identifier
        expected: crate::provenance::Provenance,
    },
    /// The WGSL carries no semantics identifier in its leading `//` block, and the reader is
    /// [`MslShader::from_wgsl_pinned`]
    Unpinned,
    /// The identifier in the leading `//` block is malformed, or there are two different ones
    InvalidFormat(String),
}

impl core::fmt::Display for MslError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WgslParse(e) => write!(f, "WGSL parse failed before MSL emit: {e}"),
            Self::Validation(e) => write!(f, "WGSL validation failed before MSL emit: {e}"),
            Self::MslWrite(e) => write!(f, "naga could not write MSL: {e}"),
            Self::NoEntryPoint => write!(f, "no entry point in the WGSL module"),
            Self::SemanticsMismatch { stored, expected } => write!(
                f,
                "WGSL: semantics identifier mismatch: stored {stored}, expected {expected}"
            ),
            Self::Unpinned => write!(
                f,
                "WGSL: no semantics identifier in the leading // block (use MslShader::from_wgsl to read it unpinned)"
            ),
            Self::InvalidFormat(e) => write!(f, "WGSL: {e}"),
        }
    }
}

impl std::error::Error for MslError {}

impl From<crate::provenance::ShaderRefusal> for MslError {
    fn from(r: crate::provenance::ShaderRefusal) -> Self {
        use crate::provenance::ShaderRefusal;
        match r {
            ShaderRefusal::Unpinned => Self::Unpinned,
            ShaderRefusal::Mismatch { stored, expected } => {
                Self::SemanticsMismatch { stored, expected }
            }
            ShaderRefusal::Invalid(e) => Self::InvalidFormat(e),
        }
    }
}

/// The Metal compile options an emitted kernel needs for its bits to match the
/// CPU laws.
///
/// The source fixes contraction itself (`#pragma METAL fp contract(off)`,
/// ALICE-Law G3); the sign of a zero it cannot fix: with fast math on, Metal drops
/// it (`-(a * b + c)` with a result of zero comes back `+0.0`), and no construct of
/// the source keeps it in general (measured on Apple silicon:
/// `tests/test_msl_metal_oracle.rs`, `metal_signed_zero_needs_fast_math_off`). The
/// caller sets these on `MTLCompileOptions` before `newLibraryWithSource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct MslCompileRequirements {
    /// `MTLCompileOptions.fastMathEnabled` (or `mathMode = .safe`, macOS 15+).
    pub fast_math_enabled: bool,
    /// `MTLCompileOptions.languageVersion` the source is written for (major, minor).
    pub language_version: (u8, u8),
}

/// An SDF evaluated as a Metal compute kernel.
#[derive(Debug, Clone)]
pub struct MslShader {
    /// Complete MSL source, ready for `MTLDevice::newLibraryWithSource`.
    pub source: String,
    /// Kernel function name to look up in the compiled library.
    ///
    /// This is naga's **renamed** function, not the WGSL entry point: `main` is
    /// reserved in MSL. Look up this name, never `wgsl_entry_point`.
    pub entry_point: String,
    /// The WGSL entry point this kernel came from (`main`), kept for
    /// diagnostics and for correlating with the WGSL oracle.
    pub wgsl_entry_point: String,
    /// `@workgroup_size` from the WGSL, i.e. the Metal threads-per-threadgroup.
    pub workgroup_size: [u32; 3],
    /// Buffer slot that must hold the `u32` lengths of the runtime-sized
    /// arrays, in declaration order.
    ///
    /// Metal has no runtime-sized arrays, so naga passes the lengths in through
    /// this extra buffer. The slot is one past the highest `@binding` in the
    /// shader, so it never collides with a real binding.
    pub sizes_buffer_slot: u8,
    /// The WGSL this MSL was derived from, kept so a parity failure can be
    /// attributed to the WGSL law or to the translation.
    pub wgsl_source: String,
}

impl MslShader {
    /// The compile options every emitted kernel needs (see [`MslCompileRequirements`]);
    /// the single source of them, also written into the head of [`Self::source`].
    pub const REQUIRED_COMPILE_OPTIONS: MslCompileRequirements = MslCompileRequirements {
        fast_math_enabled: false,
        language_version: MSL_LANG_VERSION,
    };

    /// Transpile an SDF node to a Metal compute kernel.
    ///
    /// `WgslShader::transpile` alone emits only the SDF functions (no entry
    /// point); the Metal kernel comes from `to_compute_shader`, which adds the
    /// bindings and `@compute`. Pass
    /// `WgslShader::to_compute_shader_with_normals` output to
    /// [`MslShader::from_wgsl`] for the normals variant.
    ///
    /// # Errors
    ///
    /// Returns [`MslError`] if the WGSL emit does not parse or validate (which
    /// is a bug in the WGSL transpiler, not in the caller's input), or if naga
    /// cannot write MSL for it.
    pub fn transpile(node: &SdfNode, mode: TranspileMode) -> Result<Self, MslError> {
        let wgsl = WgslShader::transpile(node, mode);
        Self::from_wgsl_pinned(&wgsl.to_compute_shader())
    }

    /// Translate an arbitrary WGSL compute shader to MSL, with what is known about the arithmetic
    /// it was transpiled under
    ///
    /// The identifier is read from the leading `//` block of `wgsl` (see [`crate::provenance`]):
    /// this crate's text is [`StoredSemantics::Current`](crate::provenance::StoredSemantics::Current),
    /// text without the key anywhere (another program's WGSL) is `Unpinned`, text under another
    /// identifier, and the key outside the leading block, are refused before naga sees it. The MSL
    /// carries the identifier the WGSL carried (none for `Unpinned`).
    /// [`MslShader::from_wgsl_pinned`] also refuses `Unpinned`.
    ///
    /// Used by the oracle tests to run the same translation over shaders that
    /// are not produced by [`MslShader::transpile`] (NPR, noise, marching
    /// cubes), so those paths get the same coverage.
    ///
    /// # Errors
    ///
    /// [`MslError::SemanticsMismatch`] for another identifier, [`MslError::InvalidFormat`] for a
    /// malformed one, otherwise see [`MslShader::transpile`].
    pub fn from_wgsl(wgsl: &str) -> Result<crate::provenance::Checked<Self>, MslError> {
        Self::judged(wgsl, true)
    }

    /// [`MslShader::from_wgsl`] that refuses WGSL without this crate's identifier
    ///
    /// # Errors
    ///
    /// [`MslError::Unpinned`] for text without an identifier, otherwise as
    /// [`MslShader::from_wgsl`].
    pub fn from_wgsl_pinned(wgsl: &str) -> Result<Self, MslError> {
        Self::judged(wgsl, false).map(|c| c.into_parts().0)
    }

    fn judged(
        wgsl: &str,
        allow_unpinned: bool,
    ) -> Result<crate::provenance::Checked<Self>, MslError> {
        let semantics = crate::provenance::shader_text_verdict(wgsl, allow_unpinned)?;
        Self::translate(wgsl, semantics)
            .map(|shader| crate::provenance::Checked::new(shader, semantics))
    }

    fn translate(
        wgsl: &str,
        semantics: crate::provenance::StoredSemantics,
    ) -> Result<Self, MslError> {
        let module = naga::front::wgsl::parse_str(wgsl)
            .map_err(|e| MslError::WgslParse(e.emit_to_string(wgsl)))?;

        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| MslError::Validation(format!("{e:?}")))?;

        let entry = module
            .entry_points
            .first()
            .ok_or(MslError::NoEntryPoint)?
            .clone();

        // WGSL `@group(0) @binding(N)` -> Metal `buffer(N)`: the identity map
        // keeps the host-side binding code readable, and the sizes buffer goes
        // one past the highest binding so it cannot collide.
        let mut resources = msl::BindingMap::new();
        let mut max_binding: u8 = 0;
        for (_, var) in module.global_variables.iter() {
            let Some(ref br) = var.binding else { continue };
            let slot = u8::try_from(br.binding).map_err(|_| {
                MslError::MslWrite(format!(
                    "binding {} exceeds the Metal buffer slot range",
                    br.binding
                ))
            })?;
            max_binding = max_binding.max(slot);
            let mutable = matches!(
                var.space,
                naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE)
            );
            resources.insert(
                br.clone(),
                msl::BindTarget {
                    buffer: Some(slot),
                    texture: None,
                    sampler: None,
                    mutable,
                },
            );
        }
        let sizes_buffer_slot = max_binding
            .checked_add(1)
            .ok_or_else(|| MslError::MslWrite("no free Metal buffer slot for sizes".into()))?;

        let mut per_entry_point_map = msl::EntryPointResourceMap::new();
        for ep in &module.entry_points {
            per_entry_point_map.insert(
                ep.name.clone(),
                msl::EntryPointResources {
                    resources: resources.clone(),
                    push_constant_buffer: None,
                    sizes_buffer: Some(sizes_buffer_slot),
                },
            );
        }

        let options = msl::Options {
            lang_version: MSL_LANG_VERSION,
            per_entry_point_map,
            // Never generate invalid MSL silently: an unmapped binding is a
            // bug in the map above, and `true` (naga's default) would emit a
            // shader that compiles and reads the wrong buffer.
            fake_missing_bindings: false,
            ..msl::Options::default()
        };

        let (naga_source, translation) =
            msl::write_string(&module, &info, &options, &msl::PipelineOptions::default())
                .map_err(|e| MslError::MslWrite(format!("{e:?}")))?;
        let required = Self::REQUIRED_COMPILE_OPTIONS;
        let body = format!(
            "{MSL_FP_CONTRACT_OFF}// compile with fastMathEnabled = {}, languageVersion = {}.{} (MslShader::REQUIRED_COMPILE_OPTIONS)\n{naga_source}",
            required.fast_math_enabled, required.language_version.0, required.language_version.1,
        );
        // the MSL carries the identifier of the WGSL it came from
        let source = match semantics {
            crate::provenance::StoredSemantics::Current => crate::provenance::tagged_shader(&body),
            crate::provenance::StoredSemantics::VerifiedPrevious(p) => {
                crate::provenance::tagged_shader_with(&p, &body)
            }
            crate::provenance::StoredSemantics::Unpinned => body,
        };

        // The MSL function name is **not** the WGSL one: `main` is reserved in
        // MSL, so naga renames entry points and reports the mapping here. Using
        // the WGSL name makes `getFunction` fail at runtime with
        // "Function 'main' does not exist" (caught by the Metal oracle).
        let entry_point = translation
            .entry_point_names
            .first()
            .ok_or(MslError::NoEntryPoint)?
            .clone()
            .map_err(|e| MslError::MslWrite(format!("entry point not translated: {e:?}")))?;

        Ok(Self {
            source,
            entry_point,
            wgsl_entry_point: entry.name,
            workgroup_size: entry.workgroup_size,
            sizes_buffer_slot,
            wgsl_source: wgsl.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere() -> SdfNode {
        SdfNode::sphere(1.0)
    }

    #[test]
    fn emits_metal_kernel_with_entry_point() {
        let shader = MslShader::transpile(&sphere(), TranspileMode::Hardcoded)
            .expect("sphere must transpile to MSL");
        assert!(
            shader.source.contains("#include <metal_stdlib>"),
            "MSL must include the Metal standard library:\n{}",
            shader.source
        );
        assert!(
            shader.source.contains("kernel void"),
            "MSL must declare a compute kernel:\n{}",
            shader.source
        );
        assert_eq!(shader.wgsl_entry_point, "main");
        // naga renames it because `main` is reserved in MSL; the renamed symbol
        // is what MTLLibrary::getFunction needs.
        assert!(
            shader.source.contains(&shader.entry_point),
            "renamed entry point {} must appear in the MSL:\n{}",
            shader.entry_point,
            shader.source
        );
    }

    #[test]
    fn workgroup_size_survives_translation() {
        let shader = MslShader::transpile(&sphere(), TranspileMode::Hardcoded).unwrap();
        let wgsl = WgslShader::transpile(&sphere(), TranspileMode::Hardcoded).to_compute_shader();
        let decl = format!("@workgroup_size({}", shader.workgroup_size[0]);
        assert!(
            wgsl.contains(&decl),
            "workgroup_size {:?} must come from the WGSL declaration",
            shader.workgroup_size
        );
        assert!(shader.workgroup_size.iter().all(|&n| n > 0));
    }

    #[test]
    fn bindings_map_to_matching_buffer_slots_and_sizes_slot_is_free() {
        let shader = MslShader::transpile(&sphere(), TranspileMode::Hardcoded).unwrap();
        // The three storage / uniform bindings of the eval kernel.
        for slot in 0..=2u8 {
            assert!(
                shader.source.contains(&format!("buffer({slot})")),
                "binding {slot} must map to buffer({slot}):\n{}",
                shader.source
            );
        }
        assert!(
            shader.sizes_buffer_slot > 2,
            "sizes buffer slot {} must not collide with a real binding",
            shader.sizes_buffer_slot
        );
        assert!(shader
            .source
            .contains(&format!("buffer({})", shader.sizes_buffer_slot)));
    }

    #[test]
    fn dynamic_mode_adds_the_params_binding() {
        let shader = MslShader::transpile(&sphere(), TranspileMode::Dynamic).unwrap();
        assert!(
            shader.source.contains("buffer(3)"),
            "Dynamic mode must bind sdf_params at buffer(3):\n{}",
            shader.source
        );
        assert!(shader.sizes_buffer_slot > 3);
    }

    #[test]
    fn invalid_wgsl_is_reported_not_swallowed() {
        let err = MslShader::from_wgsl("this is not wgsl").expect_err("must not succeed");
        assert!(matches!(err, MslError::WgslParse(_)), "got {err:?}");
    }
}
