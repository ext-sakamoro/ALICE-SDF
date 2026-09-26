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
}

impl core::fmt::Display for MslError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WgslParse(e) => write!(f, "WGSL parse failed before MSL emit: {e}"),
            Self::Validation(e) => write!(f, "WGSL validation failed before MSL emit: {e}"),
            Self::MslWrite(e) => write!(f, "naga could not write MSL: {e}"),
            Self::NoEntryPoint => write!(f, "no entry point in the WGSL module"),
        }
    }
}

impl std::error::Error for MslError {}

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
        Self::from_wgsl(&wgsl.to_compute_shader())
    }

    /// Translate an arbitrary WGSL compute shader to MSL.
    ///
    /// Used by the oracle tests to run the same translation over shaders that
    /// are not produced by [`MslShader::transpile`] (NPR, noise, marching
    /// cubes), so those paths get the same coverage.
    ///
    /// # Errors
    ///
    /// See [`MslShader::transpile`].
    pub fn from_wgsl(wgsl: &str) -> Result<Self, MslError> {
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

        let (source, translation) =
            msl::write_string(&module, &info, &options, &msl::PipelineOptions::default())
                .map_err(|e| MslError::MslWrite(format!("{e:?}")))?;

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
