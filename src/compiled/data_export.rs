//! The separate-data shader export: shader text that reads its per-node data arrays (a loft's
//! rows, a stack's sections, a polygon outline, a heightmap) from one R32F texture instead of
//! holding them as constants in the text.
//!
//! The single-file export (`transpile` in [`TranspileMode::Hardcoded`]) stays the default and is
//! unchanged: it is right for small shapes. A large shape's constants make the text large and
//! slow to compile (a driver may copy a whole constant array at each run-time index); this export
//! keeps the text small and hands the data to the engine as a texture, which every target can
//! bind (fragment and compute, Unreal / Unity / Godot / three.js / wgpu).
//!
//! Layout: every data array of the tree is laid end to end in one flat `f32` sequence, the same
//! order and the same values as the single-file text's constants; the texture is
//! [`DATA_TEXTURE_WIDTH`] texels wide, row-major, the last row padded with `0.0`. The shader reads
//! value `i` with an exact texel fetch (no filtering) at `(i % width, i / width)`, so each value
//! keeps its bits.
//!
//! [`TranspileMode::Hardcoded`]: crate::compiled::TranspileMode

/// Texels per row of the data texture (well inside every target's minimum texture size)
pub const DATA_TEXTURE_WIDTH: u32 = 4096;

/// The name the exported shader binds the data texture under
pub const DATA_TEXTURE_NAME: &str = "alice_data_tex";

/// The data of a separate-data export: an R32F texture, row-major
#[derive(Debug, Clone, PartialEq)]
pub struct DataTexture {
    /// texels per row ([`DATA_TEXTURE_WIDTH`])
    pub width: u32,
    /// rows
    pub height: u32,
    /// `width * height` values, the data followed by `0.0` padding
    pub texels: Vec<f32>,
    /// how many leading texels are data (the rest is padding)
    pub len: usize,
}

impl DataTexture {
    /// The texture holding `values` (at least one row, so an engine never binds a 0-height texture)
    #[must_use]
    pub fn from_values(values: &[f32]) -> Self {
        let width = DATA_TEXTURE_WIDTH as usize;
        let rows = values.len().div_ceil(width).max(1);
        let mut texels = values.to_vec();
        texels.resize(rows * width, 0.0);
        Self {
            width: DATA_TEXTURE_WIDTH,
            height: u32::try_from(rows).unwrap_or(u32::MAX),
            texels,
            len: values.len(),
        }
    }

    /// value `i` of the data, as the shader reads it (`None` past the data)
    #[must_use]
    pub fn value(&self, i: usize) -> Option<f32> {
        (i < self.len).then(|| self.texels[i])
    }
}

/// A separate-data export: the shader text and, when the tree has data arrays, its texture
#[derive(Debug, Clone, PartialEq)]
pub struct DataExport {
    /// the shader text (declares the texture [`DATA_TEXTURE_NAME`] when `data` is `Some`)
    pub source: String,
    /// the data texture; `None` for a tree without data arrays (the text is then the single-file
    /// text)
    pub data: Option<DataTexture>,
}

#[cfg(feature = "gpu")]
impl crate::compiled::WgslShader {
    /// The separate-data WGSL export of `node` (see the module documentation)
    ///
    /// # Errors
    ///
    /// As [`crate::compiled::WgslShader::try_transpile`]
    pub fn export_with_data_texture(
        node: &crate::SdfNode,
    ) -> Result<DataExport, crate::compiled::TranspileError> {
        let _ = node;
        todo!("STUB: A2 separate-data WGSL export not implemented")
    }
}

#[cfg(feature = "glsl")]
impl crate::compiled::glsl::GlslShader {
    /// The separate-data GLSL export of `node` (see the module documentation)
    ///
    /// # Errors
    ///
    /// As the single-file GLSL transpile
    pub fn export_with_data_texture(
        node: &crate::SdfNode,
    ) -> Result<DataExport, crate::compiled::TranspileError> {
        let _ = node;
        todo!("STUB: A2 separate-data GLSL export not implemented")
    }
}

#[cfg(feature = "hlsl")]
impl crate::compiled::HlslShader {
    /// The separate-data HLSL export of `node` (see the module documentation)
    ///
    /// # Errors
    ///
    /// As the single-file HLSL transpile
    pub fn export_with_data_texture(
        node: &crate::SdfNode,
    ) -> Result<DataExport, crate::compiled::TranspileError> {
        let _ = node;
        todo!("STUB: A2 separate-data HLSL export not implemented")
    }
}

#[cfg(feature = "gpu")]
impl crate::compiled::GpuEvaluator {
    /// An evaluator for a separate-data WGSL export, binding its data texture
    ///
    /// # Errors
    ///
    /// As [`crate::compiled::GpuEvaluator::from_shader`]
    pub fn from_data_export(export: &DataExport) -> Result<Self, crate::compiled::GpuError> {
        let _ = export;
        todo!("STUB: A2 evaluator for the separate-data export not implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_fill_rows_in_order_and_the_last_row_is_padded() {
        let values: Vec<f32> = (0..5000).map(|i| i as f32 * 0.5).collect();
        let t = DataTexture::from_values(&values);
        assert_eq!((t.width, t.height, t.len), (4096, 2, 5000));
        assert_eq!(t.texels.len(), 8192);
        assert_eq!(
            t.value(4097).map(f32::to_bits),
            Some((4097.0f32 * 0.5).to_bits())
        );
        assert_eq!(t.value(5000), None);
        assert!(t.texels[5000..].iter().all(|v| v.to_bits() == 0));
    }

    #[test]
    fn no_values_still_make_one_row() {
        let t = DataTexture::from_values(&[]);
        assert_eq!((t.height, t.len, t.texels.len()), (1, 0, 4096));
    }
}
