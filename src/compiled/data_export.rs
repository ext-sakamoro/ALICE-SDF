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
//! Limits: the texture is at most [`DATA_TEXTURE_WIDTH`] x [`MAX_DATA_TEXTURE_ROWS`] (2048 x 2048,
//! 4,194,304 values), inside the smallest maximum texture size WebGL 2 and OpenGL ES 3.0 guarantee
//! (2048); a tree with more data is refused ([`DataExportError::TooLarge`]), never truncated. The
//! targets need an R32F (single-channel 32-bit float) texture and an exact fetch: WebGL 2 / OpenGL
//! ES 3.0 (`texelFetch`, R32F is not filterable there and is not filtered), GLSL 1.30+ / Vulkan,
//! HLSL Shader Model 4+ (`Texture2D<float>::Load`), WGSL (`textureLoad` on an unfilterable float
//! texture).
//!
//! [`TranspileMode::Hardcoded`]: crate::compiled::TranspileMode

/// Texels per row of the data texture (the smallest maximum texture size WebGL 2 / OpenGL ES 3.0
/// guarantee)
pub const DATA_TEXTURE_WIDTH: u32 = 2048;

/// The most rows a data texture may have (the same guaranteed minimum): at most
/// `DATA_TEXTURE_WIDTH * MAX_DATA_TEXTURE_ROWS` values
pub const MAX_DATA_TEXTURE_ROWS: u32 = 2048;

/// Why a separate-data export was refused
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataExportError {
    /// the tree cannot be transpiled (as the single-file export refuses it)
    Transpile(crate::compiled::TranspileError),
    /// the tree's data does not fit one texture of the guaranteed size
    TooLarge {
        /// the values the tree holds
        values: usize,
        /// the most a data texture holds
        max_values: usize,
    },
}

impl core::fmt::Display for DataExportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Transpile(e) => write!(f, "{e}"),
            Self::TooLarge { values, max_values } => write!(
                f,
                "the tree holds {values} data values, more than one data texture holds ({max_values})"
            ),
        }
    }
}

impl std::error::Error for DataExportError {}

impl From<crate::compiled::TranspileError> for DataExportError {
    fn from(e: crate::compiled::TranspileError) -> Self {
        Self::Transpile(e)
    }
}

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
    ///
    /// # Errors
    ///
    /// [`DataExportError::TooLarge`] past `DATA_TEXTURE_WIDTH * MAX_DATA_TEXTURE_ROWS` values
    pub fn try_from_values(values: &[f32]) -> Result<Self, DataExportError> {
        let width = DATA_TEXTURE_WIDTH as usize;
        let max_values = width * MAX_DATA_TEXTURE_ROWS as usize;
        if values.len() > max_values {
            return Err(DataExportError::TooLarge {
                values: values.len(),
                max_values,
            });
        }
        let rows = values.len().div_ceil(width).max(1);
        let mut texels = values.to_vec();
        texels.resize(rows * width, 0.0);
        Ok(Self {
            width: DATA_TEXTURE_WIDTH,
            height: u32::try_from(rows).expect("rows <= MAX_DATA_TEXTURE_ROWS"),
            texels,
            len: values.len(),
        })
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

/// The name the node data buffer has in the emitted text; every read of it is
/// `alice_node_data[<index>]`
const BUFFER_NAME: &str = "alice_node_data";

/// `source` with every read `alice_node_data[<index>]` replaced by `read(<index>)` (the index
/// expression may itself hold brackets)
///
/// # Panics
///
/// On an unbalanced read (the emitted text is this crate's own: never)
fn reads_through(source: &str, read: &dyn Fn(&str) -> String) -> String {
    let open = format!("{BUFFER_NAME}[");
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(at) = rest.find(&open) {
        out.push_str(&rest[..at]);
        let inner_start = at + open.len();
        let mut depth = 1usize;
        let mut end = None;
        for (k, c) in rest[inner_start..].char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(inner_start + k);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.expect("a node data read has its closing bracket");
        out.push_str(&read(&rest[inner_start..end]));
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    assert!(
        !out.contains(BUFFER_NAME),
        "every node data read went through the texture"
    );
    out
}

/// `decl` placed after the text's leading `//` block (the block that carries the crate's
/// identifier, which a reader takes from the first lines of the text)
fn after_leading_comments(source: &str, decl: &str) -> String {
    let mut head = 0;
    for line in source.split_inclusive('\n') {
        if !line.trim_start().starts_with("//") {
            break;
        }
        head += line.len();
    }
    format!("{}{decl}{}", &source[..head], &source[head..])
}

#[cfg(feature = "gpu")]
impl crate::compiled::WgslShader {
    /// The separate-data WGSL export of `node` (see the module documentation): the text declares
    /// `@group(0) @binding(4) var alice_data_tex: texture_2d<f32>` (an R32F texture, unfilterable
    /// float) and reads it with `textureLoad`
    ///
    /// # Errors
    ///
    /// [`DataExportError::Transpile`] as [`crate::compiled::WgslShader::try_transpile`];
    /// [`DataExportError::TooLarge`] when the data does not fit one texture
    pub fn export_with_data_texture(node: &crate::SdfNode) -> Result<DataExport, DataExportError> {
        let with_buffer = Self::try_transpile_plain_with_node_data(node)?;
        if with_buffer.node_data.is_empty() {
            return Ok(DataExport {
                source: Self::try_transpile(node, crate::compiled::TranspileMode::Hardcoded)?
                    .source,
                data: None,
            });
        }
        let data = DataTexture::try_from_values(&with_buffer.node_data)?;
        let body = reads_through(&with_buffer.source, &|i| format!("alice_data(i32({i}))"));
        let decl = format!(
            "\n@group(0) @binding(4) var {DATA_TEXTURE_NAME}: texture_2d<f32>;\n\
             fn alice_data(i: i32) -> f32 {{\n    \
             return textureLoad({DATA_TEXTURE_NAME}, vec2<i32>(i % {w}, i / {w}), 0).x;\n}}\n",
            w = DATA_TEXTURE_WIDTH
        );
        Ok(DataExport {
            source: after_leading_comments(&body, &decl),
            data: Some(data),
        })
    }
}

#[cfg(feature = "glsl")]
impl crate::compiled::glsl::GlslShader {
    /// The separate-data GLSL export of `node` (see the module documentation)
    ///
    /// # Errors
    ///
    /// [`DataExportError::Transpile`] as the single-file GLSL transpile;
    /// [`DataExportError::TooLarge`] when the data does not fit one texture
    pub fn export_with_data_texture(node: &crate::SdfNode) -> Result<DataExport, DataExportError> {
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
    /// [`DataExportError::Transpile`] as the single-file HLSL transpile;
    /// [`DataExportError::TooLarge`] when the data does not fit one texture
    pub fn export_with_data_texture(node: &crate::SdfNode) -> Result<DataExport, DataExportError> {
        let _ = node;
        todo!("STUB: A2 separate-data HLSL export not implemented")
    }
}

#[cfg(all(feature = "gpu", feature = "glsl"))]
impl crate::compiled::GpuEvaluator {
    /// An evaluator for a separate-data GLSL export (its Vulkan compute form), binding its data
    /// texture
    ///
    /// # Errors
    ///
    /// As [`crate::compiled::GpuEvaluator::from_glsl_compute`]
    pub fn from_glsl_data_export(export: &DataExport) -> Result<Self, crate::compiled::GpuError> {
        let _ = export;
        todo!("STUB: A2 GLSL evaluator for the separate-data export not implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_fill_rows_in_order_and_the_last_row_is_padded() {
        let values: Vec<f32> = (0..5000).map(|i| i as f32 * 0.5).collect();
        let t = DataTexture::try_from_values(&values).unwrap();
        assert_eq!((t.width, t.height, t.len), (2048, 3, 5000));
        assert_eq!(t.texels.len(), 6144);
        assert_eq!(
            t.value(4097).map(f32::to_bits),
            Some((4097.0f32 * 0.5).to_bits())
        );
        assert_eq!(t.value(5000), None);
        assert!(t.texels[5000..].iter().all(|v| v.to_bits() == 0));
    }

    #[test]
    fn no_values_still_make_one_row() {
        let t = DataTexture::try_from_values(&[]).unwrap();
        assert_eq!((t.height, t.len, t.texels.len()), (1, 0, 2048));
    }

    #[test]
    fn the_largest_texture_is_accepted_and_one_value_more_is_refused() {
        let max = (DATA_TEXTURE_WIDTH * MAX_DATA_TEXTURE_ROWS) as usize;
        let full = vec![1.5f32; max];
        let t = DataTexture::try_from_values(&full).unwrap();
        assert_eq!(
            (t.width, t.height),
            (DATA_TEXTURE_WIDTH, MAX_DATA_TEXTURE_ROWS)
        );
        let over = vec![1.5f32; max + 1];
        assert_eq!(
            DataTexture::try_from_values(&over),
            Err(DataExportError::TooLarge {
                values: max + 1,
                max_values: max
            })
        );
    }
}
