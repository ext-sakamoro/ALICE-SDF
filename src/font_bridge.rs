//! ALICE-Font ↔ ALICE-SDF bridge
//!
//! Converts font glyphs (32×32 SDF grids) into 2D and 3D SDF nodes
//! for text rendering, procedural signage, and 3D typography.
//!
//! Requires the `font` feature: `cargo build --features font`
//!
//! Author: Moroya Sakamoto

use alice_font::{GlyphGenerator, GlyphSdf, MetaFontParams};

use crate::sdf2d::Sdf2dNode;
use crate::types::SdfNode;
use std::sync::Arc;

// ── Glyph → Sdf2dNode ───────────────────────────────────────

/// Convert a GlyphSdf (32×32 distance grid) into a 2D SDF node.
///
/// The resulting node uses bilinear interpolation on the grid for evaluation.
pub fn glyph_to_sdf2d(glyph: &GlyphSdf) -> Sdf2dNode {
    let mut data = Box::new([0.0f32; 1024]);
    data.copy_from_slice(&glyph.data);

    Sdf2dNode::FontGlyph {
        data,
        advance: glyph.advance,
        bbox_min: [glyph.bbox_min.x, glyph.bbox_min.y],
        bbox_max: [glyph.bbox_max.x, glyph.bbox_max.y],
    }
}

/// Convert a single ASCII character into a 2D SDF node using the given font parameters.
///
/// Non-ASCII characters are mapped to `b'?'`.
pub fn char_to_sdf2d(ch: char, params: &MetaFontParams) -> Sdf2dNode {
    let gen = GlyphGenerator::new(params);
    let byte = if ch.is_ascii() { ch as u8 } else { b'?' };
    let glyph = gen.generate(byte);
    glyph_to_sdf2d(&glyph)
}

/// Layout and convert a text string into a 2D SDF scene.
///
/// Each glyph is placed at its correct horizontal position using advance widths.
/// Returns a union of all glyph SDFs.
pub fn text_to_sdf2d(text: &str, params: &MetaFontParams) -> Option<Sdf2dNode> {
    let gen = GlyphGenerator::new(params);
    let mut result: Option<Sdf2dNode> = None;
    let mut cursor_x: f32 = 0.0;

    for ch in text.chars() {
        if ch == ' ' {
            cursor_x += 0.5; // space advance
            continue;
        }

        let byte = if ch.is_ascii() { ch as u8 } else { b'?' };
        let glyph = gen.generate(byte);
        let node = glyph_to_sdf2d(&glyph);
        let placed = node.translate(cursor_x, 0.0);
        cursor_x += glyph.advance;

        result = Some(match result {
            None => placed,
            Some(existing) => existing.union(placed),
        });
    }

    result
}

// ── Glyph → 3D SdfNode (extrusion) ──────────────────────────

/// Convert a GlyphSdf into a 3D extruded SDF node.
///
/// The glyph is evaluated in the XY plane and extruded along Z by `depth`.
/// The resulting SDF is centered at the origin in Z.
pub fn glyph_to_3d(glyph: &GlyphSdf, depth: f32) -> SdfNode {
    // Sample the glyph grid to create a mesh-free 3D SDF
    // by using the Extrude modifier on a 2D representation.
    //
    // We represent the glyph as a smooth sphere cloud approximation:
    // scan the 32×32 grid for interior cells and place small spheres.
    let half_depth = depth * 0.5;
    let w = glyph.bbox_max.x - glyph.bbox_min.x;
    let h = glyph.bbox_max.y - glyph.bbox_min.y;
    if w < 1e-6 || h < 1e-6 {
        return SdfNode::box3d(0.01, 0.01, half_depth);
    }

    let cell_w = w / 32.0;
    let cell_h = h / 32.0;
    let cell_r = cell_w.max(cell_h) * 0.6; // sphere radius per cell

    let mut spheres: Vec<SdfNode> = Vec::new();
    for iy in 0..32u32 {
        for ix in 0..32u32 {
            let d = glyph.data[(iy * 32 + ix) as usize];
            if d < 0.0 {
                // Interior cell: place a sphere
                let cx = glyph.bbox_min.x + (ix as f32 + 0.5) * cell_w;
                let cy = glyph.bbox_min.y + (iy as f32 + 0.5) * cell_h;
                spheres.push(SdfNode::sphere(cell_r).translate(cx, cy, 0.0));
            }
        }
    }

    if spheres.is_empty() {
        return SdfNode::box3d(0.01, 0.01, half_depth);
    }

    // Build balanced union tree to avoid deep recursion / stack overflow
    let k = cell_r * 0.5;
    let tree = balanced_smooth_union(spheres, k);

    // Extrude along Z
    SdfNode::Extrude {
        child: Arc::new(tree),
        half_height: half_depth,
    }
}

/// Build a balanced binary tree of SmoothUnion nodes to keep recursion depth O(log n).
fn balanced_smooth_union(mut nodes: Vec<SdfNode>, k: f32) -> SdfNode {
    while nodes.len() > 1 {
        let mut next = Vec::with_capacity(nodes.len().div_ceil(2));
        let mut i = 0;
        while i + 1 < nodes.len() {
            next.push(SdfNode::SmoothUnion {
                a: Arc::new(nodes[i].clone()),
                b: Arc::new(nodes[i + 1].clone()),
                k,
            });
            i += 2;
        }
        if i < nodes.len() {
            next.push(nodes[i].clone());
        }
        nodes = next;
    }
    nodes.into_iter().next().unwrap()
}

/// Convert a text string into a 3D extruded SDF.
pub fn text_to_3d(text: &str, params: &MetaFontParams, depth: f32) -> Option<SdfNode> {
    let gen = GlyphGenerator::new(params);
    let mut result: Option<SdfNode> = None;
    let mut cursor_x: f32 = 0.0;

    for ch in text.chars() {
        if ch == ' ' {
            cursor_x += 0.5;
            continue;
        }

        let byte = if ch.is_ascii() { ch as u8 } else { b'?' };
        let glyph = gen.generate(byte);
        let node = glyph_to_3d(&glyph, depth);
        let placed = node.translate(cursor_x, 0.0, 0.0);
        cursor_x += glyph.advance;

        result = Some(match result {
            None => placed,
            Some(existing) => SdfNode::Union {
                a: Arc::new(existing),
                b: Arc::new(placed),
            },
        });
    }

    result
}

/// Get font metrics from parameters.
pub fn font_metrics(params: &MetaFontParams) -> FontMetrics {
    FontMetrics {
        cap_height: params.cap_height,
        x_height: params.x_height,
        ascender: params.ascender,
        descender: params.descender,
        weight: params.weight,
    }
}

/// Summary of font metrics extracted from MetaFontParams.
#[derive(Debug, Clone, Copy)]
pub struct FontMetrics {
    /// Capital letter height.
    pub cap_height: f32,
    /// Lowercase x-height.
    pub x_height: f32,
    /// Ascender height (above x-height).
    pub ascender: f32,
    /// Descender depth (below baseline, typically negative).
    pub descender: f32,
    /// Weight factor (0.0 = hairline, 1.0 = black).
    pub weight: f32,
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> MetaFontParams {
        MetaFontParams::sans_regular()
    }

    #[test]
    fn glyph_to_sdf2d_conversion() {
        let gen = GlyphGenerator::new(&test_params());
        let glyph = gen.generate(b'A');
        let node = glyph_to_sdf2d(&glyph);
        match &node {
            Sdf2dNode::FontGlyph { advance, .. } => {
                assert!(*advance > 0.0);
            }
            _ => panic!("Expected FontGlyph node"),
        }
    }

    #[test]
    fn char_to_sdf2d_evaluates() {
        let node = char_to_sdf2d('O', &test_params());
        // 'O' should have interior near its center
        let gen = GlyphGenerator::new(&test_params());
        let glyph = gen.generate(b'O');
        let cx = (glyph.bbox_min.x + glyph.bbox_max.x) * 0.5;
        let cy = (glyph.bbox_min.y + glyph.bbox_max.y) * 0.5;
        let _d = crate::sdf2d::eval_2d(&node, [cx, cy]);
        // Just verify it doesn't panic
    }

    #[test]
    fn text_to_sdf2d_single_char() {
        let result = text_to_sdf2d("A", &test_params());
        assert!(result.is_some());
    }

    #[test]
    fn text_to_sdf2d_multiple_chars() {
        let result = text_to_sdf2d("HI", &test_params());
        assert!(result.is_some());
    }

    #[test]
    fn text_to_sdf2d_with_space() {
        let result = text_to_sdf2d("A B", &test_params());
        assert!(result.is_some());
    }

    #[test]
    fn text_to_sdf2d_empty() {
        let result = text_to_sdf2d("", &test_params());
        assert!(result.is_none());
    }

    #[test]
    fn glyph_to_3d_evaluates() {
        let gen = GlyphGenerator::new(&test_params());
        let glyph = gen.generate(b'T');
        let node = glyph_to_3d(&glyph, 0.5);
        let d = crate::eval::eval(&node, glam::Vec3::ZERO);
        // Just verify it doesn't panic and returns a finite value
        assert!(d.is_finite());
    }

    #[test]
    fn text_to_3d_creates_node() {
        let result = text_to_3d("A", &test_params(), 0.5);
        assert!(result.is_some());
    }

    #[test]
    fn font_metrics_extraction() {
        let params = test_params();
        let metrics = font_metrics(&params);
        assert!(metrics.cap_height > 0.0);
        assert!(metrics.x_height > 0.0);
    }

    #[test]
    fn text_to_3d_multiple_chars() {
        let result = text_to_3d("AB", &test_params(), 0.3);
        assert!(result.is_some());
        if let Some(node) = &result {
            let d = crate::eval::eval(node, glam::Vec3::new(0.0, 0.0, 0.0));
            assert!(d.is_finite());
        }
    }
}
