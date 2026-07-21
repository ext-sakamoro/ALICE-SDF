//! Text → 3D SDF demo — bilingual (ASCII + Japanese CJK).
//!
//! Demonstrates `alice_sdf::font_bridge::text_to_3d` routing through
//! `alice_font::glyph_dispatcher` for mixed ASCII + hiragana / katakana /
//! CJK unified code points.
//!
//! Run with:
//!
//! ```bash
//! cargo run --features font --example text_to_3d_demo --release
//! ```
//!
//! Output: prints SDF sign map (`X` = inside surface, ` ` = outside) on
//! a horizontal Z=0 slice, plus per-character `alice_font::glyph_dispatcher`
//! category classification.
//!
//! Author: Moroya Sakamoto

use alice_font::{glyph_dispatcher, MetaFontParams};
use alice_sdf::eval::eval;
use alice_sdf::font_bridge::text_to_3d;
use glam::Vec3;

const TEXT: &str = "HELLO AI 日本";
const DEPTH: f32 = 0.3;

// Slice resolution for the ASCII visualization on the Z = 0 plane.
const COLS: usize = 96;
const ROWS: usize = 16;

fn main() {
    let params = MetaFontParams::sans_regular();

    // 1. Report dispatcher routing category per non-space character
    println!("=== Glyph dispatcher routing for '{TEXT}' ===");
    for ch in TEXT.chars() {
        if ch == ' ' {
            continue;
        }
        let cat = glyph_dispatcher::category(ch);
        println!("  '{ch}' (U+{:04X})  →  {cat:?}", ch as u32);
    }

    // 2. Build the 3D SDF
    let Some(node) = text_to_3d(TEXT, &params, DEPTH) else {
        eprintln!("text_to_3d returned None (empty text)");
        return;
    };

    // 3. Rasterize an ASCII slice on Z = 0
    // Estimate width from the number of non-space characters (advance ≈ 1.0/char).
    let non_space = TEXT.chars().filter(|c| *c != ' ').count() as f32;
    let width = non_space + TEXT.matches(' ').count() as f32 * 0.5;
    let height = 1.2f32;

    println!(
        "\n=== SDF sign map on Z=0 slice (X = inside, {ROWS} rows × {COLS} cols, width={width:.1}, depth={DEPTH}) ==="
    );

    for row in 0..ROWS {
        let y = height * 0.5 - (row as f32 / (ROWS - 1) as f32) * height;
        let mut line = String::with_capacity(COLS);
        for col in 0..COLS {
            let x = (col as f32 / (COLS - 1) as f32) * width;
            let d = eval(&node, Vec3::new(x, y, 0.0));
            line.push(if d < 0.0 { 'X' } else { ' ' });
        }
        println!("{line}");
    }

    // 4. Center-point sanity check
    let center = Vec3::new(width * 0.5, 0.0, 0.0);
    let d_center = eval(&node, center);
    println!(
        "\ncenter d = {d_center:+.4}  (finite: {})",
        d_center.is_finite()
    );
}
