//! 3D Gaussian Splatting (`.splat`) I/O
//!
//! SDF を 3D Gaussian Splat 表現に変換し、Inria 3DGS 互換の `.splat` バイナリ形式で
//! 入出力する。WebGL ベースのリアルタイムビューア
//! (gsplat.tech / SuperSplat / antimatter15/splat) へ直接 drag&drop 可能。
//!
//! # `.splat` バイナリ形式 (1 splat = 32 bytes)
//!
//! | offset | size | type    | field    |
//! |--------|------|---------|----------|
//! | 0      | 12   | f32×3   | position |
//! | 12     | 12   | f32×3   | scale    |
//! | 24     | 4    | u8×4    | color rgba |
//! | 28     | 4    | u8×4    | rotation (compressed quat, `[0,255]` → `[-1,1]`) |
//!
//! # 使用例
//!
//! ```ignore
//! use alice_sdf::io::splat::{sdf_to_splats, save_splat, SplatConfig};
//! use alice_sdf::prelude::*;
//!
//! let node = SdfNode::sphere(1.0);
//! let cfg = SplatConfig {
//!     bounds: (-2.0, 2.0),
//!     resolution: 64,
//!     base_color: [220, 220, 240, 255],
//! };
//! let splats = sdf_to_splats(&node, &cfg);
//! save_splat("sphere.splat", &splats).unwrap();
//! ```

use crate::eval::{eval, eval_normal};
use crate::types::SdfNode;
use glam::Vec3;
use std::io::{Read, Write};
use std::path::Path;

/// 1 Splat = 32 bytes
pub const SPLAT_BYTES: usize = 32;

/// 3D Gaussian Splat (Inria 3DGS 互換レイアウト)
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Splat {
    /// XYZ 位置 (world space)
    pub position: [f32; 3],
    /// XYZ 各軸のスケール (= 標準偏差)
    pub scale: [f32; 3],
    /// RGBA 色 (各 u8)
    pub color: [u8; 4],
    /// 回転 (圧縮 quat、各 byte = (value + 1.0) * 127.5)
    pub rotation: [u8; 4],
}

impl Splat {
    /// バイト列にエンコード (32 bytes)
    #[must_use]
    pub fn to_bytes(&self) -> [u8; SPLAT_BYTES] {
        let mut buf = [0u8; SPLAT_BYTES];
        buf[0..4].copy_from_slice(&self.position[0].to_le_bytes());
        buf[4..8].copy_from_slice(&self.position[1].to_le_bytes());
        buf[8..12].copy_from_slice(&self.position[2].to_le_bytes());
        buf[12..16].copy_from_slice(&self.scale[0].to_le_bytes());
        buf[16..20].copy_from_slice(&self.scale[1].to_le_bytes());
        buf[20..24].copy_from_slice(&self.scale[2].to_le_bytes());
        buf[24..28].copy_from_slice(&self.color);
        buf[28..32].copy_from_slice(&self.rotation);
        buf
    }

    /// バイト列からデコード
    pub fn from_bytes(bytes: &[u8; SPLAT_BYTES]) -> Self {
        let f = |i: usize| f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        Self {
            position: [f(0), f(4), f(8)],
            scale: [f(12), f(16), f(20)],
            color: bytes[24..28].try_into().unwrap(),
            rotation: bytes[28..32].try_into().unwrap(),
        }
    }
}

/// SDF → Splat 変換の設定
#[derive(Clone, Copy, Debug)]
pub struct SplatConfig {
    /// グリッド範囲 (min, max)、立方体
    pub bounds: (f32, f32),
    /// 1 辺の voxel 数
    pub resolution: u32,
    /// 全 splat のベース色 (将来 material 評価で上書き)
    pub base_color: [u8; 4],
}

impl Default for SplatConfig {
    fn default() -> Self {
        Self {
            bounds: (-2.0, 2.0),
            resolution: 64,
            base_color: [200, 200, 220, 255],
        }
    }
}

/// 表面に近い voxel 中心点をピックして splat 化
///
/// 各 voxel の中心で SDF 距離を評価し、`|d| < voxel_size` の voxel を「表面近傍」として
/// 採用、その voxel のスケール = voxel_size、回転 = 法線方向 (X 軸を法線に合わせる)。
pub fn sdf_to_splats(node: &SdfNode, cfg: &SplatConfig) -> Vec<Splat> {
    let (min, max) = cfg.bounds;
    let n = cfg.resolution as usize;
    let step = (max - min) / cfg.resolution.max(1) as f32;
    let surface_eps = step;
    let mut out = Vec::new();
    for k in 0..n {
        let z = min + (k as f32 + 0.5) * step;
        for j in 0..n {
            let y = min + (j as f32 + 0.5) * step;
            for i in 0..n {
                let x = min + (i as f32 + 0.5) * step;
                let p = Vec3::new(x, y, z);
                let d = eval(node, p);
                if d.abs() < surface_eps {
                    let normal = eval_normal(node, p);
                    out.push(Splat {
                        position: [x, y, z],
                        scale: [step * 0.5, step * 0.5, step * 0.5],
                        color: cfg.base_color,
                        rotation: encode_rotation_from_normal(normal),
                    });
                }
            }
        }
    }
    out
}

/// 法線方向 → 簡易 quat (z 軸が法線に向くように回転、ロール 0) → u8×4 圧縮
fn encode_rotation_from_normal(n: Vec3) -> [u8; 4] {
    let n = if n.length_squared() < 1e-12 {
        Vec3::Z
    } else {
        n.normalize()
    };
    let z = Vec3::Z;
    let axis = z.cross(n);
    let dot = z.dot(n).clamp(-1.0, 1.0);
    let half_angle = dot.acos() * 0.5;
    let (s, c) = half_angle.sin_cos();
    let (qx, qy, qz, qw) = if axis.length_squared() < 1e-6 {
        // n はほぼ z 軸方向、回転なし or 180°
        if dot > 0.0 {
            (0.0, 0.0, 0.0, 1.0)
        } else {
            (1.0, 0.0, 0.0, 0.0)
        }
    } else {
        let axis = axis.normalize();
        (axis.x * s, axis.y * s, axis.z * s, c)
    };
    let enc = |v: f32| ((v * 127.5 + 127.5).clamp(0.0, 255.0)) as u8;
    [enc(qx), enc(qy), enc(qz), enc(qw)]
}

/// `.splat` ファイルに書き出し
pub fn save_splat(path: impl AsRef<Path>, splats: &[Splat]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for s in splats {
        f.write_all(&s.to_bytes())?;
    }
    f.flush()
}

/// `.splat` ファイルから読込
pub fn load_splat(path: impl AsRef<Path>) -> std::io::Result<Vec<Splat>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if buf.len() % SPLAT_BYTES != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("file size {} not multiple of {}", buf.len(), SPLAT_BYTES),
        ));
    }
    let n = buf.len() / SPLAT_BYTES;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * SPLAT_BYTES;
        let chunk: &[u8; SPLAT_BYTES] = buf[start..start + SPLAT_BYTES].try_into().unwrap();
        out.push(Splat::from_bytes(chunk));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    fn splat_size_is_32() {
        assert_eq!(std::mem::size_of::<Splat>(), 32);
        assert_eq!(SPLAT_BYTES, 32);
    }

    #[test]
    fn splat_roundtrip_bytes() {
        let s = Splat {
            position: [1.0, 2.0, 3.0],
            scale: [0.1, 0.2, 0.3],
            color: [10, 20, 30, 255],
            rotation: [128, 128, 128, 255],
        };
        let b = s.to_bytes();
        let s2 = Splat::from_bytes(&b);
        assert_eq!(s, s2);
    }

    #[test]
    fn sphere_produces_surface_splats() {
        let n = SdfNode::sphere(1.0);
        let cfg = SplatConfig {
            bounds: (-2.0, 2.0),
            resolution: 32,
            base_color: [255, 0, 0, 255],
        };
        let splats = sdf_to_splats(&n, &cfg);
        assert!(!splats.is_empty(), "sphere should produce some splats");
        // 各 splat の位置は半径 1 付近にあるはず (許容 voxel size)
        let voxel = (4.0_f32) / 32.0;
        for s in &splats {
            let r = (s.position[0].powi(2) + s.position[1].powi(2) + s.position[2].powi(2)).sqrt();
            assert!(
                (r - 1.0).abs() < voxel * 2.0,
                "splat off surface: r={r}, voxel={voxel}"
            );
        }
    }

    #[test]
    fn file_roundtrip() {
        let path = std::env::temp_dir().join("alice_sdf_test.splat");
        let n = SdfNode::sphere(1.0);
        let cfg = SplatConfig {
            bounds: (-2.0, 2.0),
            resolution: 16,
            base_color: [100, 100, 100, 200],
        };
        let splats = sdf_to_splats(&n, &cfg);
        save_splat(&path, &splats).unwrap();
        let loaded = load_splat(&path).unwrap();
        assert_eq!(splats.len(), loaded.len());
        for (a, b) in splats.iter().zip(loaded.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.color, b.color);
        }
        std::fs::remove_file(&path).ok();
    }
}
