//! `terrain` / `destruction` の解析解突合 test (CLAUDE.md § 解析解突合テスト規律)
//!
//! 期待値は**すべて閉形式か独立に組み直した式**で、被検査関数の出力を見て書いて
//! いない 各 test に期待値の出所を `// oracle:` で付ける
//!
//! 既存 unit test (terrain 33 / destruction 22) は `modified_voxels > 0` /
//! `!vertices.is_empty()` の**存在と範囲**しか見ておらず、`heightmap.rs` の
//! `test_bilinear_sample` は閉形式が exact (中点 = 5.0) なのに許容が ±1.0 (20%)
//! だった 棚卸しは [[feedback_alice_sdf_oracle_coverage_audit_2026_09_30]]
//!
//! # 使った閉形式の一覧
//!
//! | 対象 | oracle |
//! |---|---|
//! | `Heightmap::sample` | 双線形のテンソル積形 `Σ h_ij·w_i·w_j` (実装は入れ子 lerp + `mul_add` なので演算列が別) |
//! | `sample` / `sample_bicubic` / `normal_at` | 解析高さ場 `h = a·x + b·z` の厳密再現と `normalize(−a, 1, −b)` |
//! | `sample_bicubic` | Keys (a = −1/2) の 3 次精度 = 定数 / 1 次 / 2 次を厳密再現 |
//! | `generate_fbm` | 振幅和の上界 `|h| ≤ 1` + octave 1 は smoothstep 双線形のテンソル積 (2×2 小行列式 = 0) |
//! | thermal erosion | 総和保存 (材料を隣へ移すだけ) + 孤立峰 1 pass の閉形式 |
//! | `Splatmap` | partition of unity (`Σ w = 1`) |
//! | `ClipmapTerrain::update` | `floor((cam − half)/s)·s` と `spacing = base·2^i` |
//! | `terrain_sdf` | 平坦地形は `y − h`、洞窟付きは `max(y − h, −cave)` |
//! | `MutableVoxelGrid::from_sdf` | voxel 中心での解析 SDF (`|p| − r`) |
//! | `carve` | AABB 内は `max(old, −carve)` に厳密一致 / 占有 (符号) は全域一致 / 冪等 |
//! | `removed_volume` | 完全内包の球を削ったら `4/3·π·r³`、解像度を上げると収束する |
//! | `voronoi_fracture` | 分割の保存 (Σ `voxel_count` = 内部 ∧ radius 内の voxel 数、整数で厳密) |
//! | piece 表面積 | Cauchy の射影公式 軸平行 voxel 表面の総面積 → `6πr²` (非等方 voxel でも同値) |
//! | `remesh_chunk` | MC 頂点は解析球面上 (`|v| = r`、誤差 O(h)) |
//! | `generate_debris` | 個数 `min(max_pieces, max(1, ⌊radius/min_size⌋))` / 八面体の符号付き体積 `(1/6)Π(a₊ + a₋)` |
//!
//! # 許容の決め方
//!
//! ⚠️ 「通らないから緩める」ことはしていない 実測してから、収束の次数が分かって
//! いるものは `C·h/r` の形で、厳密なものは f32 の丸め (1e-5) で書いた
//! ⚠️ voxel 化した球の表面積 / 体積の誤差は **解像度に対して単調減少しない**
//! (表面積 res 12/24/36/48 で rel 0.057 / 0.006 / 0.022 / 0.002、体積 res
//! 16/32/64 で 0.033 / 0.036 / 0.016) ので「1 段細かくすれば必ず縮む」を assert
//! してはいけない 全点で成立するのは O(h) 上界で、収束は最粗と最細の比で見る

#![cfg(all(feature = "terrain", feature = "destruction"))]

use alice_sdf::destruction::{
    carve, carve_batch, generate_debris, voronoi_fracture, CarveShape, DebrisConfig,
    FractureConfig, FracturePiece, MutableVoxelGrid,
};
use alice_sdf::mesh::Mesh;
use alice_sdf::terrain::{erode, terrain_sdf, ClipmapTerrain, ErosionConfig, Heightmap, Splatmap};
use alice_sdf::types::SdfNode;
use glam::Vec3;

/// f32 の丸めしか許さない厳密一致用の許容
const EXACT: f32 = 1e-5;

// ---------------------------------------------------------------------------
// terrain
// ---------------------------------------------------------------------------

mod terrain_oracle {
    use super::*;

    /// 双線形補間の閉形式 (テンソル積形)
    ///
    /// 実装 (`Heightmap::sample`) は入れ子 lerp + `mul_add` なので演算列が違う
    /// world → grid の写像 `fx = wx/world_width·width` は実装の規約に合わせている
    /// (⚠️ `width − 1` ではないので最終列は `wx ≥ (width−1)/width·world_width` で
    /// clamp されて平らになる、この規約自体は仕様として扱う)
    fn bilinear_closed_form(hm: &Heightmap, wx: f32, wz: f32) -> f32 {
        let fx = (wx / hm.world_width * hm.width as f32).clamp(0.0, (hm.width - 1) as f32);
        let fz = (wz / hm.world_depth * hm.depth as f32).clamp(0.0, (hm.depth - 1) as f32);
        let x0 = fx.floor() as u32;
        let z0 = fz.floor() as u32;
        let x1 = (x0 + 1).min(hm.width - 1);
        let z1 = (z0 + 1).min(hm.depth - 1);
        let tx = fx - fx.floor();
        let tz = fz - fz.floor();
        hm.get_height(x0, z0) * (1.0 - tx) * (1.0 - tz)
            + hm.get_height(x1, z0) * tx * (1.0 - tz)
            + hm.get_height(x0, z1) * (1.0 - tx) * tz
            + hm.get_height(x1, z1) * tx * tz
    }

    fn ramp(width: u32, depth: u32, world: f32, a: f32, b: f32) -> Heightmap {
        let mut hm = Heightmap::new(width, depth, world, world);
        let sx = world / width as f32;
        let sz = world / depth as f32;
        for z in 0..depth {
            for x in 0..width {
                hm.set_height(x, z, a * (x as f32 * sx) + b * (z as f32 * sz));
            }
        }
        hm
    }

    /// oracle: 双線形補間のテンソル積閉形式 (独立に組んだ式、`mul_add` 不使用)
    #[test]
    fn bilinear_sample_matches_the_closed_form_everywhere() {
        let mut hm = Heightmap::new(8, 8, 8.0, 8.0);
        for z in 0..8 {
            for x in 0..8 {
                hm.set_height(x, z, (x * 3 + z * 7) as f32 * 0.5);
            }
        }
        let mut worst = 0.0f32;
        for i in 0..40 {
            for j in 0..40 {
                let (wx, wz) = (i as f32 * 0.17, j as f32 * 0.19);
                worst = worst.max((hm.sample(wx, wz) - bilinear_closed_form(&hm, wx, wz)).abs());
            }
        }
        assert!(worst < EXACT, "bilinear closed form drift {worst:e}");

        // 手計算の 1 点 (既存 unit test `test_bilinear_sample` が ±1.0 で通して
        // いた中点) 閉形式は exact なので 20% の許容は 4.1 でも green になる
        let mut mid = Heightmap::new(4, 4, 4.0, 4.0);
        mid.set_height(1, 0, 10.0);
        mid.set_height(1, 1, 10.0);
        assert!(
            (mid.sample(0.5, 0.0) - 5.0).abs() < EXACT,
            "midpoint of 0 and 10 must be exactly 5, got {}",
            mid.sample(0.5, 0.0)
        );
    }

    /// oracle: 補間は格子点で内挿的 (節点値をそのまま返す)
    #[test]
    fn bilinear_sample_is_interpolatory_at_grid_nodes() {
        let hm = ramp(16, 16, 32.0, 0.75, -0.25);
        let spacing = 32.0 / 16.0;
        for z in 0..15u32 {
            for x in 0..15u32 {
                let got = hm.sample(x as f32 * spacing, z as f32 * spacing);
                let want = hm.get_height(x, z);
                assert!(
                    (got - want).abs() < EXACT,
                    "node ({x},{z}): got {got} want {want}"
                );
            }
        }
    }

    /// oracle: 双線形は 1 次多項式を厳密に再現する h = a·x + b·z
    #[test]
    fn bilinear_sample_reproduces_a_linear_height_field_exactly() {
        let (a, b) = (0.75f32, -0.25f32);
        let hm = ramp(16, 16, 16.0, a, b);
        let mut worst = 0.0f32;
        for i in 0..30 {
            for j in 0..30 {
                let (wx, wz) = (2.0 + i as f32 * 0.37, 2.0 + j as f32 * 0.41);
                worst = worst.max((hm.sample(wx, wz) - a.mul_add(wx, b * wz)).abs());
            }
        }
        assert!(worst < EXACT, "linear reproduction drift {worst:e}");
    }

    /// oracle: Keys の cubic convolution (a = −1/2) は 3 次精度 = 定数 / 1 次 /
    /// 2 次を厳密再現する (重みの partition of unity + 1 次・2 次モーメント一致)
    #[test]
    fn bicubic_sample_reproduces_constant_linear_and_quadratic_fields() {
        // 定数
        let mut cst = Heightmap::new(12, 12, 12.0, 12.0);
        for z in 0..12 {
            for x in 0..12 {
                cst.set_height(x, z, 3.25);
            }
        }
        let mut worst = 0.0f32;
        for i in 0..20 {
            for j in 0..20 {
                let (wx, wz) = (1.3 + i as f32 * 0.4, 1.1 + j as f32 * 0.42);
                worst = worst.max((cst.sample_bicubic(wx, wz) - 3.25).abs());
            }
        }
        assert!(worst < EXACT, "constant reproduction drift {worst:e}");

        // 1 次
        let (a, b) = (0.5f32, 0.5f32);
        let lin = ramp(12, 12, 12.0, a, b);
        let mut worst = 0.0f32;
        for i in 0..20 {
            for j in 0..20 {
                let (wx, wz) = (2.0 + i as f32 * 0.35, 2.0 + j as f32 * 0.31);
                worst = worst.max((lin.sample_bicubic(wx, wz) - a.mul_add(wx, b * wz)).abs());
            }
        }
        assert!(worst < EXACT, "linear reproduction drift {worst:e}");

        // 既存 unit test `test_bicubic_sample` の点 (h = (x+z)/2 は 1 次なので
        // 中心は厳密に 4.0、許容 ±1.0 では 3.1 でも green になる)
        let mid = ramp(8, 8, 8.0, 0.5, 0.5);
        assert!(
            (mid.sample_bicubic(4.0, 4.0) - 4.0).abs() < EXACT,
            "got {}",
            mid.sample_bicubic(4.0, 4.0)
        );

        // 2 次 (格子間隔 1.0 なので h = 0.1·x²)
        let mut quad = Heightmap::new(12, 12, 12.0, 12.0);
        for z in 0..12 {
            for x in 0..12 {
                quad.set_height(x, z, (x as f32) * (x as f32) * 0.1);
            }
        }
        let mut worst = 0.0f32;
        for i in 0..20 {
            let wx = 2.0 + i as f32 * 0.35;
            worst = worst.max((quad.sample_bicubic(wx, 5.0) - wx * wx * 0.1).abs());
        }
        assert!(worst < EXACT, "quadratic reproduction drift {worst:e}");
    }

    /// oracle: h = a·x + b·z の単位法線は `normalize(−a, 1, −b)`
    #[test]
    fn normal_at_matches_the_analytic_normal_of_a_ramp() {
        for &(a, b) in &[(0.75f32, -0.25f32), (0.0, 0.0), (-1.5, 2.0)] {
            let hm = ramp(16, 16, 16.0, a, b);
            let want = Vec3::new(-a, 1.0, -b).normalize();
            let mut worst = 0.0f32;
            for i in 0..25 {
                for j in 0..25 {
                    let (wx, wz) = (2.0 + i as f32 * 0.41, 2.0 + j as f32 * 0.37);
                    worst = worst.max((hm.normal_at(wx, wz) - want).length());
                }
            }
            assert!(worst < EXACT, "ramp ({a},{b}) normal drift {worst:e}");
        }
    }

    /// oracle: `height_range` は格子の実 min/max、`normalize` は affine 写像
    /// `(h − min)/(max − min)` を全 cell に適用したもの
    #[test]
    fn height_range_and_normalize_match_their_definitions() {
        let mut hm = Heightmap::new(16, 16, 16.0, 16.0);
        hm.generate_fbm(4, 0.5, 2.0, 99);
        hm.scale_heights(7.5);

        let raw = hm.heights.clone();
        let want_min = raw.iter().copied().fold(f32::INFINITY, f32::min);
        let want_max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (min, max) = hm.height_range();
        assert_eq!(min, want_min);
        assert_eq!(max, want_max);

        hm.normalize();
        let span = want_max - want_min;
        for (i, &h) in hm.heights.iter().enumerate() {
            let want = (raw[i] - want_min) / span;
            assert!((h - want).abs() < EXACT, "cell {i}: {h} vs {want}");
        }
        let (nmin, nmax) = hm.height_range();
        assert!((nmin - 0.0).abs() < EXACT && (nmax - 1.0).abs() < EXACT);
    }

    /// oracle: fBm の値は振幅和で正規化されるので `|h| ≤ Σ aᵏ / Σ aᵏ = 1` が上界
    /// octave 数と persistence を振っても上界は変わらない
    #[test]
    fn generate_fbm_never_exceeds_the_amplitude_sum_bound() {
        for octaves in 1..=8u32 {
            for &persistence in &[0.25f32, 0.5, 0.9] {
                let mut hm = Heightmap::new(32, 32, 32.0, 32.0);
                hm.generate_fbm(octaves, persistence, 2.0, 42);
                let (min, max) = hm.height_range();
                assert!(
                    min >= -1.0 - 1e-6 && max <= 1.0 + 1e-6,
                    "octaves={octaves} persistence={persistence} range=({min},{max})"
                );
                assert!(
                    hm.heights.iter().all(|h| h.is_finite()),
                    "octaves={octaves} produced non-finite heights"
                );
            }
        }
    }

    /// oracle: octave 1 の値ノイズは smoothstep 双線形 = 格子 cell の 4 隅の
    /// テンソル積なので、`D(x,z) = h(x,z) − h(x,0) − h(0,z) + h(0,0)` は
    /// `sx(x)·sz(z)·C` の形になり **2×2 小行列式が全部 0** になる
    /// (既定 `scale = 1/max(width,depth)` で全体が格子 1 cell に収まることが前提)
    #[test]
    fn generate_fbm_octave_one_is_a_rank_one_smoothstep_patch() {
        let mut hm = Heightmap::new(16, 16, 16.0, 16.0);
        hm.generate_fbm(1, 0.5, 2.0, 42);
        let d = |x: u32, z: u32| {
            hm.get_height(x, z) - hm.get_height(x, 0) - hm.get_height(0, z) + hm.get_height(0, 0)
        };
        let mut worst_minor = 0.0f32;
        let mut scale = 0.0f32;
        for x in 1..16u32 {
            for z in 1..16u32 {
                scale = scale.max(d(x, z).abs());
                for x2 in 1..16u32 {
                    for z2 in 1..16u32 {
                        worst_minor =
                            worst_minor.max((d(x, z) * d(x2, z2) - d(x, z2) * d(x2, z)).abs());
                    }
                }
            }
        }
        assert!(scale > 0.1, "degenerate patch (d_max={scale:e})");
        assert!(
            worst_minor < 1e-6,
            "octave-1 noise is not a tensor product: worst 2x2 minor {worst_minor:e}"
        );
    }

    /// oracle: 同一 seed の fBm は bit 一致 (決定論)
    #[test]
    fn generate_fbm_is_bit_identical_for_the_same_seed() {
        let mut a = Heightmap::new(24, 24, 24.0, 24.0);
        let mut b = Heightmap::new(24, 24, 24.0, 24.0);
        a.generate_fbm(5, 0.5, 2.0, 2024);
        b.generate_fbm(5, 0.5, 2.0, 2024);
        assert!(a
            .heights
            .iter()
            .zip(b.heights.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    /// oracle: thermal erosion は材料を隣の cell へ移すだけなので **総和が保存**
    /// する (中心から引いた `transfer` を、比の和が 1 になる重みで低い隣へ配る)
    ///
    /// ⚠️ hydraulic は堆積物を系外に持ち出すので保存しない `rain_amount = 0.0`
    /// にすると `water < min_water` で初手 break するので thermal だけ残る
    /// ⚠️ pass 数 (= `iterations/100`) を 1 / 5 / 16 と振っても保存は破れない
    #[test]
    fn thermal_erosion_conserves_total_height() {
        for iterations in [100u32, 500, 1600] {
            let mut hm = Heightmap::new(24, 24, 24.0, 24.0);
            hm.generate_fbm(4, 0.5, 2.0, 7);
            hm.scale_heights(20.0);
            let before: f64 = hm.heights.iter().map(|&v| f64::from(v)).sum();

            erode(
                &mut hm,
                &ErosionConfig {
                    iterations,
                    rain_amount: 0.0,
                    ..Default::default()
                },
            );

            let after: f64 = hm.heights.iter().map(|&v| f64::from(v)).sum();
            let rel = ((after - before) / before.abs()).abs();
            assert!(
                rel < 1e-7,
                "iterations={iterations}: total height moved {before} -> {after} (rel {rel:e})"
            );
        }
    }

    /// oracle: 孤立した尖峰は 1 pass で
    /// `h − (h − tan(thermal_angle)·cell)·0.5·thermal_rate` まで下がる
    /// (`cell = world_width/width`)
    #[test]
    fn thermal_erosion_lowers_an_isolated_peak_by_the_closed_form_transfer() {
        let mut hm = Heightmap::new(16, 16, 16.0, 16.0);
        hm.set_height(8, 8, 10.0);
        let cfg = ErosionConfig {
            iterations: 100, // passes = iterations/100 = 1
            rain_amount: 0.0,
            thermal_rate: 1.0,
            ..Default::default()
        };
        let cell = 16.0f32 / 16.0;
        let max_slope = cfg.thermal_angle.tan() * cell;
        let want = 10.0 - (10.0 - max_slope) * 0.5 * cfg.thermal_rate;

        erode(&mut hm, &cfg);
        let got = hm.get_height(8, 8);
        assert!(
            (got - want).abs() < 1e-4,
            "peak after 1 thermal pass: got {got} want {want} (max_slope {max_slope})"
        );
    }

    /// oracle: splatmap の weight は各 texel で partition of unity
    /// (`normalize` の定義そのもの、`auto_splat` も分岐すべてが 1 を保つ)
    #[test]
    fn splatmap_weights_form_a_partition_of_unity() {
        // normalize
        let mut sp = Splatmap::new(4, 4);
        sp.add_layer("a", 0, 2.0);
        sp.add_layer("b", 1, 3.0);
        sp.add_layer("c", 2, 5.0);
        sp.normalize();
        for z in 0..4 {
            for x in 0..4 {
                let s: f32 = (0..3).map(|l| sp.get_weight(l, x, z)).sum();
                assert!((s - 1.0).abs() < EXACT, "normalize sum {s}");
            }
        }
        // 比も閉形式 2 : 3 : 5
        assert!((sp.get_weight(0, 0, 0) - 0.2).abs() < EXACT);
        assert!((sp.get_weight(1, 0, 0) - 0.3).abs() < EXACT);
        assert!((sp.get_weight(2, 0, 0) - 0.5).abs() < EXACT);

        // auto_splat は「急斜面 / 高度」の分岐が何本通っても和 1 を保つ
        let mut terr = Heightmap::new(16, 16, 16.0, 16.0);
        terr.generate_fbm(3, 0.5, 2.0, 42);
        terr.scale_heights(10.0);
        for layers in [1usize, 2, 3, 4] {
            let mut sp = Splatmap::new(16, 16);
            for l in 0..layers {
                sp.add_layer(&format!("l{l}"), l as u16, 0.0);
            }
            sp.auto_splat_from_heightmap(&terr, 0.3, 5.0);
            for z in 0..16 {
                for x in 0..16 {
                    let s: f32 = (0..layers).map(|l| sp.get_weight(l, x, z)).sum();
                    assert!(
                        (s - 1.0).abs() < EXACT,
                        "layers={layers} texel ({x},{z}) sum {s}"
                    );
                    for l in 0..layers {
                        let w = sp.get_weight(l, x, z);
                        assert!(
                            (-EXACT..=1.0 + EXACT).contains(&w),
                            "weight {w} out of [0,1]"
                        );
                    }
                }
            }
        }
    }

    /// oracle: clipmap の origin は `floor((cam − half_extent)/spacing)·spacing`
    /// で、`spacing = base·2^level` snap 後の origin は spacing の整数倍
    #[test]
    fn clipmap_update_snaps_to_the_closed_form_origin() {
        for &(cam_x, cam_z) in &[(7.3f32, 5.1f32), (-13.7, 0.0), (1024.5, -2048.25)] {
            let mut cm = ClipmapTerrain::new(5, 8, 0.5);
            cm.update(Vec3::new(cam_x, 0.0, cam_z));
            for (i, lv) in cm.levels.iter().enumerate() {
                let want_spacing = 0.5 * (1u32 << i) as f32;
                assert_eq!(lv.spacing, want_spacing, "level {i} spacing");
                let half = want_spacing * lv.resolution as f32 * 0.5;
                assert_eq!(
                    lv.origin_x,
                    ((cam_x - half) / want_spacing).floor() * want_spacing,
                    "level {i} origin_x"
                );
                assert_eq!(
                    lv.origin_z,
                    ((cam_z - half) / want_spacing).floor() * want_spacing,
                    "level {i} origin_z"
                );
                // snap の定義: origin は spacing の整数倍
                assert_eq!(
                    lv.origin_x / want_spacing,
                    (lv.origin_x / want_spacing).round(),
                    "level {i} origin_x not on the lattice"
                );
            }
        }
    }

    /// oracle: 1 次高さ場の上に張った clipmap mesh は頂点位置 / 高さ / 法線 / uv
    /// がすべて閉形式 (`origin + g·spacing`, `a·x + b·z`, `normalize(−a,1,−b)`,
    /// `g/(res−1)`)
    #[test]
    fn clipmap_level_mesh_matches_the_analytic_ramp() {
        let (a, b) = (0.75f32, -0.25f32);
        let hm = ramp(64, 64, 64.0, a, b);
        let res = 8u32;
        let mut cm = ClipmapTerrain::new(1, res, 1.0);
        cm.update(Vec3::new(16.0, 0.0, 16.0));
        let lv = cm.levels[0].clone();
        let want_n = Vec3::new(-a, 1.0, -b).normalize();

        let meshes = cm.generate_meshes(&hm);
        assert_eq!(meshes.len(), 1);
        let mesh = &meshes[0].mesh;
        assert_eq!(mesh.vertices.len(), (res * res) as usize);
        assert_eq!(mesh.indices.len(), ((res - 1) * (res - 1) * 6) as usize);

        for (i, v) in mesh.vertices.iter().enumerate() {
            let gx = (i as u32 % res) as f32;
            let gz = (i as u32 / res) as f32;
            let wx = lv.origin_x + gx * lv.spacing;
            let wz = lv.origin_z + gz * lv.spacing;
            assert_eq!(v.position.x, wx, "vertex {i} x");
            assert_eq!(v.position.z, wz, "vertex {i} z");
            assert!(
                (v.position.y - a.mul_add(wx, b * wz)).abs() < EXACT,
                "vertex {i} y = {} want {}",
                v.position.y,
                a.mul_add(wx, b * wz)
            );
            assert!((v.normal - want_n).length() < EXACT, "vertex {i} normal");
            assert!(
                (v.uv.x - gx / (res - 1) as f32).abs() < EXACT,
                "vertex {i} u"
            );
            assert!(
                (v.uv.y - gz / (res - 1) as f32).abs() < EXACT,
                "vertex {i} v"
            );
        }
    }

    /// oracle: 平坦地形 (高さ h₀) の terrain SDF は厳密に `y − h₀`、洞窟付きは
    /// `max(y − h₀, −cave)` (cave は解析球 `|p − c| − r`)
    #[test]
    fn terrain_sdf_matches_the_closed_form_for_a_flat_field_and_a_spherical_cave() {
        let h0 = 2.5f32;
        let mut hm = Heightmap::new(8, 8, 8.0, 8.0);
        for z in 0..8 {
            for x in 0..8 {
                hm.set_height(x, z, h0);
            }
        }
        for &y in &[-3.0f32, 0.0, 2.5, 4.0, 10.0] {
            let p = Vec3::new(3.0, y, 3.0);
            assert!(
                (terrain_sdf(&hm, p, None) - (y - h0)).abs() < 1e-6,
                "flat terrain at y={y}"
            );
        }

        let (c, r) = (Vec3::new(3.0, 1.0, 3.0), 1.0f32);
        let cave = SdfNode::sphere(r).translate(c.x, c.y, c.z);
        for &y in &[0.5f32, 1.0, 1.2, 1.9, 2.4] {
            let p = Vec3::new(3.0, y, 3.0);
            let want = (y - h0).max(-((p - c).length() - r));
            assert!(
                (terrain_sdf(&hm, p, Some(&cave)) - want).abs() < 1e-6,
                "carved terrain at y={y}: got {} want {want}",
                terrain_sdf(&hm, p, Some(&cave))
            );
        }
    }
}

// ---------------------------------------------------------------------------
// destruction
// ---------------------------------------------------------------------------

mod destruction_oracle {
    use super::*;

    const R_MAT: f32 = 1.5; // 材料の球半径
    const BOUND: f32 = 2.0;

    fn sphere_grid(res: u32) -> MutableVoxelGrid {
        MutableVoxelGrid::from_sdf(
            &SdfNode::sphere(R_MAT),
            [res, res, res],
            Vec3::splat(-BOUND),
            Vec3::splat(BOUND),
        )
    }

    fn ball(radius: f32) -> f32 {
        4.0 / 3.0 * std::f32::consts::PI * radius.powi(3)
    }

    /// mesh の符号付き体積 (発散定理、`Σ a·(b×c)/6`)
    fn signed_volume(mesh: &Mesh) -> f64 {
        mesh.indices
            .chunks(3)
            .map(|t| {
                let a = mesh.vertices[t[0] as usize].position;
                let b = mesh.vertices[t[1] as usize].position;
                let c = mesh.vertices[t[2] as usize].position;
                f64::from(a.dot(b.cross(c))) / 6.0
            })
            .sum()
    }

    /// mesh の総面積 (三角形ごとの `|(b−a)×(c−a)|/2`)
    fn surface_area(mesh: &Mesh) -> f64 {
        mesh.indices
            .chunks(3)
            .map(|t| {
                let a = mesh.vertices[t[0] as usize].position;
                let b = mesh.vertices[t[1] as usize].position;
                let c = mesh.vertices[t[2] as usize].position;
                f64::from((b - a).cross(c - a).length()) * 0.5
            })
            .sum()
    }

    /// oracle: `from_sdf` は voxel 中心で SDF を評価する契約なので、値は
    /// `|grid_to_world(x,y,z)| − r` に一致する (`world_to_grid` はその逆写像)
    #[test]
    fn from_sdf_stores_the_analytic_field_at_voxel_centers() {
        for res in [8u32, 16, 32] {
            let grid = sphere_grid(res);
            let mut worst = 0.0f32;
            for z in 0..res {
                for y in 0..res {
                    for x in 0..res {
                        let w = grid.grid_to_world(x, y, z);
                        worst =
                            worst.max((grid.get_distance(x, y, z) - (w.length() - R_MAT)).abs());
                        assert_eq!(
                            grid.world_to_grid(w),
                            Some([x, y, z]),
                            "world_to_grid round trip at ({x},{y},{z})"
                        );
                    }
                }
            }
            assert!(worst < 1e-6, "res={res} analytic field drift {worst:e}");

            // voxel_size は bounds/res の定義そのもの
            let vs = grid.voxel_size();
            let want = 2.0 * BOUND / res as f32;
            assert!((vs.x - want).abs() < EXACT && (vs.y - want).abs() < EXACT);
        }
    }

    /// oracle: `carve` の契約は `max(old, −carve)` — carve 形状の AABB 内では
    /// **厳密に一致**する (AABB 外は culling されるので値は据え置き、ただし
    /// **占有 (符号) は全域で一致**しなければならない = 形は正しい)
    #[test]
    fn carve_matches_the_csg_closed_form_inside_the_aabb_and_the_sign_everywhere() {
        let rc = 0.6f32;
        for res in [16u32, 32] {
            let mut grid = sphere_grid(res);
            let before = grid.distances.clone();
            carve(
                &mut grid,
                &CarveShape::Sphere {
                    center: Vec3::ZERO,
                    radius: rc,
                },
            );
            let mut inside_aabb = 0u32;
            for z in 0..res {
                for y in 0..res {
                    for x in 0..res {
                        let i = grid.voxel_index(x, y, z);
                        let w = grid.grid_to_world(x, y, z);
                        let want = before[i].max(-(w.length() - rc));
                        let got = grid.distances[i];
                        assert_eq!(
                            want < 0.0,
                            got < 0.0,
                            "occupancy differs at ({x},{y},{z}): got {got} want {want}"
                        );
                        if w.x.abs() <= rc && w.y.abs() <= rc && w.z.abs() <= rc {
                            inside_aabb += 1;
                            assert!(
                                (got - want).abs() < 1e-6,
                                "CSG drift inside AABB at ({x},{y},{z}): got {got} want {want}"
                            );
                        }
                    }
                }
            }
            assert!(inside_aabb > 0, "res={res} tested no voxel inside the AABB");
        }
    }

    /// oracle: `max` は冪等なので同じ形状を 2 回 carve しても 2 回目は 1 voxel も
    /// 動かない 逆順の batch も `max` の可換性から bit 一致する
    #[test]
    fn carve_is_idempotent_and_order_independent() {
        let shape = CarveShape::Sphere {
            center: Vec3::ZERO,
            radius: 0.6,
        };
        let mut grid = sphere_grid(24);
        let first = carve(&mut grid, &shape);
        assert!(first.modified_voxels > 0, "first carve did nothing");
        let second = carve(&mut grid, &shape);
        assert_eq!(second.modified_voxels, 0, "carve is not idempotent");
        assert_eq!(
            second.removed_volume, 0.0,
            "re-carving the same shape removed more material"
        );

        let a = CarveShape::Sphere {
            center: Vec3::new(0.4, 0.0, 0.0),
            radius: 0.5,
        };
        let b = CarveShape::Box {
            center: Vec3::new(-0.3, 0.2, 0.0),
            half_extents: Vec3::new(0.4, 0.3, 0.5),
            rotation: glam::Quat::from_rotation_y(0.7),
        };
        let mut ab = sphere_grid(24);
        carve(&mut ab, &a);
        carve(&mut ab, &b);
        let mut ba = sphere_grid(24);
        carve(&mut ba, &b);
        carve(&mut ba, &a);
        assert!(
            ab.distances
                .iter()
                .zip(ba.distances.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "carve order changed the field"
        );
    }

    /// oracle: 材料に完全内包された球を carve したときの除去体積は `4/3·π·r³`
    /// voxel 化の誤差は表面項なので O(h/r)、実測 C ≤ 0.25 (res 16/32/64)
    ///
    /// ⚠️ `removed_volume` は doc が「volume in world units cubed」と宣言して
    /// いるので、符号反転 voxel から独立に組んだ体積と**両方**突き合わせる
    /// ⚠️ 収束は 1 段ごとには単調でない (実測 rel 0.033 / 0.036 / 0.016) ので
    /// 最粗と最細の比で見る 収束しない実装 (旧: 2.099 / 1.574 / 1.567) はここで
    /// 落ちる
    #[test]
    fn carve_reports_the_analytic_sphere_volume_and_converges() {
        let rc = 0.6f32;
        let exact = ball(rc);
        let mut rels = Vec::new();
        for res in [16u32, 32, 64] {
            let mut grid = sphere_grid(res);
            let before = grid.distances.clone();
            let result = carve(
                &mut grid,
                &CarveShape::Sphere {
                    center: Vec3::ZERO,
                    radius: rc,
                },
            );
            let vs = grid.voxel_size();

            // 独立計算: 符号が反転した voxel の体積
            let flipped = (0..grid.voxel_count())
                .filter(|&i| before[i] < 0.0 && grid.distances[i] >= 0.0)
                .count();
            let independent = flipped as f32 * vs.x * vs.y * vs.z;
            assert!(
                (result.removed_volume - independent).abs() < 1e-5 * independent,
                "res={res}: removed_volume {} vs sign-flip volume {independent}",
                result.removed_volume
            );

            let rel = (result.removed_volume - exact).abs() / exact;
            let bound = 0.25 * vs.x / rc;
            assert!(
                rel < bound,
                "res={res}: removed volume {} vs analytic {exact} (rel {rel:.4} > O(h) bound {bound:.4})",
                result.removed_volume
            );
            rels.push(rel);
        }
        assert!(
            rels[2] * 1.5 < rels[0],
            "removed volume does not converge: rel {:.4} (res 16) -> {:.4} (res 64)",
            rels[0],
            rels[2]
        );
    }

    /// oracle: 重なった形状を batch で削っても、符号が反転する voxel は 1 度だけ
    /// なので体積は**和集合**の体積 (二重計上しない) 離れた 2 球なら `2·4/3πr³`
    #[test]
    fn carve_batch_reports_the_union_volume_without_double_counting() {
        let rc = 0.5f32;
        let same = CarveShape::Sphere {
            center: Vec3::ZERO,
            radius: rc,
        };
        let mut once = sphere_grid(32);
        let single = carve(&mut once, &same);
        let mut twice = sphere_grid(32);
        let doubled = carve_batch(&mut twice, &[same.clone(), same.clone()]);
        assert!(
            (doubled.removed_volume - single.removed_volume).abs() < 1e-5,
            "carving the same sphere twice double counted: {} vs {}",
            doubled.removed_volume,
            single.removed_volume
        );

        // 離れた 2 球 (材料の内側に完全内包)
        let r2 = 0.35f32;
        let mut grid = sphere_grid(48);
        let disjoint = carve_batch(
            &mut grid,
            &[
                CarveShape::Sphere {
                    center: Vec3::new(0.7, 0.0, 0.0),
                    radius: r2,
                },
                CarveShape::Sphere {
                    center: Vec3::new(-0.7, 0.0, 0.0),
                    radius: r2,
                },
            ],
        );
        let exact = 2.0 * ball(r2);
        let rel = (disjoint.removed_volume - exact).abs() / exact;
        let bound = 0.25 * grid.voxel_size().x / r2;
        assert!(
            rel < bound,
            "two disjoint spheres: {} vs {exact} (rel {rel:.4} > bound {bound:.4})",
            disjoint.removed_volume
        );
    }

    /// oracle: Voronoi 分割は「内部かつ radius 内の voxel」をちょうど分け切る
    /// (nearest-seed 割当なので重複も取りこぼしも無い) 整数で厳密に一致する
    #[test]
    fn voronoi_fracture_partitions_every_interior_voxel_exactly_once() {
        let res = 24u32;
        let grid = sphere_grid(res);
        let frac_radius = 3.0f32; // 材料全体を覆う
        let mut want = 0u32;
        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    let w = grid.grid_to_world(x, y, z);
                    if grid.get_distance(x, y, z) < 0.0 && w.length_squared() <= frac_radius.powi(2)
                    {
                        want += 1;
                    }
                }
            }
        }
        assert!(want > 0);

        for piece_count in [1u32, 4, 8] {
            let pieces = voronoi_fracture(
                &grid,
                Vec3::ZERO,
                frac_radius,
                &FractureConfig {
                    piece_count,
                    min_piece_size: 0.0,
                    ..Default::default()
                },
            );
            let total: u32 = pieces.iter().map(|p| p.voxel_count).sum();
            assert_eq!(
                total, want,
                "piece_count={piece_count}: assigned {total} voxels, interior count is {want}"
            );
            assert!(
                pieces.len() as u32 <= piece_count,
                "piece_count={piece_count} produced {} pieces",
                pieces.len()
            );
        }
    }

    /// oracle: 中心対称な材料を 1 片に割ると重心は原点 (voxel 中心が原点対称に
    /// 並ぶので厳密) 同 seed の再実行は頂点まで bit 一致
    #[test]
    fn voronoi_single_piece_centroid_is_the_analytic_center_and_is_deterministic() {
        let grid = sphere_grid(24);
        let cfg = FractureConfig {
            piece_count: 1,
            min_piece_size: 0.0,
            ..Default::default()
        };
        let a = voronoi_fracture(&grid, Vec3::ZERO, 3.0, &cfg);
        let b = voronoi_fracture(&grid, Vec3::ZERO, 3.0, &cfg);
        assert_eq!(a.len(), 1);
        assert!(
            a[0].center.length() < EXACT,
            "centroid of a centered sphere must be the origin, got {:?}",
            a[0].center
        );
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.voxel_count, pb.voxel_count);
            assert!(pa
                .mesh
                .vertices
                .iter()
                .zip(pb.mesh.vertices.iter())
                .all(|(x, y)| x.position == y.position));
        }
    }

    /// oracle: **Cauchy の射影公式** 軸平行な voxel 表面では、法線 ±X の面の枚数
    /// は X 方向の柱が材料に入る回数 = 射影面積/(vs.y·vs.z) なので、3 軸を合わせ
    /// た総面積は voxel の縦横比に関係なく `6πr²` に収束する
    ///
    /// ⚠️ 非等方 grid を必ず入れる (`vs.x` を 3 軸に流用していると 1.7 倍になる)
    /// ⚠️ 収束は単調でない (実測 res 12/24/36/48 で rel 0.057/0.006/0.022/0.002)
    /// のでここでは O(h) 上界だけを主張する
    #[test]
    fn voronoi_piece_surface_area_matches_the_cauchy_projection_formula() {
        let target = f64::from(6.0 * std::f32::consts::PI * R_MAT * R_MAT);
        let cfg = FractureConfig {
            piece_count: 1,
            min_piece_size: 0.0,
            ..Default::default()
        };
        for resolution in [
            [12u32, 12, 12],
            [24, 24, 24],
            [36, 36, 36],
            [48, 48, 48],
            [24, 48, 24], // 非等方
            [16, 24, 48], // 3 軸すべて別
        ] {
            let grid = MutableVoxelGrid::from_sdf(
                &SdfNode::sphere(R_MAT),
                resolution,
                Vec3::splat(-BOUND),
                Vec3::splat(BOUND),
            );
            let pieces = voronoi_fracture(&grid, Vec3::ZERO, 3.0, &cfg);
            assert_eq!(pieces.len(), 1, "{resolution:?}");
            let vs = grid.voxel_size();
            let h = vs.x.max(vs.y).max(vs.z);
            let area = surface_area(&pieces[0].mesh);
            let rel = (area - target).abs() / target;
            let bound = f64::from(0.4 * h / R_MAT);
            assert!(
                rel < bound,
                "{resolution:?}: voxel surface area {area:.4} vs 6*pi*r^2 {target:.4} (rel {rel:.4} > bound {bound:.4})"
            );
        }
    }

    /// oracle: piece の面は軸平行で外向き、閉じた表面なので符号付き体積は球の
    /// 体積 `4/3·π·r³` に O(h) で一致する
    #[test]
    fn voronoi_piece_faces_are_outward_and_axis_aligned() {
        let grid = sphere_grid(24);
        let pieces = voronoi_fracture(
            &grid,
            Vec3::ZERO,
            3.0,
            &FractureConfig {
                piece_count: 1,
                min_piece_size: 0.0,
                ..Default::default()
            },
        );
        let p: &FracturePiece = &pieces[0];
        let vol = signed_volume(&p.mesh);
        let exact = f64::from(ball(R_MAT));
        let rel = (vol - exact).abs() / exact;
        assert!(
            rel < 0.05,
            "signed volume {vol:.4} vs analytic {exact:.4} (rel {rel:.4})"
        );
        for t in p.mesh.indices.chunks(3) {
            let a = p.mesh.vertices[t[0] as usize].position;
            let b = p.mesh.vertices[t[1] as usize].position;
            let c = p.mesh.vertices[t[2] as usize].position;
            let geo = (b - a).cross(c - a);
            let n = p.mesh.vertices[t[0] as usize].normal;
            assert!(
                geo.dot(n) > 0.0,
                "geometric normal disagrees with the stored normal"
            );
            assert!(
                (n.abs() - Vec3::X).length() < EXACT
                    || (n.abs() - Vec3::Y).length() < EXACT
                    || (n.abs() - Vec3::Z).length() < EXACT,
                "face normal {n:?} is not axis aligned"
            );
        }
    }

    /// oracle: `remesh_chunk` の MC 頂点は解析球面上に乗る (`|v| = r`)
    /// MC の頂点は voxel 中心を繋ぐ辺の上で線形補間されるので誤差は O(h)
    /// 実測 C ≤ 0.02 (res 16 で 0.0040 / res 32 で 0.0012)
    ///
    /// ⚠️ MC cell の原点を voxel の**角**に置くと面全体が `half_step` ずれる
    /// (実測 res16 で 0.218 = `|half_step|`) 距離値は voxel 中心で評価されて
    /// いるので、cell の 8 隅は 8 個の voxel 中心
    #[test]
    fn remesh_chunk_vertices_lie_on_the_analytic_sphere() {
        for res in [16u32, 32] {
            let grid = sphere_grid(res);
            let vs = grid.voxel_size();
            let [cx_n, cy_n, cz_n] = grid.chunks_per_axis();
            let mut worst = 0.0f32;
            let mut count = 0u32;
            for cz in 0..cz_n {
                for cy in 0..cy_n {
                    for cx in 0..cx_n {
                        let mesh = grid.remesh_chunk(cx, cy, cz);
                        for v in &mesh.vertices {
                            worst = worst.max((v.position.length() - R_MAT).abs());
                            count += 1;
                        }
                        assert_eq!(
                            mesh.indices.len(),
                            mesh.vertices.len(),
                            "chunk ({cx},{cy},{cz}) index/vertex mismatch"
                        );
                    }
                }
            }
            assert!(count > 0, "res={res} produced no surface");
            let bound = 0.1 * vs.x;
            assert!(
                worst < bound,
                "res={res}: worst |dist to sphere| {worst:.6} > bound {bound:.6} ({count} verts)"
            );
        }
    }

    /// oracle: debris の個数は `min(max_pieces, max(1, ⌊radius/min_size⌋))`、
    /// 中心は carve 球内 (`|c − center| ≤ radius`)、半径は `[min_size, max_size]`、
    /// volume は球の閉形式 `4/3·π·r³`
    #[test]
    fn generate_debris_counts_and_bounds_match_the_closed_form() {
        for &(radius, min_size, max_size, max_pieces) in &[
            (1.0f32, 0.05f32, 0.2f32, 8u32),
            (0.2, 0.05, 0.2, 8),
            (1.0, 0.5, 0.9, 8),
            (10.0, 0.05, 0.2, 3),
        ] {
            let cfg = DebrisConfig {
                max_pieces,
                min_size,
                max_size,
                seed: 42,
            };
            let center = Vec3::new(1.0, -2.0, 0.5);
            let pieces = generate_debris(center, radius, &cfg);
            let want = max_pieces.min(((radius / min_size) as u32).max(1));
            assert_eq!(pieces.len() as u32, want, "piece count for r={radius}");

            for p in &pieces {
                assert!(
                    (p.center - center).length() <= radius + EXACT,
                    "debris center escaped the carve sphere"
                );
                assert!(
                    (min_size - EXACT..=max_size + EXACT).contains(&p.radius),
                    "debris radius {} outside [{min_size}, {max_size}]",
                    p.radius
                );
                let want_vol = ball(p.radius);
                assert!(
                    (p.volume - want_vol).abs() < 1e-6 * want_vol.max(1.0),
                    "debris volume {} vs 4/3 pi r^3 {want_vol}",
                    p.volume
                );
            }
        }
    }

    /// oracle: debris mesh は歪めた八面体 頂点は中心から軸方向に `r·[0.7, 1.3]`、
    /// 8 面は全部外向き、符号付き体積は `(1/6)·Π(a₊ + a₋)` (8 個の直角四面体の和)
    #[test]
    fn debris_mesh_is_a_distorted_octahedron_with_the_closed_form_volume() {
        let center = Vec3::new(0.25, -0.5, 2.0);
        let pieces = generate_debris(center, 1.0, &DebrisConfig::default());
        assert!(!pieces.is_empty());
        for p in &pieces {
            assert_eq!(p.mesh.vertices.len(), 24, "8 faces x 3 verts");
            assert_eq!(p.mesh.indices.len(), 24);

            // 頂点は **piece の中心** から軸方向、距離は r·[0.7, 1.3]
            // (carve 中心ではない — piece は carve 球内のどこかに散る)
            let mut ext = [0.0f32; 6]; // +x, -x, +y, -y, +z, -z
            for v in &p.mesh.vertices {
                let d = v.position - p.center;
                let axes = [d.x, -d.x, d.y, -d.y, d.z, -d.z];
                let which = axes
                    .iter()
                    .position(|&c| c > EXACT && (d.length() - c).abs() < EXACT)
                    .unwrap_or_else(|| panic!("vertex {d:?} is not on an axis from the center"));
                let ratio = d.length() / p.radius;
                assert!(
                    (0.7 - EXACT..=1.3 + EXACT).contains(&ratio),
                    "distortion ratio {ratio} outside [0.7, 1.3]"
                );
                ext[which] = d.length();
            }
            assert!(ext.iter().all(|&e| e > 0.0), "some axis has no vertex");

            // 外向き
            for t in p.mesh.indices.chunks(3) {
                let a = p.mesh.vertices[t[0] as usize].position;
                let b = p.mesh.vertices[t[1] as usize].position;
                let c = p.mesh.vertices[t[2] as usize].position;
                let geo = (b - a).cross(c - a);
                let centroid = (a + b + c) / 3.0;
                assert!(
                    geo.dot(centroid - p.center) > 0.0,
                    "octahedron face is inward facing"
                );
            }

            // 体積は (1/6)·Π(a₊ + a₋) (符号付き体積は原点基準なので中心を原点へ)
            let want = f64::from((ext[0] + ext[1]) * (ext[2] + ext[3]) * (ext[4] + ext[5])) / 6.0;
            let shifted = Mesh {
                vertices: p
                    .mesh
                    .vertices
                    .iter()
                    .map(|v| {
                        let mut v = *v;
                        v.position -= p.center;
                        v
                    })
                    .collect(),
                indices: p.mesh.indices.clone(),
            };
            let got = signed_volume(&shifted);
            assert!(
                (got - want).abs() < 1e-5 * want,
                "octahedron volume {got} vs (1/6)*prod(extents) {want}"
            );
        }
    }

    /// oracle: 同 seed の debris は bit 一致 (決定論)
    #[test]
    fn generate_debris_is_bit_identical_for_the_same_seed() {
        let cfg = DebrisConfig {
            seed: 123,
            ..Default::default()
        };
        let a = generate_debris(Vec3::ZERO, 1.0, &cfg);
        let b = generate_debris(Vec3::ZERO, 1.0, &cfg);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.center.to_array(), pb.center.to_array());
            assert_eq!(pa.radius.to_bits(), pb.radius.to_bits());
            assert!(pa
                .mesh
                .vertices
                .iter()
                .zip(pb.mesh.vertices.iter())
                .all(|(x, y)| x.position == y.position));
        }
    }
}
