//! `validity` の解析解突合 test (CLAUDE.md § 解析解突合テスト規律)
//!
//! 期待値は閉形式 — 板の厚さ、斜面の角度、球殻の肉厚 — から手で決めたもので、
//! 被検査関数を呼んで作っていない。
//!
//! **単位の注意**: `SdfNode::box3d(w, h, d)` は **full size** で、half-extent
//! ではない (`box3d(20, 2, 20)` の厚さは 2.0)。half-extent を持っているときは
//! `box3d_half_extents` を使う。
//!
//! overhang は `asin(-n · b)` の閉形式なので、法線が既知になる三角形を手で組んで
//! 角度を直接突き合わせる。

use alice_sdf::io::step::StepConfig;
use alice_sdf::mesh::{Mesh, Vertex};
use alice_sdf::types::SdfNode;
use alice_sdf::validity::{
    export_step_validated, local_thickness, overhang_stats, prove_erosion, validate_for_printing,
    ErosionVerdict, PrintRequirements, ValidatedExportError,
};
use glam::Vec3;

/// STEP を書く test は **file 名を test ごとに固有**にする (同 file を 2 test が
/// 並列に書くと内容が混ざる — 2026-09-27 に `test_step_export_oracle` で踏んだ)
fn step_path(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("alice-sdf-validity-oracle");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{tag}.step"));
    let _ = std::fs::remove_file(&path);
    path
}

/// 頂点 3 個の mesh (法線は `cross(b − a, c − a)` で決まる)
fn tri(a: Vec3, b: Vec3, c: Vec3) -> Mesh {
    let v = |p: Vec3| Vertex {
        position: p,
        ..Default::default()
    };
    Mesh {
        vertices: vec![v(a), v(b), v(c)],
        indices: vec![0, 1, 2],
    }
}

// ───────────────────────── overhang (閉形式) ──────────────────────────────

#[test]
fn a_vertical_wall_has_zero_overhang() {
    // oracle: 法線が +X (build 方向 +Z に垂直) なので asin(0) = 0
    // e1 = (0,1,0), e2 = (0,0,1) → cross = (1,0,0)
    let mesh = tri(
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    let (max, over) = overhang_stats(&mesh, Vec3::Z, std::f32::consts::FRAC_PI_4);
    assert!(max.abs() < 1e-5, "垂直壁の overhang は 0 のはずが {max}");
    assert_eq!(over, 0);
}

#[test]
fn an_upward_facing_triangle_reports_zero_not_a_negative_angle() {
    // oracle: 法線が +Z (build 方向と同じ) なので overhang は無い = 0
    // e1 = (1,0,0), e2 = (0,1,0) → cross = (0,0,1)
    let mesh = tri(
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    let (max, over) = overhang_stats(&mesh, Vec3::Z, std::f32::consts::FRAC_PI_4);
    assert!(
        max.abs() < 1e-5,
        "上向き面は負の角でなく 0 を返すべきが {max}"
    );
    assert_eq!(over, 0);
}

#[test]
fn a_downward_facing_horizontal_triangle_has_a_right_angle_overhang() {
    // oracle: 法線が −Z なので asin(1) = π/2
    // 上の三角形の巻き順を逆にすると cross = (0,0,−1)
    let mesh = tri(
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    );
    let (max, over) = overhang_stats(&mesh, Vec3::Z, std::f32::consts::FRAC_PI_4);
    assert!(
        (max - std::f32::consts::FRAC_PI_2).abs() < 1e-4,
        "水平な下向き面は π/2 のはずが {max}"
    );
    assert_eq!(over, 1, "45° 閾値を超えるので 1 枚が違反");
}

#[test]
fn a_45_degree_downward_face_matches_the_closed_form() {
    // oracle: 法線 (0, 1, −1)/√2 に対し −n·Z = 1/√2 → asin(1/√2) = π/4
    // e1 = (0,1,1), e2 = (1,0,0) → cross = (0,1,−1)
    let mesh = tri(
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
    );
    let (max, over) = overhang_stats(&mesh, Vec3::Z, std::f32::consts::FRAC_PI_4);
    assert!(
        (max - std::f32::consts::FRAC_PI_4).abs() < 1e-4,
        "45° 斜面は π/4 のはずが {max}"
    );
    // ちょうど閾値なので「超えて」はいない
    assert_eq!(over, 0, "閾値ちょうどは違反にしない");
}

// ───────────────────────── local thickness (SDF 実測) ─────────────────────

#[test]
fn local_thickness_of_a_plate_is_its_thickness() {
    // oracle: box3d は full size なので厚さ 2.0、y ∈ [−1, 1]
    // 上面 (0, 1, 0) から内向き (0, −1, 0) に抜けるまでの距離は 2.0
    let plate = SdfNode::box3d(20.0, 2.0, 20.0);
    let t = local_thickness(&plate, Vec3::new(0.0, 1.0, 0.0), Vec3::NEG_Y, 10.0)
        .expect("板の内部を通るので測れる");
    assert!((t - 2.0).abs() < 1e-2, "厚さ 2.0 のはずが {t}");
}

#[test]
fn local_thickness_of_a_spherical_shell_is_the_wall() {
    // oracle: 半径 2 の球から半径 1.5 を抜いた殻の肉厚は 0.5
    // 外表面 (0, 0, 2) から中心向き (0, 0, −1) に進むと 0.5 で内壁に出る
    let shell = SdfNode::sphere(2.0).subtract(SdfNode::sphere(1.5));
    let t = local_thickness(&shell, Vec3::new(0.0, 0.0, 2.0), Vec3::NEG_Z, 10.0)
        .expect("殻の肉を通るので測れる");
    assert!((t - 0.5).abs() < 1e-2, "肉厚 0.5 のはずが {t}");
}

#[test]
fn local_thickness_is_none_when_the_ray_never_enters_the_solid() {
    // oracle: 球の外を掠める向きに撃てば内部に入らない
    let ball = SdfNode::sphere(1.0);
    assert!(local_thickness(&ball, Vec3::new(5.0, 0.0, 0.0), Vec3::Z, 10.0).is_none());
}

// ───────────────────────── erosion proof (interval、三値) ─────────────────

#[test]
fn a_plate_thicker_than_the_requirement_has_a_proven_thick_region() {
    // oracle: 厚さ 2.0 の板を min_wall 1.0 (= erode 0.5) で削ると厚さ 1.0 が残る
    let plate = SdfNode::box3d(20.0, 2.0, 20.0);
    let verdict = prove_erosion(
        &plate,
        1.0,
        Vec3::new(-11.0, -2.0, -11.0),
        Vec3::new(11.0, 2.0, 11.0),
        6,
    );
    assert!(
        verdict.has_thick_region(),
        "厚さ 2.0 は min_wall 1.0 を満たすので witness が出るべきが {verdict:?}"
    );
}

#[test]
fn a_plate_thinner_than_the_requirement_is_proven_entirely_too_thin() {
    // oracle: 厚さ 0.5 の板を min_wall 1.0 (= erode 0.5) で削ると何も残らない
    // (厚さ 0.5 の中心面は表面から 0.25 しかないので erode 0.5 で消える)
    let sheet = SdfNode::box3d(20.0, 0.5, 20.0);
    let verdict = prove_erosion(
        &sheet,
        1.0,
        Vec3::new(-11.0, -2.0, -11.0),
        Vec3::new(11.0, 2.0, 11.0),
        6,
    );
    assert_eq!(
        verdict,
        ErosionVerdict::EntirelyTooThin,
        "厚さ 0.5 は min_wall 1.0 を満たせないので全体が薄いと証明されるべき"
    );
}

// ───────────────────────── 統合 (report) ─────────────────────────────────

#[test]
fn an_undecided_erosion_verdict_is_not_printable() {
    // oracle: 深さ 0 では箱が表面を跨いだまま決まらない = Undecided
    // そして Undecided は合格にしてはいけない (「決められなかった」≠「問題ない」)
    let plate = SdfNode::box3d(20.0, 2.0, 20.0);
    let verdict = prove_erosion(
        &plate,
        1.0,
        Vec3::new(-11.0, -2.0, -11.0),
        Vec3::new(11.0, 2.0, 11.0),
        0,
    );
    assert_eq!(verdict, ErosionVerdict::Undecided);

    let mesh = tri(
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    let req = PrintRequirements {
        erosion_depth: 0,
        ..PrintRequirements::fdm_0_4_nozzle()
    };
    let report = validate_for_printing(
        &plate,
        &mesh,
        Vec3::new(-11.0, -2.0, -11.0),
        Vec3::new(11.0, 2.0, 11.0),
        req,
    );
    assert!(
        !report.is_printable(),
        "Undecided を合格にしてはいけない: {report:?}"
    );
}

// ───────────────────────── export path (必ず通す) ────────────────────────

#[test]
fn export_step_validated_refuses_a_sheet_thinner_than_the_requirement() {
    // oracle: 厚さ 0.5 の板は FDM 既定の min_wall 0.8 を満たさないので
    // **file は書かれず** NotPrintable が返る (これが「必ず通す」の実証)
    let sheet = SdfNode::box3d(20.0, 0.5, 20.0);
    let path = step_path("refuse_thin_sheet");
    let cfg = StepConfig {
        bounds: (-11.0, 11.0),
        resolution: 24,
        name: "TooThin".into(),
    };
    let err = export_step_validated(&path, &sheet, &cfg, PrintRequirements::fdm_0_4_nozzle())
        .expect_err("厚さ 0.5 は min_wall 0.8 を満たさない");
    assert!(
        matches!(err, ValidatedExportError::NotPrintable(_)),
        "NotPrintable を期待したが {err}"
    );
    assert!(
        !path.exists(),
        "検証を通らない形状の file を書いてはいけない"
    );
}

#[test]
fn export_step_validated_writes_a_part_that_passes() {
    // oracle: 厚さ 4.0 の箱は min_wall 0.8 を満たす
    // overhang は接地判定を持たないので箱の水平下面が π/2 になる (module doc 参照)
    // → 閾値を π/2 にして overhang 検査を実質無効化した上で、厚さ側が通ることを見る
    let block = SdfNode::box3d(8.0, 4.0, 8.0);
    let path = step_path("accept_thick_block");
    let cfg = StepConfig {
        bounds: (-6.0, 6.0),
        resolution: 24,
        name: "ThickBlock".into(),
    };
    let req = PrintRequirements {
        max_overhang: std::f32::consts::FRAC_PI_2,
        ..PrintRequirements::fdm_0_4_nozzle()
    };
    match export_step_validated(&path, &block, &cfg, req) {
        Ok(report) => {
            assert!(report.is_printable());
            assert!(path.exists(), "通った形状は書かれるべき");
            let _ = std::fs::remove_file(&path);
        }
        Err(e) => panic!("厚さ 4.0 の箱が通らない: {e}"),
    }
}
