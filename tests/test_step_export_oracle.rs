//! STEP (ISO 10303-21) export の独立読み戻し oracle
//!
//! `src/io/step.rs` の既存 test は「header 文字列が入っている」「`CARTESIAN_POINT`
//! の数が頂点数と一致」までで、**出力が STEP として成立しているか / 立体として
//! 正しいか** を見ていない (2026-09-27 棚卸し) 本 file は書き出した file を
//! 実装を通さずに読み戻し、ISO 10303-21 と AP203 の要件で突き合わせる:
//!
//! 1. 全ての `#N` 参照が実体に解決する (dangling reference が無い)
//! 2. AP203 の必須 root 群 (`APPLICATION_CONTEXT` / `PRODUCT_DEFINITION` /
//!    `SHAPE_DEFINITION_REPRESENTATION` / `MANIFOLD_SOLID_BREP` /
//!    `CLOSED_SHELL`) が揃っている — これが無い file は CAD が開けない
//! 3. `CLOSED_SHELL` の面を loop から復元した立体の符号付き体積が解析解と一致
//! 4. 面の境界エッジが全て 2 枚で共有される (閉じている)
//!
//! Author: Moroya Sakamoto

#![allow(clippy::cast_precision_loss)]

use alice_sdf::io::step::{export_step, StepConfig};
use alice_sdf::types::SdfNode;
use std::collections::HashMap;

/// `#id=NAME(args);` の 1 レコード
///
/// STEP は 1 つの id に複数の entity を束ねる complex instance
/// (`#9=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))`) を許すので、
/// name は集合で持つ
#[derive(Debug, Clone)]
struct Entity {
    names: Vec<String>,
    args: String,
}

impl Entity {
    fn has(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    fn primary(&self) -> &str {
        self.names.first().map_or("", String::as_str)
    }
}

/// DATA section を `#id → Entity` に読む最小 Part21 パーサ
fn parse_step(text: &str) -> HashMap<u64, Entity> {
    let data = text
        .split_once("DATA;")
        .expect("DATA section")
        .1
        .split_once("ENDSEC;")
        .expect("ENDSEC after DATA")
        .0;
    let mut out = HashMap::new();
    for record in data.split(';') {
        let record = record.trim();
        let Some(rest) = record.strip_prefix('#') else {
            continue;
        };
        let Some((id, body)) = rest.split_once('=') else {
            continue;
        };
        let Ok(id) = id.trim().parse::<u64>() else {
            continue;
        };
        let body = body.trim();
        let (names, args) = if body.starts_with('(') {
            // complex instance: 含まれる各 entity 名を全部拾う
            let inner = body
                .trim_start_matches('(')
                .trim_end()
                .trim_end_matches(')');
            let mut names = Vec::new();
            let mut acc = String::new();
            let mut depth = 0usize;
            for ch in inner.chars() {
                match ch {
                    '(' => {
                        if depth == 0 && !acc.trim().is_empty() {
                            names.push(acc.trim().to_uppercase());
                        }
                        if depth == 0 {
                            acc.clear();
                        }
                        depth += 1;
                    }
                    ')' => depth = depth.saturating_sub(1),
                    c if depth == 0 => acc.push(c),
                    _ => {}
                }
            }
            (names, inner.to_string())
        } else {
            match body.split_once('(') {
                Some((n, a)) => (
                    vec![n.trim().to_uppercase()],
                    a.trim_end().trim_end_matches(')').to_string(),
                ),
                None => (vec![body.to_uppercase()], String::new()),
            }
        };
        out.insert(id, Entity { names, args });
    }
    out
}

/// 引数文字列に現れる `#N` 参照
fn refs(args: &str) -> Vec<u64> {
    let mut out = Vec::new();
    let bytes = args.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > start {
                if let Ok(n) = args[start..end].parse::<u64>() {
                    out.push(n);
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

fn write_sphere_step(r: f32, resolution: u32) -> (String, std::path::PathBuf) {
    let dir = std::env::temp_dir().join("alice-sdf-step-oracle");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("sphere_{resolution}.step"));
    let cfg = StepConfig {
        bounds: (-(r + 1.0), r + 1.0),
        resolution,
        name: "OracleSphere".into(),
    };
    export_step(&path, &SdfNode::sphere(r), &cfg).expect("export step");
    let text = std::fs::read_to_string(&path).expect("read step");
    (text, path)
}

#[test]
fn step_file_has_no_dangling_references() {
    let (text, path) = write_sphere_step(10.0, 24);
    let entities = parse_step(&text);
    assert!(!entities.is_empty(), "DATA section にレコードが無い");

    let mut dangling: Vec<String> = Vec::new();
    for (id, e) in &entities {
        for r in refs(&e.args) {
            if !entities.contains_key(&r) {
                dangling.push(format!("#{id}={} が #{r} を参照 (未定義)", e.primary()));
            }
        }
    }
    dangling.sort();
    dangling.dedup();
    assert!(
        dangling.is_empty(),
        "解決できない参照 {} 件:\n{}",
        dangling.len(),
        dangling
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    std::fs::remove_file(path).ok();
}

#[test]
fn step_declares_the_ap203_required_roots() {
    let (text, path) = write_sphere_step(10.0, 24);
    let entities = parse_step(&text);
    let have: std::collections::HashSet<&str> = entities
        .values()
        .flat_map(|e| e.names.iter().map(String::as_str))
        .collect();

    // AP203 の shape representation を成立させる最小集合
    for required in [
        "APPLICATION_CONTEXT",
        "APPLICATION_PROTOCOL_DEFINITION",
        "PRODUCT",
        "PRODUCT_DEFINITION",
        "PRODUCT_DEFINITION_SHAPE",
        "SHAPE_DEFINITION_REPRESENTATION",
        "MANIFOLD_SOLID_BREP",
        "CLOSED_SHELL",
        "ADVANCED_FACE",
        "PLANE",
        "VERTEX_POINT",
        "EDGE_CURVE",
        "ORIENTED_EDGE",
        "EDGE_LOOP",
        "LENGTH_UNIT",
    ] {
        assert!(
            have.contains(required),
            "AP203 必須 entity {required} が出力に無い (CAD が開けない)"
        );
    }
    std::fs::remove_file(path).ok();
}

/// `CLOSED_SHELL` の面を辿って三角形を復元する
fn shell_triangles(entities: &HashMap<u64, Entity>) -> Vec<[[f64; 3]; 3]> {
    let point_of = |id: u64| -> [f64; 3] {
        let e = entities.get(&id).expect("entity");
        let inner = match e.primary() {
            "VERTEX_POINT" => {
                let p = refs(&e.args)[0];
                entities.get(&p).expect("cartesian point")
            }
            "CARTESIAN_POINT" => e,
            other => panic!("点として辿れない entity {other}"),
        };
        let coords = inner
            .args
            .rsplit_once('(')
            .expect("coord tuple")
            .1
            .trim_end_matches(')');
        let v: Vec<f64> = coords
            .split(',')
            .map(|s| s.trim().parse().expect("coord"))
            .collect();
        [v[0], v[1], v[2]]
    };

    let shell = entities
        .values()
        .find(|e| e.has("CLOSED_SHELL"))
        .expect("CLOSED_SHELL");
    let mut tris = Vec::new();
    for face_id in refs(&shell.args) {
        let face = entities.get(&face_id).expect("face");
        assert!(
            face.has("ADVANCED_FACE"),
            "shell に face 以外が入っている ({})",
            face.primary()
        );
        // ADVANCED_FACE('', (#bound), #surface, .T.)
        let mut loop_points: Vec<[f64; 3]> = Vec::new();
        for bound_id in refs(&face.args) {
            let bound = entities.get(&bound_id).expect("bound");
            if !bound.has("FACE_OUTER_BOUND") && !bound.has("FACE_BOUND") {
                continue; // surface 参照
            }
            let loop_id = refs(&bound.args)[0];
            let edge_loop = entities.get(&loop_id).expect("edge loop");
            assert!(edge_loop.has("EDGE_LOOP"), "{}", edge_loop.primary());
            for oe_id in refs(&edge_loop.args) {
                let oe = entities.get(&oe_id).expect("oriented edge");
                assert!(oe.has("ORIENTED_EDGE"), "{}", oe.primary());
                let forward = oe.args.contains(".T.");
                let edge_id = *refs(&oe.args).last().expect("edge curve ref");
                let edge = entities.get(&edge_id).expect("edge curve");
                assert!(edge.has("EDGE_CURVE"), "{}", edge.primary());
                let ends = refs(&edge.args);
                let (a, b) = (ends[0], ends[1]);
                let start = if forward { a } else { b };
                loop_points.push(point_of(start));
            }
        }
        assert!(
            loop_points.len() >= 3,
            "face の loop が {} 点しかない",
            loop_points.len()
        );
        // 凸な facet を fan 分割
        for i in 1..loop_points.len() - 1 {
            tris.push([loop_points[0], loop_points[i], loop_points[i + 1]]);
        }
    }
    tris
}

#[test]
fn step_solid_volume_matches_the_analytic_sphere() {
    let r = 10.0_f64;
    let (text, path) = write_sphere_step(r as f32, 48);
    let entities = parse_step(&text);
    let tris = shell_triangles(&entities);
    assert!(!tris.is_empty(), "shell に面が無い");

    let volume: f64 = tris
        .iter()
        .map(|[a, b, c]| {
            let cross = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ];
            (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0
        })
        .sum();
    // oracle: 4/3 π r³
    let want = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
    assert!(
        volume > 0.0,
        "符号付き体積が負 = 面の向きが内向き ({volume})"
    );
    let rel = (volume - want).abs() / want;
    assert!(
        rel < 0.05,
        "STEP から復元した体積 {volume:.2}mm³ vs 4/3πr³ {want:.2}mm³ (rel {rel:.4})"
    );
    std::fs::remove_file(path).ok();
}

#[test]
fn step_shell_is_closed() {
    let (text, path) = write_sphere_step(10.0, 32);
    let entities = parse_step(&text);

    // EDGE_CURVE の端点 (VERTEX_POINT id) でエッジを同定し、ORIENTED_EDGE の
    // 参照回数を数える 閉じた shell では各エッジがちょうど 2 回使われる
    let mut used: HashMap<(u64, u64), usize> = HashMap::new();
    for e in entities.values() {
        if !e.has("ORIENTED_EDGE") {
            continue;
        }
        let edge_id = *refs(&e.args).last().expect("edge ref");
        let edge = entities.get(&edge_id).expect("edge curve");
        let ends = refs(&edge.args);
        let (a, b) = (ends[0], ends[1]);
        *used.entry((a.min(b), a.max(b))).or_insert(0) += 1;
    }
    assert!(!used.is_empty(), "ORIENTED_EDGE が無い");
    let open: Vec<_> = used.iter().filter(|(_, &n)| n != 2).collect();
    assert!(
        open.is_empty(),
        "2 回使われていないエッジ {} 本 (先頭: {:?})",
        open.len(),
        open.iter().take(3).collect::<Vec<_>>()
    );
    std::fs::remove_file(path).ok();
}

#[test]
fn box_is_exported_as_six_exact_planar_faces() {
    // 箱の面は厳密に平面なので、tessellate せず 6 面で出るのが正
    // (voxel 解像度に寄らず寸法が厳密に一致する = 三相原理 Phase 2)
    let dir = std::env::temp_dir().join("alice-sdf-step-oracle");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("box.step");
    let cfg = StepConfig {
        bounds: (-30.0, 30.0),
        resolution: 16,
        name: "OracleBox".into(),
    };
    export_step(&path, &SdfNode::box3d(20.0, 10.0, 6.0), &cfg).expect("export step");
    let text = std::fs::read_to_string(&path).expect("read step");
    let entities = parse_step(&text);

    let faces = entities.values().filter(|e| e.has("ADVANCED_FACE")).count();
    assert_eq!(faces, 6, "箱が {faces} 面で出た (6 面で厳密に書けるはず)");
    let points = entities.values().filter(|e| e.has("VERTEX_POINT")).count();
    assert_eq!(points, 8, "箱の位相頂点が {points} 個 (8 個のはず)");

    // 参照は全て解決する
    for (id, e) in &entities {
        for r in refs(&e.args) {
            assert!(
                entities.contains_key(&r),
                "#{id}={} が未定義の #{r} を参照",
                e.primary()
            );
        }
    }

    // oracle: 体積 = 20 × 10 × 6 (厳密、tessellation 誤差なし)
    let tris = shell_triangles(&entities);
    assert_eq!(tris.len(), 12, "4 角形 6 枚 = 三角形 12 枚のはず");
    let volume: f64 = tris
        .iter()
        .map(|[a, b, c]| {
            let cross = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ];
            (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0
        })
        .sum();
    let want = 20.0 * 10.0 * 6.0;
    let rel = (volume - want).abs() / want;
    assert!(
        rel < 1e-6,
        "箱の体積 {volume} vs w·h·d {want} (rel {rel:.3e})"
    );
    std::fs::remove_file(path).ok();
}
