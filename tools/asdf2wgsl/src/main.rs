//! asdf2wgsl-cf <in.asdf.json> <out.wgsl> <label>: the viewer export of an .asdf.json tree through the same
//! ALICE-SDF transpiler path as the joint assets (runs/compare-cf/tool-integ `wgsl`: transpile_with_rest_point,
//! else transpile_with_node_data; node data renamed to the viewer's `alice_data`, binding 3), plus the page's
//! parity samples (4096 points: alternately anywhere in the body box and within 2 cm of the surface, CPU values)
use alice_sdf::compiled::{TranspileMode, WgslShader};
use alice_sdf::prelude::*;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() == 4, "usage: asdf2wgsl-cf <in.asdf.json> <out.wgsl> <label>");
    let tree = alice_sdf::io::load_asdf_json(&a[1]).expect("load");
    let node = tree.root;
    let t = std::time::Instant::now();
    let sh = match WgslShader::transpile_with_rest_point(&node, TranspileMode::Hardcoded) {
        Ok(sh) => sh,
        Err(e) => {
            eprintln!("no sdf_eval_rest: {e}");
            WgslShader::transpile_with_node_data(&node, TranspileMode::Hardcoded)
        }
    };
    let mut mat = WgslShader::transpile_material(&node, TranspileMode::Hardcoded);
    if mat.is_empty() {
        mat = "fn sdf_eval_material(p: vec3<f32>) -> f32 { return 0.0; }\n".to_string();
    }
    // INTERIM post-pass (fd ruling B, 2026-11-05): the transpiler leaves loft section / knot arrays as WGSL
    // `const` arrays (some past WGSL's 32,767-element constructor limit); move every `const NAME: array<f32, N>`
    // into the node-data buffer after the transpiler's own node data and rewrite `NAME[` as `alice_data[OFF + `.
    // The root fix belongs in ALICE-SDF (transpile_with_node_data putting them in node data)
    let body = format!("{}\n{}", sh.source, mat).replace("alice_node_data", "alice_data");
    // the transpiler's output before the post-pass, for the static checker (check_postpass.py)
    std::fs::write(format!("{}.raw.wgsl", a[2]), &body).expect("write raw");
    let b0: Vec<u8> = sh.node_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{}.raw.nodedata.bin", a[2]), b0).expect("write raw node data");
    let mut data = sh.node_data.clone();
    let (body, moved) = move_arrays_to_buffer(&body, &mut data);
    let decl = if data.is_empty() { String::new() } else { "@group(0) @binding(3) var<storage, read> alice_data: array<f32>;\n".to_string() };
    let src = format!("// {}\n{}{}", a[3], decl, body);
    std::fs::write(&a[2], &src).expect("write");
    if !data.is_empty() {
        let b: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{}.data.bin", a[2]), b).expect("write data");
    }
    eprintln!("post-pass: moved {} const arrays ({} floats) into alice_data after {} transpiler node-data floats", moved.0, moved.1, sh.node_data.len());
    let left = src.lines().filter(|l| l.trim_start().starts_with("const ") && l.contains(": array<f32,")).count();
    assert!(left == 0, "const f32 arrays left in the shader: {left}");
    eprintln!("wgsl bytes {} lines {} node data {} floats rest point {} ({:.1} s)", src.len(), src.lines().count(), sh.node_data.len(), src.contains("fn sdf_eval_rest("), t.elapsed().as_secs_f64());
    let mut rnd = lcg(7);
    let mut pts: Vec<Vec3> = Vec::new();
    let mut tries = 0usize;
    while pts.len() < 4096 && tries < 2_000_000 {
        tries += 1;
        let p = Vec3::new(rnd() * 1.8 - 0.9, rnd() * 1.9 - 0.1, rnd() * 1.0 - 0.5);
        if pts.len() % 2 == 0 || eval(&node, p).abs() < 0.02 {
            pts.push(p);
        }
    }
    let s: Vec<String> = pts.iter().map(|p| format!("[{},{},{},{}]", p.x, p.y, p.z, eval(&node, *p))).collect();
    std::fs::write(format!("{}.samples.json", a[2]), format!("[{}]", s.join(","))).expect("write samples");
    eprintln!("samples {}", pts.len());
}

/// Moves every `const NAME: array<f32, N> = array<f32, N>(…);` line into `data` (appended) and rewrites each
/// whole-identifier `NAME[` as `alice_data[OFFSET + ` (the index expression and its closing bracket stay).
/// The same pass as viewer/tools/sdf2wgsl (move_arrays_to_buffer), with the offset after existing node data
fn move_arrays_to_buffer(src: &str, data: &mut Vec<f32>) -> (String, (usize, usize)) {
    let mut offsets: Vec<(String, usize)> = Vec::new();
    let mut kept = String::with_capacity(src.len());
    let start = data.len();
    for line in src.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("const ") {
            if let Some(colon) = rest.find(": array<f32, ") {
                let name = rest[..colon].trim().to_string();
                let open = rest.find(">(").expect("array constructor") + 2;
                let close = rest.rfind(");").expect("array end");
                let off = data.len();
                for v in rest[open..close].split(',') {
                    data.push(v.trim().trim_end_matches('f').parse::<f32>().unwrap_or_else(|e| panic!("f32 literal {v:?}: {e}")));
                }
                offsets.push((name, off));
                continue;
            }
        }
        kept.push_str(line);
        kept.push('\n');
    }
    let map: std::collections::HashMap<&str, usize> = offsets.iter().map(|(n, o)| (n.as_str(), *o)).collect();
    let bytes = kept.as_bytes();
    let mut out = String::with_capacity(kept.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            let s0 = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &kept[s0..i];
            if i < bytes.len() && bytes[i] == b'[' {
                if let Some(off) = map.get(word) {
                    out.push_str(&format!("alice_data[{off} + "));
                    i += 1;
                    continue;
                }
            }
            out.push_str(word);
        } else {
            out.push(c as char);
            i += 1;
        }
    }
    (out, (offsets.len(), data.len() - start))
}
