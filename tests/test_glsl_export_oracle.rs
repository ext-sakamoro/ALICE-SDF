//! Oracles for the GLSL outputs other than the bare `sdf_eval` library
//! (whose values `tests/test_gpu_law_parity.rs` already checks on the GPU):
//!
//! | output | oracle |
//! |---|---|
//! | `GlslShader::to_compute_shader` (Hardcoded / Dynamic) | run on the GPU through naga's GLSL front end; every output equals the closed form of the scene |
//! | `GlslShader::extract_params` | the literals of the tree, bit-exact, in the layout `transpile(.., Dynamic)` reads: A's Dynamic library fed B's parameters evaluates B on the GPU |
//! | `GlslShader::to_fragment_shader` | parses and validates with naga (Hardcoded and Dynamic) |
//! | `GlslShader::to_fragment_shader_full` / `RenderConfig` | parses and validates with naga for every feature flag on its own and all of them together; `max_steps` / `max_distance` reach the march loop |
//!
//! # The host-interface rewrite
//!
//! These shaders target OpenGL / OpenGL ES, whose loose `uniform` variables
//! (`uniform float uTime;`, `layout(location = 0) uniform uint point_count;`)
//! GL_KHR_vulkan_glsl — and therefore naga and wgpu — do not accept, and the
//! full pipeline is `#version 300 es`, a profile naga's GLSL front end does
//! not parse. `vulkanize` rewrites **only those declarations**: the version
//! line becomes `#version 450`, `precision` statements are dropped, the loose
//! uniforms of the global scope move into one uniform block (members keep
//! their names, so every use in the shader is unchanged), `out` variables and
//! buffer / uniform blocks get explicit `location` / `set` / `binding`
//! numbers. Every function, expression and statement the generator wrote is
//! parsed and validated as emitted; for the compute shader it also runs.
//!
//! # Tolerance
//!
//! `DIST_TOL = 1e-5` absolute, as in `tests/test_gpu_eval_api_oracle.rs`:
//! `|p − c| < 4`, one f32 ulp is ≤ 4.8e-7, and GPU `length` is not correctly
//! rounded; the reference is the closed form in f64.
//!
//! CI's gpu-parity job (lavapipe) sets `ALICE_SDF_REQUIRE_GPU=1`.
//!
//! Author: Moroya Sakamoto

#![cfg(all(feature = "glsl", feature = "gpu"))]

use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode, RenderConfig};
use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;

const DIST_TOL: f64 = 1e-5;

#[derive(Clone, Copy)]
struct Ball {
    r: f32,
    c: [f32; 3],
}

impl Ball {
    fn node(self) -> SdfNode {
        SdfNode::sphere(self.r).translate(self.c[0], self.c[1], self.c[2])
    }
    fn dist(self, p: Vec3) -> f64 {
        let q = [
            f64::from(p.x) - f64::from(self.c[0]),
            f64::from(p.y) - f64::from(self.c[1]),
            f64::from(p.z) - f64::from(self.c[2]),
        ];
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - f64::from(self.r)
    }
}

const A: Ball = Ball {
    r: 1.0,
    c: [0.25, -0.5, 0.125],
};
const B: Ball = Ball {
    r: 1.75,
    c: [-0.5, 0.25, 0.0],
};

fn points(n: usize) -> Vec<Vec3> {
    let mut s: u32 = 0x0bad_5eed;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
    };
    (0..n).map(|_| Vec3::new(next(), next(), next())).collect()
}

fn check_dist(what: &str, ball: Ball, pts: &[Vec3], got: &[f32]) -> usize {
    assert_eq!(got.len(), pts.len(), "{what}: result count");
    assert!(!pts.is_empty(), "{what}: 0 points compared");
    for (&p, &g) in pts.iter().zip(got) {
        let err = (f64::from(g) - ball.dist(p)).abs();
        assert!(
            err <= DIST_TOL,
            "{what}: at {p:?} gpu {g}, closed form {} (err {err:e})",
            ball.dist(p)
        );
    }
    pts.len()
}

/// See the module doc. Rewrites declarations only.
fn vulkanize(src: &str) -> String {
    let mut out = String::new();
    let mut loose = Vec::new();
    let mut block_at = None;
    let mut binding = 0u32;
    for (i, line) in src.lines().enumerate() {
        let t = line.trim();
        if i == 0 && t.starts_with("#version") {
            out.push_str("#version 450\n");
            continue;
        }
        if t.starts_with("precision ") {
            continue;
        }
        let is_loose_uniform = (t.starts_with("uniform ") || t.starts_with("layout(location"))
            && t.contains("uniform ")
            && !t.contains('{')
            && t.ends_with(';');
        if is_loose_uniform {
            let decl = &t[t.find("uniform ").unwrap() + "uniform ".len()..];
            loose.push(decl.to_string());
            if block_at.is_none() {
                block_at = Some(out.len());
            }
            continue;
        }
        // naga does not support write-only storage buffers (OpenGL does);
        // dropping the access qualifier changes no value the shader computes.
        let line = &line.replace("writeonly buffer", "buffer");
        let t = line.trim();
        if t.starts_with("out ") {
            out.push_str("layout(location = 0) ");
            out.push_str(t);
            out.push('\n');
            continue;
        }
        if let Some(rest) = t.strip_prefix("layout(") {
            if rest.contains("buffer ") || rest.contains("uniform ") {
                // Re-number every block: binding 0.. in declaration order,
                // set 0. The loose-uniform block takes the next free slot.
                let close = rest.find(')').unwrap();
                let quals: Vec<&str> = rest[..close]
                    .split(',')
                    .map(str::trim)
                    .filter(|q| !q.starts_with("binding") && !q.starts_with("set"))
                    .collect();
                out.push_str(&format!(
                    "layout({}, set = 0, binding = {binding}){}\n",
                    quals.join(", "),
                    &rest[close + 1..]
                ));
                binding += 1;
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    if let Some(at) = block_at {
        let members: String = loose.iter().fold(String::new(), |mut acc, d| {
            acc.push_str("    ");
            acc.push_str(d);
            acc.push('\n');
            acc
        });
        out.insert_str(
            at,
            &format!("layout(std140, set = 0, binding = {binding}) uniform HostUniforms {{\n{members}}};\n"),
        );
    }
    out
}

fn validate(stage: naga::ShaderStage, what: &str, src: &str) {
    let mut frontend = naga::front::glsl::Frontend::default();
    let module = frontend
        .parse(&naga::front::glsl::Options::from(stage), src)
        .unwrap_or_else(|e| {
            let near: Vec<String> = e
                .errors
                .iter()
                .take(4)
                .map(|err| {
                    let s = err.meta.to_range().map_or(0, |r| r.start);
                    let line = src[..s.min(src.len())].lines().count();
                    format!(
                        "{:?} at line {line}: {}",
                        err.kind,
                        src.lines().nth(line.saturating_sub(1)).unwrap_or("")
                    )
                })
                .collect();
            panic!("{what}: naga GLSL parse failed:\n{}", near.join("\n"))
        });
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{what}: naga validation failed: {e:?}"));
}

fn gpu_or_skip(what: &str, src: &str) -> Option<GpuEvaluator> {
    match GpuEvaluator::from_glsl_compute(src) {
        Ok(g) => Some(g),
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "{what}: ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            eprintln!("{what}: skipped ({e})");
            None
        }
    }
}

#[test]
fn compute_shader_runs_on_the_gpu_and_matches_the_closed_form() {
    let pts = points(1000);
    let hard = GlslShader::transpile(&A.node(), GlslTranspileMode::Hardcoded).to_compute_shader();
    assert!(hard.starts_with("#version 450\n"));
    let hard = vulkanize(&hard);
    validate(naga::ShaderStage::Compute, "compute (Hardcoded)", &hard);
    if let Some(gpu) = gpu_or_skip("compute (Hardcoded)", &hard) {
        check_dist(
            "compute (Hardcoded)",
            A,
            &pts,
            &gpu.eval_batch(&pts).unwrap(),
        );
    }
}

#[test]
fn extract_params_is_the_dynamic_layout_of_the_tree() {
    let la = GlslShader::extract_params(&A.node());
    let lb = GlslShader::extract_params(&B.node());
    let dyn_a = GlslShader::transpile(&A.node(), GlslTranspileMode::Dynamic);
    assert_eq!(la, dyn_a.param_layout);
    assert_eq!(la.len(), lb.len());
    for (layout, ball) in [(&la, A), (&lb, B)] {
        let mut got = layout.clone();
        let mut want = vec![ball.r, ball.c[0], ball.c[1], ball.c[2]];
        got.sort_by(f32::total_cmp);
        want.sort_by(f32::total_cmp);
        assert_eq!(got, want, "the four literals of the tree, bit-exact");
    }
    assert!(
        GlslShader::transpile(&A.node(), GlslTranspileMode::Hardcoded)
            .param_layout
            .is_empty()
    );

    // A's Dynamic compute shader reads `params[i].c` from the SdfParams block.
    // Fed B's layout through that block, it must evaluate B.
    let raw = dyn_a.to_compute_shader();
    validate(
        naga::ShaderStage::Compute,
        "compute (Dynamic)",
        &vulkanize(&raw),
    );
    // GpuEvaluator binds three resources, so the parameters go in as a
    // constant array in place of the uniform block: the addressing
    // `params[i / 4].{x,y,z,w}` is what is under test.
    let mut padded = lb;
    while padded.len() % 4 != 0 {
        padded.push(0.0);
    }
    let consts: Vec<String> = padded
        .chunks_exact(4)
        .map(|c| format!("vec4({:e}, {:e}, {:e}, {:e})", c[0], c[1], c[2], c[3]))
        .collect();
    let start = raw
        .find("layout(std140, binding = 2) uniform SdfParams")
        .expect("params block");
    let end = start + raw[start..].find("};").unwrap() + 2;
    let src = vulkanize(&format!(
        "{}const vec4 params[{}] = vec4[]({});{}",
        &raw[..start],
        consts.len(),
        consts.join(", "),
        &raw[end..]
    ));
    let pts = points(1000);
    if let Some(gpu) = gpu_or_skip("compute (Dynamic, B's params)", &src) {
        check_dist(
            "Dynamic A fed B's params",
            B,
            &pts,
            &gpu.eval_batch(&pts).unwrap(),
        );
    }
}

#[test]
fn fragment_shader_parses_and_validates() {
    let node = A.node().smooth_union(SdfNode::box3d(0.5, 0.4, 0.3), 0.2);
    for mode in [GlslTranspileMode::Hardcoded, GlslTranspileMode::Dynamic] {
        let frag = GlslShader::transpile(&node, mode).to_fragment_shader();
        assert!(frag.contains("void main()") && frag.contains("sdf_eval(p)"));
        validate(
            naga::ShaderStage::Fragment,
            &format!("fragment {mode:?}"),
            &vulkanize(&frag),
        );
    }
}

#[test]
fn full_pipeline_parses_and_validates_for_every_flag() {
    let node = A.node().smooth_union(SdfNode::box3d(0.5, 0.4, 0.3), 0.2);
    let shader = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
    let all_off = RenderConfig {
        day_night_cycle: false,
        weather_system: false,
        ssr_enabled: false,
        volumetric_light: false,
        post_process: false,
        ..RenderConfig::default()
    };
    type Flag = (&'static str, fn(&mut RenderConfig));
    let flags: [Flag; 11] = [
        ("day_night_cycle", |c| c.day_night_cycle = true),
        ("weather_system", |c| c.weather_system = true),
        ("ssr_enabled", |c| c.ssr_enabled = true),
        ("volumetric_light", |c| c.volumetric_light = true),
        ("post_process", |c| c.post_process = true),
        ("biome_terrain", |c| c.biome_terrain = true),
        ("spectral_rendering", |c| c.spectral_rendering = true),
        ("destruction", |c| c.destruction = true),
        ("vfx_effects", |c| c.vfx_effects = true),
        ("interior_mapping", |c| c.interior_mapping = true),
        ("micro_normal", |c| c.micro_normal = true),
    ];
    let mut configs: Vec<(String, RenderConfig)> = vec![
        ("default".into(), RenderConfig::default()),
        ("all off".into(), all_off.clone()),
    ];
    for (name, set) in flags {
        let mut c = all_off.clone();
        set(&mut c);
        configs.push((name.into(), c));
    }
    let mut every = all_off;
    for (_, set) in flags {
        set(&mut every);
    }
    configs.push(("every flag".into(), every));

    for (name, cfg) in &configs {
        let full = shader.to_fragment_shader_full(cfg);
        assert!(
            full.starts_with(&format!("#version {} es", cfg.glsl_version)),
            "{name}"
        );
        validate(
            naga::ShaderStage::Fragment,
            &format!("full pipeline [{name}]"),
            &vulkanize(&full),
        );
    }

    // `dual_sdf` reads a user-supplied `sdf_eval_lite`.
    let lite = format!(
        "{}\nfloat sdf_eval_lite(vec3 p) {{ return length(p) - 1.0; }}\n",
        shader.source
    );
    let cfg = RenderConfig {
        dual_sdf: true,
        ..RenderConfig::default()
    };
    let full = alice_sdf::compiled::glsl::render_pipeline::build_full_shader(&lite, &cfg);
    validate(
        naga::ShaderStage::Fragment,
        "full pipeline [dual_sdf]",
        &vulkanize(&full),
    );

    // The march loop uses the configured step count and distance.
    let cfg = RenderConfig {
        max_steps: 77,
        max_distance: 33.5,
        ..RenderConfig::default()
    };
    let full = shader.to_fragment_shader_full(&cfg);
    assert!(
        full.contains("for(int i=0;i<77;i++)"),
        "max_steps reaches the loop"
    );
    assert!(full.contains("33.5"), "max_distance reaches the shader");
}
