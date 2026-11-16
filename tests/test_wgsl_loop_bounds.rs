//! Every shader module this crate creates has bounded loops (`src/gpu_loop_bound.rs`): the census.
//!
//! The census builds every source kind that reaches a module-creation point, over the shared corpus of SDF trees
//! (`tests/common/corpus.rs`, every node kind) where the kind depends on a tree, and checks each with the same rule
//! the creation points apply (the loop check and the per-invocation ceiling). No GPU is used: the modules are
//! parsed with naga only. Zero modules checked, or a source kind with no module, is a failure.
//!
//! Text a caller supplies (`GpuEvaluator::from_wgsl` / `from_glsl_compute`) cannot be enumerated: it is checked
//! where it is created, before any device is requested, which is the guard itself. Trees a caller builds are
//! likewise checked at creation; the corpus here covers each node kind, not every tree shape (a union's operand
//! count multiplies the leaf loop's work).
//!
//! `every_module_creation_site_is_registered` keeps the census complete by construction: it counts the call
//! sites that create a module in `src/` and `tests/` and fails on one that is not in [`SITES`] (and on 0 found).
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::compiled::WgslShader;
use alice_sdf::gpu_loop_bound::{check_module, parse_source, HostMaxima, MAX_TRIP_PRODUCT};
use alice_sdf::mesh::gpu_mc_shaders;
use std::collections::BTreeMap;
mod common;
use common::corpus::corpus;

/// The module-creation call sites, by file: how many there are, and the source kinds they create (which the census
/// below builds). A new call site or kind must be added here and to the census.
const SITES: &[(&str, usize, &[&str])] = &[
    (
        "src/compiled/wgsl/gpu_eval.rs",
        3,
        &[
            "wgsl compute",
            "wgsl node data",
            "wgsl dynamic",
            "wgsl dynamic normals",
            "glsl compute",
            "caller text",
        ],
    ),
    (
        "src/mesh/gpu_marching_cubes.rs",
        3,
        &[
            "mc grid",
            "mc classify",
            "mc vertex",
            "prefix scan",
            "prefix propagate",
        ],
    ),
    ("src/volume/gpu_bake.rs", 2, &["volume", "volume normals"]),
    ("tests/test_npr_bytecode_gpu_parity.rs", 1, &["npr vm"]),
    ("tests/test_instanced_wgsl_gpu_parity.rs", 1, &["instanced"]),
    // the on-device check of the guard itself: fixed test text (bounded, unbounded, over the ceiling)
    ("tests/test_gpu_guard_on_device.rs", 3, &["caller text"]),
];

/// The Metal kernel compile (MSL is made from the same WGSL kinds as the wgpu evaluator)
const METAL_SITES: &[(&str, usize)] = &[("src/compiled/wgsl/gpu_metal.rs", 3)];

fn wgsl(text: String) -> wgpu::ShaderSource<'static> {
    wgpu::ShaderSource::Wgsl(text.into())
}

#[test]
fn every_module_of_the_creation_sites_has_bounded_loops() {
    let maxima = HostMaxima::new();
    // (kind, name, source)
    let mut modules: Vec<(&str, String, wgpu::ShaderSource<'static>)> = Vec::new();
    let mut instance_limits: Vec<(&str, usize)> = Vec::new();
    for (name, node) in corpus() {
        let s = WgslShader::transpile(&node, alice_sdf::compiled::TranspileMode::Hardcoded);
        if let Ok(d) = WgslShader::try_transpile_with_node_data(
            &node,
            alice_sdf::compiled::TranspileMode::Hardcoded,
        ) {
            modules.push((
                "wgsl node data",
                format!("{name}/node_data"),
                wgsl(d.to_compute_shader()),
            ));
        }
        let dynamic = WgslShader::transpile(&node, alice_sdf::compiled::TranspileMode::Dynamic);
        modules.push((
            "wgsl dynamic",
            format!("{name}/dynamic"),
            wgsl(dynamic.to_compute_shader()),
        ));
        modules.push((
            "wgsl dynamic normals",
            format!("{name}/dynamic_normals"),
            wgsl(dynamic.to_compute_shader_with_normals()),
        ));
        modules.push((
            "wgsl compute",
            format!("{name}/compute"),
            wgsl(s.to_compute_shader()),
        ));
        modules.push((
            "wgsl compute",
            format!("{name}/compute_normals"),
            wgsl(s.to_compute_shader_with_normals()),
        ));
        #[cfg(feature = "volume")]
        {
            modules.push((
                "volume",
                format!("{name}/volume"),
                wgsl(s.to_volume_shader()),
            ));
            modules.push((
                "volume normals",
                format!("{name}/volume_normals"),
                wgsl(alice_sdf::volume::gpu_bake::generate_volume_bake_shader_with_normals(&s)),
            ));
        }
        modules.push((
            "mc grid",
            format!("{name}/mc_grid"),
            wgsl(gpu_mc_shaders::generate_sdf_grid_shader(&s)),
        ));
        modules.push((
            "mc vertex",
            format!("{name}/mc_vertex"),
            wgsl(gpu_mc_shaders::generate_vertex_shader(&s)),
        ));
        // the instance limit and the emitted bound come from one function; every corpus base fits
        let limit = alice_sdf::compiled::InstancedSdf::gpu_instance_limit(&node)
            .unwrap_or_else(|e| panic!("{name}: instanced refused: {e}"));
        let text = alice_sdf::compiled::InstancedSdf::try_to_instanced_wgsl(&node)
            .unwrap_or_else(|e| panic!("{name}: instanced refused: {e}"));
        assert_eq!(
            text.matches(&format!("i < {limit}u;")).count(),
            1,
            "{name}: the emitted instance bound is not gpu_instance_limit ({limit})"
        );
        instance_limits.push((name, limit));
        modules.push(("instanced", format!("{name}/instanced"), wgsl(text)));
        #[cfg(feature = "glsl")]
        {
            let g = alice_sdf::compiled::GlslShader::transpile(
                &node,
                alice_sdf::compiled::GlslTranspileMode::Hardcoded,
            );
            modules.push((
                "glsl compute",
                format!("{name}/glsl_compute"),
                wgpu::ShaderSource::Glsl {
                    shader: g.to_vulkan_compute_shader().into(),
                    stage: wgpu::naga::ShaderStage::Compute,
                    defines: std::collections::HashMap::default(),
                },
            ));
        }
    }
    modules.push((
        "mc classify",
        "mc_classify".into(),
        wgsl(gpu_mc_shaders::generate_classify_shader()),
    ));
    modules.push((
        "prefix scan",
        "prefix_scan".into(),
        wgsl(gpu_mc_shaders::generate_prefix_sum_scan_shader()),
    ));
    modules.push((
        "prefix propagate",
        "prefix_propagate".into(),
        wgsl(gpu_mc_shaders::generate_prefix_sum_propagate_shader()),
    ));
    modules.push((
        "npr vm",
        "npr_vm".into(),
        wgsl(common::npr_shader::shader_source()),
    ));

    assert!(!instance_limits.is_empty(), "no instanced base compared");
    for (name, limit) in &instance_limits {
        if *limit < alice_sdf::compiled::MAX_INSTANCES {
            println!("[instance-limit] {name}: {limit}");
        }
    }
    println!(
        "[instance-limit] bases {} under MAX_INSTANCES {} of {}",
        instance_limits.len(),
        instance_limits
            .iter()
            .filter(|(_, l)| *l < alice_sdf::compiled::MAX_INSTANCES)
            .count(),
        instance_limits.len()
    );
    let mut failures = Vec::new();
    let mut per_kind: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut loops, mut checked) = (0usize, 0usize);
    let mut worst_all = 0u64;
    for (kind, name, src) in &modules {
        let module = parse_source(src).unwrap_or_else(|e| panic!("{name}: does not parse: {e}"));
        let r = check_module(&module, &maxima);
        checked += 1;
        *per_kind.entry(kind).or_default() += 1;
        loops += r.loops;
        println!(
            "[wgsl-loops] {name}: loops {} unbounded {} worst {}",
            r.loops,
            r.unbounded.len(),
            r.worst
        );
        if r.worst > MAX_TRIP_PRODUCT {
            failures.push(format!(
                "{name}: worst trip product {} over {MAX_TRIP_PRODUCT}",
                r.worst
            ));
        }
        worst_all = worst_all.max(r.worst);
        for u in r.unbounded {
            failures.push(format!("{name} fn {}: {}", u.function, u.reason));
        }
    }
    // every registered kind (but caller text, checked where it is created) has modules here
    let registered: Vec<&str> = SITES
        .iter()
        .flat_map(|(_, _, kinds)| kinds.iter().copied())
        .filter(|k| *k != "caller text")
        .collect();
    for k in &registered {
        let gated = (*k == "glsl compute" && !cfg!(feature = "glsl"))
            || (k.starts_with("volume") && !cfg!(feature = "volume"));
        if !gated {
            assert!(
                per_kind.get(k).copied().unwrap_or(0) > 0,
                "source kind {k} has no module in the census"
            );
        }
    }
    println!(
        "[wgsl-loops] modules {checked} kinds {} (registered {}) loops {loops} unbounded {} worst product {worst_all} (ceiling {MAX_TRIP_PRODUCT})",
        per_kind.len(),
        registered.len(),
        failures.len()
    );
    assert!(checked > 0, "no module checked");
    assert!(
        failures.is_empty(),
        "unbounded loops:\n{}",
        failures.join("\n")
    );
}

/// Count `needle` in the non-comment lines of every `.rs` file under `dir` (relative to the crate root)
fn count_calls(dir: &str, needle: &str) -> BTreeMap<String, usize> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut out = BTreeMap::new();
    let mut stack = vec![root.join(dir)];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).expect("read a source directory") {
            let p = e.expect("entry").path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let text = std::fs::read_to_string(&p).expect("read a source file");
                let n = text
                    .lines()
                    .filter(|l| !l.trim_start().starts_with("//"))
                    .filter(|l| l.contains(needle))
                    .count();
                if n > 0 {
                    let rel = p
                        .strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.insert(rel, n);
                }
            }
        }
    }
    out
}

#[test]
fn every_module_creation_site_is_registered() {
    let mut found = count_calls("src", ".shader_module(");
    found.extend(count_calls("tests", ".shader_module("));
    // the definition itself is not a call site, nor is this file (it names the call in its search)
    found.remove("src/gpu_context.rs");
    found.remove("tests/test_wgsl_loop_bounds.rs");
    let metal = count_calls("src", "Kernel::compile(&device");
    let total: usize = found.values().sum::<usize>() + metal.values().sum::<usize>();
    println!(
        "[module-sites] {} shader_module call sites in {} files, {} Metal compile sites",
        found.values().sum::<usize>(),
        found.len(),
        metal.values().sum::<usize>()
    );
    assert!(total > 0, "no module-creation call site found");
    let registered: BTreeMap<String, usize> = SITES
        .iter()
        .map(|(f, n, _)| ((*f).to_string(), *n))
        .collect();
    assert_eq!(
        found, registered,
        "unregistered or changed shader_module call sites (update SITES and the census)"
    );
    let metal_registered: BTreeMap<String, usize> = METAL_SITES
        .iter()
        .map(|(f, n)| ((*f).to_string(), *n))
        .collect();
    // the Metal module exists only on macOS with msl; on other targets it is still in the source tree
    assert_eq!(
        metal, metal_registered,
        "unregistered or changed Metal compile sites"
    );
}
