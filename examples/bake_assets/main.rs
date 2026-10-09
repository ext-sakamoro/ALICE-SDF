//! Bake a few representative SDF scenes into render and physics assets, then
//! verify every written file from disk.
//!
//! ```text
//! cargo run --example bake_assets --features gpu-mesh -- bake   <out-dir> [--res N]
//! cargo run --example bake_assets --features gpu-mesh -- verify <out-dir>
//! cargo run --example bake_assets --features gpu-mesh -- mutate <kind> <out-dir>
//! ```
//!
//! Per scene (`<out-dir>/<scene>/`):
//!
//! - `mesh_gpu.abm` — GPU marching cubes (`gpu_marching_cubes`, welded), when
//!   built with `gpu-mesh` and an adapter is present
//! - `mesh_cpu.abm` — CPU marching cubes (`sdf_to_mesh`) on the same grid
//! - `mesh.glb` — glTF 2.0 binary of the render mesh (GPU when present)
//! - `mesh.nanite` — Nanite cluster hierarchy (scenes marked for it)
//! - `law.asdf` / `law.asdf.json` — the SDF tree itself (binary / JSON)
//! - `law.bytecode` — the compiled bytecode (`CompiledSdf`), layout below
//!
//! and `<out-dir>/manifest.json` with, per scene, the grid, the mesh counts
//! and bounds, a physics collider block (tight AABB of the field, volume,
//! centre of mass, inertia tensor about it, unit density) and the SHA-256 of
//! every file.
//!
//! `bake` writes everything and then runs `verify` on the written directory,
//! so what is checked is what is stored. `verify` exits non-zero when any
//! check fails, including "no scene verified". `mutate` corrupts a baked
//! directory in one way (and refreshes the affected SHA-256 so the semantic
//! check, not the hash, has to catch it); `scripts/bake_teeth.sh` runs each
//! kind and expects `verify` to fail.
//!
//! Without the `gpu-mesh` feature or without an adapter the GPU mesh is
//! skipped and the manifest says so; `ALICE_SDF_REQUIRE_GPU=1` turns that into
//! a failure (CI runs on lavapipe with it set).
//!
//! `law.bytecode` layout (little-endian): magic `ASDFBC01`, `u32` instruction
//! count, `u32` aux length, `f32` Lipschitz bound, then per instruction
//! `7 × f32` params, `u8` opcode, `u8` flags, `u16` child count, `u32` skip
//! offset, `u32` aux offset, `u32` aux length (44 bytes), then the aux `f32`s.
//!
//! Author: Moroya Sakamoto

mod mass;

use alice_sdf::io::{
    export_glb, export_nanite, import_glb, load_abm, load_asdf, load_asdf_json, save_abm,
    save_asdf, save_asdf_json, to_json_string, GltfConfig,
};
use alice_sdf::mesh::{
    generate_nanite_mesh, sdf_to_mesh, validate_mesh, MarchingCubesConfig, Mesh, NaniteConfig,
};
use alice_sdf::prelude::*;
use alice_sdf::tight_aabb::compute_tight_aabb;
use glam::DVec3;
use mass::{mass_properties, MassProps};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const FORMAT: &str = "alice-sdf-bake/1";
const DEFAULT_RES: u32 = 64;

/// GPU and CPU meshes on the same grid: largest vertex-to-surface distance
/// in either direction, in cells.
const GPU_CPU_VERTEX_TOL_CELLS: f64 = 1e-3;
/// GPU and CPU meshes: relative difference of the total surface area.
const GPU_CPU_AREA_TOL: f64 = 1e-5;
/// Closed-form scenes: relative error of the mesh volume and of the
/// principal moments against the solid.
const ANALYTIC_TOL: f64 = 0.01;
/// Mass properties recomputed from the stored mesh against the manifest.
const MANIFEST_REL_TOL: f64 = 1e-9;

/// Closed-form volume, centre and principal moments (unit density, axes
/// aligned) of a scene.
#[derive(Clone, Copy)]
struct Analytic {
    volume: f64,
    centroid: [f64; 3],
    inertia_diag: [f64; 3],
}

struct Scene {
    name: &'static str,
    node: SdfNode,
    analytic: Option<Analytic>,
    nanite: bool,
}

fn scenes() -> Vec<Scene> {
    let r = 0.8f64;
    let v_sphere = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
    let (w, h, d) = (1.2f64, 0.8f64, 1.0f64);
    let t = [0.1f64, 0.0, -0.05];
    let v_box = w * h * d;
    vec![
        Scene {
            name: "sphere",
            node: SdfNode::sphere(r as f32),
            analytic: Some(Analytic {
                volume: v_sphere,
                centroid: [0.0; 3],
                inertia_diag: [0.4 * v_sphere * r * r; 3],
            }),
            nanite: false,
        },
        Scene {
            name: "box",
            node: SdfNode::box3d(w as f32, h as f32, d as f32).translate(
                t[0] as f32,
                t[1] as f32,
                t[2] as f32,
            ),
            analytic: Some(Analytic {
                volume: v_box,
                centroid: t,
                inertia_diag: [
                    v_box * (h * h + d * d) / 12.0,
                    v_box * (d * d + w * w) / 12.0,
                    v_box * (w * w + h * h) / 12.0,
                ],
            }),
            nanite: false,
        },
        Scene {
            name: "box_minus_sphere",
            node: SdfNode::box3d(1.2, 1.2, 1.2).subtract(SdfNode::sphere(0.75)),
            analytic: None,
            nanite: false,
        },
        Scene {
            name: "smooth_union",
            node: SdfNode::sphere(0.5)
                .translate(-0.35, 0.0, 0.0)
                .smooth_union(SdfNode::sphere(0.4).translate(0.4, 0.1, 0.0), 0.2),
            analytic: None,
            nanite: false,
        },
        Scene {
            name: "rotated_band",
            node: SdfNode::box3d(1.6, 0.16, 0.5).rotate_euler(0.3, 0.5, 0.7),
            analytic: None,
            nanite: false,
        },
        Scene {
            name: "gyroid_ball",
            node: SdfNode::sphere(0.8).intersection(SdfNode::gyroid(6.0, 0.12)),
            analytic: None,
            nanite: true,
        },
    ]
}

fn scene_by_name(name: &str) -> Option<Scene> {
    scenes().into_iter().find(|s| s.name == name)
}

// ───────────────────────────────────────────────────────────── helpers

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn mesh_aabb(m: &Mesh) -> (Vec3, Vec3) {
    m.vertices.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), v| (lo.min(v.position), hi.max(v.position)),
    )
}

fn mesh_area(m: &Mesh) -> f64 {
    m.indices
        .chunks_exact(3)
        .map(|t| {
            let p = [0, 1, 2].map(|k| m.vertices[t[k] as usize].position.as_dvec3());
            0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
        })
        .sum()
}

fn mesh_mass(m: &Mesh) -> Option<MassProps> {
    mass_properties(
        m.indices
            .chunks_exact(3)
            .map(|t| [0, 1, 2].map(|k| m.vertices[t[k] as usize].position.as_dvec3().to_array())),
    )
}

/// Closest point on triangle `abc` to `p` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn closest_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// Largest distance from a vertex of `a` to the surface of `b` (triangles
/// bucketed by centroid in `cell`-sized buckets; a marching-cubes triangle
/// lies within one cell, so the 27 neighbouring buckets hold every triangle
/// closer than a cell. A vertex with none counts as infinitely far).
/// Point-to-surface rather than vertex-to-vertex: the two pipelines weld
/// sliver vertices near lattice corners differently, which moves vertices
/// along the surface but not off it.
fn max_distance_to_surface(a: &Mesh, b: &Mesh, cell: f32) -> f64 {
    let cell = f64::from(cell);
    let key = |p: DVec3| {
        (
            (p.x / cell).floor() as i64,
            (p.y / cell).floor() as i64,
            (p.z / cell).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<[DVec3; 3]>> = HashMap::new();
    for t in b.indices.chunks_exact(3) {
        let tri = [0, 1, 2].map(|k| b.vertices[t[k] as usize].position.as_dvec3());
        grid.entry(key((tri[0] + tri[1] + tri[2]) / 3.0))
            .or_default()
            .push(tri);
    }
    let mut worst = 0.0f64;
    for v in &a.vertices {
        let p = v.position.as_dvec3();
        let (x, y, z) = key(p);
        let mut best = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for t in grid.get(&(x + dx, y + dy, z + dz)).into_iter().flatten() {
                        best = best.min(p.distance(closest_on_triangle(p, t[0], t[1], t[2])));
                    }
                }
            }
        }
        worst = worst.max(best);
    }
    worst
}

fn bytecode_bytes(c: &CompiledSdf) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"ASDFBC01");
    out.extend_from_slice(&(c.instructions().len() as u32).to_le_bytes());
    out.extend_from_slice(&(c.aux_data().len() as u32).to_le_bytes());
    out.extend_from_slice(&c.lipschitz().to_le_bytes());
    for ins in c.instructions() {
        for p in ins.params {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out.push(ins.opcode as u8);
        out.push(ins.flags);
        out.extend_from_slice(&ins.child_count.to_le_bytes());
        out.extend_from_slice(&ins.skip_offset.to_le_bytes());
        out.extend_from_slice(&ins.aux_offset.to_le_bytes());
        out.extend_from_slice(&ins.aux_len.to_le_bytes());
    }
    for a in c.aux_data() {
        out.extend_from_slice(&a.to_le_bytes());
    }
    out
}

fn v3(v: Vec3) -> Value {
    json!([v.x, v.y, v.z])
}

fn aabb_json(lo: Vec3, hi: Vec3) -> Value {
    json!({ "min": v3(lo), "max": v3(hi) })
}

fn read_v3(v: &Value) -> Result<Vec3, String> {
    let a = v.as_array().ok_or("expected [x, y, z]")?;
    if a.len() != 3 {
        return Err("expected [x, y, z]".into());
    }
    let f = |i: usize| a[i].as_f64().map(|x| x as f32).ok_or("expected a number");
    Ok(Vec3::new(f(0)?, f(1)?, f(2)?))
}

fn read_aabb(v: &Value) -> Result<(Vec3, Vec3), String> {
    Ok((read_v3(&v["min"])?, read_v3(&v["max"])?))
}

fn num(v: &Value, what: &str) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| format!("manifest: {what} is not a number"))
}

/// Cubic grid around the tight AABB with 3.5 empty cells on every side: the
/// surface never touches the grid boundary (which would open the mesh), and
/// the half cell keeps axis-aligned faces off the lattice planes. A corner
/// sitting exactly on the iso-surface makes several edge vertices coincide,
/// and welding them pinches the mesh (measured: box minus sphere at 3 cells,
/// 555 non-manifold edges on the CPU marching cubes welded by distance, the
/// same on the GPU; 0 at 3.5 cells).
fn grid_bounds(lo: Vec3, hi: Vec3, res: u32) -> (Vec3, Vec3, f32) {
    let c = (lo + hi) * 0.5;
    let m = (hi - lo).max_element() * 0.5;
    let half = m / (1.0 - 7.0 / res as f32);
    let cell = 2.0 * half / res as f32;
    (c - Vec3::splat(half), c + Vec3::splat(half), cell)
}

// ───────────────────────────────────────────────────────────── GPU

#[cfg(feature = "gpu-mesh")]
fn gpu_mesh(node: &SdfNode, lo: Vec3, hi: Vec3, res: u32, cell: f32) -> Result<Mesh, String> {
    use alice_sdf::mesh::{
        gpu_marching_cubes, remove_degenerate_triangles, GpuMarchingCubesConfig, MeshRepair,
    };
    let cfg = GpuMarchingCubesConfig {
        resolution: res,
        ..Default::default()
    };
    let soup = gpu_marching_cubes(node, lo, hi, &cfg).map_err(|e| e.to_string())?;
    // GPU MC emits one vertex per triangle corner, and the copies of an edge
    // vertex made by neighbouring cells differ in the last bits (an exact
    // weld leaves thousands of boundary edges), so weld by distance — far
    // below any real vertex spacing — and drop the triangles that collapse.
    let mut m = MeshRepair::merge_duplicate_vertices(&soup, 1e-4 * cell);
    remove_degenerate_triangles(&mut m);
    Ok(m)
}

#[cfg(not(feature = "gpu-mesh"))]
fn gpu_mesh(_: &SdfNode, _: Vec3, _: Vec3, _: u32, _: f32) -> Result<Mesh, String> {
    Err("built without the gpu-mesh feature".into())
}

// ───────────────────────────────────────────────────────────── bake

fn bake(out: &Path, res: u32) -> Result<(), String> {
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut gpu_status = json!({ "status": "ok" });
    let mut scene_entries = Vec::new();

    for s in scenes() {
        let dir = out.join(s.name);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let compiled = CompiledSdf::try_compile(&s.node).map_err(|e| format!("{}: {e}", s.name))?;
        let tight = compute_tight_aabb(&s.node);
        let (lo, hi, cell) = grid_bounds(tight.min, tight.max, res);

        let cpu = sdf_to_mesh(
            &s.node,
            lo,
            hi,
            &MarchingCubesConfig {
                resolution: res as usize,
                ..Default::default()
            },
        );
        let gpu = match gpu_mesh(&s.node, lo, hi, res, cell) {
            Ok(m) => Some(m),
            Err(e) => {
                gpu_status = json!({ "status": "skipped", "reason": e });
                None
            }
        };
        let render = gpu.as_ref().unwrap_or(&cpu);
        let props =
            mesh_mass(render).ok_or_else(|| format!("{}: empty or inverted mesh", s.name))?;

        let mut files: Vec<(&str, &str)> = Vec::new();
        save_abm(&cpu, dir.join("mesh_cpu.abm")).map_err(|e| e.to_string())?;
        files.push(("mesh_cpu.abm", "mesh/cpu"));
        if let Some(g) = &gpu {
            save_abm(g, dir.join("mesh_gpu.abm")).map_err(|e| e.to_string())?;
            files.push(("mesh_gpu.abm", "mesh/gpu"));
        }
        export_glb(render, dir.join("mesh.glb"), &GltfConfig::default(), None)
            .map_err(|e| e.to_string())?;
        files.push(("mesh.glb", "render/gltf"));
        if s.nanite {
            let cfg = NaniteConfig {
                lod_levels: 3,
                base_resolution: res,
                ..Default::default()
            };
            let nanite = generate_nanite_mesh(&s.node, lo, hi, &cfg);
            export_nanite(&nanite, dir.join("mesh.nanite")).map_err(|e| e.to_string())?;
            files.push(("mesh.nanite", "render/nanite"));
        }
        let tree = SdfTree::new(s.node.clone());
        save_asdf(&tree, dir.join("law.asdf")).map_err(|e| e.to_string())?;
        files.push(("law.asdf", "law/asdf"));
        save_asdf_json(&tree, dir.join("law.asdf.json")).map_err(|e| e.to_string())?;
        files.push(("law.asdf.json", "law/json"));
        std::fs::write(dir.join("law.bytecode"), bytecode_bytes(&compiled))
            .map_err(|e| e.to_string())?;
        files.push(("law.bytecode", "law/bytecode"));

        let mut file_entries = Vec::new();
        for (f, role) in files {
            let bytes = std::fs::read(dir.join(f)).map_err(|e| e.to_string())?;
            file_entries.push(json!({
                "path": format!("{}/{f}", s.name),
                "role": role,
                "bytes": bytes.len(),
                "sha256": sha256_hex(&bytes),
            }));
        }
        let mesh_entry = |m: &Mesh| {
            let (a, b) = mesh_aabb(m);
            json!({
                "vertices": m.vertex_count(),
                "triangles": m.triangle_count(),
                "aabb": aabb_json(a, b),
                "area": mesh_area(m),
            })
        };
        scene_entries.push(json!({
            "name": s.name,
            "grid": aabb_json(lo, hi),
            "cell": cell,
            "lipschitz": compiled.lipschitz(),
            "render_source": if gpu.is_some() { "gpu" } else { "cpu" },
            "meshes": {
                "cpu": mesh_entry(&cpu),
                "gpu": gpu.as_ref().map(mesh_entry),
            },
            "collider": {
                "aabb": aabb_json(tight.min, tight.max),
                "density": 1.0,
                "volume": props.volume,
                "centroid": props.centroid,
                "inertia": props.inertia,
            },
            "files": file_entries,
        }));
    }

    let manifest = json!({
        "format": FORMAT,
        "crate_version": env!("CARGO_PKG_VERSION"),
        "resolution": res,
        "gpu": gpu_status,
        "scenes": scene_entries,
    });
    write_manifest(out, &manifest)
}

fn write_manifest(out: &Path, manifest: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?;
    std::fs::write(out.join("manifest.json"), text + "\n").map_err(|e| e.to_string())
}

fn read_manifest(out: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(out.join("manifest.json"))
        .map_err(|e| format!("{}/manifest.json: {e}", out.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("manifest.json: {e}"))
}

// ───────────────────────────────────────────────────────────── verify

/// Collects the failed checks of one scene instead of stopping at the first,
/// so a report shows every check a corruption trips.
struct Checks {
    scene: String,
    failed: Vec<String>,
    passed: usize,
}

impl Checks {
    fn check(&mut self, ok: bool, what: impl FnOnce() -> String) {
        if ok {
            self.passed += 1;
        } else {
            let msg = what();
            eprintln!("FAIL [{}] {msg}", self.scene);
            self.failed.push(msg);
        }
    }
}

fn verify(out: &Path) -> Result<(), String> {
    let manifest = read_manifest(out)?;
    if manifest["format"] != FORMAT {
        return Err(format!(
            "manifest format {} != {FORMAT}",
            manifest["format"]
        ));
    }
    let gpu_ok = manifest["gpu"]["status"] == "ok";
    if !gpu_ok {
        let reason = manifest["gpu"]["reason"]
            .as_str()
            .unwrap_or("?")
            .to_string();
        if std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_some() {
            return Err(format!(
                "ALICE_SDF_REQUIRE_GPU is set but the GPU mesh was skipped: {reason}"
            ));
        }
        println!("SKIPPED: GPU mesh ({reason}) — GPU-vs-CPU checks not run, CPU checks only");
    }
    let scenes_json = manifest["scenes"]
        .as_array()
        .ok_or("manifest: scenes is not an array")?;
    let mut total_failed = 0;
    let mut total_passed = 0;
    let mut verified = 0;
    for sj in scenes_json {
        let name = sj["name"]
            .as_str()
            .ok_or("manifest: scene without a name")?;
        let mut c = Checks {
            scene: name.to_string(),
            failed: Vec::new(),
            passed: 0,
        };
        if let Err(e) = verify_scene(out, sj, gpu_ok, &mut c) {
            c.check(false, || e);
        }
        println!(
            "scene {name:<18} checks passed {:>3}, failed {}",
            c.passed,
            c.failed.len()
        );
        total_failed += c.failed.len();
        total_passed += c.passed;
        verified += 1;
    }
    println!("verified {verified} scenes, {total_passed} checks passed, {total_failed} failed");
    if verified == 0 {
        return Err("no scene verified (manifest lists 0 scenes)".into());
    }
    if total_passed == 0 {
        return Err("0 checks ran".into());
    }
    if total_failed > 0 {
        return Err(format!("{total_failed} checks failed"));
    }
    Ok(())
}

fn verify_scene(out: &Path, sj: &Value, gpu_ok: bool, c: &mut Checks) -> Result<(), String> {
    let name = c.scene.clone();
    let scene = scene_by_name(&name).ok_or_else(|| format!("unknown scene {name}"))?;
    let dir = out.join(&name);

    // 1. every file is present and matches its SHA-256
    let files = sj["files"].as_array().ok_or("files is not an array")?;
    c.check(!files.is_empty(), || "no files listed".into());
    for f in files {
        let path = f["path"].as_str().ok_or("file without a path")?;
        match std::fs::read(out.join(path)) {
            Ok(bytes) => {
                let got = sha256_hex(&bytes);
                c.check(f["sha256"] == got.as_str(), || {
                    format!("{path}: sha256 {got} != manifest")
                });
                c.check(f["bytes"] == bytes.len(), || {
                    format!("{path}: size differs from manifest")
                });
            }
            Err(e) => c.check(false, || format!("{path}: {e}")),
        }
    }

    // 2. the law: binary and JSON trees agree, recompiling gives the stored bytecode
    let tree = load_asdf(dir.join("law.asdf")).map_err(|e| format!("law.asdf: {e}"))?;
    let tree_json =
        load_asdf_json(dir.join("law.asdf.json")).map_err(|e| format!("law.asdf.json: {e}"))?;
    let (a, b) = (
        to_json_string(&tree).map_err(|e| e.to_string())?,
        to_json_string(&tree_json).map_err(|e| e.to_string())?,
    );
    c.check(a == b, || {
        "law.asdf and law.asdf.json describe different trees".into()
    });
    let expected = to_json_string(&SdfTree::new(scene.node.clone())).map_err(|e| e.to_string())?;
    c.check(a == expected, || "law.asdf is not the scene's tree".into());
    let compiled = CompiledSdf::try_compile(&tree.root).map_err(|e| e.to_string())?;
    let stored = std::fs::read(dir.join("law.bytecode")).map_err(|e| e.to_string())?;
    c.check(stored == bytecode_bytes(&compiled), || {
        "law.bytecode != compile(law.asdf)".into()
    });
    let node = tree.root;

    let cell = num(&sj["cell"], "cell")? as f32;
    let lip = compiled.lipschitz();
    c.check(lip.is_finite() && lip > 0.0, || {
        format!("Lipschitz bound {lip}")
    });
    let lip = lip.max(1.0);
    let eps = lip * cell;

    // 3. meshes: watertight, counts as in the manifest, vertices on the surface
    let cpu = load_abm(dir.join("mesh_cpu.abm")).map_err(|e| format!("mesh_cpu.abm: {e}"))?;
    let gpu = if gpu_ok {
        Some(load_abm(dir.join("mesh_gpu.abm")).map_err(|e| format!("mesh_gpu.abm: {e}"))?)
    } else {
        None
    };
    let mut meshes = vec![("cpu", &cpu)];
    if let Some(g) = &gpu {
        meshes.push(("gpu", g));
    }
    for (label, m) in &meshes {
        let mj = &sj["meshes"][label];
        let val = validate_mesh(m);
        c.check(m.triangle_count() > 0, || format!("{label}: 0 triangles"));
        c.check(val.boundary_edges == 0, || {
            format!(
                "{label}: {} boundary edges (not watertight)",
                val.boundary_edges
            )
        });
        c.check(val.non_manifold_edges == 0, || {
            format!("{label}: {} non-manifold edges", val.non_manifold_edges)
        });
        c.check(
            mj["vertices"] == m.vertex_count() && mj["triangles"] == m.triangle_count(),
            || format!("{label}: counts differ from the manifest"),
        );
        let worst = m
            .vertices
            .iter()
            .map(|v| eval(&node, v.position).abs())
            .fold(0.0f32, f32::max);
        c.check(worst <= eps, || {
            format!("{label}: max |SDF| at vertices {worst:.3e} > {eps:.3e}")
        });
        println!(
            "  {name}/{label}: {} vertices, {} triangles, boundary {} NME {}, max |SDF| {worst:.2e} (bound {eps:.2e})",
            m.vertex_count(),
            m.triangle_count(),
            val.boundary_edges,
            val.non_manifold_edges
        );
    }

    // 4. GPU and CPU meshes agree
    if let Some(g) = &gpu {
        let d = max_distance_to_surface(g, &cpu, cell).max(max_distance_to_surface(&cpu, g, cell))
            / f64::from(cell);
        let (ag, ac) = (mesh_area(g), mesh_area(&cpu));
        let da = (ag - ac).abs() / ac;
        println!(
            "  {name}/gpu-vs-cpu: max vertex-to-surface distance {d:.2e} cells, area rel {da:.2e}"
        );
        c.check(d <= GPU_CPU_VERTEX_TOL_CELLS, || {
            format!("gpu-vs-cpu: vertex-to-surface distance {d:.3e} cells > {GPU_CPU_VERTEX_TOL_CELLS:e}")
        });
        c.check(da <= GPU_CPU_AREA_TOL, || {
            format!("gpu-vs-cpu: area rel {da:.3e} > {GPU_CPU_AREA_TOL:e}")
        });
    }

    // 5. physics block: recomputed from the render mesh, contains the meshes
    let render = gpu.as_ref().unwrap_or(&cpu);
    let props = mesh_mass(render).ok_or("render mesh has no positive volume")?;
    let col = &sj["collider"];
    let rel =
        |a: f64, b: f64, scale: f64| (a - b).abs() <= MANIFEST_REL_TOL * scale.abs().max(1e-12);
    let vol = num(&col["volume"], "collider.volume")?;
    c.check(rel(props.volume, vol, vol), || {
        format!("collider volume {vol} != mesh {}", props.volume)
    });
    let isc = props.inertia[0][0].abs() + props.inertia[1][1].abs() + props.inertia[2][2].abs();
    for i in 0..3 {
        let ci = num(&col["centroid"][i], "collider.centroid")?;
        c.check(rel(props.centroid[i], ci, 1.0), || {
            format!("collider centroid[{i}] {ci} != mesh")
        });
        for j in 0..3 {
            let iij = num(&col["inertia"][i][j], "collider.inertia")?;
            c.check(rel(props.inertia[i][j], iij, isc), || {
                format!("collider inertia[{i}][{j}] {iij} != mesh")
            });
        }
    }
    let (clo, chi) = read_aabb(&col["aabb"])?;
    for (label, m) in &meshes {
        let (mlo, mhi) = mesh_aabb(m);
        let inside =
            (clo - Vec3::splat(eps)).cmple(mlo).all() && mhi.cmple(chi + Vec3::splat(eps)).all();
        c.check(inside, || {
            format!("collider AABB {clo:?}..{chi:?} does not contain the {label} mesh AABB {mlo:?}..{mhi:?} (eps {eps:.2e})")
        });
    }

    // 6. closed forms
    if let Some(an) = scene.analytic {
        let ev = (props.volume - an.volume).abs() / an.volume;
        let ei = (0..3)
            .map(|k| (props.inertia[k][k] - an.inertia_diag[k]).abs() / an.inertia_diag[k])
            .fold(0.0f64, f64::max);
        let ec = (0..3)
            .map(|k| (props.centroid[k] - an.centroid[k]).abs())
            .fold(0.0f64, f64::max);
        println!(
            "  {name}/closed-form: volume rel {ev:.2e}, inertia rel {ei:.2e}, centroid {ec:.2e}"
        );
        c.check(ev <= ANALYTIC_TOL, || {
            format!("volume rel {ev:.3e} > {ANALYTIC_TOL}")
        });
        c.check(ei <= ANALYTIC_TOL, || {
            format!("inertia rel {ei:.3e} > {ANALYTIC_TOL}")
        });
        c.check(ec <= f64::from(cell) * 0.1, || {
            format!("centroid off by {ec:.3e}")
        });
    }

    // 7. the glTF holds the render mesh
    let glb = import_glb(dir.join("mesh.glb")).map_err(|e| format!("mesh.glb: {e}"))?;
    let same = glb.vertex_count() == render.vertex_count()
        && glb.indices == render.indices
        && glb
            .vertices
            .iter()
            .zip(&render.vertices)
            .all(|(a, b)| a.position == b.position);
    c.check(same, || "mesh.glb differs from the render mesh".into());
    Ok(())
}

// ───────────────────────────────────────────────────────────── mutate

fn refresh_sha(manifest: &mut Value, out: &Path, path: &str) -> Result<(), String> {
    let bytes = std::fs::read(out.join(path)).map_err(|e| e.to_string())?;
    for sj in manifest["scenes"].as_array_mut().ok_or("no scenes")? {
        for f in sj["files"].as_array_mut().ok_or("no files")? {
            if f["path"] == path {
                f["sha256"] = json!(sha256_hex(&bytes));
                f["bytes"] = json!(bytes.len());
            }
        }
    }
    Ok(())
}

fn mutate(kind: &str, out: &Path) -> Result<(), String> {
    let mut manifest = read_manifest(out)?;
    let first = manifest["scenes"][0]["name"]
        .as_str()
        .ok_or("no scene to mutate")?
        .to_string();
    let gpu_path = format!("{first}/mesh_gpu.abm");
    let render_path = if manifest["gpu"]["status"] == "ok" {
        gpu_path.clone()
    } else {
        format!("{first}/mesh_cpu.abm")
    };
    match kind {
        // move one GPU vertex by half a cell
        "gpu-vertex" => {
            if manifest["gpu"]["status"] != "ok" {
                return Err("no GPU mesh in this bake".into());
            }
            let cell = num(&manifest["scenes"][0]["cell"], "cell")? as f32;
            let mut m = load_abm(out.join(&gpu_path)).map_err(|e| e.to_string())?;
            m.vertices[0].position += Vec3::new(0.5 * cell, 0.0, 0.0);
            save_abm(&m, out.join(&gpu_path)).map_err(|e| e.to_string())?;
            refresh_sha(&mut manifest, out, &gpu_path)?;
        }
        // delete one triangle of the render mesh
        "drop-face" => {
            let mut m = load_abm(out.join(&render_path)).map_err(|e| e.to_string())?;
            m.indices.truncate(m.indices.len() - 3);
            save_abm(&m, out.join(&render_path)).map_err(|e| e.to_string())?;
            refresh_sha(&mut manifest, out, &render_path)?;
        }
        // shrink the collider AABB by 10 % of its extent on every side
        "shrink-aabb" => {
            let aabb = &mut manifest["scenes"][0]["collider"]["aabb"];
            let (lo, hi) = read_aabb(aabb)?;
            let d = (hi - lo) * 0.1;
            *aabb = aabb_json(lo + d, hi - d);
        }
        "no-scenes" => manifest["scenes"] = json!([]),
        _ => {
            return Err(format!(
                "unknown mutation {kind} (gpu-vertex | drop-face | shrink-aabb | no-scenes)"
            ))
        }
    }
    write_manifest(out, &manifest)?;
    println!("mutated {} ({kind})", out.display());
    Ok(())
}

// ───────────────────────────────────────────────────────────── main

fn run(args: &[String]) -> Result<(), String> {
    let usage =
        "usage: bake_assets bake <out-dir> [--res N] | verify <out-dir> | mutate <kind> <out-dir>";
    match args.first().map(String::as_str) {
        Some("bake") => {
            let out = PathBuf::from(args.get(1).ok_or(usage)?);
            let res = match args.get(2).map(String::as_str) {
                Some("--res") => args
                    .get(3)
                    .and_then(|s| s.parse::<u32>().ok())
                    .filter(|&r| (8..=256).contains(&r))
                    .ok_or("--res takes an integer in 8..=256")?,
                Some(other) => return Err(format!("unexpected argument {other}\n{usage}")),
                None => DEFAULT_RES,
            };
            let t = std::time::Instant::now();
            bake(&out, res)?;
            println!(
                "baked {} in {:.1} s, verifying",
                out.display(),
                t.elapsed().as_secs_f64()
            );
            verify(&out)
        }
        Some("verify") => verify(Path::new(args.get(1).ok_or(usage)?)),
        Some("mutate") => mutate(
            args.get(1).ok_or(usage)?,
            Path::new(args.get(2).ok_or(usage)?),
        ),
        _ => Err(usage.into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => {
            println!("OK");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
