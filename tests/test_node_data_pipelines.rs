//! The node data buffer (binding 4) carries a tree's data arrays in the shaders this crate builds a
//! pipeline for itself, so no large module constant array is read at a run-time index (a driver
//! that copies the whole array at each such read is slow, or does not build the pipeline).
//!
//! Checked here: a polygon outline goes to the buffer like the other data arrays; the volume bake
//! and the async evaluator read the buffer and agree with the CPU; data larger than the device's
//! storage binding limit is an error, not a failed pipeline. The exported text forms
//! (`WgslShader::transpile`, the FFI and engine forms) keep their data in the text.

#![cfg(feature = "gpu")]

use alice_sdf::compiled::{TranspileMode, WgslShader};

#[path = "common/gpu_batches.rs"]
mod gpu_batches;
use alice_sdf::prelude::*;
use glam::Vec2;

/// A closed outline of `n` vertices on a wobbled circle (more than the 8 the emit unrolls)
fn outline(n: usize) -> Vec<Vec2> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32 * std::f32::consts::TAU;
            let r = 0.6 + 0.1 * alice_det_math::sin(5.0 * t);
            let (s, c) = alice_det_math::sin_cos(t);
            Vec2::new(r * c, r * s)
        })
        .collect()
}

#[test]
fn a_polygon_outline_goes_to_the_node_data_buffer() {
    let n = 40;
    let node = SdfNode::polygon_2d(outline(n), 0.3);
    let shader = WgslShader::try_transpile_with_node_data(&node, TranspileMode::Hardcoded)
        .expect("transpile");
    assert!(
        !shader.source.contains("array<f32"),
        "a module-scope array is left in the node data form:\n{}",
        shader
            .source
            .lines()
            .filter(|l| l.contains("array<f32"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(
        shader.node_data.len(),
        2 * n,
        "the buffer holds the outline's x, y pairs"
    );
    // every read of the buffer is clamped to the outline's own range [0, 2n - 1]: a corrupted or
    // stale index reads one of its values, never past it (the bound is the last element, not one past)
    let reads: Vec<&str> = shader
        .source
        .match_indices("alice_node_data[clamp(")
        .map(|(i, _)| &shader.source[i..])
        .collect();
    assert!(!reads.is_empty(), "no clamped read of the buffer");
    let want = format!(", 0, {})]", 2 * n - 1);
    for r in &reads {
        let read = &r[..r.find(")]").map_or(r.len(), |k| k + 2)];
        assert!(
            read.ends_with(&want),
            "a read is not clamped to [0, {}]: {read}",
            2 * n - 1
        );
    }
    // the exported (self-contained) form keeps the outline in the text
    let text = WgslShader::transpile(&node, TranspileMode::Hardcoded);
    assert!(text.node_data.is_empty(), "the exported form has no buffer");
    assert!(
        text.source.contains("alice_poly_v_"),
        "the exported form lost its outline array"
    );
}

/// The outline read from the buffer is the outline: the node data evaluator agrees with the CPU at
/// points inside and around it (inside, the sign and the nearest edge both depend on every vertex
/// coordinate). A lone polygon, so no other operand of a union hides it
#[cfg(feature = "gpu")]
#[test]
fn the_outline_from_the_buffer_agrees_with_the_cpu() {
    use alice_sdf::compiled::GpuEvaluator;
    let node = SdfNode::polygon_2d(outline(40), 0.25);
    let gpu = match GpuEvaluator::new(&node) {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            eprintln!("skipped ({e})");
            return;
        }
    };
    // a 21 x 21 x 3 lattice over the outline's box (radius up to 0.7) and the slab around it
    let pts: Vec<Vec3> = (0..21 * 21 * 3)
        .map(|k| {
            let (i, j, l) = (k % 21, (k / 21) % 21, k / (21 * 21));
            Vec3::new(
                -0.83 + 0.083 * i as f32,
                -0.81 + 0.081 * j as f32,
                -0.2 + 0.2 * l as f32,
            )
        })
        .collect();
    let inside = pts.iter().filter(|p| eval(&node, **p) < 0.0).count();
    assert!(inside > 100, "only {inside} points inside the outline");
    let got = gpu_batches::eval_in_batches(&gpu, &pts).expect("dispatch");
    let worst = pts
        .iter()
        .zip(&got)
        .map(|(p, g)| (eval(&node, *p) - g).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "outline: worst |gpu - cpu| {worst:e} over {} points ({inside} inside)",
        pts.len()
    );
    assert!(worst < 1e-5, "outline: worst |gpu - cpu| {worst}");
}

/// Two identical outlines in the exported (self-contained) text share one module-scope array (the
/// data arrays are deduplicated by content), and both polygon functions read it: one array, two
/// functions naming it, and the text evaluated on the GPU agrees with the CPU around both copies
#[test]
fn identical_outlines_share_one_array_in_the_exported_text() {
    let poly = || SdfNode::polygon_2d(outline(40), 0.25);
    let node = poly()
        .translate(-0.8, 0.0, 0.0)
        .union(poly().translate(0.8, 0.0, 0.0));
    let text = WgslShader::transpile(&node, TranspileMode::Hardcoded);
    let arrays: Vec<&str> = text
        .source
        .lines()
        .filter(|l| l.trim_start().starts_with("const alice_poly_v_"))
        .collect();
    assert_eq!(
        arrays.len(),
        1,
        "identical outlines: {} arrays",
        arrays.len()
    );
    let name = arrays[0]
        .trim_start()
        .trim_start_matches("const ")
        .split(':')
        .next()
        .unwrap()
        .trim()
        .to_owned();
    let functions = text
        .source
        .lines()
        .filter(|l| l.starts_with("fn alice_poly_"))
        .count();
    assert_eq!(functions, 2, "one polygon function per copy");
    let readers = text
        .source
        .split("fn alice_poly_")
        .skip(1)
        .filter(|f| {
            f.split("\nfn ")
                .next()
                .unwrap_or(f)
                .contains(&format!("{name}["))
        })
        .count();
    assert_eq!(readers, 2, "both functions read {name}");
    // a different outline gets its own array
    let other = poly().union(SdfNode::polygon_2d(outline(41), 0.25).translate(1.5, 0.0, 0.0));
    let other_arrays = WgslShader::transpile(&other, TranspileMode::Hardcoded)
        .source
        .lines()
        .filter(|l| l.trim_start().starts_with("const alice_poly_v_"))
        .count();
    assert_eq!(other_arrays, 2, "different outlines share an array");

    let gpu = match alice_sdf::compiled::GpuEvaluator::from_shader(&text) {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            eprintln!("skipped ({e})");
            return;
        }
    };
    let pts: Vec<Vec3> = (0..41 * 21)
        .map(|k| {
            let (i, j) = (k % 41, k / 41);
            Vec3::new(-1.6 + 0.08 * i as f32, -0.8 + 0.08 * j as f32, 0.05)
        })
        .collect();
    let inside = pts.iter().filter(|p| eval(&node, **p) < 0.0).count();
    assert!(inside > 200, "only {inside} points inside the two outlines");
    let got = gpu_batches::eval_in_batches(&gpu, &pts).expect("dispatch");
    let worst = pts
        .iter()
        .zip(&got)
        .map(|(p, g)| (eval(&node, *p) - g).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "shared outline: worst |gpu - cpu| {worst:e} over {} points ({inside} inside)",
        pts.len()
    );
    assert!(worst < 1e-5, "shared outline: worst |gpu - cpu| {worst}");
}

/// A tree with node data in two forms: a 70 x 70 height map on a sphere, and a 40-vertex outline
#[cfg(feature = "volume")]
fn data_tree() -> SdfNode {
    let w = 70u32;
    let map: Vec<f32> = (0..w * w)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    let field = SdfNode::sphere(0.8).heightmap_displacement(map, w, w, 0.05, 1.0);
    let outline = SdfNode::polygon_2d(outline(40), 0.25).translate(0.9, 0.0, 0.0);
    field.union(outline)
}

/// `Some(value)`, or `None` (skip) when there is no GPU, unless `ALICE_SDF_REQUIRE_GPU` is set
#[cfg(feature = "volume")]
fn gpu_or_skip<T>(what: &str, r: Result<T, alice_sdf::compiled::GpuError>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "{what}: ALICE_SDF_REQUIRE_GPU is set but the GPU path failed: {e}"
            );
            eprintln!("skipping {what}: {e}");
            None
        }
    }
}

#[cfg(feature = "volume")]
mod volume_and_async {
    use super::*;
    use alice_sdf::compiled::GpuEvaluator;
    use alice_sdf::volume::bake::bake_volume;
    use alice_sdf::volume::gpu_bake::{gpu_bake_volume, gpu_bake_volume_with_normals};
    use alice_sdf::volume::BakeConfig;

    /// The bake's box, symmetric in its axes: its grid lands on the height map's face ties
    /// (|x| = |y| and so on), where the face changes. The law blends the faces there, so the field is
    /// continuous and every voxel is compared (a hard choice of face jumped by up to the amplitude
    /// at a tie, and implementations that rounded the comparison apart disagreed there)
    fn config(res: u32) -> BakeConfig {
        BakeConfig {
            resolution: [res, res, res],
            bounds_min: Vec3::splat(-1.4),
            bounds_max: Vec3::splat(1.4),
            ..Default::default()
        }
    }

    /// The number of grid points exactly on a face tie (so the test is known to cross them)
    fn ties(cfg: &BakeConfig) -> usize {
        let r = cfg.resolution;
        let mut n = 0;
        for z in 0..r[2] {
            for y in 0..r[1] {
                for x in 0..r[0] {
                    let a = node_pos(cfg, x, y, z).abs();
                    if a.x == a.y || a.y == a.z || a.x == a.z {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    /// The point of voxel (x, y, z): `bounds_min + i·size/(res−1)`, the bake's grid
    fn node_pos(cfg: &BakeConfig, x: u32, y: u32, z: u32) -> Vec3 {
        let size = cfg.bounds_max - cfg.bounds_min;
        let r = cfg.resolution;
        cfg.bounds_min
            + Vec3::new(
                x as f32 * size.x / (r[0] - 1) as f32,
                y as f32 * size.y / (r[1] - 1) as f32,
                z as f32 * size.z / (r[2] - 1) as f32,
            )
    }

    #[test]
    fn the_volume_bake_reads_the_node_data_and_agrees_with_the_cpu() {
        let node = data_tree();
        let cfg = config(24);
        let Some(gpu) = gpu_or_skip("gpu_bake_volume", gpu_bake_volume(&node, &cfg)) else {
            return;
        };
        assert_eq!(gpu.data.len(), 24 * 24 * 24, "the bake has every voxel");
        // the reference is the tree's own field at each grid point (`eval`); the CPU bake agrees too
        let cpu = bake_volume(&node, &cfg);
        let on_tie = ties(&cfg);
        assert!(on_tie > 100, "the grid crosses only {on_tie} face ties");
        let (mut worst, mut worst_cpu, mut compared) = (0.0f32, 0.0f32, 0usize);
        for z in 0..24 {
            for y in 0..24 {
                for x in 0..24 {
                    let p = node_pos(&cfg, x, y, z);
                    let want = eval(&node, p);
                    worst = worst.max((gpu.get(x, y, z) - want).abs());
                    worst_cpu = worst_cpu.max((cpu.get(x, y, z) - want).abs());
                    compared += 1;
                }
            }
        }
        eprintln!("gpu_bake_volume: {compared} voxels compared, {on_tie} of them on a face tie");
        assert_eq!(compared, 24 * 24 * 24, "every voxel is compared");
        assert!(
            worst_cpu < 1e-4,
            "the CPU bake is not the field here: {worst_cpu}"
        );
        eprintln!("gpu_bake_volume: worst |gpu - cpu| {worst:e} over {compared} voxels");
        assert!(worst < 1e-4, "gpu_bake_volume: worst |gpu - cpu| {worst}");
    }

    #[test]
    fn the_normal_bake_is_the_tetrahedral_difference_of_the_cpu_field() {
        let node = data_tree();
        let cfg = config(16);
        let eps = 1e-3f32;
        let Some(gpu) = gpu_or_skip(
            "gpu_bake_volume_with_normals",
            gpu_bake_volume_with_normals(&node, &cfg, eps),
        ) else {
            return;
        };
        // the bake's normal: normalize(Σ kᵢ · f(p + kᵢ·ε)) over the 4 tetrahedron directions
        let ks = [
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let on_tie = ties(&cfg);
        assert!(on_tie > 50, "the grid crosses only {on_tie} face ties");
        let (mut worst_d, mut worst_n, mut compared) = (0.0f32, 0.0f32, 0usize);
        let r = cfg.resolution;
        for z in 0..r[2] {
            for y in 0..r[1] {
                for x in 0..r[0] {
                    let p = node_pos(&cfg, x, y, z);
                    let v = gpu.get(x, y, z);
                    worst_d = worst_d.max((v.distance - eval(&node, p)).abs());
                    let g: Vec3 = ks.iter().map(|k| *k * eval(&node, p + *k * eps)).sum();
                    if g.length() > 1e-3 {
                        let n = Vec3::new(v.nx, v.ny, v.nz);
                        worst_n = worst_n.max((n - g.normalize()).length());
                        compared += 1;
                    }
                }
            }
        }
        eprintln!(
            "gpu_bake_volume_with_normals: worst distance {worst_d:e}, worst normal {worst_n:e} over {compared} voxels, {on_tie} grid points on a face tie"
        );
        assert!(compared > 0, "no normal compared");
        assert!(worst_d < 1e-4, "distance: worst |gpu - cpu| {worst_d}");
        // the 4 samples differ by f32 rounding of the field (~1e-7) over 2ε: ~1e-4 in the gradient
        assert!(worst_n < 1e-2, "normal: worst |gpu - cpu| {worst_n}");
    }

    #[test]
    fn the_async_evaluator_reads_the_node_data_and_agrees_with_the_cpu() {
        let node = data_tree();
        let Some(gpu) = gpu_or_skip(
            "GpuEvaluator::new_async",
            pollster::block_on(GpuEvaluator::new_async(&node)),
        ) else {
            return;
        };
        let pts: Vec<Vec3> = (0..512)
            .map(|i| {
                let t = i as f32 * 0.618_034;
                Vec3::new(
                    alice_det_math::sin(t * 1.7),
                    alice_det_math::cos(t * 2.3),
                    alice_det_math::sin(t * 0.9),
                ) * 1.2
            })
            .collect();
        let got = pollster::block_on(gpu.eval_batch_async(&pts)).expect("dispatch");
        let worst = pts
            .iter()
            .zip(&got)
            .map(|(p, g)| (eval(&node, *p) - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "new_async: worst |gpu - cpu| {worst:e} over {} points",
            pts.len()
        );
        assert_eq!(got.len(), pts.len());
        assert!(worst < 1e-4, "new_async: worst |gpu - cpu| {worst}");
    }
}

/// GPU marching cubes reads the tree's node data in the two passes that evaluate it (the grid and
/// the vertices, each at the slot past its own bindings): on a lattice off the height map's face
/// ties it gives the CPU marching cubes' index buffer, and the same vertex positions
#[cfg(feature = "gpu-mesh")]
#[test]
fn gpu_marching_cubes_reads_the_node_data_and_matches_the_cpu() {
    use alice_sdf::mesh::{
        gpu_marching_cubes, marching_cubes, GpuMarchingCubesConfig, MarchingCubesConfig,
    };
    let w = 70u32;
    let map: Vec<f32> = (0..w * w)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    let node = SdfNode::sphere(0.8)
        .heightmap_displacement(map, w, w, 0.05, 1.0)
        .union(SdfNode::polygon_2d(outline(40), 0.25).translate(0.9, 0.0, 0.0));
    let (lo, hi) = (
        Vec3::new(-1.0711, -0.9137, -0.8911),
        Vec3::new(1.6329, 0.9317, 0.9719),
    );
    let res = 40u32;
    let gpu = match gpu_marching_cubes(
        &node,
        lo,
        hi,
        &GpuMarchingCubesConfig {
            resolution: res,
            ..Default::default()
        },
    ) {
        Ok(m) => m,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but GPU marching cubes failed: {e}"
            );
            eprintln!("skipped ({e})");
            return;
        }
    };
    let cpu = marching_cubes(
        &node,
        lo,
        hi,
        &MarchingCubesConfig {
            resolution: res as usize,
            ..Default::default()
        },
    );
    assert!(!cpu.vertices.is_empty(), "the CPU mesh is empty");
    assert_eq!(gpu.vertices.len(), cpu.vertices.len(), "vertex count");
    assert_eq!(gpu.indices, cpu.indices, "index buffer");
    let worst = gpu
        .vertices
        .iter()
        .zip(&cpu.vertices)
        .map(|(g, c)| (g.position - c.position).length())
        .fold(0.0f32, f32::max);
    eprintln!(
        "gpu marching cubes: {} vertices, {} triangles, worst position {worst:e}",
        gpu.vertices.len(),
        gpu.indices.len() / 3
    );
    // a vertex is interpolated along its lattice edge from the two grid values, so a field that
    // differs by rounding (measured 4e-7 above) moves it by a small fraction of the cell: bound it
    // by a thousandth of the cell's shortest side
    let cell = ((hi - lo) / res as f32).min_element();
    assert!(
        worst < 1e-3 * cell,
        "worst |gpu - cpu| vertex position {worst} (cell {cell})"
    );
}
