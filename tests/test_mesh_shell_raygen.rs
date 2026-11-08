//! One ray-origin formula for the CPU and the GPU, and a consumer's own pipeline with the shell
//! kernel (red until `ShellCamera`, `ShellRenderer::camera_rays` and `ShellRenderer::first_hits_camera`
//! and the public no-fusion shader compile exist; the names are the proposed API).
//!
//! Today a consumer builds pixel-centre origins on the host in f64 and its GPU builds them in its
//! own order, and the two differ in their last bits on most rays, so a GPU-made ray and a host-made
//! ray are not the same ray. The formula is to be defined once in the crate (a fixed-order f32
//! expression), with its WGSL text, and used by both.
//!
//! oracle:
//! - `ShellCamera::ray(x, y)` (CPU) and the rays the GPU makes for the same camera are equal bit for
//!   bit on all 800 × 800 pixels (origins and directions)
//! - the hits of the GPU-made rays are the hits of the CPU-made rays on the GPU, bit for bit
//!   (`first_hits_camera` against `first_hits` of `ray(x, y)`)
//! - a pipeline built by a consumer from `SHELL_KERNEL` with the public no-fusion compile gives the
//!   hits `ShellRenderer::first_hits` gives, bit for bit
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

use alice_sdf::mesh::{MeshBvh, ShellBackend, ShellCamera, ShellRay, ShellRenderer};
use glam::Vec3;

#[path = "common/shell_scenes.rs"]
mod shell_scenes;

const R: f32 = 0.01;
const SIZE: u32 = 800;

/// An orthographic camera looking at the unit sphere from far away (the scale at which host and GPU
/// origins drifted apart)
fn camera() -> ShellCamera {
    ShellCamera::orthographic(
        Vec3::new(37.0, 11.0, 220.0),
        Vec3::new(-37.0, -11.0, -220.0),
        Vec3::Y,
        2.4,
        2.4,
        SIZE,
        SIZE,
    )
}

fn renderer_or_skip(bvh: &MeshBvh) -> Option<ShellRenderer<'_>> {
    let renderer = ShellRenderer::new(bvh, R).expect("renderer");
    if renderer.backend() == ShellBackend::Cpu {
        assert!(
            std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
            "ALICE_SDF_REQUIRE_GPU is set but no GPU device was found"
        );
        eprintln!("skipped (no GPU device)");
        return None;
    }
    Some(renderer)
}

fn bits(v: Vec3) -> [u32; 3] {
    v.to_array().map(f32::to_bits)
}

#[test]
fn the_gpu_makes_the_cpu_rays_bit_for_bit() {
    let (v, ix) = shell_scenes::sphere(48, 96);
    let bvh = MeshBvh::build(&v, &ix, 4);
    let Some(renderer) = renderer_or_skip(&bvh) else {
        return;
    };
    let cam = camera();
    let gpu = renderer.camera_rays(&cam).expect("gpu rays");
    assert_eq!(gpu.len(), (SIZE * SIZE) as usize);
    let mut differ = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            let cpu: ShellRay = cam.ray(x, y);
            let g = gpu[(y * SIZE + x) as usize];
            if bits(cpu.origin) != bits(g.origin) || bits(cpu.dir) != bits(g.dir) {
                differ.push((x, y));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} of {} pixels differ, first {:?}",
        differ.len(),
        SIZE * SIZE,
        &differ[..differ.len().min(8)]
    );
}

#[test]
fn hits_of_gpu_made_rays_are_hits_of_cpu_made_rays() {
    let (v, ix) = shell_scenes::sphere(48, 96);
    let bvh = MeshBvh::build(&v, &ix, 4);
    let Some(renderer) = renderer_or_skip(&bvh) else {
        return;
    };
    let cam = camera();
    let cpu_rays: Vec<ShellRay> = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .map(|(x, y)| cam.ray(x, y))
        .collect();
    let from_cpu = renderer.first_hits(&cpu_rays).expect("gpu");
    let from_gpu = renderer.first_hits_camera(&cam).expect("gpu");
    assert!(from_cpu.iter().filter(|h| h.is_some()).count() > 10_000);
    assert_eq!(from_cpu, from_gpu);
}

#[test]
fn a_consumer_pipeline_from_the_kernel_gives_the_renderer_hits() {
    let (v, ix) = shell_scenes::sphere(48, 96);
    let bvh = MeshBvh::build(&v, &ix, 4);
    let Some(renderer) = renderer_or_skip(&bvh) else {
        return;
    };
    let cam = camera();
    let rays: Vec<ShellRay> = (0..SIZE)
        .step_by(4)
        .flat_map(|y| (0..SIZE).step_by(4).map(move |x| (x, y)))
        .map(|(x, y)| cam.ray(x, y))
        .collect();
    let want = renderer.first_hits(&rays).expect("gpu");
    // the consumer's own module, compiled with the public no-fusion compile, on the renderer's
    // buffers
    let module = renderer
        .context()
        .shader_module_no_fusion("consumer shell", alice_sdf::mesh::gpu_bvh::SHELL_KERNEL)
        .expect("module");
    let got = renderer
        .first_hits_with_module(&module, &rays)
        .expect("gpu");
    assert_eq!(want, got);
}
