//! One ray, step by step: `ray_debug <V.bin> <T.bin> <ox oy oz dx dy dz end> <shell> <hit> <max>`. Marches on
//! the CPU and asks the GPU for the nearest triangle at each CPU step (kernel from MG_KERNEL, a point
//! kernel), printing where the two first disagree.
use alice_sdf::mesh::gpu_bvh::{flatten, Tracer};
use alice_sdf::mesh::MeshBvh;
use glam::Vec3;

fn f32s(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let v: Vec<Vec3> = f32s(&a[1])
        .chunks_exact(3)
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect();
    let t: Vec<u32> = std::fs::read(&a[2])
        .unwrap()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let n: Vec<f32> = a[3..10].iter().map(|x| x.parse().unwrap()).collect();
    let (shell, hit, max): (f32, f32, u32) = (
        a[10].parse().unwrap(),
        a[11].parse().unwrap(),
        a[12].parse().unwrap(),
    );
    let bvh = MeshBvh::build(&v, &t, 4);
    let flat = flatten(&bvh);
    let (o, d, end) = (
        Vec3::new(n[0], n[1], n[2]),
        Vec3::new(n[3], n[4], n[5]),
        n[6],
    );
    // the CPU march, keeping every point it evaluates
    let mut steps = Vec::new();
    let mut s = 0.0_f32;
    while s < end && (steps.len() as u32) < max {
        let p = o + d * s;
        let q = bvh.unsigned_distance(p) - shell;
        steps.push((s, p, q, bvh.closest_triangle(p).map(|c| c.0)));
        if q < hit {
            break;
        }
        s += q;
    }
    let rays: Vec<f32> = steps
        .iter()
        .flat_map(|(_, p, _, _)| [p.x, p.y, p.z, 0.0, 0.0, 0.0, 0.0])
        .collect();
    let tracer = Tracer::new(&flat, &rays, shell, hit, max, 4096).expect("kernel");
    let (gpu, _) = tracer.trace(steps.len() as u32).expect("trace");
    println!(
        "tris {} steps {} cpu last s {} q {}",
        bvh.triangles.len(),
        steps.len(),
        steps.last().unwrap().0,
        steps.last().unwrap().2
    );
    for (k, ((s, p, q, tri), g)) in steps.iter().zip(&gpu).enumerate() {
        let cpu_d = q + shell;
        let gpu_d = g.s;
        let flag = if cpu_d.to_bits() != gpu_d.to_bits() || tri.map(|x| x as u32) != Some(g.tri) {
            " <-- differs"
        } else {
            ""
        };
        if !flag.is_empty() || k + 3 >= steps.len() || k < 2 {
            println!(
                "step {k} s {s} p {p} cpu d {cpu_d} tri {tri:?} | gpu d {gpu_d} tri {}{flag}",
                g.tri
            );
        }
    }
    println!("ALL-DONE");
}
