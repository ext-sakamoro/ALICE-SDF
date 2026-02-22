//! ALICE-SDF CLI
//!
//! Command-line interface for SDF operations.
//!
//! Author: Moroya Sakamoto

#![allow(
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::let_unit_value,
    clippy::format_push_string,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::ignored_unit_patterns,
)]

use alice_sdf::io::{export_3mf, export_ply, export_stl, PlyConfig};
use alice_sdf::prelude::*;
use std::path::PathBuf;

#[cfg(feature = "cli")]
use clap::{Parser, Subcommand};

// Deep Fried Edition: JIT + Rayon imports
#[cfg(all(feature = "cli", feature = "jit"))]
use alice_sdf::compiled::jit::JitSimdSdf;
#[cfg(feature = "cli")]
#[allow(unused_imports)]
use rayon::prelude::*;

#[cfg(feature = "cli")]
#[derive(Parser)]
#[command(name = "alice-sdf")]
#[command(author = "Moroya Sakamoto")]
#[command(version = alice_sdf::VERSION)]
#[command(about = "ALICE-SDF: 3D/Spatial Data Specialist", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[cfg(feature = "cli")]
#[derive(Subcommand)]
enum Commands {
    /// Display file information
    Info {
        /// Input file (.asdf or .asdf.json)
        file: PathBuf,
    },

    /// Convert between formats
    Convert {
        /// Input file
        input: PathBuf,
        /// Output file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Convert SDF to mesh (OBJ format)
    ToMesh {
        /// Input SDF file
        input: PathBuf,
        /// Output OBJ file
        #[arg(short, long)]
        output: PathBuf,
        /// Grid resolution
        #[arg(short, long, default_value = "64")]
        resolution: usize,
        /// Bounding box size
        #[arg(short, long, default_value = "2.0")]
        bounds: f32,
    },

    /// Create a demo SDF file
    Demo {
        /// Output file
        #[arg(short, long, default_value = "demo.asdf")]
        output: PathBuf,
    },

    /// Benchmark SDF evaluation
    Bench {
        /// Input file (or uses demo shape)
        file: Option<PathBuf>,
        /// Number of points to evaluate
        #[arg(short, long, default_value = "1000000")]
        points: usize,
    },

    /// Export SDF to mesh file (auto-detect format from extension)
    Export {
        /// Input SDF file (.asdf or .asdf.json)
        input: PathBuf,
        /// Output mesh file (.obj, .glb, .fbx, .usda, .abc)
        #[arg(short, long)]
        output: PathBuf,
        /// Grid resolution for marching cubes
        #[arg(short, long, default_value = "64")]
        resolution: usize,
        /// Bounding box half-size
        #[arg(short, long, default_value = "2.0")]
        bounds: f32,
    },

    /// Generate 3D-printable mesh from SDF (STL/3MF output)
    Print {
        /// Input SDF file (.asdf or .asdf.json)
        input: PathBuf,
        /// Output mesh file (.3mf, .stl, .obj)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Grid resolution for marching cubes (higher = finer detail)
        #[arg(short, long, default_value = "128")]
        resolution: usize,
        /// Bounding box half-size (auto-detect if omitted)
        #[arg(short, long)]
        bounds: Option<f32>,
    },

    /// Fit a texture image to procedural noise formula
    #[cfg(feature = "texture-fit")]
    TextureFit {
        /// Input image file (PNG/JPG)
        input: PathBuf,
        /// Maximum octaves
        #[arg(long, default_value = "8")]
        octaves: u32,
        /// Target PSNR (dB)
        #[arg(long, default_value = "28.0")]
        target_psnr: f32,
        /// Iterations per octave
        #[arg(long, default_value = "500")]
        iterations: u32,
        /// Output JSON file for parameters
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Shader language: wgsl, hlsl, or glsl
        #[arg(long)]
        shader: Option<String>,
        /// Shader output file
        #[arg(long)]
        shader_output: Option<PathBuf>,
    },
}

#[cfg(feature = "cli")]
fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { file } => cmd_info(file),
        Commands::Convert { input, output } => cmd_convert(input, output),
        Commands::ToMesh {
            input,
            output,
            resolution,
            bounds,
        } => cmd_to_mesh(input, output, resolution, bounds),
        Commands::Demo { output } => cmd_demo(output),
        Commands::Export {
            input,
            output,
            resolution,
            bounds,
        } => cmd_export(input, output, resolution, bounds),
        Commands::Print {
            input,
            output,
            resolution,
            bounds,
        } => cmd_print(input, output, resolution, bounds),
        Commands::Bench { file, points } => cmd_bench(file, points),
        #[cfg(feature = "texture-fit")]
        Commands::TextureFit {
            input,
            octaves,
            target_psnr,
            iterations,
            output,
            shader,
            shader_output,
        } => cmd_texture_fit(
            input,
            octaves,
            target_psnr,
            iterations,
            output,
            shader,
            shader_output,
        ),
    }
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("CLI not enabled. Build with --features cli");
    std::process::exit(1);
}

#[cfg(feature = "cli")]
fn cmd_info(path: PathBuf) {
    match get_info(&path) {
        Ok(info) => println!("{}", info),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cli")]
fn cmd_convert(input: PathBuf, output: PathBuf) {
    match load(&input) {
        Ok(tree) => match save(&tree, &output) {
            Ok(_) => println!("Converted {} -> {}", input.display(), output.display()),
            Err(e) => {
                eprintln!("Save error: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Load error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cli")]
fn cmd_to_mesh(input: PathBuf, output: PathBuf, resolution: usize, bounds: f32) {
    let tree = match load(&input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Load error: {}", e);
            std::process::exit(1);
        }
    };

    let config = MarchingCubesConfig {
        resolution,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };

    let min = Vec3::splat(-bounds);
    let max = Vec3::splat(bounds);

    println!("Generating mesh with resolution {}...", resolution);
    let mesh = sdf_to_mesh(&tree.root, min, max, &config);

    // Write OBJ file
    let mut obj = String::new();
    obj.push_str("# Generated by ALICE-SDF\n");
    obj.push_str(&format!("# Vertices: {}\n", mesh.vertex_count()));
    obj.push_str(&format!("# Triangles: {}\n\n", mesh.triangle_count()));

    for v in &mesh.vertices {
        obj.push_str(&format!(
            "v {} {} {}\n",
            v.position.x, v.position.y, v.position.z
        ));
    }

    obj.push('\n');

    for v in &mesh.vertices {
        obj.push_str(&format!(
            "vn {} {} {}\n",
            v.normal.x, v.normal.y, v.normal.z
        ));
    }

    obj.push('\n');

    for chunk in mesh.indices.chunks(3) {
        if chunk.len() == 3 {
            let a = chunk[0] + 1;
            let b = chunk[1] + 1;
            let c = chunk[2] + 1;
            obj.push_str(&format!("f {}//{} {}//{} {}//{}\n", a, a, b, b, c, c));
        }
    }

    match std::fs::write(&output, obj) {
        Ok(_) => println!(
            "Saved {} vertices, {} triangles to {}",
            mesh.vertex_count(),
            mesh.triangle_count(),
            output.display()
        ),
        Err(e) => {
            eprintln!("Write error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cli")]
fn cmd_export(input: PathBuf, output: PathBuf, resolution: usize, bounds: f32) {
    let tree = match load(&input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Load error: {}", e);
            std::process::exit(1);
        }
    };

    let config = MarchingCubesConfig {
        resolution,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };

    let min = Vec3::splat(-bounds);
    let max = Vec3::splat(bounds);

    println!("Generating mesh with resolution {}...", resolution);
    let mesh = sdf_to_mesh(&tree.root, min, max, &config);
    println!(
        "  {} vertices, {} triangles",
        mesh.vertex_count(),
        mesh.triangle_count()
    );

    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let result = match ext.as_str() {
        "obj" => export_obj(&mesh, &output, &ObjConfig::default(), None),
        "glb" => export_glb(&mesh, &output, &GltfConfig::default(), None),
        "fbx" => export_fbx(&mesh, &output, &FbxConfig::default(), None),
        "usda" | "usd" => export_usda(&mesh, &output, &UsdConfig::default(), None),
        "abc" => export_alembic(&mesh, &output, &AlembicConfig::default()),
        "stl" => export_stl(&mesh, &output),
        "3mf" => export_3mf(&mesh, &output),
        "ply" => export_ply(&mesh, &output, &PlyConfig::default()),
        _ => {
            eprintln!("Unsupported format: .{}", ext);
            eprintln!("Supported: .obj, .glb, .fbx, .usda, .abc, .stl, .3mf, .ply");
            std::process::exit(1);
        }
    };

    match result {
        Ok(_) => println!("Exported to {}", output.display()),
        Err(e) => {
            eprintln!("Export error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cli")]
fn cmd_print(
    input: PathBuf,
    output: Option<PathBuf>,
    resolution: usize,
    bounds_override: Option<f32>,
) {
    use alice_sdf::mesh::sdf_to_mesh_compiled;

    let total_start = std::time::Instant::now();

    let tree = match load(&input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Load error: {}", e);
            std::process::exit(1);
        }
    };

    let node_count = tree.node_count();
    println!("Loaded SDF: {} nodes", node_count);

    // Compile SDF for faster mesh generation (SIMD batch eval)
    let compile_start = std::time::Instant::now();
    let compiled = CompiledSdf::compile(&tree.root);
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

    // Auto-detect bounds from tight AABB if not specified
    let (min, max) = if let Some(b) = bounds_override {
        (Vec3::splat(-b), Vec3::splat(b))
    } else {
        // Use large initial search for 3D printing scale (up to 500mm)
        let aabb_config = alice_sdf::tight_aabb::TightAabbConfig {
            initial_half_size: 500.0,
            bisection_iterations: 24,
            coarse_subdivisions: 16,
        };
        let aabb = alice_sdf::tight_aabb::compute_tight_aabb_with_config(&tree.root, &aabb_config);
        if aabb.min == Vec3::ZERO && aabb.max == Vec3::ZERO {
            eprintln!("Warning: Could not detect bounds. Using default [-200, 200]³");
            (Vec3::splat(-200.0), Vec3::splat(200.0))
        } else {
            let center = (aabb.min + aabb.max) * 0.5;
            let half = (aabb.max - aabb.min) * 0.5;
            let max_extent = half.x.max(half.y).max(half.z);
            let padded = max_extent * 1.1;
            (center - Vec3::splat(padded), center + Vec3::splat(padded))
        }
    };

    let config = MarchingCubesConfig {
        resolution,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };

    println!(
        "Generating mesh: resolution={}, bounds=[{:.1}, {:.1}, {:.1}] to [{:.1}, {:.1}, {:.1}]",
        resolution, min.x, min.y, min.z, max.x, max.y, max.z
    );

    // Use compiled SDF for faster marching cubes (SIMD 8-wide batch evaluation)
    let mesh_start = std::time::Instant::now();
    let mesh = sdf_to_mesh_compiled(&compiled, min, max, &config);
    let mesh_ms = mesh_start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  {} vertices, {} triangles (compile: {:.1}ms, mesh: {:.1}ms)",
        mesh.vertex_count(),
        mesh.triangle_count(),
        compile_ms,
        mesh_ms
    );

    // Determine output path and format
    let out_path = output.unwrap_or_else(|| {
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        PathBuf::from(format!("{}.3mf", stem))
    });

    let ext = out_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("3mf")
        .to_lowercase();

    let export_start = std::time::Instant::now();
    let result = match ext.as_str() {
        "3mf" => export_3mf(&mesh, &out_path),
        "stl" => export_stl(&mesh, &out_path),
        "obj" => export_obj(&mesh, &out_path, &ObjConfig::default(), None),
        "ply" => export_ply(&mesh, &out_path, &PlyConfig::default()),
        _ => {
            eprintln!("Unsupported print format: .{}", ext);
            eprintln!("Supported: .3mf (recommended), .stl, .obj, .ply");
            std::process::exit(1);
        }
    };
    let export_ms = export_start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(_) => {
            let file_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "Exported to {} ({:.1} KB, export: {:.1}ms, total: {:.1}ms)",
                out_path.display(),
                file_size as f64 / 1024.0,
                export_ms,
                total_ms
            );
            println!("Ready for slicing (Bambu Studio / Orca Slicer)");
        }
        Err(e) => {
            eprintln!("Export error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cli")]
fn cmd_demo(output: PathBuf) {
    let shape = SdfNode::sphere(1.0)
        .smooth_subtract(SdfNode::box3d(1.2, 1.2, 1.2), 0.1)
        .smooth_union(
            SdfNode::cylinder(0.3, 2.0).rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0),
            0.1,
        )
        .smooth_union(SdfNode::cylinder(0.3, 2.0), 0.1)
        .smooth_union(
            SdfNode::cylinder(0.3, 2.0).rotate_euler(0.0, 0.0, std::f32::consts::FRAC_PI_2),
            0.1,
        );

    let tree = SdfTree::new(shape);

    match save(&tree, &output) {
        Ok(_) => println!(
            "Created demo SDF with {} nodes at {}",
            tree.node_count(),
            output.display()
        ),
        Err(e) => {
            eprintln!("Save error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Deep Fried Crispy Edition Benchmark
///
/// Uses JIT + SoA + Rayon for maximum throughput with GPU comparison:
/// - JIT: Native SIMD machine code (no interpreter overhead)
/// - SoA: Structure of Arrays for optimal SIMD loads
/// - Rayon: Multi-threaded parallel execution
/// - GPU: WebGPU compute shader for comparison
#[cfg(all(feature = "cli", feature = "jit"))]
fn cmd_bench(file: Option<PathBuf>, points: usize) {
    // 1. Load or create SDF node
    let node = if let Some(path) = file {
        match load(&path) {
            Ok(tree) => tree.root,
            Err(e) => {
                eprintln!("Load error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Demo shape (complex enough to stress test)
        SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(1.0, 1.0, 1.0), 0.2)
            .twist(0.5)
    };

    println!("=== Deep Fried Crispy Edition Benchmark ===");
    println!("Points: {}", points);
    println!("Node count: {}", node.node_count());
    println!("Threads: {}", rayon::current_num_threads());

    // 2. Compile all backends
    println!("\n--- Compilation Phase ---");
    let compile_start = std::time::Instant::now();
    let compiled_sdf = CompiledSdf::compile(&node);
    println!(
        "Stack VM: {:.3}ms",
        compile_start.elapsed().as_secs_f64() * 1000.0
    );

    let jit_start = std::time::Instant::now();
    let jit = match JitSimdSdf::compile(&compiled_sdf) {
        Ok(jit) => jit,
        Err(e) => {
            eprintln!("JIT Compile error: {}", e);
            std::process::exit(1);
        }
    };
    println!(
        "JIT SIMD : {:.3}ms",
        jit_start.elapsed().as_secs_f64() * 1000.0
    );

    // GPU compilation (optional)
    #[cfg(feature = "gpu")]
    let gpu = {
        let gpu_start = std::time::Instant::now();
        match alice_sdf::compiled::GpuEvaluator::new(&node) {
            Ok(g) => {
                println!(
                    "GPU WGSL : {:.3}ms",
                    gpu_start.elapsed().as_secs_f64() * 1000.0
                );
                Some(g)
            }
            Err(e) => {
                println!("GPU WGSL : unavailable ({:?})", e);
                None
            }
        }
    };

    // 3. Generate test data in SoA layout
    println!("\n--- Data Generation ---");
    let data_start = std::time::Instant::now();
    let mut px = vec![0.0f32; points];
    let mut py = vec![0.0f32; points];
    let mut pz = vec![0.0f32; points];

    px.par_iter_mut()
        .zip(py.par_iter_mut())
        .zip(pz.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((x, y), z))| {
            let t = i as f32 / points as f32;
            *x = (t * 1234.567).sin() * 50.0;
            *y = (t * 2345.678).cos() * 50.0;
            *z = (t * 3456.789).sin() * 50.0;
        });
    println!(
        "SoA data : {:.3}ms",
        data_start.elapsed().as_secs_f64() * 1000.0
    );

    // Also prepare AoS for GPU
    let points_vec: Vec<Vec3> = (0..points)
        .map(|i| Vec3::new(px[i], py[i], pz[i]))
        .collect();

    // 4. Benchmark all modes
    println!("\n--- Benchmark Results ---");
    println!(
        "{:<20} {:>12} {:>15} {:>12}",
        "Mode", "Time (ms)", "Throughput", "ns/point"
    );
    println!("{}", "-".repeat(62));

    // CPU Scalar (interpreted)
    let start = std::time::Instant::now();
    let _results_scalar = eval_batch_parallel(&node, &points_vec);
    let elapsed = start.elapsed();
    print_result("CPU Scalar", elapsed, points);

    // CPU SIMD (stack VM)
    let start = std::time::Instant::now();
    let _results_simd =
        alice_sdf::compiled::eval_compiled_batch_simd_parallel(&compiled_sdf, &points_vec);
    let elapsed = start.elapsed();
    print_result("CPU SIMD (VM)", elapsed, points);

    // CPU JIT SIMD
    let start = std::time::Instant::now();
    let num_threads = rayon::current_num_threads();
    let chunk_size = (points + num_threads - 1) / num_threads;
    let _results_jit: Vec<f32> = px
        .par_chunks(chunk_size)
        .zip(py.par_chunks(chunk_size))
        .zip(pz.par_chunks(chunk_size))
        .flat_map(|((cx, cy), cz)| jit.eval_batch(cx, cy, cz))
        .collect();
    let elapsed = start.elapsed();
    print_result("CPU JIT SIMD", elapsed, points);

    // GPU Compute
    #[cfg(feature = "gpu")]
    if let Some(ref gpu) = gpu {
        let start = std::time::Instant::now();
        let _results_gpu = gpu.eval_batch(&points_vec);
        let elapsed = start.elapsed();
        print_result("GPU Compute", elapsed, points);
    }

    println!("{}", "-".repeat(62));
}

#[cfg(feature = "cli")]
#[allow(dead_code)]
fn print_result(mode: &str, elapsed: std::time::Duration, points: usize) {
    let seconds = elapsed.as_secs_f64();
    let throughput = points as f64 / seconds / 1_000_000.0;
    let ns_per_point = seconds * 1_000_000_000.0 / points as f64;
    println!(
        "{:<20} {:>10.3}ms {:>12.2} M/s {:>10.2} ns",
        mode,
        seconds * 1000.0,
        throughput,
        ns_per_point
    );
}

/// Fallback benchmark (no JIT feature)
#[cfg(all(feature = "cli", not(feature = "jit")))]
fn cmd_bench(file: Option<PathBuf>, points: usize) {
    let node = if let Some(path) = file {
        match load(&path) {
            Ok(tree) => tree.root,
            Err(e) => {
                eprintln!("Load error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Demo shape
        SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(1.0, 1.0, 1.0), 0.2)
            .twist(0.5)
    };

    println!("=== Interpreter Benchmark (JIT not enabled) ===");
    println!("Hint: Build with --features jit for 20x+ speedup");
    println!("Points: {}", points);
    println!("Node count: {}", node.node_count());

    // Generate random points
    let test_points: Vec<Vec3> = (0..points)
        .map(|i| {
            let t = i as f32 / points as f32;
            Vec3::new(
                (t * 123.456).sin() * 2.0,
                (t * 234.567).sin() * 2.0,
                (t * 345.678).sin() * 2.0,
            )
        })
        .collect();

    // Benchmark
    let start = std::time::Instant::now();
    let _results = eval_batch_parallel(&node, &test_points);
    let elapsed = start.elapsed();

    let points_per_sec = points as f64 / elapsed.as_secs_f64();

    println!("\n--------------------------------------------------");
    println!("Time       : {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "Throughput : {:.2} M points/sec",
        points_per_sec / 1_000_000.0
    );
    println!("--------------------------------------------------");
}

#[cfg(all(feature = "cli", feature = "texture-fit"))]
fn cmd_texture_fit(
    input: PathBuf,
    octaves: u32,
    target_psnr: f32,
    iterations: u32,
    output: Option<PathBuf>,
    shader: Option<String>,
    shader_output: Option<PathBuf>,
) {
    use alice_sdf::texture::{fit_texture, generate_shader, ShaderLanguage, TextureFitConfig};

    let config = TextureFitConfig {
        max_octaves: octaves,
        target_psnr_db: target_psnr,
        iterations_per_octave: iterations,
        tileable: true,
    };

    println!("Fitting texture: {}", input.display());
    println!("  Max octaves: {}", octaves);
    println!("  Target PSNR: {:.1} dB", target_psnr);
    println!("  Iterations/octave: {}", iterations);

    let start = std::time::Instant::now();
    let result = match fit_texture(&input, &config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    let total_octaves: usize = result.octaves.iter().map(|o| o.len()).sum();
    println!("\nResult:");
    println!("  Image: {}x{}", result.width, result.height);
    println!("  Octaves: {}", total_octaves);
    println!("  PSNR: {:.2} dB", result.psnr_db);
    println!("  NMSE: {:.4}", result.nmse);
    println!("  Time: {:.3}s", elapsed.as_secs_f64());

    // Save JSON
    if let Some(out_path) = output {
        let json = serde_json::to_string_pretty(&result).unwrap();
        match std::fs::write(&out_path, json) {
            Ok(_) => println!("  Saved parameters to {}", out_path.display()),
            Err(e) => eprintln!("  Failed to save JSON: {}", e),
        }
    }

    // Generate shader
    if let Some(lang_str) = shader {
        let lang: ShaderLanguage = match lang_str.parse() {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        };

        let source_name = input
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let shader_code = generate_shader(&result, lang, &source_name);

        if let Some(shader_path) = shader_output {
            match std::fs::write(&shader_path, &shader_code) {
                Ok(_) => println!("  Saved shader to {}", shader_path.display()),
                Err(e) => eprintln!("  Failed to save shader: {}", e),
            }
        } else {
            println!("\n--- Generated Shader ({:?}) ---", lang);
            println!("{}", shader_code);
        }
    }
}
