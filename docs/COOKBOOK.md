# ALICE-SDF Cookbook

Practical recipes for common SDF tasks.

## Basic Shapes

```rust
use alice_sdf::prelude::*;

let sphere = SdfNode::sphere(1.0);
let box3d = SdfNode::box3d(1.0, 0.5, 0.5);
let cylinder = SdfNode::cylinder(0.5, 2.0);
let torus = SdfNode::torus(1.0, 0.3);
let plane = SdfNode::plane(0.0, 1.0, 0.0, 0.0); // Y-up
```

## CSG Operations

```rust
// Boolean
let union = a.clone().union(b.clone());
let intersection = a.clone().intersection(b.clone());
let subtraction = a.clone().subtract(b.clone());

// Smooth blending (k = blend radius)
let smooth = a.smooth_union(b, 0.3);
```

## Transforms

```rust
let moved = shape.translate(1.0, 2.0, 0.0);
let rotated = shape.rotate_y(std::f32::consts::FRAC_PI_4);
let scaled = shape.scale(2.0);
```

## Evaluate Distance

```rust
use alice_sdf::eval::eval;
use glam::Vec3;

let d = eval(&node, Vec3::new(1.0, 0.0, 0.0));
// d < 0: inside, d == 0: surface, d > 0: outside
```

## Compiled SDF (Faster Evaluation)

```rust
use alice_sdf::compiled::{CompiledSdf, eval_compiled};
use glam::Vec3;

let compiled = CompiledSdf::compile(&node);
let d = eval_compiled(&compiled, Vec3::new(1.0, 0.0, 0.0));
```

## SIMD Batch Evaluation (8 Points at Once)

```rust
use alice_sdf::compiled::eval_compiled_8wide;

let xs = [0.0_f32; 8];
let ys = [0.0; 8];
let zs = [1.0; 8];
let distances = eval_compiled_8wide(&compiled, &xs, &ys, &zs);
```

## GPU Evaluation (Millions of Points)

```rust
// Requires feature "gpu"
use alice_sdf::compiled::GpuEvaluator;
use glam::Vec3;

let gpu = GpuEvaluator::new(&node)?;
let points: Vec<Vec3> = /* your points */;
let distances = gpu.eval_batch(&points)?;
```

## Mesh Generation (Marching Cubes)

```rust
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig};
use glam::Vec3;

let config = MarchingCubesConfig {
    resolution: 128,
    ..Default::default()
};
let mesh = sdf_to_mesh(&node, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
println!("Vertices: {}, Triangles: {}", mesh.vertices.len(), mesh.indices.len() / 3);
```

## Shader Transpilation

```rust
// WGSL (WebGPU) — requires feature "gpu"
use alice_sdf::compiled::{WgslShader, TranspileMode};
let wgsl = WgslShader::transpile(&node, TranspileMode::Hardcoded);
println!("{}", wgsl.code);

// HLSL (DirectX/Unreal) — requires feature "hlsl"
use alice_sdf::compiled::{HlslShader, HlslTranspileMode};
let hlsl = HlslShader::transpile(&node, HlslTranspileMode::Hardcoded);

// GLSL (OpenGL/Unity) — requires feature "glsl"
use alice_sdf::compiled::{GlslShader, GlslTranspileMode};
let glsl = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
```

## Materials (PBR)

```rust
use alice_sdf::material::{Material, StandardMaterials, material_lerp};

let gold = StandardMaterials::gold();
let chrome = StandardMaterials::chrome();
let blend = material_lerp(&gold, &chrome, 0.5);
```

## Collision Detection

```rust
use alice_sdf::collision::{sdf_contact_points, compute_manifold, sdf_ccd};
use glam::Vec3;

// Contact points between two SDF shapes
let contacts = sdf_contact_points(&node_a, &node_b, resolution);

// Contact manifold
let manifold = compute_manifold(&contacts);

// Continuous collision detection (ray vs SDF)
let toi = sdf_ccd(&node, Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 10.0);
```

## Automatic Differentiation

```rust
use alice_sdf::autodiff::{eval_with_gradient, principal_curvatures, gaussian_curvature};
use glam::Vec3;

let (distance, gradient) = eval_with_gradient(&node, Vec3::new(1.0, 0.0, 0.0));
let (k1, k2) = principal_curvatures(&node, Vec3::new(1.0, 0.0, 0.0));
let kg = gaussian_curvature(&node, Vec3::new(1.0, 0.0, 0.0));
```

## Animation / Morph

```rust
use alice_sdf::animation::{morph, Timeline, Track, Keyframe};

// Morph between two shapes
let morphed = morph(&sphere, &box3d, 0.5); // 50% blend

// Keyframe animation
let track = Track::new("blend")
    .add_keyframe(Keyframe::new(0.0, 0.0))
    .add_keyframe(Keyframe::new(1.0, 1.0));
let timeline = Timeline::new().add_track(track);
```

## JSON Serialization

```rust
// Save
let json = serde_json::to_string_pretty(&node).unwrap();
std::fs::write("shape.asdf.json", &json).unwrap();

// Load
let json = std::fs::read_to_string("shape.asdf.json").unwrap();
let node: SdfNode = serde_json::from_str(&json).unwrap();
```

## Environment Rendering (RenderConfig)

Generate a complete fragment shader with environment (sky, ground, weather, destruction) from an SDF scene. Requires feature `glsl`.

### Basic: SDF Scene with Full Environment

```rust
use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode, RenderConfig};

let node = SdfNode::sphere(1.0).translate(0.0, 1.0, 0.0)
    .union(SdfNode::plane(0.0, 1.0, 0.0, 0.0));

let glsl = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
let config = RenderConfig::default(); // PBR + sky + weather + day/night + post-process
let fragment_shader = glsl.to_fragment_shader_full(&config);
// → Complete GLSL ES 300 fragment shader ready for WebGL2
```

### RenderConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_steps` | u32 | 128 | Raymarching iteration limit |
| `max_distance` | f32 | 200.0 | Raymarching termination distance |
| `day_night_cycle` | bool | true | Sun angle from `uDayPhase` (0-1 = 24h) |
| `weather_system` | bool | true | Fog + rain + lightning (uniforms: `uWxFog`, `uWxRain`, `uLightning`) |
| `ssr_enabled` | bool | true | Screen-space reflection on floor (16-step) |
| `volumetric_light` | bool | true | God rays + volumetric scatter (8-step) |
| `post_process` | bool | true | ACES tonemap + bloom + CA + vignette + grain + color grading |
| `biome_terrain` | bool | false | Voronoi erosion + biome phase transition terrain |
| `glsl_version` | u32 | 300 | GLSL ES version |
| `spectral_rendering` | bool | false | Planck blackbody + CIE 1931 + λ^-4 Rayleigh scattering |
| `material_slots` | u32 | 1 | 1 = single material, 2+ = multi-material (`sdf_eval` returns `vec2`) |
| `destruction` | bool | false | Voronoi cracking + debris + meteor + shockwave |
| `vfx_effects` | bool | false | Domain warp + DBM discharge + analytic bloom |
| `interior_mapping` | bool | false | Pseudo-rooms behind glass walls |
| `micro_normal` | bool | false | Nanoscale surface detail via vnoise3 |
| `dual_sdf` | bool | false | `sdf_eval_lite()` for AO/shadow (performance) |

### Ground + Sky (Minimal)

```rust
let config = RenderConfig::default();
// Generates: Rayleigh/Mie sky, sun/moon/stars, dual-layer clouds,
// PBR Cook-Torrance with shadow + AO, fog, rain, ACES tonemap
```

The ground is defined by your SDF scene. Use a plane:
```rust
let ground = SdfNode::plane(0.0, 1.0, 0.0, 0.0); // Y=0 infinite ground
let scene = ground.union(your_objects);
```

### Metaverse-Grade Environment

```rust
let config = RenderConfig {
    spectral_rendering: true,  // physical blackbody + spectral sky
    material_slots: 8,         // multi-material scene
    destruction: true,          // Voronoi cracking, debris, meteor
    vfx_effects: true,          // domain warp plasma, DBM discharge, bloom
    interior_mapping: true,     // pseudo-rooms behind glass
    micro_normal: true,         // nanoscale surface detail
    dual_sdf: true,             // map_lite for AO/shadow performance
    max_steps: 64,              // TDR-safe for Windows
    ..Default::default()
};
```

When `destruction: true`, these uniforms become available:
- `uEntropy` (0-1), `uShatter` (0-1), `uMeteorY`, `uMeteorActive`, `uMeteorImpact` (vec2), `uImpact`, `uImpactRing`, `uShake` (vec2), `uTimeDilation`

### Multi-Material Scene

When `material_slots > 1`, your `sdf_eval` must return `vec2(distance, material_id)` and you must provide a `getMat(float id, vec3 p)` function in your SDF source:

```rust
let sdf_source = r#"
vec2 sdf_eval(vec3 p) {
    float ground = p.y;
    float orb = length(p - vec3(0,2,0)) - 1.0;
    float id = step(orb, ground); // 0 = ground, 1 = orb
    return vec2(min(ground, orb), id);
}
Mat getMat(float id, vec3 p) {
    Mat m; m.emission = vec3(0); m.sss = 0.0;
    if (id < 0.5) { // ground
        m.albedo = vec3(0.03); m.metallic = 0.08; m.roughness = 0.18;
    } else { // orb
        m.albedo = vec3(0.9, 0.7, 0.2); m.metallic = 0.95; m.roughness = 0.02;
    }
    return m;
}
"#;
let config = RenderConfig { material_slots: 2, ..Default::default() };
let shader = render_pipeline::build_full_shader(sdf_source, &config);
```

### Dual SDF (Performance)

When `dual_sdf: true`, provide `sdf_eval_lite()` — a simplified SDF used only for AO, shadow, and rain occlusion. Omit expensive objects (energy orbs, animated rings, debris):

```rust
let sdf_source = r#"
float sdf_eval(vec3 p) { /* full scene */ }
float sdf_eval_lite(vec3 p) { /* ground + static structures only */ }
"#;
```

### Uniform Interface

The generated shader expects these uniforms from your host application:

| Uniform | Type | Description |
|---------|------|-------------|
| `uRes` | vec2 | Viewport resolution |
| `uTime` | float | Elapsed time (seconds) |
| `uCamPos` | vec3 | Camera position |
| `uCamFwd` | vec3 | Camera forward vector |
| `uCamRight` | vec3 | Camera right vector |
| `uCamUp` | vec3 | Camera up vector |
| `uDayPhase` | float | Time of day (0.0 = midnight, 0.5 = noon) |
| `uMaxDist` | float | Raymarching max distance |
| `uWxFog` | float | Fog intensity (0-1) |
| `uWxRain` | float | Rain intensity (0-1) |
| `uLightning` | float | Lightning flash (0-1, fast decay) |

## Physics Integration (ALICE-Physics)

The destruction uniforms in `RenderConfig { destruction: true }` can be driven by ALICE-Physics simulation. The physics engine runs thermal, fracture, erosion, and pressure simulations on the CPU, and the application layer bridges the data to the GPU shader.

### Uniform Mapping

| Physics Module | Method | Shader Uniform | Description |
|---------------|--------|----------------|-------------|
| `ThermalModifier` | `temperature.sample(pos)` | `uShatter` | Melt threshold → destruction intensity |
| `ThermalModifier` | `melt_accumulator.sample(pos)` | `uEntropy` | Cumulative material loss (0-1) |
| `FractureModifier` | `stress_field.sample(pos)` | `uShatter` | Crack propagation |
| `PressureModifier` | `pressure_field.sample(pos)` | `uImpact` | Contact deformation |
| `ForceField::Explosion` | `center`, `radius` | `uMeteorImpact`, `uImpactRing` | Blast center and radius |

### Example

```rust
use alice_physics::{ThermalModifier, ThermalConfig, HeatSource};

let mut thermal = ThermalModifier::new(ThermalConfig {
    melt_temperature: 800.0,
    melt_rate: 0.01,
    ..Default::default()
});
thermal.add_heat_source(HeatSource::point(impact_pos, 2000.0));

// Each frame: step physics, sample fields, upload uniforms
thermal.step(&mut world, dt);
let shatter = thermal.melt_accumulator.sample(pos).min(1.0);
gl.uniform1f(u_shatter_loc, shatter);
```

See [ALICE-Physics README](../../ALICE-Physics/README.md#rendering-pipeline-integration-alice-sdf-v140) for the full architecture.

## Feature Flags

| Feature | Description |
|---------|-------------|
| (default) | Core SDF eval, mesh, compiled VM, SIMD |
| `jit` | Cranelift JIT compilation |
| `gpu` | WebGPU evaluation + WGSL transpiler |
| `hlsl` | HLSL shader transpiler |
| `glsl` | GLSL shader transpiler |
| `ffi` | C/C++/C# FFI bindings |
| `unity` | Unity integration (FFI + GLSL) |
| `unreal` | Unreal Engine integration (FFI + HLSL) |
| `aaa` | All premium features |
