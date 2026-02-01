# SDF Universe - "5MB Procedural Universe" Demo

> **The Future of Real-time Graphics: Mathematical Surfaces, Infinite Resolution**

![ALICE-SDF](https://img.shields.io/badge/ALICE--SDF-Deep%20Fried-orange)
![Particles](https://img.shields.io/badge/Particles-10M%2B-blue)
![FPS](https://img.shields.io/badge/FPS-60%2B-green)
![GPU](https://img.shields.io/badge/GPU-Compute%20Shader-purple)

## What is This?

A Unity demo that creates an **entire procedural universe** using only **5MB** of code.

- No meshes
- No textures
- No baking
- Just pure mathematics

## The Magic

```
Traditional Game Engine        ALICE-SDF
═══════════════════════        ═════════════════════
15-30 GB install               5 MB library
850 MB for 1M shapes           120 MB for 1M shapes
Polygon resolution limit       INFINITE resolution
12 FPS during morphing         60 FPS during morphing
Cannot do infinite repeat      Infinite repeat possible
Pre-baked boolean ops          Real-time boolean ops
```

## Architecture

Three rendering modes with increasing performance:

| Mode | Pipeline | Performance | Use Case |
|------|----------|-------------|----------|
| **CPU Standard** | Parallel.For + EvalGradientSoA | 100K particles | Debugging, compatibility |
| **CPU Burst** | Rust SIMD + Burst Jobs + Indirect | 500K particles | CPU-bound systems |
| **GPU Compute** | Full GPU Compute Shader | **10M+ particles** | Maximum performance |

```
Performance Stack:

┌─────────────────────────────────────────────────────────┐
│  GPU Compute Shader (God Tier)                         │
│  └─ SDF Eval + Physics + Render = ALL ON GPU           │
│     └─ CPU Transfer: ZERO                              │
├─────────────────────────────────────────────────────────┤
│  CPU Burst + DrawMeshInstancedIndirect                 │
│  └─ Rust SIMD Eval + Burst Physics                     │
│     └─ 24 bytes/particle (vs 64 bytes Matrix4x4)       │
├─────────────────────────────────────────────────────────┤
│  CPU Standard                                          │
│  └─ Parallel.For + EvalGradientSoA                     │
└─────────────────────────────────────────────────────────┘
```

## GPU Scene Variants

Four pre-built compute shader scenes:

| Scene | Description | Key Features |
|-------|-------------|--------------|
| **Cosmic** | Solar System | Sun + Planet + Ring + Moon + Asteroids |
| **Terrain** | FBM Landscape | Noise terrain + Water + Floating islands |
| **Abstract** | Generative Art | Gyroid + Metaballs + Rotating torus |
| **Fractal** | Menger Sponge | Infinite zoom fractal with surface-adhering particles |

Switch scenes in real-time with `[1]` `[2]` `[3]` `[4]` keys.

### The Fractal Dive (Microscope Demo)

A special demo showcasing **TRUE infinite resolution** - zoom in x10,000+ and the fractal remains perfectly sharp with **zero polygons**.

**Two Render Modes:**

| Mode | Method | Resolution | Best For |
|------|--------|------------|----------|
| **Raymarching** (Default) | Per-pixel SDF evaluation | **INFINITE** | Maximum quality, true solid surface |
| **Particles** | GPU particles on surface | High (density-dependent) | Visual effects, performance |

**Raymarching Mode (TRUE Infinite Resolution):**
- Per-pixel raymarching with 128 steps
- Procedural texturing via FBM noise (never pixelates)
- Same SDF formula: `Subtract(Box, Repeat(Cross))`
- Solid surface rendering with lighting and AO

**Features:**
- Logarithmic zoom camera (microscope-level precision)
- SDF formula: `Subtract(Box, Repeat(Cross))` - single mathematical object
- View-dependent density (particles concentrate in camera frustum)
- Real-time twist deformation

**Controls:**
- Mouse Wheel: Logarithmic zoom
- RMB + Mouse: Rotate view
- **[R] key: Toggle Raymarching / Particles mode**
- 1-4 keys: Change fractal iterations
- Space: Reset zoom

### Zoom Comparison Demo

Side-by-side comparison showing why SDF raymarching is superior:

| Left Camera | Right Camera |
|-------------|--------------|
| Traditional Mesh + Texture | SDF Raymarching + Procedural |
| Pixelates at x100 zoom | **Perfect at x10,000+ zoom** |
| Fixed polygon count | Zero polygons |

Run `ZoomComparisonDemo.cs` to see the difference in real-time.

## Time Slicing

Reduce GPU load by updating particles across multiple frames:

```
updateDivisions = 3:

Frame 0: Update particles 0, 3, 6, 9, ...
Frame 1: Update particles 1, 4, 7, 10, ...
Frame 2: Update particles 2, 5, 8, 11, ...

1M particles → 333K/frame
GPU load: 1/3
Visual difference: Negligible
```

## Quick Start

### 1. Build the Rust Library

```bash
cd ../   # ALICE-SDF root
cargo build --release

# Copy to Unity
cp target/release/libalice_sdf.dylib unity-sdf-universe/Assets/Plugins/  # macOS
cp target/release/alice_sdf.dll unity-sdf-universe/Assets/Plugins/       # Windows
cp target/release/libalice_sdf.so unity-sdf-universe/Assets/Plugins/     # Linux
```

### 2. Open in Unity

- Unity 2022.3+ recommended
- Open `unity-sdf-universe` as a Unity project
- Open scene `Assets/Scenes/SdfUniverse.unity`

### 3. Run

- Press Play
- Demo starts immediately with camera tour
- After tour: WASD to move, RMB to look

## Controls

| Key | Action |
|-----|--------|
| WASD | Move camera |
| QE | Up/Down |
| Shift | Boost speed |
| RMB + Mouse | Look around |
| Arrow Up/Down | Adjust sun size (Cosmic scene) |
| Arrow Left/Right | Adjust smoothness/blend |
| **1** | Switch to Cosmic scene |
| **2** | Switch to Terrain scene |
| **3** | Switch to Abstract scene |
| **4** | Switch to Fractal scene |
| **R** | Toggle Raymarching/Particles (Fractal scene) |

## Inspector Configuration

```
CosmicDemo (Script)
├── === MODE ===
│   ├── Particle Mode: [CPU_Standard | CPU_RustSIMD_Burst | GPU_ComputeShader]
│   └── Gpu Scene Type: [Cosmic | Terrain | Abstract | Fractal]
│
├── === TIME SLICING ===
│   └── Update Divisions: 1-10 (1=full, 3=1/3 load)
│
├── === COSMIC PARAMETERS ===
│   ├── Sun Radius: 1-50
│   ├── Planet Radius: 0.5-10
│   ├── Planet Distance: 5-30
│   └── Smoothness: 0-10
│
├── === PARTICLES ===
│   └── Particle Count: 10K-10M
│
└── === DEMO TIMING ===
    └── Tour Duration: seconds
```

## Project Structure

```
unity-sdf-universe/
├── Assets/
│   ├── Plugins/
│   │   └── AliceSdf.cs                      ← C# FFI bindings
│   ├── Scripts/
│   │   ├── SdfWorld.cs                      ← SDF world definition
│   │   ├── SdfParticleSystem.cs             ← CPU Standard mode
│   │   ├── SdfParticleSystem_Ultimate.cs    ← CPU Burst mode
│   │   ├── SdfParticleSystem_GPU.cs         ← GPU Compute mode
│   │   ├── CosmicDemo.cs                    ← Demo controller
│   │   ├── FractalDemo.cs                   ← Fractal Dive demo (Raymarching/Particles)
│   │   ├── ZoomComparisonDemo.cs            ← Side-by-side comparison demo
│   │   └── InfiniteZoomCamera.cs            ← Logarithmic zoom camera
│   ├── Shaders/
│   │   ├── SdfCompute_Cosmic.compute        ← GPU: Solar system
│   │   ├── SdfCompute_Terrain.compute       ← GPU: FBM landscape
│   │   ├── SdfCompute_Abstract.compute      ← GPU: Generative art
│   │   ├── SdfCompute_Fractal.compute       ← GPU: Menger Sponge (Particles)
│   │   ├── SdfSurface_Raymarching.shader    ← TRUE infinite resolution raymarching
│   │   ├── ParticleRender_Indirect.shader   ← GPU instancing
│   │   └── ParticleRender.shader            ← Standard rendering
│   └── Scenes/
│       └── SdfUniverse.unity
├── README.md
└── SETUP_GUIDE.md
```

## Performance Benchmarks

Tested on M3 MacBook Air:

| Mode | Particles | FPS | GPU Time |
|------|-----------|-----|----------|
| GPU Compute | 1,000,000 | 60+ | 0.15ms |
| GPU Compute | 5,000,000 | 45+ | 0.8ms |
| GPU Compute + Time Slice (1/3) | 10,000,000 | 60+ | 0.5ms |
| CPU Burst | 500,000 | 60 | - |
| CPU Standard | 100,000 | 60 | - |

## Technical Details

### SoA Layout (The Secret Sauce)

```
Traditional (AoS):  [x0,y0,z0, x1,y1,z1, ...]  ← Cache unfriendly
ALICE (SoA):        [x0,x1,x2,...], [y0,y1,...], [z0,z1,...]  ← SIMD heaven
```

### Why It's Fast

1. **Rust Core**: Zero-cost abstractions, no GC
2. **SIMD**: 8 points evaluated simultaneously (AVX2)
3. **Pre-compilation**: SDF tree → bytecode → 10x speedup
4. **SoA Layout**: Direct SIMD loads, no shuffle
5. **GPU Compute**: Zero CPU-GPU transfer
6. **Time Slicing**: Amortize updates across frames

### GPU Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Compute Shader (CSMain)                                 │
│ ├─ SDF Evaluation (sceneSDF)                           │
│ ├─ Normal Calculation (calcNormal)                     │
│ ├─ Physics Simulation                                  │
│ └─ Position Update                                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Vertex Shader (Billboard)                              │
│ └─ Read position from StructuredBuffer                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Fragment Shader (Glow)                                 │
│ └─ Additive blending, velocity-based color            │
└─────────────────────────────────────────────────────────┘
```

## Comparison with Traditional Engines

| Feature | Unity/DOTS | Unreal/Chaos | ALICE-SDF |
|---------|------------|--------------|-----------|
| Engine Size | 15+ GB | 30+ GB | **5 MB** |
| Memory (1M shapes) | 850 MB | 1+ GB | **120 MB** |
| Resolution | Vertex-dependent | Vertex-dependent | **INFINITE** |
| Real-time Morphing | Rebake needed | Rebake needed | **Instant** |
| Infinite Repeat | Impossible | Impossible | **Possible** |
| Boolean Ops | Pre-baked | Pre-baked | **Real-time** |
| Zoom Limit | x100 | x100 | **x10^6+** |
| Max Particles (60 FPS) | 500K | 1M | **10M+** |

## License

**ALICE Community License** - Free for game development!

| Use Case | License |
|----------|---------|
| Personal projects | FREE |
| Indie games (any revenue) | FREE |
| AAA game studios | FREE |
| Education & research | FREE |
| Metaverse platforms (10,000+ MAU) | Commercial |
| Cloud/SaaS infrastructure | Commercial |

Content you create is 100% yours. No royalties.

See [LICENSE](LICENSE) for details.

## Credits

- **ALICE-SDF**: Moroya Sakamoto
- **Project A.L.I.C.E.**: Universe Physics, Barnes-Hut, NeuroWired

---

**"5MB to render infinity. That's not a bug, that's mathematics."**
