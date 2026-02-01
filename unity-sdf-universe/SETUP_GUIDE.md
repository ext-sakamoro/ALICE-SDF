# ALICE-SDF Unity Setup Guide

Complete setup guide for running the SDF Universe demo.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Rust Library Setup](#rust-library-setup)
3. [Unity Project Setup](#unity-project-setup)
4. [Scene Configuration](#scene-configuration)
5. [Inspector Settings](#inspector-settings)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10, macOS 10.15+, Ubuntu 20.04+ |
| Unity | 2022.3 LTS or later |
| Rust | 1.70+ (for building native library) |
| GPU | Any GPU with Compute Shader support |
| RAM | 8 GB |

### Recommended

| Component | Requirement |
|-----------|-------------|
| Unity | 2022.3.20f1+ |
| GPU | Metal/Vulkan/DX12 capable |
| RAM | 16 GB |
| CPU | Apple M1+ / Intel i7+ / AMD Ryzen 7+ |

---

## Rust Library Setup

### Step 1: Install Rust

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Step 2: Build the Library

```bash
# Navigate to ALICE-SDF root
cd /path/to/ALICE-SDF

# Build release version
cargo build --release
```

### Step 3: Copy to Unity

**macOS (Apple Silicon / Intel):**
```bash
cp target/release/libalice_sdf.dylib unity-sdf-universe/Assets/Plugins/
```

**Windows:**
```bash
copy target\release\alice_sdf.dll unity-sdf-universe\Assets\Plugins\
```

**Linux:**
```bash
cp target/release/libalice_sdf.so unity-sdf-universe/Assets/Plugins/
```

### Step 4: macOS Gatekeeper (if blocked)

```bash
# Sign the library
codesign --force --sign - unity-sdf-universe/Assets/Plugins/libalice_sdf.dylib

# Or remove quarantine attribute
xattr -d com.apple.quarantine unity-sdf-universe/Assets/Plugins/libalice_sdf.dylib
```

---

## Unity Project Setup

### Step 1: Open Project

1. Open Unity Hub
2. Click "Add" -> Navigate to `unity-sdf-universe`
3. Select Unity 2022.3+ as the editor version
4. Click "Open"

### Step 2: Package Dependencies

The project requires these packages (auto-installed via manifest.json):

| Package | Version | Purpose |
|---------|---------|---------|
| com.unity.burst | 1.8.12+ | Burst compiler for CPU mode |
| com.unity.mathematics | 1.2.6+ | Math operations |
| com.unity.collections | 1.4.0+ | Native collections |

### Step 3: First Run

1. Open `Assets/Scenes/SdfUniverse.unity`
2. Press **Play**
3. Camera tour starts automatically
4. After tour: WASD to move, RMB to look

---

## Scene Configuration

### Creating CosmicDemo from Scratch

1. **Create empty scene:** `File -> New Scene -> Basic`
2. **Save:** `File -> Save As -> Assets/Scenes/MyDemo.unity`
3. **Create controller:**
   - `GameObject -> Create Empty`
   - Rename to "CosmicDemo"
   - `Add Component -> CosmicDemo`
4. **Press Play** - Everything auto-creates!

The CosmicDemo script automatically creates:
- SdfWorld component
- Particle system (GPU/CPU based on mode)
- Camera (if none exists)
- UI elements (stats, controls)
- Lighting

### Creating FractalDemo (The Fractal Dive)

1. **Create empty scene:** `File -> New Scene -> Basic`
2. **Save:** `File -> Save As -> Assets/Scenes/FractalDemo.unity`
3. **Create controller:**
   - `GameObject -> Create Empty`
   - Rename to "FractalDemo"
   - `Add Component -> FractalDemo`
4. **Press Play** - Fractal world auto-creates!

The FractalDemo script automatically creates:
- InfiniteZoomCamera (logarithmic zoom)
- Raymarching quad (default) or SdfParticleSystem_GPU
- UI elements (zoom stats, controls)
- Lighting

**Fractal Demo Features:**
- **Two Render Modes:**
  - **Raymarching** (Default): TRUE infinite resolution via per-pixel SDF evaluation
  - **Particles**: GPU particles adhering to surface
- Zoom x10,000+ with infinite resolution
- SDF formula: `Subtract(Box, Repeat(Cross))`
- Procedural texturing (raymarching mode) - never pixelates
- Initial zoom: x0.5 magnification
- Press **[R]** to toggle between modes

**Why Raymarching is TRUE Infinite Resolution:**
- Per-pixel SDF evaluation (128 raymarching steps)
- Procedural texturing via FBM noise (colors from math, not textures)
- No particles = no gaps at any zoom level
- Solid surface rendering with lighting, AO, fresnel effects

---

## Inspector Settings

### Mode Selection

```
=== MODE ===
Particle Mode: [Dropdown]
├── CPU_Standard       : Parallel.For (for debugging)
├── CPU_RustSIMD_Burst : Burst Jobs (balanced)
└── GPU_ComputeShader  : Full GPU (maximum perf) [Recommended]
```

### GPU Scene Type

```
=== GPU SCENE (GPU Mode Only) ===
Gpu Scene Type: [Dropdown]
├── Cosmic   : Sun + Planet + Ring + Moon + Asteroids
├── Terrain  : FBM noise terrain + Water + Islands
├── Abstract : Gyroid + Metaballs + Torus
└── Fractal  : Menger Sponge with infinite zoom
```

**Runtime switching:** Press `[1]` `[2]` `[3]` `[4]` keys

### Time Slicing

```
=== TIME SLICING ===
Update Divisions: [Slider 1-10]

| Value | Updates/Frame | GPU Load | Best For         |
|-------|---------------|----------|------------------|
|   1   | 100%          | Full     | < 1M particles   |
|   2   | 50%           | 1/2      | 1-3M particles   |
|   3   | 33%           | 1/3      | 3-5M particles   |
|   5   | 20%           | 1/5      | 5-8M particles   |
|  10   | 10%           | 1/10     | 8M+ particles    |
```

### Particle Count

```
=== PARTICLES ===
Particle Count: [Slider 10K - 10M]

| Count | Recommended Mode      | Expected FPS |
|-------|-----------------------|--------------|
| 100K  | Any                   | 60+ FPS      |
| 500K  | CPU Burst or GPU      | 60 FPS       |
| 1M    | GPU                   | 60 FPS       |
| 5M    | GPU + Slice(3)        | 45-60 FPS    |
| 10M   | GPU + Slice(5+)       | 30-60 FPS    |
```

### Scene-Specific Parameters

**Cosmic Scene:**
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Sun Radius | 1-50 | 8 | Central sun size |
| Planet Radius | 0.5-10 | 2.5 | Orbiting planet size |
| Planet Distance | 5-30 | 18 | Orbit radius |
| Smoothness | 0-10 | 1.5 | Blend between shapes |

**Terrain Scene:**
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Terrain Height | 1-30 | 10 | Height variation |
| Terrain Scale | 0.1-5 | 1 | Noise frequency |
| Water Level | -10 to 10 | 0 | Water plane height |
| Rock Size | 0.5-5 | 1.5 | Scattered rock size |

**Abstract Scene:**
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Gyroid Scale | 0.1-2 | 0.5 | Gyroid frequency |
| Gyroid Thickness | 0.1-1 | 0.3 | Surface thickness |
| Metaball Radius | 0.5-5 | 2 | Metaball size |
| Morph Amount | 0-1 | 0.5 | Shape morphing |

**Fractal Scene:**
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Box Size | 10-200 | 50 | Fractal bounding size |
| Hole Size | 0.5-10 | 2 | Cross subtraction size |
| Repeat Scale | 5-50 | 15 | Pattern repetition scale |
| Twist Amount | 0-0.2 | 0.02 | Organic twist deformation |
| Fractal Iterations | 1-5 | 3 | Menger Sponge depth |

---

## Performance Tuning

### Target: 60 FPS

| Situation | Solution |
|-----------|----------|
| FPS < 30 | Reduce particle count or increase time slice |
| FPS 30-45 | Increase time slice (e.g., 3 -> 5) |
| FPS 45-55 | Minor adjustments or acceptable |
| FPS 60+ | Can increase particles or reduce slice |

### Optimization Checklist

1. **Use GPU Mode** - Always prefer `GPU_ComputeShader`
2. **Enable Time Slicing** - For >1M particles, use divisions 3+
3. **Reduce Particle Size** - Smaller `particleSize` = less overdraw
4. **Limit Max Distance** - Smaller bounds = fewer particles visible

### Memory Usage

| Particles | GPU Buffer | Notes |
|-----------|------------|-------|
| 100K | 3.2 MB | No concerns |
| 1M | 32 MB | Normal |
| 5M | 160 MB | Monitor VRAM |
| 10M | 320 MB | Requires 4GB+ VRAM |

---

## Troubleshooting

### DllNotFoundException: alice_sdf

**Cause:** Native library not found

**Solution:**
1. Verify library exists in `Assets/Plugins/`
2. Check file extension matches OS (.dylib/.dll/.so)
3. Rebuild with `cargo build --release`
4. On macOS: Sign library (see Rust Library Setup)
5. Restart Unity

### Particles Not Visible

**Cause:** Material or shader issue

**Solution:**
1. Check Console for shader compile errors
2. Verify `ParticleRender_Indirect.shader` exists
3. Try reducing particle count to 10K
4. Switch to `CPU_Standard` mode to test

### Black Screen

**Cause:** Camera or lighting issue

**Solution:**
1. Check if camera exists and is tagged "MainCamera"
2. Verify camera clear flags (should be Solid Color)
3. Check background color isn't black

### Low FPS in GPU Mode

**Cause:** GPU bottleneck

**Solution:**
1. Enable Time Slicing (set to 3+)
2. Reduce particle count
3. Check GPU usage in Unity Profiler
4. Verify compute shaders compiled (check Console)

### Scene Switching Not Working

**Cause:** Compute shaders not assigned

**Solution:**
1. Select the `SdfParticleSystem_GPU` component
2. Assign shaders in Inspector:
   - Cosmic Shader -> `SdfCompute_Cosmic`
   - Terrain Shader -> `SdfCompute_Terrain`
   - Abstract Shader -> `SdfCompute_Abstract`
   - Fractal Shader -> `SdfCompute_Fractal`

### Compute Shader Errors

**Cause:** GPU doesn't support compute shaders

**Solution:**
1. Check Unity Player Settings -> Graphics API
2. Use Metal (macOS) / DX12 (Windows) / Vulkan (Linux)
3. Update GPU drivers
4. Fall back to `CPU_RustSIMD_Burst` mode

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| **Movement** | |
| W/S | Forward/Backward |
| A/D | Left/Right |
| Q/E | Down/Up |
| Shift | Boost speed |
| RMB + Mouse | Look around |
| **Scene Control (GPU Mode)** | |
| 1 | Switch to Cosmic scene |
| 2 | Switch to Terrain scene |
| 3 | Switch to Abstract scene |
| 4 | Switch to Fractal scene |
| **Fractal Scene Controls** | |
| Mouse Wheel | Logarithmic zoom |
| Space | Reset zoom |
| **R** | **Toggle Raymarching / Particles mode** |
| **Parameters** | |
| Arrow Up/Down | Adjust sun size |
| Arrow Left/Right | Adjust smoothness |

---

## Project Structure

```
unity-sdf-universe/
├── Assets/
│   ├── Plugins/
│   │   └── AliceSdf.cs                      <- C# FFI bindings
│   ├── Scripts/
│   │   ├── SdfWorld.cs                      <- SDF world definition
│   │   ├── SdfParticleSystem.cs             <- CPU Standard mode
│   │   ├── SdfParticleSystem_Ultimate.cs    <- CPU Burst mode
│   │   ├── SdfParticleSystem_GPU.cs         <- GPU Compute mode
│   │   ├── CosmicDemo.cs                    <- Demo controller
│   │   ├── FractalDemo.cs                   <- Fractal Dive (Raymarching/Particles)
│   │   ├── ZoomComparisonDemo.cs            <- Side-by-side comparison demo
│   │   └── InfiniteZoomCamera.cs            <- Logarithmic zoom camera
│   ├── Shaders/
│   │   ├── SdfCompute_Cosmic.compute        <- GPU: Solar system
│   │   ├── SdfCompute_Terrain.compute       <- GPU: FBM landscape
│   │   ├── SdfCompute_Abstract.compute      <- GPU: Generative art
│   │   ├── SdfCompute_Fractal.compute       <- GPU: Menger Sponge (Particles)
│   │   ├── SdfSurface_Raymarching.shader    <- TRUE infinite resolution raymarching
│   │   ├── ParticleRender_Indirect.shader   <- GPU instancing
│   │   └── ParticleRender.shader            <- Standard rendering
│   └── Scenes/
│       └── SdfUniverse.unity
├── README.md
└── SETUP_GUIDE.md                           <- You are here
```

---

## Quick Reference

### Recommended Settings for 60 FPS

| Hardware | Particles | Mode | Time Slice |
|----------|-----------|------|------------|
| M1 MacBook Air | 1M | GPU | 1 |
| M3 MacBook Pro | 5M | GPU | 3 |
| RTX 3060 | 5M | GPU | 2 |
| RTX 4080 | 10M | GPU | 3 |
| Intel iGPU | 100K | CPU Burst | - |

### Demo Sequence

1. **Press Play** -> Camera tour starts immediately
2. **Tour completes** -> Free roam mode
3. **WASD** to move, **RMB** to look
4. **[1][2][3][4]** to switch GPU scenes
5. **Arrow keys** to morph shapes
6. **[4]** for Fractal scene -> **Mouse Wheel** for infinite zoom
7. **[R]** to toggle Raymarching (solid surface) / Particles mode

---

**"5MB to render infinity. That's not a bug, that's mathematics."**
