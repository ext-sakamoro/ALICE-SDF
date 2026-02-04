# ALICE-SDF for VRChat

**Stand on math.** A VRChat package that lets players walk on, collide with, and interact with Signed Distance Function surfaces — all via Drag & Drop.

[Japanese / 日本語版](README_JP.md)

## Install

### Via Unity Package Manager (Recommended)

1. In Unity: **Window > Package Manager**
2. Click **+** > **Add package from disk...**
3. Select `package.json` from the ALICE-SDF folder

Or add via git URL in `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.alice.sdf": "https://github.com/sakamoro/ALICE-SDF.git?path=vrchat-package"
  }
}
```

### Via .unitypackage (Legacy)

1. Download `ALICE-SDF-VRChat.unitypackage`
2. In Unity: **Assets > Import Package > Custom Package...**
3. Select all and import

### Requirements

- Unity 2022.3.x (VRChat recommended version)
- VRChat SDK3 + UdonSharp (for collision — shader works without it)

## Three Components

### 1. ALICE-Shader (Raymarching Kernel)

Pixel-level SDF rendering via HLSL raymarching.

- `AliceSDF_Include.cginc` — Full primitive & operation function library
- `AliceSDF_LOD.cginc` — Dynamic LOD for VRChat GPU budget (Deep Fried)
- `AliceSDF_Raymarcher.shader` — Main shader with SV_Depth output

**Deep Fried LOD**: Automatically adjusts ray step count based on camera distance.

| Distance | Steps | Epsilon | Quality |
|----------|-------|---------|---------|
| < 20m    | 128   | 0.0001  | High    |
| 20–60m   | 64    | 0.001   | Medium  |
| > 60m    | 32    | 0.005   | Low     |

### 2. ALICE-Udon (SDF Collider)

Evaluates the same SDF formula in UdonSharp (pure C#) to push players out of solid geometry.

- `AliceSDF_Primitives.cs` — 6 primitives + CSG operations + transforms (pure C#)
- `AliceSDF_Math.cs` — Vector math helpers
- `AliceSDF_Collider.cs` — Player collision detection & push-back

**How it works:**

1. Get player position **P**
2. Compute **d = SDF(P)**
3. If **d < 0** (inside), push along **gradSDF(P)** by **|d|**

### 3. ALICE-Baker v0.2 (Deep Fried Editor Tool)

Paste `.asdf.json` and auto-generate optimized Shader + Udon + Prefab.

- Open via **Window > ALICE-SDF Baker**
- Paste JSON or drag & drop a `.asdf.json` TextAsset
- Click **Bake!** to generate everything at once
- **Live preview**: code updates in real-time as you edit JSON

#### Baker v0.2 Optimizations

| Optimization | Target | Effect |
|-------------|--------|--------|
| Instruction Fusion | HLSL | Leaf nodes inlined directly — no temp variables for primitives |
| Branchless CSG | HLSL | `opUnion` expanded to `min()`, `opSubtraction` to `max(a, -(b))` |
| Division Exorcism | HLSL | `1/k` pre-computed at top of `map()` for smooth operations |
| Scalar Expansion | Udon | `Sdf.Union()` replaced with `Mathf.Min()` — eliminates function call overhead |
| Smooth Op Inline | Udon | `SmoothUnion/Intersection/Subtraction` fully expanded as scalar math with pre-computed `inv_k` |
| Translate Scalarization | Udon | `Vector3` subtraction split into `float x,y,z` when child is a leaf — reduces Udon VM struct copies |
| Smart Float Format | Both | `0.000000` becomes `0.0`, `1.500000` becomes `1.5` |
| Live Preview | Editor | JSON changes detected via hash — preview updates automatically |

## Samples (SDF Gallery)

Four ready-to-play samples are included. Import via **Package Manager > Samples** tab.

| Sample | Description | SDF Formula |
|--------|-------------|-------------|
| **Basic** | Ground plane + floating sphere. The simplest SDF world. | `min(plane, sphere)` |
| **Cosmic** | Animated solar system — Sun, orbiting planet, tilted ring, moon, asteroid belt. | `SmoothUnion(sun, planet, ring, moon, asteroids)` |
| **Fractal** | Walk inside a Menger Sponge labyrinth with twist deformation. | `Subtract(Box, Repeat(Cross))` — one formula, infinite complexity |
| **Mix** | Cosmic x Fractal fusion — fractal planet + torus ring + onion shell. | `SmoothUnion(Intersect(Sphere, Menger), Torus, Onion(Sphere))` |

Each sample includes:
- `*_Raymarcher.shader` — Raymarching shader with SV_Depth, LOD, AO, fog
- `*_Collider.cs` — UdonSharp collider (with `#if UDONSHARP` guard)
- `*.asdf.json` — Source definition for the Baker

### Generate Sample Scenes

After importing samples, generate ready-to-play scenes:

1. **ALICE-SDF > Generate Sample Scenes** (Unity menu)
2. Scenes are created in `Assets/AliceSDF_SampleScenes/`
3. Open any `SDF_*.unity` scene and press **Play**

The generator auto-detects which samples have been imported and creates a scene with Camera, Light, and a Cube with the SDF shader applied.

## Quick Start

### Manual Setup

1. Create a Material from `AliceSDF_Raymarcher.shader`
2. Place a Cube in the scene (bounding volume)
3. Assign the material to the Cube
4. Attach `AliceSDF_Collider.cs` to a GameObject
5. Write your SDF formula in the `Evaluate()` method (must match the shader's `map()`)

### ALICE-Baker (Recommended)

1. **Window > ALICE-SDF Baker**
2. Paste your `.asdf.json`
3. Click **Bake!** — Shader + Udon + Prefab are generated
4. Drag the generated Prefab into your scene

## File Structure (UPM)

```
com.alice.sdf/
├── package.json                     # UPM manifest
├── CHANGELOG.md
├── README.md / README_JP.md
├── Runtime/
│   ├── AliceSDF.Runtime.asmdef      # Assembly Definition
│   ├── Shaders/
│   │   ├── AliceSDF_Include.cginc   # SDF function library
│   │   ├── AliceSDF_LOD.cginc       # Deep Fried dynamic LOD
│   │   └── AliceSDF_Raymarcher.shader # Main raymarching shader
│   └── Udon/
│       ├── AliceSDF_Math.cs         # Vector math helpers
│       ├── AliceSDF_Primitives.cs   # SDF functions (C#)
│       └── AliceSDF_Collider.cs     # Player collision
├── Editor/
│   ├── AliceSDF.Editor.asmdef       # Editor Assembly Definition
│   ├── AliceSDF_Baker.cs            # Baker v0.2 (Deep Fried)
│   └── SampleSceneGenerator.cs      # Menu: ALICE-SDF > Generate Sample Scenes
├── Samples~/                        # UPM Samples (import via Package Manager)
│   └── SDF Gallery/
│       ├── SampleBasic/             # Ground + Sphere
│       ├── SampleCosmic/            # Solar system
│       ├── SampleFractal/           # Menger Sponge labyrinth
│       └── SampleMix/              # Cosmic x Fractal fusion
└── Prefabs~/                        # Hidden from Unity import
```

## Supported Primitives (11)

| Primitive | HLSL | C# | Formula |
|-----------|------|----|---------|
| Sphere    | `sdSphere`     | `Sdf.Sphere`     | `length(p) - r` |
| Box       | `sdBox`        | `Sdf.Box`        | Branchless min/max |
| Cylinder  | `sdCylinder`   | `Sdf.Cylinder`   | Capped vertical |
| Torus     | `sdTorus`      | `Sdf.Torus`      | XZ ring |
| Plane     | `sdPlane`      | `Sdf.Plane`      | `dot(p,n) + d` |
| Capsule   | `sdCapsule`    | `Sdf.Capsule`    | Line segment + r |
| Cone      | `sdCone`       | `Sdf.Cone`       | Capped Y-axis cone |
| Ellipsoid | `sdEllipsoid`  | `Sdf.Ellipsoid`  | Bound-corrected approx |
| HexPrism  | `sdHexPrism`   | `Sdf.HexPrism`   | Hexagonal prism (Z-axis) |
| Triangle  | `sdTriangle`   | `Sdf.Triangle`   | Exact 3D triangle |
| Bezier    | `sdBezier`     | `Sdf.Bezier`     | Quadratic curve + radius |

## Supported Operations (16)

| Operation | HLSL | C# | Effect |
|-----------|------|----|--------|
| Union              | `opUnion` / `min`             | `Sdf.Union` / `Mathf.Min`     | Combine shapes |
| Intersection       | `opIntersection` / `max`      | `Sdf.Intersection` / `Mathf.Max` | Intersect shapes |
| Subtraction        | `opSubtraction` / `max(a,-b)` | `Sdf.Subtraction` / `Mathf.Max(a,-(b))` | Carve out |
| Smooth Union       | `opSmoothUnion`               | `Sdf.SmoothUnion` (inlined by Baker) | Smooth blend |
| Smooth Intersection| `opSmoothIntersection`        | `Sdf.SmoothIntersection` (inlined) | Smooth intersect |
| Smooth Subtraction | `opSmoothSubtraction`         | `Sdf.SmoothSubtraction` (inlined) | Smooth carve |
| Repeat Infinite    | `opRepeatInfinite`            | `Sdf.RepeatInfinite`          | Infinite tiling |
| Repeat Finite      | `opRepeatFinite`              | `Sdf.RepeatFinite`            | Bounded tiling |
| Polar Repeat       | `opPolarRepeat`               | `Sdf.PolarRepeat`             | Circular array (Y-axis) |
| Twist              | `opTwist`                     | `Sdf.Twist`                   | Y-axis twist |
| Bend               | `opBend`                      | `Sdf.Bend`                    | X-axis bend |
| Round              | `opRound`                     | `Sdf.Round`                   | Round edges |
| Onion              | `opOnion`                     | `Sdf.Onion`                   | Hollow shell |
| Taper              | `opTaper`                     | `Sdf.Taper`                   | Y-axis taper |
| Displacement       | `opDisplacement`              | `Sdf.Displacement`            | Noise surface |
| Symmetry           | `opSymmetry`                  | `Sdf.Symmetry`                | Axis mirroring |
| Elongate           | `opElongate`                  | `Sdf.Elongate`                | Stretch along axes |

## VRChat Compatibility

- **No native plugins**: Everything runs as Shader + UdonSharp — no DllImport
- **SV_Depth output**: Proper depth buffer occlusion with VRM avatars
- **`#if UDONSHARP` guard**: Compiles without VRC SDK installed (MonoBehaviour stub)
- **Performance Rank safe**: Deep Fried LOD keeps GPU within budget at any distance

## License

ALICE Community License

## Author

Moroya Sakamoto
