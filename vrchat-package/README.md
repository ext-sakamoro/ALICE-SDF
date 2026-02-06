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

Seven ready-to-play samples are included. Import via **Package Manager > Samples** tab.

| Sample | Description | SDF Formula |
|--------|-------------|-------------|
| **Basic** | Ground plane + floating sphere. The simplest SDF world. | `min(plane, sphere)` |
| **Cosmic** | Animated solar system — Sun, orbiting planet, tilted ring, moon, asteroid belt. | `SmoothUnion(sun, planet, ring, moon, asteroids)` |
| **Fractal** | Walk inside a Menger Sponge labyrinth with twist deformation. | `Subtract(Box, Repeat(Cross))` — one formula, infinite complexity |
| **Mix** | Cosmic x Fractal fusion — fractal planet + torus ring + onion shell. | `SmoothUnion(Intersect(Sphere, Menger), Torus, Onion(Sphere))` |
| **DeformableWall** | Touch/hit the wall and it dents. Dents recover over time. VR hand interaction. | `min(ground, SmoothSubtract(wall, dent_spheres...))` |
| **Mochi** | Squishy mochi blobs. Grab, merge, split, and grow. SmoothUnion soft-body physics. | `SmoothUnion(ground, SmoothUnion(mochi1, mochi2, ..., k))` |
| **TerrainSculpt** | Dig holes & build hills with VR hands. You fall into holes you dig. **Only possible with SDF.** | `SmoothUnion(SmoothSub(plane, digs...), hills...)` |

Each sample includes:
- `*_Raymarcher.shader` — Raymarching shader with SV_Depth, LOD, AO, fog
- `*_Collider.cs` — UdonSharp collider (with `#if UDONSHARP` guard)
- `*.asdf.json` — Source definition for the Baker

### Interactive Samples (VR)

The **DeformableWall**, **Mochi**, and **TerrainSculpt** samples demonstrate real-time SDF deformation driven by VR hand tracking. Unlike the static samples above, these send dynamic data from UdonSharp to the shader every frame via `Material.SetVectorArray`.

#### DeformableWall — Touch & Dent

A flat wall standing on a ground plane. When a VR player's hand touches the wall surface, a dent appears at the contact point and gradually recovers over time.

**How it works:**
1. UdonSharp detects hand proximity to the wall via `SdfBox()` distance check
2. On contact, the impact position and timestamp are recorded (circular buffer, max 16)
3. Each frame, the array is sent to the shader via `Material.SetVectorArray("_ImpactPoints", ...)`
4. The shader carves each dent using `opSmoothSubtraction(wall, sphere)`, where the sphere radius decays exponentially: `r = DentRadius * exp(-age * DecaySpeed)`
5. Fully decayed dents (radius < 0.005) are automatically recycled

**VR Interaction:**
- Move your hand close to the wall surface — a dent appears
- Hit the wall repeatedly — up to 16 dents at once
- Wait — dents smoothly recover back to the flat wall
- Walk into the wall — player collision pushes you back

**Inspector Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Impact Distance | 0.08 | How close the hand must be to register a dent |
| Impact Cooldown | 0.15s | Minimum time between impacts from the same hand |
| Dent Radius | 0.35 | Size of each dent |
| Decay Speed | 0.5 | Recovery speed (higher = faster recovery) |
| Dent Smoothness | 0.08 | SmoothSubtraction blend factor |

#### Mochi — Grab, Merge, Split & Grow

Soft mochi (rice cake) blobs sitting on a ground plane. In VR, you can grab them, pull them apart, push them together, and watch them grow.

**How it works:**
1. Up to 16 mochi spheres are tracked as `(position, radius)` pairs
2. All mochis are blended together using `opSmoothUnion` — nearby mochis naturally appear to merge visually
3. Ground contact uses a separate `opSmoothUnion` with a lower `k` for a "squishy sitting on the floor" look
4. UdonSharp sends the mochi array to the shader every frame via `Material.SetVectorArray("_MochiData", ...)`

**VR Interaction:**

| Action | How | What Happens |
|--------|-----|--------------|
| **Grab** | Hold hand inside a mochi for 0.08s | Mochi sticks to your hand |
| **Move** | Move hand while grabbing | Mochi follows your hand |
| **Split** | Pull hand far from grab origin (2.5x radius) | Mochi splits into two pieces (volume conserved: `r_new = r * cbrt(0.5)`) |
| **Release** | Move hand very far (4x radius) | Mochi drops and settles to the ground |
| **Merge** | Push two free mochis close together | They merge into one bigger mochi (`r = cbrt(r1^3 + r2^3)`) |
| **Grow** | Keep merging mochis | The merged mochi gets bigger and bigger |

**Inspector Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Blend K | 0.5 | SmoothUnion factor between mochis (higher = stickier) |
| Ground K | 0.15 | SmoothUnion factor with ground (squishy floor contact) |
| Min Radius | 0.1 | Smallest allowed mochi (won't split below this) |
| Grab Threshold | 0.8 | Hand must be within this fraction of radius to grab |
| Grab Dwell Time | 0.08s | Hold time before grab activates (prevents accidental grabs) |
| Split Distance | 2.5 | Pull distance (x radius) to trigger split |
| Merge Threshold | 0.7 | Distance (x combined radii) for auto-merge |

#### TerrainSculpt — Dig Holes & Build Hills

**The first VRChat experience where you can dig a hole and actually fall into it.**

A flat ground plane that players can sculpt in real-time. Left hand adds terrain, right hand digs. Both rendering and collision use the exact same SDF formula — what you see is what you collide with, even as the terrain changes.

This is fundamentally impossible with VRChat's mesh-based approach because MeshColliders cannot be recalculated at runtime. ALICE-SDF evaluates the same math for both pixels and physics.

**How it works:**
1. Base terrain is a ground plane at Y=0
2. Left hand near surface → `opSmoothUnion(terrain, sphere)` — adds a hill at hand position
3. Right hand near surface → `opSmoothSubtraction(terrain, sphere)` — digs a hole at hand position
4. Operations are stored in a circular buffer (max 48). When full, oldest operations are overwritten
5. UdonSharp sends the operation array to the shader every frame
6. Player collision evaluates the same formula — fall into holes, climb hills

**VR Interaction:**

| Action | Hand | What Happens |
|--------|------|--------------|
| **Dig** | Right hand near ground | A hemispherical hole is carved. You can fall in |
| **Build** | Left hand near ground | A hill/mound appears. You can climb it |
| **Sculpt deeper** | Keep right hand in the hole | Dig deeper with each operation |
| **Build higher** | Keep left hand on the mound | Stack more terrain on top |

**Visual feedback:**
- Blue glow around left hand = add mode
- Red glow around right hand = dig mode
- Glows only appear when hand is near the terrain surface

**Terrain coloring:**
- Green grass on flat surfaces
- Brown dirt on steep slopes
- Gray rock when digging deep underground

**Inspector Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Sculpt Radius | 0.3 | Size of each sculpt brush stroke |
| Sculpt Distance | 0.15 | How close hand must be to terrain surface to sculpt |
| Sculpt Cooldown | 0.12s | Minimum time between operations (prevents buffer spam) |
| Add Smooth | 0.25 | SmoothUnion blend for hills (higher = smoother) |
| Sub Smooth | 0.15 | SmoothSubtraction blend for holes (higher = smoother edges) |

#### Setup (All Interactive Samples)

1. Place a **Cube** in your scene (this is the raymarching bounding volume)
2. Scale it to cover the desired area (e.g. `(12, 8, 12)` for DeformableWall, `(20, 10, 20)` for TerrainSculpt)
3. Create a **Material** from `AliceSDF/Samples/DeformableWall`, `AliceSDF/Samples/Mochi`, or `AliceSDF/Samples/TerrainSculpt`
4. Assign the material to the Cube's **MeshRenderer**
5. Add the corresponding `*_Collider.cs` script to the same GameObject
6. **Build & Test** in VRChat — use your VR hands to interact

**Desktop mode:** The interactive features require VR hand tracking. In desktop mode, the SDF rendering and player collision still work, but you cannot trigger sculpting, dents, or mochi grabs.

**Multiplayer note:** All interactive samples run in local-only mode (each player sees their own state). To sync across players, add `[UdonSynced]` to the data arrays and call `RequestSerialization()` on state changes.

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
│       ├── SampleMix/              # Cosmic x Fractal fusion
│       ├── SampleDeformableWall/    # Interactive: touch wall → dent → recover
│       ├── SampleMochi/            # Interactive: grab, merge, split, grow
│       └── SampleTerrainSculpt/   # Interactive: dig holes, build hills, fall in
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
