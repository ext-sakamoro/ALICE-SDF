# Changelog

## [Unreleased]

### Added
- `HostTests~/MochiParity`: the Mochi collider is now compiled on every push
  without Unity (C# 7.3, warnings as errors, against a ~60-line
  `UnityEngineStub.cs`), its `EvaluateSdf` is compared with
  `alice_sdf::eval` of the same scene (`examples/vrchat_mochi_golden.rs`,
  1521 points, 1e-5) and a grab → split → release → settle → merge scenario
  is replayed. `scripts/vrchat-host-parity.sh` runs it locally
  (`brew install dotnet`); CI job `vrchat-host`. Unity never imports the
  folder (trailing `~`). This is the first host-side check of any UdonSharp
  code in the package; the 53-primitive `AliceSDF_Primitives.cs` is next.

### Fixed
- Mochi sample: the player-collision test used the rendered SDF (ground
  plane included), so standing on the floor was a permanent penetration and
  the player bobbed up and down every frame. Player collision now uses the
  mochis alone (`EvaluateMochiSdf`); `EvaluateSdf` is unchanged.
- `Runtime/AliceSDF.Runtime.asmdef` now references `UdonSharp.Runtime`,
  `VRC.Udon` and `VRC.SDKBase`. It defined `UDONSHARP` but referenced
  nothing, so `AliceSDF_Collider.cs` did not compile in a VRChat project.
  Verified in Unity 2022.3.22f1 + VRChat SDK 3.10.1 (ClientSim).

### Changed
- Mochi sample, same features, tightened:
  - Shader: the ground / mochi material weight comes from the blend factor
    of the same `opSmoothUnion` that shapes the surface (IQ smooth-min with
    material), so the neck where a mochi squishes onto the floor shades
    continuously instead of switching at `mochiRaw < groundRaw`; normal
    half-width follows the LOD tier (0.001 / 0.003 / 0.01) instead of a
    fixed 0.001 finer than the LOW surface epsilon; ground grain is
    interpolated value noise instead of a per-pixel hash (no sparkle in
    VR); soft contact shadow (32 / 16 / 8 steps by tier, `_ShadowEnabled`
    default on); light direction and fog density are material properties
    (`_LightDir`, `_FogDensity`, defaults equal to the former constants).
  - Collider: `blendK` / `groundK` are pushed to the material every frame
    (one source of truth for collision and rendering, the "match shader"
    tooltips are gone); the per-hand grab / origin / split / dwell state is
    indexed by hand instead of duplicated `if (isLeft)` branches (same
    logic, one `ProcessHand`); gravity settle is `1 - exp(-g dt)` instead of
    `Lerp(..., g dt)` (frame-rate independent); a merged mochi is clamped
    to `y >= r` so it does not spend a frame under the floor; tracking
    validity rejects the exact zero vector instead of a 10 cm radius
    around the world origin, where the mochis sit; `MaxMochi` constant
    documents the shader's `MOCHI_MAX = 16` coupling.
  - Verified on the host: the CGPROGRAM block compiles under glslang
    (HLSL mode, stub UnityCG.cginc) for both stages; the collider compiles
    under .NET 10 / C# 7.3 with warnings as errors against a UnityEngine
    stub, `EvaluateSdf` matches `alice_sdf::eval` of the same scene on a
    1521-point grid (max |Δ| 6e-8), and a grab → split → release → settle →
    merge scenario holds volume conservation. Not verified in Unity or
    VRChat in this environment.
- One shader source: `Runtime/Shaders/` (the VPM path the Baker and the
  Samples include) is the only copy. The legacy `Assets/AliceSDF/Shaders/`
  layout had forked in February and carried the PBR surface (GGX /
  Schlick / Smith, per-material colour / roughness / metallic,
  material-id aware `op*Mat` ops) that `Runtime/` lacked, while `Runtime/`
  had 40 more primitives; the PBR surface and the material ops are now in
  `Runtime/` and the fork is deleted. The June `ALICE-SDF-VRChat.unitypackage`
  (a snapshot of the old layout) is removed with the "Legacy" install
  section; install via VPM / Package Manager.
- Not verified in Unity yet: the merged `AliceSDF_Raymarcher.shader` and
  `AliceSDF_Include.cginc` were merged by hand without a Unity compile in
  this environment — open a sample scene and check the console.


## [0.4.0] - 2026-02-07

### Added
- 22 new IQ-based SDF primitives (31 → 53 total):
  Tube, Barrel, Diamond, ChamferedCube, SchwarzP, Superellipsoid, RoundedX,
  Pie, Trapezoid, Parallelogram, Tunnel, UnevenCapsule, Egg,
  ArcShape, Moon, CrossShape, BlobbyCross, ParabolaSegment,
  RegularPolygon, StarPolygon, Stairs, Helix
- GLSL/WGSL/HLSL shader transpiler support for all 22 new primitives
- 88 new unit tests (458 → 546 total)
- New categories: 2D→3D prisms (HD2D), 3D revolution bodies, native 3D shapes

## [0.3.0] - 2026-02-07

### Added
- 16 new IQ-based SDF primitives (15 → 31 total):
  RoundedBox, CappedCone, CappedTorus, InfiniteCylinder, RoundedCylinder,
  TriangularPrism, CutSphere, CutHollowSphere, DeathStar, SolidAngle,
  Rhombus, Horseshoe, Vesica, InfiniteCone, Heart, Gyroid
- GLSL/WGSL/HLSL shader transpiler support for all 16 new primitives
- 57 new unit tests (401 → 458 total)

## [0.2.0] - 2026-02-03

### Added
- ALICE-Baker v0.2 (Deep Fried Edition)
  - Instruction Fusion: inline leaf nodes, reduce temp variables
  - Division Exorcism: pre-compute 1/k for smooth operations
  - Scalar Expansion: Udon CSG ops expanded to Mathf.Min/Max
  - Translate Scalarization: Vector3 split into float xyz for leaf children
  - Smooth Op Inline: SmoothUnion/Intersection/Subtraction fully expanded
  - Live Preview: JSON change detection with auto-update
- UPM (Unity Package Manager) support
- `#if UDONSHARP` compile guard for non-VRC environments
- Assembly Definitions (.asmdef) for Runtime and Editor

## [0.1.0] - 2026-02-03

### Added
- Initial release
- AliceSDF_Include.cginc: 6 primitives + 10 operations
- AliceSDF_LOD.cginc: Deep Fried dynamic LOD (128/64/32 steps)
- AliceSDF_Raymarcher.shader: Main shader with SV_Depth
- AliceSDF_Primitives.cs: Pure C# SDF mirror
- AliceSDF_Math.cs: Vector math helpers
- AliceSDF_Collider.cs: UdonSharp player collision
- AliceSDF_Baker.cs: Editor tool for JSON → Shader + Udon generation
