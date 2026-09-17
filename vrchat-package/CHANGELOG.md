# Changelog

## [Unreleased]

### Added
- Mochi sample on desktop: hold Use (left click) on a mochi and the point
  on the view ray nearest its centre becomes a virtual right hand, so grab /
  drag / split / merge run through the same `ProcessHand` as VR; the
  distance is fixed on the click (`CursorDistance` = sphere-traced hit,
  then the depth of that mochi's centre, so the grab threshold is met; a
  click past every mochi does nothing), releasing the button drops the
  mochi (`ReleaseHand`), and a held mochi is kept above the floor when the
  view ray is steep. Host scenario: hit distance, cursor depth, miss, grab
  after the dwell, drag, release. ClientSim: `_inputUse` via
  `RunInputEvent` grabbed mochi 1 at the cursor, followed a pitch change,
  dropped on release.
- Mochi sample: the player's body presses into the mochis. The collider
  sends a body capsule (feet to eyes, `playerRadius` 0.3 m) to the shader
  (`_PlayerCapA` / `_PlayerCapB` / `_PlayerDentK`) and the shader
  smooth-subtracts it from the mochi union, so a dent forms where the player
  leans in and springs back when they step away (the collider still resolves
  against the undented surface: the dent is where the player already is).
  The pushed mochi also gives way: the separation is split by mass
  (`playerMass` 60 kg vs 4/3 pi r^3 x `mochiDensity`), the mochi slides on
  the floor by its share (0.25 for r = 0.35) and the player takes the rest;
  a mochi held in a hand does not yield. Host scenario: mass share, slide
  on the floor, held mochi. ClientSim: player 0.25 m from a mochi's centre
  moved +0.142 m, the mochi -0.047 m, capsule following the player, dent
  visible with the avatar hidden.

## [0.5.0] - 2026-09-17

`package.json` had stayed at 0.2.0 through the 0.3.0 and 0.4.0 entries
below, so a project that imported the samples in February kept its
February copy in `Assets/Samples/ALICE-SDF for VRChat/0.2.0/` and the
Package Manager saw nothing to update. This release bumps it so the
samples re-import.

### Added
- `Editor/SampleSceneGenerator.cs` also builds the three interactive
  samples (DeformableWall / Mochi / TerrainSculpt): bounding cube at the
  README size with its bottom face on the SDF ground, the sample material,
  and the `*_Collider` UdonSharp behaviour on the cube (UdonSharp creates
  the backing `UdonBehaviour`). Before, only Basic / Cosmic / Fractal / Mix
  were generated and the interactive ones were a manual setup. In edit mode
  the Mochi scene shows the ground only: the mochis are placed by the
  behaviour at Play.
- `.meta` files for every asset in `Editor/`, `Runtime/`, `Packages/` and the
  package root are in the repository, as a VPM package ships them. They
  were never tracked, so every `file:`-linked project generated its own
  GUIDs (different per machine) and left 21 untracked files in the working
  tree.
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
- Mochi sample, player collision: a single sample at the feet (5 cm under
  the floor) sat below every mochi's centre, so walking into a mochi from
  the side barely registered and what push there was pointed up — the
  player ended up standing on the mochi (seen in the VRChat client). The
  body is now sampled from the feet to the eyes
  (`GetAvatarEyeHeightAsMeters`, `bodySamples` = 5) and the deepest sample
  decides the push, sideways whenever the surface allows it (lifting the
  player only hands them to gravity and the next frame's push); a 5 mm dead
  band stops the geometric approach from teleporting the player every
  frame forever. `PlayerPushOut` is a public pure function; the host
  scenario checks a sideways walk-in (pushed back along the approach line,
  never lifted, push reaches exactly zero, clear of the mochi), the floor
  away from mochis, and the mochi's own column (never pushed down).
  ClientSim: player at 0.25 m from a mochi's centre pushed to 0.45 m in
  0.5 s with y = 0 throughout, then no further teleport.
- Mochi sample, shader: a thin dark line followed every mochi / ground
  contour. Not lighting: it survived `_ShadowEnabled = 0` and AO = 1. A ray
  grazing a silhouette takes ever smaller steps (smaller still inside a
  smooth union's blend zone, where |grad| < 1) and ran out of the 128-step
  budget a hair short of the surface; treated as a miss, it wrote the far
  depth and the world's floor behind the volume showed through (confirmed
  by colouring the miss path magenta). The march now keeps its closest
  approach and, when the budget runs out within one pixel footprint of the
  surface (`NEAR_MISS_PER_M` = 0.002 × ray length), shades that point.
- Mochi sample, shader: a sharp dark ring around every mochi's base, ending
  abruptly at the edge of the ground blend zone. The AO integral (h − d)
  sampled the polynomial smooth union, which under-reports distance inside
  its blend zone (up to k/4 short), and read the shortfall as occlusion.
  AO now samples the hard union (`min(ground, mochis)`, exact outside the
  geometry), so only real geometry occludes; the contact shading is a
  smooth gradient.
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
