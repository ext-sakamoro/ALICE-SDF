# Changelog

## [Unreleased]

### Fixed
- Mochi: the law was anchored to the world origin (ground `y = 0`, the
  initial mochis around `(0, 0)`), so three product prefabs placed 6 m apart
  all spawned their mochis at the origin (first Build & Test of the
  product: "only green"). The collider now anchors everything to
  `transform + Ground Offset` (the cube's bottom centre) and pushes it to
  the shader as `_Origin`: spawn ring, resting height, merge floor, the
  desktop cursor floor clamp and the rendered ground plane all follow the
  prefab. The golden layout is unchanged at the origin (host tests anchor
  there); a placement scenario checks an offset.
- The four static samples (Basic / Cosmic / Fractal / Mix) had no working
  collision in VRChat. Their colliders derive from the package's
  `AliceSDF_Collider`, but that assembly had no UdonSharp assembly
  definition (`AliceSDF.Runtime.UdonSharp.asset`, now shipped), so the base
  class could not run as Udon, and the samples hid `Evaluate` with `new`
  instead of overriding it, so a base that did run would have pushed against
  the base-class demo sphere. Now: the U# assembly definition, `override`
  (UdonSharp resolves virtual calls), and the base collider samples the body
  feet to eyes and pushes the deepest wall-like contact sideways with a dead
  band, leaving floor-like contacts to the scene's colliders (no bobbing).
  The scene generator also creates the base class's program asset (UdonSharp
  throws an internal `ArgumentNullException` without it).
- Cosmic and Mix colliders were static snapshots (and Cosmic lacked the
  ring's tilt and twist and the asteroids) while the shaders animate:
  rendering and collision disagreed. The colliders now carry the full shader
  law with `animTime` (`Time.timeSinceLevelLoad`, the shader's `_Time.y`);
  the shader's 15-degree ring tilt uses exact constants (0.9659 → 0.96592582).
- Static sample shaders: `Cull Off`, closest-approach acceptance,
  `_LightDir`, Basic gets a soft contact shadow and a `_FogDensity`, Mix's
  AO samples the hard union (no dark ring at the ring / planet junction).
- DeformableWall: the player collided with the flat box while the shader
  drew dents (the collider "simplified" the wall for speed), and the same
  foot-only push as Mochi's first version. The collider now evaluates the
  dented wall (16 spheres per sample is cheap), samples the body from the
  feet to the eyes and pushes the deepest point out sideways with a dead
  band. Dent strengths are stored instead of `Time.time` stamps, so they
  can be synced; a hand or the cursor following a fresh dent inward no
  longer drills through the 0.4 m wall (impacts are measured against the
  undented face and refused inside a live dent's hollow; a hit at a live
  dent refreshes it). Shader: `Cull Off`, closest-approach acceptance,
  hard-union AO, `_LightDir` / soft shadow / `_FogDensity`, wall size and
  blend factors pushed from the collider; the duplicate `_WallWidth`
  declaration is gone.
- DeformableWall `Decay Speed` default 0.5 → 0.15: at 0.5 a dent was
  visually gone in 3 s (user, first Build & Test: "it heals too fast");
  now half strength after 4.6 s, gone after ~30 s.
- TerrainSculpt: the player could not stand on the terrain. The collider
  pushed the feet out of the SDF every frame (the same bobbing loop Mochi
  had) and the scene still needed a flat floor collider, so a dug hole was
  drawn but never entered. The script now moves a small invisible box
  (`TerrainSupport`, created by the scene generator at the scene root) onto
  the highest SDF surface under the foot every frame (level; five points
  across the foot so a ridge holds until the feet leave it), so VRChat's
  controller stands on the terrain as it is now: dig
  under yourself and you fall, build under yourself and you are lifted onto
  the top, walk into the steep flank of a tall hill and it pushes you back
  like a wall (with the Mochi dead band, so the push stops), anything under
  a 0.3 m step you walk up. No floor collider in a TerrainSculpt scene.
  Sculpt buffer 48 → `Sculpt Capacity` (Inspector, default 96, array 128):
  in a 7-minute session the 48-slot ring wrapped eight times and the first
  hills vanished under the player's feet.
- TerrainSculpt shader: `Cull Off` (the player walks inside the volume
  cube; with back-face culling the terrain vanished from inside), the
  closest-approach acceptance of the Mochi raymarcher (no dark seam along
  hill silhouettes), ambient occlusion against the hard union (no dark ring
  at the foot of a hill), `_LightDir` / soft contact shadow / `_FogDensity`
  properties instead of constants; `_AddSmooth` / `_SubSmooth` are pushed
  from the collider every frame so collision and rendering cannot drift.
- TerrainSculpt shader: the per-sphere distance cull (skip a sculpt farther
  than r + 2k) returned the plane distance, which above a tall stack is
  larger than the true distance; rays overshot into the column and stopped
  inside it (a black cavity, a floating cap, a dark dot on every apex — seen
  in the VRChat client the first time a column was built). Every sculpt is
  now folded, as in the collider.
- `SampleSceneGenerator`: a generated DeformableWall / TerrainSculpt scene
  had the `*_Collider` component but no backing `UdonBehaviour`, so nothing
  ran at Play. UdonSharp backs a proxy only when a `UdonSharpProgramAsset`
  exists for its script, and it does not create one inside the unsaved
  scene the generator builds in (Mochi worked only because an earlier scene
  had made its asset). The generator now creates the program asset next to
  the scenes (`SDF_<Sample>_UdonProgram.asset`), compiles it, and after all
  scenes are saved opens each interactive scene again so UdonSharp creates
  the backing, then saves; the log says `1/1 UdonSharp behaviours have a
  backing UdonBehaviour.` per scene. `AliceSDF.Editor.asmdef` references
  `UdonSharp.Runtime` / `UdonSharp.Editor` / `VRC.Udon` / `VRC.Udon.Editor`
  under the same `UDONSHARP` version define as the runtime assembly.
  Verified in the Editor and headless (`GenerateAllBatch`) on 2022.3.22f1 +
  SDK 3.10.1.
- `AGENTS.md`: strings and checks aligned with the real logs (`click
  missed`, the backing-behaviour log line, the headless expectations and
  exit codes, the GUID-preserving way to move an old sample folder,
  T12 SDK-panel `IndexOutOfRangeException` on scenes without a descriptor,
  T13 collider without `UdonBehaviour`).

### Added
- Mochi: optional textures from the Inspector (`Mochi Texture` / `Ground
  Texture` + scale + strength, pushed to `_MochiTex` / `_GroundTex` with
  `Apply Colors`). The shader projects them triplanar: the mochi texture in
  the nearest mochi's own frame scaled by its radius (it moves, splits and
  grows with the mochi), the ground texture in world space around the
  ground point. Strength 1 shows the texture as is, lower tints it.
- **AliceMochi product** (`ALICE-SDF/Build Mochi Product`, `Editor/MochiProductBuilder.cs`):
  generates a stand-alone `Assets/AliceMochi/` (script, shader, three preset
  materials + prefabs Mochi / Slime / Water, program asset, README EN / JP,
  LICENSE) from the Mochi sample and exports `Product~/AliceMochi_<version>.unitypackage`
  for BOOTH. No dependency on this package: verified by importing it into a
  fresh VCC World project (0 compile errors, UdonSharp compiled, prefab backed,
  shader 0 errors). Three phases across the script compile (write → program
  asset → prefabs on a later tick, UdonSharp backs scene proxies on its own
  tick). The sample gains the product's knobs: `Look` (colours pushed to the
  material when `Apply Colors` is on) and `Mochis` (`Use Custom Layout`:
  count / radius / ring; off keeps the five-mochi golden layout).
- The TerrainSculpt and DeformableWall samples are published as the VRChat
  worlds **TerrainSculpt** (`wrld_6c134920-6d49-4d66-8b5d-11bbf79ef061`) and
  **DeformableWall** (`wrld_3fd67c2b-f712-4d6c-be7f-202eb2ecb2b9`), private
  while testing; **Mochi** is in Community Labs (2026-09-18). Both READMEs
  link them. The scene generator adds the `PipelineManager` to `VRCWorld`
  (the SDK adds it in the Inspector, not via `AddComponent`; without it
  there is no blueprint id to upload) and sets each material's `_LightDir`
  from the scene light.
- Static sample scenes are playable: the generator adds the `VRCSceneDescriptor`
  + spawn to all seven, a floor collider to Basic, a viewing platform under
  the spawn of Cosmic / Fractal / Mix (skybox cleared: a raymarch miss is
  black space, not the world's sky).
- `HostTests~/StaticParity` + four goldens (`examples/vrchat_{basic,cosmic,fractal,mix}_golden.rs`,
  2535 / 2907 / 3375 / 1989 points, the animated ones at t = 0) + the base
  collider's push scenario; `scripts/vrchat-host-parity.sh` runs all seven
  samples (the example name is passed to each project).
- `Log Events` on the base collider (`[SDF] push ...`).
- DeformableWall on desktop: hold the left button (Use) to punch the wall
  where you look (`Cursor Max Dist` 4 m); walking into the wall presses
  your body capsule into it (`_PlayerCapA/B`, local) while it pushes you
  back. Multiplayer: the dent array is `[UdonSynced]` (manual, the hitting
  player takes ownership per contact, 10 Hz while any dent is alive,
  everyone recovers locally between packets). `Log Events` writes
  `[Wall] ...` per impact / click miss / push / ownership / received.
- `HostTests~/DeformableWallParity` + `examples/vrchat_deformable_wall_golden.rs`
  (5100 points, four dents of mixed strength) + a 29-check impact / decay /
  slot / push / click scenario; the parity script and CI job run it.
- Scene generator: the Mochi / DeformableWall scenes get an invisible floor
  collider at y = 0 (their SDF ground is drawn, not walked on).
- TerrainSculpt on desktop: hold the left button (Use) to build and the
  right button (Drop) to dig at the point where the view meets the terrain
  (`Cursor Max Dist`, 6 m); the cursor glow shows what a click would do.
- TerrainSculpt multiplayer: the sculpt buffer is `[UdonSynced]` (manual,
  owner-authoritative, 10 Hz while changed; the sculpting player takes
  ownership at the start of a stroke), so everyone stands on the same
  terrain and late joiners receive it. `Log Events` writes one
  `[Terrain] ...` line per add / dig / click / ownership / floor drop or
  rise under the player / lift / wall push.
- `HostTests~/TerrainSculptParity` + `examples/vrchat_terrain_sculpt_golden.rs`:
  the terrain collider is compiled on the host and its `EvaluateSdf`
  compared with `alice_sdf::eval` (4335 points) plus a sculpt / stand /
  wall / buried scenario; `scripts/vrchat-host-parity.sh` now runs every
  sample project. `UnityEngineStub.cs` moved to `HostTests~/Shared/`.
- The Mochi sample is published as the VRChat world **Mochi**
  (`wrld_0cb72970-948e-4212-b955-fd3dd567aa42`, private while testing, PC
  only); both READMEs say how to get in.

### Changed
- Mochi sample: splitting is an explicit action. Pulling a held mochi no
  longer tears it (`splitOnPull`, off by default; the old 2.5 r rule was
  meant for a VR hand, and the desktop cursor crosses it with a glance, so
  every carried mochi split "at a certain height" and, its radius now
  smaller, dropped at the shrunken 4 r a moment later — both seen in the
  client log). Grip / right click splits. The desktop cursor drops only on
  button up; a VR hand still drops the mochi after carrying it 4 r and,
  being inside it, grabs it again after the dwell, so a slow hand carries
  and a flick lets go. Host scenario: carry 3.5 r without a split, drop
  past 4 r, immediate re-grab; the tear-on-pull scenario runs with
  `splitOnPull` on.

### Added
- Mochi sample is networked: `mochiPos` / `mochiR` / `mochiCount` are
  `[UdonSynced]` (manual sync), the owner runs gravity and merging and
  serializes at 10 Hz while anything changed, grabbing or walking into a
  mochi takes ownership once per grab / contact, late joiners spawn nothing
  and receive the owner's state (`[Mochi] received N mochis (was M) from
  <owner>` in the log when the count changes), each player's body dent is
  local. One player sculpts at a time in practice. Host scenario: the owner
  spawns, spawning and moving mark the state dirty. Two VRChat clients on
  one PC (Steam + `--profile=1`, same local room URL): the second logged
  `received 5 mochis (was -1) from sakamoro` on join.
- Mochi sample: right click on desktop (`InputDrop`) / the grip in VR (`InputGrab`) splits
  the mochi you hold without pulling (`SplitHeld`, same volume-conserving
  split as the pull, the other half stays at the grab origin; refused at
  the minimum size or 16 mochis, logged). With Log Events on, a release
  says how far you pulled and how far a split needed (`max pull 0.42 m,
  split at 0.75 m`).
- `AGENTS.md`: for coding agents a user points at this package. One install
  path with a file- or log-level check per step, the headless commands, the
  logs to read per platform, what can be verified without Unity
  (`scripts/vrchat-host-parity.sh`) and what only the user's client can, the
  Mochi sample split into law (shape, split / merge volume conservation,
  settle, mass share — identical in every host) and binding (data channel,
  input, player collision, bounding volume — rewritten per host), the seven
  invariants to re-check after an edit, design guidance for moving Mochi to
  an avatar accessory (no Udon: N explicit `Vector` properties driven by the
  Animator, march in object space, no body dent, PC only) or a pickup, the
  request text to send the user for a Build & Test, and eleven known errors
  with the string to grep, the cause and the fix. Both READMEs link to it.
- Menu **ALICE-SDF > Import All Samples** (`SampleSceneGenerator.ImportAllSamples`):
  imports every sample of this package that is not yet in `Assets/Samples/`
  (`PackageManager.UI.Sample.FindByPackage` + `Import()`), so Generate Sample
  Scenes has its shaders and colliders without visiting the Package Manager.
- Headless entry points for scripts and agents:
  `SampleSceneGenerator.ImportAllSamplesBatch` and `GenerateAllBatch`
  (`Unity -batchmode -quit -executeMethod …`, two invocations so the imported
  scripts compile in between). No dialog, no scene opened, exit code 1 when
  nothing was imported / no scene created. Not verified in Unity in this
  environment (no editor on the machine); the menu path is unchanged apart
  from the shared core.
- README / README_JP: Requirements state the Creator Companion Worlds
  project and the ASCII-only project path; a Troubleshooting table (seven
  rows) and the agent pointer.
- `Documentation~/mochi_desktop.gif` (11 s from the VRChat client, 3.5 MB)
  at the top of the Mochi section of both READMEs: body dent, click grab,
  split, merge on desktop. Unity skips the `~` folder.
- Mochi sample: `Log Events` (Inspector, off by default) writes one
  `Debug.Log` line per grab / split / release / merge / desktop click
  (hit or miss) / push (once per contact, with the mochi's mass share) as
  `[Mochi] ...`, so the VRChat client `output_log_*.txt` shows what
  happened without a debugger. Nothing is logged per frame. The host
  scenario runs with it on and prints the four VR events.
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

### Fixed
- README / README_JP install: the git URL pointed at `sakamoro/ALICE-SDF`,
  which does not exist; the repository is `ext-sakamoro/ALICE-SDF`.

### Removed
- `Packages/manifest.json` + `packages-lock.json` (and their `.meta`) that
  had been committed inside the package since the first commit: they were a
  Unity project's manifest, not part of a UPM package, and Unity imported
  them as assets.

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
