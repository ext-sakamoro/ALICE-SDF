# AGENTS.md — ALICE-SDF for VRChat (`com.alice.sdf`)

This file is for coding agents (Claude Code, Codex, Cursor, …) that a user has
asked to install, verify, debug or adapt this package. It states what is true
about the package, what can be checked without Unity, what only the user can
check, and which parts of the Mochi sample are portable law versus
host-specific binding. `README.md` is the human introduction; nothing here
contradicts it, but this file is written to be followed step by step.

Package facts (read `package.json`, do not guess):

| Fact | Value |
|------|-------|
| Package name | `com.alice.sdf` |
| Repository | `https://github.com/ext-sakamoro/ALICE-SDF` (public), package at `vrchat-package/` |
| Git URL for `Packages/manifest.json` | `https://github.com/ext-sakamoro/ALICE-SDF.git?path=vrchat-package` |
| Unity | 2022.3.x (the VRChat Creator Companion version) |
| Host | VRChat **world** project (VRChat SDK Worlds, which bundles UdonSharp) |
| Native code | none — HLSL shaders + UdonSharp (C#) only |
| Layout | `Runtime/Shaders` (raymarch kernel + LOD), `Runtime/Udon` (53 primitives in C#, player collider), `Editor/` (Baker, scene generator), `Samples~/SDF Gallery/<7 samples>` (each = `*_Raymarcher.shader` + `*_Collider.cs` + `*.asdf.json`), `HostTests~/` (checks that run without Unity), `Documentation~/` |

Folders ending in `~` are invisible to Unity; they are for git, CI and you.

---

## 1. Install → Mochi running: one path, every step verifiable

Do the steps in order. Each has a check whose result you can read from a file
or a log; "it should work" is not a check.

| # | Step | How | Check |
|---|------|-----|-------|
| 1 | Project | A VRChat **Worlds** project created by the VRChat Creator Companion (VCC), Unity 2022.3. The project path must be ASCII-only (see §5 T1). | `Packages/vpm-manifest.json` lists `com.vrchat.worlds`; the path has no non-ASCII characters |
| 2 | Add the package | Add to `Packages/manifest.json` → `"dependencies"`: `"com.alice.sdf": "https://github.com/ext-sakamoro/ALICE-SDF.git?path=vrchat-package"` (or `"file:<absolute path to vrchat-package>"` for a local checkout). Do not edit `vpm-manifest.json`. Unity resolves on focus + refresh (Ctrl+R); it takes minutes. | `Packages/packages-lock.json` has a `com.alice.sdf` entry; `Packages/com.alice.sdf/package.json` exists (git install) or the `file:` path resolves; Console has no red line mentioning `AliceSDF` |
| 3 | Import the Mochi sample | Menu **ALICE-SDF > Import All Samples** (imports every sample that is not yet imported), or Package Manager > ALICE-SDF for VRChat > Samples > Import. Headless: see below. | `Assets/Samples/ALICE-SDF for VRChat/<version>/SDF Gallery - Mochi/SampleMochi_Collider.cs` exists; `<version>` equals `package.json`'s `version` |
| 4 | Generate the scene | Menu **ALICE-SDF > Generate Sample Scenes**. Headless: see below. It sizes the bounding cube (Mochi: centre (0, 1, 0), scale (4, 2, 4)), assigns the material, adds `SampleMochi_Collider`, creates the `UdonSharpProgramAsset` the script needs if none exists (`SDF_<Sample>_UdonProgram.asset` next to the scenes) and, after every scene is saved, opens each interactive scene again so UdonSharp creates the backing UdonBehaviour, and saves it. The menu shows a modal dialog ("7 created, 0 skipped") that blocks the Editor until OK; an agent driving the menu must dismiss it (or use the headless method, which has no dialog). | `Assets/AliceSDF_SampleScenes/SDF_Mochi.unity` exists; Editor.log contains `[ALICE-SDF] Created scene: Assets/AliceSDF_SampleScenes/SDF_Mochi.unity`, `[ALICE-SDF] Assets/AliceSDF_SampleScenes/SDF_Mochi.unity: 1/1 UdonSharp behaviours have a backing UdonBehaviour.` and **no** `[ALICE-SDF] AliceSDF.Samples.SampleMochi_Collider not found` |
| 5 | Play in the Editor | Open `SDF_Mochi.unity`, press Play. With the VRChat SDK's ClientSim a local player spawns; on desktop, hold left click on a mochi to grab it, right click while holding to split. Set `Log Events` on the collider first. | Five mochis on a plane are visible; Console / Editor.log shows `[Mochi] grab …` after a click on a mochi, `[Mochi] click missed (view from …)` after a click on empty space |
| 6 | Build & Test in VRChat | **User only** (needs the VRChat client logged in on that machine). Ask for it with the request in §4. | The client's `output_log_*.txt` contains `[Mochi] grab` / `split` / `merge` / `push` lines |

Headless equivalents of steps 3 and 4 (two separate Unity invocations — the
imported scripts compile between them; exit code 1 on failure). Verified
2022.3.22f1 + SDK 3.10.1 (Windows): `ImportAllSamplesBatch` logs
`[ALICE-SDF] Import All Samples: N imported, M already present, 0 failed
(com.alice.sdf 0.5.0).` and exits 0 (also when everything is already
present: `0 imported, 7 already present`); `GenerateAllBatch` logs seven
`Created scene` lines plus `1/1 UdonSharp behaviours have a backing
UdonBehaviour.` for Mochi / DeformableWall / TerrainSculpt and exits 0;
with no sample imported it logs `[ALICE-SDF] No sample scene created (7
skipped). Run ImportAllSamplesBatch (or Package Manager > Samples > Import)
first.` and exits 1:

```
# Windows
"E:\2022.3.22f1\Editor\Unity.exe" -batchmode -quit -nographics -projectPath "E:\VRChatProjects\Mochi" -logFile "%TEMP%\alice-import.log" -executeMethod AliceSDF.Editor.SampleSceneGenerator.ImportAllSamplesBatch
"E:\2022.3.22f1\Editor\Unity.exe" -batchmode -quit -nographics -projectPath "E:\VRChatProjects\Mochi" -logFile "%TEMP%\alice-scenes.log" -executeMethod AliceSDF.Editor.SampleSceneGenerator.GenerateAllBatch
# macOS
/Applications/Unity/Hub/Editor/2022.3.22f1/Unity.app/Contents/MacOS/Unity -batchmode -quit -nographics -projectPath ~/VRChatProjects/Mochi -logFile /tmp/alice-import.log -executeMethod AliceSDF.Editor.SampleSceneGenerator.ImportAllSamplesBatch
```

Adjust the editor path and version to the machine (`vpm list`/VCC shows them).
The Unity Editor must not already have the project open. Read the `-logFile`
afterwards; grep for `[ALICE-SDF]` and `error CS`.

Logs to read, per platform:

| What | Where |
|------|-------|
| Editor console | Windows `%LOCALAPPDATA%\Unity\Editor\Editor.log`, macOS `~/Library/Logs/Unity/Editor.log` (strip rich-text tags: `sed 's/<[^>]*>//g'`) |
| VRChat client | Windows `%USERPROFILE%\AppData\LocalLow\VRChat\VRChat\output_log_*.txt` (newest file) |
| Shader compiler | `<project>/Logs/shadercompiler-UnityShaderCompiler.exe*.log` (`ok=1` is the preprocess; real errors appear in the Console on first draw) |

---

## 2. What you can verify without Unity — and what you cannot

**Can (from a clone of the repository, `cargo` + `dotnet` installed):**

```
scripts/vrchat-host-parity.sh        # from the repository root; CI job "vrchat-host"
```

It compiles each interactive sample's `*_Collider.cs` as C# 7.3 with warnings
as errors against `HostTests~/Shared/UnityEngineStub.cs`, compares its
`EvaluateSdf` with `alice_sdf::eval` of the same scene (tolerance 1e-5) and
replays a behaviour scenario:

| Project | Golden | Scenario |
|---------|--------|----------|
| `HostTests~/MochiParity` | `examples/vrchat_mochi_golden.rs`, 1521 points | grab → split → release → settle → merge, desktop click, player push |
| `HostTests~/TerrainSculptParity` | `examples/vrchat_terrain_sculpt_golden.rs`, 4335 points | sculpt (cooldown / stroke / buffer wrap), surface search over a hole and a stacked hill, step / wall / buried contact, desktop cursor ray |
| `HostTests~/DeformableWallParity` | `examples/vrchat_deformable_wall_golden.rs`, 5100 points | impact (undented face, cooldown, hollow refusal, refresh, slot reuse), recovery, dented law, body push, desktop ray |
| `HostTests~/StaticParity` | `examples/vrchat_{basic,cosmic,fractal,mix}_golden.rs` (2535 / 2907 / 3375 / 1989 points, t = 0) | base collider: wall push sideways, floor contact ignored, dead band, override dispatch; Cosmic / Mix move with `animTime` |

Green means: the C# is valid, and the collider's law equals the Rust law. It
says nothing about Udon, the shader or VR.

**Cannot (Unity is the only compiler / runtime):**

- UdonSharp acceptance of the collider (the `#if UDONSHARP` branch).
- Shader compilation and rendering — `Samples~/…/SampleMochi_Raymarcher.shader`
  has no host-side compile in CI; the Unity Console on first draw is the gate.
- Hand tracking, ClientSim input, networking, the VRChat client.

When a change touches any of these, report it as "code in, not verified in
Unity" and hand the user the §4 request. Do not report "works".

---

## 3. Mochi: law versus binding

The sample is two layers. Keep the first when you move it anywhere; rewrite
the second per host.

### Law (host-independent, must stay identical wherever it runs)

| Law | Shader (`SampleMochi_Raymarcher.shader`) | Collider (`SampleMochi_Collider.cs`) |
|-----|-------------------------------------------|--------------------------------------|
| Shape: `smin(ground, smin_i(sphere_i, blendK), groundK)` — polynomial smooth-min, `k` is a length | `map()` / `mapMochi()` | `EvaluateSdf()` (with ground, what the golden test checks), `EvaluateMochiSdf()` (mochis only, what the player collides with) |
| Player body dent: smooth-subtract of a capsule (feet → eyes, radius `playerRadius`) from the mochi union | `_PlayerCapA/_PlayerCapB/_PlayerDentK` | `playerCapA/B`, `dentK` |
| Split conserves volume: `r_new = r · cbrt(0.5)`; triggered by a button (`SplitHeld`), the pull-to-tear rule is behind `splitOnPull` (off) | — | `SplitRadiusScale = 0.7937005f`, `SplitHeld` |
| Merge conserves volume: `r = cbrt(r1³ + r2³)` | — | `CheckMerge` |
| Settle: `y ← lerp(y, r, 1 − exp(−g·dt))` (frame-rate independent) | — | `ApplyGravity` |
| Push share by mass: `share = m_player / (m_player + ρ·4/3·π·r³)` | — | `YieldMochi` (`mochiDensity`, `playerMass`) |
| Static snapshot of the law for the Baker | `mochi.asdf.json` (5 spheres, `k` 0.5 / 0.15) | — |
| Reference implementation of the same formula | `alice_sdf` crate, `examples/vrchat_mochi_golden.rs` | — |

### Binding (host-specific; this sample binds to a VRChat world)

| Binding | Where | Replace with, on another host |
|---------|-------|-------------------------------|
| Data channel collider → shader | `SyncShader()`: `Material.SetVectorArray("_MochiData", …)` + `SetFloat`/`SetVector` every frame | anything that writes material properties (see §3.2) |
| Input | VR: `VRCPlayerApi.GetTrackingData(LeftHand/RightHand)`, grip (`InputGrab`) splits the held mochi; desktop: `InputUse` (left click) + the view ray (`ProcessDesktopCursor`), right click (`InputDrop`) splits, button up drops | the host's hand / pointer / button source |
| Player collision | `localPlayer.GetPosition()` → `EvaluateMochiSdf` on body sample points → `TeleportTo` | the host's character controller, or nothing |
| Bounding volume | a Unity Cube (0, 1, 0) × (4, 2, 4); rays start on its surface, march in **world space** (`_WorldSpaceCameraPos`) | any mesh; march in object space if the mesh moves |
| Quality | LOD tiers by camera distance (128 / 64 / 32 steps; ε 1e-4 / 1e-3 / 5e-3) in the shader | tune to the host's budget |
| Network | owner-authoritative manual sync: `mochiPos` / `mochiR` / `mochiCount` are `[UdonSynced]`, the owner runs gravity + merging and serializes at 10 Hz while dirty, a grab or a body contact takes ownership, late joiners receive the state (`OnDeserialization`); the body dent is local | the host's replication, or none |

### Invariants — check these after any edit, in any host

1. `SampleMochi_Collider.MaxMochi == MOCHI_MAX (shader) == 16`. `SetVectorArray` fixes the array length on its first call; a mismatch is silent.
2. Every term of the collider's `EvaluateSdf` equals the shader's `map()` (same `k`, same order, same primitives). The golden test is the check for the C# side; the shader side has only Unity. Change both or neither.
3. `blendK` / `groundK` are owned by the collider and pushed every frame; the shader's Inspector values are fallbacks for a material with no script.
4. The bounding volume contains every mochi plus its dent at all times; anything outside is never rendered and never collides.
5. The player collides with `EvaluateMochiSdf` (no ground term). With the ground included, feet 5 cm under `y = 0` are a permanent penetration and the player is teleported every frame (0.5.0 fix).
6. The marcher keeps the closest approach (`bestD`, `bestT`) and shades it when the step budget runs out; without it silhouettes show a dark line where the world behind shows through. Do not "simplify" the loop.
7. Polynomial smooth-min has `|∇d| < 1` inside its blend band, so the distance is an under-estimate there. AO and shadows sample the hard union; any new domain distortion (twist, scale, shear) needs the returned distance multiplied by ≈ 0.8.

---

### 3.1 Re-using Mochi in another VRChat world

Copy `SampleMochi_Raymarcher.shader` + `SampleMochi_Collider.cs` (or import the
sample) and re-run the generator, or place the cube by hand (§1 step 4).
Change the shape by editing `mapMochi()` and `EvaluateSdf()`/`EvaluateMochiSdf()`
together, then run `scripts/vrchat-host-parity.sh` — it fails the moment the
two laws differ. New static shapes come from the Baker
(`Window > ALICE-SDF Baker`, paste an `.asdf.json`), which emits shader + Udon
+ prefab from one definition.

### 3.2 Transplanting to an avatar (costume / accessory) — design guidance, not verified on an avatar

An avatar cannot run Udon, so every binding in §3 changes; the law does not.

| Constraint | Consequence |
|------------|-------------|
| No scripts on avatars | No `SetVectorArray`. The 16-entry array becomes N explicit material properties (`_Mochi0` … `_Mochi{N-1}` as `Vector`, xyz = centre in **object space**, w = radius). Keep N small (2–4 for an accessory); the loop bound becomes N. |
| Values change only through the Animator | Animator layers write material `float`/`Vector` components (e.g. radius from a PhysBone stretch/squish parameter, position offsets from a Contact Receiver's proximity). Anything that must move per frame without an Animator has to be in the vertex shader. |
| The mesh follows a bone | The bounding mesh is a child of the bone (Head, Chest, hand). March in **object space**: transform `_WorldSpaceCameraPos` and the ray by `unity_WorldToObject`, do `map()` in object space, transform the hit back for depth. The sample marches in world space and must be changed here. |
| No world player position | Drop the body dent (`_PlayerCap*`), or drive a single dent from a Contact: the Contact's proximity → an Animator float → `_PlayerDentK`/a dent position preset. |
| Depth | Keep `SV_Depth`; the accessory then occludes and is occluded correctly. |
| Quest | Quest avatars accept only the VRChat mobile shaders; this is PC-only. Quest **worlds** accept custom shaders, at a raymarch cost the world's own performance budget has to carry. |
| Performance ranking | A raymarch shader does not change the avatar's rank, but its GPU cost is real; keep the bounding mesh small and N low. |

What stays byte-for-byte: `sdSphere`, `opSmoothUnion`, the material blend
weight, the normal / AO / soft shadow code, the LOD tiers, the closest-approach
fallback. What you rewrite: the property block, `mapMochi()`'s loop bound and
source of centres, the vertex stage (object space), and the removal of
`_MochiCount`. What disappears: everything in the collider (input, split /
merge, settle, mass share, `[UdonSynced]` state) — an avatar's mochi is
shape only, its motion comes from the Animator. There is no collider on an avatar, so `HostTests~` does not
apply; the only gates are the Unity Console and the avatar in the client.

### 3.3 A pickup / prop in a world

Same binding as the world sample plus `VRC_Pickup` on the bounding object;
the collider keeps working (the cube moves, the mochis are in world space —
either keep them in world space and let the cube be large enough, or switch
to object space as in §3.2 and offset the player capsule into object space).
The pickup's owner and the mochi state's owner (`Networking.SetOwner` on
grab / contact) are the same object, so ownership already follows whoever
holds it.

---

### 3.4 The other interactive samples: the same split

TerrainSculpt and DeformableWall follow the Mochi layout exactly (law in
`Evaluate*Sdf` = the shader's `map()`, binding in `PostLateUpdate` /
`SyncShader` / the input overrides, `[UdonSynced]` state, `Log Events`).
Their laws, for the same re-use:

| Sample | Law (identical in shader and collider, golden-checked) | Host-specific part |
|--------|--------------------------------------------------------|--------------------|
| TerrainSculpt | plane `y`, folded in slot order with `smin(., sphere, addSmooth)` for an add and `smax(., -sphere, subSmooth)` for a dig (`EvaluateSdf`, `examples/vrchat_terrain_sculpt_golden.rs`); the surface under a point is a downward sphere trace (`SurfaceHeight`) | standing on the SDF: a Unity box collider (`TerrainSupport`, scene root) placed every frame on the highest surface under the foot (`SupportHeight`, five samples, level, step / wall / buried classification in `ContactKind`); desktop sculpting casts against the terrain as it was at the press (`EvaluateSdfSkipping`) |
| DeformableWall | box (half `wallWidth/Height/Thickness`) on the plane, folded with `smax(., -sphere(dentRadius · strength), dentSmooth)` per live dent (`EvaluateWallSdf`, `examples/vrchat_deformable_wall_golden.rs`); strength recovers as `exp(−decaySpeed · t)` (`Decay`) | impacts measured against the undented face and refused inside a live dent's hollow (`TryImpact`); the body capsule pressed in by the shader only (`_PlayerCapA/B`) |
| Basic / Cosmic / Fractal / Mix | `Evaluate` override of the package's `AliceSDF_Collider` (`examples/vrchat_{basic,cosmic,fractal,mix}_golden.rs`); Cosmic / Mix take `animTime` = the shader's `_Time.y` | the base collider's body push (wall contacts only); the scene's floor or viewing platform |

Invariants 1–6 hold for each of them with the names above (`MaxSculpts` /
`_SculptData[128]`, `MaxImpacts` / `_ImpactPoints[16]`); the host parity
projects in `HostTests~` are the check for every collider.

## 4. What to ask the user to do (they have the client; you do not)

Copy, fill in, send:

> Please open `Assets/AliceSDF_SampleScenes/SDF_Mochi.unity`, turn on
> **Log Events** on `SDF_Mochi`, run **VRChat SDK > Build & Test**, and in the
> client: walk into a mochi, left-click and hold one, right-click while
> holding (split), carry one onto another (merge). Then send me the newest
> `%USERPROFILE%\AppData\LocalLow\VRChat\VRChat\output_log_*.txt` (or the lines containing
> `[Mochi]`), and one screenshot. I am looking for `[Mochi] push`, `grab`,
> `split`, `merge`, and no `Shader error` / `UdonSharp` lines.

---

## 5. Known errors → cause → fix

| # | Symptom (grep for it) | Cause | Fix |
|---|-----------------------|-------|-----|
| T1 | `Illegal byte sequence` from `UnityEventFilter` / `Assembly.GetCodeBase` at Play | Non-ASCII characters in the project path (e.g. a user name in kanji) — the VRChat SDK's event filter cannot read its own assembly path | Move the project to an ASCII path (e.g. `E:\VRChatProjects\Mochi`) and re-add it in VCC |
| T2 | `The type or namespace name 'UdonSharp' could not be found` / `VRC` not found | Not a VRChat Worlds project (SDK missing), or the package was copied into `Assets/` instead of installed as a package | Create the project with VCC (Worlds) and install via `manifest.json` (§1 step 2); `Runtime/AliceSDF.Runtime.asmdef` references `UdonSharp.Runtime`, `VRC.Udon`, `VRC.SDKBase` |
| T3 | `[ALICE-SDF] Shader 'AliceSDF/Samples/Mochi' not found` | Sample not imported | §1 step 3 |
| T4 | `[ALICE-SDF] AliceSDF.Samples.SampleMochi_Collider not found` | The sample's script did not compile (see Console) or the scene was generated in the same domain the sample was imported in (headless: both methods in one invocation) | Fix the compile error; headless: run `ImportAllSamplesBatch` and `GenerateAllBatch` as two invocations |
| T5 | After updating the package the Mochi behaves like the old version | Package Manager never overwrites `Assets/Samples/ALICE-SDF for VRChat/<old version>/` | Rename the folder to the new version with `AssetDatabase.MoveAsset("Assets/Samples/ALICE-SDF for VRChat/<old>", "…/<new>")` (keeps every GUID, so existing scenes and materials stay bound; `Import All Samples` then reports it as already present and imports only the missing samples), copy the updated files over it, and regenerate. Deleting the folder and re-importing also works but assigns new GUIDs (`Samples~` carries no `.meta`): a scene generated before then points at a shader that no longer exists (pink material) |
| T6 | `EPERM` / rename of `…\packages.unity.com\.tmp-*` during resolve, then hundreds of `UnityEditor.TestTools` errors | A registry package download was locked (antivirus) mid-rename, leaving `com.unity.test-framework` missing | Back up `Packages/packages-lock.json`, delete the broken package's entry, focus Unity, Ctrl+R |
| T7 | `Build & Test` disabled / UdonSharp exception watcher error at startup | The VRChat client is not installed on this machine | Install the client, or ask the user (§4) |
| T8 | Dark line along every silhouette (mochi and ground edge) | The marcher's miss path writes far depth and the world floor behind shows through — you removed or broke the closest-approach fallback | Restore `bestD`/`bestT` handling (0.5.0, invariant 6) |
| T9 | Player bounces up every frame while standing still on the plane | Collider evaluated the ground term | Collide with `EvaluateMochiSdf` (invariant 5) |
| T10 | Desktop click never grabs (`[Mochi] click missed` every time) | The click is evaluated on the view ray; the ray misses every mochi or the hold is shorter than `Grab Dwell Time` (0.08 s) | Aim at the mochi's centre and hold; `[Mochi] click hit` then `grab` after the dwell |
| T11 | Mochis render but do not react to hands in VR | Tracking positions were the exact zero vector (no VR) or the script is not on the same object as the material | Confirm `SDF_Mochi` has both the MeshRenderer with the Mochi material and `SampleMochi_Collider`; in the Editor without VR use desktop click |
| T12 | `IndexOutOfRangeException` at `VRCSdkControlPanelWorldBuilder.CreateValidationsGUI` whenever a scene is opened | The VRChat SDK control panel re-runs its validations on scene change and a scene without a `VRC_SceneDescriptor` (every generated sample scene) trips it | Harmless; close the SDK panel while generating, or ignore |
| T13 | A generated interactive scene has `SampleX_Collider` but no `UdonBehaviour` (nothing reacts at Play) | Generated with a package before the program-asset step: UdonSharp only backs a proxy whose script has a `UdonSharpProgramAsset`, and does not create one inside an unsaved scene | Regenerate with the current package; or open the scene in the Editor (UdonSharp's "has not been fully setup, running setup" creates the backing) and save |

---

## 6. Scope of this file

It covers the package as published; it does not promise an avatar version of
Mochi (§3.2 is guidance for someone building one), a VPM listing, or Quest
support. Report anything you find that is not here to
`https://github.com/ext-sakamoro/ALICE-SDF/issues`.
