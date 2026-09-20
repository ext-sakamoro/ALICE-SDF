# HostTests~ — checks that run without Unity

Unity never sees this folder (the trailing `~` hides it from the importer);
it exists so the UdonSharp code in this package is compiled and compared
against the Rust crate on every push, not only when someone opens a scene.

| Project | What it checks |
|---------|----------------|
| `MochiParity/` | `SampleMochi_Collider.cs` compiles as C# 7.3 with warnings as errors against `Shared/UnityEngineStub.cs`; `EvaluateSdf` matches `alice_sdf::eval` of the same scene (`examples/vrchat_mochi_golden.rs`) on a 1521-point grid to 1e-5; a grab → split → release → settle → merge scenario conserves volume |
| `TerrainSculptParity/` | `SampleTerrainSculpt_Collider.cs` likewise; the six sculpts of `examples/vrchat_terrain_sculpt_golden.rs` are recorded and `EvaluateSdf` matches on a 4335-point grid to 1e-5; the scenario covers the sculpt cooldown / stroke / 48-slot wrap, the surface search over a hole and a stacked hill, the step / wall / buried contact classification (no push on flat ground, a lift onto a hill built under the feet, a sideways push off a 0.9 m column that stops) and the desktop cursor ray |
| `DeformableWallParity/` | `SampleDeformableWall_Collider.cs` likewise; the four dents of `examples/vrchat_deformable_wall_golden.rs` are loaded and `EvaluateSdf` matches on a 5100-point grid to 1e-5; the scenario covers impacts against the undented face (cooldown, no drilling inside a live dent, refresh, weakest-slot reuse), recovery to zero, the dented law, the sideways body push with a dead band and the desktop ray |
| `StaticParity/` | `AliceSDF_Collider` (the base) and the four static sample colliders compile; the sample named on the command line (`vrchat_<name>_golden`) is compared with its golden (Basic 2535 / Cosmic 2907 / Fractal 3375 / Mix 1989 points, the animated ones at `animTime = 0`) to 1e-5; the scenario checks the base collider: the override is what it evaluates, a wall-like contact pushes sideways and stops (dead band), a floor-like contact is ignored, Cosmic / Mix move with `animTime` |
| `KitCompile/` | The Kit product scripts: `AliceSDF_Collider.cs` and the seven sample colliders renamed the way `Editor/KitProductBuilder.cs` renames them (`namespace AliceSDFKit`, `Sample*_Collider` → `AliceMochi` / `AliceWall` / `AliceTerrain` / `AliceDecor*`) are written to `gen/` and `Compiled/` compiles the eight together against the stub, then checks by reflection that the eight types exist under `AliceSDFKit`, the decor four derive from the base and override `Evaluate`, the interactive three expose `EvaluateSdf`, and nothing is left in `AliceSDF.*`. A class renamed in a sample but not in the builder's table, or a new cross-sample reference, fails here; the rename table is a copy of the builder's and must move with it |
| `Shared/` | `UnityEngineStub.cs`, the minimum `UnityEngine` surface the colliders touch on their non-`UDONSHARP` path (`Vector3` / `Vector4` / `Quaternion` / `Mathf` / `Time` / `Material` / `Transform` / `GameObject` / `MonoBehaviour` / the Inspector attributes). It is not Unity: what it proves is the C# and the law, not the Udon runtime |

Run from the repository root:

```bash
scripts/vrchat-host-parity.sh        # needs cargo + dotnet (brew install dotnet)
```

Opening the sample in Unity / VRChat is still the only test of the Udon
runtime and of the picture. Everything between — do the shaders compile, does
UdonSharp accept the scripts, do the scenes and the Kit build — is the Unity
CI below.

## Unity CI (shaders / UdonSharp / scenes / Kit)

`Editor/AliceSDF_CiChecks.cs` is a batch entry (`AliceSDF.Editor.AliceSDF_CiChecks.RunBatch`,
one Unity invocation per stage, the stage from `ALICE_CI_STAGE`) that runs
inside a VRChat Worlds project linking this package:

| Stage | What it does / checks |
|-------|----------------------|
| `setup` | Writes the scripting defines the SDK and UdonSharp add on editor ticks (`UDON` / `VRC_SDK_VRCSDK3` / `UDONSHARP`, Standalone + Android); a fresh headless project never runs those ticks and the samples' `#if UDONSHARP` would pick the `MonoBehaviour` branch |
| `compile` | No C# errors; `UdonSharpCompilerV1.CompileSync` with 0 errors (its assembly cache is primed on the main thread first, else the worker thread throws); every shader under `Assets` and the package re-imported and free of errors for the active target |
| `samples` | `SampleSceneGenerator.ImportAllSamples` (the seven `Samples~` into `Assets/Samples`) |
| `scenes` | `SampleSceneGenerator.GenerateAll`, then each `SDF_<name>.unity` has a `VRCSceneDescriptor` with a spawn, a `PipelineManager`, the volume with a material whose shader compiles and an `UdonBehaviour` whose program is compiled |
| `kit` | `KitProductBuilder.BuildCore`; the build spans domain reloads, so the caller loops this stage until `Library/alice_ci_kit.txt` says `done <path>` |
| `verify` | `compile` + `scenes` + the Kit: 8 scripts and compiled programs, 10 prefabs each with a `Volume` (shadows off, backing `UdonBehaviour`, `TerrainSupport` on Terrain), 7 `AliceSDFKit/*` shaders, 2 textures, 2 includes, README EN / JP |
| `android` | Unity started with `-buildTarget Android`: `compile` again, so every shader is compiled for GLES3 (the Quest `ALICE_MOBILE_BUDGET` branch) |

Two drivers run the same stages:

- **Local** (installed Unity): `pwsh scripts/unity-preflight.ps1` — defaults to
  Unity 2022.3.22f1 and a Creator Companion project named `MochiProductTest`
  that links the package as `"com.alice.sdf": "file:<path to vrchat-package>"`;
  `-Project` / `-Unity` / `-Stages` override. It first checks that the
  `Assets/Samples/…` copies of the samples in the developer project match
  `Samples~` (Unity compiles the copy, an unsynced edit ships a stale
  sample), then clears the generated folders and runs the stages, logs under
  `target/unity-preflight/`.
- **GitHub Actions** (`.github/workflows/unity-vrchat.yml`, on changes under
  `vrchat-package/`): `scripts/unity-ci-project.sh` builds a throw-away
  project (`vrchat-package/CI~/` template, `com.vrchat.worlds` pinned in the
  script and resolved with [vrc-get](https://github.com/vrc-get/vrc-get)),
  `scripts/unity-ci-stages.sh` runs the stages in the GameCI image
  `unityci/editor:ubuntu-2022.3.22f1-android-3`, and the exported
  `AliceSDFKit_<version>.unitypackage` is uploaded as an artifact. The job
  needs the repository secret **`UNITY_LICENSE`**: the contents of a
  `Unity_lic.ulf` (a Personal licence activated for 2022.3.22f1 — Unity Hub
  writes it to `%PROGRAMDATA%\Unity\Unity_lic.ulf` on Windows,
  `/Library/Application Support/Unity/Unity_lic.ulf` on macOS). Without the
  secret the job prints a notice and passes without checking anything.
