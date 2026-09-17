# HostTests~ — checks that run without Unity

Unity never sees this folder (the trailing `~` hides it from the importer);
it exists so the UdonSharp code in this package is compiled and compared
against the Rust crate on every push, not only when someone opens a scene.

| Project | What it checks |
|---------|----------------|
| `MochiParity/` | `SampleMochi_Collider.cs` compiles as C# 7.3 with warnings as errors against `Shared/UnityEngineStub.cs`; `EvaluateSdf` matches `alice_sdf::eval` of the same scene (`examples/vrchat_mochi_golden.rs`) on a 1521-point grid to 1e-5; a grab → split → release → settle → merge scenario conserves volume |
| `TerrainSculptParity/` | `SampleTerrainSculpt_Collider.cs` likewise; the six sculpts of `examples/vrchat_terrain_sculpt_golden.rs` are recorded and `EvaluateSdf` matches on a 4335-point grid to 1e-5; the scenario covers the sculpt cooldown / stroke / 48-slot wrap, the surface search over a hole and a stacked hill, the step / wall / buried contact classification (no push on flat ground, a lift onto a hill built under the feet, a sideways push off a 0.9 m column that stops) and the desktop cursor ray |
| `Shared/` | `UnityEngineStub.cs`, the minimum `UnityEngine` surface the colliders touch on their non-`UDONSHARP` path (`Vector3` / `Vector4` / `Quaternion` / `Mathf` / `Time` / `Material` / `Transform` / `GameObject` / `MonoBehaviour` / the Inspector attributes). It is not Unity: what it proves is the C# and the law, not the Udon runtime |

Run from the repository root:

```bash
scripts/vrchat-host-parity.sh        # needs cargo + dotnet (brew install dotnet)
```

Opening the sample in Unity / VRChat is still the only test of UdonSharp
acceptance and of the shader.
