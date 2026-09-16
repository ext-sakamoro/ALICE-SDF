# HostTests~ — checks that run without Unity

Unity never sees this folder (the trailing `~` hides it from the importer);
it exists so the UdonSharp code in this package is compiled and compared
against the Rust crate on every push, not only when someone opens a scene.

| Project | What it checks |
|---------|----------------|
| `MochiParity/` | `SampleMochi_Collider.cs` compiles as C# 7.3 with warnings as errors against `UnityEngineStub.cs`; `EvaluateSdf` matches `alice_sdf::eval` of the same scene (`examples/vrchat_mochi_golden.rs`) on a 1521-point grid to 1e-5; a grab → split → release → settle → merge scenario conserves volume |

Run from the repository root:

```bash
scripts/vrchat-host-parity.sh        # needs cargo + dotnet (brew install dotnet)
```

`UnityEngineStub.cs` is the minimum surface the collider touches on its
non-`UDONSHARP` path (`Vector3` / `Vector4` / `Mathf` / `Material` /
`MonoBehaviour` / the Inspector attributes). It is not Unity: what it proves
is the C# and the law, not the Udon runtime. Opening the sample in Unity /
VRChat is still the only test of UdonSharp acceptance and of the shader.
