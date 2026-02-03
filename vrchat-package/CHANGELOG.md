# Changelog

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
