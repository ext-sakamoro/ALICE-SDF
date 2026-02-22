# Changelog

All notable changes to ALICE-SDF are documented in this file.

## [v1.1.0] - 2026-02-22

### Added
- **Auxiliary data buffer** (`aux_data: Vec<f32>`) on `CompiledSdf` for variable-length instruction data
- `aux_offset` / `aux_len` fields on `Instruction` struct
- **ProjectiveTransform** compiled eval — perspective projection with 4x4 inverse matrix from aux_data
- **LatticeDeform** compiled eval — FFD grid deformation with control points from aux_data
- **SdfSkinning** compiled eval — bone-weight skeletal deformation with BoneTransform array from aux_data
- **IcosahedralSymmetry** compiled eval — proper 120-fold icosahedral fold (was abs() approximation)
- **IFS** compiled eval — Iterated Function System with transform matrices from aux_data
- **HeightmapDisplacement** compiled eval — bilinear heightmap sampling from aux_data
- **SurfaceRoughness** compiled eval — FBM noise with child distance
- 5 roundtrip tests (compiled vs tree-walker consistency)
- 33 doc comments on `transpiler_common.rs` public API
- All dependency stubs in release CI workflow

### Fixed
- 220 pedantic clippy warnings (raw string hashes, implicit clone, useless format, dead code, clamp pattern, manual Debug)
- `surface_roughness` private module path and argument count in eval.rs
- Missing `inst_idx` variable in new CoordFrame initializers
- Release workflow missing alice-cache/alice-codec/libasp/alice-font stubs

### Quality
- 1003 tests passing, 0 failed
- 0 clippy pedantic warnings
- 0 doc warnings
- 0 TODO/FIXME/unimplemented

## [v1.0.0] - 2026-02-08

### Added
- Initial stable release
- 72 SDF primitives (Platonic solids, TPMS surfaces, 2D primitives)
- 24 CSG operations (smooth, chamfer, stairs, exp-smooth, columns, pipe, groove, tongue)
- 7 evaluation modes (interpreted, compiled VM, SIMD 8-wide, BVH, SoA, JIT, GPU)
- 3 shader transpilers (GLSL, WGSL, HLSL)
- PBR material system (metallic-roughness)
- Keyframe animation system
- 15 I/O formats (ASDF, OBJ, GLB, FBX, USD, Alembic, STL, PLY, 3MF, ABM, Nanite, Unity, UE5)
- Neural SDF (MLP approximation)
- SDF-to-SDF collision detection
- Analytic gradient computation
- Dual Contouring mesh generation
- CSG tree optimization
- Interval arithmetic evaluation
- Text-to-3D pipeline (FastAPI server)
- FFI bindings (C/C++/C#)
- Python bindings (PyO3 + NumPy)
- Godot GDExtension
- Unity and UE5 plugins
- VRChat package

## [v0.1.0] - 2026-02-08

### Added
- Initial pre-release
- Core SDF primitives and operations
- Basic compiled evaluator
- CI/CD pipeline
