# Changelog

All notable changes to ALICE-SDF are documented in this file.

## [Unreleased]

### Added

- **3D Gaussian Splatting I/O** (`src/io/splat.rs`) — Inria 3DGS 互換 `.splat` バイナリ (32 bytes/splat: pos + scale + RGBA + compressed quat) の読書き、`sdf_to_splats()` で SDF 表面近傍を Gaussian Splat 化、4 tests
- **Blender Add-on** (`bindings/blender/`) — Blender 4.0+ プラグイン: `.asdf` Import operator + sphere/box/torus 生成 operator + N-panel UI、PyO3 alice_sdf binding 経由
- **Houdini Python plugin** (`bindings/houdini/`) — Houdini 20+ 用 Python module + Python SOP body (loader + primitive generator) + 自動 install.sh、HSITE/HOUDINI_USER_PREF_DIR/HFS 自動検出

## [v1.6.0] - 2026-06-06

### Added

- **`wasm` feature** — WebAssembly bindings (browser): wasm-bindgen + js-sys。`sdf_sphere` / `sdf_box` / `sdf_torus_w` / `sdf_cylinder_w` / `sdf_plane_w` / 6 op + `render_sphere_slice_rgba` を JavaScript から呼び出し可能。`cargo build --target wasm32-unknown-unknown --features wasm` で動作
- **`openvdb` feature** — OpenVDB Float Grid I/O (Houdini / Maya / Nuke 等の VFX/DCC ツール連携): `bake_dense_grid()` / `bake_to_vdb()` / `load_dense_grid_from_vdb()`。vdb-rs 0.6 ベース。`io::vdb` モジュール、4 tests
- **Bevy plugin** (`bindings/bevy/alice-sdf-bevy/`) — Bevy 0.18 用 ECS 統合: `AliceSdfPlugin` + `SdfShape` Component (Sphere/Box/Torus/Cylinder)、Mesh 自動生成 system、`examples/sphere_demo.rs`、4 tests
- **CI matrix 拡張**: `wasm` / `openvdb` / `bevy` の 3 ジョブ追加、`physics` strict 化 (continue-on-error 削除 + 実 ALICE-Physics clone + 1088 tests カバー)

### Changed

- README (英日) に "Web (WebAssembly) / VFX (OpenVDB) / Bevy エンジン" セクション追加
- `AliceSDF.uplugin`: VersionName 1.5.0 → 1.6.0、Version 2 → 3

### Quality

- 全 CI matrix green: macOS ARM64 / Linux x86_64 / Windows x86_64 + Mobile + Physics strict + wasm + openvdb + bevy + clippy + clippy-strict + fmt
- 全 strict job pass (continue-on-error なし)

## [v1.5.0] - 2026-06-06

### Added

- **Mobile SDK (iOS / Android)** — `mobile/` 配下に [UniFFI](https://mozilla.github.io/uniffi-rs/) ベースの Swift / Kotlin 公開 SDK
  - `mobile/uniffi-wrapper/` — UDL 定義 + Rust ラッパークレート (`sdfSphere` / `sdfBox` / `sdfTorus` / `sdfCylinder` / `sdfPlane` / `sdfRoundedBox` + 6 op + `sphereBatch` + version)
  - `mobile/packaging/ios/build-xcframework.sh` — `AliceSDF.xcframework` (device 44MB + sim fat 88MB) 自動生成
  - `mobile/packaging/android/build-aar.sh` — 4 ABI `libuniffi_alice_sdf.so` (250-400KB) + Kotlin bindings 自動生成
  - `mobile/swift-package/Package.swift` — SwiftPM パッケージ (binaryTarget + Swift bindings 2層)
  - `mobile/samples/ios-swiftui/` — SwiftUI サンプルアプリ (xcodegen + Bridging Header 方式)
  - `mobile/samples/android-compose/` — Jetpack Compose サンプルアプリ (AGP 8.5.2 + Kotlin 2.0)
  - 実機検証: iPhone 17 Pro Simulator (iOS 26.0) + Pixel 6 Emulator (Android 14 / API 34) で iOS と Android 数値完全一致 (sphere d=0.2806, smooth union=0.2056)
- **Rendering metaverse features** — RenderConfig に分光レンダリング / 破壊 / VFX / マイクロ法線 / インテリアマッピング / dual SDF 還元
- **`WgslShader::transpile_material()`** — WithMaterial サブツリーからマテリアル評価関数を WGSL 生成
- **Terrain primitive** — 地形プリミティブ + フルレンダリングパイプライン
- **`examples/sword.lol`** — LOL DSL で記述した剣のサンプル
- **Unreal Engine 5.8 互換性確認** — `AliceSDF.uplugin` に `"EngineVersion": "5.7.0"` 明示、UE 5.7.0 〜 5.7.4 stable + 5.8.0-preview-1 で改修不要を実証 (Shader Parameter API: `FRHIBatchedShaderParameters` + `SetBatchedShaderParameters` + `FRHIBufferCreateDesc::CreateStructured` + `FRHIViewDesc::CreateBufferSRV/UAV`)
- **README リンク**: ALICE SDF Metaverse demo (https://alicelaw.net/sdf-metaverse) + alicelaw.net repo を Related Projects に追加 (英日)

### Changed

- **CI/CD 大規模強化**:
  - `concurrency: cancel-in-progress` で連続 push 時の前 run 自動 cancel
  - `dorny/paths-filter@v3` で README/docs only push の full CI skip
  - `.github/actions/alice-stubs` composite action で dep stub 生成を DRY 化 (60行 × 2 jobs)
  - `mobile` job 新規: iOS 3 target + Android 4 ABI cross-compile + Swift/Kotlin bindings 生成検証 + gpu (Metal) feature ビルド
  - `clippy-strict` job: mobile/uniffi-wrapper のみ `RUSTFLAGS="-Dwarnings"` 厳格
  - `nick-fields/retry@v3`: cargo build 3 リトライ (HTTP/2 framing layer 一過性失敗対策)
  - `CARGO_NET_RETRY=5` + `CARGO_HTTP_MULTIPLEXING=false` env
  - `fmt` job 拡張: core + mobile/uniffi-wrapper 両方
- **Author email**: `Moroya Sakamoto <sakamoro@alicelaw.net>` に統一 (Cargo.toml authors)

### Fixed

- **`optimize.rs`**: 4 パスを値渡し化、未最適化ノードの deep clone 除去 (perf)
- **BVH**: `split_off` 化、mipchain clone 除去、abm `read_to_end` 事前確保 (perf)
- **`check_min_tests`**: 算術エラー修正、cargo 失敗時の安全な skip
- **`ecosystem-tests` schedule**: 削除 (CI では兄弟クレート不在で動作不可)
- **`transpile_material`** ヘルパー重複定義の排除
- **cargo fmt** 差分修正 (CI rustfmt 互換)

### Quality

- **1,379 tests passing** (src/ 1,375 + mobile/uniffi-wrapper 4), 0 failed (+205 from v1.3.0)
- 0 clippy pedantic+nursery warnings (core)
- 0 clippy `-D warnings` (mobile wrapper、strict mode)
- 0 fmt diffs (core + mobile)
- CI matrix: macOS ARM64 + Linux x86_64 + Windows x86_64 + macOS Mobile cross-compile

### Compatibility

| Platform | Status |
|----------|--------|
| Linux x86_64 / aarch64 | 🟢 |
| macOS Apple Silicon / Intel | 🟢 |
| Windows x86_64 | 🟢 |
| **iOS aarch64 / sim** | 🟢 v1.5.0 新規 |
| **Android arm64-v8a / armv7 / x86_64 / x86** | 🟢 v1.5.0 新規 |
| Unreal Engine 5.7.0 〜 5.7.4 (stable) | 🟢 |
| Unreal Engine 5.8.0-preview-1 | 🟢 改修不要見込み |

## [v1.3.0] - 2026-03-14

### Added
- **LLM × 3D Pipeline docs**: README に SDF + LOL + View + Physics 統合パイプラインセクション追加（EN/JP両対応）
- **SDF圧縮パイプライン**: API公開、GPU Marching Cubes物理統合

### Changed
- **License**: MIT → MIT OR Apache-2.0 デュアルライセンスに変更
- **profile.release**: `panic = "abort"` を削除（ベンチコンパイルエラー修正）

### Fixed
- **autodiff.rs**: `suboptimal_flops` 警告を `mul_add` に修正
- **clippy**: pedantic+nursery 0 warnings 達成

### Quality
- 1,174 tests passing, 0 failed (+97 from v1.2.0)
- 0 clippy pedantic+nursery warnings
- 0 fmt diffs

## [v1.2.0] - 2026-03-05

### Added
- **shell**: `eval_shell_batch_parallel`, `eval_shell_gradient`, `eval_shell_compiled`, `eval_shell_compiled_batch_parallel`
- **diff**: `invert_patch`, `merge_patches`, `insert_at_path`, `delete_at_path`, `DiffOp::Insert`/`Delete`, `TreePatch::op_count`/`is_empty`
- **constraint**: `Product`/`Min`/`Max`/`Range` constraint kinds, `ratio()`/`product()`/`range()` convenience methods
- **sdf2d**: `Ring`, `RegularPolygon`, `Star`, `Ellipse`, `Onion` node types, `eval_2d_normal()`
- **interval**: `width()`, `midpoint()`, `contains()`, `overlaps()`, `intersect()`, `hull()`
- **autodiff**: `principal_curvatures()`, `gaussian_curvature()`
- **collision**: `ContactManifold` struct, `compute_manifold()`, `sdf_ccd()`, `sdf_closest_point()`
- **material**: `material_lerp()`, `MaterialLibrary::find_by_name()`, `StandardMaterials` (15 PBR presets)
- **neural**: `eval_batch()`, `eval_with_gradient()`, `hidden_layer_count()`, `input_dimension()`
- 74 new tests (1003 → 1077)

### Fixed
- `[profile.bench]` panic strategy conflict — added explicit `panic = "unwind"` to fix CI bench compilation on Linux/Windows

### Quality
- 1077 tests passing, 0 failed
- 0 clippy pedantic+nursery warnings
- 0 fmt diffs

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
