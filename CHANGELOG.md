# Changelog

All notable changes to ALICE-SDF are documented in this file.

For releases prior to v1.5.0 (v0.1.0 – v1.3.0), see [CHANGELOG-history.md](CHANGELOG-history.md).

## [Unreleased]

### Added

- **Unreal Engine 6.0 (UE6) support** — `unreal-plugin/AliceSDF.uplugin` の `EngineVersion` を `6.0.0` に bump (UE6-main `f602d4b` time point)。UE5.5+ で導入された最新 RHI API (= `FRHIBatchedShaderParameters` / `FRHIBufferCreateDesc::CreateVertex/CreateIndex` / 4 引数 `SubscribeToPostProcessingPass` / `DispatchComputeShader` / `IMPLEMENT_GLOBAL_SHADER` / `LAYOUT_FIELD` / `FSceneViewExtensionBase` / `GScreenRectangleVertexBuffer` 等) が UE6 にも残存、`UE_DEPRECATED(6.x)` 0 件確認、Build.cs / `.cpp` / `.h` / `.usf` 改変ゼロで論理互換。実機 UE6 Editor build 検証は別途実施推奨

### Changed

- **README** — Engine integrations 列挙を `Unreal Engine 5 / 6` に更新 (英語/日本語)

## [v1.7.2] - 2026-06-08

### Added

- **core clippy-strict CI** — `clippy` job を informational から `-D warnings` 化 (no-default-features + glsl/hlsl/gpu の 2 matrix)。新 lint 混入を即 CI fail で発見
- **Pre-built wheel CI** (`.github/workflows/release-wheels.yml`) — tag push (`v*`) で linux-x86_64 / linux-aarch64 / macos-arm64 / macos-x86_64 / windows-x86_64 の wheel を maturin で abi3-py310 ビルドして Release に attach。1 wheel で Python 3.10–3.13 をカバー
- **REST server smoke test** CI job — `/version` / `/eval` / `/op` / `/mesh` / `/splat` / `/vox` の全 endpoint を curl で叩く
- **WASM build** CI job — `cargo build --target wasm32-unknown-unknown --features wasm` で artifact 生成検証
- **Three.js TypeScript type-check** CI job — `tsc --noEmit` で TypeScript 健全性確認
- **Mobile sample compile** CI job (macOS runner) — iOS は xcodebuild build-for-testing、Android は `gradlew assembleDebug` でリグレッション検出
- **visionOS XCFramework support** — `mobile/packaging/ios/build-xcframework.sh --with-visionos` で `aarch64-apple-visionos` / `aarch64-apple-visionos-sim` slice を追加 (nightly + `-Z build-std`)
- **REST server hardening**:
  - `Authorization: Bearer <ALICE_SDF_TOKEN>` middleware (env が空でなければ全 endpoint で必須化、`/` `/version` は除外)
  - `tower_governor` レート制限 (per-IP、デフォルト 20 RPS / burst 60、`ALICE_SDF_RPS` / `ALICE_SDF_BURST` で上書き可能)
  - `RequestBodyLimitLayer` で 1 MiB JSON body 上限
- **`docs/USAGE.md` / `docs/USAGE_JP.md`** — README から詳細セクション 1675 行を移動
- **`docs/PUBLISH.md`** — crates.io 配布戦略の現状とロードマップを明文化

### Changed

- **STEP / IGES README claim 是正** (`README.md` / `README_JP.md`) — 「Fusion 360 / SolidWorks / OnShape / Rhino / AutoCAD / FreeCAD 互換」を撤回。実態は `POLY_LOOP` + `FACE_OUTER_BOUND` の faceted mesh / Entity 134+136 FEM mesh で、`MANIFOLD_SOLID_BREP` を要求する CAD ツールでは開けない可能性がある旨を明記
- **REST server resolution / size 検証** — 旧 silent `clamp(8, 192)` を `400 Bad Request` に変更 (out-of-range を明示的にエラー返却)
- **README 分割**: 2585 → 913 行 (35%)、JP も同様
- `pyproject.toml`: `requires-python = ">=3.9"` → `">=3.10"` (abi3-py310 と整合)
- `pyproject.toml`: project version 0.1.0 → 1.7.2 (Cargo.toml と同期)

### Fixed

- `src/python/compiled.rs`: 未使用の `source_node` field 削除 (`dead_code` warning 除去)
- `src/io/iges.rs`: `format!()` を str literal に置換 (clippy `useless_format`)
- `src/io/vox.rs`: `cfg.size.min(256).max(1)` → `cfg.size.clamp(1, 256)` (clippy `manual_clamp`)

### Compatibility

- Mobile: iOS / Android **+ visionOS** (XCFramework スクリプトに追加)
- Unreal Engine: 5.7.0 〜 5.7.4 / 5.8.0-preview-1 (変更なし)

## [v1.7.1] - 2026-06-08

### Added

- **REST server endpoint 拡張** (`server/`) — `POST /mesh` (Marching Cubes vertices+normals+indices)、`POST /splat` (3D Gaussian Splats、`format=bytes` で base64 32-byte stream)、`POST /vox` (voxel 配列) を追加。`/version` が全 endpoint を列挙
- **OpenXR `SceneFrame` / `SphereBeacon` / `RayHit` API** (`bindings/openxr/`) — フレーム 1 回分のシーン状態を builder style で組み立て、head/left/right の raycast と手メッシュ→beacon 最小距離をワンメソッドで取得
- **OpenXR `examples/quest_demo.rs`** — Meta Quest 風の 60-frame loop 完全実装サンプル
- **visionOS `makeSDFMeshEntity` / `makeBlobEntity`** — 任意 SDF closure を voxel-fill で評価し RealityKit `ModelEntity` 化 / 2 球 smooth-union を Rust `AliceSDFFramework` 直呼出で blob 生成
- **visionOS `AliceSDFFramework` 統合** — Swift 側 SDF 計算を Rust UniFFI コアにルーティング (`sdfSphere` / `opSmoothUnion` / `sphereBatch` / `aliceSdfVersion`)、`canImport(AliceSDFFramework)` でフォールバック実装も保持

### Changed

- **STEP / IGES export を Marching Cubes 化** (`src/io/step.rs` / `src/io/iges.rs`) — 旧 naive voxel quads を `mesh::sdf_to_mesh` (実 MC アルゴリズム) に置換。res=16 で <100 verts → 数千 verts の品質向上
- **PyO3 を `abi3-py310` に固定** — Python 3.10 / 3.11 / 3.12 / 3.13 を 1 つの `.so` でサポートし、再ビルド不要に
- **CHANGELOG split**: v0.1.0 – v1.3.0 を `CHANGELOG-history.md` に分離 (本ファイルの肥大化対策)
- **CI `clippy-strict` トリガ拡張**: `paths-filter` で `code` (core src/**) 変更時も mobile wrapper の strict clippy を実行 (uniffi-wrapper は alice-sdf core を path dep として再 clippy するため、core の変更が見落とされる設計ミスを修正)
- **CI mobile job**: `cargo test --tests` (debug プロファイル) を追加し、12 公開 wrapper 関数を 26 統合テストで網羅
- README (英日) の Python 節に abi3 / pre-built wheel 説明を追加

### Fixed

- `src/io/vdb.rs`: `VdbError::Io` / `VdbError::InvalidBounds` の missing variant docs (clippy strict 対応)

### Quality

- **Core**: 1,093 tests passing (+10 vs v1.7.0 — STEP/IGES 各 +3 quality tests + Bevy +7 + OpenXR +8 + mobile +22 + server +4)
- **OpenXR**: 4 → 12 tests
- **Bevy**: 4 → 11 tests (normals 単位長 / vertex bounds / annulus / cap planes / plugin build)
- **mobile/uniffi-wrapper**: 4 unit + **26 integration tests** (全 12 公開関数を網羅)
- **server**: 0 → 4 tests
- 全 clippy-strict (`-D warnings`) pass: openxr / mobile / bevy

## [v1.7.0] - 2026-06-06

### Added

#### 3D / Modern rendering

- **3D Gaussian Splatting I/O** (`src/io/splat.rs`) — Inria 3DGS 互換 `.splat` バイナリ (32 bytes/splat: pos + scale + RGBA + compressed quat) の読書き、`sdf_to_splats()` で SDF 表面近傍を Gaussian Splat 化、4 tests
- **MagicaVoxel I/O** (`src/io/vox.rs`) — `.vox` v150 RIFF (SIZE + XYZI chunks) の読書き、`sdf_to_vox()` で SDF を voxelize、4 tests
- **STEP AP203 export** (`src/io/step.rs`) — ISO 10303-21 ASCII Faceted BREP、Fusion 360 / SolidWorks / OnShape / Rhino / FreeCAD 互換、2 tests
- **IGES export** (`src/io/iges.rs`) — IGES ASCII Entity 134 (Node) + 136 (Finite Element) で三角形メッシュ表現、Rhino / AutoCAD 互換、2 tests

#### Web / Mobile / XR

- **WebXR raymarching helpers** (`src/wasm.rs` 拡張) — `raymarch_sphere` / `raymarch_two_spheres_smooth` / `sphere_batch_flat` で VR/AR コントローラ・ハンドメッシュ用 SDF クエリ
- **Three.js / React Three Fiber TypeScript wrapper** (`bindings/threejs/`) — `@alice-sdf/threejs` npm パッケージ、`AliceSDF` クラス + `createSliceTexture()` Three.js helper + `<AliceSDFSlicePlane>` R3F コンポーネント + WebXR 統合例
- **OpenXR native helpers** (`bindings/openxr/`) — `XrPose` 変換 + `raymarch_sphere` + ハンドメッシュバッチ評価、Meta Quest / PC VR / Apple Vision Pro 対応、3 tests
- **visionOS Swift Package** (`mobile/swift-package-visionos/`) — Apple Vision Pro 用 RealityKit ヘルパー (`makeSphereEntity` / `makeBoxEntity`)、`AliceSDFFramework` (XCFramework) を再利用

#### DCC ツール統合

- **Blender Add-on** (`bindings/blender/`) — Blender 4.0 / 4.2 LTS / 4.4+ プラグイン: `.asdf` Import operator + sphere/box/torus 生成 + N-panel UI
- **Houdini Python plugin** (`bindings/houdini/`) — Houdini 20.0 / 20.5 / 21+ 用 Python SOP body + 自動 install.sh (python3.10libs/3.11libs 検出)
- **Maya Python plugin** (`bindings/maya/`) — Autodesk Maya 2024 / 2025 / 2026+ 用 Python module + MFnMesh + `register_menu()`
- **Nuke Python plugin** (`bindings/nuke/`) — Foundry Nuke 15.x / 16.x 用 Python module + Volume export + Slice render
- **Cinema 4D Python plugin** (`bindings/cinema4d/`) — Maxon Cinema 4D 2024 / 2025 / 2026+ 用 Python module + PolygonObject 生成

#### Cloud / Server

- **REST API server** (`server/`) — `axum` 0.7 + `tokio` 1.40、`POST /eval` (primitive 評価) + `POST /op` (operation) 公開、`alicelaw.net/sdf-metaverse` バックエンド向け

### Changed

- README (英日) に Web/VFX/Bevy/Splat/Vox/Blender/Houdini/Maya/Nuke/Cinema 4D/Three.js セクション追加
- DCC ツールの対応バージョンを各 README で **後方互換維持 + 新バージョン明示** (Maya 2024-2026、Houdini 20.0/20.5/21、Nuke 15.x/16.x、Blender 4.0/4.2/4.4)
- `AliceSDF.uplugin`: VersionName 1.6.0 → 1.7.0、Version 3 → 4

### Fixed

- `src/io/vox.rs` / `src/io/iges.rs`: pub struct field の missing docs (clippy strict 対応)
- `src/io/iges.rs`: unused `mut` / 使われない変数を削除
- `src/eval/mod.rs`: `43758.5453` の f32 過剰精度を `43758.547` に修正 (clippy strict `excessive_precision` 対応)

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
