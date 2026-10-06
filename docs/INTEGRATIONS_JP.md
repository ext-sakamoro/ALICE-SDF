# プラットフォームとツールの統合

モバイル / Web / ゲームエンジン / DCC / XR / CAD の統合 [README](../README_JP.md#バインディング) は一覧表のみ、
各統合の手順はこのページ

[English](INTEGRATIONS.md)

## Mobile (iOS / Android)

ALICE-SDF は [UniFFI](https://mozilla.github.io/uniffi-rs/) ベースの mobile SDK を同梱しており、Rust コアを **Swift** (iOS) と **Kotlin** (Android) から直接呼び出せます。

### 対応ターゲット

| プラットフォーム | アーキテクチャ | 配布物 |
|-----------------|-------------|--------|
| **iOS** | `aarch64-apple-ios` (実機), `aarch64-apple-ios-sim`, `x86_64-apple-ios` | `AliceSDF.xcframework` (static lib + Swift bindings) |
| **Android** | `arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86` | `libuniffi_alice_sdf.so` + Kotlin bindings |

### 実機動作確認済 (2026-06-06)

| プラットフォーム | デバイス | 結果 |
|-----------------|---------|------|
| iOS | iPhone 17 Pro Simulator (iOS 26.0, Xcode 26.2) | ✅ アプリ起動、2D SDF スライス描画 ([screenshot](../mobile/samples/ios-swiftui/screenshots/AliceSDF-demo.png)) |
| Android | Pixel 6 emulator (Android 14 / API 34, arm64-v8a) | ✅ アプリ起動、2D SDF スライス描画 ([screenshot](../mobile/samples/android-compose/screenshots/AliceSDF-android-demo.png)) |

両プラットフォームで **完全に同じ数値** (`sphere d = 0.2806`、`smooth_union(k=0.3) = 0.2056`) を出力 — Rust コアの Apple Silicon / Android ARM 間移植正確性を実機で実証。

### Swift クイックスタート

```swift
import AliceSDF

let d = sdfSphere(
    point:  Vec3(x: 1, y: 0, z: 0),
    center: Vec3(x: 0, y: 0, z: 0),
    radius: 1.0
)
// d ≈ 0 (球面上の点)

let blended = opSmoothUnion(a: 0.5, b: 0.6, k: 0.1)
// blended < 0.5 (smooth union が min より下に引っ張る)
```

### Kotlin クイックスタート

```kotlin
import uniffi.alice_sdf.*

val d = sdfSphere(
    point  = Vec3(1f, 0f, 0f),
    center = Vec3(0f, 0f, 0f),
    radius = 1.0f
)
// d ≈ 0

val blended = opSmoothUnion(a = 0.5f, b = 0.6f, k = 0.1f)
```

### SDK ビルド

```bash
# iOS XCFramework (実機 + シミュレータ)
cd mobile/packaging/ios && ./build-xcframework.sh

# Android .so + Kotlin bindings (4 ABI)
export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk
cd mobile/packaging/android && ./build-aar.sh
```

サンプルアプリ・統合手順は [`mobile/`](../mobile/) を参照。

---

## Web (WebAssembly) / VFX (OpenVDB) / Bevy エンジン

### `wasm` feature — WebAssembly バインディング

ブラウザ側で SDF 評価 + スライス描画。`wasm-bindgen` ベース。

```bash
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
```

JavaScript 使用例:

```js
import init, { sdf_sphere, op_smooth_union, render_sphere_slice_rgba } from './alice_sdf.js';
await init();
const d = sdf_sphere(1, 0, 0, /*center*/ 0, 0, 0, /*radius*/ 1.0);  // ≈ 0
const rgba = render_sphere_slice_rgba(256, 256, 0, 0, 0, 1.0, 2.5);  // Uint8Array (canvas へ putImageData)
```

### `openvdb` feature — OpenVDB Float Grid I/O

SDF を voxel grid に bake、Houdini / Maya / Nuke / Blender 等の VFX/DCC ツール連携。

```rust
use alice_sdf::io::vdb::{bake_to_vdb, load_dense_grid_from_vdb};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
let bytes = bake_to_vdb(&node, (-2.0, 2.0), 64).unwrap();
std::fs::write("sphere.vdb", &bytes).unwrap();
```

[`vdb-rs`](https://crates.io/crates/vdb-rs) 0.6 (pure Rust) ベース。現状は `ALICEVDB1` コンパクト形式、`vdb-rs` の write API 整備に合わせて OpenVDB 正規バイナリへ移行予定。

### `alice-sdf-bevy` — Bevy 0.18 プラグイン

`SdfShape` Component を持つ Entity を spawn すれば、自動的に Mesh が生成・attach される ECS 統合。

```rust
use bevy::prelude::*;
use alice_sdf_bevy::{AliceSdfPlugin, SdfShape};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(AliceSdfPlugin)
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(SdfShape::Sphere { radius: 1.0 });
        })
        .run();
}
```

`bindings/bevy/alice-sdf-bevy/examples/sphere_demo.rs` にカメラ + ライト付きの 3 形状デモあり。

### 3D Gaussian Splatting (`.splat`)

SDF 表面を Inria 3DGS 互換 `.splat` ファイル (32 bytes/splat: position + scale + RGBA + 圧縮 quat) に変換。WebGL ベースのビューア (gsplat.tech / SuperSplat / antimatter15/splat) に drag&drop で即読込可能。

```rust
use alice_sdf::io::splat::{sdf_to_splats, save_splat, SplatConfig};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
let cfg = SplatConfig { bounds: (-2.0, 2.0), resolution: 64, base_color: [220, 220, 240, 255] };
let splats = sdf_to_splats(&node, &cfg);
save_splat("sphere.splat", &splats).unwrap();
```

### DCC 統合は reference integration

以下の Blender / Houdini / Maya / Nuke / Cinema 4D プラグインは **reference integration** (各 200〜700 行) です `.asdf` の読込経路と数個の primitive 生成をホスト内で示すもので、本番プラグインの出発点であって完成品ではありません Bevy (`bindings/bevy/`) / Three.js (`bindings/threejs/`) / OpenXR が比較的厚い統合で、CI で compile / type-check されています

### Blender アドオン (`bindings/blender/`)

Blender 4.0+ アドオン。`.asdf` を直接 import + N-panel に "ALICE-SDF" タブを追加して sphere / box / torus を生成。`alice_sdf` Python モジュール (`cargo build --release --features python`) が前提。

インストール: `alice_sdf_blender/` を zip 化し、`Edit > Preferences > Add-ons > Install...` から有効化。

### Houdini Python プラグイン (`bindings/houdini/`)

SideFX Houdini 20+ 用 Python モジュール + Python SOP body (`.asdf` ローダー / プリミティブ生成)。`install.sh` が `$HSITE` / `$HOUDINI_USER_PREF_DIR` を自動検出してコピー。

```python
import alice_sdf_hou
sdf = alice_sdf_hou.sphere(1.0)
alice_sdf_hou.sdf_to_hou_geo(sdf, hou.pwd().geometry(), bounds=(-2.0, 2.0), resolution=64)
```

### MagicaVoxel `.vox` IO

SDF を voxelize して MagicaVoxel `.vox` (v150 RIFF) で書き出し。indie / voxel art パイプライン向け。

```rust
use alice_sdf::io::vox::{sdf_to_vox, save_vox, VoxConfig};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
let cfg = VoxConfig { size: 64, bounds: (-1.5, 1.5), color_index: 79 };
save_vox("sphere.vox", &sdf_to_vox(&node, &cfg)).unwrap();
```

### `@alice-sdf/threejs` — Three.js / React Three Fiber ラッパー

`wasm` feature の上の TypeScript npm パッケージ。型付き `AliceSDF` クラス + Three.js `DataTexture` ヘルパ + R3F 用 `<AliceSDFSlicePlane>` + WebXR raymarching ヘルパを提供。

```ts
import { AliceSDF } from "@alice-sdf/threejs";
const sdf = await AliceSDF.load("/alice_sdf.js");
const tex = await sdf.createSliceTexture(512, 512, [0, 0, 0], 1.0, 2.5);
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: tex })));
```

### Maya Python プラグイン (`bindings/maya/`)

Autodesk Maya 2024+ 用 Python モジュール — メイン メニューに「ALICE-SDF」を登録し、`MFnMesh` API でポリゴンメッシュを直接構築。

```python
import alice_sdf_maya
alice_sdf_maya.register_menu()
alice_sdf_maya.add_sphere(radius=1.5, resolution=64)
```

### Nuke Python プラグイン (`bindings/nuke/`)

Foundry Nuke 15 / 16+ 用 Python モジュール — `.asdf` をボリュームバイナリと 2D RGBA スライスへ書き出し、VFX コンポジット連携。

```python
import alice_sdf_nuke
alice_sdf_nuke.export_asdf_as_volume("/path/to/model.asdf", out_path="/tmp/model.alicevdb")
```

### Cinema 4D Python プラグイン (`bindings/cinema4d/`)

Maxon Cinema 4D 2024 / 2025 / 2026+ 用 Python モジュール — SDF プリミティブから `PolygonObject` を生成、`.asdf` を直接 C4D シーンに読み込み。

```python
import alice_sdf_c4d
alice_sdf_c4d.add_sphere(radius=100.0, resolution=64)   # C4D 単位は cm
alice_sdf_c4d.import_asdf("/path/to/model.asdf", bounds=(-300.0, 300.0), resolution=128)
```

### CAD 交換 — STEP / IGES (FEM 風メッシュエクスポート)

SDF tree を Marching Cubes で tessellate して以下のいずれかを書き出し可能:

- **STEP AP214** (ISO 10303-21 ASCII、`AUTOMOTIVE_DESIGN`) — **faceted BREP**。三角形 1 枚が `PLANE` 上の `ADVANCED_FACE` で、`ORIENTED_EDGE`/`EDGE_CURVE` の `EDGE_LOOP` が境界、全面を `CLOSED_SHELL` → `MANIFOLD_SOLID_BREP` → `ADVANCED_BREP_SHAPE_REPRESENTATION` にまとめ、単位は mm。**注:** 面は平面なので、球は `SPHERICAL_SURFACE` ではなく三角形分割として届く。独立な読み戻し oracle (`tests/test_step_export_oracle.rs`) が (1) `#id` 参照の未解決なし (2) AP214 必須 root の存在 (3) shell が閉じている (各エッジが 2 回使われる) (4) 復元した体積が解析値と 5% 以内 を検証する。2026-09-27 以前は点と loop だけで shell / solid / 表現 / 単位が無く、未定義の `#0` を参照していたため CAD では開けなかった
- **箱は厳密** — 原点中心の軸平行箱 (`SdfNode::Box3d`) は tessellate せず 4 角形 6 枚の `ADVANCED_FACE` として書くので、`resolution` に依らず寸法が厳密 (oracle が体積を `w·h·d` と 1e-6 以内で照合)。球 / 円柱は従来通り tessellate する。
- **IGES** (Entity 134 Node + Entity 136 Finite Element) — FEM mesh entity。FEM ソルバや該当 entity 対応 viewer 向け。Entity 144 (Trimmed Surface) を期待する標準 CAD では未対応

両方とも unit test で round-trip 検証済みだが、実 CAD ツールでの相互運用は個別検証が必要。

```rust
use alice_sdf::io::step::{export_step, StepConfig};
use alice_sdf::io::iges::{export_iges, IgesConfig};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
export_step("sphere.step", &node, &StepConfig::default()).unwrap();
export_iges("sphere.igs",  &node, &IgesConfig::default()).unwrap();
```

### `alice-sdf-openxr` — Native VR / AR ヘルパー (`bindings/openxr/`)

[`openxr`](https://crates.io/crates/openxr) Rust バインディングの上に乗る薄いヘルパー。Meta Quest standalone (Android APK)、PC VR (SteamVR / Oculus PC)、Microsoft Mixed Reality、Apple Vision Pro (OpenXR backend) で動く。

```rust
use alice_sdf_openxr::{XrPose, raymarch_sphere};
use glam::Vec3;

// XR frame コールバック内で
let head_pose: XrPose = openxr_pose.into();
let hit_dist = raymarch_sphere(head_pose, Vec3::new(0.0, 1.5, -1.0), 0.3, 5.0);
if hit_dist > 0.0 {
    // コントローラ / ヘッドが球を見ている
}
```

### `AliceSDFVisionOS` — Apple Vision Pro SwiftPM パッケージ (`mobile/swift-package-visionos/`)

iOS / iPadOS / macOS と同じ `AliceSDF.xcframework` を再利用しつつ、visionOS / RealityKit 向けの `ModelEntity` ファクトリヘルパーを追加した SwiftPM パッケージ。

```swift
import SwiftUI
import RealityKit
import AliceSDFVisionOS

struct ImmersiveView: View {
    var body: some View {
        RealityView { content in
            let sphere = AliceSDFRealityKit.makeSphereEntity(radius: 0.1)
            sphere.position = SIMD3(0, 1.5, -1.0)
            content.add(sphere)
        }
    }
}
```

### REST API サーバー (`server/`)

`axum` + `tokio` ベースの HTTP サーバー。ALICE-SDF の primitive 評価と operation を JSON で公開。クラウド配信型 SDF UI (例: `alicelaw.net/sdf-metaverse`) のバックエンドを想定。

```bash
cd server && cargo run --release
# → ALICE-SDF server listening on http://0.0.0.0:8787
```

```http
POST /eval
Content-Type: application/json

{ "shape": "sphere", "point": [1, 0, 0], "params": { "radius": 1.0, "center": [0, 0, 0] } }
```

レスポンス: `{ "distance": 0.0 }`
