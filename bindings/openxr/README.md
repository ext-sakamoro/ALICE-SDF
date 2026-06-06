# alice-sdf-openxr

Native VR / AR (OpenXR) アプリで ALICE-SDF を呼び出すためのヘルパー集。
Meta Quest standalone (Android APK)、PC VR (Oculus / Steam VR)、Apple Vision Pro
(visionOS OpenXR backend) で動作。

## 役割

このクレートは **OpenXR Session 管理 / レンダリング自体は扱わない**。
ユーザーが既存の OpenXR Rust エコシステム
([`openxr`](https://crates.io/crates/openxr) + [`wgpu`](https://crates.io/crates/wgpu)
等) を使って構築するアプリの中で、

- ヘッド / コントローラ / ハンド pose の **SDF シーンとの衝突判定**
- ハンドメッシュ全頂点に対する **バッチ SDF 距離評価**

を行うユーティリティ関数を提供する。

## サポート対象 OpenXR ランタイム

- Meta Quest (Quest 2 / 3 / Pro standalone Android、または Quest Link/AirLink PC VR)
- Steam VR (Valve Index / HTC Vive / HP Reverb / Pico 等)
- Microsoft Mixed Reality (Hololens 2 等)
- Apple Vision Pro (visionOS の OpenXR backend、Apple 公式 2026 リリース予定)
- Pico 4 / 4 Ultra (Pico OpenXR Runtime)

## API

```rust
use alice_sdf_openxr::{XrPose, raymarch_sphere};
use glam::Vec3;

// OpenXR frame callback 内で session.locate_view() 等から取得した pose
let head_pose: XrPose = openxr_pose.into();

// 球 SDF への raymarching
let hit_dist = raymarch_sphere(
    head_pose,
    Vec3::new(0.0, 1.5, -1.0),  // 球中心
    0.3,                          // 半径
    5.0,                          // 最大距離
);
if hit_dist > 0.0 {
    // ヘッドが球を見ている
}
```

詳細は `src/lib.rs` の docstring と単体テストを参照。

## ビルド

このクレートは `alice-sdf` workspace 外の独立クレート。

```bash
cd bindings/openxr
cargo build --release
cargo test
```

## 依存関係

- `openxr = "0.19"` (公式 Rust OpenXR バインディング、loader 経由でランタイム呼出)
- `alice-sdf` (path 依存、default-features = false)
- `glam = "0.29"` (ALICE-SDF と同バージョン)

## 制限事項

- 本クレートは API helper のみ、`openxr::Session` の作成は呼出側の責務
- OpenXR ランタイムのインストール状況は各プラットフォーム依存
- Apple Vision Pro 向けは `mobile/swift-package-visionos/` のほうがネイティブ感が高い
