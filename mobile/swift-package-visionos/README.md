# AliceSDFVisionOS — visionOS / Apple Vision Pro Swift Package

ALICE-SDF Mobile (UniFFI) の上に乗る、`RealityKit` 連携用 Swift Package。
Apple Vision Pro / visionOS 1.0+ で `ModelEntity` として SDF プリミティブを
直接 RealityKit シーンに追加できる。

## 対応プラットフォーム

| Platform | 最小バージョン |
|----------|---------------|
| visionOS | 1.0 |
| iOS | 17.0 |
| macOS | 14.0 (RealityKit 部分は visionOS / iOS 専用) |

## 使い方

`Package.swift`:
```swift
.package(path: "../ALICE-SDF/mobile/swift-package-visionos")
.executableTarget(
    name: "MyVisionApp",
    dependencies: [.product(name: "AliceSDFVisionOS", package: "AliceSDFVisionOS")]
)
```

`MyApp.swift`:
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

            let box = AliceSDFRealityKit.makeBoxEntity(size: SIMD3(0.1, 0.05, 0.05))
            box.position = SIMD3(0.3, 1.5, -1.0)
            content.add(box)
        }
    }
}
```

## API

| 関数 | 用途 |
|------|------|
| `makeSphereEntity(radius:, material:)` | 球 ModelEntity 生成 |
| `makeBoxEntity(size:, material:)` | 直方体 ModelEntity 生成 |
| `sphereDistance(point:, center:, radius:)` | SDF 距離評価 |
| `smoothUnion(_:_:k:)` | smooth union (hand-tracking 判定向け) |
| `sphereBatch(points:, center:, radius:)` | ハンドメッシュ全頂点に対するバッチ評価 |

## ビルド

```bash
cd mobile/swift-package-visionos
# Package が認識されるか確認 (visionOS SDK が必要)
swift package describe
```

実機ビルドは Xcode の visionOS シミュレータ or 実機 (Vision Pro / Vision Pro 2) を選択して
Run。

## 既存 `mobile/swift-package` との関係

| Package | プラットフォーム | 主な目的 |
|---------|------------------|----------|
| `mobile/swift-package` | iOS / iPadOS / macOS | UniFFI 経由の汎用 ALICE-SDF アクセス |
| `mobile/swift-package-visionos` (本パッケージ) | visionOS + 上記 | RealityKit Entity / hand-tracking 統合 |

両方使う場合は visionOS 向けには本パッケージのみ依存すれば足りる
(内部で `AliceSDFFramework` (XCFramework) を再利用)。

## 制限事項

- v1.6 時点では `MeshResource.generateSphere/Box` 等の RealityKit 標準 mesh を使用
  (本格的な dual_contouring / marching_cubes での SDF→Mesh 変換は ALICE-SDF Mobile UniFFI
  経由で別途実装予定)
- hand-tracking との完全連携 (`ARKitSession.HandTrackingProvider`) は別サンプル
- Material 設定は `SimpleMaterial` の単純な色のみ、PBR は将来対応
