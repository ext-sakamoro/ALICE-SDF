# AliceSDF — Swift Package

ALICE-SDF Mobile XCFramework を **Swift Package Manager** 経由で配信するためのパッケージ。

## 使い方 (アプリ側)

```swift
// Package.swift
.package(path: "../ALICE-SDF/mobile/swift-package")

// targets:
.executableTarget(
    name: "MyApp",
    dependencies: [
        .product(name: "AliceSDF", package: "AliceSDF")
    ]
)
```

```swift
// MyApp source
import AliceSDF

let d = sdfSphere(
    point: Vec3(x: 1, y: 0, z: 0),
    center: Vec3(x: 0, y: 0, z: 0),
    radius: 1.0
)
```

## 構成

```
swift-package/
├── Package.swift                 # binaryTarget + 通常 target の 2 層
└── Sources/AliceSDF/
    └── AliceSDF.swift            # UniFFI 生成 Swift bindings (xcframework からコピー)
```

`Package.swift` は2つの target を持つ:

1. **`AliceSDFFramework`** (binaryTarget) — `../uniffi-wrapper/target/xcframework/AliceSDF.xcframework` を参照
2. **`AliceSDF`** (target) — Swift bindings を含み、上の framework に依存

## 配信先候補 (Lv4 範囲)

- **SwiftPM (リポジトリ単体)**: GitHub Tag 付け → アプリ側で `.package(url: ..., from: "0.1.0")`
- **CocoaPods**: `AliceSDF.podspec` 追加 + `pod trunk push` (要 Apple Developer Account)
- **Maven Central相当の Apple 公式版** は無し (将来 Swift Package Index 登録は無料)
