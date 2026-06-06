# ALICE-SDF Demo (iOS SwiftUI)

ALICE-SDF Mobile XCFramework を SwiftUI アプリから呼び出す最小サンプル。

## 構成

- `project.yml` — xcodegen spec
- `AliceSDFDemo/` — Swift ソース
  - `AliceSDFDemoApp.swift` — `@main` エントリ
  - `ContentView.swift` — UI (スライダ + SDF 距離表示 + 2D Canvas スライス)
  - `alice_sdf.swift` — UniFFI 生成 Swift bindings (xcframework からコピー)
- 参照: `../../uniffi-wrapper/target/xcframework/AliceSDF.xcframework` (上位ディレクトリの生成物)

## ビルド方法

```bash
brew install xcodegen           # 初回のみ

# 1. xcframework 生成 (Phase A 前提)
cd ../../packaging/ios && ./build-xcframework.sh

# 2. SwiftUI サンプル生成 + ビルド
cd ../../samples/ios-swiftui
xcodegen                        # AliceSDFDemo.xcodeproj 生成
xcodebuild -project AliceSDFDemo.xcodeproj \
  -scheme AliceSDFDemo \
  -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath ./DerivedData \
  build
```

## ✅ 動作確認済 (2026-06-06, iPhone 17 Pro Simulator)

`Bridging Header 方式` で完全動作。`screenshots/AliceSDF-demo.png` 参照。

### 動作する設定の鍵

1. `AliceSDFDemo-Bridging-Header.h` で `#import "alice_sdfFFI.h"` (C ヘッダ直接 import)
2. `project.yml` の `SWIFT_OBJC_BRIDGING_HEADER` に bridging header path 指定
3. `HEADER_SEARCH_PATHS` に xcframework の `ios-arm64/Headers` と `ios-arm64_x86_64-simulator/Headers` 両方追加
4. xcframework 自体は `dependencies.framework` で参照 (`embed: false`、static lib なので)

### 罠だった点

xcodegen + XCFramework 直接依存だと、UniFFI が生成する `alice_sdfFFI.modulemap`
は自動的に Swift compiler に認識されない。Swift bindings の
`#if canImport(alice_sdfFFI)` チェックが false 倒れし、

```
cannot find 'uniffi_alice_sdf_mobile_checksum_func_*' in scope
```

エラーで失敗する。Bridging Header 経由で C ABI を bridge することで Swift 側に
シンボルを露出させ、`canImport` 失敗時のフォールバック (C 関数の直接呼出) が動く。

## 期待される画面 (実装目標)

```
┌──────────────────────────────┐
│ ALICE-SDF Mobile             │
│ v1.4.0 • iOS demo            │
├──────────────────────────────┤
│ query point  (1, 0, 0)       │
│ sphere1 d    0.4536          │
│ sphere2 d    1.5874          │
│ union        0.4536          │
│ smooth union 0.4216 (k=0.30) │
├──────────────────────────────┤
│ [2D SDF スライス Canvas]      │
│  ├─ 280x280 px               │
│  └─ z=0 平面、青=内部/黒=外部 │
├──────────────────────────────┤
│ Sphere Y offset    0.80      │
│ ━━━━●━━━━━━━━━━━━━           │
│ Radius             1.00      │
│ ━━━━━━●━━━━━━━━━━            │
│ Smooth K           0.30      │
│ ━━●━━━━━━━━━━━━━━            │
└──────────────────────────────┘
```
