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

## ⚠️ 既知の問題 (Lv3 残課題)

xcodegen + XCFramework 直接依存だと、UniFFI が生成する `alice_sdfFFI` モジュールが
Swift compiler から見えず、

```
cannot find 'uniffi_alice_sdf_mobile_checksum_func_*' in scope
```

エラーで失敗する。`#if canImport(alice_sdfFFI)` のチェックが false に倒れるため。

### 暫定回避策 (検証中)

1. SwiftPM パッケージ (`mobile/swift-package/`) 経由で取り込む — モジュール解決を SPM に委譲
2. Bridging Header (`AliceSDFDemo-Bridging-Header.h`) で `#import "alice_sdfFFI.h"` 直接 import
3. 手動で Xcode を開き、Drag & Drop で XCFramework を追加 (Xcode の自動セットアップ任せ)

### 一方確実に動く構成

- iOS 実機 / シミュレータ向けに xcframework ビルド: **OK** ✅
- Swift bindings (alice_sdf.swift) 生成: **OK** ✅
- SwiftPM package 構造定義 (`mobile/swift-package/Package.swift`): **OK** ✅
- アプリ統合 / SwiftUI 起動: **検証中** (上記)

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
