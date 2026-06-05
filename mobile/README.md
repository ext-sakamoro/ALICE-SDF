# ALICE-SDF Mobile — iOS / Android SDK

ALICE-SDF v1.4.0+ を iOS (Swift) と Android (Kotlin) から呼べる UniFFI ベース SDK。

## 動作確認済構成

- **iOS targets**: `aarch64-apple-ios` (実機), `aarch64-apple-ios-sim` (M-series simulator), `x86_64-apple-ios` (Intel simulator)
- **Android targets**: `arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86`
- **toolchain**: rustc stable, UniFFI 0.28, cargo-ndk 3+
- **配布物**: iOS は `.xcframework`、Android は `.so` + Kotlin bindings

## ビルド方法

### 事前準備

```bash
# Rust targets
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android

# Tools
cargo install cargo-ndk

# Android NDK
brew install --cask android-ndk
export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk
```

### iOS XCFramework

```bash
cd mobile/packaging/ios
./build-xcframework.sh
```

出力:
- `mobile/uniffi-wrapper/target/xcframework/AliceSDF.xcframework` (device + simulator fat)
- `mobile/uniffi-wrapper/target/xcframework/swift/alice_sdf.swift` (Swift API)

### Android .so + Kotlin

```bash
cd mobile/packaging/android
./build-aar.sh
```

出力:
- `mobile/uniffi-wrapper/target/aar/jniLibs/{abi}/libalice_sdf_mobile.so` (4 ABI)
- `mobile/uniffi-wrapper/target/aar/kotlin/uniffi/alice_sdf/alice_sdf.kt`

## 公開 API (Lv2 最小実装)

### Primitives
- `sdf_sphere(point, center, radius) -> f32`
- `sdf_box(point, center, half_extents) -> f32`
- `sdf_rounded_box(point, center, half_extents, round_radius) -> f32`
- `sdf_cylinder(point, radius, half_height) -> f32`
- `sdf_torus(point, major_radius, minor_radius) -> f32`
- `sdf_plane(point, normal, distance) -> f32`

### Operations
- `op_union(a, b)` / `op_intersection(a, b)` / `op_subtraction(a, b)`
- `op_smooth_union(a, b, k)` / `op_smooth_intersection(a, b, k)` / `op_smooth_subtraction(a, b, k)`

### Batch
- `sphere_batch(points, center, radius) -> [f32]` — 配列で一括評価

### Vec3
```rust
struct Vec3 { x: f32, y: f32, z: f32 }
```

## Swift 使用例

```swift
import AliceSDF

let center = Vec3(x: 0, y: 0, z: 0)
let point  = Vec3(x: 1, y: 0, z: 0)
let d = sdfSphere(point: point, center: center, radius: 1.0)
// d ≈ 0 (on sphere surface)

let dSmooth = opSmoothUnion(a: 0.5, b: 0.6, k: 0.1)
// dSmooth < 0.5 (smooth union pulls below min)
```

## Kotlin 使用例

```kotlin
import uniffi.alice_sdf.*

val center = Vec3(0f, 0f, 0f)
val point  = Vec3(1f, 0f, 0f)
val d = sdfSphere(point, center, 1.0f)
// d ≈ 0

val dSmooth = opSmoothUnion(0.5f, 0.6f, 0.1f)
```

## バイナリサイズ

| Platform | Library | Size |
|----------|---------|------|
| iOS device (arm64) | `libalice_sdf_mobile.a` (static) | 44 MB |
| iOS simulator (arm64+x86_64) | `libalice_sdf_mobile.a` (static fat) | 88 MB |
| Android arm64-v8a | `libalice_sdf_mobile.so` (dynamic) | 360 KB |
| Android armv7 | `libalice_sdf_mobile.so` (dynamic) | 248 KB |
| Android x86_64 | `libalice_sdf_mobile.so` (dynamic) | 396 KB |
| Android x86 | `libalice_sdf_mobile.so` (dynamic) | 388 KB |
| Swift bindings | `alice_sdf.swift` | 26 KB |
| Kotlin bindings | `alice_sdf.kt` | 50 KB |

iOS 側が大きいのは **static lib + 全シンボル + 全ターゲット fat** のため。App 側の dead code elimination で実際の app size は < 5MB に収束する。

## AAR パッケージング (Android)

`build-aar.sh` は .so + Kotlin bindings まで生成。最終 .aar 化は Android Studio で:

1. 新規 Android Library module 作成
2. `target/aar/jniLibs/*` を `src/main/jniLibs/` にコピー
3. `target/aar/kotlin/*.kt` を `src/main/java/` にコピー
4. `build.gradle` に `implementation 'net.java.dev.jna:jna:5.13.0@aar'` 追加 (UniFFI runtime)
5. `./gradlew :assembleRelease` で AAR 生成

## XCFramework 統合 (iOS)

Xcode プロジェクトに XCFramework をドラッグ&ドロップ + Swift ファイルを追加するだけ。

```
ios/App/
├── AliceSDF.xcframework/   ← ドラッグ
└── alice_sdf.swift         ← コピー
```

## 制限事項 (Lv2 範囲)

- 既存 ALICE-SDF の 1077+ tests カバーする全 API のうち、**12 関数だけ wrap**
- メッシュ生成 / SVO / GPU rendering / JIT などは未対応 (Lv3 で wgpu Metal/Vulkan 検証予定)
- 構造体ベースの SDF tree builder (`SdfBuilder`) は未実装、現状は pure function のみ

## 関連ドキュメント

- 本体 API: `/API.md`
- アーキテクチャ: `/ARCHITECTURE.md`
- C-ABI FFI (Unity/Unreal 用): `/src/ffi/`
