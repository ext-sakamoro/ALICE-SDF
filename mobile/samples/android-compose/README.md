# ALICE-SDF Demo (Android Jetpack Compose)

ALICE-SDF Mobile を Kotlin + Jetpack Compose アプリから呼び出す最小サンプル。

## ✅ 動作確認済 (2026-06-06, Pixel 6 emulator, Android 14 / API 34, arm64-v8a)

`screenshots/AliceSDF-android-demo.png` 参照。iOS 版と数学的に完全一致 (sphere d=0.2806、smooth union=0.2056) — bindings の Linux/Android 移植正常性を実証。

## 構成

```
android-compose/
├── settings.gradle.kts
├── build.gradle.kts                 # root
├── gradle.properties
├── local.properties                 # SDK location (gitignore 済)
├── app/
│   ├── build.gradle.kts             # AGP 8.5.2 + Kotlin 2.0.0 + Compose
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/
│       │   ├── net/alicelaw/alicesdf/MainActivity.kt   # Compose UI
│       │   └── uniffi/alice_sdf/alice_sdf.kt           # UniFFI 生成 Kotlin bindings
│       └── jniLibs/{arm64-v8a,armeabi-v7a,x86_64,x86}/
│           └── libuniffi_alice_sdf.so                  # Rust core (ABI ごと)
└── screenshots/AliceSDF-android-demo.png
```

## ビルド方法

```bash
# 1. ALICE-SDF Mobile の AAR コンポーネント (.so + Kotlin) を生成
cd ../../packaging/android
./build-aar.sh

# 2. (.so をリネーム) UniFFI Kotlin は libuniffi_<namespace>.so を期待
for abi in arm64-v8a armeabi-v7a x86_64 x86; do
  cp ../../uniffi-wrapper/target/aar/jniLibs/$abi/libalice_sdf_mobile.so \
     app/src/main/jniLibs/$abi/libuniffi_alice_sdf.so
done
cp ../../uniffi-wrapper/target/aar/kotlin/uniffi/alice_sdf/alice_sdf.kt \
   app/src/main/java/uniffi/alice_sdf/

# 3. local.properties 作成
echo "sdk.dir=/Users/ys/Library/Android/sdk" > local.properties

# 4. gradle wrapper 初期化 (初回のみ)
gradle wrapper --gradle-version 8.7 --distribution-type all

# 5. ビルド + emulator install + 起動
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n net.alicelaw.alicesdf/.MainActivity
```

## ⚠️ 罠 (重要)

### .so のファイル名は `libuniffi_<namespace>.so` でなければならない

UniFFI Kotlin は `Native.load("uniffi_<namespace>")` で .so を探す
(`<namespace>` は UDL の namespace 名 = `alice_sdf`)。

ALICE-SDF Mobile のクレート名は `alice-sdf-mobile`、lib name は
`alice_sdf_mobile` なので、cargo-ndk が生成する .so 名は `libalice_sdf_mobile.so`。
これを **`libuniffi_alice_sdf.so` にリネーム** しないと、

```
java.lang.UnsatisfiedLinkError: Unable to load library 'uniffi_alice_sdf':
dlopen failed: library "libuniffi_alice_sdf.so" not found
```

で即 crash する。

### Kotlin 2.0+ では Compose Compiler Gradle Plugin が必須

```kts
// root build.gradle.kts
id("org.jetbrains.kotlin.android") version "2.0.0" apply false
id("org.jetbrains.kotlin.plugin.compose") version "2.0.0" apply false

// app/build.gradle.kts
id("org.jetbrains.kotlin.plugin.compose")
```

`kotlinCompilerExtensionVersion` の手動指定は不要 (plugin が自動でセット)。

## 期待される画面

(iOS 版とほぼ同等 UI、Material 3 のスタイル違いのみ)

```
┌──────────────────────────────┐
│ ALICE-SDF Mobile             │
│ v1.4.0 · Android demo        │
├──────────────────────────────┤
│ query point  (1, 0, 0)       │
│ sphere1 d    0.2806          │
│ sphere2 d    0.2806          │
│ union        0.2806          │
│ smooth union 0.2056 (k=0.30) │
├──────────────────────────────┤
│ [2D SDF スライス Canvas]      │
│  ├─ 2 球 smooth_union         │
│  └─ dumbbell / 図 8 形状      │
├──────────────────────────────┤
│ Sphere Y offset    0.80      │
│ Radius             1.00      │
│ Smooth K           0.30      │
└──────────────────────────────┘
```
