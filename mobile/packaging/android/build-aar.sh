#!/bin/bash
# ALICE-SDF Mobile — Android AAR build script (.so + Kotlin bindings)
#
# Output: ../../uniffi-wrapper/target/aar/{jniLibs/<abi>/libalice_sdf_mobile.so, kotlin/}
#
# Note: 完全な .aar パッケージング (META-INF + manifest) には Android Gradle Plugin
# が必要だが、本スクリプトは .so + Kotlin bindings を生成するところまで。
# 受け取った側で Android Studio の "Import .jar/.aar" or Gradle module 化で取り込む。
#
# Usage:
#   export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk
#   ./build-aar.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER_DIR="$SCRIPT_DIR/../../uniffi-wrapper"
TARGET_DIR="$WRAPPER_DIR/target"
OUT_DIR="$TARGET_DIR/aar"
NDK="${ANDROID_NDK_HOME:-/opt/homebrew/share/android-ndk}"

if [ ! -d "$NDK" ]; then
  echo "ERROR: ANDROID_NDK_HOME not found at $NDK"
  echo "Install: brew install --cask android-ndk"
  exit 1
fi

echo "=== ALICE-SDF Android Build ==="
echo "Wrapper: $WRAPPER_DIR"
echo "NDK:     $NDK"
echo "Out:     $OUT_DIR"

# 1. Build .so for 4 Android ABIs via cargo-ndk
echo ""
echo ">>> Building 4 ABIs..."
(cd "$WRAPPER_DIR" && \
  ANDROID_NDK_HOME="$NDK" cargo ndk \
    -t arm64-v8a -t armeabi-v7a -t x86_64 -t x86 \
    -o "$OUT_DIR/jniLibs" \
    build --release)

# 2. Generate Kotlin bindings
echo ""
echo ">>> Generating Kotlin bindings..."
mkdir -p "$OUT_DIR/kotlin"
(cd "$WRAPPER_DIR" && cargo run --release --bin uniffi-bindgen -- \
  generate src/alice_sdf.udl \
  --language kotlin \
  --out-dir "$OUT_DIR/kotlin")

# 3. Report
echo ""
echo "=== Done ==="
echo " JNI libs: $OUT_DIR/jniLibs/{arm64-v8a,armeabi-v7a,x86_64,x86}/libalice_sdf_mobile.so"
echo " Kotlin:   $OUT_DIR/kotlin/uniffi/alice_sdf/alice_sdf.kt"
echo ""
echo "AAR 化手順 (Android Studio):"
echo "  1. 新規 Android Library module 作成"
echo "  2. \$OUT_DIR/jniLibs/* を src/main/jniLibs/ にコピー"
echo "  3. \$OUT_DIR/kotlin/*.kt を src/main/java/ にコピー"
echo "  4. build.gradle に net.java.dev.jna:jna:5.13.0@aar を追加"
echo "  5. ./gradlew :assembleRelease で AAR 生成"
