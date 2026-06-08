#!/bin/bash
# ALICE-SDF Mobile — iOS / visionOS XCFramework build script
#
# Output: ../../uniffi-wrapper/target/xcframework/AliceSDF.xcframework + Swift bindings
#
# Targets:
#   - iOS device (aarch64-apple-ios)
#   - iOS simulator (aarch64 + x86_64 via lipo)
#   - visionOS device (aarch64-apple-visionos) — Xcode 15.2+ + nightly Rust 推奨
#   - visionOS simulator (aarch64-apple-visionos-sim)
#
# Usage:
#   ./build-xcframework.sh                  # iOS のみ
#   ./build-xcframework.sh --with-visionos  # iOS + visionOS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER_DIR="$SCRIPT_DIR/../../uniffi-wrapper"
TARGET_DIR="$WRAPPER_DIR/target"
OUT_DIR="$TARGET_DIR/xcframework"
LIB_NAME="libalice_sdf_mobile.a"

WITH_VISIONOS=false
for arg in "$@"; do
  if [ "$arg" = "--with-visionos" ]; then
    WITH_VISIONOS=true
  fi
done

echo "=== ALICE-SDF iOS/visionOS XCFramework Build ==="
echo "Wrapper: $WRAPPER_DIR"
echo "Out:     $OUT_DIR"
echo "visionOS: $WITH_VISIONOS"

# 1. iOS targets
IOS_TARGETS=("aarch64-apple-ios" "aarch64-apple-ios-sim" "x86_64-apple-ios")
for target in "${IOS_TARGETS[@]}"; do
  echo ""
  echo ">>> Building $target..."
  (cd "$WRAPPER_DIR" && cargo build --release --target "$target")
done

# 2. visionOS targets (optional, nightly Rust 必要)
if [ "$WITH_VISIONOS" = "true" ]; then
  VISIONOS_TARGETS=("aarch64-apple-visionos" "aarch64-apple-visionos-sim")
  for target in "${VISIONOS_TARGETS[@]}"; do
    echo ""
    echo ">>> Building $target (nightly + -Z build-std)..."
    # visionOS の Rust target は Tier 3 で nightly + -Z build-std=panic_abort,std が必要
    (cd "$WRAPPER_DIR" && \
      rustup target list --installed | grep -q "$target" || rustup +nightly target add "$target" || true)
    (cd "$WRAPPER_DIR" && \
      cargo +nightly build --release --target "$target" \
      -Z build-std=panic_abort,std \
      --target-dir "$TARGET_DIR")
  done
fi

# 3. lipo simulator slices (arm64 + x86_64) for iOS
mkdir -p "$OUT_DIR/sim"
echo ""
echo ">>> lipo iOS simulator slices..."
lipo -create \
  "$TARGET_DIR/aarch64-apple-ios-sim/release/$LIB_NAME" \
  "$TARGET_DIR/x86_64-apple-ios/release/$LIB_NAME" \
  -output "$OUT_DIR/sim/$LIB_NAME"

# 4. Generate Swift bindings via UniFFI
echo ""
echo ">>> Generating Swift bindings..."
mkdir -p "$OUT_DIR/swift"
(cd "$WRAPPER_DIR" && cargo run --release --bin uniffi-bindgen -- \
  generate src/alice_sdf.udl \
  --language swift \
  --out-dir "$OUT_DIR/swift")

# 5. Build XCFramework (iOS device + simulator [+ visionOS])
echo ""
echo ">>> Building XCFramework..."
rm -rf "$OUT_DIR/AliceSDF.xcframework"

XCFRAMEWORK_ARGS=(
  -library "$TARGET_DIR/aarch64-apple-ios/release/$LIB_NAME"
  -headers "$OUT_DIR/swift"
  -library "$OUT_DIR/sim/$LIB_NAME"
  -headers "$OUT_DIR/swift"
)

if [ "$WITH_VISIONOS" = "true" ]; then
  XCFRAMEWORK_ARGS+=(
    -library "$TARGET_DIR/aarch64-apple-visionos/release/$LIB_NAME"
    -headers "$OUT_DIR/swift"
    -library "$TARGET_DIR/aarch64-apple-visionos-sim/release/$LIB_NAME"
    -headers "$OUT_DIR/swift"
  )
fi

xcodebuild -create-xcframework \
  "${XCFRAMEWORK_ARGS[@]}" \
  -output "$OUT_DIR/AliceSDF.xcframework"

echo ""
echo "=== Done ==="
echo " XCFramework: $OUT_DIR/AliceSDF.xcframework"
echo " Swift bindings: $OUT_DIR/swift/"
echo ""
echo "Swift package usage:"
echo "  import AliceSDF"
echo "  let d = sdfSphere(point: Vec3(x: 1, y: 0, z: 0), center: Vec3(x: 0, y: 0, z: 0), radius: 1.0)"
