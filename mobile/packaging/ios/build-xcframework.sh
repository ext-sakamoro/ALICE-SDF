#!/bin/bash
# ALICE-SDF Mobile — iOS XCFramework build script
#
# Output: ../../uniffi-wrapper/target/xcframework/AliceSDF.xcframework + Swift bindings
#
# Usage: ./build-xcframework.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER_DIR="$SCRIPT_DIR/../../uniffi-wrapper"
TARGET_DIR="$WRAPPER_DIR/target"
OUT_DIR="$TARGET_DIR/xcframework"
LIB_NAME="libalice_sdf_mobile.a"

echo "=== ALICE-SDF iOS XCFramework Build ==="
echo "Wrapper: $WRAPPER_DIR"
echo "Out:     $OUT_DIR"

# 1. Build static libraries for 3 iOS targets
TARGETS=("aarch64-apple-ios" "aarch64-apple-ios-sim" "x86_64-apple-ios")
for target in "${TARGETS[@]}"; do
  echo ""
  echo ">>> Building $target..."
  (cd "$WRAPPER_DIR" && cargo build --release --target "$target")
done

# 2. Combine simulator slices (arm64 + x86_64) via lipo
mkdir -p "$OUT_DIR/sim"
echo ""
echo ">>> lipo simulator slices..."
lipo -create \
  "$TARGET_DIR/aarch64-apple-ios-sim/release/$LIB_NAME" \
  "$TARGET_DIR/x86_64-apple-ios/release/$LIB_NAME" \
  -output "$OUT_DIR/sim/$LIB_NAME"

# 3. Generate Swift bindings via UniFFI
echo ""
echo ">>> Generating Swift bindings..."
mkdir -p "$OUT_DIR/swift"
(cd "$WRAPPER_DIR" && cargo run --release --bin uniffi-bindgen -- \
  generate src/alice_sdf.udl \
  --language swift \
  --out-dir "$OUT_DIR/swift")

# 4. Build XCFramework (device + simulator)
echo ""
echo ">>> Building XCFramework..."
rm -rf "$OUT_DIR/AliceSDF.xcframework"
xcodebuild -create-xcframework \
  -library "$TARGET_DIR/aarch64-apple-ios/release/$LIB_NAME" \
  -headers "$OUT_DIR/swift" \
  -library "$OUT_DIR/sim/$LIB_NAME" \
  -headers "$OUT_DIR/swift" \
  -output "$OUT_DIR/AliceSDF.xcframework"

echo ""
echo "=== Done ==="
echo " XCFramework: $OUT_DIR/AliceSDF.xcframework"
echo " Swift bindings: $OUT_DIR/swift/"
echo ""
echo "Swift package usage:"
echo "  import AliceSDF"
echo "  let d = sdfSphere(point: Vec3(x: 1, y: 0, z: 0), center: Vec3(x: 0, y: 0, z: 0), radius: 1.0)"
