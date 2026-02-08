#!/usr/bin/env bash
# ALICE-SDF UE5 Plugin Build & Package Script
# Author: Moroya Sakamoto
#
# Usage:
#   ./scripts/build_ue5_plugin.sh          # Build for current platform
#   ./scripts/build_ue5_plugin.sh --zip    # Build + create distributable zip
#
# Output:
#   unreal-plugin/ThirdParty/AliceSDF/lib/{Platform}/  - Native library
#   unreal-plugin/ThirdParty/AliceSDF/include/         - C header
#   AliceSDF-UE5-Plugin-{platform}.zip                 - Distributable archive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_DIR="$ROOT_DIR/unreal-plugin"
THIRDPARTY="$PLUGIN_DIR/ThirdParty/AliceSDF"
FEATURES="ffi hlsl glsl gpu"
CREATE_ZIP=false

if [[ "${1:-}" == "--zip" ]]; then
    CREATE_ZIP=true
fi

echo "=== ALICE-SDF UE5 Plugin Builder ==="
echo ""

# Detect platform
case "$(uname -s)" in
    Darwin*)
        PLATFORM="Mac"
        LIB_NAME="libalice_sdf.dylib"
        ;;
    Linux*)
        PLATFORM="Linux"
        LIB_NAME="libalice_sdf.so"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        PLATFORM="Win64"
        LIB_NAME="alice_sdf.dll"
        ;;
    *)
        echo "ERROR: Unsupported platform: $(uname -s)"
        exit 1
        ;;
esac

echo "[1/4] Building ALICE-SDF (release, features: $FEATURES)..."
cd "$ROOT_DIR"
cargo build --release --features "$FEATURES"
echo "      Done."

echo "[2/4] Setting up ThirdParty directory..."
mkdir -p "$THIRDPARTY/include"
mkdir -p "$THIRDPARTY/lib/Win64"
mkdir -p "$THIRDPARTY/lib/Mac"
mkdir -p "$THIRDPARTY/lib/Linux"

# Copy header
cp "$ROOT_DIR/include/alice_sdf.h" "$THIRDPARTY/include/"

# Copy native library
cp "$ROOT_DIR/target/release/$LIB_NAME" "$THIRDPARTY/lib/$PLATFORM/"
echo "      Copied $LIB_NAME -> ThirdParty/AliceSDF/lib/$PLATFORM/"

# On Windows, also copy .lib (import library)
if [[ "$PLATFORM" == "Win64" ]]; then
    if [[ -f "$ROOT_DIR/target/release/alice_sdf.lib" ]]; then
        cp "$ROOT_DIR/target/release/alice_sdf.lib" "$THIRDPARTY/lib/Win64/"
        echo "      Copied alice_sdf.lib -> ThirdParty/AliceSDF/lib/Win64/"
    fi
fi

echo "[3/4] Verifying plugin structure..."
EXPECTED_FILES=(
    "$PLUGIN_DIR/AliceSDF.uplugin"
    "$PLUGIN_DIR/Source/AliceSDF/AliceSDF.Build.cs"
    "$PLUGIN_DIR/Source/AliceSDF/Public/AliceSdfComponent.h"
    "$PLUGIN_DIR/Source/AliceSDF/Private/AliceSdfComponent.cpp"
    "$PLUGIN_DIR/Source/AliceSDF/Private/AliceSdfModule.cpp"
    "$THIRDPARTY/include/alice_sdf.h"
    "$THIRDPARTY/lib/$PLATFORM/$LIB_NAME"
)

ALL_OK=true
for f in "${EXPECTED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "      OK: $(basename "$f")"
    else
        echo "      MISSING: $f"
        ALL_OK=false
    fi
done

if [[ "$ALL_OK" != "true" ]]; then
    echo ""
    echo "ERROR: Some files are missing. Check the output above."
    exit 1
fi

LIB_SIZE=$(du -sh "$THIRDPARTY/lib/$PLATFORM/$LIB_NAME" | cut -f1)
echo "      Library size: $LIB_SIZE"

echo "[4/4] Plugin ready!"

if [[ "$CREATE_ZIP" == "true" ]]; then
    echo ""
    echo "Creating distributable zip..."
    PLATFORM_LOWER=$(echo "$PLATFORM" | tr '[:upper:]' '[:lower:]')
    ZIP_NAME="AliceSDF-UE5-Plugin-${PLATFORM_LOWER}.zip"
    cd "$ROOT_DIR"
    zip -r "$ZIP_NAME" unreal-plugin/ \
        -x "unreal-plugin/.git/*" \
        -x "unreal-plugin/.DS_Store" \
        -x "*/Intermediate/*" \
        -x "*/Binaries/*"
    ZIP_SIZE=$(du -sh "$ZIP_NAME" | cut -f1)
    echo "Created: $ZIP_NAME ($ZIP_SIZE)"
fi

echo ""
echo "=== Installation ==="
echo "1. Copy 'unreal-plugin/' to YourProject/Plugins/AliceSDF/"
echo "2. Open your .uproject in UE5"
echo "3. Enable the ALICE-SDF plugin"
echo "4. Restart the editor"
echo ""
echo "Or download from GitHub Releases (no Rust required)."
