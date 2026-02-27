#!/bin/bash
# update-ue5-dylib.sh — Copy latest ALICE-SDF dylib to unreal-plugin
# Run after: cargo build --release
#
# Author: Moroya Sakamoto

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SRC="$PROJECT_ROOT/target/release/libalice_sdf.dylib"
DST="$PROJECT_ROOT/unreal-plugin/ThirdParty/AliceSDF/lib/Mac/libalice_sdf.dylib"

if [ ! -f "$SRC" ]; then
    echo "ERROR: Source dylib not found: $SRC"
    echo "Run 'cargo build --release' first."
    exit 1
fi

cp "$SRC" "$DST"
echo "Updated: $DST"
echo "  Size: $(wc -c < "$DST" | tr -d ' ') bytes"
echo "  Date: $(stat -f '%Sm' "$DST")"
