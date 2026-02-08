#!/bin/bash
# update_counts.sh — Parse src/types.rs section markers and update counts everywhere
# Author: Moroya Sakamoto
#
# Usage: ./scripts/update_counts.sh
#
# Parses the SdfNode enum in types.rs to count variants in each category,
# then updates:
#   - src/types.rs (SdfCategory::count constants)
#   - README.md / README_JP.md
#   - unreal-plugin/README.md

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TYPES_FILE="$REPO_ROOT/src/types.rs"

if [ ! -f "$TYPES_FILE" ]; then
    echo "Error: $TYPES_FILE not found"
    exit 1
fi

# Extract enum body and count variants per section
enum_body() {
    sed -n '/^pub enum SdfNode/,/^impl SdfNode/p' "$TYPES_FILE"
}

PRIM_3D=$(enum_body | sed -n '/=== Primitives ===/,/=== 2D Primitives/p' | grep -cE '^\s+[A-Z][a-zA-Z0-9_]+ \{')
PRIM_2D=$(enum_body | sed -n '/=== 2D Primitives/,/=== Operations ===/p' | grep -cE '^\s+[A-Z][a-zA-Z0-9_]+ \{')
PRIMITIVES=$((PRIM_3D + PRIM_2D))
OPERATIONS=$(enum_body | sed -n '/=== Operations ===/,/=== Transforms ===/p' | grep -cE '^\s+[A-Z][a-zA-Z0-9_]+ \{')
TRANSFORMS=$(enum_body | sed -n '/=== Transforms ===/,/=== Modifiers ===/p' | grep -cE '^\s+[A-Z][a-zA-Z0-9_]+ \{')
MODIFIERS=$(enum_body | sed -n '/=== Modifiers ===/,/^}/p' | grep -cE '^\s+[A-Z][a-zA-Z0-9_]+ \{')
TOTAL=$((PRIMITIVES + OPERATIONS + TRANSFORMS + MODIFIERS))

echo "=== SDF Variant Counts ==="
echo "  Primitives:  $PRIMITIVES (3D: $PRIM_3D, 2D: $PRIM_2D)"
echo "  Operations:  $OPERATIONS"
echo "  Transforms:  $TRANSFORMS"
echo "  Modifiers:   $MODIFIERS"
echo "  Total:       $TOTAL"
echo ""

# --- Update src/types.rs SdfCategory::count() ---
echo "Updating src/types.rs..."
sed -i '' "s/SdfCategory::Primitive => [0-9]*/SdfCategory::Primitive => $PRIMITIVES/" "$TYPES_FILE"
sed -i '' "s/SdfCategory::Operation => [0-9]*/SdfCategory::Operation => $OPERATIONS/" "$TYPES_FILE"
sed -i '' "s/SdfCategory::Transform => [0-9]*/SdfCategory::Transform => $TRANSFORMS/" "$TYPES_FILE"
sed -i '' "s/SdfCategory::Modifier => [0-9]*/SdfCategory::Modifier => $MODIFIERS/" "$TYPES_FILE"

# --- Update READMEs ---
update_readme_counts() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "  Skipping $file (not found)"
        return
    fi
    echo "Updating $file..."
    # Only update summary lines (e.g. "72 primitives, 24 operations, 4 transforms, 19 modifiers")
    # and "all N primitives" lines. Do NOT touch gradient tables or JIT-specific counts.
    sed -i '' -E "s/[0-9]+ primitives, [0-9]+ (CSG )?operations, [0-9]+ transforms, [0-9]+ modifiers/$PRIMITIVES primitives, $OPERATIONS \1operations, $TRANSFORMS transforms, $MODIFIERS modifiers/g" "$file"
    sed -i '' -E "s/[0-9]+ primitives, [0-9]+ CSG operations/$PRIMITIVES primitives, $OPERATIONS CSG operations/g" "$file"
    sed -i '' -E "s/all [0-9]+ primitives/all $PRIMITIVES primitives/g" "$file"
}

update_readme_counts "$REPO_ROOT/README.md"
update_readme_counts "$REPO_ROOT/README_JP.md"
update_readme_counts "$REPO_ROOT/unreal-plugin/README.md"

echo ""
echo "Done! Run 'cargo check' to verify."
