#!/bin/bash
# ALICE-SDF Houdini Python module installer
#
# Houdini 20+ の HSITE / HOUDINI_USER_PREF_DIR / HFS どこに置くか自動検出して
# python3.11libs/ にコピーする。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/python/alice_sdf_hou"

if [ ! -d "$SRC" ]; then
  echo "ERROR: $SRC not found"
  exit 1
fi

candidates=()
[ -n "${HSITE:-}" ] && candidates+=("$HSITE")
[ -n "${HOUDINI_USER_PREF_DIR:-}" ] && candidates+=("$HOUDINI_USER_PREF_DIR")

# macOS default
HOME_HOUDINI=$(ls -d "$HOME/Library/Preferences/houdini/"* 2>/dev/null | sort -r | head -1 || true)
[ -n "$HOME_HOUDINI" ] && candidates+=("$HOME_HOUDINI")
# Linux default
LINUX_HOUDINI=$(ls -d "$HOME/houdini"* 2>/dev/null | sort -r | head -1 || true)
[ -n "$LINUX_HOUDINI" ] && candidates+=("$LINUX_HOUDINI")

if [ ${#candidates[@]} -eq 0 ]; then
  echo "ERROR: no Houdini install dir detected"
  echo "  set \$HSITE or \$HOUDINI_USER_PREF_DIR and re-run"
  exit 1
fi

TARGET_BASE="${candidates[0]}"
# Houdini 20.0 = python3.10libs, 20.5+ = python3.11libs。
# 既存ディレクトリを優先、無ければデフォルト 3.11
if [ -d "$TARGET_BASE/python3.11libs" ]; then
  PYLIBS="python3.11libs"
elif [ -d "$TARGET_BASE/python3.10libs" ]; then
  PYLIBS="python3.10libs"
else
  PYLIBS="python3.11libs"  # default for Houdini 20.5/21+
fi
TARGET="$TARGET_BASE/$PYLIBS/alice_sdf_hou"
echo ">>> Installing to: $TARGET"
mkdir -p "$(dirname "$TARGET")"
rm -rf "$TARGET"
cp -r "$SRC" "$TARGET"
echo "OK ($(find "$TARGET" -type f | wc -l | tr -d ' ') files)"
echo ""
echo "Next steps:"
echo "  1. Houdini を起動"
echo "  2. Python SOP を作成"
echo "  3. python/alice_sdf_hou/sop_sdf_load.py または sop_sdf_primitive.py の中身を"
echo "     Python SOP の Code フィールドに貼り付け、parameter を作成"
