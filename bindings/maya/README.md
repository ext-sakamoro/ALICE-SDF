# ALICE-SDF Maya Plugin

Autodesk Maya 2024 / 2025 / 2026+ 用 ALICE-SDF Python モジュール。`.asdf` import と SDF プリミティブ生成。

## 対応バージョン

| Maya | Python ABI | 状態 |
|------|-----------|------|
| 2024 | 3.10 | 🟢 |
| 2025 | 3.11 | 🟢 |
| 2026 | 3.11+ | 🟢 |

## インストール

### 1. ALICE-SDF Python binding を該当 Python ABI でビルド

```bash
cd /path/to/ALICE-SDF
cargo build --release --features python
```

Maya 内蔵 Python の site-packages にコピー (バージョンを `<YEAR>`、Python を `<PYVER>` で置換):

| Platform | site-packages |
|----------|---------------|
| macOS | `/Applications/Autodesk/maya<YEAR>/Maya.app/Contents/MacOS/lib/python<PYVER>/site-packages/` |
| Linux | `/usr/autodesk/maya<YEAR>/lib/python<PYVER>/site-packages/` |
| Windows | `C:\Program Files\Autodesk\Maya<YEAR>\bin\Python\Lib\site-packages\` |

例:
```bash
# Maya 2024 (Python 3.10) - macOS
cp target/release/libalice_sdf.dylib \
   /Applications/Autodesk/maya2024/Maya.app/Contents/MacOS/lib/python3.10/site-packages/alice_sdf.so

# Maya 2025 (Python 3.11) - macOS
cp target/release/libalice_sdf.dylib \
   /Applications/Autodesk/maya2025/Maya.app/Contents/MacOS/lib/python3.11/site-packages/alice_sdf.so

# Maya 2026 (Python 3.11+) - macOS
cp target/release/libalice_sdf.dylib \
   /Applications/Autodesk/maya2026/Maya.app/Contents/MacOS/lib/python3.11/site-packages/alice_sdf.so
```

### 2. Maya scripts dir にモジュールをコピー

```bash
# macOS (例: Maya 2026)
cp -r bindings/maya/alice_sdf_maya ~/Library/Preferences/Autodesk/maya/2026/scripts/
# Maya 2024 / 2025 でも同じパターン、年だけ変える
cp -r bindings/maya/alice_sdf_maya ~/Library/Preferences/Autodesk/maya/2025/scripts/

# Linux
cp -r bindings/maya/alice_sdf_maya ~/maya/2026/scripts/

# Windows: %USERPROFILE%\Documents\maya\2026\scripts\
```

### 3. Maya で有効化

Maya 起動 → Script Editor (Python タブ):

```python
import alice_sdf_maya
alice_sdf_maya.register_menu()  # 上部メニューに「ALICE-SDF」追加
```

`userSetup.py` に上記を入れれば起動時自動登録。

## 使い方

### メニュー
- `ALICE-SDF > Add Sphere / Box / Torus` — プリミティブを mesh 化して spawn
- `ALICE-SDF > Import .asdf...` — ファイル選択ダイアログから .asdf 読込

### Python API

```python
import alice_sdf_maya

# プリミティブ追加
alice_sdf_maya.add_sphere(radius=1.5, resolution=64)
alice_sdf_maya.add_box(sx=1, sy=2, sz=3)
alice_sdf_maya.add_torus(R=1, r=0.3)

# .asdf 読込
alice_sdf_maya.import_asdf(
    "/path/to/model.asdf",
    bounds=(-3.0, 3.0),
    resolution=128,
)
```

## API 概要

| 関数 | 用途 |
|------|------|
| `is_available() -> bool` | alice_sdf backend 確認 |
| `version() -> str` | "1.6.0" |
| `add_sphere(radius, resolution, name)` | 球 mesh を MFnMesh で生成 |
| `add_box(sx, sy, sz, resolution, name)` | 箱 |
| `add_torus(R, r, resolution, name)` | トーラス |
| `import_asdf(filepath, bounds, resolution, name)` | .asdf import |
| `register_menu()` | "ALICE-SDF" トップメニュー登録 |

## 制限事項

- v1.6 時点で Shader Network 連携 (Arnold/Maya Shader) 未実装
- アニメーション再評価は手動 (将来 Expression Driver 経由で自動化予定)
- macOS Maya 2024 は Python 3.10 (Windows/Linux 同様)。Python 3.11+ 環境では `alice_sdf` の Python ABI を 3.10 でビルド必要
