# ALICE-SDF Cinema 4D Plugin

Maxon Cinema 4D 2024 / 2025 / 2026+ 用 ALICE-SDF Python モジュール。`.asdf` import と SDF プリミティブ生成、C4D の PolygonObject として scene に追加。

## 対応バージョン

| C4D | Python ABI | 状態 |
|-----|-----------|------|
| 2024 (R26 系) | 3.11 | 🟢 |
| 2025 | 3.11 | 🟢 |
| 2026+ | 3.11+ | 🟢 |

## インストール

### 1. ALICE-SDF Python binding をビルド (C4D Python に合わせて)

```bash
cd /path/to/ALICE-SDF
cargo build --release --features python
```

C4D 内蔵 Python の site-packages にコピー (`<YEAR>` 置換):

| Platform | site-packages |
|----------|---------------|
| macOS | `/Applications/Maxon Cinema 4D <YEAR>/resource/modules/python/libs/macosx/python311.framework/lib/python3.11/site-packages/` |
| Windows | `C:\Program Files\Maxon Cinema 4D <YEAR>\resource\modules\python\libs\win64\python311.framework\lib\site-packages\` |

例 (macOS, C4D 2026):
```bash
cp target/release/libalice_sdf.dylib \
   "/Applications/Maxon Cinema 4D 2026/resource/modules/python/libs/macosx/python311.framework/lib/python3.11/site-packages/alice_sdf.so"
```

### 2. C4D plugins dir にモジュールをコピー

```bash
# macOS (例: C4D 2026)
cp -r bindings/cinema4d/alice_sdf_c4d "~/Library/Preferences/Maxon/Cinema 4D 2026/plugins/"

# Windows: %APPDATA%\Maxon\Cinema 4D 2026\plugins\
```

### 3. C4D で使う

Script Manager (Python tab):

```python
import alice_sdf_c4d

# プリミティブ追加 (C4D 単位は cm)
alice_sdf_c4d.add_sphere(radius=100.0, resolution=64)
alice_sdf_c4d.add_box(sx=100, sy=200, sz=300)
alice_sdf_c4d.add_torus(R=100, r=30)

# .asdf 読込
alice_sdf_c4d.import_asdf(
    "/path/to/model.asdf",
    bounds=(-300.0, 300.0),
    resolution=128,
)
```

## API 概要

| 関数 | 用途 |
|------|------|
| `is_available() -> bool` | alice_sdf backend 存在チェック |
| `version() -> str` | "1.6.0" |
| `add_sphere(radius, resolution, name)` | PolygonObject で球を生成 |
| `add_box(sx, sy, sz, resolution, name)` | 直方体 |
| `add_torus(R, r, resolution, name)` | トーラス |
| `import_asdf(filepath, bounds, resolution, name)` | .asdf import |

## 制限事項

- Material (C4D Standard / Redshift / Octane) との連携は v1.6 時点で未実装
- 本格的なメニュー登録には `.pyp` plugin ファイル + `RegisterCommandPlugin` が必要 (本モジュールは Script Manager 経由の helper)
- C4D 単位はメートル系 (1 unit = 1 cm が標準)、SDF 評価結果のスケーリングに注意
