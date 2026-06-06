# ALICE-SDF Houdini Python Plugin

SideFX Houdini 20 / 20.5 / 21+ で ALICE-SDF を呼び出すための Python モジュール集。

## 対応バージョン

| Houdini | Python ABI | 状態 |
|---------|-----------|------|
| 20.0    | 3.10 | 🟢 |
| 20.5    | 3.11 | 🟢 |
| 21+     | 3.11+ | 🟢 |

## 構成

```
houdini/
├── python/alice_sdf_hou/        # Houdini Python に置く本体
│   ├── __init__.py              # backend wrapper (alice_sdf を遅延 import)
│   ├── sop_sdf_load.py          # Python SOP body: .asdf loader
│   └── sop_sdf_primitive.py     # Python SOP body: sphere/box/torus generator
├── install.sh                    # python3.11libs/ への自動インストール
└── sample_hda/                   # 将来の .hda アセット用 (Houdini で作成)
```

## インストール

### 1. ALICE-SDF Python binding をビルド

```bash
cd /path/to/ALICE-SDF
cargo build --release --features python
```

Houdini 内蔵 Python の site-packages に配置 (macOS 例):
```bash
cp target/release/libalice_sdf.dylib \
   /Applications/Houdini/Houdini20.5/Frameworks/Houdini.framework/Versions/Current/Resources/houdini/python3.11libs/alice_sdf.so
```

### 2. Houdini モジュールを install

```bash
cd bindings/houdini
./install.sh   # $HSITE or $HOUDINI_USER_PREF_DIR を自動検出
```

または手動で `python/alice_sdf_hou/` を以下のいずれかへ:
- `$HSITE/python3.11libs/` (推奨、複数 Houdini 版で共有)
- `$HOUDINI_USER_PREF_DIR/python3.11libs/` (ユーザー固有)
- `$HFS/houdini/python3.11libs/` (Houdini インストール先)

## 使い方

### Python SOP — .asdf Loader

1. `Object > Geometry` を作成、内部に `Python` SOP を追加
2. `Code` タブを開く
3. `python/alice_sdf_hou/sop_sdf_load.py` の `# Houdini Python SOP body` 内容を貼り付け
4. Parameter Editor で以下を追加:
   - `file` (String) — .asdf ファイルパス
   - `bounds_min` (Float, default -2.0)
   - `bounds_max` (Float, default 2.0)
   - `resolution` (Integer, default 64)
5. Cook ボタンで .asdf → Mesh

### Python SOP — SDF Primitive Generator

`sop_sdf_primitive.py` の中身を Python SOP に貼り付け、以下 parameter を追加:
- `shape` (Menu) — "sphere", "box", "torus"
- `radius` (Float)
- `minor_radius` (Float) — torus
- `size_x`, `size_y`, `size_z` (Float) — box
- `resolution` (Integer)

### .hda 化 (Tools Developer → Digital Asset 作成)

上記 Python SOP を作成後、

```
Tools > Save as Digital Asset
   Operator Type: alice_sdf_sphere (任意)
   File: $HSITE/otls/alice_sdf_primitive.hda
```

で `.hda` 化、再利用可能。`sample_hda/` は将来の公式アセット配布用ディレクトリ。

## API 概要

`alice_sdf_hou` モジュールが提供:

| 関数 | 用途 |
|------|------|
| `is_available() -> bool` | alice_sdf backend 存在チェック |
| `version() -> str` | "1.6.0" |
| `load_asdf(path) -> SdfNode` | .asdf 読込 |
| `sphere(r) -> SdfNode` | プリミティブ生成 |
| `box(x, y, z) -> SdfNode` | 〃 |
| `torus(R, r) -> SdfNode` | 〃 |
| `sdf_to_hou_geo(sdf, geo, bounds, resolution)` | SDF → hou.Geometry にメッシュ流し込み |

## 制限事項

- v1.6 時点で `.hda` バイナリアセット同梱なし (Houdini 起動環境がないと作成不可)
- Material / Houdini Shader への連携は未実装
- VOP (vex) からの SDF 評価は未対応 (将来 vex C++ HDK プラグイン化予定)
