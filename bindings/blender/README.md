# ALICE-SDF Blender Add-on

Blender 4.0+ 用 ALICE-SDF プラグイン。`.asdf` のインポートと SDF プリミティブの
Mesh 生成を Blender 内 UI から実行できる。

## インストール

### 1. ALICE-SDF Python binding をビルド

```bash
cd /path/to/ALICE-SDF
cargo build --release --features python
```

これで `target/release/libalice_sdf.dylib` (macOS) / `.so` (Linux) / `.dll` (Windows) が
できる。Blender 内蔵 Python の `site-packages` にリネームコピー:

| Platform | Blender Python | Target name |
|----------|----------------|-------------|
| macOS    | `/Applications/Blender.app/Contents/Resources/4.x/python/lib/python3.11/site-packages/` | `alice_sdf.so` |
| Linux    | `~/.config/blender/4.x/python/lib/python3.11/site-packages/` | `alice_sdf.so` |
| Windows  | `%APPDATA%\Blender Foundation\Blender\4.x\python\Lib\site-packages\` | `alice_sdf.pyd` |

例 (macOS):
```bash
cp target/release/libalice_sdf.dylib \
   /Applications/Blender.app/Contents/Resources/4.0/python/lib/python3.11/site-packages/alice_sdf.so
```

### 2. Blender Add-on を install

```bash
zip -r alice_sdf_blender.zip alice_sdf_blender
```

Blender を起動 → `Edit > Preferences > Add-ons > Install...` → 上記 zip を選択 →
"ALICE-SDF" を check して有効化。

## 使い方

### N-Panel (3D Viewport)

`N` キーで side panel を開き、`ALICE-SDF` タブで:
- `Add SDF Sphere` — 球の SDF を Mesh 化して spawn
- `Add SDF Box` — 直方体
- `Add SDF Torus` — トーラス
- `Import .asdf` — 既存ファイル読み込み

### Add メニュー

3D Viewport で `Shift + A > ALICE-SDF Sphere` 等から直接追加。

### Import メニュー

`File > Import > ALICE-SDF (.asdf)` で `.asdf` / `.asdf.json` を読み込み。

## API バックエンド要件

Add-on は Blender 内蔵 Python から `import alice_sdf` で以下を呼ぶ:

- `alice_sdf.sphere(radius)` → `SdfNode`
- `alice_sdf.box(size_x, size_y, size_z)` → `SdfNode`
- `alice_sdf.torus(major_radius, minor_radius)` → `SdfNode`
- `alice_sdf.load_asdf(path)` → `SdfNode`
- `alice_sdf.sdf_to_mesh(node, bounds, resolution)` → `(verts: List[Tuple[f32,3]], faces: List[Tuple[int,3]])`

ALICE-SDF 本体 v1.6 系の PyO3 binding (`--features python`) が提供する API と一致。

## 制限事項

- Material 評価 (PBR) は v1.6 時点では未実装 (Mesh のみ生成)
- 高 resolution (>128) では Blender 側 Mesh 構築がボトルネック
- modifier / animation の再 evaluation は手動 (将来 driver 経由で自動化予定)
