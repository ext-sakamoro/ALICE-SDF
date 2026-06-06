# ALICE-SDF Nuke Plugin

Foundry Nuke 15 / 16+ で ALICE-SDF を呼び出す Python モジュール。VFX コンポジット内で `.asdf` を
ボリューム / スライス画像として展開する。

## 対応バージョン

| Nuke | Python ABI | 状態 |
|------|-----------|------|
| 15.x | 3.10 | 🟢 |
| 16.x | 3.11 | 🟢 |

## インストール

### 1. ALICE-SDF Python binding をビルド (Nuke Python に合わせて)

```bash
cd /path/to/ALICE-SDF
cargo build --release --features python
```

### 2. Nuke 内蔵 Python の site-packages にコピー (`<VER>`, `<PYVER>` 置換)

| Platform | site-packages |
|----------|---------------|
| macOS | `/Applications/Nuke<VER>/Nuke<VER>.app/Contents/MacOS/lib/python<PYVER>/site-packages/` |
| Linux | `/usr/local/Nuke<VER>/lib/python<PYVER>/site-packages/` |
| Windows | `C:\Program Files\Nuke<VER>\lib\python<PYVER>\site-packages\` |

例:
```bash
# Nuke 15.0v0 (Python 3.10) - macOS
cp target/release/libalice_sdf.dylib \
   /Applications/Nuke15.0v0/Nuke15.0v0.app/Contents/MacOS/lib/python3.10/site-packages/alice_sdf.so

# Nuke 16.0v0 (Python 3.11) - macOS
cp target/release/libalice_sdf.dylib \
   /Applications/Nuke16.0v0/Nuke16.0v0.app/Contents/MacOS/lib/python3.11/site-packages/alice_sdf.so
```

### 3. Nuke plugin dir にモジュールをコピー

```bash
cp -r bindings/nuke/alice_sdf_nuke ~/.nuke/
```

### 4. `~/.nuke/menu.py` に追加

```python
import alice_sdf_nuke
alice_sdf_nuke.register_menu()
```

## 使い方

### メニュー

- `ALICE-SDF > Export .asdf as Volume...` — .asdf 選択ダイアログ → ALICEVDB1 バイナリ書出
- `ALICE-SDF > Render .asdf slice...` — z 平面スライスを RGBA 画像化

### Python API

```python
import alice_sdf_nuke

# Volume export (DeepRead / Volume node 向け)
out = alice_sdf_nuke.export_asdf_as_volume(
    "/path/to/model.asdf",
    bounds=(-2.0, 2.0),
    resolution=128,
    out_path="/tmp/model.alicevdb",
)

# Slice render (numpy.ndarray, shape=(h, w, 4))
arr = alice_sdf_nuke.render_sdf_slice(
    "/path/to/model.asdf",
    z_world=0.0,
    width=1024,
    height=1024,
    bounds=(-2.5, 2.5),
)
```

## 制限事項

- ALICEVDB1 形式は Nuke ネイティブのボリュームノードでは直接読めない (専用 reader が必要)
- Deep image 連携 (DeepRead/DeepWrite) は未実装
- Foundry Mari (テクスチャ) や Katana (ルックデブ) への連携も将来予定
