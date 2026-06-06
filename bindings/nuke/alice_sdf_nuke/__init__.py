"""
ALICE-SDF Nuke plugin
======================

Foundry Nuke 15 / 16+ 用 ALICE-SDF 連携 Python モジュール。
VFX コンポジット内で SDF → ボリュームレンダリング (DeepRecolor 等) や
.asdf スキャンライン化に使用。

サポート対象 (Python ABI):
    - Nuke 15.x: Python 3.10
    - Nuke 16.x: Python 3.11

Install:
    1. ALICE-SDF Python binding を Nuke の Python に合わせてビルド:
        cargo build --release --features python
    2. Nuke 内蔵 Python の site-packages にコピー (<VER>, <PYVER> 置換):
        macOS:   /Applications/Nuke<VER>/Nuke<VER>.app/Contents/MacOS/lib/python<PYVER>/site-packages/
        Linux:   /usr/local/Nuke<VER>/lib/python<PYVER>/site-packages/
        Windows: C:\\Program Files\\Nuke<VER>\\lib\\python<PYVER>\\site-packages\\
    3. bindings/nuke/alice_sdf_nuke/ を Nuke plugin dir にコピー:
        ~/.nuke/  (全 OS 共通、Nuke 全バージョン共有)
    4. ~/.nuke/menu.py に以下を追加:
        import alice_sdf_nuke
        alice_sdf_nuke.register_menu()
"""

import importlib
import importlib.util

__version__ = "1.6.0"

_alice_sdf = None


def get_backend():
    global _alice_sdf
    if _alice_sdf is None:
        _alice_sdf = importlib.import_module("alice_sdf")
    return _alice_sdf


def is_available() -> bool:
    return importlib.util.find_spec("alice_sdf") is not None


def version() -> str:
    return __version__


def export_asdf_as_volume(filepath, bounds=(-2.0, 2.0), resolution=128, out_path=None):
    """`.asdf` を読み込み、密 voxel 配列としてバイナリ書き出し。

    Nuke の DeepRead や Volume ノードへ渡す中間ファイルとして利用。
    出力形式: ALICEVDB1 (alice_sdf.io.vdb 互換)。
    """
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed")
    backend = get_backend()
    node = backend.load_asdf(filepath)
    if out_path is None:
        out_path = filepath.replace(".asdf", ".alicevdb").replace(".json", "")
    bytes_data = backend.bake_to_vdb(node, bounds, resolution)
    with open(out_path, "wb") as f:
        f.write(bytes_data)
    return out_path


def render_sdf_slice(filepath, z_world, width=512, height=512, bounds=(-2.0, 2.0)):
    """`.asdf` の z 平面スライスを numpy.ndarray (RGBA uint8) で返す。

    Nuke の Python ノード内で `nuke.Image` に変換して下流に流せる。
    """
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed")
    backend = get_backend()
    node = backend.load_asdf(filepath)
    return backend.render_slice_2d(node, z_world, width, height, bounds)


def register_menu():
    """Nuke のメイン メニューに「ALICE-SDF」を追加。"""
    import nuke

    menu = nuke.menu("Nuke").addMenu("ALICE-SDF")
    menu.addCommand("Export .asdf as Volume...", _menu_export_volume)
    menu.addCommand("Render .asdf slice...", _menu_render_slice)
    return menu


def _menu_export_volume():
    import nuke

    asdf = nuke.getFilename("Select .asdf to export", "*.asdf *.asdf.json")
    if not asdf:
        return
    out = export_asdf_as_volume(asdf)
    nuke.message(f"Exported: {out}")


def _menu_render_slice():
    import nuke

    asdf = nuke.getFilename("Select .asdf to render slice", "*.asdf *.asdf.json")
    if not asdf:
        return
    arr = render_sdf_slice(asdf, z_world=0.0)
    nuke.message(f"Rendered slice: shape={arr.shape}")
