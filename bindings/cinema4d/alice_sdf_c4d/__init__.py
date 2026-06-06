"""
ALICE-SDF Cinema 4D plugin
============================

Maxon Cinema 4D 2024 / 2025 / 2026+ 用 ALICE-SDF 連携 Python モジュール。

サポート対象 (Python ABI):
    - C4D 2024 (R26 系列): Python 3.11
    - C4D 2025:            Python 3.11
    - C4D 2026:            Python 3.11+

Install:
    1. ALICE-SDF Python binding を Cinema 4D Python に合わせてビルド:
        cargo build --release --features python
    2. C4D 内蔵 Python の site-packages にコピー:
        macOS:   /Applications/Maxon Cinema 4D <YEAR>/resource/modules/python/libs/macosx/python311.framework/lib/python3.11/site-packages/
        Windows: C:\\Program Files\\Maxon Cinema 4D <YEAR>\\resource\\modules\\python\\libs\\win64\\python311.framework\\lib\\site-packages\\
    3. bindings/cinema4d/alice_sdf_c4d/ を C4D plugins dir にコピー:
        macOS:   ~/Library/Preferences/Maxon/Cinema 4D <YEAR>/plugins/
        Windows: %APPDATA%\\Maxon\\Cinema 4D <YEAR>\\plugins\\
    4. C4D 起動 → Script Manager で:
        import alice_sdf_c4d
        alice_sdf_c4d.register_menu()
"""

import importlib
import importlib.util

__version__ = "1.6.0"
_alice_sdf = None


def get_backend():
    """alice_sdf module (PyO3 binding) を遅延 import。"""
    global _alice_sdf
    if _alice_sdf is None:
        _alice_sdf = importlib.import_module("alice_sdf")
    return _alice_sdf


def is_available() -> bool:
    return importlib.util.find_spec("alice_sdf") is not None


def version() -> str:
    return __version__


def _sdf_to_c4d_polygon_object(node, bounds=(-2.0, 2.0), resolution=48, name="aliceSdfMesh"):
    """SdfNode → Cinema 4D PolygonObject。"""
    import c4d

    backend = get_backend()
    verts_py, faces_py = backend.sdf_to_mesh(node, bounds, resolution)

    poly = c4d.PolygonObject(len(verts_py), len(faces_py))
    poly.SetName(name)

    for i, v in enumerate(verts_py):
        poly.SetPoint(i, c4d.Vector(v[0], v[1], v[2]))

    for i, f in enumerate(faces_py):
        if len(f) == 3:
            poly.SetPolygon(i, c4d.CPolygon(f[0], f[1], f[2], f[2]))
        elif len(f) == 4:
            poly.SetPolygon(i, c4d.CPolygon(f[0], f[1], f[2], f[3]))

    poly.Message(c4d.MSG_UPDATE)
    return poly


def add_sphere(radius=100.0, resolution=48, name="aliceSdfSphere"):
    """SDF 球を C4D scene に追加 (C4D 単位は cm が標準)。"""
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed (cargo build --features python)")
    import c4d

    node = get_backend().sphere(radius)
    poly = _sdf_to_c4d_polygon_object(node, (-radius * 1.5, radius * 1.5), resolution, name)
    doc = c4d.documents.GetActiveDocument()
    doc.InsertObject(poly)
    doc.SetActiveObject(poly)
    c4d.EventAdd()
    return poly


def add_box(sx=100.0, sy=100.0, sz=100.0, resolution=48, name="aliceSdfBox"):
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed")
    import c4d

    node = get_backend().box(sx, sy, sz)
    bmax = max(sx, sy, sz) * 1.5
    poly = _sdf_to_c4d_polygon_object(node, (-bmax, bmax), resolution, name)
    doc = c4d.documents.GetActiveDocument()
    doc.InsertObject(poly)
    doc.SetActiveObject(poly)
    c4d.EventAdd()
    return poly


def add_torus(R=100.0, r=30.0, resolution=64, name="aliceSdfTorus"):
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed")
    import c4d

    node = get_backend().torus(R, r)
    bmax = (R + r) * 1.3
    poly = _sdf_to_c4d_polygon_object(node, (-bmax, bmax), resolution, name)
    doc = c4d.documents.GetActiveDocument()
    doc.InsertObject(poly)
    doc.SetActiveObject(poly)
    c4d.EventAdd()
    return poly


def import_asdf(filepath, bounds=(-2.0, 2.0), resolution=64, name="aliceSdfImported"):
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed")
    import c4d

    node = get_backend().load_asdf(filepath)
    poly = _sdf_to_c4d_polygon_object(node, bounds, resolution, name)
    doc = c4d.documents.GetActiveDocument()
    doc.InsertObject(poly)
    doc.SetActiveObject(poly)
    c4d.EventAdd()
    return poly


def register_menu():
    """C4D の Plugins メニューに「ALICE-SDF」を追加 (Script Manager から呼ぶ)。

    Note: C4D の menu 登録は通常 c4d.plugins.RegisterCommandPlugin で行うため、
    本 helper は便宜的にトップメニューへ追加する形を取る。本格的な plugin 化は
    `alice_sdf_c4d_command.py` 等を別途用意。
    """
    import c4d

    # シンプルにメッセージ表示のみ (本格 plugin 登録は別 .pyp ファイル化を推奨)
    c4d.gui.MessageDialog(
        "ALICE-SDF helpers loaded.\n"
        "Use: alice_sdf_c4d.add_sphere(), .add_box(), .add_torus(), .import_asdf()"
    )
