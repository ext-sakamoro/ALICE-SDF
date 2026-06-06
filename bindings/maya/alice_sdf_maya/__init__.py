"""
ALICE-SDF Maya plugin
======================

Autodesk Maya 2024 / 2025 / 2026+ 用 ALICE-SDF 連携 Python モジュール。

サポート対象 (Python ABI):
    - Maya 2024: Python 3.10
    - Maya 2025: Python 3.11
    - Maya 2026: Python 3.11+

Install:
    1. ALICE-SDF Python binding を Maya の Python と同じバージョンでビルド:
        cargo build --release --features python
    2. Maya 起動 Python の site-packages にコピー (<YEAR>, <PYVER> 置換):
        macOS:   /Applications/Autodesk/maya<YEAR>/Maya.app/Contents/MacOS/lib/python<PYVER>/site-packages/
        Linux:   /usr/autodesk/maya<YEAR>/lib/python<PYVER>/site-packages/
        Windows: C:\\Program Files\\Autodesk\\Maya<YEAR>\\bin\\Python\\Lib\\site-packages\\
        ファイル名: alice_sdf.so (macOS/Linux) / alice_sdf.pyd (Windows)
    3. bindings/maya/alice_sdf_maya/ を Maya scripts dir にコピー:
        macOS:   ~/Library/Preferences/Autodesk/maya/<YEAR>/scripts/
        Linux:   ~/maya/<YEAR>/scripts/
        Windows: %USERPROFILE%\\Documents\\maya\\<YEAR>\\scripts\\
    4. Maya の Script Editor で実行:
        import alice_sdf_maya
        alice_sdf_maya.register_menu()
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


def _sdf_to_maya_mesh(node, bounds=(-2.0, 2.0), resolution=48, name="aliceSdfMesh"):
    """SdfNode → Maya MFnMesh API で polygon mesh を作成。"""
    import maya.api.OpenMaya as om

    backend = get_backend()
    verts_py, faces_py = backend.sdf_to_mesh(node, bounds, resolution)

    points = om.MPointArray()
    for v in verts_py:
        points.append(om.MPoint(v[0], v[1], v[2]))

    face_counts = om.MIntArray()
    face_connects = om.MIntArray()
    for face in faces_py:
        face_counts.append(len(face))
        for vi in face:
            face_connects.append(vi)

    mesh_fn = om.MFnMesh()
    mesh_obj = mesh_fn.create(points, face_counts, face_connects)
    dag = om.MFnDagNode(mesh_obj)
    dag.setName(name)
    return mesh_obj


def add_sphere(radius=1.0, resolution=48, name="aliceSdfSphere"):
    """SDF 球を Maya scene に追加。"""
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed (cargo build --features python)")
    node = get_backend().sphere(radius)
    return _sdf_to_maya_mesh(node, (-radius * 1.5, radius * 1.5), resolution, name)


def add_box(sx=1.0, sy=1.0, sz=1.0, resolution=48, name="aliceSdfBox"):
    """SDF 箱を Maya scene に追加。"""
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed")
    node = get_backend().box(sx, sy, sz)
    bmax = max(sx, sy, sz) * 1.5
    return _sdf_to_maya_mesh(node, (-bmax, bmax), resolution, name)


def add_torus(R=1.0, r=0.3, resolution=64, name="aliceSdfTorus"):
    """SDF トーラスを Maya scene に追加。"""
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed")
    node = get_backend().torus(R, r)
    bmax = (R + r) * 1.3
    return _sdf_to_maya_mesh(node, (-bmax, bmax), resolution, name)


def import_asdf(filepath, bounds=(-2.0, 2.0), resolution=64, name="aliceSdfImported"):
    """`.asdf` を読み込んで Maya scene に mesh として追加。"""
    if not is_available():
        raise RuntimeError("alice_sdf backend not installed")
    node = get_backend().load_asdf(filepath)
    return _sdf_to_maya_mesh(node, bounds, resolution, name)


def register_menu():
    """Maya のメイン メニューに「ALICE-SDF」を追加。"""
    import maya.cmds as cmds

    menu_name = "AliceSdfMenu"
    if cmds.menu(menu_name, exists=True):
        cmds.deleteUI(menu_name)
    menu = cmds.menu(
        menu_name,
        label="ALICE-SDF",
        parent="MayaWindow",
        tearOff=True,
    )
    cmds.menuItem(parent=menu, label="Add Sphere", command=lambda *_: add_sphere())
    cmds.menuItem(parent=menu, label="Add Box", command=lambda *_: add_box())
    cmds.menuItem(parent=menu, label="Add Torus", command=lambda *_: add_torus())
    cmds.menuItem(parent=menu, divider=True)
    cmds.menuItem(
        parent=menu,
        label="Import .asdf...",
        command=lambda *_: _file_dialog_import(),
    )
    return menu


def _file_dialog_import():
    import maya.cmds as cmds

    result = cmds.fileDialog2(fileMode=1, fileFilter="ALICE-SDF (*.asdf *.asdf.json)")
    if result:
        import_asdf(result[0])
