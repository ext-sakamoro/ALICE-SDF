"""
ALICE-SDF Houdini Python plugin
================================

SideFX Houdini 20 / 20.5 / 21+ 用 ALICE-SDF 連携 Python モジュール。

Houdini 内の Python SOP / Python パネル / HDA 内 Python から `import alice_sdf_hou`
で各種 SDF 操作を呼び出せる。

サポート対象 (Python ABI):
    - Houdini 20.0: Python 3.10
    - Houdini 20.5: Python 3.11
    - Houdini 21+:  Python 3.11+

Install:
    bindings/houdini/python/ 配下を以下にコピー (Houdini Python に合わせて
    python3.10libs/ または python3.11libs/ を選ぶ):
        $HSITE/python<PYVER>libs/                  (推奨、複数 Houdini 版で共有)
        $HOUDINI_USER_PREF_DIR/python<PYVER>libs/  (ユーザー固有)
        $HFS/houdini/python<PYVER>libs/             (Houdini インストール先)

Backend: ALICE-SDF Python binding (PyO3 経由) を import する必要あり。
    cargo build --release --features python → libalice_sdf.dylib/.so/.dll
    Houdini 内蔵 Python の site-packages に Python ABI 別 にリネームコピー。
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
    """alice_sdf backend がロード可能か。"""
    return importlib.util.find_spec("alice_sdf") is not None


def version() -> str:
    return __version__


def load_asdf(path: str):
    """Load .asdf / .asdf.json file → SdfNode."""
    return get_backend().load_asdf(path)


def sphere(radius: float):
    """Create sphere SDF."""
    return get_backend().sphere(radius)


def box(size_x: float, size_y: float, size_z: float):
    """Create box SDF."""
    return get_backend().box(size_x, size_y, size_z)


def torus(major_radius: float, minor_radius: float):
    """Create torus SDF."""
    return get_backend().torus(major_radius, minor_radius)


def sdf_to_hou_geo(node, geo, bounds=(-2.0, 2.0), resolution: int = 64):
    """SdfNode から Houdini hou.Geometry に頂点+面を流し込む。

    Args:
        node: SDF node (alice_sdf.SdfNode)
        geo: Houdini hou.Geometry instance (Python SOP 内では hou.Geometry())
        bounds: (min, max) 立方体範囲
        resolution: 1 辺の voxel 数
    """
    verts, faces = get_backend().sdf_to_mesh(node, bounds, resolution)
    point_handles = [geo.createPoint() for _ in verts]
    for ph, v in zip(point_handles, verts):
        ph.setPosition(v)
    for f in faces:
        poly = geo.createPolygon()
        for vi in f:
            poly.addVertex(point_handles[vi])
    return geo
