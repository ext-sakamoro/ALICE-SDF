"""
ALICE-SDF Blender Add-on
========================

Blender 内から ALICE-SDF の .asdf ファイルを Mesh として import する add-on。

サポート対象:
    - Blender 4.0 / 4.1 / 4.2 LTS / 4.3 / 4.4+ (内蔵 Python 3.11)
    - bl_info の minimum は 4.0、新バージョンでも互換

Install (Blender 4.0+):
    Edit > Preferences > Add-ons > Install... → alice_sdf_blender.zip
    or copy this directory to:
        ~/Library/Application Support/Blender/<X.Y>/scripts/addons/   (macOS)
        ~/.config/blender/<X.Y>/scripts/addons/                       (Linux)
        %APPDATA%\\Blender Foundation\\Blender\\<X.Y>\\scripts\\addons\\ (Windows)
    例: Blender 4.4 なら 4.4、4.2 LTS なら 4.2

Backend: ALICE-SDF Python binding (PyO3 経由) を import する必要あり。
    pip install alice-sdf  (or cargo build --features python で .so 生成 → site-packages 配置)
"""

bl_info = {
    "name": "ALICE-SDF",
    "author": "Moroya Sakamoto",
    "version": (1, 6, 0),
    "blender": (4, 0, 0),
    "location": "File > Import / Object > ALICE-SDF",
    "description": "Import / generate Signed Distance Field meshes via ALICE-SDF",
    "warning": "Requires the `alice_sdf` Python package (PyO3 build)",
    "doc_url": "https://github.com/ext-sakamoro/ALICE-SDF",
    "category": "Import-Export",
}


def register():
    from . import operators, panels

    operators.register()
    panels.register()


def unregister():
    from . import operators, panels

    panels.unregister()
    operators.unregister()


if __name__ == "__main__":
    register()
