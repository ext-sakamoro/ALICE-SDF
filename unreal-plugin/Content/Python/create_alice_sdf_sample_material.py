"""Create the ALICE-SDF sample material in the open project.

A `.uasset` cannot be shipped as text, so the plugin ships the shader
(`Shaders/Public/AliceSdfSample.ush`) plus this script, which builds the
material that uses it:

    /Game/AliceSDF/M_AliceSDF_Sample   Unlit, two-sided, a Custom expression
                                       that raymarches the sample SDF

Run it from the editor (the Python Editor Script Plugin must be enabled):

    Tools > Execute Python Script...  -> create_alice_sdf_sample_material.py

or from the console:

    py create_alice_sdf_sample_material.py

or headless:

    UnrealEditor-Cmd.exe <Project>.uproject -run=pythonscript
        -script="create_alice_sdf_sample_material.py"

Assign the material to any mesh (a cube works): the SDF is raymarched inside
the object's bounds, centred on the actor origin.

Author: Moroya Sakamoto
"""

import unreal

PACKAGE_PATH = "/Game/AliceSDF"
ASSET_NAME = "M_AliceSDF_Sample"

CUSTOM_CODE = """// ALICE-SDF sample: raymarched SDF surface.
// Swap AliceSdfSample_Scene in the .ush for the output of
// UAliceSdfComponent::GenerateHlsl() to show your own shape.
#include "/Plugin/AliceSDF/Public/AliceSdfSample.ush"
return AliceSdfSample_Shade(WorldPosition, ObjectPosition, CameraPosition);
"""


def _custom_input(name):
    """A named input pin on a Custom expression.

    `unreal.CustomInput(input_name=...)` is not constructible with keywords
    in UE 5.7 (`call() takes at most 0 arguments`), so the property is set
    afterwards.
    """
    pin = unreal.CustomInput()
    pin.set_editor_property("input_name", name)
    return pin


def _add(material, expression_class, x, y):
    return unreal.MaterialEditingLibrary.create_material_expression(
        material, expression_class, x, y
    )


def create_sample_material():
    """Create (or replace) the sample material and return it."""
    asset_path = "{}/{}".format(PACKAGE_PATH, ASSET_NAME)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        unreal.EditorAssetLibrary.delete_asset(asset_path)

    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        ASSET_NAME, PACKAGE_PATH, unreal.Material, unreal.MaterialFactoryNew()
    )
    if material is None:
        raise RuntimeError("could not create {}".format(asset_path))

    # Unlit: the shader returns finished colour, so no lighting is applied on
    # top of it. Two-sided + masked-free so the volume is visible from inside.
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    material.set_editor_property("two_sided", True)

    custom = _add(material, unreal.MaterialExpressionCustom, -400, 0)
    custom.set_editor_property("code", CUSTOM_CODE)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    custom.set_editor_property("description", "ALICE-SDF raymarch")

    world_position = _add(material, unreal.MaterialExpressionWorldPosition, -900, -120)
    object_position = _add(material, unreal.MaterialExpressionObjectPositionWS, -900, 20)
    camera_position = _add(material, unreal.MaterialExpressionCameraPositionWS, -900, 160)

    custom.set_editor_property(
        "inputs",
        [
            _custom_input("WorldPosition"),
            _custom_input("ObjectPosition"),
            _custom_input("CameraPosition"),
        ],
    )
    # Custom inputs are addressed by the names declared above.
    for node, input_name in (
        (world_position, "WorldPosition"),
        (object_position, "ObjectPosition"),
        (camera_position, "CameraPosition"),
    ):
        unreal.MaterialEditingLibrary.connect_material_expressions(
            node, "", custom, input_name
        )

    unreal.MaterialEditingLibrary.connect_material_property(
        custom, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR
    )

    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_asset(asset_path)
    unreal.log("ALICE-SDF: created {}".format(asset_path))
    return material


if __name__ == "__main__":
    create_sample_material()
