"""Tests for SDF service."""

import json
import pytest
from server.services import sdf_service


VALID_SPHERE_JSON = json.dumps({
    "version": "0.1.0",
    "root": {"Sphere": {"radius": 1.0}},
    "metadata": None,
})

VALID_COMPLEX_JSON = json.dumps({
    "version": "0.1.0",
    "root": {
        "SmoothUnion": {
            "a": {
                "Translate": {
                    "child": {"Sphere": {"radius": 1.0}},
                    "offset": [0.0, 1.0, 0.0],
                }
            },
            "b": {"Box3d": {"half_extents": [2.0, 0.1, 2.0]}},
            "k": 0.2,
        }
    },
    "metadata": None,
})

INVALID_JSON = '{"version":"0.1.0","root":{"InvalidNode":{}}}'
MALFORMED_JSON = '{not valid json'


class TestValidateSdfJson:
    def test_valid_sphere(self):
        valid, count, err = sdf_service.validate_sdf_json(VALID_SPHERE_JSON)
        assert valid is True
        assert count == 1
        assert err is None

    def test_valid_complex(self):
        valid, count, err = sdf_service.validate_sdf_json(VALID_COMPLEX_JSON)
        assert valid is True
        assert count > 1
        assert err is None

    def test_invalid_node_type(self):
        valid, count, err = sdf_service.validate_sdf_json(INVALID_JSON)
        assert valid is False
        assert err is not None

    def test_malformed_json(self):
        valid, count, err = sdf_service.validate_sdf_json(MALFORMED_JSON)
        assert valid is False
        assert err is not None


class TestParseSdfJson:
    def test_parse_sphere(self):
        node = sdf_service.parse_sdf_json(VALID_SPHERE_JSON)
        assert node.node_count() == 1
        assert node.eval(0, 0, 0) == -1.0  # Inside sphere

    def test_parse_invalid_raises(self):
        with pytest.raises(Exception):
            sdf_service.parse_sdf_json(MALFORMED_JSON)


class TestNodeToJson:
    def test_roundtrip(self):
        node = sdf_service.parse_sdf_json(VALID_SPHERE_JSON)
        json_str = sdf_service.node_to_json(node)
        parsed = json.loads(json_str)
        assert "root" in parsed
        assert "Sphere" in parsed["root"]


class TestGenerateMesh:
    def test_sphere_mesh(self):
        node = sdf_service.parse_sdf_json(VALID_SPHERE_JSON)
        vertices, indices, stats = sdf_service.generate_mesh(
            node, (-2, -2, -2), (2, 2, 2), 16
        )
        assert vertices.shape[0] > 0
        assert vertices.shape[1] == 3
        assert indices.shape[0] > 0
        assert stats["vertices"] > 0
        assert stats["triangles"] > 0

    def test_complex_mesh(self):
        node = sdf_service.parse_sdf_json(VALID_COMPLEX_JSON)
        vertices, indices, stats = sdf_service.generate_mesh(
            node, (-5, -5, -5), (5, 5, 5), 16
        )
        assert stats["vertices"] > 0


class TestExportMesh:
    def test_export_glb(self):
        node = sdf_service.parse_sdf_json(VALID_SPHERE_JSON)
        vertices, indices, _ = sdf_service.generate_mesh(
            node, (-2, -2, -2), (2, 2, 2), 16
        )
        data = sdf_service.export_mesh_glb(vertices, indices)
        assert len(data) > 0
        # GLB magic number
        assert data[:4] == b"glTF"

    def test_export_obj(self):
        node = sdf_service.parse_sdf_json(VALID_SPHERE_JSON)
        vertices, indices, _ = sdf_service.generate_mesh(
            node, (-2, -2, -2), (2, 2, 2), 16
        )
        data = sdf_service.export_mesh_obj(vertices, indices)
        assert len(data) > 0
        text = data.decode("utf-8")
        assert "v " in text
        assert "f " in text


class TestFullPipeline:
    def test_pipeline_glb(self):
        data, stats = sdf_service.full_pipeline(VALID_SPHERE_JSON, resolution=16)
        assert len(data) > 0
        assert data[:4] == b"glTF"
        assert stats["node_count"] == 1
        assert stats["vertices"] > 0
        assert "timings" in stats

    def test_pipeline_obj(self):
        data, stats = sdf_service.full_pipeline(
            VALID_SPHERE_JSON, resolution=16, output_format="obj"
        )
        assert len(data) > 0
        assert b"v " in data
