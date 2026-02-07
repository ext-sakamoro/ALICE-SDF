"""Tests for FastAPI endpoints (no LLM calls)."""

import json
import pytest
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)


VALID_SPHERE_JSON = json.dumps({
    "version": "0.1.0",
    "root": {"Sphere": {"radius": 1.0}},
    "metadata": None,
})


class TestHealthEndpoint:
    def test_health(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "providers" in data


class TestValidateEndpoint:
    def test_valid_sdf(self):
        resp = client.post(
            "/api/validate",
            json={"sdf_json": VALID_SPHERE_JSON},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["node_count"] == 1

    def test_invalid_sdf(self):
        resp = client.post(
            "/api/validate",
            json={"sdf_json": '{"version":"0.1.0","root":{"Bad":{}},"metadata":null}'},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert data["error"] is not None


class TestMeshEndpoint:
    def test_mesh_glb(self):
        resp = client.post(
            "/api/mesh",
            json={
                "sdf_json": VALID_SPHERE_JSON,
                "resolution": 16,
                "format": "glb",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "model/gltf-binary"
        assert resp.content[:4] == b"glTF"

    def test_mesh_obj(self):
        resp = client.post(
            "/api/mesh",
            json={
                "sdf_json": VALID_SPHERE_JSON,
                "resolution": 16,
                "format": "obj",
            },
        )
        assert resp.status_code == 200
        assert b"v " in resp.content


class TestExamplesEndpoint:
    def test_examples(self):
        resp = client.get("/api/examples")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 5
        for ex in data:
            assert "prompt" in ex
            assert "sdf_json" in ex


class TestViewerEndpoint:
    def test_viewer(self):
        resp = client.get("/api/viewer")
        assert resp.status_code == 200
        assert "Three.js" in resp.text or "three" in resp.text
