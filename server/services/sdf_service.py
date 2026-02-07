"""SDF Service - wraps alice_sdf for JSON parsing, meshing, and export."""

import time
import tempfile
import os
from pathlib import Path
from typing import Optional

import alice_sdf


def validate_sdf_json(json_str: str) -> tuple[bool, Optional[int], Optional[str]]:
    """Validate an SDF JSON string.

    Returns (valid, node_count, error_message).
    """
    try:
        node = alice_sdf.from_json(json_str)
        return True, node.node_count(), None
    except Exception as e:
        return False, None, str(e)


def parse_sdf_json(json_str: str) -> alice_sdf.SdfNode:
    """Parse an SDF JSON string into an SdfNode."""
    return alice_sdf.from_json(json_str)


def node_to_json(node: alice_sdf.SdfNode) -> str:
    """Serialize an SdfNode to JSON string."""
    return alice_sdf.to_json(node)


def generate_mesh(
    node: alice_sdf.SdfNode,
    bounds_min: tuple[float, float, float] = (-5.0, -5.0, -5.0),
    bounds_max: tuple[float, float, float] = (5.0, 5.0, 5.0),
    resolution: int = 64,
):
    """Generate mesh from SdfNode. Returns (vertices, indices, stats)."""
    t0 = time.perf_counter()
    vertices, indices = alice_sdf.to_mesh(node, bounds_min, bounds_max, resolution)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    num_vertices = vertices.shape[0]
    num_triangles = indices.shape[0] // 3

    return vertices, indices, {
        "vertices": num_vertices,
        "triangles": num_triangles,
        "resolution": resolution,
        "mesh_time_ms": round(elapsed_ms, 2),
    }


def export_mesh_glb(vertices, indices) -> bytes:
    """Export mesh to GLB bytes."""
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        tmp_path = f.name

    try:
        alice_sdf.export_glb(vertices, indices, tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def export_mesh_obj(vertices, indices) -> bytes:
    """Export mesh to OBJ bytes."""
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp_path = f.name

    try:
        alice_sdf.export_obj(vertices, indices, tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def full_pipeline(
    sdf_json: str,
    resolution: int = 64,
    output_format: str = "glb",
    bounds_min: tuple[float, float, float] = (-5.0, -5.0, -5.0),
    bounds_max: tuple[float, float, float] = (5.0, 5.0, 5.0),
) -> tuple[bytes, dict]:
    """Full pipeline: JSON → parse → mesh → export.

    Returns (file_bytes, stats).
    """
    timings = {}

    t0 = time.perf_counter()
    node = parse_sdf_json(sdf_json)
    timings["parse_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    compiled = alice_sdf.compile_sdf(node)
    timings["compile_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    vertices, indices, mesh_stats = generate_mesh(
        node, bounds_min, bounds_max, resolution
    )
    timings["mesh_ms"] = mesh_stats["mesh_time_ms"]

    t0 = time.perf_counter()
    if output_format == "obj":
        data = export_mesh_obj(vertices, indices)
    else:
        data = export_mesh_glb(vertices, indices)
    timings["export_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    stats = {
        "node_count": node.node_count(),
        "vertices": mesh_stats["vertices"],
        "triangles": mesh_stats["triangles"],
        "timings": timings,
    }

    return data, stats
