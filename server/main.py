"""ALICE-SDF Text-to-3D FastAPI server."""

import base64
import json
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import alice_sdf

from server.models import (
    GenerateRequest,
    GenerateResponse,
    ValidateRequest,
    ValidateResponse,
    MeshRequest,
    HealthResponse,
)
from server.services import sdf_service, llm_service
from server.prompts.examples import EXAMPLES
from server import config

app = FastAPI(
    title="ALICE-SDF Text-to-3D",
    description="Generate real 3D geometry from text descriptions using LLM + SDF",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"


# ── REST Endpoints ──────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=alice_sdf.version(),
        providers={
            "claude": bool(config.ANTHROPIC_API_KEY),
            "gemini": bool(config.GOOGLE_API_KEY),
        },
    )


@app.post("/api/validate", response_model=ValidateResponse)
async def validate(req: ValidateRequest):
    valid, node_count, error = sdf_service.validate_sdf_json(req.sdf_json)
    return ValidateResponse(valid=valid, node_count=node_count, error=error)


@app.post("/api/mesh")
async def mesh(req: MeshRequest):
    node = sdf_service.parse_sdf_json(req.sdf_json)
    vertices, indices, stats = sdf_service.generate_mesh(
        node, req.bounds_min, req.bounds_max, req.resolution
    )

    if req.format == "obj":
        data = sdf_service.export_mesh_obj(vertices, indices)
        media_type = "text/plain"
        filename = "mesh.obj"
    else:
        data = sdf_service.export_mesh_glb(vertices, indices)
        media_type = "model/gltf-binary"
        filename = "mesh.glb"

    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    total_t0 = time.perf_counter()

    try:
        sdf_json, llm_meta = await llm_service.generate_sdf(
            req.prompt, req.provider, req.model
        )
    except Exception as e:
        return GenerateResponse(error=f"LLM error: {e}")

    # Auto-select resolution if not specified
    resolution = req.resolution
    if resolution is None:
        resolution = sdf_service.recommend_resolution(sdf_json, req.quality)

    if req.format == "json":
        valid, node_count, err = sdf_service.validate_sdf_json(sdf_json)
        # Generate low-res preview GLB
        preview_b64 = None
        try:
            node = sdf_service.parse_sdf_json(sdf_json)
            pv, pi, _ = sdf_service.generate_mesh(node, (-5, -5, -5), (5, 5, 5), 16)
            preview_bytes = sdf_service.export_mesh_glb(pv, pi)
            preview_b64 = base64.b64encode(preview_bytes).decode()
        except Exception:
            pass
        return GenerateResponse(
            sdf_json=sdf_json,
            node_count=node_count,
            timings=llm_meta,
            preview_glb_base64=preview_b64,
        )

    try:
        data, stats = sdf_service.full_pipeline(
            sdf_json, resolution, req.format
        )
    except Exception as e:
        return GenerateResponse(
            sdf_json=sdf_json,
            error=f"Mesh error: {e}",
            timings=llm_meta,
        )

    total_ms = round((time.perf_counter() - total_t0) * 1000, 2)
    timings = {**llm_meta, **stats.get("timings", {}), "total_ms": total_ms}

    media_type = "model/gltf-binary" if req.format == "glb" else "text/plain"
    filename = f"scene.{req.format}"

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-SDF-JSON": base64.b64encode(sdf_json.encode()).decode(),
            "X-Stats": json.dumps(timings),
        },
    )


@app.post("/api/cache/clear")
async def cache_clear():
    count = llm_service.cache_clear()
    return {"cleared": count}


@app.get("/api/examples")
async def examples():
    return [
        {"prompt": ex["prompt"], "sdf_json": ex["sdf_json"]}
        for ex in EXAMPLES
    ]


@app.get("/api/viewer", response_class=HTMLResponse)
async def viewer():
    viewer_path = STATIC_DIR / "viewer.html"
    return HTMLResponse(content=viewer_path.read_text())


# ── WebSocket Endpoint ──────────────────────────────────────────


async def _safe_send(ws: WebSocket, data: dict) -> bool:
    """Send JSON to WebSocket, returning False if disconnected."""
    try:
        await ws.send_json(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()
            prompt = data.get("prompt", "")
            provider = data.get("provider", "claude")
            model = data.get("model")
            resolution = data.get("resolution", 64)
            quality = data.get("quality", "medium")

            if not prompt:
                await _safe_send(ws, {"type": "error", "message": "Empty prompt"})
                continue

            total_t0 = time.perf_counter()

            # Phase 1: Stream LLM tokens
            if not await _safe_send(ws, {
                "type": "progress", "phase": "llm", "percent": 0,
                "message": "Generating SDF..."
            }):
                continue

            accumulated = ""
            try:
                if provider == "claude":
                    stream_model = model or "haiku"
                    async for token in llm_service.stream_sdf_claude(
                        prompt, stream_model
                    ):
                        accumulated += token
                        if not await _safe_send(ws, {"type": "tokens", "content": token}):
                            break
                elif provider == "gemini":
                    stream_model = model or "flash"
                    async for token in llm_service.stream_sdf_gemini(
                        prompt, stream_model
                    ):
                        accumulated += token
                        if not await _safe_send(ws, {"type": "tokens", "content": token}):
                            break
                else:
                    await _safe_send(ws, {"type": "error", "message": f"Unknown provider: {provider}"})
                    continue
            except Exception as e:
                await _safe_send(ws, {"type": "error", "message": f"LLM error: {e}"})
                continue

            if not await _safe_send(ws, {
                "type": "progress", "phase": "llm", "percent": 100,
                "message": "SDF generation complete"
            }):
                continue

            # Extract JSON from accumulated text
            try:
                sdf_json = llm_service._extract_json(accumulated)
            except Exception as e:
                await _safe_send(ws, {"type": "error", "message": f"JSON extraction error: {e}"})
                continue

            # Validate
            valid, node_count, err = sdf_service.validate_sdf_json(sdf_json)
            if not valid:
                await _safe_send(ws, {"type": "error", "message": f"Invalid SDF: {err}"})
                continue

            if not await _safe_send(ws, {
                "type": "sdf", "json": json.loads(sdf_json), "node_count": node_count
            }):
                continue

            # Auto-select resolution if default
            if resolution is None or resolution == 0:
                resolution = sdf_service.recommend_resolution(sdf_json, quality)

            # Phase 2: Progressive mesh generation
            node = sdf_service.parse_sdf_json(sdf_json)

            # Adaptive preview resolution based on node count
            if node_count is not None and node_count <= 5:
                preview_res = 12
            elif node_count is not None and node_count <= 10:
                preview_res = 16
            else:
                preview_res = 24
            preview_res = min(preview_res, resolution)

            if not await _safe_send(ws, {
                "type": "progress", "phase": "preview", "percent": 0,
                "message": "Generating preview..."
            }):
                continue

            try:
                verts, idxs, _ = sdf_service.generate_mesh(
                    node, (-5, -5, -5), (5, 5, 5), preview_res
                )
                preview_bytes = sdf_service.export_mesh_glb(verts, idxs)
                await _safe_send(ws, {
                    "type": "preview",
                    "glb_base64": base64.b64encode(preview_bytes).decode(),
                })
            except Exception:
                pass  # Preview is optional

            # Final mesh
            if not await _safe_send(ws, {
                "type": "progress", "phase": "mesh", "percent": 0,
                "message": f"Generating final mesh (res={resolution})..."
            }):
                continue

            try:
                verts, idxs, mesh_stats = sdf_service.generate_mesh(
                    node, (-5, -5, -5), (5, 5, 5), resolution
                )

                # Send mesh stats before the mesh data
                if not await _safe_send(ws, {
                    "type": "mesh_stats",
                    "vertices": mesh_stats["vertices"],
                    "triangles": mesh_stats["triangles"],
                    "resolution": mesh_stats["resolution"],
                    "mesh_time_ms": mesh_stats["mesh_time_ms"],
                }):
                    continue

                if not await _safe_send(ws, {
                    "type": "progress", "phase": "export", "percent": 50,
                    "message": "Exporting GLB..."
                }):
                    continue

                mesh_bytes = sdf_service.export_mesh_glb(verts, idxs)
                total_ms = round((time.perf_counter() - total_t0) * 1000, 2)

                await _safe_send(ws, {
                    "type": "mesh",
                    "glb_base64": base64.b64encode(mesh_bytes).decode(),
                    "stats": mesh_stats,
                })
                await _safe_send(ws, {
                    "type": "done",
                    "total_time_ms": total_ms,
                })
            except Exception as e:
                await _safe_send(ws, {"type": "error", "message": f"Mesh error: {e}"})

    except WebSocketDisconnect:
        pass


# ── Main ────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
    )
