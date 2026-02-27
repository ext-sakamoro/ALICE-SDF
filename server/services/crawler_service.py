"""OS3A Crawler Service — fetch GLB assets and convert to .asdf.json via ALICE-SDF."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

import alice_sdf

logger = logging.getLogger(__name__)

OS3A_DATA_BASE = (
    "https://raw.githubusercontent.com/ToxSam/open-source-3D-assets/main/data"
)
PROJECTS_URL = f"{OS3A_DATA_BASE}/projects.json"

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "asset" / "os3a-sdf"


@dataclass
class AssetResult:
    id: str
    name: str
    collection: str
    license: str
    glb_url: str
    glb_size: int
    sdf_json_path: str
    sdf_size: int
    node_count: int
    convert_ms: float
    error: Optional[str] = None


@dataclass
class CrawlerState:
    running: bool = False
    total_assets: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    current_asset: str = ""
    results: list[AssetResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_glb_bytes: int = 0
    total_sdf_bytes: int = 0


_state = CrawlerState()


def get_state() -> dict:
    elapsed = 0.0
    if _state.start_time > 0:
        end = _state.end_time if _state.end_time > 0 else time.time()
        elapsed = round(end - _state.start_time, 2)

    return {
        "running": _state.running,
        "total_assets": _state.total_assets,
        "processed": _state.processed,
        "succeeded": _state.succeeded,
        "failed": _state.failed,
        "current_asset": _state.current_asset,
        "elapsed_seconds": elapsed,
        "total_glb_bytes": _state.total_glb_bytes,
        "total_sdf_bytes": _state.total_sdf_bytes,
        "compression_ratio": (
            round(_state.total_glb_bytes / max(_state.total_sdf_bytes, 1), 1)
            if _state.total_sdf_bytes > 0
            else 0
        ),
    }


async def _fetch_json(client: httpx.AsyncClient, url: str) -> list | dict:
    resp = await client.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


async def _download_glb(client: httpx.AsyncClient, url: str) -> bytes:
    resp = await client.get(url, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _convert_glb_to_asdf(
    glb_bytes: bytes, asset_meta: dict, collection_meta: dict
) -> tuple[str, int]:
    """Convert GLB bytes to .asdf.json string. Returns (json_str, node_count)."""
    vertices, indices = alice_sdf.import_glb_bytes(glb_bytes)
    node = alice_sdf.mesh_to_sdf(vertices, indices)
    sdf_json_str = alice_sdf.to_json(node)

    sdf_data = json.loads(sdf_json_str)

    metadata = {
        "name": asset_meta.get("name", ""),
        "description": asset_meta.get("description", ""),
        "author": collection_meta.get("creator_id", ""),
        "license": collection_meta.get("license", "CC0"),
        "source": "opensource3dassets.com",
        "original_format": "GLB",
        "original_size_bytes": asset_meta.get("metadata", {}).get("file_size", 0),
        "collection": collection_meta.get("name", ""),
        "os3a_id": asset_meta.get("id", ""),
    }
    attributes = asset_meta.get("metadata", {}).get("attributes", [])
    if attributes:
        metadata["attributes"] = {
            a["trait_type"]: a["value"] for a in attributes
        }

    output = {
        "version": alice_sdf.version(),
        "root": sdf_data.get("root", sdf_data),
        "metadata": metadata,
    }

    output_str = json.dumps(output, indent=2, ensure_ascii=False)
    node_count = node.node_count()
    return output_str, node_count


async def _process_asset(
    client: httpx.AsyncClient,
    asset: dict,
    collection: dict,
    output_dir: Path,
) -> AssetResult:
    asset_id = asset.get("id", "unknown")
    name = asset.get("name", "unknown")
    glb_url = asset.get("model_file_url", "")
    collection_name = collection.get("id", "unknown")
    license_type = collection.get("license", "CC0")

    _state.current_asset = f"{collection_name}/{name}"

    result = AssetResult(
        id=asset_id,
        name=name,
        collection=collection_name,
        license=license_type,
        glb_url=glb_url,
        glb_size=0,
        sdf_json_path="",
        sdf_size=0,
        node_count=0,
        convert_ms=0.0,
    )

    try:
        glb_bytes = await _download_glb(client, glb_url)
        result.glb_size = len(glb_bytes)

        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        sdf_json_str, node_count = await loop.run_in_executor(
            None, _convert_glb_to_asdf, glb_bytes, asset, collection
        )
        result.convert_ms = round((time.perf_counter() - t0) * 1000, 2)
        result.node_count = node_count

        col_dir = output_dir / collection_name
        col_dir.mkdir(parents=True, exist_ok=True)
        out_path = col_dir / f"{name}.asdf.json"
        out_path.write_text(sdf_json_str, encoding="utf-8")

        result.sdf_json_path = str(out_path.relative_to(output_dir))
        result.sdf_size = len(sdf_json_str.encode("utf-8"))

        logger.info(
            "OK %s/%s  GLB=%dB → SDF=%dB (%dx) nodes=%d  %.0fms",
            collection_name, name,
            result.glb_size, result.sdf_size,
            result.glb_size // max(result.sdf_size, 1),
            node_count, result.convert_ms,
        )

    except Exception as e:
        result.error = str(e)
        logger.warning("FAIL %s/%s: %s", collection_name, name, e)

    return result


def _build_index(results: list[AssetResult], output_dir: Path) -> None:
    """Generate index.json with all asset metadata and compression stats."""
    succeeded = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    total_glb = sum(r.glb_size for r in succeeded)
    total_sdf = sum(r.sdf_size for r in succeeded)

    index = {
        "generator": "ALICE-SDF OS3A Crawler",
        "version": alice_sdf.version(),
        "source": "https://www.opensource3dassets.com",
        "stats": {
            "total_assets": len(results),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "total_glb_bytes": total_glb,
            "total_sdf_bytes": total_sdf,
            "compression_ratio": round(total_glb / max(total_sdf, 1), 1),
        },
        "assets": [
            {
                "id": r.id,
                "name": r.name,
                "collection": r.collection,
                "license": r.license,
                "file": r.sdf_json_path,
                "sdf_size": r.sdf_size,
                "original_glb_size": r.glb_size,
                "node_count": r.node_count,
                "convert_ms": r.convert_ms,
            }
            for r in succeeded
        ],
    }

    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Index written: %s (%d assets)", index_path, len(succeeded))


async def run_crawler(output_dir: Optional[Path] = None) -> dict:
    """Run the full OS3A → SDF conversion pipeline."""
    global _state

    if _state.running:
        return {"error": "Crawler is already running"}

    _state = CrawlerState(running=True, start_time=time.time())
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient() as client:
            projects = await _fetch_json(client, PROJECTS_URL)
            logger.info("Fetched %d collections from OS3A", len(projects))

            all_assets: list[tuple[dict, dict]] = []
            for project in projects:
                asset_file = project.get("asset_data_file", "")
                if not asset_file:
                    continue
                asset_url = f"{OS3A_DATA_BASE}/{asset_file}"
                try:
                    assets = await _fetch_json(client, asset_url)
                    for asset in assets:
                        if asset.get("model_file_url", "").endswith(".glb"):
                            all_assets.append((asset, project))
                except Exception as e:
                    logger.warning("Skip collection %s: %s", project.get("id"), e)

            _state.total_assets = len(all_assets)
            logger.info("Found %d GLB assets to convert", len(all_assets))

            for asset, project in all_assets:
                result = await _process_asset(client, asset, project, out)
                _state.results.append(result)
                _state.processed += 1
                if result.error:
                    _state.failed += 1
                else:
                    _state.succeeded += 1
                    _state.total_glb_bytes += result.glb_size
                    _state.total_sdf_bytes += result.sdf_size

        _build_index(_state.results, out)

    except Exception as e:
        logger.exception("Crawler fatal error: %s", e)
        return {"error": str(e)}
    finally:
        _state.running = False
        _state.end_time = time.time()

    return get_state()


def get_converted_assets() -> list[dict]:
    """Return list of successfully converted assets."""
    out = OUTPUT_DIR
    index_path = out / "index.json"
    if not index_path.exists():
        return []
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return data.get("assets", [])


def get_asset_sdf(asset_id: str) -> Optional[str]:
    """Return .asdf.json content for a specific asset."""
    assets = get_converted_assets()
    for a in assets:
        if a["id"] == asset_id:
            path = OUTPUT_DIR / a["file"]
            if path.exists():
                return path.read_text(encoding="utf-8")
    return None
