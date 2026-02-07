"""Pydantic models for ALICE-SDF Text-to-3D API."""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the 3D scene")
    provider: Literal["claude", "gemini"] = "claude"
    model: Optional[str] = None
    resolution: int = Field(64, ge=8, le=256)
    format: Literal["glb", "obj", "json"] = "glb"


class ValidateRequest(BaseModel):
    sdf_json: str = Field(..., description="SDF JSON string to validate")


class ValidateResponse(BaseModel):
    valid: bool
    node_count: Optional[int] = None
    error: Optional[str] = None


class MeshRequest(BaseModel):
    sdf_json: str = Field(..., description="SDF JSON string")
    resolution: int = Field(64, ge=8, le=256)
    format: Literal["glb", "obj"] = "glb"
    bounds_min: tuple[float, float, float] = (-5.0, -5.0, -5.0)
    bounds_max: tuple[float, float, float] = (5.0, 5.0, 5.0)


class GenerateResponse(BaseModel):
    sdf_json: Optional[str] = None
    node_count: Optional[int] = None
    mesh_vertices: Optional[int] = None
    mesh_triangles: Optional[int] = None
    timings: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    providers: dict = {}
