//! ALICE-SDF REST HTTP server
//!
//! axum ベースの薄い HTTP wrapper。SDF プリミティブ評価 / メッシュ生成 /
//! Splat / Vox / OpenVDB エクスポートを REST API として公開する。
//!
//! 用途: cloud SaaS (alicelaw.net 系) のバックエンド。

use alice_sdf::io::splat::{sdf_to_splats, Splat, SplatConfig};
use alice_sdf::io::vox::{sdf_to_vox, VoxConfig};
use alice_sdf::mesh::MarchingCubesConfig;
use alice_sdf::types::SdfNode;
use axum::{
    extract::{Json, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

/// Resolution / size の許容範囲
const MESH_RESOLUTION_MIN: usize = 8;
const MESH_RESOLUTION_MAX: usize = 192;
const SPLAT_RESOLUTION_MIN: u32 = 8;
const SPLAT_RESOLUTION_MAX: u32 = 96;
const VOX_SIZE_MIN: u32 = 4;
const VOX_SIZE_MAX: u32 = 96;

/// JSON body 上限 1 MiB
const BODY_LIMIT_BYTES: usize = 1 * 1024 * 1024;

/// アプリ全体で共有する設定
#[derive(Clone)]
struct AppState {
    /// `ALICE_SDF_TOKEN` env が空でない場合に bearer 必須化
    auth_token: Option<Arc<String>>,
}

#[derive(Serialize)]
struct VersionResponse {
    name: &'static str,
    version: &'static str,
    endpoints: &'static [&'static str],
}

async fn handler_version() -> impl IntoResponse {
    Json(VersionResponse {
        name: "alice-sdf-server",
        version: env!("CARGO_PKG_VERSION"),
        endpoints: &[
            "GET  /",
            "GET  /version",
            "POST /eval",
            "POST /op",
            "POST /mesh",
            "POST /splat",
            "POST /vox",
        ],
    })
}

#[derive(Deserialize)]
struct ShapeSpec {
    /// "sphere" | "box" | "torus" | "cylinder"
    shape: String,
    /// shape-specific parameters
    params: serde_json::Value,
}

impl ShapeSpec {
    fn to_node(&self) -> Result<SdfNode, (StatusCode, String)> {
        match self.shape.as_str() {
            "sphere" => {
                let radius = self
                    .params
                    .get("radius")
                    .and_then(|v| v.as_f64())
                    .ok_or((StatusCode::BAD_REQUEST, "missing radius".into()))?
                    as f32;
                Ok(SdfNode::sphere(radius))
            }
            "box" => {
                let he = parse_vec3(&self.params, "half_extents")
                    .ok_or((StatusCode::BAD_REQUEST, "missing half_extents".into()))?;
                Ok(SdfNode::box3d(he.x * 2.0, he.y * 2.0, he.z * 2.0))
            }
            "torus" => {
                let major = self
                    .params
                    .get("major_radius")
                    .and_then(|v| v.as_f64())
                    .ok_or((StatusCode::BAD_REQUEST, "missing major_radius".into()))?
                    as f32;
                let minor = self
                    .params
                    .get("minor_radius")
                    .and_then(|v| v.as_f64())
                    .ok_or((StatusCode::BAD_REQUEST, "missing minor_radius".into()))?
                    as f32;
                Ok(SdfNode::torus(major, minor))
            }
            "cylinder" => {
                let radius = self
                    .params
                    .get("radius")
                    .and_then(|v| v.as_f64())
                    .ok_or((StatusCode::BAD_REQUEST, "missing radius".into()))?
                    as f32;
                let half_height = self
                    .params
                    .get("half_height")
                    .and_then(|v| v.as_f64())
                    .ok_or((StatusCode::BAD_REQUEST, "missing half_height".into()))?
                    as f32;
                Ok(SdfNode::cylinder(radius, half_height * 2.0))
            }
            other => Err((StatusCode::BAD_REQUEST, format!("unknown shape: {other}"))),
        }
    }
}

#[derive(Deserialize)]
struct EvalRequest {
    /// "sphere" | "box" | "torus" | "cylinder"
    shape: String,
    point: [f32; 3],
    /// shape-specific parameters
    params: serde_json::Value,
}

#[derive(Serialize)]
struct EvalResponse {
    distance: f32,
}

async fn handler_eval(
    Json(req): Json<EvalRequest>,
) -> Result<Json<EvalResponse>, (StatusCode, String)> {
    let p = glam::Vec3::new(req.point[0], req.point[1], req.point[2]);
    let d = match req.shape.as_str() {
        "sphere" => {
            let radius = req
                .params
                .get("radius")
                .and_then(|v| v.as_f64())
                .ok_or((StatusCode::BAD_REQUEST, "missing radius".into()))?
                as f32;
            let c = parse_vec3(&req.params, "center").unwrap_or(glam::Vec3::ZERO);
            alice_sdf::primitives::sdf_sphere_at(p, c, radius)
        }
        "box" => {
            let he = parse_vec3(&req.params, "half_extents")
                .ok_or((StatusCode::BAD_REQUEST, "missing half_extents".into()))?;
            let c = parse_vec3(&req.params, "center").unwrap_or(glam::Vec3::ZERO);
            alice_sdf::primitives::sdf_box3d_at(p, c, he)
        }
        "torus" => {
            let major = req
                .params
                .get("major_radius")
                .and_then(|v| v.as_f64())
                .ok_or((StatusCode::BAD_REQUEST, "missing major_radius".into()))?
                as f32;
            let minor = req
                .params
                .get("minor_radius")
                .and_then(|v| v.as_f64())
                .ok_or((StatusCode::BAD_REQUEST, "missing minor_radius".into()))?
                as f32;
            alice_sdf::primitives::sdf_torus(p, major, minor)
        }
        "cylinder" => {
            let radius = req
                .params
                .get("radius")
                .and_then(|v| v.as_f64())
                .ok_or((StatusCode::BAD_REQUEST, "missing radius".into()))?
                as f32;
            let half_height = req
                .params
                .get("half_height")
                .and_then(|v| v.as_f64())
                .ok_or((StatusCode::BAD_REQUEST, "missing half_height".into()))?
                as f32;
            alice_sdf::primitives::sdf_cylinder(p, radius, half_height)
        }
        other => return Err((StatusCode::BAD_REQUEST, format!("unknown shape: {other}"))),
    };
    Ok(Json(EvalResponse { distance: d }))
}

fn parse_vec3(v: &serde_json::Value, key: &str) -> Option<glam::Vec3> {
    let arr = v.get(key)?.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    Some(glam::Vec3::new(
        arr[0].as_f64()? as f32,
        arr[1].as_f64()? as f32,
        arr[2].as_f64()? as f32,
    ))
}

#[derive(Deserialize)]
struct OpRequest {
    op: String,
    a: f32,
    b: f32,
    #[serde(default)]
    k: f32,
}

async fn handler_op(
    Json(req): Json<OpRequest>,
) -> Result<Json<EvalResponse>, (StatusCode, String)> {
    use alice_sdf::operations::*;
    let d = match req.op.as_str() {
        "union" => sdf_union(req.a, req.b),
        "intersection" => sdf_intersection(req.a, req.b),
        "subtraction" => sdf_subtraction(req.a, req.b),
        "smooth_union" => sdf_smooth_union(req.a, req.b, req.k),
        "smooth_intersection" => sdf_smooth_intersection(req.a, req.b, req.k),
        "smooth_subtraction" => sdf_smooth_subtraction(req.a, req.b, req.k),
        other => return Err((StatusCode::BAD_REQUEST, format!("unknown op: {other}"))),
    };
    Ok(Json(EvalResponse { distance: d }))
}

// ── /mesh ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct MeshRequest {
    shape: String,
    params: serde_json::Value,
    #[serde(default)]
    bounds: Option<[[f32; 3]; 2]>,
    #[serde(default)]
    resolution: Option<usize>,
}

#[derive(Serialize)]
struct MeshResponse {
    vertex_count: usize,
    triangle_count: usize,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

async fn handler_mesh(
    Json(req): Json<MeshRequest>,
) -> Result<Json<MeshResponse>, (StatusCode, String)> {
    let spec = ShapeSpec {
        shape: req.shape,
        params: req.params,
    };
    let node = spec.to_node()?;

    let (lo, hi) = if let Some([a, b]) = req.bounds {
        (Vec3::from_array(a), Vec3::from_array(b))
    } else {
        (Vec3::splat(-2.0), Vec3::splat(2.0))
    };
    let res = req.resolution.unwrap_or(64);
    if !(MESH_RESOLUTION_MIN..=MESH_RESOLUTION_MAX).contains(&res) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "resolution {res} out of range [{}, {}]",
                MESH_RESOLUTION_MIN, MESH_RESOLUTION_MAX
            ),
        ));
    }
    let cfg = MarchingCubesConfig {
        resolution: res,
        iso_level: 0.0,
        compute_normals: true,
        compute_uvs: false,
        uv_scale: 1.0,
        compute_tangents: false,
        compute_materials: false,
    };

    let mesh = alice_sdf::mesh::sdf_to_mesh(&node, lo, hi, &cfg);
    let positions: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|v| v.position.to_array())
        .collect();
    let normals: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.normal.to_array()).collect();
    let triangle_count = mesh.indices.len() / 3;
    Ok(Json(MeshResponse {
        vertex_count: mesh.vertices.len(),
        triangle_count,
        positions,
        normals,
        indices: mesh.indices,
    }))
}

// ── /splat ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SplatRequest {
    shape: String,
    params: serde_json::Value,
    #[serde(default)]
    bounds: Option<(f32, f32)>,
    #[serde(default)]
    resolution: Option<u32>,
    #[serde(default)]
    base_color: Option<[u8; 4]>,
    #[serde(default)]
    format: Option<String>,
}

#[derive(Serialize)]
struct SplatResponse {
    splat_count: usize,
    /// JSON-encoded splat list (only when format != "bytes")
    splats: Option<Vec<SplatJson>>,
    /// base64 of raw 32-byte splat stream (when format == "bytes")
    bytes_base64: Option<String>,
}

#[derive(Serialize)]
struct SplatJson {
    position: [f32; 3],
    scale: [f32; 3],
    color: [u8; 4],
    rotation: [u8; 4],
}

fn b64_encode(bytes: &[u8]) -> String {
    const CS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n =
            (u32::from(bytes[i]) << 16) | (u32::from(bytes[i + 1]) << 8) | u32::from(bytes[i + 2]);
        out.push(CS[((n >> 18) & 0x3F) as usize] as char);
        out.push(CS[((n >> 12) & 0x3F) as usize] as char);
        out.push(CS[((n >> 6) & 0x3F) as usize] as char);
        out.push(CS[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = u32::from(bytes[i]) << 16;
        out.push(CS[((n >> 18) & 0x3F) as usize] as char);
        out.push(CS[((n >> 12) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = (u32::from(bytes[i]) << 16) | (u32::from(bytes[i + 1]) << 8);
        out.push(CS[((n >> 18) & 0x3F) as usize] as char);
        out.push(CS[((n >> 12) & 0x3F) as usize] as char);
        out.push(CS[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

async fn handler_splat(
    Json(req): Json<SplatRequest>,
) -> Result<Json<SplatResponse>, (StatusCode, String)> {
    let spec = ShapeSpec {
        shape: req.shape,
        params: req.params,
    };
    let node = spec.to_node()?;

    let res = req.resolution.unwrap_or(32);
    if !(SPLAT_RESOLUTION_MIN..=SPLAT_RESOLUTION_MAX).contains(&res) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "resolution {res} out of range [{}, {}]",
                SPLAT_RESOLUTION_MIN, SPLAT_RESOLUTION_MAX
            ),
        ));
    }
    let cfg = SplatConfig {
        bounds: req.bounds.unwrap_or((-2.0, 2.0)),
        resolution: res,
        base_color: req.base_color.unwrap_or([255, 255, 255, 255]),
    };
    let splats = sdf_to_splats(&node, &cfg);

    if req.format.as_deref() == Some("bytes") {
        let mut bytes = Vec::with_capacity(splats.len() * 32);
        for s in &splats {
            bytes.extend_from_slice(&s.to_bytes());
        }
        Ok(Json(SplatResponse {
            splat_count: splats.len(),
            splats: None,
            bytes_base64: Some(b64_encode(&bytes)),
        }))
    } else {
        let arr: Vec<SplatJson> = splats
            .iter()
            .map(|s: &Splat| SplatJson {
                position: s.position,
                scale: s.scale,
                color: s.color,
                rotation: s.rotation,
            })
            .collect();
        Ok(Json(SplatResponse {
            splat_count: splats.len(),
            splats: Some(arr),
            bytes_base64: None,
        }))
    }
}

// ── /vox ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct VoxRequest {
    shape: String,
    params: serde_json::Value,
    #[serde(default)]
    size: Option<u32>,
    #[serde(default)]
    bounds: Option<(f32, f32)>,
    #[serde(default)]
    color_index: Option<u8>,
}

#[derive(Serialize)]
struct VoxResponse {
    size: [u32; 3],
    voxel_count: usize,
    voxels: Vec<[u32; 4]>,
}

async fn handler_vox(
    Json(req): Json<VoxRequest>,
) -> Result<Json<VoxResponse>, (StatusCode, String)> {
    let spec = ShapeSpec {
        shape: req.shape,
        params: req.params,
    };
    let node = spec.to_node()?;
    let size = req.size.unwrap_or(32);
    if !(VOX_SIZE_MIN..=VOX_SIZE_MAX).contains(&size) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "size {size} out of range [{}, {}]",
                VOX_SIZE_MIN, VOX_SIZE_MAX
            ),
        ));
    }
    let cfg = VoxConfig {
        size,
        bounds: req.bounds.unwrap_or((-1.5, 1.5)),
        color_index: req.color_index.unwrap_or(1).max(1),
    };
    let model = sdf_to_vox(&node, &cfg);
    let voxels: Vec<[u32; 4]> = model
        .voxels
        .iter()
        .map(|v| {
            [
                u32::from(v.x),
                u32::from(v.y),
                u32::from(v.z),
                u32::from(v.color),
            ]
        })
        .collect();
    Ok(Json(VoxResponse {
        size: [model.size.0, model.size.1, model.size.2],
        voxel_count: voxels.len(),
        voxels,
    }))
}

/// Bearer token middleware
///
/// `state.auth_token` が `Some(_)` の時のみ `Authorization: Bearer <token>` を要求する。
/// `None` (= env 未設定) なら no-op。`GET /` と `GET /version` は常に通す。
async fn auth_middleware(
    State(state): State<AppState>,
    headers: HeaderMap,
    req: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.auth_token.as_deref() else {
        return next.run(req).await;
    };
    let path = req.uri().path();
    if path == "/" || path == "/version" {
        return next.run(req).await;
    }
    let presented = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));
    if presented == Some(expected.as_str()) {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, "missing or invalid Bearer token").into_response()
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let state = AppState {
        auth_token: std::env::var("ALICE_SDF_TOKEN")
            .ok()
            .filter(|s| !s.is_empty())
            .map(Arc::new),
    };
    let auth_enabled = state.auth_token.is_some();

    // Rate limit: per-IP, 60 burst / 1 req/sec sustain (env で上書き可能)
    let per_sec: u64 = std::env::var("ALICE_SDF_RPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let burst: u32 = std::env::var("ALICE_SDF_BURST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let governor_cfg = Arc::new(
        GovernorConfigBuilder::default()
            .per_second(per_sec)
            .burst_size(burst)
            .finish()
            .expect("governor config"),
    );

    let app = Router::new()
        .route("/", get(handler_version))
        .route("/version", get(handler_version))
        .route("/eval", post(handler_eval))
        .route("/op", post(handler_op))
        .route("/mesh", post(handler_mesh))
        .route("/splat", post(handler_splat))
        .route("/vox", post(handler_vox))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state)
        .layer(RequestBodyLimitLayer::new(BODY_LIMIT_BYTES))
        .layer(GovernorLayer {
            config: governor_cfg,
        })
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr = std::env::var("ALICE_SDF_SERVER_ADDR").unwrap_or_else(|_| "0.0.0.0:8787".into());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("bind failed");
    tracing::info!(
        "ALICE-SDF server listening on http://{addr} (auth={}, rate={per_sec}/s burst={burst}, body_limit={}MB)",
        if auth_enabled { "enabled" } else { "disabled" },
        BODY_LIMIT_BYTES / (1024 * 1024)
    );
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await
    .expect("serve failed");
}

// ── tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_sphere_node() {
        let s = ShapeSpec {
            shape: "sphere".into(),
            params: serde_json::json!({"radius": 1.0}),
        };
        assert!(s.to_node().is_ok());
    }

    #[test]
    fn shape_box_requires_half_extents() {
        let s = ShapeSpec {
            shape: "box".into(),
            params: serde_json::json!({}),
        };
        assert!(s.to_node().is_err());
    }

    #[test]
    fn shape_unknown_rejected() {
        let s = ShapeSpec {
            shape: "nope".into(),
            params: serde_json::json!({}),
        };
        assert!(s.to_node().is_err());
    }

    #[test]
    fn base64_roundtrip_known_vector() {
        // "ALICE-SDF" -> "QUxJQ0UtU0RG"
        let s = b64_encode(b"ALICE-SDF");
        assert_eq!(s, "QUxJQ0UtU0RG");
    }
}
