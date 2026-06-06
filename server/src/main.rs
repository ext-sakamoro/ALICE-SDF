//! ALICE-SDF REST HTTP server
//!
//! axum ベースの薄い HTTP wrapper。SDF プリミティブ評価 / メッシュ生成 /
//! Splat / Vox / OpenVDB エクスポートを REST API として公開する。
//!
//! 用途: cloud SaaS (alicelaw.net 系) のバックエンド。

use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

#[derive(Serialize)]
struct VersionResponse {
    name: &'static str,
    version: &'static str,
}

async fn handler_version() -> impl IntoResponse {
    Json(VersionResponse {
        name: "alice-sdf-server",
        version: env!("CARGO_PKG_VERSION"),
    })
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

async fn handler_eval(Json(req): Json<EvalRequest>) -> Result<Json<EvalResponse>, (StatusCode, String)> {
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

async fn handler_op(Json(req): Json<OpRequest>) -> Result<Json<EvalResponse>, (StatusCode, String)> {
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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let app = Router::new()
        .route("/", get(handler_version))
        .route("/version", get(handler_version))
        .route("/eval", post(handler_eval))
        .route("/op", post(handler_op))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr = std::env::var("ALICE_SDF_SERVER_ADDR").unwrap_or_else(|_| "0.0.0.0:8787".into());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("bind failed");
    tracing::info!("ALICE-SDF server listening on http://{addr}");
    axum::serve(listener, app).await.expect("serve failed");
}
