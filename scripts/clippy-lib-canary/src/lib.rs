//! Each function uses one form that `.github/clippy-lib/clippy.toml` forbids
//! `scripts/clippy_lib_gate.py` runs clippy with that configuration and
//! requires a `disallowed_methods` diagnostic for every one of them

pub fn method_f32(a: f32, b: f32) -> f32 {
    a.max(b)
}

pub fn path_f32(a: f32, b: f32) -> f32 {
    f32::min(a, b)
}

pub fn method_f64(a: f64, b: f64) -> f64 {
    a.min(b)
}

pub fn fold_f64(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

pub fn libm_f32(x: f32) -> f32 {
    x.sin()
}

pub fn simd_max(v: wide::f32x8, w: wide::f32x8) -> wide::f32x8 {
    v.max(w)
}

pub fn simd_min(v: wide::f32x8, w: wide::f32x8) -> wide::f32x8 {
    v.min(w)
}

pub fn clock_instant() -> std::time::Instant {
    std::time::Instant::now()
}

pub fn clock_system() -> std::time::SystemTime {
    std::time::SystemTime::now()
}

pub fn instant_elapsed(i: std::time::Instant) -> std::time::Duration {
    i.elapsed()
}

pub fn system_time_elapsed(
    s: std::time::SystemTime,
) -> Result<std::time::Duration, std::time::SystemTimeError> {
    s.elapsed()
}

pub fn random_state() -> std::collections::hash_map::RandomState {
    std::collections::hash_map::RandomState::new()
}

pub fn temp_dir() -> std::path::PathBuf {
    std::env::temp_dir()
}
