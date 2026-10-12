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
