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

// glam's vector max / min / clamp and the lane reductions (each lane through f32::max / f32::min)

pub fn glam_v2_min(v2: glam::Vec2, o: glam::Vec2) -> glam::Vec2 {
    v2.min(o)
}

pub fn glam_v2_max(v2: glam::Vec2, o: glam::Vec2) -> glam::Vec2 {
    v2.max(o)
}

pub fn glam_v2_clamp(v2: glam::Vec2, lo: glam::Vec2, hi: glam::Vec2) -> glam::Vec2 {
    v2.clamp(lo, hi)
}

pub fn glam_v2_min_element(v2: glam::Vec2) -> f32 {
    v2.min_element()
}

pub fn glam_v2_max_element(v2: glam::Vec2) -> f32 {
    v2.max_element()
}

pub fn glam_v3_min(v3: glam::Vec3, o: glam::Vec3) -> glam::Vec3 {
    v3.min(o)
}

pub fn glam_v3_max(v3: glam::Vec3, o: glam::Vec3) -> glam::Vec3 {
    v3.max(o)
}

pub fn glam_v3_clamp(v3: glam::Vec3, lo: glam::Vec3, hi: glam::Vec3) -> glam::Vec3 {
    v3.clamp(lo, hi)
}

pub fn glam_v3_min_element(v3: glam::Vec3) -> f32 {
    v3.min_element()
}

pub fn glam_v3_max_element(v3: glam::Vec3) -> f32 {
    v3.max_element()
}

pub fn glam_v3a_min(v3a: glam::Vec3A, o: glam::Vec3A) -> glam::Vec3A {
    v3a.min(o)
}

pub fn glam_v3a_max(v3a: glam::Vec3A, o: glam::Vec3A) -> glam::Vec3A {
    v3a.max(o)
}

pub fn glam_v3a_clamp(v3a: glam::Vec3A, lo: glam::Vec3A, hi: glam::Vec3A) -> glam::Vec3A {
    v3a.clamp(lo, hi)
}

pub fn glam_v3a_min_element(v3a: glam::Vec3A) -> f32 {
    v3a.min_element()
}

pub fn glam_v3a_max_element(v3a: glam::Vec3A) -> f32 {
    v3a.max_element()
}

pub fn glam_v4_min(v4: glam::Vec4, o: glam::Vec4) -> glam::Vec4 {
    v4.min(o)
}

pub fn glam_v4_max(v4: glam::Vec4, o: glam::Vec4) -> glam::Vec4 {
    v4.max(o)
}

pub fn glam_v4_clamp(v4: glam::Vec4, lo: glam::Vec4, hi: glam::Vec4) -> glam::Vec4 {
    v4.clamp(lo, hi)
}

pub fn glam_v4_min_element(v4: glam::Vec4) -> f32 {
    v4.min_element()
}

pub fn glam_v4_max_element(v4: glam::Vec4) -> f32 {
    v4.max_element()
}

pub fn glam_d2_min(d2: glam::DVec2, o: glam::DVec2) -> glam::DVec2 {
    d2.min(o)
}

pub fn glam_d2_max(d2: glam::DVec2, o: glam::DVec2) -> glam::DVec2 {
    d2.max(o)
}

pub fn glam_d2_clamp(d2: glam::DVec2, lo: glam::DVec2, hi: glam::DVec2) -> glam::DVec2 {
    d2.clamp(lo, hi)
}

pub fn glam_d2_min_element(d2: glam::DVec2) -> f64 {
    d2.min_element()
}

pub fn glam_d2_max_element(d2: glam::DVec2) -> f64 {
    d2.max_element()
}

pub fn glam_d3_min(d3: glam::DVec3, o: glam::DVec3) -> glam::DVec3 {
    d3.min(o)
}

pub fn glam_d3_max(d3: glam::DVec3, o: glam::DVec3) -> glam::DVec3 {
    d3.max(o)
}

pub fn glam_d3_clamp(d3: glam::DVec3, lo: glam::DVec3, hi: glam::DVec3) -> glam::DVec3 {
    d3.clamp(lo, hi)
}

pub fn glam_d3_min_element(d3: glam::DVec3) -> f64 {
    d3.min_element()
}

pub fn glam_d3_max_element(d3: glam::DVec3) -> f64 {
    d3.max_element()
}

pub fn glam_d4_min(d4: glam::DVec4, o: glam::DVec4) -> glam::DVec4 {
    d4.min(o)
}

pub fn glam_d4_max(d4: glam::DVec4, o: glam::DVec4) -> glam::DVec4 {
    d4.max(o)
}

pub fn glam_d4_clamp(d4: glam::DVec4, lo: glam::DVec4, hi: glam::DVec4) -> glam::DVec4 {
    d4.clamp(lo, hi)
}

pub fn glam_d4_min_element(d4: glam::DVec4) -> f64 {
    d4.min_element()
}

pub fn glam_d4_max_element(d4: glam::DVec4) -> f64 {
    d4.max_element()
}

// ALICE-Law G13: the hash collections and their iteration (the method calls are
// found through inference, without the type being written on the line)
pub fn hash_map_type(hm: &std::collections::HashMap<u32, u32>) -> usize {
    hm.len()
}

pub fn hash_set_type(hs: &std::collections::HashSet<u32>) -> usize {
    hs.len()
}

pub fn hash_map_iter_type(it: std::collections::hash_map::Iter<'_, u32, u32>) -> usize {
    it.count()
}

fn made() -> std::collections::HashMap<u32, u32> {
    std::collections::HashMap::new()
}

fn made_set() -> std::collections::HashSet<u32> {
    std::collections::HashSet::new()
}

pub fn hash_map_keys_inferred() -> u32 {
    let m = made();
    m.keys().sum()
}

pub fn hash_map_values_inferred() -> u32 {
    let m = made();
    m.values().sum()
}

pub fn hash_map_iter_mut_inferred() -> usize {
    let mut m = made();
    m.iter_mut().count()
}

pub fn hash_set_iter_inferred() -> u32 {
    let s = made_set();
    s.iter().sum()
}

// ALICE-Law G8: a bare read of a stored form, in each spelling clippy resolves to the
// forbidden path (an alias, a brace import, the value module, bincode's reader API)
pub fn json_from_str(js: &str) -> Option<serde_json::Value> {
    serde_json::from_str(js).ok()
}

pub fn json_alias(js: &str) -> Option<serde_json::Value> {
    use serde_json as sj;
    sj::from_str(js).ok()
}

pub fn json_brace_import(js: &str) -> Option<serde_json::Value> {
    use serde_json::{from_str as read_text, Value};
    read_text::<Value>(js).ok()
}

pub fn json_value_module(jv: serde_json::Value) -> Option<u32> {
    serde_json::value::from_value(jv).ok()
}

pub fn json_from_value(jv: serde_json::Value) -> Option<u32> {
    serde_json::from_value(jv.clone()).ok()
}

pub fn json_from_slice(jb: &[u8]) -> Option<serde_json::Value> {
    serde_json::from_slice(jb).ok()
}

pub fn json_from_reader(jr: std::io::Cursor<Vec<u8>>) -> Option<serde_json::Value> {
    serde_json::from_reader(jr).ok()
}

pub fn json_deserializer(js: &str) -> Option<serde_json::Value> {
    use serde::Deserialize;
    let mut dj = serde_json::Deserializer::from_str(js);
    serde_json::Value::deserialize(&mut dj).ok()
}

pub fn bincode_slice(bb: &[u8]) -> Option<u32> {
    bincode::serde::decode_from_slice(bb, bincode::config::standard()).ok().map(|r| r.0)
}

pub fn bincode_alias(bb: &[u8]) -> Option<u32> {
    use bincode::serde as bs;
    bs::decode_from_slice(bb, bincode::config::legacy()).ok().map(|r| r.0)
}

pub fn bincode_reader(bb: &[u8]) -> Option<u32> {
    let rd = bincode::de::read::SliceReader::new(bb);
    bincode::serde::decode_from_reader(rd, bincode::config::standard()).ok()
}

pub fn bincode_std_read(br: &mut std::io::Cursor<Vec<u8>>) -> Option<u32> {
    bincode::serde::decode_from_std_read(br, bincode::config::standard()).ok()
}

pub fn bincode_native(bn: &[u8]) -> Option<u32> {
    bincode::decode_from_slice(bn, bincode::config::standard()).ok().map(|r| r.0)
}

// the trait method: a serde_json::Value is itself a Deserializer, so a type's own
// `deserialize` reads a stored form without any free function
pub fn trait_from_value(tv: serde_json::Value) -> Option<u32> {
    use serde::Deserialize;
    u32::deserialize(tv).ok()
}

pub fn trait_from_constructed(ts: &str) -> Option<u32> {
    use serde::Deserialize;
    let mut td = serde_json::Deserializer::new(serde_json::de::StrRead::new(ts));
    <u32 as Deserialize>::deserialize(&mut td).ok()
}

pub fn trait_from_parsed(tp: &str) -> Option<u32> {
    use serde::Deserialize;
    let pv: serde_json::Value = tp.parse().ok()?;
    u32::deserialize(&pv).ok()
}

pub fn trait_seed(sv: serde_json::Value) -> Option<u32> {
    use serde::de::DeserializeSeed;
    std::marker::PhantomData::<u32>.deserialize(sv).ok()
}
