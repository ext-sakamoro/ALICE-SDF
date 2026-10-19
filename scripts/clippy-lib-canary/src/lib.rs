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

pub fn simd_f32x8_mul_add(f32x8_mul_add_a: wide::f32x8, b: wide::f32x8, c: wide::f32x8) -> wide::f32x8 {
    f32x8_mul_add_a.mul_add(b, c)
}

pub fn simd_f32x8_mul_sub(f32x8_mul_sub_a: wide::f32x8, b: wide::f32x8, c: wide::f32x8) -> wide::f32x8 {
    f32x8_mul_sub_a.mul_sub(b, c)
}

pub fn simd_f32x8_mul_neg_add(f32x8_mul_neg_add_a: wide::f32x8, b: wide::f32x8, c: wide::f32x8) -> wide::f32x8 {
    f32x8_mul_neg_add_a.mul_neg_add(b, c)
}

pub fn simd_f32x8_mul_neg_sub(f32x8_mul_neg_sub_a: wide::f32x8, b: wide::f32x8, c: wide::f32x8) -> wide::f32x8 {
    f32x8_mul_neg_sub_a.mul_neg_sub(b, c)
}

pub fn simd_f32x4_mul_add(f32x4_mul_add_a: wide::f32x4, b: wide::f32x4, c: wide::f32x4) -> wide::f32x4 {
    f32x4_mul_add_a.mul_add(b, c)
}

pub fn simd_f32x4_mul_sub(f32x4_mul_sub_a: wide::f32x4, b: wide::f32x4, c: wide::f32x4) -> wide::f32x4 {
    f32x4_mul_sub_a.mul_sub(b, c)
}

pub fn simd_f32x4_mul_neg_add(f32x4_mul_neg_add_a: wide::f32x4, b: wide::f32x4, c: wide::f32x4) -> wide::f32x4 {
    f32x4_mul_neg_add_a.mul_neg_add(b, c)
}

pub fn simd_f32x4_mul_neg_sub(f32x4_mul_neg_sub_a: wide::f32x4, b: wide::f32x4, c: wide::f32x4) -> wide::f32x4 {
    f32x4_mul_neg_sub_a.mul_neg_sub(b, c)
}

pub fn simd_f64x4_mul_add(f64x4_mul_add_a: wide::f64x4, b: wide::f64x4, c: wide::f64x4) -> wide::f64x4 {
    f64x4_mul_add_a.mul_add(b, c)
}

pub fn simd_f64x4_mul_sub(f64x4_mul_sub_a: wide::f64x4, b: wide::f64x4, c: wide::f64x4) -> wide::f64x4 {
    f64x4_mul_sub_a.mul_sub(b, c)
}

pub fn simd_f64x4_mul_neg_add(f64x4_mul_neg_add_a: wide::f64x4, b: wide::f64x4, c: wide::f64x4) -> wide::f64x4 {
    f64x4_mul_neg_add_a.mul_neg_add(b, c)
}

pub fn simd_f64x4_mul_neg_sub(f64x4_mul_neg_sub_a: wide::f64x4, b: wide::f64x4, c: wide::f64x4) -> wide::f64x4 {
    f64x4_mul_neg_sub_a.mul_neg_sub(b, c)
}

pub fn simd_f64x2_mul_add(f64x2_mul_add_a: wide::f64x2, b: wide::f64x2, c: wide::f64x2) -> wide::f64x2 {
    f64x2_mul_add_a.mul_add(b, c)
}

pub fn simd_f64x2_mul_sub(f64x2_mul_sub_a: wide::f64x2, b: wide::f64x2, c: wide::f64x2) -> wide::f64x2 {
    f64x2_mul_sub_a.mul_sub(b, c)
}

pub fn simd_f64x2_mul_neg_add(f64x2_mul_neg_add_a: wide::f64x2, b: wide::f64x2, c: wide::f64x2) -> wide::f64x2 {
    f64x2_mul_neg_add_a.mul_neg_add(b, c)
}

pub fn simd_f64x2_mul_neg_sub(f64x2_mul_neg_sub_a: wide::f64x2, b: wide::f64x2, c: wide::f64x2) -> wide::f64x2 {
    f64x2_mul_neg_sub_a.mul_neg_sub(b, c)
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

// the rest of the library-only entries, so every entry has a form (a mistyped or removed entry
// forbids nothing, and clippy does not say so for a method path)

pub fn hash_map_iter_inferred() -> usize {
    let mi = made();
    mi.iter().count()
}

pub fn hash_map_values_mut_inferred() -> usize {
    let mut mv = made();
    mv.values_mut().count()
}

pub fn hash_map_into_keys_inferred() -> u32 {
    let mk = made();
    mk.into_keys().sum()
}

pub fn hash_map_into_values_inferred() -> u32 {
    let mu = made();
    mu.into_values().sum()
}

pub fn hash_map_drain_inferred() -> usize {
    let mut md = made();
    md.drain().count()
}

pub fn hash_map_retain_inferred() -> usize {
    let mut mr = made();
    mr.retain(|_, v| *v > 0);
    mr.len()
}

pub fn hash_set_drain_inferred() -> usize {
    let mut sd = made_set();
    sd.drain().count()
}

pub fn hash_set_retain_inferred() -> usize {
    let mut sr = made_set();
    sr.retain(|v| *v > 0);
    sr.len()
}

pub fn hash_map_iter_mut_type(it: std::collections::hash_map::IterMut<'_, u32, u32>) -> usize {
    it.count()
}

pub fn hash_map_keys_type(it: std::collections::hash_map::Keys<'_, u32, u32>) -> usize {
    it.count()
}

pub fn hash_map_values_type(it: std::collections::hash_map::Values<'_, u32, u32>) -> usize {
    it.count()
}

pub fn hash_map_values_mut_type(it: std::collections::hash_map::ValuesMut<'_, u32, u32>) -> usize {
    it.count()
}

pub fn hash_map_into_iter_type(it: std::collections::hash_map::IntoIter<u32, u32>) -> usize {
    it.count()
}

pub fn hash_map_drain_type(it: std::collections::hash_map::Drain<'_, u32, u32>) -> usize {
    it.count()
}

pub fn hash_set_iter_type(it: std::collections::hash_set::Iter<'_, u32>) -> usize {
    it.count()
}

pub fn hash_set_into_iter_type(it: std::collections::hash_set::IntoIter<u32>) -> usize {
    it.count()
}

pub fn hash_set_drain_type(it: std::collections::hash_set::Drain<'_, u32>) -> usize {
    it.count()
}

// the rest of the stored-form entries, one function each (the deserializers are built without
// calling deserialize, so each line carries the one diagnostic its form names)

pub fn json_deserializer_from_slice(ds: &[u8]) -> () {
    let _dsl = serde_json::Deserializer::from_slice(ds);
}

pub fn json_deserializer_from_reader(dr: std::io::Cursor<Vec<u8>>) -> () {
    let _drd = serde_json::Deserializer::from_reader(dr);
}

pub fn json_stream_deserializer(st: &str) -> () {
    let _sd = serde_json::StreamDeserializer::<_, serde_json::Value>::new(serde_json::de::StrRead::new(st));
}

pub fn bincode_serde_borrow(bs2: &[u8]) -> Option<usize> {
    bincode::serde::borrow_decode_from_slice::<&str, _>(bs2, bincode::config::standard()).ok().map(|r| r.1)
}

pub fn bincode_native_reader(br2: &[u8]) -> Option<u32> {
    bincode::decode_from_reader::<u32, _, _>(bincode::de::read::SliceReader::new(br2), bincode::config::standard()).ok()
}

pub fn bincode_native_std_read(bsr: &mut std::io::Cursor<Vec<u8>>) -> Option<u32> {
    bincode::decode_from_std_read::<u32, _, _>(bsr, bincode::config::standard()).ok()
}

pub fn bincode_native_borrow(bb2: &[u8]) -> Option<usize> {
    bincode::borrow_decode_from_slice::<&str, _>(bb2, bincode::config::standard()).ok().map(|r| r.1)
}

pub fn jit_fma<'f, B: cranelift_codegen::ir::InstBuilder<'f>>(
    ins: B,
    a_fma: cranelift_codegen::ir::Value,
    b: cranelift_codegen::ir::Value,
    c: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    ins.fma(a_fma, b, c)
}

pub fn glam_vec2_mul_add(glam_vec2_a: glam::Vec2, b: glam::Vec2, c: glam::Vec2) -> glam::Vec2 {
    glam_vec2_a.mul_add(b, c)
}

pub fn glam_vec3_mul_add(glam_vec3_a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> glam::Vec3 {
    glam_vec3_a.mul_add(b, c)
}

pub fn glam_vec3a_mul_add(glam_vec3a_a: glam::Vec3A, b: glam::Vec3A, c: glam::Vec3A) -> glam::Vec3A {
    glam_vec3a_a.mul_add(b, c)
}

pub fn glam_vec4_mul_add(glam_vec4_a: glam::Vec4, b: glam::Vec4, c: glam::Vec4) -> glam::Vec4 {
    glam_vec4_a.mul_add(b, c)
}

pub fn glam_dvec2_mul_add(glam_dvec2_a: glam::DVec2, b: glam::DVec2, c: glam::DVec2) -> glam::DVec2 {
    glam_dvec2_a.mul_add(b, c)
}

pub fn glam_dvec3_mul_add(glam_dvec3_a: glam::DVec3, b: glam::DVec3, c: glam::DVec3) -> glam::DVec3 {
    glam_dvec3_a.mul_add(b, c)
}

pub fn glam_dvec4_mul_add(glam_dvec4_a: glam::DVec4, b: glam::DVec4, c: glam::DVec4) -> glam::DVec4 {
    glam_dvec4_a.mul_add(b, c)
}

pub fn wide_f32x8_sin(f32x8_sin_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_sin_arg.sin()
}

pub fn wide_f32x8_cos(f32x8_cos_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_cos_arg.cos()
}

pub fn wide_f32x8_tan(f32x8_tan_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_tan_arg.tan()
}

pub fn wide_f32x8_sin_cos(f32x8_sin_cos_arg: wide::f32x8) -> (wide::f32x8, wide::f32x8) {
    f32x8_sin_cos_arg.sin_cos()
}

pub fn wide_f32x8_asin(f32x8_asin_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_asin_arg.asin()
}

pub fn wide_f32x8_acos(f32x8_acos_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_acos_arg.acos()
}

pub fn wide_f32x8_asin_acos(f32x8_asin_acos_arg: wide::f32x8) -> (wide::f32x8, wide::f32x8) {
    f32x8_asin_acos_arg.asin_acos()
}

pub fn wide_f32x8_atan(f32x8_atan_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_atan_arg.atan()
}

pub fn wide_f32x8_atan2(f32x8_atan2_arg: wide::f32x8, y: wide::f32x8) -> wide::f32x8 {
    f32x8_atan2_arg.atan2(y)
}

pub fn wide_f32x8_exp(f32x8_exp_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_exp_arg.exp()
}

pub fn wide_f32x8_ln(f32x8_ln_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_ln_arg.ln()
}

pub fn wide_f32x8_log2(f32x8_log2_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_log2_arg.log2()
}

pub fn wide_f32x8_log10(f32x8_log10_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_log10_arg.log10()
}

pub fn wide_f32x8_powf(f32x8_powf_arg: wide::f32x8, y: f32) -> wide::f32x8 {
    f32x8_powf_arg.powf(y)
}

pub fn wide_f32x8_pow_f32x8(f32x8_pow_f32x8_arg: wide::f32x8, y: wide::f32x8) -> wide::f32x8 {
    f32x8_pow_f32x8_arg.pow_f32x8(y)
}

pub fn wide_f32x4_sin(f32x4_sin_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_sin_arg.sin()
}

pub fn wide_f32x4_cos(f32x4_cos_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_cos_arg.cos()
}

pub fn wide_f32x4_tan(f32x4_tan_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_tan_arg.tan()
}

pub fn wide_f32x4_sin_cos(f32x4_sin_cos_arg: wide::f32x4) -> (wide::f32x4, wide::f32x4) {
    f32x4_sin_cos_arg.sin_cos()
}

pub fn wide_f32x4_asin(f32x4_asin_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_asin_arg.asin()
}

pub fn wide_f32x4_acos(f32x4_acos_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_acos_arg.acos()
}

pub fn wide_f32x4_asin_acos(f32x4_asin_acos_arg: wide::f32x4) -> (wide::f32x4, wide::f32x4) {
    f32x4_asin_acos_arg.asin_acos()
}

pub fn wide_f32x4_atan(f32x4_atan_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_atan_arg.atan()
}

pub fn wide_f32x4_atan2(f32x4_atan2_arg: wide::f32x4, y: wide::f32x4) -> wide::f32x4 {
    f32x4_atan2_arg.atan2(y)
}

pub fn wide_f32x4_exp(f32x4_exp_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_exp_arg.exp()
}

pub fn wide_f32x4_ln(f32x4_ln_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_ln_arg.ln()
}

pub fn wide_f32x4_log2(f32x4_log2_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_log2_arg.log2()
}

pub fn wide_f32x4_log10(f32x4_log10_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_log10_arg.log10()
}

pub fn wide_f32x4_powf(f32x4_powf_arg: wide::f32x4, y: f32) -> wide::f32x4 {
    f32x4_powf_arg.powf(y)
}

pub fn wide_f32x4_pow_f32x4(f32x4_pow_f32x4_arg: wide::f32x4, y: wide::f32x4) -> wide::f32x4 {
    f32x4_pow_f32x4_arg.pow_f32x4(y)
}

pub fn wide_f64x4_sin(f64x4_sin_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_sin_arg.sin()
}

pub fn wide_f64x4_cos(f64x4_cos_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_cos_arg.cos()
}

pub fn wide_f64x4_tan(f64x4_tan_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_tan_arg.tan()
}

pub fn wide_f64x4_sin_cos(f64x4_sin_cos_arg: wide::f64x4) -> (wide::f64x4, wide::f64x4) {
    f64x4_sin_cos_arg.sin_cos()
}

pub fn wide_f64x4_asin(f64x4_asin_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_asin_arg.asin()
}

pub fn wide_f64x4_acos(f64x4_acos_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_acos_arg.acos()
}

pub fn wide_f64x4_asin_acos(f64x4_asin_acos_arg: wide::f64x4) -> (wide::f64x4, wide::f64x4) {
    f64x4_asin_acos_arg.asin_acos()
}

pub fn wide_f64x4_atan(f64x4_atan_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_atan_arg.atan()
}

pub fn wide_f64x4_atan2(f64x4_atan2_arg: wide::f64x4, y: wide::f64x4) -> wide::f64x4 {
    f64x4_atan2_arg.atan2(y)
}

pub fn wide_f64x4_exp(f64x4_exp_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_exp_arg.exp()
}

pub fn wide_f64x4_ln(f64x4_ln_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_ln_arg.ln()
}

pub fn wide_f64x4_log2(f64x4_log2_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_log2_arg.log2()
}

pub fn wide_f64x4_log10(f64x4_log10_arg: wide::f64x4) -> wide::f64x4 {
    f64x4_log10_arg.log10()
}

pub fn wide_f64x4_powf(f64x4_powf_arg: wide::f64x4, y: f64) -> wide::f64x4 {
    f64x4_powf_arg.powf(y)
}

pub fn wide_f64x4_pow_f64x4(f64x4_pow_f64x4_arg: wide::f64x4, y: wide::f64x4) -> wide::f64x4 {
    f64x4_pow_f64x4_arg.pow_f64x4(y)
}

pub fn wide_f64x2_sin(f64x2_sin_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_sin_arg.sin()
}

pub fn wide_f64x2_cos(f64x2_cos_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_cos_arg.cos()
}

pub fn wide_f64x2_tan(f64x2_tan_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_tan_arg.tan()
}

pub fn wide_f64x2_sin_cos(f64x2_sin_cos_arg: wide::f64x2) -> (wide::f64x2, wide::f64x2) {
    f64x2_sin_cos_arg.sin_cos()
}

pub fn wide_f64x2_asin(f64x2_asin_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_asin_arg.asin()
}

pub fn wide_f64x2_acos(f64x2_acos_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_acos_arg.acos()
}

pub fn wide_f64x2_asin_acos(f64x2_asin_acos_arg: wide::f64x2) -> (wide::f64x2, wide::f64x2) {
    f64x2_asin_acos_arg.asin_acos()
}

pub fn wide_f64x2_atan(f64x2_atan_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_atan_arg.atan()
}

pub fn wide_f64x2_atan2(f64x2_atan2_arg: wide::f64x2, y: wide::f64x2) -> wide::f64x2 {
    f64x2_atan2_arg.atan2(y)
}

pub fn wide_f64x2_exp(f64x2_exp_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_exp_arg.exp()
}

pub fn wide_f64x2_ln(f64x2_ln_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_ln_arg.ln()
}

pub fn wide_f64x2_log2(f64x2_log2_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_log2_arg.log2()
}

pub fn wide_f64x2_log10(f64x2_log10_arg: wide::f64x2) -> wide::f64x2 {
    f64x2_log10_arg.log10()
}

pub fn wide_f64x2_powf(f64x2_powf_arg: wide::f64x2, y: f64) -> wide::f64x2 {
    f64x2_powf_arg.powf(y)
}

pub fn wide_f64x2_pow_f64x2(f64x2_pow_f64x2_arg: wide::f64x2, y: wide::f64x2) -> wide::f64x2 {
    f64x2_pow_f64x2_arg.pow_f64x2(y)
}

pub fn wide_f32x8_recip(f32x8_recip_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_recip_arg.recip()
}

pub fn wide_f32x8_recip_sqrt(f32x8_recip_sqrt_arg: wide::f32x8) -> wide::f32x8 {
    f32x8_recip_sqrt_arg.recip_sqrt()
}

pub fn wide_f32x4_recip(f32x4_recip_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_recip_arg.recip()
}

pub fn wide_f32x4_recip_sqrt(f32x4_recip_sqrt_arg: wide::f32x4) -> wide::f32x4 {
    f32x4_recip_sqrt_arg.recip_sqrt()
}
