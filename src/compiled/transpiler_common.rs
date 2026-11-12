//! Shared shader transpiler framework: Generic transpiler parameterized by shader language.
//!
//! This module provides a `ShaderLang` trait and `GenericTranspiler<L>` that contains
//! the shared transpilation logic (~2600 lines of `transpile_node_inner`), eliminating
//! ~90% code duplication between WGSL, GLSL, and HLSL transpilers.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use crate::modifiers::TERRAIN_NOISE_SEED;
use crate::operations::BlendMode;
use crate::types::SdfNode;
use std::fmt::Write;

/// Epsilon for constant folding (skip operations that are no-ops)
pub const FOLD_EPSILON: f32 = 1e-6;

// ============================================================================
// ShaderLang trait
// ============================================================================

/// Node kinds the shader transpilers do not express.
///
/// Their child is evaluated as-is and a comment is emitted. This is the
/// single list — the GPU parity oracles skip exactly these, so implementing
/// one of them (and removing it here) puts it under the oracle.
///
/// `LatticeDeform` / `HeightmapDisplacement` / `SdfSkinning` / `IFS` carry
/// per-node data (control points, height field, bones, matrices) that the
/// hardcoded shader has no binding for.
///
/// **Stale since 2.2.0**: every node kind is transpiled (these four included),
/// so [`shader_unsupported_nodes`] returns an empty list for every tree and
/// this array no longer describes the transpilers. Deprecated since 5.0.0;
/// call [`shader_unsupported_nodes`] on the tree instead.
#[deprecated(
    since = "5.0.0",
    note = "stale: every node kind is transpiled since 2.2.0; use shader_unsupported_nodes(&node)"
)]
pub const SHADER_UNSUPPORTED: [&str; 4] = [
    "LatticeDeform",
    "HeightmapDisplacement",
    "SdfSkinning",
    "IFS",
];

/// Polygons with more vertices than this are emitted as a vertex array and a loop function
pub const POLYGON_2D_UNROLL_MAX: usize = 8;

const fn unsupported_name(node: &SdfNode) -> Option<&'static str> {
    // Every node kind is transpiled (IFS / SdfSkinning / ProjectiveTransform
    // with literal matrices, LatticeDeform / HeightmapDisplacement as
    // module-scope data); the list stays as the hook for a future node that
    // cannot be. `ProjectiveTransform` forwarded its child unchanged until
    // the release after 5.0.0 while this returned `None`;
    // `tests/test_node_backend_matrix.rs` now fails when an arm forwards the
    // child of a node that changes the distance and this list is silent.
    match node {
        // neighbours / variation have no shader law yet (refused, not ignored)
        SdfNode::RepeatCells {
            neighbours,
            variation,
            ..
        } if *neighbours || variation.is_some() => Some("RepeatCells (neighbours / variation)"),
        _ => None,
    }
}

/// Names of the nodes in `node`'s tree that the transpilers pass through
/// unchanged; empty when the shader is a faithful port of the tree.
///
/// Since the release after 5.0.0 that is every tree (`ProjectiveTransform`
/// was the last node passed through).
#[must_use]
pub fn shader_unsupported_nodes(node: &SdfNode) -> Vec<&'static str> {
    let mut out = Vec::new();
    let mut stack: Vec<std::sync::Arc<SdfNode>> = vec![std::sync::Arc::new(node.clone())];
    while let Some(n) = stack.pop() {
        if let Some(name) = unsupported_name(&n) {
            out.push(name);
        }
        let mut owned = (*n).clone();
        let mut kids = Vec::new();
        owned.take_children_into(&mut kids);
        stack.extend(kids);
    }
    out
}

/// glam column-major `Mat4::transform_point3` (no perspective divide) as a
/// vec3 constructor expression with literal matrix entries.
fn transform_point3_expr<L: ShaderLang>(m: &[f32; 16], v: &str) -> String {
    L::vec3_ctor(
        &format!(
            "{} * {v}.x + {} * {v}.y + {} * {v}.z + {}",
            lit(m[0]),
            lit(m[4]),
            lit(m[8]),
            lit(m[12])
        ),
        &format!(
            "{} * {v}.x + {} * {v}.y + {} * {v}.z + {}",
            lit(m[1]),
            lit(m[5]),
            lit(m[9]),
            lit(m[13])
        ),
        &format!(
            "{} * {v}.x + {} * {v}.y + {} * {v}.z + {}",
            lit(m[2]),
            lit(m[6]),
            lit(m[10]),
            lit(m[14])
        ),
    )
}

pub(in crate::compiled) mod private {
    /// Seals [`super::ShaderLang`]: only the three language markers
    /// (`WgslLang`, `GlslLang`, `HlslLang`) implement it.
    ///
    /// MSL is deliberately **not** a fourth marker: it is derived from the WGSL
    /// emit through naga (`compiled::msl`) so the 56 `helper_source` laws keep a
    /// single source and cannot drift a fourth way.
    pub trait Sealed {}
}

/// Trait that captures the syntactic differences between WGSL, GLSL, and HLSL.
///
/// **Sealed** (since 3.0): implemented for `WgslLang`, `GlslLang` and
/// `HlslLang` only. Every node kind that needs new syntax adds a required
/// method (`cast_int` and the module-scope declarations in 3.0), and an
/// implementation also has to supply every helper function the emitted
/// laws reference (`helper_source`), so external implementations are not
/// supported.
///
/// Metal is reached without a fourth implementation: `compiled::msl` translates
/// the WGSL emit with naga, which keeps one copy of every law instead of four
/// (see that module's docs).
pub trait ShaderLang: private::Sealed + 'static {
    /// Lowercase language name used in diagnostics (`"wgsl"`, `"glsl"`, `"hlsl"`).
    const NAME: &'static str;

    // ---- Type constructors ----
    /// Construct a 2-component vector from scalar strings.
    fn vec2_ctor(x: &str, y: &str) -> String;
    /// Construct a 3-component vector from scalar strings.
    fn vec3_ctor(x: &str, y: &str, z: &str) -> String;
    /// Construct a 4-component vector from scalar strings.
    fn vec4_ctor(x: &str, y: &str, z: &str, w: &str) -> String;
    /// Return the zero literal for a 2-component vector.
    fn vec2_zero() -> &'static str;
    /// Return the zero literal for a 3-component vector.
    fn vec3_zero() -> &'static str;
    /// Splat a scalar into a 2-component vector.
    fn vec2_splat(v: &str) -> String;
    /// Splat a scalar into a 3-component vector.
    fn vec3_splat(v: &str) -> String;

    // ---- Variable declarations (returns full "    TYPE name = expr;\n") ----
    /// Declare an immutable float variable.
    fn decl_float(name: &str, expr: &str) -> String;
    /// Declare an immutable vec2 variable.
    fn decl_vec2(name: &str, expr: &str) -> String;
    /// Declare an immutable vec3 variable.
    fn decl_vec3(name: &str, expr: &str) -> String;
    /// Declare a mutable float variable.
    fn decl_mut_float(name: &str, expr: &str) -> String;
    /// Declare a mutable vec2 variable.
    fn decl_mut_vec2(name: &str, expr: &str) -> String;
    /// Declare a mutable vec3 variable.
    fn decl_mut_vec3(name: &str, expr: &str) -> String;
    /// Declare a mutable float with type annotation only (no initializer).
    fn decl_mut_float_typed(name: &str) -> String;
    /// Declare a mutable vec3 with type annotation only (no initializer).
    fn decl_mut_vec3_typed(name: &str) -> String;

    // ---- Expressions ----
    /// select(false_val, true_val, cond) or (cond) ? true_val : false_val
    fn select_expr(cond: &str, true_val: &str, false_val: &str) -> String;
    /// Floor modulo `a - b * floor(a / b)` (GLSL `mod`); never the truncated
    /// `%` / `fmod`, which differ for negative operands.
    fn modulo_expr(a: &str, b: &str) -> String;
    /// Two-argument arctangent `atan2(y, x)`: GLSL spells it `atan(y, x)`.
    /// `atan2(y, x)` with the CPU law's exact axis cases (`alice_atan2`
    /// helper, 3.1.0): `atan2(±0, x < 0) = ±π`, `atan2(y, ±0) = ±π/2` as
    /// exact constants, the builtin elsewhere. Callers register the helper.
    fn atan2_expr(y: &str, x: &str) -> String {
        format!("alice_atan2({y}, {x})")
    }
    /// Linear interpolation `a + (b - a) * t`: GLSL / WGSL spell it `mix`,
    /// HLSL `lerp` (a `mix` in HLSL is a compile error — the Morph node
    /// emitted one until 3.2.0, which nothing compiled until the UE5 job).
    fn mix_expr(a: &str, b: &str, t: &str) -> String {
        format!("mix({a}, {b}, {t})")
    }
    /// "f32(x)" / "float(x)"
    fn cast_float(expr: &str) -> String;
    /// A quiet NaN (bits 0x7fc00000): the value of a node the shader has no law for
    fn nan_expr() -> &'static str;
    /// Scalar select: `cond ? a : b`, used where the CPU law is a branch-free
    /// `select` so the tie / sign convention matches. Default is the C-style
    /// ternary (GLSL / HLSL); WGSL overrides with `select(b, a, cond)`.
    fn select_float(cond: &str, a: &str, b: &str) -> String {
        format!("(({cond}) ? ({a}) : ({b}))")
    }
    /// For loop: "for(var i: i32 = 0" / "for(int i = 0"
    fn for_loop_int(name: &str, init: i32, cond: &str, incr: &str) -> String;

    // ---- Param prefix ----
    /// Return the dynamic parameter accessor expression.
    fn param_dynamic(vec_idx: usize, comp: &str) -> String;

    // ---- Function signature ----
    /// Return the SDF entry-point function signature.
    fn func_signature() -> &'static str;

    // ---- Capsule behavior ----
    /// Whether to use max(dot(ba,ba), 1e-10) guard
    const CAPSULE_DEGENERATE_GUARD: bool;
    /// Whether Capsule re-emits point_a params for ba
    const CAPSULE_RE_EMIT_PARAMS: bool;

    // ---- Helper functions ----
    /// Return the source code for a named helper function, if known.
    fn helper_source(name: &str) -> Option<&'static str>;

    // ---- Module-scope data (per-node arrays / functions) ----
    /// "i32(x)" / "int(x)" / "(int)(x)"
    fn cast_int(expr: &str) -> String;
    /// Declare an immutable int variable.
    fn decl_int(name: &str, expr: &str) -> String;
    /// A module-scope float array with an initializer (dynamically indexable).
    fn global_float_array(name: &str, values: &[f32]) -> String;
    /// A module-scope `vec3 name(vec3 p)` function with the given body lines.
    fn global_vec3_fn(name: &str, body: &str) -> String;
    /// A module-scope `float name(int i)` returning `arr[i]`: one function frame holds the array
    /// when a target lowers a dynamically indexed constant array to a local copy per access
    fn global_float_at_fn(name: &str, arr: &str) -> String;
    /// A module-scope `float name(vec2 q)` function: the signed distance to the closed polygon whose
    /// `n` vertices are the pairs read by `at` (the expression of the element at an integer index:
    /// a module-scope array or the node data buffer, [`DataArr::at`]) (the loop of
    /// `sdf_polygon_2d_xy`: the same edge order, clamp and even-odd sign rule)
    fn global_polygon_2d_fn(name: &str, at: &dyn Fn(&str) -> String, n: usize) -> String;
    /// The module-scope data and functions of a [`crate::primitives::SectionStack`]: `name(p)` is
    /// its field (the law of `SectionStack::eval`: per level the even-odd signed distance over all
    /// its loops, linear in y between levels, the end caps like an extrusion, divided by the bound)
    fn global_section_stack(name: &str, st: &SectionStackData) -> String;
}

/// A [`crate::primitives::SectionStack`] flattened for a shader: level heights, every loop's
/// vertices as (x, z) pairs, per level the range of its loops, per loop the range of its vertices
pub struct SectionStackData {
    /// Level heights
    pub levels: Vec<f32>,
    /// All vertices, (x, z) pairs, loop after loop
    pub verts: Vec<f32>,
    /// Level k's loops are `loop_start[k]..loop_start[k + 1]`
    pub loop_start: Vec<i32>,
    /// Loop l's vertices are `vert_start[l]..vert_start[l + 1]`
    pub vert_start: Vec<i32>,
    /// The bound the field is divided by
    pub lipschitz: f32,
    /// The most loops on one level and the most vertices in one loop: the emitted loops' constant
    /// trip counts (a loop reads its count from the data and stops at these, so bad data cannot run
    /// it past them)
    pub max_loops: usize,
    /// See [`SectionStackData::max_loops`]
    pub max_verts: usize,
}

impl SectionStackData {
    /// Flatten a stack
    #[must_use]
    pub fn new(st: &crate::primitives::SectionStack) -> Self {
        let mut verts = Vec::new();
        let mut loop_start = vec![0i32];
        let mut vert_start = vec![0i32];
        for level in &st.loops {
            for lp in level {
                for q in lp {
                    verts.push(q.x);
                    verts.push(q.y);
                }
                vert_start.push((verts.len() / 2) as i32);
            }
            loop_start.push((vert_start.len() - 1) as i32);
        }
        Self {
            levels: st.levels.clone(),
            verts,
            loop_start,
            vert_start,
            lipschitz: st.lipschitz,
            max_loops: st.loops.iter().map(Vec::len).max().unwrap_or(0).max(1),
            max_verts: st
                .loops
                .iter()
                .flatten()
                .map(Vec::len)
                .max()
                .unwrap_or(0)
                .max(1),
        }
    }
}

#[cfg(any(feature = "glsl", feature = "hlsl"))]
fn lits(v: &[f32]) -> String {
    v.iter().map(|x| lit(*x)).collect::<Vec<_>>().join(", ")
}
#[cfg(any(feature = "glsl", feature = "hlsl"))]
fn ints(v: &[i32]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The section stack in a C-like shading language (GLSL / HLSL): `v2` is the vec2 type,
/// `farr(name, values)` / `iarr(name, values)` declare constant arrays
#[cfg(any(feature = "glsl", feature = "hlsl"))]
pub(crate) fn c_like_section_stack(
    name: &str,
    st: &SectionStackData,
    v2: &str,
    farr: &dyn Fn(&str, usize, &str) -> String,
    iarr: &dyn Fn(&str, usize, &str) -> String,
) -> String {
    let n = st.levels.len();
    let mut s = String::new();
    s.push_str(&farr(&format!("{name}_y"), n, &lits(&st.levels)));
    s.push_str(&farr(
        &format!("{name}_v"),
        st.verts.len(),
        &lits(&st.verts),
    ));
    s.push_str(&iarr(
        &format!("{name}_ls"),
        st.loop_start.len(),
        &ints(&st.loop_start),
    ));
    s.push_str(&iarr(
        &format!("{name}_vs"),
        st.vert_start.len(),
        &ints(&st.vert_start),
    ));
    s.push_str(&format!(
        "float {name}_d2({v2} q, int k) {{
    float d = 1e30; // above any squared distance; not the f32 maximum, whose decimal some WGSL compilers refuse
    float s = 1.0;
    for (int lc = 0; lc < {lmax}; lc++) {{
        int l = {name}_ls[k @ 0..{lsk}] + lc;
        if (l >= {name}_ls[k + 1 @ 1..{lsk1}]) {{ break; }}
        int a = {name}_vs[l @ 0..{vsl}];
        int m = {name}_vs[l + 1 @ 1..{vsl1}] - a;
        int j = m - 1;
        for (int i = 0; i < {vmax}; i++) {{
            if (i >= m) {{ break; }}
            {v2} vi = {v2}({name}_v[2 * (a + i) @ 0..{v0}], {name}_v[2 * (a + i) + 1 @ 1..{v1}]);
            {v2} vj = {v2}({name}_v[2 * (a + j) @ 0..{v0}], {name}_v[2 * (a + j) + 1 @ 1..{v1}]);
            {v2} e = vj - vi;
            {v2} w = q - vi;
            float ee = dot(e, e);
            {v2} b = w;
            if (ee > 0.0) {{ b = w - e * clamp(dot(w, e) / ee, 0.0, 1.0); }}
            d = min(d, dot(b, b));
            bool c1 = q.y >= vi.y;
            bool c2 = q.y < vj.y;
            bool c3 = e.x * w.y > e.y * w.x;
            if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) {{ s = -s; }}
            j = i;
        }}
    }}
    return s * sqrt(d);
}}
float {name}(vec3_t p) {{
    {v2} q = {v2}(p.x, p.z);
    float y0 = {name}_y[0 @ 0..0];
    float yn = {name}_y[{last} @ {last}..{last}];
    float d;
    float h;
    if (p.y < y0) {{
        d = {name}_d2(q, 0);
        h = y0 - p.y;
    }} else if (p.y > yn) {{
        d = {name}_d2(q, {last});
        h = p.y - yn;
    }} else {{
        int c = 0;
        for (int k = 0; k < {n}; k++) {{ if ({name}_y[k @ 0..{last}] <= p.y) {{ c = k + 1; }} }}
        int i = (c < 1 ? 1 : (c > {last} ? {last} : c)) - 1;
        float t = (p.y - {name}_y[i @ 0..{lastm1}]) / ({name}_y[i + 1 @ 1..{last}] - {name}_y[i @ 0..{lastm1}]);
        d = (1.0 - t) * {name}_d2(q, i) + t * {name}_d2(q, i + 1);
        h = max(y0 - p.y, p.y - yn);
    }}
    float raw = min(max(d, h), 0.0) + length({v2}(max(d, 0.0), max(h, 0.0)));
    return raw / {lip};
}}
",
        last = n - 1,
        lip = lit(st.lipschitz),
        lmax = st.max_loops,
        vmax = st.max_verts,
        // the reads' index ranges: level k < n, loop l < loops, vertex a + i < vertices
        lsk = n - 1,
        lsk1 = n,
        vsl = st.vert_start.len() - 2,
        vsl1 = st.vert_start.len() - 1,
        v0 = st.verts.len() - 2,
        v1 = st.verts.len() - 1,
        lastm1 = n - 2,
    ));
    s
}

/// The floats of one far-bound piece in the C-like dialects' per-piece arrays
const PIECE_FLOATS_SHADER: usize = crate::primitives::loft::PIECE_FLOATS;

/// Float literal for shader source: shortest round-trip form with a decimal
/// point (`2.0`, `0.33333334`), so the GPU sees the same f32 the CPU has.
pub fn lit(v: f32) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    }
}

// ============================================================================
// Transpile mode (language-independent)
// ============================================================================

/// Transpilation mode: hardcoded constants or dynamic parameter buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranspileModeLang {
    /// Inline numeric literals directly into shader source.
    Hardcoded,
    /// Reference a runtime parameter buffer for SDF constants.
    Dynamic,
}

// ============================================================================
// GenericTranspiler<L: ShaderLang>
// ============================================================================

/// Language-generic SDF-to-shader transpiler parameterized by [`ShaderLang`].
pub struct GenericTranspiler<L: ShaderLang> {
    /// Monotonic counter for generating unique variable names.
    pub var_counter: usize,
    /// Accumulated helper function names required by the generated shader.
    pub helper_functions: Vec<&'static str>,
    /// Whether to inline constants or use a parameter buffer.
    pub mode: TranspileModeLang,
    /// Collected float parameters for dynamic mode.
    pub params: Vec<f32>,
    /// Module-scope declarations the body needs (per-node data arrays and
    /// functions for lattice / heightmap nodes), emitted before `sdf_eval`.
    pub globals: String,
    /// Node data buffer (WGSL GPU evaluation only): when `Some`, per-node data arrays are
    /// appended here and read from the storage buffer `alice_node_data` (binding 4) instead of
    /// being emitted as module-scope constants (see [`GenericTranspiler::data_array`])
    pub node_data: Option<Vec<f32>>,
    /// Data arrays already emitted, by (length, FNV-1a of the bits): the same values are emitted once
    /// (a second walk of the tree, e.g. the rest point, reuses the first walk's arrays), with whether
    /// the accessor function was emitted
    data_seen: Vec<((usize, u64), DataArr, bool)>,
    /// Emit the loft and stack data reads as checked reads (counting reads outside their proven
    /// index range; see `Reads`): tests and diagnostics only, the values are the same
    pub checked_reads: bool,
    /// Reads the emit could not prove in range (an emitter's range annotation outside its array):
    /// a non-empty list refuses the tree (`try_transpile*`)
    pub read_errors: Vec<String>,
    _phantom: std::marker::PhantomData<L>,
}

/// A per-node data array as the emitted code reads it: a module-scope constant array, or a
/// range of the node data buffer
#[derive(Debug, Clone)]
pub struct DataArr {
    name: String,
    base: Option<usize>,
    len: usize,
}

impl DataArr {
    /// [`DataArr::at`] through the accessor function `{name}_at` in constant mode (emitted by
    /// [`GenericTranspiler::data_array_accessed`]), the buffer read otherwise
    #[must_use]
    pub fn at_fn(&self, idx: &str) -> String {
        match self.base {
            None => format!("{}_at({idx})", self.name),
            Some(_) => self.at(idx),
        }
    }

    /// The element at the (integer) index expression `idx`
    #[must_use]
    pub fn at(&self, idx: &str) -> String {
        match self.base {
            None => format!("{}[{idx}]", self.name),
            // clamped to this node's own range: a corrupted or stale index reads one of the node's
            // values, never past the buffer or another node's data
            Some(b) => format!(
                "alice_node_data[clamp({b} + ({idx}), {b}, {})]",
                b + self.len.max(1) - 1
            ),
        }
    }

    /// The node data buffer range `(base, len)`, or `None` for a module-scope constant array
    #[must_use]
    pub fn range(&self) -> Option<(usize, usize)> {
        self.base.map(|b| (b, self.len))
    }
}

/// A tree the shader emit refuses: data reads (of a loft's or a stack's arrays) whose index range,
/// proven from the node's validated data at emit time, would leave their array
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranspileError {
    /// One line per refused read: the array, the read and its range
    pub reads: Vec<String>,
}

impl TranspileError {
    /// `Ok` for no refused reads, the error otherwise
    pub(crate) fn from_reads(reads: Vec<String>) -> Result<(), Self> {
        if reads.is_empty() {
            Ok(())
        } else {
            Err(Self { reads })
        }
    }
}

impl std::fmt::Display for TranspileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shader emit refused {} data read(s) outside their arrays: {}",
            self.reads.len(),
            self.reads.join("; ")
        )
    }
}

impl std::error::Error for TranspileError {}

/// How an emitter's annotated reads `{arr}[expr @ lo..hi]` are written out (see [`render_reads`])
pub(crate) enum Reads<'a> {
    /// `arr[expr]`, a module-scope array of the dialect
    Plain,
    /// `alice_node_data[clamp(base + (expr), base + lo, base + hi)]`, a range of the node data buffer
    Buffer(&'a DataArr),
    /// `alice_nd(base + (expr), base + lo, base + hi)`: the buffer read that counts the reads outside
    /// `lo..=hi` (WGSL, [`crate::compiled::WgslShader::try_transpile_with_node_data_checked`])
    CheckedBuffer(&'a DataArr),
    /// `ALICE_RD(arr, (expr), lo, hi)`: a module-scope array read through a macro the host defines
    /// (the C-like dialects, [`crate::compiled::HlslShader::try_transpile_checked`])
    CheckedPlain,
}

/// The most elements a module-scope array is emitted with in GLSL and HLSL: larger arrays are written
/// as consecutive arrays of at most this many elements, read through one function. A SPIR-V
/// instruction holds at most 65535 words, so a constant composite of more than 65532 constituents
/// cannot be encoded (naga writes it anyway, with a word count of 0, and the module is invalid on
/// every Vulkan driver). A loft at a body's size carries about 85000 values. Reported upstream:
/// <https://github.com/gfx-rs/wgpu/issues/10877>
pub(crate) const MODULE_ARRAY_CHUNK: usize = 16384;

/// The most elements of one WGSL array value constructor (W3C WGSL, section 2.4 "Limits": "Maximum
/// number of elements in value constructor expression of array type", 2047); WGSL arrays are split at
/// this size. A read of a constant array at a run-time index copies the whole array it reads from
/// (naga spills the value to a function variable), so smaller chunks also make each read cheaper
pub(crate) const WGSL_ARRAY_CONSTRUCTOR_LIMIT: usize = 2047;

/// The chunk size of a dialect's module-scope arrays
pub(crate) fn module_array_chunk(dialect: &str) -> usize {
    match dialect {
        "wgsl" => WGSL_ARRAY_CONSTRUCTOR_LIMIT,
        _ => MODULE_ARRAY_CHUNK,
    }
}

/// One module-scope array declaration of a dialect, on one line: `head` up to the first element,
/// the elements, `tail` from after the last
struct ArrayDecl<'a> {
    name: &'a str,
    elem: &'static str,
    len: usize,
    head: &'a str,
    items: Vec<&'a str>,
    tail: &'a str,
}

/// The array declaration on `line`, as each dialect writes it (`ShaderLang::global_float_array`,
/// the loft and stack emitters): WGSL `const NAME: array<f32, N> = array<f32, N>(…);`, GLSL
/// `const float NAME[N] = float[N](…);`, HLSL `static const float NAME[N] = { … };` (and `int`)
fn array_decl<'a>(line: &'a str, dialect: &str) -> Option<ArrayDecl<'a>> {
    let (name, elem, len) = match dialect {
        "wgsl" => {
            let rest = line.strip_prefix("const ")?;
            let (name, ty) = rest.split_once(": array<")?;
            let (elem, rest) = ty.split_once(", ")?;
            let elem = match elem {
                "f32" => "f32",
                "i32" => "i32",
                _ => return None,
            };
            (name, elem, rest.split_once('>')?.0)
        }
        "glsl" | "hlsl" => {
            let rest = line.strip_prefix(if dialect == "glsl" {
                "const "
            } else {
                "static const "
            })?;
            let (elem, rest) = rest.split_once(' ')?;
            let elem = match elem {
                "float" => "float",
                "int" => "int",
                _ => return None,
            };
            let (name, rest) = rest.split_once('[')?;
            (name, elem, rest.split_once(']')?.0)
        }
        _ => return None,
    };
    let len = len.trim().parse::<usize>().ok()?;
    let eq = line.find(" = ")?;
    let open = eq + line[eq..].find(['(', '{'])? + 1;
    let close = line.rfind([')', '}'])?;
    let inner = &line[open..close];
    let lead = inner.len() - inner.trim_start().len();
    let trail = inner.len() - inner.trim_end().len();
    let items: Vec<&str> = inner
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .collect();
    (items.len() == len).then_some(ArrayDecl {
        name,
        elem,
        len,
        head: &line[..open + lead],
        items,
        tail: &line[close - trail..],
    })
}

/// `src` with every module-scope array longer than `chunk` written as consecutive arrays
/// `NAME_c0, NAME_c1, …` of at most `chunk` elements and a function `NAME_r(i)` reading element `i`;
/// each read `NAME[e]` becomes `NAME_r(e)` (as an integer). The values and the order of every read are
/// those of the one array, so the field is the same
pub(crate) fn split_large_arrays(src: &str, dialect: &str, chunk: usize) -> String {
    let decls: Vec<ArrayDecl<'_>> = src
        .lines()
        .filter_map(|l| array_decl(l, dialect))
        .filter(|d| d.len > chunk)
        .collect();
    if decls.is_empty() {
        return src.to_owned();
    }
    let mut out = String::with_capacity(src.len() + 1024);
    for line in src.split_inclusive('\n') {
        let Some(d) = decls.iter().find(|d| {
            array_decl(line.trim_end_matches('\n'), dialect).is_some_and(|x| x.name == d.name)
        }) else {
            out.push_str(line);
            continue;
        };
        let parts = d.len.div_ceil(chunk);
        for k in 0..parts {
            let items = &d.items[k * chunk..((k + 1) * chunk).min(d.len)];
            let n = items.len();
            let head = d
                .head
                .replacen(d.name, &format!("{}_c{k}", d.name), 1)
                .replace(&format!(", {}>", d.len), &format!(", {n}>"))
                .replace(&format!("[{}]", d.len), &format!("[{n}]"));
            out.push_str(&format!("{head}{}{}\n", items.join(", "), d.tail));
        }
        out.push_str(&chunk_reader(d, dialect, parts, chunk));
    }
    let mut src = out;
    for d in &decls {
        src = rewrite_reads(&src, d.name, dialect);
    }
    src
}

/// The function `NAME_r(i)`: element `i` of the chunks of `d`
fn chunk_reader(d: &ArrayDecl<'_>, dialect: &str, parts: usize, chunk: usize) -> String {
    let name = d.name;
    let mut body = String::new();
    for k in 0..parts {
        let read = format!("{name}_c{k}[i - {}]", k * chunk);
        if k + 1 < parts {
            body.push_str(&format!(
                "    if (i < {}) {{ return {read}; }}\n",
                (k + 1) * chunk
            ));
        } else {
            body.push_str(&format!("    return {read};\n"));
        }
    }
    match dialect {
        "wgsl" => format!("fn {name}_r(i: i32) -> {} {{\n{body}}}\n", d.elem),
        _ => format!("{} {name}_r(int i) {{\n{body}}}\n", d.elem),
    }
}

/// Every read `name[e]` in `src` (not a declaration, not a longer identifier ending in `name`)
/// written `name_r(int(e))`
fn rewrite_reads(src: &str, name: &str, dialect: &str) -> String {
    let pat = format!("{name}[");
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while let Some(rel) = src[i..].find(&pat) {
        let at = i + rel;
        let prev_ident = at > 0 && (bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_');
        out.push_str(&src[i..at]);
        if prev_ident {
            out.push_str(&pat);
            i = at + pat.len();
            continue;
        }
        let open = at + pat.len();
        let (mut depth, mut j) = (1usize, open);
        while depth > 0 && j < bytes.len() {
            match bytes[j] {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            j += 1;
        }
        let expr = &src[open..j - 1];
        let int = match dialect {
            "wgsl" => format!("i32({expr})"),
            "hlsl" => format!("(int)({expr})"),
            _ => format!("int({expr})"),
        };
        out.push_str(&format!("{name}_r({int})"));
        i = j;
    }
    out.push_str(&src[i..]);
    out
}

/// The emitter's reads of the array `arr` (a block of `len` values, at `offset` in the node's data
/// for the buffer modes) written out in `mode`, `wrap` applied to each read. Every read must carry
/// its index range `@ lo..hi` (integers, inclusive): a read without one, or a range outside `0..len`,
/// is recorded in `errs` (the emit refuses the tree). A read whose index leaves its range is caught at
/// run time by the checked modes; the plain buffer read clamps it to the range (GPU safety only, the
/// checked modes show it never happens)
pub(crate) fn render_reads(
    src: &str,
    arr: &str,
    len: usize,
    offset: usize,
    mode: &Reads<'_>,
    wrap: &dyn Fn(String) -> String,
    errs: &mut Vec<String>,
) -> String {
    let pat = format!("{arr}[");
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while let Some(rel) = src[i..].find(&pat) {
        let at = i + rel;
        let prev_ident = at > 0 && (bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_');
        // a C-like declaration `float NAME[N]` / `int NAME[N]` is not a read
        let decl = src[..at].ends_with("float ") || src[..at].ends_with("int ");
        out.push_str(&src[i..at]);
        if prev_ident || decl {
            out.push_str(&pat);
            i = at + pat.len();
            continue;
        }
        let open = at + pat.len();
        let (mut depth, mut j) = (1usize, open);
        while depth > 0 && j < bytes.len() {
            match bytes[j] {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            j += 1;
        }
        let inner = &src[open..j - 1];
        let Some((expr, range)) = inner.rsplit_once(" @ ") else {
            errs.push(format!("{arr}: a read without its index range: [{inner}]"));
            out.push_str(&src[at..j]);
            i = j;
            continue;
        };
        let bounds = range.split_once("..").and_then(|(a, b)| {
            Some((
                a.trim().parse::<usize>().ok()?,
                b.trim().parse::<usize>().ok()?,
            ))
        });
        let Some((lo, hi)) = bounds else {
            errs.push(format!(
                "{arr}: an index range that is not two integers: {range}"
            ));
            out.push_str(&src[at..j]);
            i = j;
            continue;
        };
        if lo > hi || hi >= len {
            errs.push(format!(
                "{arr}: the read [{expr}] ranges over {lo}..={hi}, outside the array's 0..{len}"
            ));
        }
        let read = match mode {
            Reads::Plain => format!("{arr}[{expr}]"),
            Reads::CheckedPlain => format!("ALICE_RD({arr}, ({expr}), {lo}, {hi})"),
            Reads::Buffer(d) | Reads::CheckedBuffer(d) => {
                let (base, _) = d.range().expect("a buffer range");
                let b = base + offset;
                if matches!(mode, Reads::Buffer(_)) {
                    format!(
                        "alice_node_data[clamp({b} + ({expr}), {}, {})]",
                        b + lo,
                        b + hi
                    )
                } else {
                    format!("alice_nd({b} + ({expr}), {}, {})", b + lo, b + hi)
                }
            }
        };
        out.push_str(&wrap(read));
        i = j;
    }
    out.push_str(&src[i..]);
    out
}

impl<L: ShaderLang> GenericTranspiler<L> {
    /// Create a new transpiler in the given mode.
    pub const fn new(mode: TranspileModeLang) -> Self {
        Self {
            var_counter: 0,
            helper_functions: Vec::new(),
            mode,
            params: Vec::new(),
            globals: String::new(),
            node_data: None,
            data_seen: Vec::new(),
            checked_reads: false,
            read_errors: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// The bilinear sample of a height map at the face coordinates `(u, v)` (before the scale and
    /// the mapping to [0, w-1] x [0, h-1]), as `modifiers::heightmap_displacement`'s face sample
    #[allow(
        clippy::too_many_arguments,
        reason = "the face's coordinates and the map's constants"
    )]
    fn heightmap_face_sample(
        &mut self,
        code: &mut String,
        arr: &DataArr,
        u: &str,
        v: &str,
        sc: &str,
        wm1: &str,
        hm1: &str,
        width: u32,
        height: u32,
    ) -> String {
        let (uu, vv) = (self.next_var(), self.next_var());
        code.push_str(&L::decl_float(
            &uu,
            &format!("clamp(({u} * {sc} * 0.5 + 0.5) * {wm1}, 0.0, {wm1})"),
        ));
        code.push_str(&L::decl_float(
            &vv,
            &format!("clamp(({v} * {sc} * 0.5 + 0.5) * {hm1}, 0.0, {hm1})"),
        ));
        let (u0, v0, u1, v1) = (
            self.next_var(),
            self.next_var(),
            self.next_var(),
            self.next_var(),
        );
        let (fu, fv) = (self.next_var(), self.next_var());
        let wi = i64::from(width);
        code.push_str(&L::decl_int(&u0, &L::cast_int(&format!("floor({uu})"))));
        code.push_str(&L::decl_int(&v0, &L::cast_int(&format!("floor({vv})"))));
        code.push_str(&L::decl_int(&u1, &format!("min({u0} + 1, {})", wi - 1)));
        code.push_str(&L::decl_int(
            &v1,
            &format!("min({v0} + 1, {})", i64::from(height) - 1),
        ));
        code.push_str(&L::decl_float(&fu, &format!("{uu} - floor({uu})")));
        code.push_str(&L::decl_float(&fv, &format!("{vv} - floor({vv})")));
        let (a, b, s) = (self.next_var(), self.next_var(), self.next_var());
        code.push_str(&L::decl_float(
            &a,
            &format!(
                "{} * (1.0 - {fu}) + {} * {fu}",
                arr.at(&format!("{v0} * {wi} + {u0}")),
                arr.at(&format!("{v0} * {wi} + {u1}"))
            ),
        ));
        code.push_str(&L::decl_float(
            &b,
            &format!(
                "{} * (1.0 - {fu}) + {} * {fu}",
                arr.at(&format!("{v1} * {wi} + {u0}")),
                arr.at(&format!("{v1} * {wi} + {u1}"))
            ),
        ));
        code.push_str(&L::decl_float(
            &s,
            &format!("{a} * (1.0 - {fv}) + {b} * {fv}"),
        ));
        s
    }

    /// Register a per-node data array: appended to the node data buffer when it is enabled,
    /// otherwise emitted as a module-scope constant array named `name`.
    pub fn data_array(&mut self, name: &str, values: &[f32]) -> DataArr {
        let key = (
            values.len(),
            values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                (h ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
            }),
        );
        if let Some((_, d, _)) = self.data_seen.iter().find(|(k, _, _)| *k == key) {
            return d.clone();
        }
        let d = self.data_array_new(name, values);
        self.data_seen.push((key, d.clone(), false));
        d
    }

    fn data_array_new(&mut self, name: &str, values: &[f32]) -> DataArr {
        if let Some(buf) = self.node_data.as_mut() {
            let base = buf.len();
            buf.extend_from_slice(values);
            DataArr {
                name: name.to_owned(),
                base: Some(base),
                len: values.len(),
            }
        } else {
            self.globals.push_str(&L::global_float_array(name, values));
            DataArr {
                name: name.to_owned(),
                base: None,
                len: values.len(),
            }
        }
    }

    /// [`GenericTranspiler::data_array`] with the accessor function `{name}_at` in constant mode,
    /// for arrays read with many dynamic indices (read them with [`DataArr::at_fn`])
    pub fn data_array_accessed(&mut self, name: &str, values: &[f32]) -> DataArr {
        let d = self.data_array(name, values);
        let seen = self
            .data_seen
            .iter_mut()
            .find(|(_, e, _)| e.name == d.name)
            .expect("data_array registers every array");
        if d.base.is_none() && !seen.2 {
            self.globals
                .push_str(&L::global_float_at_fn(&format!("{}_at", d.name), &d.name));
        }
        seen.2 = true;
        d
    }

    /// A loft's module-scope data and functions: in WGSL with the node data buffer the data is a
    /// clamped range of `alice_node_data` (no module-scope array: a WGSL array constructor is limited
    /// to 32767 elements), otherwise the loft's own module-scope arrays
    fn loft_globals(&mut self, arr: &str, func: &str, flat: &[f32]) {
        let funcs = crate::compiled::loft_shader::loft_functions(func, arr, flat, L::NAME);
        if self.node_data.is_some() && L::NAME == "wgsl" {
            let values = crate::compiled::loft_shader::wgsl_values(flat);
            let d = self.data_array(arr, &values);
            let src = self.render_data(&funcs, arr, values.len(), Some((&d, 0)), &|x| x);
            self.globals.push_str(&src);
        } else {
            self.globals
                .push_str(&crate::compiled::loft_shader::loft_array(
                    arr,
                    flat,
                    L::NAME,
                    L::global_float_array,
                ));
            let values = crate::compiled::loft_shader::wgsl_values(flat);
            let pieces = flat[16] as usize;
            let mut src = funcs;
            if L::NAME != "wgsl" && pieces > 0 {
                // the C-like dialects: the block without its pieces, each piece and the directions on
                // their own (see `loft_shader::loft_array`)
                src = self.render_data(
                    &src,
                    arr,
                    crate::compiled::loft_shader::pieces_at(flat),
                    None,
                    &|x| x,
                );
                for q in 0..pieces {
                    src = self.render_data(
                        &src,
                        &format!("{arr}_p{q}"),
                        PIECE_FLOATS_SHADER,
                        None,
                        &|x| x,
                    );
                }
                src = self.render_data(&src, &format!("{arr}_dir"), 32, None, &|x| x);
            } else {
                src = self.render_data(&src, arr, values.len(), None, &|x| x);
            }
            self.globals.push_str(&src);
        }
    }

    /// An emitter's annotated reads of `arr` (`len` values) written out for this transpiler: in the
    /// node data buffer at `(range, offset)`, or a module-scope array; checked when
    /// [`GenericTranspiler::checked_reads`] is set. Ranges the emit cannot prove go to `read_errors`
    fn render_data(
        &mut self,
        src: &str,
        arr: &str,
        len: usize,
        buffer: Option<(&DataArr, usize)>,
        wrap: &dyn Fn(String) -> String,
    ) -> String {
        let mode = match (buffer, self.checked_reads) {
            (Some((d, _)), false) => Reads::Buffer(d),
            (Some((d, _)), true) => Reads::CheckedBuffer(d),
            (None, false) => Reads::Plain,
            (None, true) => Reads::CheckedPlain,
        };
        let offset = buffer.map_or(0, |b| b.1);
        render_reads(src, arr, len, offset, &mode, wrap, &mut self.read_errors)
    }

    /// Allocate the next unique variable name (d0, d1, ...).
    pub fn next_var(&mut self) -> String {
        let var = format!("d{}", self.var_counter);
        self.var_counter += 1;
        var
    }

    /// Register a helper function by name, deduplicating.
    ///
    /// Helpers that take a polar angle go through `alice_atan2` (3.1.0), which
    /// is emitted first so it is defined before its users.
    pub fn ensure_helper(&mut self, name: &'static str) {
        if matches!(
            name,
            "sdf_regular_polygon" | "sdf_star_polygon" | "sdf_helix"
        ) && !self.helper_functions.contains(&"alice_atan2")
        {
            self.helper_functions.push("alice_atan2");
        }
        if !self.helper_functions.contains(&name) {
            self.helper_functions.push(name);
        }
    }

    /// Register a float parameter and return its shader expression.
    pub fn param(&mut self, value: f32) -> String {
        match self.mode {
            TranspileModeLang::Hardcoded => lit(value),
            TranspileModeLang::Dynamic => {
                let idx = self.params.len();
                self.params.push(value);
                let vec_idx = idx / 4;
                let comp = match idx % 4 {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => "w",
                };
                L::param_dynamic(vec_idx, comp)
            }
        }
    }

    /// Emit inline code for the stairs-style smooth union operator.
    pub fn emit_stairs_union_inline(
        &mut self,
        code: &mut String,
        d_a: &str,
        d_b: &str,
        r_s: &str,
        n_s: &str,
        out_var: &str,
    ) {
        let s_str = self.param(std::f32::consts::FRAC_1_SQRT_2);
        let s2_str = self.param(std::f32::consts::SQRT_2);
        let rn = self.next_var();
        let off = self.next_var();
        let step = self.next_var();
        let px = self.next_var();
        let py = self.next_var();
        let px2 = self.next_var();
        let t = self.next_var();
        let px3 = self.next_var();
        let d2 = self.next_var();
        let npx = self.next_var();
        let npy = self.next_var();
        let edge = self.next_var();

        code.push_str(&L::decl_float(&rn, &format!("{} / {}", r_s, n_s)));
        code.push_str(&L::decl_float(
            &off,
            &format!("({} - {}) * 0.5 * {}", r_s, rn, s2_str),
        ));
        code.push_str(&L::decl_float(
            &step,
            &format!("{} * {} / {}", r_s, s2_str, n_s),
        ));
        code.push_str(&L::decl_float(
            &px,
            &format!("({} - {}) * {} - {}", d_b, d_a, s_str, off),
        ));
        code.push_str(&L::decl_float(
            &py,
            &format!("({} + {}) * {} - {}", d_a, d_b, s_str, off),
        ));
        code.push_str(&L::decl_float(
            &px2,
            &format!("{} + 0.5 * {} * {}", px, s2_str, rn),
        ));
        code.push_str(&L::decl_float(&t, &format!("{} + {} * 0.5", px2, step)));
        code.push_str(&L::decl_float(
            &px3,
            &format!(
                "{} - {} * floor({} / {}) - {} * 0.5",
                t, step, t, step, step
            ),
        ));
        code.push_str(&L::decl_float(
            &d2,
            &format!("min(min({}, {}), {})", d_a, d_b, py),
        ));
        code.push_str(&L::decl_float(
            &npx,
            &format!("({} + {}) * {}", px3, py, s_str),
        ));
        code.push_str(&L::decl_float(
            &npy,
            &format!("({} - {}) * {}", py, px3, s_str),
        ));
        code.push_str(&L::decl_float(&edge, &format!("0.5 * {}", rn)));
        code.push_str(&L::decl_float(
            out_var,
            &format!("min({}, max({} - {}, {} - {}))", d2, npx, edge, npy, edge),
        ));
    }

    /// Assemble the final shader string: helpers + module-scope globals +
    /// function signature + body.
    ///
    /// # Panics
    ///
    /// Panics if a referenced helper has no source in [`ShaderLang::helper_source`]:
    /// that is a transpiler bug which would otherwise surface as a shader
    /// compile error at runtime (until 1.10.3 the five GDF polyhedra were
    /// skipped silently by a duplicated table).
    pub fn generate_shader(&self, body: &str) -> String {
        let mut shader = String::new();
        for helper in &self.helper_functions {
            let src = L::helper_source(helper).unwrap_or_else(|| {
                panic!(
                    "{} transpiler: no helper source registered for `{helper}`",
                    L::NAME
                )
            });
            shader.push_str(src);
            shader.push('\n');
        }
        // Per-node data arrays / functions (lattice, heightmap)
        shader.push_str(&self.globals);
        writeln!(shader, "{}", L::func_signature()).unwrap();
        shader.push_str(body);
        shader.push_str("}\n");
        split_large_arrays(&shader, L::NAME, module_array_chunk(L::NAME))
    }

    /// The grid warp's point computation (shared by the field and the rest point): emits `q`, the box
    /// distance and the blended map, and returns the names of (x = M q, e, b, Λ(q), F(q))
    fn emit_grid_warp_point(
        &mut self,
        warp: &crate::transforms::GridWarp,
        point_var: &str,
        code: &mut String,
    ) -> (String, String, String, String, String) {
        let data = warp.data();
        let id = self.var_counter;
        self.var_counter += 1;
        let arr = self.data_array_accessed(&format!("alice_gw_{id}"), &data.maps);
        let lam_arr = self.data_array_accessed(&format!("alice_gwl_{id}"), warp.node_lambda());
        let floor_arr = (!data.floor.is_empty())
            .then(|| self.data_array_accessed(&format!("alice_gwf_{id}"), &data.floor));
        let bmin = warp.box_min();
        let bmax = warp.box_max();
        let [nx, ny, nz] = data.dims.map(i64::from);
        let p = point_var;
        let (q, e, g) = (self.next_var(), self.next_var(), self.next_var());
        code.push_str(&L::decl_vec3(
            &q,
            &format!(
                "clamp({p}, {}, {})",
                L::vec3_ctor(&lit(bmin.x), &lit(bmin.y), &lit(bmin.z)),
                L::vec3_ctor(&lit(bmax.x), &lit(bmax.y), &lit(bmax.z))
            ),
        ));
        code.push_str(&L::decl_float(&e, &format!("length({p} - {q})")));
        let (bw, bx) = (self.next_var(), self.next_var());
        code.push_str(&L::decl_vec3(
            &bw,
            &format!(
                "max({} - {p}, {p} - {})",
                L::vec3_ctor(&lit(bmin.x), &lit(bmin.y), &lit(bmin.z)),
                L::vec3_ctor(&lit(bmax.x), &lit(bmax.y), &lit(bmax.z))
            ),
        ));
        code.push_str(&L::decl_float(
            &bx,
            &format!("{e} + min(max({bw}.x, max({bw}.y, {bw}.z)), 0.0)"),
        ));
        code.push_str(&L::decl_vec3(
            &g,
            &format!(
                "({q} - {}) * {}",
                L::vec3_ctor(&lit(bmin.x), &lit(bmin.y), &lit(bmin.z)),
                lit(1.0 / data.cell)
            ),
        ));
        let (ix, iy, iz) = (self.next_var(), self.next_var(), self.next_var());
        for (iv, c, n) in [(&ix, "x", nx), (&iy, "y", ny), (&iz, "z", nz)] {
            code.push_str(&L::decl_int(
                iv,
                &format!(
                    "clamp({}, 0, {})",
                    L::cast_int(&format!("floor({g}.{c})")),
                    n - 2
                ),
            ));
        }
        let (fx, fy, fz) = (self.next_var(), self.next_var(), self.next_var());
        for (fv, c, iv) in [(&fx, "x", &ix), (&fy, "y", &iy), (&fz, "z", &iz)] {
            code.push_str(&L::decl_float(
                fv,
                &format!("{g}.{c} - {}", L::cast_float(iv)),
            ));
        }
        let (node, base) = (self.next_var(), self.next_var());
        code.push_str(&L::decl_int(
            &node,
            &format!("{ix} + {nx} * ({iy} + {ny} * {iz})"),
        ));
        code.push_str(&L::decl_int(&base, &format!("{node} * 12")));
        let mut w = Vec::new();
        for c in 0..8i64 {
            let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
            let f = |d: i64, v: &str| {
                if d == 1 {
                    v.to_owned()
                } else {
                    format!("(1.0 - {v})")
                }
            };
            let wv = self.next_var();
            code.push_str(&L::decl_float(
                &wv,
                &format!("{} * {} * {}", f(dx, &fx), f(dy, &fy), f(dz, &fz)),
            ));
            w.push((wv, (dx + nx * (dy + ny * dz)) * 12));
        }
        // Λ(q): the node bounds blended with the same weights, in the CPU order
        let mut lexpr = "0.0".to_owned();
        for (wv, off) in &w {
            lexpr = format!(
                "({lexpr} + {wv} * {})",
                lam_arr.at_fn(&format!("{node} + {}", off / 12))
            );
        }
        let lam = self.next_var();
        code.push_str(&L::decl_float(&lam, &lexpr));
        // F(q): the floor blended with the same weights (−reach without a floor)
        let fl = self.next_var();
        match &floor_arr {
            Some(fa) => {
                let mut fexpr = "0.0".to_owned();
                for (wv, off) in &w {
                    fexpr = format!(
                        "({fexpr} + {wv} * {})",
                        fa.at_fn(&format!("{node} + {}", off / 12))
                    );
                }
                code.push_str(&L::decl_float(&fl, &fexpr));
            }
            None => code.push_str(&L::decl_float(&fl, &format!("-{}", lit(data.reach)))),
        }
        let mut m = Vec::new();
        for t in 0..12i64 {
            let mut expr = "0.0".to_owned();
            for (wv, off) in &w {
                expr = format!(
                    "({expr} + {wv} * {})",
                    arr.at_fn(&format!("{base} + {}", off + t))
                );
            }
            let mv = self.next_var();
            code.push_str(&L::decl_float(&mv, &expr));
            m.push(mv);
        }
        let x = self.next_var();
        let row = |r: usize| {
            format!(
                "{} * {q}.x + {} * {q}.y + {} * {q}.z + {}",
                m[r * 4],
                m[r * 4 + 1],
                m[r * 4 + 2],
                m[r * 4 + 3]
            )
        };
        code.push_str(&L::decl_vec3(&x, &L::vec3_ctor(&row(0), &row(1), &row(2))));
        (x, e, bx, lam, fl)
    }

    /// Code for the rest point of `node` at `point_var` (the point the tree's leaves are evaluated at,
    /// following the branch a union / intersection selects by distance, the grid warp's map and the
    /// translate / rotate / scale transforms); the name of the vec3. Err for a node that moves points
    /// in another way (twist, bend, repetition, mirror, ...): no rest point is invented for it
    ///
    /// # Errors
    ///
    /// A node kind the rest point does not follow
    pub fn transpile_rest_inner(
        &mut self,
        node: &SdfNode,
        point_var: &str,
        code: &mut String,
    ) -> Result<String, String> {
        let pick = |t: &mut Self,
                    a: &SdfNode,
                    b: &SdfNode,
                    op: &str,
                    code: &mut String|
         -> Result<String, String> {
            let da = t.transpile_node_inner(a, point_var, code);
            let db = t.transpile_node_inner(b, point_var, code);
            let ra = t.transpile_rest_inner(a, point_var, code)?;
            let rb = t.transpile_rest_inner(b, point_var, code)?;
            let r = t.next_var();
            code.push_str(&L::decl_mut_vec3(&r, &rb));
            code.push_str(&format!("    if ({da} {op} {db}) {{ {r} = {ra}; }}\n"));
            Ok(r)
        };
        match node {
            SdfNode::Union { a, b }
            | SdfNode::SmoothUnion { a, b, .. }
            | SdfNode::ChamferUnion { a, b, .. }
            | SdfNode::StairsUnion { a, b, .. }
            | SdfNode::ColumnsUnion { a, b, .. }
            | SdfNode::ExpSmoothUnion { a, b, .. } => pick(self, a, b, "<=", code),
            SdfNode::Intersection { a, b }
            | SdfNode::SmoothIntersection { a, b, .. }
            | SdfNode::ChamferIntersection { a, b, .. }
            | SdfNode::StairsIntersection { a, b, .. }
            | SdfNode::ColumnsIntersection { a, b, .. }
            | SdfNode::ExpSmoothIntersection { a, b, .. } => pick(self, a, b, ">=", code),
            SdfNode::Subtraction { a, .. }
            | SdfNode::SmoothSubtraction { a, .. }
            | SdfNode::ChamferSubtraction { a, .. }
            | SdfNode::StairsSubtraction { a, .. }
            | SdfNode::ColumnsSubtraction { a, .. }
            | SdfNode::ExpSmoothSubtraction { a, .. } => {
                self.transpile_rest_inner(a, point_var, code)
            }
            SdfNode::SmoothBlend { a, b, mode, .. } => match mode {
                BlendMode::Union => pick(self, a, b, "<=", code),
                BlendMode::Intersection => pick(self, a, b, ">=", code),
                BlendMode::Subtraction => self.transpile_rest_inner(a, point_var, code),
            },
            SdfNode::WithMaterial { child, .. } => {
                self.transpile_rest_inner(child, point_var, code)
            }
            SdfNode::Translate { child, offset } => {
                let (o, new_p) = self.emit_rest_translate(*offset, point_var, code);
                let r = self.transpile_rest_inner(child, &new_p, code)?;
                let out = self.next_var();
                code.push_str(&L::decl_vec3(&out, &format!("{r} + {o}")));
                Ok(out)
            }
            SdfNode::Rotate { child, rotation } => {
                let new_p = self.emit_rest_rotate(rotation.inverse(), point_var, code);
                let r = self.transpile_rest_inner(child, &new_p, code)?;
                Ok(self.emit_rest_rotate(*rotation, &r, code))
            }
            SdfNode::Scale { child, factor } => {
                let new_p = self.next_var();
                let inv = self.param(1.0 / factor);
                code.push_str(&L::decl_vec3(&new_p, &format!("{point_var} * {inv}")));
                let r = self.transpile_rest_inner(child, &new_p, code)?;
                let out = self.next_var();
                let f = self.param(*factor);
                code.push_str(&L::decl_vec3(&out, &format!("{r} * {f}")));
                Ok(out)
            }
            SdfNode::Pose { child } => self.transpile_rest_posed(child, point_var, code),
            SdfNode::GridWarp { child, warp } => {
                let (x, _, _, _, _) = self.emit_grid_warp_point(warp, point_var, code);
                self.transpile_rest_inner(child, &x, code)
            }
            n if crate::diff::node_children(n).is_empty() => Ok(point_var.to_owned()),
            n => Err(format!(
                "rest point: {} moves points in a way the rest point does not follow",
                node_kind(n)
            )),
        }
    }

    /// The rest point below a [`SdfNode::Pose`]: its rigid motion (the translate right below it and
    /// the rotate right below that) is undone, not mapped back; Err for any other shape under a
    /// pose (see [`crate::eval::eval_rest_point`])
    fn transpile_rest_posed(
        &mut self,
        node: &SdfNode,
        point_var: &str,
        code: &mut String,
    ) -> Result<String, String> {
        if let SdfNode::Translate { child, offset } = node {
            if let SdfNode::Rotate { child, rotation } = child.as_ref() {
                let p = self.emit_rest_translate(*offset, point_var, code).1;
                let new_p = self.emit_rest_rotate(rotation.inverse(), &p, code);
                return self.transpile_rest_inner(child, &new_p, code);
            }
        }
        Err("rest point: a pose is a translate over a rotate over the part (as rigid_pose builds it)".to_owned())
    }

    /// `point − offset` for the rest point: (the offset vec3 expression, the new point's name)
    fn emit_rest_translate(
        &mut self,
        offset: glam::Vec3,
        point_var: &str,
        code: &mut String,
    ) -> (String, String) {
        let new_p = self.next_var();
        let (ox, oy, oz) = (
            self.param(offset.x),
            self.param(offset.y),
            self.param(offset.z),
        );
        let o = L::vec3_ctor(&ox, &oy, &oz);
        code.push_str(&L::decl_vec3(&new_p, &format!("{point_var} - {o}")));
        (o, new_p)
    }

    /// `q · point` for the rest point: the new point's name
    fn emit_rest_rotate(&mut self, q: glam::Quat, point_var: &str, code: &mut String) -> String {
        self.ensure_helper("quat_rotate");
        let new_p = self.next_var();
        let (qx, qy, qz, qw) = (
            self.param(q.x),
            self.param(q.y),
            self.param(q.z),
            self.param(q.w),
        );
        code.push_str(&L::decl_vec3(
            &new_p,
            &format!(
                "quat_rotate({}, {})",
                point_var,
                L::vec4_ctor(&qx, &qy, &qz, &qw)
            ),
        ));
        new_p
    }

    /// Transpile an SDF node tree into shader code, returning the body with a final `return`.
    pub fn transpile_node(&mut self, node: &SdfNode, point_var: &str) -> String {
        let mut code = String::new();
        let result_var = self.transpile_node_inner(node, point_var, &mut code);
        writeln!(code, "    return {};", result_var).unwrap();
        code
    }

    /// The main transpilation dispatcher - converts an SdfNode tree into shader code.
    ///
    /// This is the single source of truth for all shader languages.
    /// Language differences are handled via the `ShaderLang` trait methods.
    #[allow(clippy::too_many_lines)]
    pub fn transpile_node_inner(
        &mut self,
        node: &SdfNode,
        point_var: &str,
        code: &mut String,
    ) -> String {
        match node {
            // ============ Primitives ============
            SdfNode::Sphere { radius } => {
                let var = self.next_var();
                let r = self.param(*radius);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("length({}) - {}", point_var, r),
                ));
                var
            }

            SdfNode::Box3d { half_extents } => {
                let q_var = self.next_var();
                let var = self.next_var();
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                code.push_str(&L::decl_vec3(
                    &q_var,
                    &format!("abs({}) - {}", point_var, L::vec3_ctor(&hx, &hy, &hz)),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "length(max({}, {})) + min(max({}.x, max({}.y, {}.z)), 0.0)",
                        q_var,
                        L::vec3_zero(),
                        q_var,
                        q_var,
                        q_var
                    ),
                ));
                var
            }

            SdfNode::Cylinder {
                radius,
                half_height,
            } => {
                let d_var = self.next_var();
                let var = self.next_var();
                let r = self.param(*radius);
                let hh = self.param(*half_height);
                code.push_str(&L::decl_vec2(
                    &d_var,
                    &L::vec2_ctor(
                        &format!("length({}.xz) - {}", point_var, r),
                        &format!("abs({}.y) - {}", point_var, hh),
                    ),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "min(max({}.x, {}.y), 0.0) + length(max({}, {}))",
                        d_var,
                        d_var,
                        d_var,
                        L::vec2_zero()
                    ),
                ));
                var
            }

            SdfNode::Torus {
                major_radius,
                minor_radius,
            } => {
                let q_var = self.next_var();
                let var = self.next_var();
                let mr = self.param(*major_radius);
                let mnr = self.param(*minor_radius);
                code.push_str(&L::decl_vec2(
                    &q_var,
                    &L::vec2_ctor(
                        &format!("length({}.xz) - {}", point_var, mr),
                        &format!("{}.y", point_var),
                    ),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("length({}) - {}", q_var, mnr),
                ));
                var
            }

            SdfNode::Plane { normal, distance } => {
                let var = self.next_var();
                let nx = self.param(normal.x);
                let ny = self.param(normal.y);
                let nz = self.param(normal.z);
                let d = self.param(*distance);
                code.push_str(&L::decl_float(
                    &var,
                    // sdf_plane law: dot(p, n) - distance ("distance from origin")
                    &format!(
                        "dot({}, {}) - {}",
                        point_var,
                        L::vec3_ctor(&nx, &ny, &nz),
                        d
                    ),
                ));
                var
            }

            SdfNode::Capsule {
                point_a,
                point_b,
                radius,
            } => {
                let pax = self.param(point_a.x);
                let pay = self.param(point_a.y);
                let paz = self.param(point_a.z);
                let pbx = self.param(point_b.x);
                let pby = self.param(point_b.y);
                let pbz = self.param(point_b.z);

                // GLSL/HLSL re-emit point_a params for ba computation
                let (pax2, pay2, paz2) = if L::CAPSULE_RE_EMIT_PARAMS {
                    (
                        self.param(point_a.x),
                        self.param(point_a.y),
                        self.param(point_a.z),
                    )
                } else {
                    (pax.clone(), pay.clone(), paz.clone())
                };
                let r = self.param(*radius);

                let pa_var = self.next_var();
                let ba_var = self.next_var();
                let h_var = self.next_var();
                let var = self.next_var();

                code.push_str(&L::decl_vec3(
                    &pa_var,
                    &format!("{} - {}", point_var, L::vec3_ctor(&pax, &pay, &paz)),
                ));
                code.push_str(&L::decl_vec3(
                    &ba_var,
                    &format!(
                        "{} - {}",
                        L::vec3_ctor(&pbx, &pby, &pbz),
                        L::vec3_ctor(&pax2, &pay2, &paz2)
                    ),
                ));
                if L::CAPSULE_DEGENERATE_GUARD {
                    code.push_str(&L::decl_float(
                        &h_var,
                        &format!(
                            "clamp(dot({}, {}) / max(dot({}, {}), 1e-10), 0.0, 1.0)",
                            pa_var, ba_var, ba_var, ba_var
                        ),
                    ));
                } else {
                    code.push_str(&L::decl_float(
                        &h_var,
                        &format!(
                            "clamp(dot({}, {}) / dot({}, {}), 0.0, 1.0)",
                            pa_var, ba_var, ba_var, ba_var
                        ),
                    ));
                }
                code.push_str(&L::decl_float(
                    &var,
                    &format!("length({} - {} * {}) - {}", pa_var, ba_var, h_var, r),
                ));
                var
            }

            SdfNode::Cone {
                radius,
                half_height,
            } => {
                let k2x = -radius;
                let k2y = 2.0 * half_height;
                let p_hh = self.param(*half_height);
                let p_r = self.param(*radius);
                let p_k2x = self.param(k2x);
                let p_k2y = self.param(k2y);
                let p_k2sq = self.param(k2x * k2x + (k2y * k2y));

                let qx_var = self.next_var();
                let h_var = self.next_var();
                let ca_var = self.next_var();
                let t_var = self.next_var();
                let cb_var = self.next_var();
                let s_var = self.next_var();
                let d2_var = self.next_var();
                let var = self.next_var();

                code.push_str(&L::decl_float(
                    &qx_var,
                    &format!("length({}.xz)", point_var),
                ));
                code.push_str(&L::decl_float(&h_var, &p_hh));
                // ca = vec2(qx - min(qx, select(0.0, r, p.y<0.0)), abs(p.y) - h)
                let cone_select = L::select_expr(&format!("{}.y < 0.0", point_var), &p_r, "0.0");
                code.push_str(&L::decl_vec2(
                    &ca_var,
                    &L::vec2_ctor(
                        &format!("{} - min({}, {})", qx_var, qx_var, cone_select),
                        &format!("abs({}.y) - {}", point_var, h_var),
                    ),
                ));
                code.push_str(&L::decl_float(
                    &t_var,
                    &format!(
                        "clamp((-{} * {} + ({} - {}.y) * {}) / {}, 0.0, 1.0)",
                        qx_var, p_k2x, h_var, point_var, p_k2y, p_k2sq
                    ),
                ));
                code.push_str(&L::decl_vec2(
                    &cb_var,
                    &L::vec2_ctor(
                        &format!("{} + {} * {}", qx_var, p_k2x, t_var),
                        &format!("{}.y - {} + {} * {}", point_var, h_var, p_k2y, t_var),
                    ),
                ));
                let cone_sign = L::select_expr(
                    &format!("{}.x < 0.0 && {}.y < 0.0", cb_var, ca_var),
                    "-1.0",
                    "1.0",
                );
                code.push_str(&L::decl_float(&s_var, &cone_sign));
                code.push_str(&L::decl_float(
                    &d2_var,
                    &format!(
                        "min(dot({}, {}), dot({}, {}))",
                        ca_var, ca_var, cb_var, cb_var
                    ),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{} * sqrt({})", s_var, d2_var),
                ));
                var
            }

            // Exact ellipsoid (Eberly nearest point), one helper per language
            SdfNode::Ellipsoid { radii } => {
                self.ensure_helper("sdf_ellipsoid");
                let var = self.next_var();
                let rx = self.param(radii.x);
                let ry = self.param(radii.y);
                let rz = self.param(radii.z);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_ellipsoid({}, {})",
                        point_var,
                        L::vec3_ctor(&rx, &ry, &rz)
                    ),
                ));
                var
            }

            SdfNode::RoundedCone {
                r1,
                r2,
                half_height,
            } => {
                self.ensure_helper("sdf_rounded_cone");
                let var = self.next_var();
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_rounded_cone({}, {}, {}, {})",
                        point_var, p_r1, p_r2, p_hh
                    ),
                ));
                var
            }

            SdfNode::Pyramid { half_height } => {
                self.ensure_helper("sdf_pyramid");
                let var = self.next_var();
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_pyramid({}, {})", point_var, p_hh),
                ));
                var
            }

            SdfNode::Octahedron { size } => {
                self.ensure_helper("sdf_octahedron");
                let var = self.next_var();
                let p_s = self.param(*size);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_octahedron({}, {})", point_var, p_s),
                ));
                var
            }

            SdfNode::HexPrism {
                hex_radius,
                half_height,
            } => {
                self.ensure_helper("sdf_hex_prism");
                let var = self.next_var();
                let p_hr = self.param(*hex_radius);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_hex_prism({}, {}, {})", point_var, p_hr, p_hh),
                ));
                var
            }

            SdfNode::Link {
                half_length,
                r1,
                r2,
            } => {
                self.ensure_helper("sdf_link");
                let var = self.next_var();
                let p_hl = self.param(*half_length);
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_link({}, {}, {}, {})", point_var, p_hl, p_r1, p_r2),
                ));
                var
            }

            SdfNode::Triangle {
                point_a,
                point_b,
                point_c,
            } => {
                self.ensure_helper("sdf_triangle");
                let var = self.next_var();
                let ax = self.param(point_a.x);
                let ay = self.param(point_a.y);
                let az = self.param(point_a.z);
                let bx = self.param(point_b.x);
                let by = self.param(point_b.y);
                let bz = self.param(point_b.z);
                let cx = self.param(point_c.x);
                let cy = self.param(point_c.y);
                let cz = self.param(point_c.z);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_triangle({}, {}, {}, {})",
                        point_var,
                        L::vec3_ctor(&ax, &ay, &az),
                        L::vec3_ctor(&bx, &by, &bz),
                        L::vec3_ctor(&cx, &cy, &cz),
                    ),
                ));
                var
            }

            SdfNode::Bezier {
                point_a,
                point_b,
                point_c,
                radius,
            } => {
                self.ensure_helper("sdf_bezier");
                let var = self.next_var();
                let p0x = self.param(point_a.x);
                let p0y = self.param(point_a.y);
                let p0z = self.param(point_a.z);
                let p1x = self.param(point_b.x);
                let p1y = self.param(point_b.y);
                let p1z = self.param(point_b.z);
                let p2x = self.param(point_c.x);
                let p2y = self.param(point_c.y);
                let p2z = self.param(point_c.z);
                let r = self.param(*radius);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_bezier({}, {}, {}, {}, {})",
                        point_var,
                        L::vec3_ctor(&p0x, &p0y, &p0z),
                        L::vec3_ctor(&p1x, &p1y, &p1z),
                        L::vec3_ctor(&p2x, &p2y, &p2z),
                        r,
                    ),
                ));
                var
            }

            SdfNode::RoundedBox {
                half_extents,
                round_radius,
            } => {
                let q_var = self.next_var();
                let var = self.next_var();
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                let r = self.param(*round_radius);
                code.push_str(&L::decl_vec3(
                    &q_var,
                    &format!("abs({}) - {}", point_var, L::vec3_ctor(&hx, &hy, &hz)),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "length(max({}, {})) + min(max({}.x, max({}.y, {}.z)), 0.0) - {}",
                        q_var,
                        L::vec3_zero(),
                        q_var,
                        q_var,
                        q_var,
                        r
                    ),
                ));
                var
            }

            SdfNode::CappedCone {
                half_height,
                r1,
                r2,
            } => {
                self.ensure_helper("sdf_capped_cone");
                let var = self.next_var();
                let p_hh = self.param(*half_height);
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_capped_cone({}, {}, {}, {})",
                        point_var, p_hh, p_r1, p_r2
                    ),
                ));
                var
            }

            SdfNode::CappedTorus {
                major_radius,
                minor_radius,
                cap_angle,
            } => {
                self.ensure_helper("sdf_capped_torus");
                let var = self.next_var();
                let p_rm = self.param(*major_radius);
                let p_rn = self.param(*minor_radius);
                let p_an = self.param(*cap_angle);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_capped_torus({}, {}, {}, {})",
                        point_var, p_rm, p_rn, p_an
                    ),
                ));
                var
            }

            SdfNode::RoundedCylinder {
                radius,
                round_radius,
                half_height,
            } => {
                self.ensure_helper("sdf_rounded_cylinder");
                let var = self.next_var();
                let p_r = self.param(*radius);
                let p_rr = self.param(*round_radius);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_rounded_cylinder({}, {}, {}, {})",
                        point_var, p_r, p_rr, p_hh
                    ),
                ));
                var
            }

            SdfNode::TriangularPrism { width, half_depth } => {
                self.ensure_helper("sdf_triangular_prism");
                let var = self.next_var();
                let p_w = self.param(*width);
                let p_hd = self.param(*half_depth);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_triangular_prism({}, {}, {})", point_var, p_w, p_hd),
                ));
                var
            }

            SdfNode::CutSphere { radius, cut_height } => {
                self.ensure_helper("sdf_cut_sphere");
                let var = self.next_var();
                let p_r = self.param(*radius);
                let p_ch = self.param(*cut_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_cut_sphere({}, {}, {})", point_var, p_r, p_ch),
                ));
                var
            }

            SdfNode::CutHollowSphere {
                radius,
                cut_height,
                thickness,
            } => {
                self.ensure_helper("sdf_cut_hollow_sphere");
                let var = self.next_var();
                let p_r = self.param(*radius);
                let p_ch = self.param(*cut_height);
                let p_t = self.param(*thickness);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_cut_hollow_sphere({}, {}, {}, {})",
                        point_var, p_r, p_ch, p_t
                    ),
                ));
                var
            }

            SdfNode::DeathStar { ra, rb, d } => {
                self.ensure_helper("sdf_death_star");
                let var = self.next_var();
                let p_ra = self.param(*ra);
                let p_rb = self.param(*rb);
                let p_d = self.param(*d);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_death_star({}, {}, {}, {})", point_var, p_ra, p_rb, p_d),
                ));
                var
            }

            SdfNode::SolidAngle { angle, radius } => {
                self.ensure_helper("sdf_solid_angle");
                let var = self.next_var();
                let p_an = self.param(*angle);
                let p_r = self.param(*radius);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_solid_angle({}, {}, {})", point_var, p_an, p_r),
                ));
                var
            }

            SdfNode::Rhombus {
                la,
                lb,
                half_height,
                round_radius,
            } => {
                self.ensure_helper("sdf_rhombus");
                let var = self.next_var();
                let p_la = self.param(*la);
                let p_lb = self.param(*lb);
                let p_h = self.param(*half_height);
                let p_ra = self.param(*round_radius);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_rhombus({}, {}, {}, {}, {})",
                        point_var, p_la, p_lb, p_h, p_ra
                    ),
                ));
                var
            }

            SdfNode::Horseshoe {
                angle,
                radius,
                half_length,
                width,
                thickness,
            } => {
                self.ensure_helper("sdf_horseshoe");
                let var = self.next_var();
                let p_an = self.param(*angle);
                let p_r = self.param(*radius);
                let p_le = self.param(*half_length);
                let p_w = self.param(*width);
                let p_t = self.param(*thickness);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_horseshoe({}, {}, {}, {}, {}, {})",
                        point_var, p_an, p_r, p_le, p_w, p_t
                    ),
                ));
                var
            }

            SdfNode::Vesica { radius, half_dist } => {
                self.ensure_helper("sdf_vesica");
                let var = self.next_var();
                let p_r = self.param(*radius);
                let p_hd = self.param(*half_dist);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_vesica({}, {}, {})", point_var, p_r, p_hd),
                ));
                var
            }

            SdfNode::InfiniteCylinder { radius } => {
                let var = self.next_var();
                let r = self.param(*radius);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("length({}.xz) - {}", point_var, r),
                ));
                var
            }

            SdfNode::InfiniteCone { angle } => {
                self.ensure_helper("sdf_infinite_cone");
                let var = self.next_var();
                let p_an = self.param(*angle);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_infinite_cone({}, {})", point_var, p_an),
                ));
                var
            }

            SdfNode::Gyroid { scale, thickness } => {
                let var = self.next_var();
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, sc)));
                code.push_str(&L::decl_float(&var, &format!(
                    "abs(sin({sp}.x)*cos({sp}.y) + sin({sp}.y)*cos({sp}.z) + sin({sp}.z)*cos({sp}.x)) / {sc} - {th}",
                    sp = sp, sc = sc, th = th
                )));
                var
            }

            SdfNode::MetricBall {
                radius,
                w_l1,
                w_l2,
                w_linf,
            } => {
                // Same left-to-right sum as `sdf_metric_ball` / the VM law.
                let var = self.next_var();
                let p_r = self.param(*radius);
                let p_w1 = self.param(*w_l1);
                let p_w2 = self.param(*w_l2);
                let p_wi = self.param(*w_linf);
                let a = self.next_var();
                code.push_str(&L::decl_vec3(&a, &format!("abs({point_var})")));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "{w1} * ({a}.x + {a}.y + {a}.z) + {w2} * length({p}) \
                         + {wi} * max(max({a}.x, {a}.y), {a}.z) - {r}",
                        w1 = p_w1,
                        w2 = p_w2,
                        wi = p_wi,
                        a = a,
                        p = point_var,
                        r = p_r
                    ),
                ));
                var
            }

            SdfNode::MetricBlend {
                inner,
                outer,
                center,
                radius,
                skin,
            } => {
                let d_in = self.transpile_node_inner(inner, point_var, code);
                let d_out = self.transpile_node_inner(outer, point_var, code);
                let p_cx = self.param(center.x);
                let p_cy = self.param(center.y);
                let p_cz = self.param(center.z);
                let p_r = self.param(*radius);
                let p_w = self.param(*skin);
                // The cubic is written out rather than calling the builtin
                // `smoothstep`, whose last ulp is not specified across
                // drivers; this is the same expression the CPU law uses.
                // `skin` must be positive here — the CPU treats a zero skin
                // as a step, which this expression cannot reproduce.
                let len = self.next_var();
                code.push_str(&L::decl_float(
                    &len,
                    &format!(
                        "length({p} - {ctor})",
                        p = point_var,
                        ctor = L::vec3_ctor(&p_cx, &p_cy, &p_cz)
                    ),
                ));
                let tt = self.next_var();
                code.push_str(&L::decl_float(
                    &tt,
                    &format!(
                        "clamp(({len} - {r}) / {w}, 0.0, 1.0)",
                        len = len,
                        r = p_r,
                        w = p_w
                    ),
                ));
                let t = self.next_var();
                code.push_str(&L::decl_float(
                    &t,
                    &format!("{tt} * {tt} * (3.0 - 2.0 * {tt})", tt = tt),
                ));
                let var = self.next_var();
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "(1.0 - {t}) * {din} + {t} * {dout}",
                        t = t,
                        din = d_in,
                        dout = d_out
                    ),
                ));
                var
            }

            SdfNode::Heart { size } => {
                self.ensure_helper("sdf_heart");
                let var = self.next_var();
                let p_s = self.param(*size);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_heart({}, {})", point_var, p_s),
                ));
                var
            }

            SdfNode::Tube {
                outer_radius,
                thickness,
                half_height,
            } => {
                self.ensure_helper("sdf_tube");
                let var = self.next_var();
                let p_or = self.param(*outer_radius);
                let p_th = self.param(*thickness);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_tube({}, {}, {}, {})", point_var, p_or, p_th, p_hh),
                ));
                var
            }

            SdfNode::Barrel {
                radius,
                half_height,
                bulge,
            } => {
                self.ensure_helper("sdf_barrel");
                let var = self.next_var();
                let p_r = self.param(*radius);
                let p_hh = self.param(*half_height);
                let p_b = self.param(*bulge);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_barrel({}, {}, {}, {})", point_var, p_r, p_hh, p_b),
                ));
                var
            }

            SdfNode::Diamond {
                radius,
                half_height,
            } => {
                self.ensure_helper("sdf_diamond");
                let var = self.next_var();
                let p_r = self.param(*radius);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_diamond({}, {}, {})", point_var, p_r, p_hh),
                ));
                var
            }

            SdfNode::ChamferedCube {
                half_extents,
                chamfer,
            } => {
                self.ensure_helper("sdf_chamfered_cube");
                let var = self.next_var();
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                let ch = self.param(*chamfer);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_chamfered_cube({}, {}, {}, {}, {})",
                        point_var, hx, hy, hz, ch
                    ),
                ));
                var
            }

            SdfNode::SchwarzP { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, sc)));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "abs(cos({sp}.x) + cos({sp}.y) + cos({sp}.z)) / {sc} - {th}",
                        sp = sp,
                        sc = sc,
                        th = th
                    ),
                ));
                var
            }

            SdfNode::Superellipsoid {
                half_extents,
                e1,
                e2,
            } => {
                self.ensure_helper("sdf_superellipsoid");
                let var = self.next_var();
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                let p_e1 = self.param(*e1);
                let p_e2 = self.param(*e2);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_superellipsoid({}, {}, {}, {}, {}, {})",
                        point_var, hx, hy, hz, p_e1, p_e2
                    ),
                ));
                var
            }

            SdfNode::RoundedX {
                width,
                round_radius,
                half_height,
            } => {
                self.ensure_helper("sdf_rounded_x");
                let var = self.next_var();
                let p_w = self.param(*width);
                let p_r = self.param(*round_radius);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_rounded_x({}, {}, {}, {})", point_var, p_w, p_r, p_hh),
                ));
                var
            }

            SdfNode::Pie {
                angle,
                radius,
                half_height,
            } => {
                self.ensure_helper("sdf_pie");
                let var = self.next_var();
                let p_an = self.param(*angle);
                let p_r = self.param(*radius);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_pie({}, {}, {}, {})", point_var, p_an, p_r, p_hh),
                ));
                var
            }

            SdfNode::Trapezoid {
                r1,
                r2,
                trap_height,
                half_depth,
            } => {
                self.ensure_helper("sdf_trapezoid");
                let var = self.next_var();
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                let p_hh = self.param(*trap_height);
                let p_hd = self.param(*half_depth);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_trapezoid({}, {}, {}, {}, {})",
                        point_var, p_r1, p_r2, p_hh, p_hd
                    ),
                ));
                var
            }

            SdfNode::Parallelogram {
                width,
                para_height,
                skew,
                half_depth,
            } => {
                self.ensure_helper("sdf_parallelogram");
                let var = self.next_var();
                let p_w = self.param(*width);
                let p_hh = self.param(*para_height);
                let p_sk = self.param(*skew);
                let p_hd = self.param(*half_depth);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_parallelogram({}, {}, {}, {}, {})",
                        point_var, p_w, p_hh, p_sk, p_hd
                    ),
                ));
                var
            }

            SdfNode::Tunnel {
                width,
                height_2d,
                half_depth,
            } => {
                self.ensure_helper("sdf_tunnel");
                let var = self.next_var();
                let p_w = self.param(*width);
                let p_h2d = self.param(*height_2d);
                let p_hd = self.param(*half_depth);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_tunnel({}, {}, {}, {})", point_var, p_w, p_h2d, p_hd),
                ));
                var
            }

            SdfNode::UnevenCapsule {
                r1,
                r2,
                cap_height,
                half_depth,
            } => {
                self.ensure_helper("sdf_uneven_capsule");
                let var = self.next_var();
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                let p_chh = self.param(*cap_height);
                let p_hd = self.param(*half_depth);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_uneven_capsule({}, {}, {}, {}, {})",
                        point_var, p_r1, p_r2, p_chh, p_hd
                    ),
                ));
                var
            }

            SdfNode::Egg { ra, rb } => {
                self.ensure_helper("sdf_egg");
                let var = self.next_var();
                let p_ra = self.param(*ra);
                let p_rb = self.param(*rb);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_egg({}, {}, {})", point_var, p_ra, p_rb),
                ));
                var
            }

            SdfNode::ArcShape {
                aperture,
                radius,
                thickness,
                half_height,
            } => {
                self.ensure_helper("sdf_arc_shape");
                let var = self.next_var();
                let p_ap = self.param(*aperture);
                let p_r = self.param(*radius);
                let p_th = self.param(*thickness);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_arc_shape({}, {}, {}, {}, {})",
                        point_var, p_ap, p_r, p_th, p_hh
                    ),
                ));
                var
            }

            SdfNode::Moon {
                d,
                ra,
                rb,
                half_height,
            } => {
                self.ensure_helper("sdf_moon");
                let var = self.next_var();
                let p_d = self.param(*d);
                let p_ra = self.param(*ra);
                let p_rb = self.param(*rb);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_moon({}, {}, {}, {}, {})",
                        point_var, p_d, p_ra, p_rb, p_hh
                    ),
                ));
                var
            }

            SdfNode::CrossShape {
                length,
                thickness,
                round_radius,
                half_height,
            } => {
                self.ensure_helper("sdf_cross_shape");
                let var = self.next_var();
                let p_l = self.param(*length);
                let p_th = self.param(*thickness);
                let p_rr = self.param(*round_radius);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_cross_shape({}, {}, {}, {}, {})",
                        point_var, p_l, p_th, p_rr, p_hh
                    ),
                ));
                var
            }

            SdfNode::BlobbyCross { size, half_height } => {
                self.ensure_helper("sdf_blobby_cross");
                let var = self.next_var();
                let p_s = self.param(*size);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_blobby_cross({}, {}, {})", point_var, p_s, p_hh),
                ));
                var
            }

            SdfNode::ParabolaSegment {
                width,
                para_height,
                half_depth,
            } => {
                self.ensure_helper("sdf_parabola_segment");
                let var = self.next_var();
                let p_w = self.param(*width);
                let p_ph = self.param(*para_height);
                let p_hd = self.param(*half_depth);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_parabola_segment({}, {}, {}, {})",
                        point_var, p_w, p_ph, p_hd
                    ),
                ));
                var
            }

            SdfNode::RegularPolygon {
                radius,
                n_sides,
                half_height,
            } => {
                self.ensure_helper("sdf_regular_polygon");
                let var = self.next_var();
                let p_r = self.param(*radius);
                let p_n = self.param(*n_sides);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_regular_polygon({}, {}, {}, {})",
                        point_var, p_r, p_n, p_hh
                    ),
                ));
                var
            }

            SdfNode::StarPolygon {
                radius,
                n_points,
                m,
                half_height,
            } => {
                self.ensure_helper("sdf_star_polygon");
                let var = self.next_var();
                let p_r = self.param(*radius);
                let p_n = self.param(*n_points);
                let p_m = self.param(*m);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_star_polygon({}, {}, {}, {}, {})",
                        point_var, p_r, p_n, p_m, p_hh
                    ),
                ));
                var
            }

            SdfNode::Stairs {
                step_width,
                step_height,
                n_steps,
                half_depth,
            } => {
                self.ensure_helper("sdf_stairs");
                let var = self.next_var();
                let p_sw = self.param(*step_width);
                let p_sh = self.param(*step_height);
                let p_ns = self.param(*n_steps);
                let p_hd = self.param(*half_depth);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_stairs({}, {}, {}, {}, {})",
                        point_var, p_sw, p_sh, p_ns, p_hd
                    ),
                ));
                var
            }

            SdfNode::Helix {
                major_r,
                minor_r,
                pitch,
                half_height,
            } => {
                self.ensure_helper("sdf_helix");
                let var = self.next_var();
                let p_mr = self.param(*major_r);
                let p_mnr = self.param(*minor_r);
                let p_p = self.param(*pitch);
                let p_hh = self.param(*half_height);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "sdf_helix({}, {}, {}, {}, {})",
                        point_var, p_mr, p_mnr, p_p, p_hh
                    ),
                ));
                var
            }

            // Polyhedra (no dedicated helpers - inline approximation)
            SdfNode::Tetrahedron { size } => {
                self.ensure_helper("sdf_tetrahedron");
                let var = self.next_var();
                let p_s = self.param(*size);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_tetrahedron({}, {})", point_var, p_s),
                ));
                var
            }

            SdfNode::Dodecahedron { radius } => {
                self.ensure_helper("sdf_dodecahedron");
                let var = self.next_var();
                let p_r = self.param(*radius);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_dodecahedron({}, {})", point_var, p_r),
                ));
                var
            }

            SdfNode::Icosahedron { radius } => {
                self.ensure_helper("sdf_icosahedron");
                let var = self.next_var();
                let p_r = self.param(*radius);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_icosahedron({}, {})", point_var, p_r),
                ));
                var
            }

            SdfNode::TruncatedOctahedron { radius } => {
                self.ensure_helper("sdf_truncated_octahedron");
                let var = self.next_var();
                let p_r = self.param(*radius);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_truncated_octahedron({}, {})", point_var, p_r),
                ));
                var
            }

            SdfNode::TruncatedIcosahedron { radius } => {
                self.ensure_helper("sdf_truncated_icosahedron");
                let var = self.next_var();
                let p_r = self.param(*radius);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("sdf_truncated_icosahedron({}, {})", point_var, p_r),
                ));
                var
            }

            SdfNode::BoxFrame { half_extents, edge } => {
                let var = self.next_var();
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                let e = self.param(*edge);
                let p = self.next_var();
                let q = self.next_var();
                code.push_str(&L::decl_vec3(
                    &p,
                    &format!("abs({}) - {}", point_var, L::vec3_ctor(&hx, &hy, &hz)),
                ));
                code.push_str(&L::decl_vec3(&q, &format!("abs({} + {}) - {}", p, e, e)));
                let z = L::vec3_zero();
                let v1 = L::vec3_ctor(
                    &format!("{}.x", p),
                    &format!("{}.y", q),
                    &format!("{}.z", q),
                );
                let v2 = L::vec3_ctor(
                    &format!("{}.x", q),
                    &format!("{}.y", p),
                    &format!("{}.z", q),
                );
                let v3 = L::vec3_ctor(
                    &format!("{}.x", q),
                    &format!("{}.y", q),
                    &format!("{}.z", p),
                );
                let expr = format!(
                    "min(min(\
                    length(max({v1}, {z})) + min(max({p}.x, max({q}.y, {q}.z)), 0.0), \
                    length(max({v2}, {z})) + min(max({q}.x, max({p}.y, {q}.z)), 0.0)), \
                    length(max({v3}, {z})) + min(max({q}.x, max({q}.y, {p}.z)), 0.0))",
                    v1 = v1,
                    v2 = v2,
                    v3 = v3,
                    z = z,
                    p = p,
                    q = q,
                );
                code.push_str(&L::decl_float(&var, &expr));
                var
            }

            // TPMS surfaces
            SdfNode::DiamondSurface { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, sc)));
                code.push_str(&L::decl_float(&var, &format!(
                    "abs(sin({sp}.x)*sin({sp}.y)*sin({sp}.z) + sin({sp}.x)*cos({sp}.y)*cos({sp}.z) + cos({sp}.x)*sin({sp}.y)*cos({sp}.z) + cos({sp}.x)*cos({sp}.y)*sin({sp}.z)) / {sc} - {th}",
                    sp = sp, sc = sc, th = th
                )));
                var
            }

            SdfNode::Neovius { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, sc)));
                code.push_str(&L::decl_float(&var, &format!(
                    "abs(3.0*(cos({sp}.x) + cos({sp}.y) + cos({sp}.z)) + 4.0*cos({sp}.x)*cos({sp}.y)*cos({sp}.z)) / {sc} - {th}",
                    sp = sp, sc = sc, th = th
                )));
                var
            }

            SdfNode::Lidinoid { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, sc)));
                code.push_str(&L::decl_float(&var, &format!(
                    // `sdf_lidinoid` on the CPU: 0.5 (sin2x cos y sin z + sin x sin2y cos z + cos x sin y sin2z) - 0.5 (cos2x cos2y + cos2y cos2z + cos2z cos2x) + 0.15
                    "abs(0.5 * (sin(2.0*{sp}.x)*cos({sp}.y)*sin({sp}.z) + sin({sp}.x)*sin(2.0*{sp}.y)*cos({sp}.z) + cos({sp}.x)*sin({sp}.y)*sin(2.0*{sp}.z)) - 0.5 * (cos(2.0*{sp}.x)*cos(2.0*{sp}.y) + cos(2.0*{sp}.y)*cos(2.0*{sp}.z) + cos(2.0*{sp}.z)*cos(2.0*{sp}.x)) + 0.15) / {sc} - {th}",
                    sp = sp, sc = sc, th = th
                )));
                var
            }

            SdfNode::IWP { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, sc)));
                code.push_str(&L::decl_float(&var, &format!(
                    // `sdf_iwp` on the CPU: 2 (cx cy + cy cz + cz cx) - (cos2x + cos2y + cos2z)
                    "abs(2.0 * (cos({sp}.x)*cos({sp}.y) + cos({sp}.y)*cos({sp}.z) + cos({sp}.z)*cos({sp}.x)) - (cos(2.0*{sp}.x) + cos(2.0*{sp}.y) + cos(2.0*{sp}.z))) / {sc} - {th}",
                    sp = sp, sc = sc, th = th
                )));
                var
            }

            SdfNode::FRD { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, sc)));
                code.push_str(&L::decl_float(&var, &format!(
                    // `sdf_frd` on the CPU: sx cy cos2z + cos2x sy cz + cx cos2y sz
                    "abs(sin({sp}.x)*cos({sp}.y)*cos(2.0*{sp}.z) + cos(2.0*{sp}.x)*sin({sp}.y)*cos({sp}.z) + cos({sp}.x)*cos(2.0*{sp}.y)*sin({sp}.z)) / {sc} - {th}",
                    sp = sp, sc = sc, th = th
                )));
                var
            }

            SdfNode::FischerKochS { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, sc)));
                code.push_str(&L::decl_float(&var, &format!(
                    "abs(cos(2.0*{sp}.x)*sin({sp}.y)*cos({sp}.z) + cos({sp}.x)*cos(2.0*{sp}.y)*sin({sp}.z) + sin({sp}.x)*cos({sp}.y)*cos(2.0*{sp}.z) - 0.4) / {sc} - {th}",
                    sp = sp, sc = sc, th = th
                )));
                var
            }

            SdfNode::PMY { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, sc)));
                code.push_str(&L::decl_float(&var, &format!(
                    "abs(2.0*cos({sp}.x)*cos({sp}.y)*cos({sp}.z) + sin(2.0*{sp}.x)*sin({sp}.y) + sin({sp}.x)*sin(2.0*{sp}.z) + sin(2.0*{sp}.y)*sin({sp}.z)) / {sc} - {th}",
                    sp = sp, sc = sc, th = th
                )));
                var
            }

            // ============ Boolean Operations ============
            SdfNode::Union { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("min({}, {})", d_a, d_b)));
                var
            }

            SdfNode::Intersection { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("max({}, {})", d_a, d_b)));
                var
            }

            SdfNode::Subtraction { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("max({}, -{})", d_a, d_b)));
                var
            }

            // Division Exorcism for SmoothUnion
            SdfNode::SmoothUnion { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && k.abs() < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("min({}, {})", d_a, d_b)));
                    return var;
                }
                let k_str = self.param(*k);
                let inv_k_str = self.param(1.0 / k);
                let h_var = self.next_var();
                code.push_str(&L::decl_float(
                    &h_var,
                    &format!("max(1.0 - abs({} - {}) * {}, 0.0)", d_a, d_b, inv_k_str),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "min({}, {}) - {} * {} * ({} * 0.25)",
                        d_a, d_b, h_var, h_var, k_str
                    ),
                ));
                var
            }

            SdfNode::SmoothIntersection { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && k.abs() < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("max({}, {})", d_a, d_b)));
                    return var;
                }
                let k_str = self.param(*k);
                let inv_k_str = self.param(1.0 / k);
                let h_var = self.next_var();
                code.push_str(&L::decl_float(
                    &h_var,
                    &format!("max(1.0 - abs({} - {}) * {}, 0.0)", d_a, d_b, inv_k_str),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "max({}, {}) + {} * {} * ({} * 0.25)",
                        d_a, d_b, h_var, h_var, k_str
                    ),
                ));
                var
            }

            SdfNode::SmoothSubtraction { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && k.abs() < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("max({}, -{})", d_a, d_b)));
                    return var;
                }
                let k_str = self.param(*k);
                let inv_k_str = self.param(1.0 / k);
                let h_var = self.next_var();
                let neg_b = self.next_var();
                code.push_str(&L::decl_float(&neg_b, &format!("-{}", d_b)));
                code.push_str(&L::decl_float(
                    &h_var,
                    &format!("max(1.0 - abs({} - {}) * {}, 0.0)", d_a, neg_b, inv_k_str),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "max({}, {}) + {} * {} * ({} * 0.25)",
                        d_a, neg_b, h_var, h_var, k_str
                    ),
                ));
                var
            }

            // Chamfer blends
            SdfNode::ChamferUnion { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && *r < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("min({}, {})", d_a, d_b)));
                    return var;
                }
                let r_str = self.param(*r);
                let s_str = self.param(std::f32::consts::FRAC_1_SQRT_2);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "min(min({}, {}), ({} + {}) * {} - {})",
                        d_a, d_b, d_a, d_b, s_str, r_str
                    ),
                ));
                var
            }

            SdfNode::ChamferIntersection { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && *r < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("max({}, {})", d_a, d_b)));
                    return var;
                }
                let r_str = self.param(*r);
                let s_str = self.param(std::f32::consts::FRAC_1_SQRT_2);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "max(max({}, {}), ({} + {}) * {} + {})",
                        d_a, d_b, d_a, d_b, s_str, r_str
                    ),
                ));
                var
            }

            SdfNode::ChamferSubtraction { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && *r < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("max({}, -{})", d_a, d_b)));
                    return var;
                }
                let r_str = self.param(*r);
                let s_str = self.param(std::f32::consts::FRAC_1_SQRT_2);
                let neg_b = self.next_var();
                code.push_str(&L::decl_float(&neg_b, &format!("-{}", d_b)));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "max(max({}, {}), ({} + {}) * {} + {})",
                        d_a, neg_b, d_a, neg_b, s_str, r_str
                    ),
                ));
                var
            }

            // Stairs blends
            SdfNode::StairsUnion { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && *r < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("min({}, {})", d_a, d_b)));
                    return var;
                }
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                self.emit_stairs_union_inline(code, &d_a, &d_b, &r_s, &n_s, &var);
                var
            }

            SdfNode::StairsIntersection { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && *r < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("max({}, {})", d_a, d_b)));
                    return var;
                }
                let na = self.next_var();
                let nb = self.next_var();
                code.push_str(&L::decl_float(&na, &format!("-{}", d_a)));
                code.push_str(&L::decl_float(&nb, &format!("-{}", d_b)));
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                let su = self.next_var();
                self.emit_stairs_union_inline(code, &na, &nb, &r_s, &n_s, &su);
                code.push_str(&L::decl_float(&var, &format!("-{}", su)));
                var
            }

            SdfNode::StairsSubtraction { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && *r < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("max({}, -{})", d_a, d_b)));
                    return var;
                }
                let na = self.next_var();
                code.push_str(&L::decl_float(&na, &format!("-{}", d_a)));
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                let su = self.next_var();
                self.emit_stairs_union_inline(code, &na, &d_b, &r_s, &n_s, &su);
                code.push_str(&L::decl_float(&var, &format!("-{}", su)));
                var
            }

            SdfNode::XOR { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(
                    &var,
                    &format!("max(min({}, {}), -max({}, {}))", d_a, d_b, d_a, d_b),
                ));
                var
            }

            SdfNode::Morph { a, b, t } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let t_s = self.param(*t);
                code.push_str(&L::decl_float(&var, &L::mix_expr(&d_a, &d_b, &t_s)));
                var
            }

            // Columns operations: one helper per language, same law as the CPU
            SdfNode::ColumnsUnion { a, b, r, n } => {
                self.ensure_helper("columns");
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("alice_columns_union({d_a}, {d_b}, {r_s}, {n_s})"),
                ));
                var
            }
            SdfNode::ColumnsIntersection { a, b, r, n } => {
                self.ensure_helper("columns");
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("alice_columns_intersection({d_a}, {d_b}, {r_s}, {n_s})"),
                ));
                var
            }
            SdfNode::ColumnsSubtraction { a, b, r, n } => {
                self.ensure_helper("columns");
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("alice_columns_subtraction({d_a}, {d_b}, {r_s}, {n_s})"),
                ));
                var
            }
            SdfNode::Pipe { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                code.push_str(&L::decl_float(
                    &var,
                    &format!("length({}) - {}", L::vec2_ctor(&d_a, &d_b), r_s),
                ));
                var
            }

            SdfNode::Engrave { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "max({}, ({} + {} - abs({})) * 0.70710678)",
                        d_a, d_a, r_s, d_b
                    ),
                ));
                var
            }

            SdfNode::Groove { a, b, ra, rb } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let ra_s = self.param(*ra);
                let rb_s = self.param(*rb);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "max({}, min({} + {}, {} - abs({})))",
                        d_a, d_a, ra_s, rb_s, d_b
                    ),
                ));
                var
            }

            SdfNode::Tongue { a, b, ra, rb } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let ra_s = self.param(*ra);
                let rb_s = self.param(*rb);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "min({}, max({} - {}, abs({}) - {}))",
                        d_a, d_a, ra_s, d_b, rb_s
                    ),
                ));
                var
            }

            // ============ Transforms ============
            SdfNode::Translate { child, offset } => {
                let new_p = self.next_var();
                let ox = self.param(offset.x);
                let oy = self.param(offset.y);
                let oz = self.param(offset.z);
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!("{} - {}", point_var, L::vec3_ctor(&ox, &oy, &oz)),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Rotate { child, rotation } => {
                self.ensure_helper("quat_rotate");
                let inv_rot = rotation.inverse();
                let new_p = self.next_var();
                let qx = self.param(inv_rot.x);
                let qy = self.param(inv_rot.y);
                let qz = self.param(inv_rot.z);
                let qw = self.param(inv_rot.w);
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!(
                        "quat_rotate({}, {})",
                        point_var,
                        L::vec4_ctor(&qx, &qy, &qz, &qw)
                    ),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Scale { child, factor } => {
                let new_p = self.next_var();
                let inv_factor = 1.0 / factor;
                let p_inv = self.param(inv_factor);
                let p_factor = self.param(*factor);
                code.push_str(&L::decl_vec3(&new_p, &format!("{} * {}", point_var, p_inv)));
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("{} * {}", d, p_factor)));
                var
            }

            SdfNode::ScaleNonUniform { child, factors } => {
                let new_p = self.next_var();
                let inv_x = self.param(1.0 / factors.x);
                let inv_y = self.param(1.0 / factors.y);
                let inv_z = self.param(1.0 / factors.z);
                let min_scale = factors.x.law_min(factors.y).law_min(factors.z);
                let p_min = self.param(min_scale);
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!("{} * {}", point_var, L::vec3_ctor(&inv_x, &inv_y, &inv_z)),
                ));
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("{} * {}", d, p_min)));
                var
            }

            // ============ Modifiers ============
            SdfNode::Twist { child, strength } => {
                // The native `sin` / `cos` are bounded (WGSL: 2⁻¹¹ absolute) only on [−π, π], so
                // the angle `a = strength · y` is reduced by quadrants first:
                //   n = floor(a · 2/π + 0.5),  r = (a − n·P1) − n·P2,  P1 = 1.5703125 (201/128,
                //   8 significant bits), P2 = fl(π/2 − P1) = 4.8382679e-4
                // `n·P1` is exact while |n| ≤ 2¹⁶ (8 + 16 ≤ 24 bits) and `a − n·P1` is a single
                // rounding, so |r − (a − n·π/2)| ≤ |n|·(|π/2 − P1 − P2| + ulp(n·P2)) + ulp(r)
                // ≤ 2¹⁶ · 2⁻³⁴ + 2⁻²⁴ ≈ 4e-6. The rounding of `a · 2/π` in n can leave |r| past
                // π/4 (0.793 measured), inside r ∈ [−π, π], which is what the native pair needs,
                // for |a| ≤ 2¹⁶ · π/2 (≈ 1.03e5). The quadrant q = n mod 4 picks (sin, cos) of r or
                // their swap / negation exactly (a select and a ±1 product), so c² + s² is that of
                // the native pair on r: √(c² + s²) ∈ [1 − √2·2⁻¹¹, 1 + √2·2⁻¹¹], the twist's
                // factor in `bound::SHADER_TWIST_DELTA`. The angle itself is within ~4e-6 of the
                // CPU's (|a| up to 1.03e5); outside that range the reduction is not derived
                let angle_var = self.next_var();
                let n_var = self.next_var();
                let r_var = self.next_var();
                let q_var = self.next_var();
                let c0_var = self.next_var();
                let s0_var = self.next_var();
                let odd_var = self.next_var();
                let sg_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();
                let str_val = self.param(*strength);
                code.push_str(&L::decl_float(
                    &angle_var,
                    &format!("{} * {}.y", str_val, point_var),
                ));
                code.push_str(&L::decl_float(
                    &n_var,
                    &format!("floor({angle_var} * 0.63661975 + 0.5)"),
                ));
                code.push_str(&L::decl_float(
                    &r_var,
                    &format!("({angle_var} - {n_var} * 1.5703125) - {n_var} * 0.00048382679"),
                ));
                code.push_str(&L::decl_float(
                    &q_var,
                    &format!("{n_var} - 4.0 * floor({n_var} * 0.25)"),
                ));
                code.push_str(&L::decl_float(&c0_var, &format!("cos({r_var})")));
                code.push_str(&L::decl_float(&s0_var, &format!("sin({r_var})")));
                code.push_str(&L::decl_float(
                    &odd_var,
                    &format!("{q_var} - 2.0 * floor({q_var} * 0.5)"),
                ));
                code.push_str(&L::decl_float(
                    &sg_var,
                    &format!("1.0 - 2.0 * floor({q_var} * 0.5)"),
                ));
                // q = 0: (s0, c0); 1: (c0, −s0); 2: (−s0, −c0); 3: (−c0, s0)  (sin, cos of r + q·π/2)
                code.push_str(&L::decl_float(
                    &s_var,
                    &format!(
                        "{sg_var} * {}",
                        L::select_float(&format!("{odd_var} > 0.5"), &c0_var, &s0_var)
                    ),
                ));
                code.push_str(&L::decl_float(
                    &c_var,
                    &format!(
                        "{sg_var} * {}",
                        L::select_float(
                            &format!("{odd_var} > 0.5"),
                            &format!("-{s0_var}"),
                            &c0_var
                        )
                    ),
                ));
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &L::vec3_ctor(
                        &format!("{} * {}.x - {} * {}.z", c_var, point_var, s_var, point_var),
                        &format!("{}.y", point_var),
                        &format!("{} * {}.x + {} * {}.z", s_var, point_var, c_var, point_var),
                    ),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Bend { child, curvature } => {
                let angle_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();
                let curv = self.param(*curvature);
                code.push_str(&L::decl_float(
                    &angle_var,
                    &format!("{} * {}.x", curv, point_var),
                ));
                code.push_str(&L::decl_float(&c_var, &format!("cos({})", angle_var)));
                code.push_str(&L::decl_float(&s_var, &format!("sin({})", angle_var)));
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &L::vec3_ctor(
                        // `real::bend`: (c x - s y, s x + c y, z)
                        &format!("{} * {}.x - {} * {}.y", c_var, point_var, s_var, point_var),
                        &format!("{} * {}.x + {} * {}.y", s_var, point_var, c_var, point_var),
                        &format!("{}.z", point_var),
                    ),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Round { child, radius } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let var = self.next_var();
                let r = self.param(*radius);
                code.push_str(&L::decl_float(&var, &format!("{} - {}", d, r)));
                var
            }

            SdfNode::Onion { child, thickness } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let var = self.next_var();
                let th = self.param(*thickness);
                code.push_str(&L::decl_float(&var, &format!("abs({}) - {}", d, th)));
                var
            }

            SdfNode::Elongate { child, amount } => {
                // Same law as `crate::eval::mod::eval`'s `SdfNode::Elongate`
                // arm and `real::elongate` (tree, VM bytecode, scalar/SIMD
                // JIT): `q = p - clamp(p, -a, a)`, the child evaluated
                // as-is at `q` (IQ's cheap elongate, no box correction —
                // the previous "exact" shader law disagreed by up to 1.0 on
                // an interior point, e.g. `elongate(1,2,3, sphere(1))` at
                // (0.011, 0.666, 0.515): cpu -1.0 vs the old shader -1.99).
                let bound = self.next_var();
                let q_var = self.next_var();
                let ax = self.param(amount.x);
                let ay = self.param(amount.y);
                let az = self.param(amount.z);
                code.push_str(&L::decl_vec3(&bound, &L::vec3_ctor(&ax, &ay, &az)));
                code.push_str(&L::decl_vec3(
                    &q_var,
                    &format!("{p} - clamp({p}, -{b}, {b})", p = point_var, b = bound),
                ));
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::RepeatInfinite { child, spacing } => {
                let new_p = self.next_var();
                let sx = self.param(spacing.x);
                let sy = self.param(spacing.y);
                let sz = self.param(spacing.z);
                let half_s = self.next_var();
                code.push_str(&L::decl_vec3(
                    &half_s,
                    &format!("{} * 0.5", L::vec3_ctor(&sx, &sy, &sz)),
                ));
                // `p * (1 / s)` with the reciprocal baked as an f32 constant: the
                // CPU paths multiply by the same reciprocal, so a cell-boundary tie
                // resolves identically (`p / s` differs by an ulp).
                let isx = self.param(1.0 / spacing.x);
                let isy = self.param(1.0 / spacing.y);
                let isz = self.param(1.0 / spacing.z);
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!(
                        "{p} - {s} * floor({p} * {inv} + 0.5)",
                        p = point_var,
                        s = L::vec3_ctor(&sx, &sy, &sz),
                        inv = L::vec3_ctor(&isx, &isy, &isz),
                    ),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::RepeatCells {
                child,
                spacing,
                count,
                mirror,
                neighbours,
                variation,
            } => {
                if *neighbours || variation.is_some() {
                    // no shader law yet (shader_unsupported_nodes names it): NaN, not the
                    // single-cell value with the field ignored
                    let v = self.next_var();
                    code.push_str(&L::decl_float(&v, L::nan_expr()));
                    return v;
                }
                // the law of modifiers::repeat_cells, operation for operation: per repeated axis
                // c = floor(p·(1/s) [+ h] + 0.5) (clamped to [0, n−1] when finite),
                // q = p − c·s (or p − (c − h)·s), and q·(1 − 2·(c − 2·floor(c/2))) when mirrored
                let c = count.unwrap_or([0; 3]);
                let comps = ["x", "y", "z"];
                let mut axes = Vec::with_capacity(3);
                for i in 0..3 {
                    let s = spacing[i];
                    let pc = format!("{point_var}.{}", comps[i]);
                    if s.is_nan() || s <= 0.0 {
                        axes.push(pc);
                        continue;
                    }
                    let sv = self.param(s);
                    let inv = self.param(1.0 / s);
                    let cell = self.next_var();
                    let q = self.next_var();
                    if c[i] == 0 {
                        code.push_str(&L::decl_float(&cell, &format!("floor({pc} * {inv} + 0.5)")));
                        code.push_str(&L::decl_float(&q, &format!("{pc} - {cell} * {sv}")));
                    } else {
                        let h = self.param((c[i] as f32 - 1.0) * 0.5);
                        let top = self.param(c[i] as f32 - 1.0);
                        code.push_str(&L::decl_float(
                            &cell,
                            &format!("min(max(floor({pc} * {inv} + {h} + 0.5), 0.0), {top})"),
                        ));
                        code.push_str(&L::decl_float(&q, &format!("{pc} - ({cell} - {h}) * {sv}")));
                    }
                    if *mirror {
                        let m = self.next_var();
                        code.push_str(&L::decl_float(
                            &m,
                            &format!("{q} * (1.0 - ({cell} - floor({cell} * 0.5) * 2.0) * 2.0)"),
                        ));
                        axes.push(m);
                    } else {
                        axes.push(q);
                    }
                }
                let new_p = self.next_var();
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &L::vec3_ctor(&axes[0], &axes[1], &axes[2]),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::RepeatFinite {
                child,
                spacing,
                count,
            } => {
                let new_p = self.next_var();
                let sx = self.param(spacing.x);
                let sy = self.param(spacing.y);
                let sz = self.param(spacing.z);
                // clamp to ±count/2 like `real::repeat_finite` (the shader used to
                // clamp to ±count, twice the CPU extent)
                let cx = self.param(count[0] as f32 * 0.5);
                let cy = self.param(count[1] as f32 * 0.5);
                let cz = self.param(count[2] as f32 * 0.5);
                let isx = self.param(1.0 / spacing.x);
                let isy = self.param(1.0 / spacing.y);
                let isz = self.param(1.0 / spacing.z);
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!(
                        "{p} - {s} * clamp(floor({p} * {inv} + 0.5), -{c}, {c})",
                        p = point_var,
                        s = L::vec3_ctor(&sx, &sy, &sz),
                        inv = L::vec3_ctor(&isx, &isy, &isz),
                        c = L::vec3_ctor(&cx, &cy, &cz),
                    ),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Noise {
                child,
                amplitude,
                frequency,
                seed,
            } => {
                // Same Perlin law as `modifiers::perlin_noise_3d` (CPU / SIMD / bytecode)
                self.ensure_helper("perlin_noise");
                let d = self.transpile_node_inner(child, point_var, code);
                let n_var = self.next_var();
                let var = self.next_var();
                let freq = self.param(*frequency);
                let amp = self.param(*amplitude);
                code.push_str(&L::decl_float(
                    &n_var,
                    &format!("perlin_noise_3d({} * {}, {}u)", point_var, freq, seed),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{} + {} * {}", d, n_var, amp),
                ));
                var
            }

            SdfNode::Mirror { child, axes } => {
                let new_p = self.next_var();
                let mx = if axes.x != 0.0 {
                    format!("abs({}.x)", point_var)
                } else {
                    format!("{}.x", point_var)
                };
                let my = if axes.y != 0.0 {
                    format!("abs({}.y)", point_var)
                } else {
                    format!("{}.y", point_var)
                };
                let mz = if axes.z != 0.0 {
                    format!("abs({}.z)", point_var)
                } else {
                    format!("{}.z", point_var)
                };
                code.push_str(&L::decl_vec3(&new_p, &L::vec3_ctor(&mx, &my, &mz)));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::OctantMirror { child } => {
                self.ensure_helper("octant_mirror");
                let new_p = self.next_var();
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!("alice_octant_mirror({})", point_var),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::ProjectiveTransform {
                child,
                inv_matrix,
                lipschitz_bound,
            } => {
                // `projective_transform`: w = row 3 of M⁻¹ · (p, 1), q = rows
                // 0-2 times `1 / w`, the child distance times
                // `min(|1 / w|, lipschitz_bound)`. The matrix entries are
                // literals; every `a * b + c` is written out in the CPU's
                // order (two roundings, no `fma`).
                let m = inv_matrix;
                let w = self.next_var();
                let inv_w = self.next_var();
                let q = self.next_var();
                code.push_str(&L::decl_float(
                    &w,
                    &format!(
                        "{} * {point_var}.x + {} * {point_var}.y + {} * {point_var}.z + {}",
                        lit(m[3]),
                        lit(m[7]),
                        lit(m[11]),
                        lit(m[15])
                    ),
                ));
                code.push_str(&L::decl_float(&inv_w, &format!("1.0 / {w}")));
                code.push_str(&L::decl_vec3(
                    &q,
                    &format!("{} * {inv_w}", transform_point3_expr::<L>(m, point_var)),
                ));
                let d = self.transpile_node_inner(child, &q, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{d} * min(abs({inv_w}), {})", lit(*lipschitz_bound)),
                ));
                var
            }

            SdfNode::LatticeDeform {
                child,
                control_points,
                nx,
                ny,
                nz,
                bbox_min,
                bbox_max,
            } => {
                // `lattice_deform`: outside the box → identity (correction 1);
                // inside → tricubic Bernstein sum over the control points, the
                // correction |∂q/∂x| by central difference (ε = 0.001, ≥ 0.1)
                // divides the child distance. Control points are a module-scope
                // array, the FFD a module-scope function (called three times).
                let id = self.var_counter;
                self.var_counter += 1;
                let arr = format!("alice_ffd_cp_{id}");
                let func = format!("alice_ffd_{id}");
                let flat: Vec<f32> = control_points
                    .iter()
                    .flat_map(|c| [c.x, c.y, c.z])
                    .collect();
                let arr = self.data_array(&arr, &flat);
                let cpx = (nx + 1) as usize;
                let cpy = (ny + 1) as usize;
                let cpz = (nz + 1) as usize;
                let size = *bbox_max - *bbox_min;
                let bmin = L::vec3_ctor(&lit(bbox_min.x), &lit(bbox_min.y), &lit(bbox_min.z));
                let bmax = L::vec3_ctor(&lit(bbox_max.x), &lit(bbox_max.y), &lit(bbox_max.z));
                let inv = L::vec3_ctor(&lit(1.0 / size.x), &lit(1.0 / size.y), &lit(1.0 / size.z));
                let mut body = String::new();
                body.push_str(&L::decl_vec3(
                    "s",
                    &format!(
                        "clamp((p - {bmin}) * {inv}, {}, {})",
                        L::vec3_zero(),
                        L::vec3_splat("1.0")
                    ),
                ));
                body.push_str(&L::decl_mut_vec3("r", L::vec3_zero()));
                let bern = |t: &str, i: usize| -> String {
                    match i {
                        0 => format!("((1.0 - {t}) * (1.0 - {t}) * (1.0 - {t}))"),
                        1 => format!("(3.0 * {t} * (1.0 - {t}) * (1.0 - {t}))"),
                        2 => format!("(3.0 * {t} * {t} * (1.0 - {t}))"),
                        _ => format!("({t} * {t} * {t})"),
                    }
                };
                for i in 0..cpx.min(4) {
                    for j in 0..cpy.min(4) {
                        for k in 0..cpz.min(4) {
                            let idx = i * cpy * cpz + j * cpz + k;
                            if idx >= control_points.len() {
                                continue;
                            }
                            let cp = L::vec3_ctor(
                                &arr.at(&(idx * 3).to_string()),
                                &arr.at(&(idx * 3 + 1).to_string()),
                                &arr.at(&(idx * 3 + 2).to_string()),
                            );
                            writeln!(
                                body,
                                "    r = r + {cp} * ({} * {} * {});",
                                bern("s.x", i),
                                bern("s.y", j),
                                bern("s.z", k)
                            )
                            .unwrap();
                        }
                    }
                }
                body.push_str("    return r;\n");
                self.globals.push_str(&L::global_vec3_fn(&func, &body));

                let q = self.next_var();
                let corr = self.next_var();
                let p = point_var;
                code.push_str(&L::decl_mut_vec3(&q, p));
                code.push_str(&L::decl_mut_float(&corr, "1.0"));
                let ex = L::vec3_ctor("0.001", "0.0", "0.0");
                writeln!(
                    code,
                    "    if (!({p}.x < {bmin}.x || {p}.y < {bmin}.y || {p}.z < {bmin}.z || {p}.x > {bmax}.x || {p}.y > {bmax}.y || {p}.z > {bmax}.z)) {{"
                )
                .unwrap();
                writeln!(code, "        {q} = {func}({p});").unwrap();
                writeln!(
                    code,
                    "        {corr} = max(length({func}({p} + {ex}) - {func}({p} - {ex})) / 0.002, 0.1);"
                )
                .unwrap();
                code.push_str("    }\n");
                let d = self.transpile_node_inner(child, &q, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("{d} / {corr}")));
                var
            }

            SdfNode::GridWarp { child, warp } => {
                // `transforms::grid_warp`: q = clamp(p, box), the 8 node maps of q's cell blended
                // trilinearly (corner order 000, 100, 010, 110, 001, 101, 011, 111, each sum
                // started at 0.0, the CPU order), x = M q; the child at x divided by the computed
                // bound, extended outside the box by hypot(e, max(d, 0)) and clipped by the box
                // (max with the box's signed distance b, the CPU order)
                let (x, e, bx, lam, fl) = self.emit_grid_warp_point(warp, point_var, code);
                let data = warp.data();
                let dc = self.transpile_node_inner(child, &x, code);
                let (dv, dp, out) = (self.next_var(), self.next_var(), self.next_var());
                let r = lit(data.reach);
                let lg = lit(warp.lambda_global());
                code.push_str(&L::decl_float(
                    &dv,
                    &format!(
                        "max(min(max({dc} / {lam}, min(-{r}, {dc} / {lg})), max({r}, {dc} / {lg})), clamp({fl}, -{r}, {r}))"
                    ),
                ));
                code.push_str(&L::decl_float(&dp, &format!("max({dv}, 0.0)")));
                code.push_str(&L::decl_float(
                    &out,
                    &format!("max({dv} + (sqrt({e} * {e} + {dp} * {dp}) - {dp}), {bx})"),
                ));
                out
            }

            SdfNode::SdfSkinning { child, bones } => {
                // `sdf_skinning`: Σ w · bind(current(p)) / Σ w over bones with
                // w ≥ 1e-6 (glam column-major `transform_point3` twice);
                // no bone or no weight → identity.
                let active: Vec<&crate::transforms::skinning::BoneTransform> =
                    bones.iter().filter(|b| b.weight >= 1e-6).collect();
                let total: f32 = active.iter().map(|b| b.weight).sum();
                if active.is_empty() || total <= 1e-6 {
                    return self.transpile_node_inner(child, point_var, code);
                }
                let acc = self.next_var();
                code.push_str(&L::decl_mut_vec3(&acc, L::vec3_zero()));
                for bone in &active {
                    let skinned = self.next_var();
                    let rest = self.next_var();
                    code.push_str(&L::decl_vec3(
                        &skinned,
                        &transform_point3_expr::<L>(&bone.current_pose, point_var),
                    ));
                    code.push_str(&L::decl_vec3(
                        &rest,
                        &transform_point3_expr::<L>(&bone.inv_bind_pose, &skinned),
                    ));
                    writeln!(code, "    {acc} = {acc} + {rest} * {};", lit(bone.weight)).unwrap();
                }
                let q = self.next_var();
                code.push_str(&L::decl_vec3(&q, &format!("{acc} / {}", lit(total))));
                self.transpile_node_inner(child, &q, code)
            }

            SdfNode::IcosahedralSymmetry { child } => {
                self.ensure_helper("icosahedral_fold");
                let new_p = self.next_var();
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!("alice_icosahedral_fold({})", point_var),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::IFS {
                child,
                transforms,
                iterations,
            } => {
                // `ifs_fold_with_scale`: per iteration keep the transform whose
                // image is nearest the origin; the scale accumulates that
                // matrix's X-axis length; the child distance is divided by it.
                // Transforms and iterations are data, unrolled as literals.
                let q = self.next_var();
                let sc = self.next_var();
                code.push_str(&L::decl_mut_vec3(&q, point_var));
                code.push_str(&L::decl_mut_float(&sc, "1.0"));
                for _ in 0..*iterations {
                    let best = self.next_var();
                    let bestd = self.next_var();
                    let bests = self.next_var();
                    code.push_str(&L::decl_mut_vec3(&best, &q));
                    code.push_str(&L::decl_mut_float(&bestd, &format!("dot({q}, {q})")));
                    code.push_str(&L::decl_mut_float(&bests, "1.0"));
                    for m in transforms {
                        let t = self.next_var();
                        let td = self.next_var();
                        code.push_str(&L::decl_vec3(&t, &transform_point3_expr::<L>(m, &q)));
                        code.push_str(&L::decl_float(&td, &format!("dot({t}, {t})")));
                        let x_axis_len = (m[2] * m[2] + (m[1] * m[1] + (m[0] * m[0]))).sqrt();
                        writeln!(
                            code,
                            "    if ({td} < {bestd}) {{ {bestd} = {td}; {best} = {t}; {bests} = {}; }}",
                            lit(x_axis_len)
                        )
                        .unwrap();
                    }
                    writeln!(code, "    {q} = {best};").unwrap();
                    writeln!(code, "    {sc} = {sc} * {bests};").unwrap();
                }
                let d = self.transpile_node_inner(child, &q, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("{d} / max({sc}, 1e-6)")));
                var
            }

            SdfNode::HeightmapDisplacement {
                child,
                heightmap,
                width,
                height,
                amplitude,
                scale,
            } => {
                // `heightmap_displacement`: dominant-axis projection to (u, v),
                // mapped to [0, w-1] × [0, h-1], bilinear sample × amplitude,
                // subtracted from the child distance. The map is a module-scope
                // array.
                let d = self.transpile_node_inner(child, point_var, code);
                if heightmap.is_empty() || *width == 0 || *height == 0 {
                    return d;
                }
                let id = self.var_counter;
                self.var_counter += 1;
                let arr = format!("alice_hm_{id}");
                let arr = self.data_array(&arr, heightmap);
                let (wm1, hm1) = (lit(*width as f32 - 1.0), lit(*height as f32 - 1.0));
                let sc = self.param(*scale);
                let amp = self.param(*amplitude);
                let p = point_var;
                let ap = self.next_var();
                code.push_str(&L::decl_vec3(&ap, &format!("abs({p})")));
                // the law of `heightmap_displacement`: the dominant face, with the other faces blended
                // in within FACE_BLEND_BAND (1/16) of the dominant component, as
                // h_dom + sum_{i != dom} (w_i / sum w) * (h_i - h_dom) in axis order (a face with no
                // weight adds exactly 0)
                let x_dom = format!("{ap}.x > {ap}.y && {ap}.x > {ap}.z");
                let y_dom = format!("{ap}.y > {ap}.z");
                let (m, lo) = (self.next_var(), self.next_var());
                code.push_str(&L::decl_float(
                    &m,
                    &format!("max(max({ap}.x, {ap}.y), {ap}.z)"),
                ));
                code.push_str(&L::decl_float(
                    &lo,
                    &format!("{m} - {m} * {}", lit(crate::modifiers::FACE_BLEND_BAND)),
                ));
                let wts: Vec<String> = ["x", "y", "z"]
                    .iter()
                    .map(|c| {
                        let w = self.next_var();
                        code.push_str(&L::decl_float(&w, &format!("max({ap}.{c} - {lo}, 0.0)")));
                        w
                    })
                    .collect();
                let total = self.next_var();
                code.push_str(&L::decl_float(
                    &total,
                    &format!("({} + {}) + {}", wts[0], wts[1], wts[2]),
                ));
                let faces = [
                    (format!("{p}.y"), format!("{p}.z")),
                    (format!("{p}.x"), format!("{p}.z")),
                    (format!("{p}.x"), format!("{p}.y")),
                ];
                let samples: Vec<String> = faces
                    .iter()
                    .map(|(u, v)| {
                        self.heightmap_face_sample(
                            code, &arr, u, v, &sc, &wm1, &hm1, *width, *height,
                        )
                    })
                    .collect();
                let h_dom = self.next_var();
                code.push_str(&L::decl_float(
                    &h_dom,
                    &L::select_float(
                        &x_dom,
                        &samples[0],
                        &L::select_float(&y_dom, &samples[1], &samples[2]),
                    ),
                ));
                let not_dom = [
                    format!("!({x_dom})"),
                    format!("(({x_dom}) || !({y_dom}))"),
                    format!("(({x_dom}) || ({y_dom}))"),
                ];
                let mut acc = h_dom.clone();
                for axis in 0..3 {
                    let next = self.next_var();
                    code.push_str(&L::decl_float(
                        &next,
                        &L::select_float(
                            &format!("{} && {} > 0.0", not_dom[axis], wts[axis]),
                            &format!(
                                "{acc} + ({} / {total}) * ({} - {h_dom})",
                                wts[axis], samples[axis]
                            ),
                            &acc,
                        ),
                    ));
                    acc = next;
                }
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("{d} - {acc} * {amp}")));
                var
            }

            SdfNode::SurfaceRoughness {
                child,
                amplitude,
                frequency,
                octaves,
                ..
            } => {
                self.ensure_helper("hash_noise");
                let d = self.transpile_node_inner(child, point_var, code);
                let var = self.next_var();
                let amp = self.param(*amplitude);
                let freq = self.param(*frequency);
                let noise_var = self.next_var();
                let sp = self.next_var();
                code.push_str(&L::decl_vec3(&sp, &format!("{} * {}", point_var, freq)));
                code.push_str(&L::decl_mut_float(&noise_var, "0.0"));
                code.push_str(&L::decl_mut_float(&format!("{}_a", noise_var), "1.0"));
                for i in 0..*octaves {
                    let scale = 1u32 << i;
                    writeln!(
                        code,
                        "    {n} = {n} + {n}_a * hash_noise_3d({sp} * {s}.0, 42u);",
                        n = noise_var,
                        sp = sp,
                        s = scale
                    )
                    .unwrap();
                    writeln!(code, "    {n}_a = {n}_a * 0.5;", n = noise_var).unwrap();
                }
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{} + {} * {}", d, amp, noise_var),
                ));
                var
            }

            SdfNode::Revolution { child, offset } => {
                let new_p = self.next_var();
                let off = self.param(*offset);
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &L::vec3_ctor(
                        &format!("length({}.xz) - {}", point_var, off),
                        &format!("{}.y", point_var),
                        "0.0",
                    ),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Extrude { child, half_height } => {
                // `real::extrude_point` / `extrude_distance`: the 2-D child lives in
                // XY (z = 0) and the slab is |z| ≤ half_height
                let new_p_2d = self.next_var();
                code.push_str(&L::decl_vec3(
                    &new_p_2d,
                    &L::vec3_ctor(
                        &format!("{}.x", point_var),
                        &format!("{}.y", point_var),
                        "0.0",
                    ),
                ));
                let d = self.transpile_node_inner(child, &new_p_2d, code);
                let var = self.next_var();
                let hh = self.param(*half_height);
                let w_var = self.next_var();
                code.push_str(&L::decl_vec2(
                    &w_var,
                    &L::vec2_ctor(&d, &format!("abs({}.z) - {}", point_var, hh)),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "min(max({w}.x, {w}.y), 0.0) + length(max({w}, {z}))",
                        w = w_var,
                        z = L::vec2_zero()
                    ),
                ));
                var
            }

            SdfNode::Taper {
                child,
                factor,
                reach,
            } => {
                self.ensure_helper("taper_bound");
                let new_p = self.next_var();
                let f = self.param(*factor);
                // `reach` may be INFINITY (unbounded child); shaders have no
                // inf literal, so ≥ 1e30 is the "no cone bound" sentinel on
                // every path (`real::taper_bound` uses the same test).
                let rx = self.param(if reach[0] < 1e30 { reach[0] } else { 1e30 });
                let ry = self.param(if reach[1] < 1e30 { reach[1] } else { 1e30 });
                // CPU law (`real::taper`): den = 1 - y * f, |den| >= 1e-6 with the
                // sign kept. The transpilers used to emit `1 + y * f` (mirrored
                // taper) with a `max(den, 0.001)` clamp — a different law.
                let taper_var = self.next_var();
                let mag_var = self.next_var();
                let den_var = self.next_var();
                code.push_str(&L::decl_float(
                    &taper_var,
                    &format!("1.0 - {}.y * {}", point_var, f),
                ));
                code.push_str(&L::decl_float(
                    &mag_var,
                    &format!("max(abs({}), 1e-6)", taper_var),
                ));
                code.push_str(&L::decl_float(
                    &den_var,
                    &L::select_float(
                        &format!("{} < 0.0", taper_var),
                        &format!("-{}", mag_var),
                        &mag_var,
                    ),
                ));
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &L::vec3_ctor(
                        &format!("{}.x / {}", point_var, den_var),
                        &format!("{}.y", point_var),
                        &format!("{}.z / {}", point_var, den_var),
                    ),
                ));
                // Child distance at the tapered point → parent-space bound
                // (`real::taper_bound`, same law as every CPU path).
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(
                    &var,
                    &format!("alice_taper_bound({d}, {point_var}, {f}, {rx}, {ry})"),
                ));
                var
            }

            SdfNode::Displacement { child, strength } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let var = self.next_var();
                let s = self.param(*strength);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        // `modifier_displacement`: sin(5x) sin(5y) sin(5z)
                        "{} + {} * sin({p}.x * 5.0) * sin({p}.y * 5.0) * sin({p}.z * 5.0)",
                        d,
                        s,
                        p = point_var
                    ),
                ));
                var
            }

            // Phase 28: SineDisplacement (= 軸別 frequency Vec3) を transpile
            SdfNode::SineDisplacement {
                child,
                amplitude,
                frequency,
            } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let var = self.next_var();
                let a = self.param(*amplitude);
                let fx = self.param(frequency.x);
                let fy = self.param(frequency.y);
                let fz = self.param(frequency.z);
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "{} + {} * sin({p}.x * {}) * sin({p}.y * {}) * sin({p}.z * {})",
                        d,
                        a,
                        fx,
                        fy,
                        fz,
                        p = point_var
                    ),
                ));
                var
            }

            SdfNode::SweepBezier { child, p0, p1, p2 } => {
                // p0, p1, p2 are Vec2 (XZ plane). We project p into XZ, find closest on bezier,
                // then evaluate child in local frame.
                let var = self.next_var();
                let p0x = self.param(p0.x);
                let p0y = self.param(p0.y);
                let p1x = self.param(p1.x);
                let p1y = self.param(p1.y);
                let p2x = self.param(p2.x);
                let p2y = self.param(p2.y);
                // Closed-form distance to the curve (IQ sdBezier), one helper
                // per language mirroring `modifiers::sweep::bezier_distance_2d`.
                self.ensure_helper("bezier_distance_2d");
                let local_p = self.next_var();
                code.push_str(&L::decl_vec3(
                    &local_p,
                    &L::vec3_ctor(
                        &format!(
                            "bezier_distance_2d({}, {}, {}, {})",
                            L::vec2_ctor(&format!("{point_var}.x"), &format!("{point_var}.z")),
                            L::vec2_ctor(&p0x, &p0y),
                            L::vec2_ctor(&p1x, &p1y),
                            L::vec2_ctor(&p2x, &p2y)
                        ),
                        &format!("{point_var}.y"),
                        "0.0",
                    ),
                ));
                let d = self.transpile_node_inner(child, &local_p, code);
                code.push_str(&L::decl_float(&var, &d));
                var
            }

            SdfNode::PolarRepeat { child, count } => {
                let var = self.next_var();
                let n = self.param(*count as f32);
                let angle_var = self.next_var();
                let sector_var = self.next_var();
                let recip_var = self.next_var();
                let snapped_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();
                let pi2 = self.param(std::f32::consts::TAU);
                self.ensure_helper("alice_atan2");
                code.push_str(&L::decl_float(
                    &angle_var,
                    &L::atan2_expr(&format!("{point_var}.z"), &format!("{point_var}.x")),
                ));
                code.push_str(&L::decl_float(&sector_var, &format!("{} / {}", pi2, n)));
                // `angle * (n / TAU)` with the same operands as the CPU law
                // (`real::polar_repeat`), so a boundary angle snaps to the same
                // sector on every path (`angle / sector` differs by an ulp).
                code.push_str(&L::decl_float(&recip_var, &format!("{} / {}", n, pi2)));
                code.push_str(&L::decl_float(
                    &snapped_var,
                    &format!(
                        "floor({} * {} + 0.5) * {}",
                        angle_var, recip_var, sector_var
                    ),
                ));
                code.push_str(&L::decl_float(&c_var, &format!("cos({})", snapped_var)));
                code.push_str(&L::decl_float(&s_var, &format!("sin({})", snapped_var)));
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &L::vec3_ctor(
                        &format!(
                            "{c} * {p}.x + {s} * {p}.z",
                            c = c_var,
                            s = s_var,
                            p = point_var
                        ),
                        &format!("{}.y", point_var),
                        &format!(
                            "-{s} * {p}.x + {c} * {p}.z",
                            c = c_var,
                            s = s_var,
                            p = point_var
                        ),
                    ),
                ));
                let d = self.transpile_node_inner(child, &new_p, code);
                code.push_str(&L::decl_float(&var, &d));
                var
            }

            SdfNode::WithMaterial { child, .. } | SdfNode::Pose { child } => {
                self.transpile_node_inner(child, point_var, code)
            }

            // ============ 2D Primitives (extruded to 3D) ============
            SdfNode::Circle2D {
                radius,
                half_height,
            } => {
                let r = self.param(*radius);
                let hh = self.param(*half_height);
                let v = self.next_var();
                code.push_str(&L::decl_float(
                    &format!("{}_d2d", v),
                    &format!("length({}.xy) - {}", point_var, r),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_dz", v),
                    &format!("abs({}.z) - {}", point_var, hh),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wx", v),
                    &format!("max({}_d2d, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wy", v),
                    &format!("max({}_dz, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &v,
                    &format!(
                        "sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0)",
                        v = v
                    ),
                ));
                v
            }

            SdfNode::Rect2D {
                half_extents,
                half_height,
            } => {
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hh = self.param(*half_height);
                let v = self.next_var();
                code.push_str(&L::decl_float(
                    &format!("{}_dx", v),
                    &format!("abs({}.x) - {}", point_var, hx),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_dy", v),
                    &format!("abs({}.y) - {}", point_var, hy),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_d2d", v),
                    &format!(
                        "length(max({}, {})) + min(max({v}_dx, {v}_dy), 0.0)",
                        L::vec2_ctor(&format!("{}_dx", v), &format!("{}_dy", v)),
                        L::vec2_zero(),
                        v = v
                    ),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_dz", v),
                    &format!("abs({}.z) - {}", point_var, hh),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wx", v),
                    &format!("max({}_d2d, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wy", v),
                    &format!("max({}_dz, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &v,
                    &format!(
                        "sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0)",
                        v = v
                    ),
                ));
                v
            }

            SdfNode::Segment2D {
                a,
                b,
                thickness,
                half_height,
            } => {
                let ax = self.param(a.x);
                let ay = self.param(a.y);
                let bx = self.param(b.x);
                let by = self.param(b.y);
                let th = self.param(*thickness);
                let hh = self.param(*half_height);
                let v = self.next_var();
                code.push_str(&L::decl_vec2(
                    &format!("{}_pa", v),
                    &format!("{}.xy - {}", point_var, L::vec2_ctor(&ax, &ay)),
                ));
                code.push_str(&L::decl_vec2(
                    &format!("{}_ba", v),
                    &format!("{} - {}", L::vec2_ctor(&bx, &by), L::vec2_ctor(&ax, &ay)),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_h", v),
                    &format!(
                        "clamp(dot({v}_pa, {v}_ba) / dot({v}_ba, {v}_ba), 0.0, 1.0)",
                        v = v
                    ),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_d2d", v),
                    &format!("length({v}_pa - {v}_ba * {v}_h) - {th}", v = v, th = th),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_dz", v),
                    &format!("abs({}.z) - {}", point_var, hh),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wx", v),
                    &format!("max({}_d2d, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wy", v),
                    &format!("max({}_dz, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &v,
                    &format!(
                        "sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0)",
                        v = v
                    ),
                ));
                v
            }

            SdfNode::SectionStack(st) => {
                // module-scope arrays and a loop function, like a large polygon: the field of
                // `SectionStack::eval` (no parameters, so Dynamic mode emits the same text)
                let id = self.var_counter;
                self.var_counter += 1;
                let name = format!("alice_stack_{id}");
                let data = SectionStackData::new(st);
                let src = L::global_section_stack(&name, &data);
                let (ny, nv, nl, ns) = (
                    data.levels.len(),
                    data.verts.len(),
                    data.loop_start.len(),
                    data.vert_start.len(),
                );
                if self.node_data.is_some() && L::NAME == "wgsl" {
                    // the four arrays as one range of the node data buffer (integers stored exactly as
                    // f32, all below 2^24), the module-scope declarations dropped
                    let (oy, ov, ol, os) = (0, ny, ny + nv, ny + nv + nl);
                    let mut values = data.levels.clone();
                    values.extend_from_slice(&data.verts);
                    values.extend(data.loop_start.iter().map(|&x| x as f32));
                    values.extend(data.vert_start.iter().map(|&x| x as f32));
                    let d = self.data_array(&format!("{name}_data"), &values);
                    let mut kept = String::with_capacity(src.len());
                    for l in src
                        .lines()
                        .filter(|l| !l.starts_with(&format!("const {name}_")))
                    {
                        writeln!(kept, "{l}").unwrap();
                    }
                    let src = kept;
                    let src =
                        self.render_data(&src, &format!("{name}_y"), ny, Some((&d, oy)), &|x| x);
                    let src =
                        self.render_data(&src, &format!("{name}_v"), nv, Some((&d, ov)), &|x| x);
                    let src =
                        self.render_data(&src, &format!("{name}_ls"), nl, Some((&d, ol)), &|x| {
                            format!("i32({x})")
                        });
                    let src =
                        self.render_data(&src, &format!("{name}_vs"), ns, Some((&d, os)), &|x| {
                            format!("i32({x})")
                        });
                    self.globals.push_str(&src);
                } else {
                    let src = self.render_data(&src, &format!("{name}_y"), ny, None, &|x| x);
                    let src = self.render_data(&src, &format!("{name}_v"), nv, None, &|x| x);
                    let src = self.render_data(&src, &format!("{name}_ls"), nl, None, &|x| x);
                    let src = self.render_data(&src, &format!("{name}_vs"), ns, None, &|x| x);
                    self.globals.push_str(&src);
                }
                let v = self.next_var();
                code.push_str(&L::decl_float(&v, &format!("{name}({point_var})")));
                v
            }

            SdfNode::SmoothSectionStack(st) => {
                // the stack's law as module-scope functions over a module-scope data array
                // (`compiled::smooth_stack_shader`, calling the loft's closed-curve emitter)
                let id = self.var_counter;
                self.var_counter += 1;
                let arr = format!("alice_sss_d_{id}");
                let func = format!("alice_sss_{id}");
                let flat = crate::compiled::smooth_stack_shader::flatten(st);
                let funcs = crate::compiled::smooth_stack_shader::stack_functions(
                    &func,
                    &arr,
                    &flat,
                    L::NAME,
                );
                if self.node_data.is_some() && L::NAME == "wgsl" {
                    // the node data buffer: no module-scope array (a WGSL array constructor is limited
                    // to 32767 elements), every read a range of `alice_node_data`
                    let d = self.data_array(&arr, &flat);
                    let src = self.render_data(&funcs, &arr, flat.len(), Some((&d, 0)), &|x| x);
                    self.globals.push_str(&src);
                } else {
                    // a plain array: the loft's array emitter reads the loft's own slots (pieces, fade)
                    self.globals
                        .push_str(&crate::compiled::loft_shader::const_array(
                            &arr,
                            &flat,
                            L::NAME,
                            L::global_float_array,
                        ));
                    let src = self.render_data(&funcs, &arr, flat.len(), None, &|x| x);
                    self.globals.push_str(&src);
                }
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("{func}({point_var})")));
                var
            }

            SdfNode::Loft {
                axis,
                up,
                harmonics,
                rows,
                caps,
                bounds,
            } => {
                let (sec, caps) = (
                    crate::primitives::loft::LoftSection::Polar(*harmonics),
                    *caps,
                );
                // the loft's law as module-scope functions over a module-scope data array
                // (`compiled::loft_shader`, one template rendered per dialect)
                let id = self.var_counter;
                self.var_counter += 1;
                let arr = format!("alice_loft_d_{id}");
                let func = format!("alice_loft_{id}");
                let flat =
                    crate::primitives::loft::flatten(axis, *up, sec, rows, &[], caps, bounds);
                self.loft_globals(&arr, &func, &flat);
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("{func}({point_var})")));
                var
            }

            SdfNode::LoftCurves {
                axis,
                up,
                section_points,
                points: rows,
                knots,
                caps,
                bounds,
            } => {
                let (sec, caps) = (
                    crate::primitives::loft::LoftSection::Curve(*section_points),
                    [caps[0][0], caps[1][0]],
                );
                // the loft's law as module-scope functions over a module-scope data array
                // (`compiled::loft_shader`, one template rendered per dialect)
                let id = self.var_counter;
                self.var_counter += 1;
                let arr = format!("alice_loft_d_{id}");
                let func = format!("alice_loft_{id}");
                let flat =
                    crate::primitives::loft::flatten(axis, *up, sec, rows, knots, caps, bounds);
                self.loft_globals(&arr, &func, &flat);
                let var = self.next_var();
                code.push_str(&L::decl_float(&var, &format!("{func}({point_var})")));
                var
            }

            SdfNode::Polygon2D {
                vertices,
                half_height,
            } => {
                let hh = self.param(*half_height);
                let v = self.next_var();
                let n = vertices.len();
                if n < 3 {
                    code.push_str(&L::decl_float(&v, "1e10"));
                    return v;
                }
                // Dynamic mode keeps the vertices as parameters, so that `update_params` moves them
                if n > POLYGON_2D_UNROLL_MAX && matches!(self.mode, TranspileModeLang::Hardcoded) {
                    // a module-scope vertex array and one loop function: the shader's length does not
                    // grow with the outline (an unrolled outline of hundreds of edges makes a shader
                    // some drivers take seconds to compile)
                    let id = self.var_counter;
                    self.var_counter += 1;
                    let arr = format!("alice_poly_v_{id}");
                    let func = format!("alice_poly_{id}");
                    let flat: Vec<f32> = vertices.iter().flat_map(|q| [q.x, q.y]).collect();
                    // a module-scope array, or with the node data buffer a range of it
                    let d = self.data_array(&arr, &flat);
                    self.globals
                        .push_str(&L::global_polygon_2d_fn(&func, &|i| d.at(i), n));
                    code.push_str(&L::decl_float(
                        &format!("{v}_d2d"),
                        &format!("{func}({point_var}.xy)"),
                    ));
                    code.push_str(&L::decl_float(
                        &format!("{v}_dz"),
                        &format!("abs({point_var}.z) - {hh}"),
                    ));
                    code.push_str(&L::decl_float(
                        &format!("{v}_wx"),
                        &format!("max({v}_d2d, 0.0)"),
                    ));
                    code.push_str(&L::decl_float(
                        &format!("{v}_wy"),
                        &format!("max({v}_dz, 0.0)"),
                    ));
                    code.push_str(&L::decl_float(
                        &v,
                        &format!(
                            "sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0)"
                        ),
                    ));
                    return v;
                }
                // Emit vertex params
                let mut vx: Vec<String> = Vec::new();
                let mut vy: Vec<String> = Vec::new();
                for vert in vertices {
                    vx.push(self.param(vert.x));
                    vy.push(self.param(vert.y));
                }
                // Start with first edge distance
                code.push_str(&L::decl_mut_float(
                    &format!("{}_d", v),
                    &format!(
                        "dot({p}.xy - {v0}, {p}.xy - {v0})",
                        p = point_var,
                        v0 = L::vec2_ctor(&vx[0], &vy[0])
                    ),
                ));
                code.push_str(&L::decl_mut_float(&format!("{}_s", v), "1.0"));
                for i in 0..n {
                    let j = (i + 1) % n;
                    let ei = self.next_var();
                    let wi = self.next_var();
                    code.push_str(&L::decl_vec2(
                        &ei,
                        &format!(
                            "{} - {}",
                            L::vec2_ctor(&vx[j], &vy[j]),
                            L::vec2_ctor(&vx[i], &vy[i])
                        ),
                    ));
                    code.push_str(&L::decl_vec2(
                        &wi,
                        &format!("{}.xy - {}", point_var, L::vec2_ctor(&vx[i], &vy[i])),
                    ));
                    let ci = self.next_var();
                    code.push_str(&L::decl_float(
                        &ci,
                        &format!(
                            "clamp(dot({w}, {e}) / dot({e}, {e}), 0.0, 1.0)",
                            w = wi,
                            e = ei
                        ),
                    ));
                    let bi = self.next_var();
                    code.push_str(&L::decl_vec2(&bi, &format!("{} - {} * {}", wi, ei, ci)));
                    writeln!(
                        code,
                        "    {v}_d = min({v}_d, dot({b}, {b}));",
                        v = v,
                        b = bi
                    )
                    .unwrap();
                    // Winding number
                    let cond1 = format!("{p}.y >= {vy_i}", p = point_var, vy_i = vy[i]);
                    let cond2 = format!("{p}.y < {vy_j}", p = point_var, vy_j = vy[j]);
                    let cond3 = format!("{e}.x * {w}.y > {e}.y * {w}.x", e = ei, w = wi);
                    writeln!(
                        code,
                        "    if ({c1} && {c2} && {c3}) {{ {v}_s = -{v}_s; }}",
                        c1 = cond1,
                        c2 = cond2,
                        c3 = cond3,
                        v = v
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "    if (!({c1}) && !({c2}) && !({c3})) {{ {v}_s = -{v}_s; }}",
                        c1 = cond1,
                        c2 = cond2,
                        c3 = cond3,
                        v = v
                    )
                    .unwrap();
                }
                code.push_str(&L::decl_float(
                    &format!("{}_d2d", v),
                    &format!("{v}_s * sqrt({v}_d)", v = v),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_dz", v),
                    &format!("abs({}.z) - {}", point_var, hh),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wx", v),
                    &format!("max({}_d2d, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wy", v),
                    &format!("max({}_dz, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &v,
                    &format!(
                        "sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0)",
                        v = v
                    ),
                ));
                v
            }

            SdfNode::RoundedRect2D {
                half_extents,
                round_radius,
                half_height,
            } => {
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let rr = self.param(*round_radius);
                let hh = self.param(*half_height);
                let v = self.next_var();
                code.push_str(&L::decl_float(
                    &format!("{}_dx", v),
                    &format!("abs({}.x) - {} + {}", point_var, hx, rr),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_dy", v),
                    &format!("abs({}.y) - {} + {}", point_var, hy, rr),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_d2d", v),
                    &format!(
                        "length(max({}, {})) + min(max({v}_dx, {v}_dy), 0.0) - {}",
                        L::vec2_ctor(&format!("{}_dx", v), &format!("{}_dy", v)),
                        L::vec2_zero(),
                        rr,
                        v = v
                    ),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_dz", v),
                    &format!("abs({}.z) - {}", point_var, hh),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wx", v),
                    &format!("max({}_d2d, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wy", v),
                    &format!("max({}_dz, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &v,
                    &format!(
                        "sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0)",
                        v = v
                    ),
                ));
                v
            }

            SdfNode::Annular2D {
                outer_radius,
                thickness,
                half_height,
            } => {
                let r = self.param(*outer_radius);
                let th = self.param(*thickness);
                let hh = self.param(*half_height);
                let v = self.next_var();
                code.push_str(&L::decl_float(
                    &format!("{}_d2d", v),
                    &format!("abs(length({}.xy) - {}) - {}", point_var, r, th),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_dz", v),
                    &format!("abs({}.z) - {}", point_var, hh),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wx", v),
                    &format!("max({}_d2d, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_wy", v),
                    &format!("max({}_dz, 0.0)", v),
                ));
                code.push_str(&L::decl_float(
                    &v,
                    &format!(
                        "sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0)",
                        v = v
                    ),
                ));
                v
            }

            SdfNode::Terrain { scale, amplitude } => {
                // Same law as `crate::eval::mod::eval`'s `SdfNode::Terrain`
                // arm: 3 octaves of `hash_noise_3d` on the xz plane (y held
                // at 0), gain 0.48, lacunarity 2.1 with a rotation. Portable
                // across GLSL/WGSL/HLSL via `ensure_helper` + `L::decl_*`
                // (the previous raw-GLSL `format!` emitted an undefined
                // `vnoise` and was invalid WGSL/HLSL syntax).
                self.ensure_helper("hash_noise");
                let sc = self.param(*scale);
                let amp = self.param(*amplitude);
                let v_acc = self.next_var();
                let a_fbm = self.next_var();
                let fx = self.next_var();
                let fz = self.next_var();
                code.push_str(&L::decl_mut_float(&v_acc, "0.0"));
                code.push_str(&L::decl_mut_float(&a_fbm, "0.5"));
                code.push_str(&L::decl_mut_float(
                    &fx,
                    &format!("{}.x * {}", point_var, sc),
                ));
                code.push_str(&L::decl_mut_float(
                    &fz,
                    &format!("{}.z * {}", point_var, sc),
                ));
                for _ in 0..3 {
                    let sample_p = self.next_var();
                    let n = self.next_var();
                    code.push_str(&L::decl_vec3(&sample_p, &L::vec3_ctor(&fx, "0.0", &fz)));
                    code.push_str(&L::decl_float(
                        &n,
                        &format!("hash_noise_3d({}, {}u)", sample_p, TERRAIN_NOISE_SEED),
                    ));
                    writeln!(
                        code,
                        "    {v} = {v} + {a} * {n};",
                        v = v_acc,
                        a = a_fbm,
                        n = n
                    )
                    .unwrap();
                    let nx = self.next_var();
                    let nz = self.next_var();
                    code.push_str(&L::decl_float(&nx, &format!("0.6 * {} + 0.8 * {}", fz, fx)));
                    code.push_str(&L::decl_float(&nz, &format!("0.8 * {} - 0.6 * {}", fz, fx)));
                    writeln!(code, "    {fx} = {nx} * 2.1;", fx = fx, nx = nx).unwrap();
                    writeln!(code, "    {fz} = {nz} * 2.1;", fz = fz, nz = nz).unwrap();
                    writeln!(code, "    {a} = {a} * 0.48;", a = a_fbm).unwrap();
                }
                let v = self.next_var();
                code.push_str(&L::decl_float(
                    &v,
                    &format!("{}.y - {} * {}", point_var, v_acc, amp),
                ));
                v
            }

            // Exponential smooth operations
            SdfNode::ExpSmoothUnion { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && k.abs() < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("min({}, {})", d_a, d_b)));
                    return var;
                }
                // stable form: min(a, b) - k * log(1 + exp(-|a - b| / k)), same as
                // `sdf_exp_smooth_union_r` (no underflow to log(0), no clamp)
                let inv_k = self.param(1.0 / k.law_max(1e-6));
                let m = self.next_var();
                let delta = self.next_var();
                let s = self.next_var();
                code.push_str(&L::decl_float(&m, &format!("min({}, {})", d_a, d_b)));
                code.push_str(&L::decl_float(&delta, &format!("abs({} - {})", d_a, d_b)));
                code.push_str(&L::decl_float(
                    &s,
                    &format!("1.0 + exp(-{} * {})", inv_k, delta),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{} - log({}) * {}", m, s, self.param(k.law_max(1e-6))),
                ));
                var
            }

            SdfNode::ExpSmoothIntersection { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && k.abs() < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("max({}, {})", d_a, d_b)));
                    return var;
                }
                // stable form: max(a, b) + k * log(1 + exp(-|a - b| / k))
                let inv_k = self.param(1.0 / k.law_max(1e-6));
                let m = self.next_var();
                let delta = self.next_var();
                let s = self.next_var();
                code.push_str(&L::decl_float(&m, &format!("max({}, {})", d_a, d_b)));
                code.push_str(&L::decl_float(&delta, &format!("abs({} - {})", d_a, d_b)));
                code.push_str(&L::decl_float(
                    &s,
                    &format!("1.0 + exp(-{} * {})", inv_k, delta),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{} + log({}) * {}", m, s, self.param(k.law_max(1e-6))),
                ));
                var
            }

            SdfNode::ExpSmoothSubtraction { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && k.abs() < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("max({}, -{})", d_a, d_b)));
                    return var;
                }
                // stable form of the intersection with -b: max(a, -b) + k * log(1 + exp(-|a + b| / k))
                let inv_k = self.param(1.0 / k.law_max(1e-6));
                let m = self.next_var();
                let delta = self.next_var();
                let s = self.next_var();
                code.push_str(&L::decl_float(&m, &format!("max({}, -{})", d_a, d_b)));
                code.push_str(&L::decl_float(&delta, &format!("abs({} + {})", d_a, d_b)));
                code.push_str(&L::decl_float(
                    &s,
                    &format!("1.0 + exp(-{} * {})", inv_k, delta),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{} + log({}) * {}", m, s, self.param(k.law_max(1e-6))),
                ));
                var
            }

            SdfNode::SmoothBlend {
                a,
                b,
                k,
                kernel,
                mode,
            } => {
                // the kernel table's shader helper (operations::smooth_blend), with
                // the mode's negations as on the CPU: intersection −smin(−a, −b),
                // subtraction −smin(−a, b)
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let helper = crate::operations::smooth_blend_helper_name(*kernel);
                self.ensure_helper(helper);
                let kk = self.param(*k);
                let var = self.next_var();
                let expr = match mode {
                    BlendMode::Union => format!("{helper}({d_a}, {d_b}, {kk})"),
                    BlendMode::Intersection => format!("-{helper}(-{d_a}, -{d_b}, {kk})"),
                    BlendMode::Subtraction => format!("-{helper}(-{d_a}, {d_b}, {kk})"),
                };
                code.push_str(&L::decl_float(&var, &expr));
                var
            }

            SdfNode::Shear { child, shear } => {
                let sx = self.param(shear.x);
                let sy = self.param(shear.y);
                let sz = self.param(shear.z);
                let new_p = self.next_var();
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &L::vec3_ctor(
                        &format!("{p}.x", p = point_var),
                        &format!("{p}.y - {sx} * {p}.x", p = point_var, sx = sx),
                        &format!(
                            "{p}.z - {sy} * {p}.x - {sz} * {p}.y",
                            p = point_var,
                            sy = sy,
                            sz = sz
                        ),
                    ),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Animated { child, .. } => self.transpile_node_inner(child, point_var, code),
        }
    }
}

/// The node kind's name (the enum variant), for messages
fn node_kind(node: &SdfNode) -> String {
    let s = format!("{node:?}");
    s.split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or("?")
        .to_owned()
}

#[cfg(test)]
mod read_range_tests {
    use super::{render_reads, Reads};

    fn render(src: &str, len: usize) -> (String, Vec<String>) {
        let mut errs = Vec::new();
        let out = render_reads(src, "a", len, 0, &Reads::Plain, &|x| x, &mut errs);
        (out, errs)
    }

    #[test]
    fn a_read_inside_its_array_renders_without_its_range() {
        let (out, errs) = render("x = a[i + 1 @ 1..4] + ba[2];", 5);
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(out, "x = a[i + 1] + ba[2];");
    }

    #[test]
    fn a_range_past_the_array_is_refused() {
        let (_, errs) = render("x = a[i @ 0..5];", 5);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(
            errs[0].contains("0..=5") && errs[0].contains("0..5"),
            "{errs:?}"
        );
    }

    #[test]
    fn an_inverted_range_is_refused() {
        let (_, errs) = render("x = a[i @ 3..2];", 5);
        assert_eq!(errs.len(), 1, "{errs:?}");
    }

    #[test]
    fn a_read_without_its_range_is_refused() {
        let (_, errs) = render("x = a[i];", 5);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("without its index range"), "{errs:?}");
    }

    #[test]
    fn a_declaration_is_not_a_read() {
        let (out, errs) = render("static const float a[5] = { 1.0 };\nx = a[0 @ 0..0];", 5);
        assert!(errs.is_empty(), "{errs:?}");
        assert!(out.starts_with("static const float a[5]"));
    }
}

#[cfg(test)]
mod array_chunk_tests {
    use super::split_large_arrays;

    fn items(n: usize) -> (Vec<f32>, String) {
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 1.5 + 0.25).collect();
        let s = v
            .iter()
            .map(|x| format!("{x:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        (v, s)
    }

    #[test]
    fn every_dialect_writes_chunks_of_at_most_the_limit() {
        let (_, s) = items(10);
        for (dialect, decl, read) in [
            (
                "wgsl",
                format!("const A: array<f32, 10> = array<f32, 10>({s});"),
                "A[j]",
            ),
            (
                "glsl",
                format!("const float A[10] = float[10]({s});"),
                "A[j]",
            ),
            (
                "hlsl",
                format!("static const float A[10] = {{ {s} }};"),
                "A[j]",
            ),
        ] {
            let out = split_large_arrays(&format!("{decl}\nx = {read} + BA[1];\n"), dialect, 3);
            for (k, n) in [(0, 3), (1, 3), (2, 3), (3, 1)] {
                let want = match dialect {
                    "wgsl" => format!("const A_c{k}: array<f32, {n}> = array<f32, {n}>("),
                    "glsl" => format!("const float A_c{k}[{n}] = float[{n}]("),
                    _ => format!("static const float A_c{k}[{n}] = {{ "),
                };
                assert!(out.contains(&want), "{dialect}: no {want}\n{out}");
            }
            assert!(
                !out.contains("A[j]"),
                "{dialect}: a read left as it was\n{out}"
            );
            assert!(out.contains("A_r("), "{dialect}: no reader call\n{out}");
            assert!(
                out.contains("BA[1]"),
                "{dialect}: a longer name was rewritten\n{out}"
            );
        }
        // an array within the limit is left alone
        let small = format!("const A: array<f32, 10> = array<f32, 10>({s});\n");
        assert_eq!(split_large_arrays(&small, "wgsl", 10), small);
    }

    /// The reader on the GPU returns every element, across every chunk boundary
    #[cfg(feature = "gpu")]
    #[test]
    fn a_chunked_array_reads_every_element_on_the_gpu() {
        if !crate::gpu_test_support::gpu_ready("a_chunked_array_reads_every_element_on_the_gpu") {
            return;
        }
        let n = 10;
        let (values, s) = items(n);
        let src = format!(
            "const A: array<f32, {n}> = array<f32, {n}>({s});\nfn sdf_eval(p: vec3<f32>) -> f32 {{\n    return A[i32(p.x)];\n}}\n"
        );
        let split = split_large_arrays(&src, "wgsl", 3);
        assert!(split.contains("A_c3"), "{split}");
        let mut shader = crate::compiled::WgslShader::transpile(
            &crate::types::SdfNode::sphere(1.0),
            crate::compiled::TranspileMode::Hardcoded,
        );
        shader.source = split;
        let gpu = crate::compiled::GpuEvaluator::from_shader(&shader).expect("evaluator");
        let pts: Vec<glam::Vec3> = (0..10)
            .map(|i| glam::Vec3::new(i as f32 + 0.25, 0.0, 0.0))
            .collect();
        assert_eq!(gpu.eval_batch(&pts).expect("gpu"), values);
    }
}
