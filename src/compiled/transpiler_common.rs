//! Shared shader transpiler framework: Generic transpiler parameterized by shader language.
//!
//! This module provides a `ShaderLang` trait and `GenericTranspiler<L>` that contains
//! the shared transpilation logic (~2600 lines of `transpile_node_inner`), eliminating
//! ~90% code duplication between WGSL, GLSL, and HLSL transpilers.
//!
//! Author: Moroya Sakamoto

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
pub const SHADER_UNSUPPORTED: [&str; 4] = [
    "LatticeDeform",
    "HeightmapDisplacement",
    "SdfSkinning",
    "IFS",
];

const fn unsupported_name(node: &SdfNode) -> Option<&'static str> {
    // Every node kind is transpiled since 2.2.0 (IFS / SdfSkinning unrolled
    // as literals, LatticeDeform / HeightmapDisplacement as module-scope
    // data); the list stays as the hook for a future node that cannot be.
    let _ = node;
    None
}

/// Names of the nodes in `node`'s tree that the transpilers pass through
/// unchanged (see [`SHADER_UNSUPPORTED`]); empty when the shader is a
/// faithful port of the tree.
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
pub trait ShaderLang: private::Sealed + 'static {
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
    fn atan2_expr(y: &str, x: &str) -> String {
        format!("atan2({y}, {x})")
    }
    /// "f32(x)" / "float(x)"
    fn cast_float(expr: &str) -> String;
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
}

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
    _phantom: std::marker::PhantomData<L>,
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
            _phantom: std::marker::PhantomData,
        }
    }

    /// Allocate the next unique variable name (d0, d1, ...).
    pub fn next_var(&mut self) -> String {
        let var = format!("d{}", self.var_counter);
        self.var_counter += 1;
        var
    }

    /// Register a helper function by name, deduplicating.
    pub fn ensure_helper(&mut self, name: &'static str) {
        if !self.helper_functions.contains(&name) {
            self.helper_functions.push(name);
        }
    }

    /// Register a float parameter and return its shader expression.
    pub fn param(&mut self, value: f32) -> String {
        match self.mode {
            TranspileModeLang::Hardcoded => format!("{:.6}", value),
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

    /// Assemble the final shader string: helpers + function signature + body.
    pub fn generate_shader(&self, body: &str) -> String {
        let mut shader = String::new();
        for helper in &self.helper_functions {
            if let Some(src) = L::helper_source(helper) {
                shader.push_str(src);
                shader.push('\n');
            }
        }
        writeln!(shader, "{}", L::func_signature()).unwrap();
        shader.push_str(body);
        shader.push_str("}\n");
        shader
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
                let p_k2sq = self.param(k2x.mul_add(k2x, k2y * k2y));

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
                    &format!(
                        "max({} - abs({} - {}), 0.0) * {}",
                        k_str, d_a, d_b, inv_k_str
                    ),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "min({}, {}) - {} * {} * {} * 0.25",
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
                    &format!(
                        "max({} - abs({} - {}), 0.0) * {}",
                        k_str, d_a, d_b, inv_k_str
                    ),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "max({}, {}) + {} * {} * {} * 0.25",
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
                    &format!(
                        "max({} - abs({} - {}), 0.0) * {}",
                        k_str, d_a, neg_b, inv_k_str
                    ),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!(
                        "max({}, {}) + {} * {} * {} * 0.25",
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
                code.push_str(&L::decl_float(
                    &var,
                    &format!("mix({}, {}, {})", d_a, d_b, t_s),
                ));
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
                let min_scale = factors.x.min(factors.y).min(factors.z);
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
                let angle_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();
                let str_val = self.param(*strength);
                code.push_str(&L::decl_float(
                    &angle_var,
                    &format!("{} * {}.y", str_val, point_var),
                ));
                code.push_str(&L::decl_float(&c_var, &format!("cos({})", angle_var)));
                code.push_str(&L::decl_float(&s_var, &format!("sin({})", angle_var)));
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
                let new_p = self.next_var();
                let q_var = self.next_var();
                let ax = self.param(amount.x);
                let ay = self.param(amount.y);
                let az = self.param(amount.z);
                code.push_str(&L::decl_vec3(
                    &q_var,
                    &format!("abs({}) - {}", point_var, L::vec3_ctor(&ax, &ay, &az)),
                ));
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!("max({}, {})", q_var, L::vec3_zero()),
                ));
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{} + min(max({q}.x, max({q}.y, {q}.z)), 0.0)", d, q = q_var),
                ));
                var
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

            // Unsupported / pass-through transforms
            SdfNode::ProjectiveTransform { child, .. } => {
                self.transpile_node_inner(child, point_var, code)
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
                self.globals.push_str(&L::global_float_array(&arr, &flat));
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
                                &format!("{arr}[{}]", idx * 3),
                                &format!("{arr}[{}]", idx * 3 + 1),
                                &format!("{arr}[{}]", idx * 3 + 2),
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
                        let x_axis_len = m[2].mul_add(m[2], m[1].mul_add(m[1], m[0] * m[0])).sqrt();
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
                self.globals
                    .push_str(&L::global_float_array(&arr, heightmap));
                let (wm1, hm1) = (lit(*width as f32 - 1.0), lit(*height as f32 - 1.0));
                let sc = self.param(*scale);
                let amp = self.param(*amplitude);
                let p = point_var;
                let ap = self.next_var();
                let u = self.next_var();
                let v = self.next_var();
                code.push_str(&L::decl_vec3(&ap, &format!("abs({p})")));
                let x_dom = format!("{ap}.x > {ap}.y && {ap}.x > {ap}.z");
                let y_dom = format!("{ap}.y > {ap}.z");
                code.push_str(&L::decl_float(
                    &u,
                    &L::select_float(
                        &x_dom,
                        &format!("{p}.y"),
                        &L::select_float(&y_dom, &format!("{p}.x"), &format!("{p}.x")),
                    ),
                ));
                code.push_str(&L::decl_float(
                    &v,
                    &L::select_float(
                        &x_dom,
                        &format!("{p}.z"),
                        &L::select_float(&y_dom, &format!("{p}.z"), &format!("{p}.y")),
                    ),
                ));
                let uu = self.next_var();
                let vv = self.next_var();
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
                let wi = *width as i64;
                code.push_str(&L::decl_int(&u0, &L::cast_int(&format!("floor({uu})"))));
                code.push_str(&L::decl_int(&v0, &L::cast_int(&format!("floor({vv})"))));
                code.push_str(&L::decl_int(&u1, &format!("min({u0} + 1, {})", wi - 1)));
                code.push_str(&L::decl_int(
                    &v1,
                    &format!("min({v0} + 1, {})", *height as i64 - 1),
                ));
                code.push_str(&L::decl_float(&fu, &format!("{uu} - floor({uu})")));
                code.push_str(&L::decl_float(&fv, &format!("{vv} - floor({vv})")));
                let a = self.next_var();
                let b = self.next_var();
                code.push_str(&L::decl_float(
                    &a,
                    &format!(
                        "{arr}[{v0} * {wi} + {u0}] * (1.0 - {fu}) + {arr}[{v0} * {wi} + {u1}] * {fu}"
                    ),
                ));
                code.push_str(&L::decl_float(
                    &b,
                    &format!(
                        "{arr}[{v1} * {wi} + {u0}] * (1.0 - {fu}) + {arr}[{v1} * {wi} + {u1}] * {fu}"
                    ),
                ));
                let var = self.next_var();
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{d} - ({a} * (1.0 - {fv}) + {b} * {fv}) * {amp}"),
                ));
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

            SdfNode::WithMaterial { child, .. } => {
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
                let sc = self.param(*scale);
                let amp = self.param(*amplitude);
                let v = self.next_var();
                // terrainHeight の GLSL インライン展開 (3-octave FBM)
                code.push_str(&format!(
                    "    float {v}_th = 0.0; float {v}_a = 0.5;\n    vec2 {v}_fp = {p}.xz * {sc};\n    for(int {v}_i=0;{v}_i<3;{v}_i++) {{\n        {v}_th += {v}_a * vnoise({v}_fp);\n        vec2 {v}_nfp = vec2(0.8*{v}_fp.x+0.6*{v}_fp.y, -0.6*{v}_fp.x+0.8*{v}_fp.y);\n        {v}_fp = {v}_nfp * 2.1; {v}_a *= 0.48;\n    }}\n",
                    v = v, p = point_var, sc = sc
                ));
                code.push_str(&L::decl_float(
                    &v,
                    &format!("{}.y - {}_th * {}", point_var, v, amp),
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
                let inv_k = self.param(1.0 / k.max(1e-6));
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
                    &format!("{} - log({}) * {}", m, s, self.param(k.max(1e-6))),
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
                let inv_k = self.param(1.0 / k.max(1e-6));
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
                    &format!("{} + log({}) * {}", m, s, self.param(k.max(1e-6))),
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
                let inv_k = self.param(1.0 / k.max(1e-6));
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
                    &format!("{} + log({}) * {}", m, s, self.param(k.max(1e-6))),
                ));
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
