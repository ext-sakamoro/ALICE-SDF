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

/// Trait that captures the syntactic differences between WGSL, GLSL, and HLSL.
pub trait ShaderLang: 'static {
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
    /// a % b or fmod(a, b)
    fn modulo_expr(a: &str, b: &str) -> String;
    /// "f32(x)" / "float(x)"
    fn cast_float(expr: &str) -> String;
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
    _phantom: std::marker::PhantomData<L>,
}

impl<L: ShaderLang> GenericTranspiler<L> {
    /// Create a new transpiler in the given mode.
    pub fn new(mode: TranspileModeLang) -> Self {
        Self {
            var_counter: 0,
            helper_functions: Vec::new(),
            mode,
            params: Vec::new(),
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
                    )
                    .clone(),
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
                    )
                    .clone(),
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
                    &format!(
                        "dot({}, {}) + {}",
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
                let p_k2sq = self.param(k2x * k2x + k2y * k2y);

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
                code.push_str(&L::decl_float(&h_var, &p_hh.clone()));
                // ca = vec2(qx - min(qx, select(0.0, r, p.y<0.0)), abs(p.y) - h)
                let cone_select = L::select_expr(&format!("{}.y < 0.0", point_var), &p_r, "0.0");
                code.push_str(&L::decl_vec2(
                    &ca_var,
                    &L::vec2_ctor(
                        &format!("{} - min({}, {})", qx_var, qx_var, cone_select),
                        &format!("abs({}.y) - {}", point_var, h_var),
                    )
                    .clone(),
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
                    )
                    .clone(),
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

            // Division Exorcism for Ellipsoid
            SdfNode::Ellipsoid { radii } => {
                let var = self.next_var();
                let k0_var = self.next_var();
                let k1_var = self.next_var();
                let inv_rx = self.param(1.0 / radii.x.max(1e-10));
                let inv_ry = self.param(1.0 / radii.y.max(1e-10));
                let inv_rz = self.param(1.0 / radii.z.max(1e-10));
                let inv_rx2 = self.param(1.0 / (radii.x * radii.x).max(1e-10));
                let inv_ry2 = self.param(1.0 / (radii.y * radii.y).max(1e-10));
                let inv_rz2 = self.param(1.0 / (radii.z * radii.z).max(1e-10));
                code.push_str(&L::decl_float(
                    &k0_var,
                    &format!(
                        "length({} * {})",
                        point_var,
                        L::vec3_ctor(&inv_rx, &inv_ry, &inv_rz)
                    ),
                ));
                code.push_str(&L::decl_float(
                    &k1_var,
                    &format!(
                        "length({} * {})",
                        point_var,
                        L::vec3_ctor(&inv_rx2, &inv_ry2, &inv_rz2)
                    ),
                ));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{} * ({} - 1.0) / max({}, 1e-10)", k0_var, k0_var, k1_var),
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
                    length(max({v1}, {z})) + min(max({p}.x, max({q}.y, {p}.z)), 0.0), \
                    length(max({v2}, {z})) + min(max({p}.x, max({q}.y, {q}.z)), 0.0)), \
                    length(max({v3}, {z})) + min(max({q}.x, max({p}.y, {q}.z)), 0.0))",
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
                    "abs(sin(2.0*{sp}.x)*cos({sp}.y)*sin({sp}.z) + sin(2.0*{sp}.y)*cos({sp}.z)*sin({sp}.x) + sin(2.0*{sp}.z)*cos({sp}.x)*sin({sp}.y) - cos(2.0*{sp}.x)*cos(2.0*{sp}.y) - cos(2.0*{sp}.y)*cos(2.0*{sp}.z) - cos(2.0*{sp}.z)*cos(2.0*{sp}.x) + 0.3) / {sc} - {th}",
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
                    "abs(cos({sp}.x)*cos({sp}.y) + cos({sp}.y)*cos({sp}.z) + cos({sp}.z)*cos({sp}.x) - cos({sp}.x)*cos({sp}.y)*cos({sp}.z)) / {sc} - {th}",
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
                    "abs(4.0*cos({sp}.x)*cos({sp}.y)*cos({sp}.z) - cos(2.0*{sp}.x)*cos(2.0*{sp}.y) - cos(2.0*{sp}.y)*cos(2.0*{sp}.z) - cos(2.0*{sp}.z)*cos(2.0*{sp}.x)) / {sc} - {th}",
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

            // Columns operations use modulo (language-specific: % vs fmod)
            SdfNode::ColumnsUnion { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                code.push_str(&L::decl_mut_float(
                    &format!("{}_m", var),
                    &format!("min({}, {})", d_a, d_b),
                ));
                writeln!(
                    code,
                    "    {}  = min({}, {});",
                    L::decl_mut_float(&format!("{}_a2", var), &format!("min({}, {})", d_a, d_b))
                        .trim_start(),
                    d_a,
                    d_b
                )
                .ok();
                // Actually, the above is getting complex. Let me use a simpler writeln approach for these complex arms.
                // We need: var _m, _a2, _b2, _cs, _ra, _rb, then modulo, then select/ternary.
                // Let me rewrite using direct writeln.
                // Clear the partial writes and redo:
                let code_len = code.len(); // Save for rollback if needed
                                           // Actually let's not complicate. Let me just emit directly with writeln.
                                           // Remove the decl_mut_float we just added
                code.truncate(
                    code_len
                        - L::decl_mut_float(
                            &format!("{}_m", var),
                            &format!("min({}, {})", d_a, d_b),
                        )
                        .len(),
                );

                // Emit ColumnsUnion with direct string formatting
                writeln!(
                    code,
                    "{}",
                    L::decl_mut_float(&format!("{}_m", var), &format!("min({}, {})", d_a, d_b))
                        .trim_end()
                )
                .unwrap();
                writeln!(
                    code,
                    "    {} {}_b2 = max({}, {});",
                    L::decl_mut_float(&format!("{}_a2", var), &format!("min({}, {})", d_a, d_b))
                        .trim_end(),
                    var,
                    d_a,
                    d_b
                )
                .ok();
                // This is getting too messy with the decl_ approach for complex multi-statement arms.
                // Let me use a different strategy: for ColumnsUnion and similar complex arms,
                // emit raw writeln! with language-specific tokens from the trait.
                code.truncate(code_len);

                // Start fresh for ColumnsUnion - use trait methods for just the syntax differences
                let mod_expr = L::modulo_expr(
                    &format!("{v}_ra + {v}_cs * 0.5", v = var),
                    &format!("{v}_cs", v = var),
                );
                let select_columns = L::select_expr(
                    &format!("{v}_m > {r}", v = var, r = r_s),
                    &format!("{v}_m", v = var),
                    &format!("min(min(0.70710678 * ({v}_ra + {v}_rb), 0.70710678 * ({v}_rb - {v}_ra)), {v}_m)", v = var),
                );

                code.push_str(&L::decl_mut_float(
                    &format!("{}_m", var),
                    &format!("min({}, {})", d_a, d_b),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_a2", var),
                    &format!("min({}, {})", d_a, d_b),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_b2", var),
                    &format!("max({}, {})", d_a, d_b),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_cs", var),
                    &format!("{} * 2.0 / {}", r_s, n_s),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_ra", var),
                    &format!(
                        "0.70710678 * ({v}_a2 + {v}_b2) - {r} * 0.70710678",
                        v = var,
                        r = r_s
                    ),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_rb", var),
                    &format!("0.70710678 * ({v}_b2 - {v}_a2)", v = var),
                ));
                writeln!(
                    code,
                    "    {v}_ra = {mod_e} - {v}_cs * 0.5;",
                    v = var,
                    mod_e = mod_expr
                )
                .unwrap();
                code.push_str(&L::decl_float(&var, &select_columns));
                var
            }

            SdfNode::ColumnsIntersection { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                let mod_expr = L::modulo_expr(
                    &format!("{v}_ra + {v}_cs * 0.5", v = var),
                    &format!("{v}_cs", v = var),
                );
                let select_columns = L::select_expr(
                    &format!("{v}_m > {r}", v = var, r = r_s),
                    &format!("-{v}_m", v = var),
                    &format!("-min(min(0.70710678 * ({v}_ra + {v}_rb), 0.70710678 * ({v}_rb - {v}_ra)), {v}_m)", v = var),
                );

                code.push_str(&L::decl_float(
                    &format!("{}_na", var),
                    &format!("-({})", d_a),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_m", var),
                    &format!("min({v}_na, {})", d_b, v = var),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_a2", var),
                    &format!("min({v}_na, {})", d_b, v = var),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_b2", var),
                    &format!("max({v}_na, {})", d_b, v = var),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_cs", var),
                    &format!("{} * 2.0 / {}", r_s, n_s),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_ra", var),
                    &format!(
                        "0.70710678 * ({v}_a2 + {v}_b2) - {r} * 0.70710678",
                        v = var,
                        r = r_s
                    ),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_rb", var),
                    &format!("0.70710678 * ({v}_b2 - {v}_a2)", v = var),
                ));
                writeln!(
                    code,
                    "    {v}_ra = {mod_e} - {v}_cs * 0.5;",
                    v = var,
                    mod_e = mod_expr
                )
                .unwrap();
                code.push_str(&L::decl_float(&var, &select_columns));
                var
            }

            SdfNode::ColumnsSubtraction { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                let mod_expr = L::modulo_expr(
                    &format!("{v}_ra + {v}_cs * 0.5", v = var),
                    &format!("{v}_cs", v = var),
                );
                let select_columns = L::select_expr(
                    &format!("{v}_m > {r}", v = var, r = r_s),
                    &format!("-{v}_m", v = var),
                    &format!("-min(min(0.70710678 * ({v}_ra + {v}_rb), 0.70710678 * ({v}_rb - {v}_ra)), {v}_m)", v = var),
                );

                code.push_str(&L::decl_float(
                    &format!("{}_na", var),
                    &format!("-({})", d_a),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_m", var),
                    &format!("min({v}_na, {})", d_b, v = var),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_a2", var),
                    &format!("min({v}_na, {})", d_b, v = var),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_b2", var),
                    &format!("max({v}_na, {})", d_b, v = var),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_cs", var),
                    &format!("{} * 2.0 / {}", r_s, n_s),
                ));
                code.push_str(&L::decl_mut_float(
                    &format!("{}_ra", var),
                    &format!(
                        "0.70710678 * ({v}_a2 + {v}_b2) - {r} * 0.70710678",
                        v = var,
                        r = r_s
                    ),
                ));
                code.push_str(&L::decl_float(
                    &format!("{}_rb", var),
                    &format!("0.70710678 * ({v}_b2 - {v}_a2)", v = var),
                ));
                writeln!(
                    code,
                    "    {v}_ra = {mod_e} - {v}_cs * 0.5;",
                    v = var,
                    mod_e = mod_expr
                )
                .unwrap();
                code.push_str(&L::decl_float(&var, &select_columns));
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
                    )
                    .clone(),
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
                        &format!("{} * {}.x + {} * {}.y", c_var, point_var, s_var, point_var),
                        &format!("{} * {}.y - {} * {}.x", c_var, point_var, s_var, point_var),
                        &format!("{}.z", point_var),
                    )
                    .clone(),
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
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!(
                        "{p} - {s} * round({p} / {s})",
                        p = point_var,
                        s = L::vec3_ctor(&sx, &sy, &sz),
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
                let cx = self.param(count[0] as f32);
                let cy = self.param(count[1] as f32);
                let cz = self.param(count[2] as f32);
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &format!(
                        "{p} - {s} * clamp(round({p} / {s}), -{c}, {c})",
                        p = point_var,
                        s = L::vec3_ctor(&sx, &sy, &sz),
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
                self.ensure_helper("hash_noise");
                let d = self.transpile_node_inner(child, point_var, code);
                let n_var = self.next_var();
                let var = self.next_var();
                let freq = self.param(*frequency);
                let amp = self.param(*amplitude);
                code.push_str(&L::decl_float(
                    &n_var,
                    &format!("hash_noise_3d({} * {}, {}u)", point_var, freq, seed),
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
                let new_p = self.next_var();
                code.push_str(&L::decl_vec3(&new_p, &format!("abs({})", point_var)));
                self.transpile_node_inner(child, &new_p, code)
            }

            // Unsupported / pass-through transforms
            SdfNode::ProjectiveTransform { child, .. } => {
                self.transpile_node_inner(child, point_var, code)
            }

            SdfNode::LatticeDeform { child, .. } => {
                self.transpile_node_inner(child, point_var, code)
            }

            SdfNode::SdfSkinning { child, .. } => self.transpile_node_inner(child, point_var, code),

            SdfNode::IcosahedralSymmetry { child } => {
                self.transpile_node_inner(child, point_var, code)
            }

            SdfNode::IFS { child, .. } => self.transpile_node_inner(child, point_var, code),

            SdfNode::HeightmapDisplacement { child, .. } => {
                self.transpile_node_inner(child, point_var, code)
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
                    )
                    .clone(),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Extrude { child, half_height } => {
                let new_p_2d = self.next_var();
                code.push_str(&L::decl_vec3(
                    &new_p_2d,
                    &L::vec3_ctor(
                        &format!("{}.x", point_var),
                        &format!("{}.z", point_var),
                        "0.0",
                    )
                    .clone(),
                ));
                let d = self.transpile_node_inner(child, &new_p_2d, code);
                let var = self.next_var();
                let hh = self.param(*half_height);
                let w_var = self.next_var();
                code.push_str(&L::decl_vec2(
                    &w_var,
                    &L::vec2_ctor(&d, &format!("abs({}.y) - {}", point_var, hh)).clone(),
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

            SdfNode::Taper { child, factor } => {
                let new_p = self.next_var();
                let f = self.param(*factor);
                let taper_var = self.next_var();
                code.push_str(&L::decl_float(
                    &taper_var,
                    &format!("1.0 + {}.y * {}", point_var, f),
                ));
                code.push_str(&L::decl_vec3(
                    &new_p,
                    &L::vec3_ctor(
                        &format!("{}.x / max({}, 0.001)", point_var, taper_var),
                        &format!("{}.y", point_var),
                        &format!("{}.z / max({}, 0.001)", point_var, taper_var),
                    )
                    .clone(),
                ));
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                code.push_str(&L::decl_float(
                    &var,
                    &format!("{} * max({}, 0.001)", d, taper_var),
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
                        "{} + {} * sin({p}.x * 10.0) * sin({p}.y * 10.0) * sin({p}.z * 10.0)",
                        d,
                        s,
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
                // Compute closest point on 2D Bezier in XZ plane
                let a_var = self.next_var();
                let b_var = self.next_var();
                let c_var = self.next_var();
                let d_var = self.next_var();
                code.push_str(&L::decl_vec2(
                    &a_var,
                    &format!(
                        "{} - {}",
                        L::vec2_ctor(&p1x, &p1y),
                        L::vec2_ctor(&p0x, &p0y)
                    ),
                ));
                code.push_str(&L::decl_vec2(
                    &b_var,
                    &format!(
                        "{} - 2.0 * {} + {}",
                        L::vec2_ctor(&p0x, &p0y),
                        L::vec2_ctor(&p1x, &p1y),
                        L::vec2_ctor(&p2x, &p2y)
                    ),
                ));
                code.push_str(&L::decl_vec2(&c_var, &format!("{} * 2.0", a_var)));
                code.push_str(&L::decl_vec2(
                    &d_var,
                    &format!(
                        "{} - {}",
                        L::vec2_ctor(&p0x, &p0y),
                        L::vec2_ctor(&format!("{}.x", point_var), &format!("{}.z", point_var))
                    ),
                ));
                // Newton iteration to find closest t
                let t_var = self.next_var();
                code.push_str(&L::decl_mut_float(&t_var, "0.5"));
                for _ in 0..4 {
                    let q_var = self.next_var();
                    let qp_var = self.next_var();
                    let qd_var = self.next_var();
                    code.push_str(&L::decl_vec2(
                        &q_var,
                        &format!(
                            "{d} + ({c} + {b} * {t}) * {t}",
                            d = d_var,
                            c = c_var,
                            b = b_var,
                            t = t_var
                        ),
                    ));
                    code.push_str(&L::decl_vec2(
                        &qd_var,
                        &format!("{c} + 2.0 * {b} * {t}", c = c_var, b = b_var, t = t_var),
                    ));
                    code.push_str(&L::decl_float(
                        &qp_var,
                        &format!("dot({}, {})", q_var, qd_var),
                    ));
                    writeln!(code, "    {t} = clamp({t} - {qp} / max(dot({qd}, {qd}) + dot({q}, {b}) * 2.0, 1e-10), 0.0, 1.0);",
                        t = t_var, qp = qp_var, qd = qd_var, q = q_var, b = b_var
                    ).unwrap();
                }
                // Compute bezier point at t (2D in XZ)
                let bp_var = self.next_var();
                let omt = self.next_var();
                code.push_str(&L::decl_float(&omt, &format!("1.0 - {}", t_var)));
                code.push_str(&L::decl_vec2(
                    &bp_var,
                    &format!(
                        "{p0} * {omt} * {omt} + {p1} * 2.0 * {omt} * {t} + {p2} * {t} * {t}",
                        p0 = L::vec2_ctor(&p0x, &p0y),
                        p1 = L::vec2_ctor(&p1x, &p1y),
                        p2 = L::vec2_ctor(&p2x, &p2y),
                        omt = omt,
                        t = t_var
                    ),
                ));
                // Local 3D point: (distance from bezier path, y, 0)
                let local_p = self.next_var();
                code.push_str(&L::decl_vec3(
                    &local_p,
                    &L::vec3_ctor(
                        &format!("length({} - {}.xz)", bp_var, point_var),
                        &format!("{}.y", point_var),
                        "0.0",
                    )
                    .clone(),
                ));
                let d = self.transpile_node_inner(child, &local_p, code);
                code.push_str(&L::decl_float(&var, &d.clone()));
                var
            }

            SdfNode::PolarRepeat { child, count } => {
                let var = self.next_var();
                let n = self.param(*count as f32);
                let angle_var = self.next_var();
                let sector_var = self.next_var();
                let snapped_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();
                let pi2 = self.param(std::f32::consts::TAU);
                code.push_str(&L::decl_float(
                    &angle_var,
                    &format!("atan2({p}.z, {p}.x)", p = point_var),
                ));
                code.push_str(&L::decl_float(&sector_var, &format!("{} / {}", pi2, n)));
                code.push_str(&L::decl_float(
                    &snapped_var,
                    &format!("round({} / {}) * {}", angle_var, sector_var, sector_var),
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
                    )
                    .clone(),
                ));
                let d = self.transpile_node_inner(child, &new_p, code);
                code.push_str(&L::decl_float(&var, &d.clone()));
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

            // Exponential smooth operations
            SdfNode::ExpSmoothUnion { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                if self.mode == TranspileModeLang::Hardcoded && k.abs() < FOLD_EPSILON {
                    code.push_str(&L::decl_float(&var, &format!("min({}, {})", d_a, d_b)));
                    return var;
                }
                let inv_k = self.param(1.0 / k.max(1e-10));
                let ea = self.next_var();
                let eb = self.next_var();
                let s = self.next_var();
                code.push_str(&L::decl_float(&ea, &format!("exp(-{} * {})", inv_k, d_a)));
                code.push_str(&L::decl_float(&eb, &format!("exp(-{} * {})", inv_k, d_b)));
                code.push_str(&L::decl_float(&s, &format!("{} + {}", ea, eb)));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("-log(max({}, 1e-10)) * {}", s, self.param(*k)),
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
                let inv_k = self.param(1.0 / k.max(1e-10));
                let ea = self.next_var();
                let eb = self.next_var();
                let s = self.next_var();
                code.push_str(&L::decl_float(&ea, &format!("exp({} * {})", inv_k, d_a)));
                code.push_str(&L::decl_float(&eb, &format!("exp({} * {})", inv_k, d_b)));
                code.push_str(&L::decl_float(&s, &format!("{} + {}", ea, eb)));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("log(max({}, 1e-10)) * {}", s, self.param(*k)),
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
                let inv_k = self.param(1.0 / k.max(1e-10));
                let neg_b = self.next_var();
                let ea = self.next_var();
                let eb = self.next_var();
                let s = self.next_var();
                code.push_str(&L::decl_float(&neg_b, &format!("-{}", d_b)));
                code.push_str(&L::decl_float(&ea, &format!("exp({} * {})", inv_k, d_a)));
                code.push_str(&L::decl_float(&eb, &format!("exp({} * {})", inv_k, neg_b)));
                code.push_str(&L::decl_float(&s, &format!("{} + {}", ea, eb)));
                code.push_str(&L::decl_float(
                    &var,
                    &format!("log(max({}, 1e-10)) * {}", s, self.param(*k)),
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
                    )
                    .clone(),
                ));
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Animated { child, .. } => self.transpile_node_inner(child, point_var, code),
        }
    }
}
