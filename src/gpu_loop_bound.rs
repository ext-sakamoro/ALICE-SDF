//! Loop bounds of WGSL modules, checked on the naga IR before a module is compiled.
//!
//! A compute dispatch cannot be stopped once submitted: a shader loop without a bound keeps the GPU busy until the
//! driver gives up, and on some systems the whole display stalls with it. So every loop of a module this crate
//! creates must carry a counter with a bound, and the bound must have a stated maximum:
//!
//! - the loop has a local counter `v`, the only store to `v` inside the loop is `v = v + c` with a literal `c > 0`,
//!   done in the `continuing` block (or at the top level of the body with no `continue` before it);
//! - a top-level `if` of the body (or the `break if`) breaks when `v` reaches the bound: `v < B` / `v <= B` /
//!   `B > v` / `B >= v` keep looping, `v >= B` / `v > B` / `v == B` break;
//! - `B` is a literal, a module constant, or a named host value (a uniform member, an override or an array
//!   length) whose maximum the caller states in [`HostMaxima`]; the host checks the value against that maximum
//!   before it submits.
//!
//! The iteration count of a loop is `B + 1` at most (the start value is not read: a counter starting above 0 only
//! shortens the loop). Nested loops and loops in called functions multiply; [`LoopReport::worst`] is the largest
//! product over all paths of an entry point.
//!
//! Anything else (a `while` on a data value, a stack drained until empty, a counter that is reset inside the loop)
//! is an [`UnboundedLoop`].

use std::collections::BTreeMap;

use wgpu::naga;
use wgpu::naga::{BinaryOperator, Block, Expression, Function, Handle, Literal, Module, Statement};

/// The most loop iterations one invocation of a module may do: the product of nested loop bounds (through calls)
/// over any path of an entry point ([`LoopReport::worst`]).
///
/// Measured 2026-11-16 over every module the crate creates for its full test corpus (1211 modules): the worst
/// product is 4097 (the staircase helper at `primitives::MAX_STAIR_STEPS`), the next 2091 (a loft's volume shader).
/// 2^16 is 16 times that: room for a short loop around the largest helper, while one invocation stays at most 2^16
/// iterations. A module over it is refused like an unbounded one. Bounding the work of a whole submit (invocations
/// times this) is the dispatching code's budget, not this check's.
pub const MAX_TRIP_PRODUCT: u64 = 1 << 16;

/// The stated maxima of the host values a module's loops are bounded by.
///
/// Keyed by name: `"<global>.<member>"` for a uniform struct member (`"U.steps"`), `"<global>"` for a whole uniform scalar, `"<override>"` for an override,
/// `"len(<global>)"` for a runtime array length.
pub type HostMaxima = BTreeMap<String, u64>;

/// A loop of a module whose iteration count has no stated bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnboundedLoop {
    /// The function holding the loop.
    pub function: String,
    /// Why the loop is not bounded.
    pub reason: String,
}

/// The loop bounds of a module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoopReport {
    /// Loops found (all functions, each counted once).
    pub loops: usize,
    /// Loops without a stated bound.
    pub unbounded: Vec<UnboundedLoop>,
    /// The largest product of nested loop bounds over the entry points (1 for a module without loops); `u64::MAX`
    /// when it overflows.
    pub worst: u64,
}

/// Bound the loops of a parsed module.
#[must_use]
pub fn check_module(module: &Module, maxima: &HostMaxima) -> LoopReport {
    let mut cx = Cx {
        module,
        maxima,
        report: LoopReport::default(),
        fn_bound: BTreeMap::new(),
    };
    // functions first (callees precede callers in naga's arena)
    for (h, f) in module.functions.iter() {
        let name = f.name.clone().unwrap_or_else(|| format!("fn{}", h.index()));
        let b = cx.block(f, &name, &f.body);
        cx.fn_bound.insert(h, b);
    }
    let mut worst = 1u64;
    for ep in &module.entry_points {
        let b = cx.block(&ep.function, &ep.name, &ep.function.body);
        worst = worst.max(b);
    }
    cx.report.worst = worst;
    cx.report
}

struct Cx<'a> {
    module: &'a Module,
    maxima: &'a HostMaxima,
    report: LoopReport,
    fn_bound: BTreeMap<Handle<Function>, u64>,
}

impl Cx<'_> {
    /// The largest iteration product of any path through `block`.
    fn block(&mut self, f: &Function, name: &str, block: &Block) -> u64 {
        let mut worst = 1u64;
        for st in block.iter() {
            let b = match st {
                Statement::Block(b) => self.block(f, name, b),
                Statement::If { accept, reject, .. } => {
                    self.block(f, name, accept).max(self.block(f, name, reject))
                }
                Statement::Switch { cases, .. } => cases
                    .iter()
                    .map(|c| self.block(f, name, &c.body))
                    .max()
                    .unwrap_or(1),
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    self.report.loops += 1;
                    let inner = self
                        .block(f, name, body)
                        .max(self.block(f, name, continuing));
                    match self.loop_bound(f, body, continuing, *break_if) {
                        Ok(n) => n.saturating_add(1).saturating_mul(inner),
                        Err(reason) => {
                            self.report.unbounded.push(UnboundedLoop {
                                function: name.to_string(),
                                reason,
                            });
                            u64::MAX
                        }
                    }
                }
                Statement::Call { function, .. } => *self.fn_bound.get(function).unwrap_or(&1),
                _ => 1,
            };
            worst = worst.max(b);
        }
        worst
    }

    /// The bound `B` of a loop's counter, or why there is none.
    fn loop_bound(
        &self,
        f: &Function,
        body: &Block,
        continuing: &Block,
        break_if: Option<Handle<Expression>>,
    ) -> Result<u64, String> {
        // candidate exits: top-level `if` of the body whose one branch is just `break`, and `break if`
        let mut exits: Vec<(Handle<Expression>, bool)> = Vec::new(); // (condition, breaks when true)
        for st in body.iter() {
            if let Statement::If {
                condition,
                accept,
                reject,
            } = st
            {
                if is_break(accept) && reject.is_empty() {
                    exits.push((*condition, true));
                } else if accept.is_empty() && is_break(reject) {
                    exits.push((*condition, false));
                }
            }
        }
        if let Some(c) = break_if {
            exits.push((c, true));
        }
        if exits.is_empty() {
            return Err(
                "no exit at the top level of the loop (a break inside a branch is not a bound)"
                    .into(),
            );
        }
        let mut why = Vec::new();
        for (cond, breaks_when_true) in exits {
            match self.exit_bound(f, cond, breaks_when_true) {
                Ok((var, bound)) => match Self::counter_ok(f, var, body, continuing) {
                    Ok(()) => return Ok(bound),
                    Err(e) => why.push(e),
                },
                Err(e) => why.push(e),
            }
        }
        Err(why.join("; "))
    }

    /// For an exit condition, the counter local it compares and the bound it compares with.
    fn exit_bound(
        &self,
        f: &Function,
        cond: Handle<Expression>,
        breaks_when_true: bool,
    ) -> Result<(Handle<naga::LocalVariable>, u64), String> {
        let (mut cond, mut breaks) = (cond, breaks_when_true);
        while let Expression::Unary {
            op: naga::UnaryOperator::LogicalNot,
            expr,
        } = f.expressions[cond]
        {
            cond = expr;
            breaks = !breaks;
        }
        let Expression::Binary { op, left, right } = f.expressions[cond] else {
            return Err("the exit is not a comparison".into());
        };
        // normalise to `counter OP bound`
        let (var, bound_e, op) = match (local_of(f, left), local_of(f, right)) {
            (Some(v), _) if self.value(f, right).is_ok() => (v, right, op),
            (_, Some(v)) if self.value(f, left).is_ok() => (v, left, flip(op)),
            _ => {
                return Err(format!(
                    "the exit compares no counter with a bounded value ({})",
                    self.value(f, right)
                        .err()
                        .or_else(|| self.value(f, left).err())
                        .unwrap_or_default()
                ))
            }
        };
        let keeps_going_below = matches!(
            (op, breaks),
            (BinaryOperator::Less | BinaryOperator::LessEqual, false)
                | (
                    BinaryOperator::GreaterEqual | BinaryOperator::Greater | BinaryOperator::Equal,
                    true
                )
        );
        if !keeps_going_below {
            return Err("the exit does not stop the counter on its way up".into());
        }
        Ok((var, self.value(f, bound_e)?))
    }

    /// The counter is stored only by `v = v + c`, `c > 0` literal, once per iteration before any `continue`.
    fn counter_ok(
        f: &Function,
        var: Handle<naga::LocalVariable>,
        body: &Block,
        continuing: &Block,
    ) -> Result<(), String> {
        let mut stores = Vec::new();
        collect_stores(f, var, body, &mut stores);
        collect_stores(f, var, continuing, &mut stores);
        if stores.len() != 1 {
            return Err(format!(
                "the counter is stored {} times in the loop (needs exactly 1)",
                stores.len()
            ));
        }
        if !is_increment(f, var, stores[0]) {
            return Err("the counter's store is not `v = v + <literal > 0>`".into());
        }
        let top = |b: &Block| {
            b.iter()
                .any(|s| matches!(s, Statement::Store { pointer, value } if local_of_ptr(f, *pointer) == Some(var) && *value == stores[0]))
        };
        if top(continuing) {
            return Ok(());
        }
        // in the body: at the top level, and no `continue` before it
        for st in body.iter() {
            if let Statement::Store { pointer, value } = st {
                if local_of_ptr(f, *pointer) == Some(var) && *value == stores[0] {
                    return Ok(());
                }
            }
            if has_continue(st) {
                return Err("a `continue` can skip the counter's increment".into());
            }
        }
        Err("the counter's increment is inside a branch".into())
    }

    /// The value of a bound expression: a literal, a constant, or a named host value with a stated maximum.
    fn value(&self, f: &Function, e: Handle<Expression>) -> Result<u64, String> {
        match &f.expressions[e] {
            Expression::Literal(l) => {
                lit(l).ok_or_else(|| "a negative or non-integer literal bound".into())
            }
            Expression::Constant(c) => {
                let init = self.module.constants[*c].init;
                match &self.module.global_expressions[init] {
                    Expression::Literal(l) => {
                        lit(l).ok_or_else(|| "a non-integer constant bound".into())
                    }
                    _ => Err("a constant bound that is not a literal".into()),
                }
            }
            Expression::As { expr, .. } => self.value(f, *expr),
            _ => {
                let name = self.host_name(f, e).ok_or_else(|| {
                    "a bound that is a computed value (only literals, constants and named host values)".to_string()
                })?;
                self.maxima
                    .get(&name)
                    .copied()
                    .ok_or_else(|| format!("the host value `{name}` has no stated maximum"))
            }
        }
    }

    /// `"G.member"` / `"G"` / `"len(G)"` / override name for a host-provided value.
    fn host_name(&self, f: &Function, e: Handle<Expression>) -> Option<String> {
        match &f.expressions[e] {
            Expression::Load { pointer } => self.place(f, *pointer),
            Expression::Override(o) => self.module.overrides[*o].name.clone(),
            Expression::ArrayLength(p) => self.place(f, *p).map(|n| format!("len({n})")),
            Expression::As { expr, .. } => self.host_name(f, *expr),
            _ => None,
        }
    }

    fn place(&self, f: &Function, p: Handle<Expression>) -> Option<String> {
        match &f.expressions[p] {
            Expression::GlobalVariable(g) => {
                let gv = &self.module.global_variables[*g];
                if !matches!(
                    gv.space,
                    naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
                ) {
                    return None;
                }
                gv.name.clone()
            }
            Expression::AccessIndex { base, index } => {
                let b = self.place(f, *base)?;
                let member = self.member_name(f, *base, *index)?;
                Some(format!("{b}.{member}"))
            }
            _ => None,
        }
    }

    fn member_name(&self, f: &Function, base: Handle<Expression>, index: u32) -> Option<String> {
        let Expression::GlobalVariable(g) = f.expressions[base] else {
            return Some(index.to_string());
        };
        let ty = &self.module.types[self.module.global_variables[g].ty];
        match &ty.inner {
            naga::TypeInner::Struct { members, .. } => {
                members.get(index as usize).and_then(|m| m.name.clone())
            }
            _ => Some(index.to_string()),
        }
    }
}

/// Parse a shader source (WGSL or GLSL) into a naga module
///
/// # Errors
///
/// The parse error, as text.
pub fn parse_source(source: &wgpu::ShaderSource<'_>) -> Result<Module, String> {
    match source {
        wgpu::ShaderSource::Wgsl(text) => {
            naga::front::wgsl::parse_str(text).map_err(|e| e.emit_to_string(text))
        }
        wgpu::ShaderSource::Glsl {
            shader,
            stage,
            defines,
        } => naga::front::glsl::Frontend::default()
            .parse(
                &naga::front::glsl::Options {
                    stage: *stage,
                    defines: defines.clone(),
                },
                shader,
            )
            .map_err(|e| e.emit_to_string(shader)),
        _ => Err("unsupported shader source".to_string()),
    }
}

/// The module's loops must all be bounded: [`crate::compiled::GpuError::ShaderCompilation`] starting with
/// [`crate::gpu_context::refusal::UNBOUNDED_SHADER_LOOP`] otherwise
///
/// # Errors
///
/// `ShaderCompilation`: an unbounded loop (the prefix above), or the parse error.
pub fn require_bounded(
    label: &str,
    source: &wgpu::ShaderSource<'_>,
    maxima: &HostMaxima,
) -> Result<LoopReport, crate::compiled::GpuError> {
    let module = parse_source(source).map_err(crate::compiled::GpuError::ShaderCompilation)?;
    let report = check_module(&module, maxima);
    if report.unbounded.is_empty() && report.worst > MAX_TRIP_PRODUCT {
        return Err(crate::compiled::GpuError::ShaderCompilation(format!(
            "{}shader {label}: one invocation may run {} loop iterations, over the ceiling {MAX_TRIP_PRODUCT} (gpu_loop_bound::MAX_TRIP_PRODUCT); a union of many polygon operands costs rows x the largest polygon's vertices per invocation: split the tree or evaluate it on the CPU path",
            crate::gpu_context::refusal::UNBOUNDED_SHADER_LOOP,
            report.worst
        )));
    }
    if let Some(first) = report.unbounded.first() {
        return Err(crate::compiled::GpuError::ShaderCompilation(format!(
            "{}shader {label}: {} loop(s) without a stated bound: fn {}: {}",
            crate::gpu_context::refusal::UNBOUNDED_SHADER_LOOP,
            report.unbounded.len(),
            first.function,
            first.reason
        )));
    }
    Ok(report)
}

fn lit(l: &Literal) -> Option<u64> {
    match *l {
        Literal::U32(v) => Some(u64::from(v)),
        Literal::U64(v) => Some(v),
        Literal::I32(v) => u64::try_from(v).ok(),
        Literal::I64(v) | Literal::AbstractInt(v) => u64::try_from(v).ok(),
        _ => None,
    }
}

const fn flip(op: BinaryOperator) -> BinaryOperator {
    match op {
        BinaryOperator::Less => BinaryOperator::Greater,
        BinaryOperator::LessEqual => BinaryOperator::GreaterEqual,
        BinaryOperator::Greater => BinaryOperator::Less,
        BinaryOperator::GreaterEqual => BinaryOperator::LessEqual,
        o => o,
    }
}

fn is_break(b: &Block) -> bool {
    b.len() == 1 && matches!(b.iter().next(), Some(Statement::Break))
}

fn has_continue(st: &Statement) -> bool {
    match st {
        Statement::Continue => true,
        Statement::Block(b) => b.iter().any(has_continue),
        Statement::If { accept, reject, .. } => {
            accept.iter().any(has_continue) || reject.iter().any(has_continue)
        }
        Statement::Switch { cases, .. } => cases.iter().any(|c| c.body.iter().any(has_continue)),
        _ => false, // a nested loop's `continue` is its own
    }
}

/// `Load(LocalVariable v)` -> `v`
fn local_of(f: &Function, e: Handle<Expression>) -> Option<Handle<naga::LocalVariable>> {
    match f.expressions[e] {
        Expression::Load { pointer } => local_of_ptr(f, pointer),
        Expression::As { expr, .. } => local_of(f, expr),
        _ => None,
    }
}

fn local_of_ptr(f: &Function, p: Handle<Expression>) -> Option<Handle<naga::LocalVariable>> {
    match f.expressions[p] {
        Expression::LocalVariable(v) => Some(v),
        _ => None,
    }
}

fn collect_stores(
    f: &Function,
    var: Handle<naga::LocalVariable>,
    b: &Block,
    out: &mut Vec<Handle<Expression>>,
) {
    for st in b.iter() {
        match st {
            Statement::Store { pointer, value } if local_of_ptr(f, *pointer) == Some(var) => {
                out.push(*value);
            }
            Statement::Block(b) => collect_stores(f, var, b, out),
            Statement::If { accept, reject, .. } => {
                collect_stores(f, var, accept, out);
                collect_stores(f, var, reject, out);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    collect_stores(f, var, &c.body, out);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                collect_stores(f, var, body, out);
                collect_stores(f, var, continuing, out);
            }
            _ => {}
        }
    }
}

fn is_increment(f: &Function, var: Handle<naga::LocalVariable>, value: Handle<Expression>) -> bool {
    let Expression::Binary {
        op: BinaryOperator::Add,
        left,
        right,
    } = f.expressions[value]
    else {
        return false;
    };
    let pos = |e: Handle<Expression>| matches!(&f.expressions[e], Expression::Literal(l) if lit(l).is_some_and(|v| v > 0));
    (local_of(f, left) == Some(var) && pos(right)) || (local_of(f, right) == Some(var) && pos(left))
}

#[cfg(test)]
mod tests {
    use super::{check_module, parse_source, HostMaxima, LoopReport};

    fn check_wgsl(wgsl: &str, maxima: &HostMaxima) -> Result<LoopReport, String> {
        Ok(check_module(
            &parse_source(&wgpu::ShaderSource::Wgsl(wgsl.into()))?,
            maxima,
        ))
    }

    fn wrap(body: &str) -> String {
        format!(
            "struct U {{ steps: u32, n: u32 }}\n@group(0) @binding(0) var<uniform> u: U;\n@group(0) @binding(1) var<storage, read_write> o: array<f32>;\n@compute @workgroup_size(1)\nfn main() {{\n var acc = 0.0;\n{body}\n o[0] = acc;\n}}\n"
        )
    }

    fn maxima() -> HostMaxima {
        HostMaxima::from([("u.steps".to_string(), 4000)])
    }

    #[test]
    fn a_for_loop_on_a_literal_is_bounded() {
        let r = check_wgsl(
            &wrap("for (var i = 0u; i < 30u; i = i + 1u) { acc = acc + 1.0; }"),
            &maxima(),
        )
        .unwrap();
        assert_eq!((r.loops, r.unbounded.len(), r.worst), (1, 0, 31));
    }

    #[test]
    fn a_uniform_bound_needs_a_stated_maximum() {
        let src = wrap("for (var i = 0u; i < u.steps; i = i + 1u) { acc = acc + 1.0; }");
        assert_eq!(check_wgsl(&src, &maxima()).unwrap().worst, 4001);
        let r = check_wgsl(&src, &HostMaxima::new()).unwrap();
        assert_eq!(r.unbounded.len(), 1, "{r:?}");
        assert!(r.unbounded[0].reason.contains("u.steps"), "{r:?}");
    }

    #[test]
    fn nested_loops_multiply() {
        let r = check_wgsl(
            &wrap("for (var i = 0u; i < 9u; i = i + 1u) { for (var j = 0u; j < 3u; j = j + 1u) { acc = acc + 1.0; } }"),
            &maxima(),
        )
        .unwrap();
        assert_eq!((r.loops, r.unbounded.len(), r.worst), (2, 0, 40));
    }

    #[test]
    fn a_stack_drained_until_empty_is_unbounded() {
        let r = check_wgsl(
            &wrap("var n = 1u; loop { if (n == 0u) { break; } n = n - 1u; if (acc < 3.0) { n = n + 2u; } acc = acc + 1.0; }"),
            &maxima(),
        )
        .unwrap();
        assert_eq!((r.loops, r.unbounded.len()), (1, 1), "{r:?}");
    }

    #[test]
    fn an_unbounded_module_is_refused_with_the_stable_prefix() {
        let src = wrap("while (acc < 10.0) { acc = acc + 0.5; }");
        let e = super::require_bounded(
            "probe",
            &wgpu::ShaderSource::Wgsl(src.as_str().into()),
            &maxima(),
        )
        .expect_err("an unbounded loop is refused");
        let crate::compiled::GpuError::ShaderCompilation(m) = e else {
            panic!("refused as ShaderCompilation, got {e:?}");
        };
        assert!(
            m.starts_with(crate::gpu_context::refusal::UNBOUNDED_SHADER_LOOP),
            "{m}"
        );
        assert!(m.contains("shader probe"), "{m}");
        let ok = wrap("for (var i = 0u; i < 30u; i = i + 1u) { acc = acc + 1.0; }");
        assert!(super::require_bounded(
            "ok",
            &wgpu::ShaderSource::Wgsl(ok.as_str().into()),
            &maxima()
        )
        .is_ok());
    }

    #[test]
    fn a_bounded_module_over_the_trip_ceiling_is_refused() {
        // 300 x 300 nested = 301 * 301 = 90601 iterations per invocation > 2^16
        let deep = wrap("for (var i = 0u; i < 300u; i = i + 1u) { for (var j = 0u; j < 300u; j = j + 1u) { acc = acc + 1.0; } }");
        let e = super::require_bounded(
            "deep",
            &wgpu::ShaderSource::Wgsl(deep.as_str().into()),
            &maxima(),
        )
        .expect_err("over the ceiling");
        assert!(e.to_string().contains("over the ceiling"), "{e}");
        // 200 x 200 = 40401 is under it
        let ok = wrap("for (var i = 0u; i < 200u; i = i + 1u) { for (var j = 0u; j < 200u; j = j + 1u) { acc = acc + 1.0; } }");
        assert!(super::require_bounded(
            "ok",
            &wgpu::ShaderSource::Wgsl(ok.as_str().into()),
            &maxima()
        )
        .is_ok());
    }

    #[test]
    fn a_while_on_data_is_unbounded() {
        let r = check_wgsl(&wrap("while (acc < 10.0) { acc = acc + 0.5; }"), &maxima()).unwrap();
        assert_eq!(r.unbounded.len(), 1, "{r:?}");
    }

    #[test]
    fn a_counter_reset_inside_the_loop_is_unbounded() {
        let r = check_wgsl(
            &wrap("for (var i = 0u; i < 30u; i = i + 1u) { if (acc > 2.0) { i = 0u; } acc = acc + 1.0; }"),
            &maxima(),
        )
        .unwrap();
        assert_eq!(r.unbounded.len(), 1, "{r:?}");
    }

    #[test]
    fn a_continue_before_a_body_increment_is_unbounded() {
        let r = check_wgsl(
            &wrap("var i = 0u; loop { if (i >= 30u) { break; } if (acc > 2.0) { continue; } i = i + 1u; acc = acc + 1.0; }"),
            &maxima(),
        )
        .unwrap();
        assert_eq!(r.unbounded.len(), 1, "{r:?}");
    }

    #[test]
    fn a_capped_manual_loop_is_bounded() {
        let r = check_wgsl(
            &wrap("var it = 0u; var n = 1u; loop { if (it >= 60u) { break; } it = it + 1u; if (n == 0u) { break; } n = n - 1u; acc = acc + 1.0; }"),
            &maxima(),
        )
        .unwrap();
        assert_eq!((r.unbounded.len(), r.worst), (0, 61), "{r:?}");
    }
}
