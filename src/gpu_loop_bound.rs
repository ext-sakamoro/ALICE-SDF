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

/// The longest shader text [`parse_source`] reads, in bytes.
///
/// Longer text is refused before it is parsed, so a caller's text
/// ([`crate::compiled::GpuEvaluator::from_wgsl`] / `from_glsl_compute`) cannot make the parser run without bound.
///
/// MEASURED_LONGEST is the longest module the crate generates over its test corpus (the shader loop census prints
/// it and asserts it is under this); the maximum is a stated ceiling above it.
pub const MAX_SHADER_SOURCE_BYTES: usize = 16 << 20;

/// The stated maxima of the host values a module's loops are bounded by.
///
/// Keyed by name: `"<global>.<member>"` for a uniform struct member (`"U.steps"`), `"<global>"` for a whole uniform scalar, `"<override>"` for an override,
/// `"len(<global>)"` for a runtime array length.
pub type HostMaxima = BTreeMap<String, u64>;

/// A stated loop-iteration ceiling for one kernel whose work is bounded by its own counter.
///
/// Its loops are all bounded, but their product is over [`MAX_TRIP_PRODUCT`] (the sheet march's per-ray work).
///
/// Only `GpuContext::shader_module_with_ceiling` (crate-private) takes it, and every kernel that does is listed
/// by label in `tests/test_gpu_lifecycle_oracle.rs` (the census is red on an unlisted one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCeiling {
    /// The most loop iterations one invocation may do (the product [`LoopReport::worst`] is checked against).
    pub max_trip_product: u64,
    /// Why this kernel may go over [`MAX_TRIP_PRODUCT`], and what bounds its real work.
    pub reason: &'static str,
    /// The work counter every loop iteration of the kernel ticks ([`check_counted`]); the ceiling is checked against
    /// [`CountedReport::bound`], not the product of the loop bounds.
    pub counter: WorkCounter,
}

/// A kernel's work counter: a private `u32` global stored only by the function `tick`, which adds a literal > 0 and
/// returns whether the count is still within the host value `budget`.
///
/// Every loop of the kernel starts its body with `if (!tick()) { break; }` (after its own exit checks only), so no
/// iteration past the budget does work; the loops still each have their own bound ([`check_module`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkCounter {
    /// The tick function's name.
    pub tick: &'static str,
    /// The host value of the budget, as [`HostMaxima`] names it (`"U.work_budget"`); its stated maximum bounds the
    /// ticks.
    pub budget: &'static str,
}

/// What [`check_counted`] found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CountedReport {
    /// Why the kernel's loops are not all counted (empty when they are).
    pub problems: Vec<String>,
    /// The stated maximum of the budget.
    pub budget: u64,
    /// Loop instances in the expanded call tree of the entry points: once the budget is spent each can be entered
    /// at most once more (its first tick fails).
    pub instances: u64,
    /// The most loops entered between two ticks (directly in one loop body or entry function, calls expanded).
    pub unit: u64,
    /// The most loop iterations one invocation can do: `(budget + instances + 1) x (unit + 1)` (saturating).
    pub bound: u64,
}

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
                Statement::Call { function, .. } => {
                    if let Some(b) = self.fn_bound.get(function) {
                        *b
                    } else {
                        // naga orders callees first; a callee not checked yet has no bound to use
                        self.report.unbounded.push(UnboundedLoop {
                            function: name.to_string(),
                            reason: format!(
                                "calls fn{} before its loops were checked",
                                function.index()
                            ),
                        });
                        u64::MAX
                    }
                }
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
    let len = match source {
        wgpu::ShaderSource::Wgsl(text) => text.len(),
        wgpu::ShaderSource::Glsl { shader, .. } => shader.len(),
        _ => 0,
    };
    if len > MAX_SHADER_SOURCE_BYTES {
        return Err(format!(
            "shader text of {len} bytes, over the most {MAX_SHADER_SOURCE_BYTES} (gpu_loop_bound::MAX_SHADER_SOURCE_BYTES): not parsed"
        ));
    }
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
    require_bounded_under(label, source, maxima, None)
}

/// [`require_bounded`] against `ceiling` instead of [`MAX_TRIP_PRODUCT`] when one is given.
///
/// # Errors
///
/// As [`require_bounded`]; also a ceiling without a reason.
pub fn require_bounded_under(
    label: &str,
    source: &wgpu::ShaderSource<'_>,
    maxima: &HostMaxima,
    ceiling: Option<KernelCeiling>,
) -> Result<LoopReport, crate::compiled::GpuError> {
    if ceiling.is_some_and(|c| c.reason.trim().is_empty()) {
        return Err(crate::compiled::GpuError::ShaderCompilation(format!(
            "{}shader {label}: a kernel ceiling without a reason",
            crate::gpu_context::refusal::UNBOUNDED_SHADER_LOOP
        )));
    }
    let module = parse_source(source).map_err(crate::compiled::GpuError::ShaderCompilation)?;
    let report = check_module(&module, maxima);
    if let (Some(c), true) = (ceiling, report.unbounded.is_empty()) {
        let counted = check_counted(&module, maxima, c.counter);
        if let Some(first) = counted.problems.first() {
            return Err(crate::compiled::GpuError::ShaderCompilation(format!(
                "{}shader {label}: {} loop(s) not counted by the work counter: {first}",
                crate::gpu_context::refusal::UNBOUNDED_SHADER_LOOP,
                counted.problems.len()
            )));
        }
        if counted.bound > c.max_trip_product {
            return Err(crate::compiled::GpuError::ShaderCompilation(format!(
                "{}shader {label}: one invocation may run {} counted loop iterations, over its ceiling {} ({})",
                crate::gpu_context::refusal::UNBOUNDED_SHADER_LOOP,
                counted.bound,
                c.max_trip_product,
                c.reason
            )));
        }
        return Ok(report);
    }
    let limit = MAX_TRIP_PRODUCT;
    if report.unbounded.is_empty() && report.worst > limit {
        return Err(crate::compiled::GpuError::ShaderCompilation(format!(
            "{}shader {label}: one invocation may run {} loop iterations, over the ceiling {limit} ({}); a union of many polygon operands costs rows x the largest polygon's vertices per invocation: split the tree or evaluate it on the CPU path",
            crate::gpu_context::refusal::UNBOUNDED_SHADER_LOOP,
            report.worst,
            "gpu_loop_bound::MAX_TRIP_PRODUCT"
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

/// Check that every loop of `module` is counted by `counter` and bound the iterations of one invocation.
#[must_use]
pub fn check_counted(module: &Module, maxima: &HostMaxima, counter: WorkCounter) -> CountedReport {
    let mut r = CountedReport::default();
    let Some(budget) = maxima.get(counter.budget).copied() else {
        r.problems.push(format!(
            "the budget `{}` has no stated maximum",
            counter.budget
        ));
        return r;
    };
    r.budget = budget;
    let Some((tick, tf)) = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some(counter.tick))
    else {
        r.problems
            .push(format!("no tick function `{}`", counter.tick));
        return r;
    };
    let cx = Cx {
        module,
        maxima,
        report: LoopReport::default(),
        fn_bound: BTreeMap::new(),
    };
    let Some((g, _step)) = tick_counter(&cx, tf, counter.budget) else {
        r.problems.push(format!(
            "the tick function `{}` is not `G = G + <literal > 0>; return G <= {}` on a private global G",
            counter.tick, counter.budget
        ));
        return r;
    };
    if module.global_variables[g].init.is_some() {
        r.problems
            .push("the work counter has an initializer".into());
    }
    if !matches!(module.types[module.global_variables[g].ty].inner, naga::TypeInner::Scalar(s) if s == naga::Scalar::U32)
    {
        r.problems
            .push("the work counter is not a u32 (the overflow bound below is for u32)".into());
    }
    if !budget_read_only(module, counter.budget) {
        r.problems.push(format!(
            "the budget `{}` is not a uniform or read-only storage value: the shader could rewrite it",
            counter.budget
        ));
    }
    let mut inst: BTreeMap<Handle<Function>, u64> = BTreeMap::new();
    let mut entered: BTreeMap<Handle<Function>, u64> = BTreeMap::new();
    let mut unit = 0u64;
    let walk = |name: &str,
                f: &Function,
                inst: &BTreeMap<_, _>,
                entered: &BTreeMap<_, _>,
                problems: &mut Vec<String>,
                unit: &mut u64| {
        if stores_global(f, &f.body, g) {
            problems.push(format!(
                "fn {name}: stores the work counter outside the tick function"
            ));
        }
        if passes_global(f, &f.body, g) {
            problems.push(format!(
                "fn {name}: passes the work counter to a call (the callee could store to it)"
            ));
        }
        count_loops(f, name, &f.body, tick, inst, entered, problems, unit)
    };
    for (h, f) in module.functions.iter() {
        if h == tick {
            continue;
        }
        let name = f.name.clone().unwrap_or_else(|| format!("fn{}", h.index()));
        let (i, e) = walk(&name, f, &inst, &entered, &mut r.problems, &mut unit);
        inst.insert(h, i);
        entered.insert(h, e);
    }
    let mut instances = 0u64;
    for ep in &module.entry_points {
        let (i, e) = walk(
            &ep.name,
            &ep.function,
            &inst,
            &entered,
            &mut r.problems,
            &mut unit,
        );
        instances = instances.saturating_add(i);
        unit = unit.max(e);
    }
    r.instances = instances;
    r.unit = unit;
    r.bound = budget
        .saturating_add(instances)
        .saturating_add(1)
        .saturating_mul(unit.saturating_add(1));
    r
}

/// The private global the tick function counts with, when its body is exactly the counted form.
fn tick_counter(
    cx: &Cx<'_>,
    f: &Function,
    budget: &str,
) -> Option<(Handle<naga::GlobalVariable>, u64)> {
    let st: Vec<&Statement> = flat(&f.body)
        .into_iter()
        .filter(|s| !matches!(s, Statement::Emit(_)))
        .collect();
    let [Statement::Store { pointer, value }, Statement::Return { value: Some(ret) }] = st[..]
    else {
        return None;
    };
    let g = private_global(cx.module, f, *pointer)?;
    let Expression::Binary {
        op: BinaryOperator::Add,
        left,
        right,
    } = f.expressions[*value]
    else {
        return None;
    };
    let loads_g = |e: Handle<Expression>| matches!(f.expressions[e], Expression::Load { pointer } if private_global(cx.module, f, pointer) == Some(g));
    let step = |e: Handle<Expression>| match &f.expressions[e] {
        Expression::Literal(l) => lit(l).filter(|&v| v > 0),
        _ => None,
    };
    let c = match (loads_g(left), loads_g(right)) {
        (true, _) => step(right),
        (_, true) => step(left),
        _ => None,
    }?;
    let Expression::Binary { op, left, right } = f.expressions[*ret] else {
        return None;
    };
    let names_budget = |e: Handle<Expression>| cx.host_name(f, e).as_deref() == Some(budget);
    let within = matches!(
        (
            op,
            loads_g(left),
            names_budget(right),
            loads_g(right),
            names_budget(left)
        ),
        (
            BinaryOperator::LessEqual | BinaryOperator::Less,
            true,
            true,
            _,
            _
        ) | (
            BinaryOperator::GreaterEqual | BinaryOperator::Greater,
            _,
            _,
            true,
            true
        )
    );
    within.then_some((g, c))
}

/// Whether the budget `name` (`global.member`) is a value the shader cannot write: a uniform, or read-only storage.
fn budget_read_only(module: &Module, name: &str) -> bool {
    let root = name.split('.').next().unwrap_or(name);
    module
        .global_variables
        .iter()
        .find(|(_, g)| g.name.as_deref() == Some(root))
        .is_some_and(|(_, g)| match g.space {
            naga::AddressSpace::Uniform => true,
            naga::AddressSpace::Storage { access } => !access.contains(naga::StorageAccess::STORE),
            _ => false,
        })
}

/// The global a pointer expression points into, through member and element accesses.
fn ptr_root(f: &Function, mut p: Handle<Expression>) -> Option<Handle<naga::GlobalVariable>> {
    loop {
        match f.expressions[p] {
            Expression::GlobalVariable(g) => return Some(g),
            Expression::AccessIndex { base, .. } | Expression::Access { base, .. } => p = base,
            _ => return None,
        }
    }
}

/// Whether `block` passes a pointer into `g` to a call (the callee could then store to it through its argument).
fn passes_global(f: &Function, block: &Block, g: Handle<naga::GlobalVariable>) -> bool {
    block.iter().any(|st| match st {
        Statement::Call { arguments, .. } => arguments.iter().any(|a| ptr_root(f, *a) == Some(g)),
        Statement::Block(b) => passes_global(f, b, g),
        Statement::If { accept, reject, .. } => {
            passes_global(f, accept, g) || passes_global(f, reject, g)
        }
        Statement::Switch { cases, .. } => cases.iter().any(|c| passes_global(f, &c.body, g)),
        Statement::Loop {
            body, continuing, ..
        } => passes_global(f, body, g) || passes_global(f, continuing, g),
        _ => false,
    })
}

fn private_global(
    module: &Module,
    f: &Function,
    p: Handle<Expression>,
) -> Option<Handle<naga::GlobalVariable>> {
    match f.expressions[p] {
        Expression::GlobalVariable(g)
            if module.global_variables[g].space == naga::AddressSpace::Private =>
        {
            Some(g)
        }
        _ => None,
    }
}

/// Whether `block` stores to `g` (or into it).
fn stores_global(f: &Function, block: &Block, g: Handle<naga::GlobalVariable>) -> bool {
    block.iter().any(|st| match st {
        Statement::Store { pointer, .. } => ptr_root(f, *pointer) == Some(g),
        Statement::Atomic { pointer, .. } => ptr_root(f, *pointer) == Some(g),
        Statement::Block(b) => stores_global(f, b, g),
        Statement::If { accept, reject, .. } => {
            stores_global(f, accept, g) || stores_global(f, reject, g)
        }
        Statement::Switch { cases, .. } => cases.iter().any(|c| stores_global(f, &c.body, g)),
        Statement::Loop {
            body, continuing, ..
        } => stores_global(f, body, g) || stores_global(f, continuing, g),
        _ => false,
    })
}

/// The statements of `block` with nested plain blocks spliced in.
fn flat(block: &Block) -> Vec<&Statement> {
    let mut out = Vec::new();
    for st in block.iter() {
        if let Statement::Block(b) = st {
            out.extend(flat(b));
        } else {
            out.push(st);
        }
    }
    out
}

/// Whether a statement may run before a loop's tick: an emit, or an `if` whose branches only break.
fn exit_check(st: &Statement) -> bool {
    match st {
        Statement::Emit(_) => true,
        Statement::If { accept, reject, .. } => {
            (is_break(accept) || accept.is_empty()) && (is_break(reject) || reject.is_empty())
        }
        _ => false,
    }
}

/// Whether a loop body starts (after exit checks only) with `if (!tick()) { break; }`.
fn ticks_first(f: &Function, body: &Block, tick: Handle<Function>) -> bool {
    let st = flat(body);
    let mut i = 0;
    while i < st.len() && exit_check(st[i]) {
        i += 1;
    }
    let Some(Statement::Call {
        function,
        result: Some(res),
        ..
    }) = st.get(i)
    else {
        return false;
    };
    if *function != tick {
        return false;
    }
    let mut j = i + 1;
    while matches!(st.get(j), Some(Statement::Emit(_))) {
        j += 1;
    }
    let Some(Statement::If {
        condition,
        accept,
        reject,
    }) = st.get(j)
    else {
        return false;
    };
    let not_res = matches!(f.expressions[*condition], Expression::Unary { op: naga::UnaryOperator::LogicalNot, expr } if expr == *res);
    not_res && is_break(accept) && reject.is_empty()
}

/// For `block`: (loop instances inside it, calls expanded; loops entered directly, calls expanded). Records loops
/// that do not tick first and updates `unit` with each loop body's direct entries.
#[allow(clippy::too_many_arguments)]
fn count_loops(
    f: &Function,
    name: &str,
    block: &Block,
    tick: Handle<Function>,
    inst: &BTreeMap<Handle<Function>, u64>,
    entered: &BTreeMap<Handle<Function>, u64>,
    problems: &mut Vec<String>,
    unit: &mut u64,
) -> (u64, u64) {
    let (mut i, mut e) = (0u64, 0u64);
    for st in block.iter() {
        let (a, b) = match st {
            Statement::Block(b) => count_loops(f, name, b, tick, inst, entered, problems, unit),
            Statement::If { accept, reject, .. } => {
                let x = count_loops(f, name, accept, tick, inst, entered, problems, unit);
                let y = count_loops(f, name, reject, tick, inst, entered, problems, unit);
                (x.0.saturating_add(y.0), x.1.saturating_add(y.1))
            }
            Statement::Switch { cases, .. } => cases.iter().fold((0, 0), |acc: (u64, u64), c| {
                let x = count_loops(f, name, &c.body, tick, inst, entered, problems, unit);
                (acc.0.saturating_add(x.0), acc.1.saturating_add(x.1))
            }),
            Statement::Loop {
                body, continuing, ..
            } => {
                if !ticks_first(f, body, tick) {
                    problems.push(format!(
                        "fn {name}: a loop whose body does not start with `if (!tick()) {{ break; }}`"
                    ));
                }
                let x = count_loops(f, name, body, tick, inst, entered, problems, unit);
                let y = count_loops(f, name, continuing, tick, inst, entered, problems, unit);
                *unit = (*unit).max(x.1.saturating_add(y.1));
                (x.0.saturating_add(y.0).saturating_add(1), 1)
            }
            Statement::Call { function, .. } if *function != tick => {
                match (inst.get(function), entered.get(function)) {
                    (Some(&a), Some(&b)) => (a, b),
                    _ => {
                        problems.push(format!(
                            "fn {name}: calls fn{} before its loops were counted",
                            function.index()
                        ));
                        (0, 0)
                    }
                }
            }
            _ => (0, 0),
        };
        i = i.saturating_add(a);
        e = e.saturating_add(b);
    }
    (i, e)
}

#[cfg(test)]
mod tests {
    use super::{
        check_module, parse_source, require_bounded, HostMaxima, LoopReport,
        MAX_SHADER_SOURCE_BYTES,
    };

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
    fn shader_text_over_the_size_maximum_is_refused_before_parsing() {
        // text one byte over the maximum (not valid WGSL either: the size refusal comes first, so the parser never
        // sees it); at the maximum the parser runs and reports the syntax error instead
        let over = "x".repeat(MAX_SHADER_SOURCE_BYTES + 1);
        let e = parse_source(&wgpu::ShaderSource::Wgsl(over.as_str().into())).unwrap_err();
        assert!(
            e.contains("MAX_SHADER_SOURCE_BYTES") && e.contains("not parsed"),
            "{e}"
        );
        let glsl = wgpu::ShaderSource::Glsl {
            shader: over.as_str().into(),
            stage: naga::ShaderStage::Compute,
            defines: std::collections::HashMap::default(),
        };
        assert!(parse_source(&glsl).unwrap_err().contains("not parsed"));
        let at = "x".repeat(MAX_SHADER_SOURCE_BYTES);
        let e = parse_source(&wgpu::ShaderSource::Wgsl(at.as_str().into())).unwrap_err();
        assert!(!e.contains("not parsed"), "the maximum itself is parsed");
        // and require_bounded carries it as a typed refusal before any device
        let e = require_bounded(
            "big",
            &wgpu::ShaderSource::Wgsl(over.as_str().into()),
            &HostMaxima::new(),
        )
        .unwrap_err();
        assert!(
            matches!(e, crate::compiled::GpuError::ShaderCompilation(ref m) if m.contains("not parsed")),
            "{e}"
        );
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

    /// A counted kernel: `tick` and a body of loops, with a work budget of at most 100.
    fn counted(
        tick_body: &str,
        body: &str,
    ) -> (
        super::CountedReport,
        Result<LoopReport, crate::compiled::GpuError>,
    ) {
        counted_with(COUNTED_DECLS, "", tick_body, body, 100)
    }

    const COUNTED_DECLS: &str = "struct U { steps: u32, work_budget: u32 }\n@group(0) @binding(0) var<uniform> u: U;\nvar<private> work: u32;";

    /// [`counted`] with its own declarations of `u` and `work`, extra functions or entry points, and the budget's
    /// stated maximum.
    fn counted_with(
        decls: &str,
        extra: &str,
        tick_body: &str,
        body: &str,
        budget_max: u64,
    ) -> (
        super::CountedReport,
        Result<LoopReport, crate::compiled::GpuError>,
    ) {
        let src = format!(
            "{decls}\n@group(0) @binding(1) var<storage, read_write> o: array<f32>;\nvar<private> other: u32;\nfn tick() -> bool {{ {tick_body} }}\nfn inner() -> f32 {{ var a = 0.0; for (var j = 0u; j < 4u; j = j + 1u) {{ if (!tick()) {{ break; }} a = a + 1.0; }} return a; }}\n{extra}\n@compute @workgroup_size(1)\nfn main() {{\n var acc = 0.0;\n{body}\n o[0] = acc;\n}}\n"
        );
        let m = HostMaxima::from([
            ("u.steps".to_string(), 4000),
            ("u.work_budget".to_string(), budget_max),
        ]);
        let counter = super::WorkCounter {
            tick: "tick",
            budget: "u.work_budget",
        };
        let module = parse_source(&wgpu::ShaderSource::Wgsl(src.clone().into())).unwrap();
        let r = super::check_counted(&module, &m, counter);
        let ceiling = super::KernelCeiling {
            max_trip_product: 1000,
            reason: "test",
            counter,
        };
        (
            r,
            super::require_bounded_under(
                "t",
                &wgpu::ShaderSource::Wgsl(src.into()),
                &m,
                Some(ceiling),
            ),
        )
    }

    const TICK: &str = "work = work + 1u; return work <= u.work_budget;";
    const COUNTED: &str = "for (var i = 0u; i < u.steps; i = i + 1u) { if (!tick()) { break; } acc = acc + inner(); }";

    #[test]
    fn every_loop_counted_bounds_the_invocation_by_the_budget() {
        let (r, ok) = counted(TICK, COUNTED);
        assert!(r.problems.is_empty(), "{r:?}");
        // main's loop and inner's loop: 2 instances; main's loop body enters 1 loop (inner's)
        assert_eq!(
            (r.budget, r.instances, r.unit, r.bound),
            (100, 2, 1, (100 + 2 + 1) * 2),
            "{r:?}"
        );
        // the static product (4001 x 5) is over the default ceiling's reach only through steps; the counted bound
        // decides, and it is under the stated 1000
        assert!(ok.is_ok(), "{ok:?}");
    }

    #[test]
    fn a_loop_without_a_tick_is_refused() {
        let (r, ok) = counted(
            TICK,
            "for (var i = 0u; i < u.steps; i = i + 1u) { acc = acc + inner(); }",
        );
        assert_eq!(r.problems.len(), 1, "{r:?}");
        assert!(ok.is_err());
    }

    #[test]
    fn a_tick_after_work_in_the_body_is_refused() {
        let (r, _) = counted(TICK, "for (var i = 0u; i < u.steps; i = i + 1u) { acc = acc + inner(); if (!tick()) { break; } }");
        assert!(!r.problems.is_empty(), "{r:?}");
    }

    #[test]
    fn a_tick_that_does_not_count_or_compare_with_the_budget_is_refused() {
        for t in [
            "return work <= u.work_budget;",
            "work = work + 0u; return work <= u.work_budget;",
            "work = work + 1u; return work <= u.steps;",
            "other = work + 1u; return work <= u.work_budget;",
            "work = work + 1u; return true;",
        ] {
            let (r, ok) = counted(t, COUNTED);
            assert!(!r.problems.is_empty(), "{t}: {r:?}");
            assert!(ok.is_err(), "{t}");
        }
    }

    #[test]
    fn a_store_to_the_counter_outside_the_tick_is_refused() {
        let (r, _) = counted(TICK, &format!("work = 0u;\n{COUNTED}"));
        assert_eq!(r.problems.len(), 1, "{r:?}");
    }

    #[test]
    fn a_counted_bound_over_the_stated_ceiling_is_refused() {
        // three nested loops: the unit grows, the bound passes 1000
        let (r, ok) = counted(TICK, "for (var i = 0u; i < u.steps; i = i + 1u) { if (!tick()) { break; } acc = acc + inner() + inner() + inner() + inner() + inner() + inner() + inner() + inner() + inner() + inner(); }");
        assert!(r.problems.is_empty() && r.bound > 1000, "{r:?}");
        assert!(ok.is_err(), "{ok:?}");
    }

    /// A function that stores through a pointer argument, called with `&work` inside a counted loop, resets the
    /// counter as surely as a direct store: passing the counter to any call is refused.
    #[test]
    fn passing_the_counter_to_a_call_is_refused() {
        let (r, ok) = counted_with(
            COUNTED_DECLS,
            "fn reset(p: ptr<private, u32>) { *p = 0u; }",
            TICK,
            &format!("{COUNTED}\nreset(&work);"),
            100,
        );
        assert!(
            r.problems
                .iter()
                .any(|p| p.contains("passes the work counter")),
            "{r:?}"
        );
        assert!(ok.is_err());
    }

    /// A budget the shader can write is no bound: storage read_write is refused, read-only storage and a uniform
    /// are accepted.
    #[test]
    fn a_budget_the_shader_can_write_is_refused() {
        let rw = "struct U { steps: u32, work_budget: u32 }\n@group(0) @binding(0) var<storage, read_write> u: U;\nvar<private> work: u32;";
        let (r, ok) = counted_with(rw, "", TICK, COUNTED, 100);
        assert!(
            r.problems
                .iter()
                .any(|p| p.contains("not a uniform or read-only")),
            "{r:?}"
        );
        assert!(ok.is_err());
        let ro = "struct U { steps: u32, work_budget: u32 }\n@group(0) @binding(0) var<storage, read> u: U;\nvar<private> work: u32;";
        let (r, _) = counted_with(ro, "", TICK, COUNTED, 100);
        assert!(r.problems.is_empty(), "{r:?}");
    }

    /// The counter's rules on its own declaration: no initializer, and a plain u32.
    #[test]
    fn a_counter_with_an_initializer_or_another_type_is_refused() {
        let init = "struct U { steps: u32, work_budget: u32 }\n@group(0) @binding(0) var<uniform> u: U;\nvar<private> work: u32 = 0u;";
        let (r, ok) = counted_with(init, "", TICK, COUNTED, 100);
        assert!(
            r.problems.iter().any(|p| p.contains("initializer")),
            "{r:?}"
        );
        assert!(ok.is_err());
        let wide = "struct U { steps: u32, work_budget: i32 }\n@group(0) @binding(0) var<uniform> u: U;\nvar<private> work: i32;";
        let (r, ok) = counted_with(
            wide,
            "",
            "work = work + 1i; return work <= u.work_budget;",
            COUNTED,
            100,
        );
        assert!(r.problems.iter().any(|p| p.contains("not a u32")), "{r:?}");
        assert!(ok.is_err());
    }

    /// The tick's only accepted use in a loop is `if (!tick()) { break; }` first in the body (after exit checks):
    /// a tick whose result only guards work, a tick inside a branch, and a statement other than an exit check
    /// before it are each refused.
    #[test]
    fn a_tick_used_in_another_form_or_place_is_refused() {
        for body in [
            "for (var i = 0u; i < u.steps; i = i + 1u) { if (tick()) { acc = acc + inner(); } }",
            "for (var i = 0u; i < u.steps; i = i + 1u) { if (acc > 1.0) { if (!tick()) { break; } } acc = acc + inner(); }",
            "for (var i = 0u; i < u.steps; i = i + 1u) { if (acc > 1.0) { acc = 0.0; } if (!tick()) { break; } acc = acc + inner(); }",
        ] {
            let (r, ok) = counted(TICK, body);
            assert!(!r.problems.is_empty(), "{body}: {r:?}");
            assert!(ok.is_err(), "{body}");
        }
    }

    /// A loop in a loop's `continuing` block is a loop like any other: without its own tick it is refused.
    #[test]
    fn a_loop_in_continuing_without_a_tick_is_refused() {
        let body = "var i = 0u; loop { if (i >= u.steps) { break; } if (!tick()) { break; } acc = acc + 1.0; continuing { for (var j = 0u; j < 3u; j = j + 1u) { acc = acc + 0.5; } i = i + 1u; } }";
        let (r, ok) = counted(TICK, body);
        assert!(!r.problems.is_empty(), "{r:?}");
        assert!(ok.is_err());
        let ticked = "var i = 0u; loop { if (i >= u.steps) { break; } if (!tick()) { break; } acc = acc + 1.0; continuing { for (var j = 0u; j < 3u; j = j + 1u) { if (!tick()) { break; } acc = acc + 0.5; } i = i + 1u; } }";
        let (r, _) = counted(TICK, ticked);
        assert!(r.problems.is_empty(), "{r:?}");
    }

    /// The loop instances of every entry point are summed: a second entry point with its own loop adds to them.
    #[test]
    fn instances_are_summed_across_entry_points() {
        let (one, _) = counted(TICK, COUNTED);
        let (two, _) = counted_with(
            COUNTED_DECLS,
            "@compute @workgroup_size(1)\nfn second() { var b = 0.0; for (var k = 0u; k < 5u; k = k + 1u) { if (!tick()) { break; } b = b + 1.0; } o[1] = b; }",
            TICK,
            COUNTED,
            100,
        );
        assert!(
            one.problems.is_empty() && two.problems.is_empty(),
            "{two:?}"
        );
        assert_eq!(two.instances, one.instances + 1, "{one:?} {two:?}");
    }

    #[test]
    fn a_ceiling_without_a_reason_is_refused() {
        let src = wrap("for (var i = 0u; i < 3u; i = i + 1u) { acc = acc + 1.0; }");
        let c = super::KernelCeiling {
            max_trip_product: 10,
            reason: " ",
            counter: super::WorkCounter {
                tick: "tick",
                budget: "u.n",
            },
        };
        assert!(super::require_bounded_under(
            "t",
            &wgpu::ShaderSource::Wgsl(src.into()),
            &maxima(),
            Some(c)
        )
        .is_err());
    }
}
