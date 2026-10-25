//! ALICE-Law G3 on the shader text the transpilers emit: no fused multiply-add
//! intrinsic (`fma(`, `fmaf(`, `mad(`) in any of it.
//!
//! The CPU paths round `a * b + c` twice (clippy bans `f32::mul_add`, wide's
//! fused methods and cranelift's `fma`), so a fused intrinsic in the GPU text
//! would give the GPU path other bits, and a negated one is what G3 forbids.
//! `scripts/shader_fma_guard.py` reads the string literals of `src/`; this test
//! reads what the transpilers actually return, so text composed from pieces at
//! run time is seen too. Every entry of the shared node corpus goes through
//! every backend whose feature is on, in both modes:
//!
//! | backend     | feature       | text scanned                                    |
//! |-------------|---------------|-------------------------------------------------|
//! | GLSL        | `glsl`        | `GlslShader::transpile(..).source`              |
//! | WGSL        | `gpu`         | `WgslShader::transpile(..).source`              |
//! | HLSL        | `hlsl`        | `HlslShader::transpile(..).source`              |
//! | BlinkScript | `blinkscript` | `BlinkScriptShader::transpile(..).to_kernel(..)` |
//! | MSL         | `msl`         | `MslShader::transpile(..)`: `source` (naga's MSL) and `wgsl_source` |
//!
//! Fails on a hit, and when a backend whose feature is on compared no text
//! (an empty corpus or a transpiler that failed on every entry). CI runs it
//! with `msl,glsl,hlsl,blinkscript` so every backend is compared.
//!
//! A GPU driver may contract `a * b + c` into an fma by itself (Metal through
//! wgpu does, measured in tests/test_gpu_fma_contraction.rs), so the text must
//! also never negate a whole multiply-add: after a contraction that negation is
//! the outer fneg of an fma G3 forbids. The same scan flags, in every text, a
//! unary minus over a parenthesised expression with a top-level product and a
//! top-level binary `+` / `-` (`-(a * b + c)`, `0.0 - (a * b + c)`,
//! `-1.0 * (a * b + c)`), and a unary minus on a name whose assignment earlier
//! in the same text is such an expression (`p = ky - kx * kx; .. sqrt(-p)`).
//! The emitted form negates the operands instead (`kx * kx - ky`), which a
//! contraction turns into `fma(kx, kx, -ky)`, the form the Law allows.
//!
//! Author: Moroya Sakamoto

#[cfg(any(
    feature = "glsl",
    feature = "gpu",
    feature = "hlsl",
    feature = "blinkscript",
    feature = "msl"
))]
mod common;

/// The fused multiply-add intrinsics of the shader languages this crate emits,
/// as `name(` with any spaces before the parenthesis and not part of a longer
/// identifier (`nomad(` and `fmadd(` do not count; `metal::fma(` does).
const INTRINSICS: [&str; 3] = ["fmaf", "fma", "mad"];

fn fused_calls(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut found = Vec::new();
    for name in INTRINSICS {
        let mut from = 0;
        while let Some(off) = text[from..].find(name) {
            let at = from + off;
            let end = at + name.len();
            let before_ok = at == 0 || !ident(bytes[at - 1]);
            let mut k = end;
            while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                k += 1;
            }
            if before_ok
                && k < bytes.len()
                && bytes[k] == b'('
                && (end == bytes.len() || !ident(bytes[end]))
            {
                let line = text[..at].matches('\n').count() + 1;
                found.push(format!("line {line}: `{}`", &text[at..=k]));
            }
            from = end;
        }
    }
    found
}

/// The index of the `)` that closes the `(` at `open`, if any
fn group_end(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (k, c) in text[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + k);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether an expression has a top-level `*` and a top-level binary `+` / `-`
/// (an exponent such as `1e-5` and a sign after an operator do not count)
fn is_multiply_add(expr: &str) -> bool {
    let (mut depth, mut star, mut add) = (0i32, false, false);
    let mut prev: Option<char> = None;
    let mut prev2: Option<char> = None;
    for c in expr.chars() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '*' if depth == 0 => star = true,
            '+' | '-' if depth == 0 => {
                let after_operand = prev.is_some_and(|p| !"(,=+-*/?:<>!&|[".contains(p));
                let exponent = prev.is_some_and(|p| p == 'e' || p == 'E')
                    && prev2.is_some_and(|q| q.is_ascii_digit() || q == '.');
                if after_operand && !exponent {
                    add = true;
                }
            }
            _ => {}
        }
        if !c.is_whitespace() {
            prev2 = prev;
            prev = Some(c);
        }
    }
    star && add
}

/// Whether the `-` at `at` is a unary minus (after an operator, an opening bracket, a
/// separator, `return` or the start of the text)
fn is_unary_minus(text: &str, at: usize) -> bool {
    let before = text[..at].trim_end();
    before.is_empty()
        || before.ends_with("return")
        || before
            .chars()
            .next_back()
            .is_some_and(|c| "(,=+-*/?:<>!&|[{;".contains(c))
}

const fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Negations of a whole multiply-add in shader text: `-( .. * .. ± .. )`,
/// `0.0 - ( .. )`, `-1.0 * ( .. )`, and `-name` where `name` was assigned a
/// multiply-add earlier in the text
fn negated_multiply_adds(text: &str) -> Vec<String> {
    let line_of = |at: usize| text[..at].matches('\n').count() + 1;
    let mut found = Vec::new();
    let bytes = text.as_bytes();
    for (at, _) in text.match_indices('-') {
        let rest = text[at + 1..].trim_start();
        let open = text.len() - rest.len();
        if rest.starts_with('(') && is_unary_minus(text, at) {
            if let Some(end) = group_end(text, open) {
                if is_multiply_add(&text[open + 1..end]) {
                    found.push(format!("line {}: `{}`", line_of(at), &text[at..=end]));
                }
            }
        }
    }
    for lead in ["0.0 - (", "0.0f - (", "-1.0 * (", "-1.0f * ("] {
        for (at, _) in text.match_indices(lead) {
            if at > 0 && is_ident(char::from(bytes[at - 1])) {
                continue;
            }
            let open = at + lead.len() - 1;
            if let Some(end) = group_end(text, open) {
                if is_multiply_add(&text[open + 1..end]) {
                    found.push(format!("line {}: `{}`", line_of(at), &text[at..=end]));
                }
            }
        }
    }
    // names assigned a multiply-add: `name = expr;` and `name: type = expr;`
    // every assignment of a name, in order: the last one before a use decides
    // (`h = q * q + p; h = sqrt(h); .. -h` negates a square root, not a multiply-add)
    let mut assigned: Vec<(String, usize, bool)> = Vec::new();
    for (eq, _) in text.match_indices('=') {
        let next = text[eq + 1..].chars().next();
        let prev = text[..eq].chars().next_back();
        if next == Some('=') || prev.is_some_and(|c| "=<>!+-*/".contains(c)) {
            continue;
        }
        let Some(stmt_end) = text[eq..].find(';').map(|k| eq + k) else {
            continue;
        };
        let mut lhs = text[..eq].trim_end();
        if let Some(colon) = lhs.rfind(':') {
            if !lhs[colon + 1..].contains([';', '\n']) {
                lhs = lhs[..colon].trim_end();
            }
        }
        let name: String = lhs
            .chars()
            .rev()
            .take_while(|&c| is_ident(c))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if !name.is_empty() {
            let multiply_add = is_multiply_add(&text[eq + 1..stmt_end]);
            assigned.push((name, eq, multiply_add));
        }
    }
    for (at, _) in text.match_indices('-') {
        if !is_unary_minus(text, at) {
            continue;
        }
        let rest = text[at + 1..].trim_start();
        let name: String = rest.chars().take_while(|&c| is_ident(c)).collect();
        if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        let after = rest[name.len()..].trim_start().chars().next();
        // a call, or the negated name as an operand of a product (`-t * 2.0`) or a field access
        if matches!(after, Some('(' | '*' | '/' | '.')) {
            continue;
        }
        let last = assigned
            .iter()
            .rev()
            .find(|(n, eq, _)| *n == name && *eq < at);
        if let Some((_, eq, true)) = last {
            let line = text[..*eq].rfind('\n').map_or(0, |k| k + 1);
            found.push(format!(
                "line {}: `-{name}`, assigned at line {}",
                line_of(at),
                line_of(line.max(*eq))
            ));
        }
    }
    found
}

#[test]
fn the_negation_scanner_flags_a_whole_multiply_add_and_nothing_else() {
    // canaries: each negates a whole multiply-add, directly or through a name
    for text in [
        "let d = -(p.x * p.y + p.z);",
        "float d = - (a*b - c);",
        "float d = 0.0 - (a * b + c);",
        "float d = -1.0 * (a * b + c);",
        "float t = a * b + c;\nreturn -t;",
        "let p = ky - kx * kx;\nlet z = sqrt(-p);",
        "precise float p: f32 = ky - kx * kx; float z = sqrt(-p);",
    ] {
        assert_eq!(negated_multiply_adds(text).len(), 1, "{text}");
    }
    // operand negations and other minus signs
    for text in [
        "let d = -p.x * p.y + p.z;",
        "float d = c - a * b;",
        "float t = a * b + c; return -t * 2.0;",
        "x = y - (a * b + c);",
        "float d = -(a * b);",
        "let z = sqrt(kx * kx - ky);",
        "float h = q * q + 4.0 * p3; h = sqrt(h); vec2 x = vec2(h, -h);",
        "float e = -(x + 1e-5);",
        "float v = clamp(q, -1.0, 1.0);",
        "float s = -sin(a * b + c);",
    ] {
        assert!(
            negated_multiply_adds(text).is_empty(),
            "{text}: {:?}",
            negated_multiply_adds(text)
        );
    }
}

#[test]
fn the_scanner_flags_each_language_spelling() {
    // a canary per backend, written the way that language spells it
    for text in [
        "let d = -fma(a, b, c);",                   // WGSL
        "float d = -fma(a, b, c);",                 // GLSL
        "float d = -mad(a, b, c);",                 // HLSL / BlinkScript
        "float d = -metal::precise::fma(a, b, c);", // MSL
        "float d = -fmaf (a, b, c);",
    ] {
        assert_eq!(fused_calls(text).len(), 1, "{text}");
    }
    for text in [
        "nomad(x)",
        "my_fma(a)",
        "fmadd(b)",
        "madness(c)",
        "a * b + c",
        "fma",
    ] {
        assert!(fused_calls(text).is_empty(), "{text}");
    }
}

#[cfg(any(
    feature = "glsl",
    feature = "gpu",
    feature = "hlsl",
    feature = "blinkscript",
    feature = "msl"
))]
struct Tally {
    backend: &'static str,
    compared: usize,
    hits: Vec<String>,
}

#[cfg(any(
    feature = "glsl",
    feature = "gpu",
    feature = "hlsl",
    feature = "blinkscript",
    feature = "msl"
))]
impl Tally {
    const fn new(backend: &'static str) -> Self {
        Self {
            backend,
            compared: 0,
            hits: Vec::new(),
        }
    }

    fn scan(&mut self, entry: &str, mode: &str, text: &str) {
        self.compared += 1;
        for hit in fused_calls(text)
            .into_iter()
            .chain(negated_multiply_adds(text))
        {
            self.hits
                .push(format!("{} {entry} ({mode}) {hit}", self.backend));
        }
    }

    fn check(&self) {
        assert!(
            self.compared > 0,
            "{}: no emitted text was compared",
            self.backend
        );
        assert!(
            self.hits.is_empty(),
            "fused or negated multiply-add in emitted shader text:\n{}",
            self.hits.join("\n")
        );
        eprintln!(
            "{}: {} emitted texts, 0 fused multiply-add intrinsics, 0 negated multiply-adds",
            self.backend, self.compared
        );
    }
}

#[cfg(feature = "glsl")]
#[test]
fn glsl_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
    let mut t = Tally::new("GLSL");
    for (name, node) in common::corpus::corpus() {
        t.scan(
            name,
            "hardcoded",
            &GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).source,
        );
        t.scan(
            name,
            "dynamic",
            &GlslShader::transpile(&node, GlslTranspileMode::Dynamic).source,
        );
    }
    t.check();
}

#[cfg(feature = "gpu")]
#[test]
fn wgsl_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::{TranspileMode, WgslShader};
    let mut t = Tally::new("WGSL");
    for (name, node) in common::corpus::corpus() {
        t.scan(
            name,
            "hardcoded",
            &WgslShader::transpile(&node, TranspileMode::Hardcoded).source,
        );
        t.scan(
            name,
            "dynamic",
            &WgslShader::transpile(&node, TranspileMode::Dynamic).source,
        );
    }
    t.check();
}

#[cfg(feature = "hlsl")]
#[test]
fn hlsl_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::hlsl::{HlslShader, HlslTranspileMode};
    let mut t = Tally::new("HLSL");
    for (name, node) in common::corpus::corpus() {
        t.scan(
            name,
            "hardcoded",
            &HlslShader::transpile(&node, HlslTranspileMode::Hardcoded).source,
        );
        t.scan(
            name,
            "dynamic",
            &HlslShader::transpile(&node, HlslTranspileMode::Dynamic).source,
        );
    }
    t.check();
}

#[cfg(feature = "blinkscript")]
#[test]
fn blinkscript_kernel_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::blinkscript::{BlinkScriptShader, BlinkScriptTranspileMode};
    let mut t = Tally::new("BlinkScript");
    for (name, node) in common::corpus::corpus() {
        for (mode, m) in [
            ("hardcoded", BlinkScriptTranspileMode::Hardcoded),
            ("dynamic", BlinkScriptTranspileMode::Dynamic),
        ] {
            t.scan(
                name,
                mode,
                &BlinkScriptShader::transpile(&node, m).to_kernel(0.0, (-2.0, 2.0)),
            );
        }
    }
    t.check();
}

#[cfg(feature = "msl")]
#[test]
fn msl_emits_no_fused_multiply_add() {
    use alice_sdf::compiled::msl::MslShader;
    use alice_sdf::compiled::TranspileMode;
    let mut msl = Tally::new("MSL");
    let mut wgsl = Tally::new("MSL's WGSL input");
    for (name, node) in common::corpus::corpus() {
        for (mode, m) in [
            ("hardcoded", TranspileMode::Hardcoded),
            ("dynamic", TranspileMode::Dynamic),
        ] {
            // a node MSL refuses is the matrix test's business (docs/node-support.md); here only emitted text counts
            if let Ok(shader) = MslShader::transpile(&node, m) {
                msl.scan(name, mode, &shader.source);
                wgsl.scan(name, mode, &shader.wgsl_source);
            }
        }
    }
    msl.check();
    wgsl.check();
}
