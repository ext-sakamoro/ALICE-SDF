//! ALICE-Law G3 on the GLSL and HLSL the transpilers emit: every floating-point
//! result is computed into a `precise` variable, so a compiler may not contract
//! `a * b + c` into one rounding.
//!
//! GLSL and HLSL drivers may fuse a multiply and an add that the text writes
//! separately (both languages allow it unless the result is `precise`).
//! `precise` covers the computation of a variable's value, not a bare
//! `return a * b + c`, so the emitted text must:
//!
//! - declare every `float` / vector local with the qualifier: HLSL `precise`,
//!   GLSL `ALICE_PRECISE`, a macro the source defines as `precise` where the
//!   version has it (GLSL 4.00+, GLSL ES 3.20+) and as nothing elsewhere
//!   (GLSL ES 3.00 and GLSL 3.30 have no `precise`; that is a residual on the
//!   caller's choice of version);
//! - return, from a function that returns a float or a vector, only a name, a
//!   negated name or a literal, so every arithmetic result passed out was a
//!   `precise` local first.
//!
//! Every entry of the shared node corpus goes through both modes. Fails on a
//! miss, and when nothing was compared. What a compiler then does with
//! `precise` is checked on its output by CI (SPIR-V `NoContraction`, DXIL
//! without fused multiply-add); a real Vulkan or DX12 driver is not measured.
//!
//! Author: Moroya Sakamoto

#[cfg(any(feature = "glsl", feature = "hlsl"))]
mod common;

/// The floating-point types of both languages whose locals must be precise.
const FLOAT_TYPES: [&str; 7] = [
    "float", "vec2", "vec3", "vec4", "float2", "float3", "float4",
];

#[derive(Default)]
struct Scan {
    declarations: usize,
    returns: usize,
    misses: Vec<String>,
}

const fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

const fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `name`, `-name`, `name.xyz` or a number: nothing left to compute.
fn is_plain_value(expr: &str) -> bool {
    let e = expr.trim();
    let e = e.strip_prefix('-').unwrap_or(e).trim();
    if e.is_empty() {
        return false;
    }
    let first = e.chars().next().unwrap_or(' ');
    if is_ident_start(first) {
        return e.chars().all(|c| is_ident(c) || c == '.');
    }
    e.chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-' | 'f' | 'u'))
}

/// Split `type name ...` off a statement: `Some((type, rest after the name))`
/// when it declares a local named variable of a float or vector type.
fn float_declaration(stmt: &str) -> Option<(&str, &str)> {
    let mut words = stmt.splitn(2, char::is_whitespace);
    let ty = words.next()?;
    if !FLOAT_TYPES.contains(&ty) {
        return None;
    }
    let rest = words.next()?.trim_start();
    let name_len = rest.find(|c: char| !is_ident(c)).unwrap_or(rest.len());
    if name_len == 0 || !rest.starts_with(is_ident_start) {
        return None;
    }
    let after = rest[name_len..].trim_start();
    // `name = ...` or `name` (a later assignment): a local; `name(` a function,
    // `name,` a struct member list (see `local_declaration` for locals), `name[` an
    // array in a uniform block.
    (after.is_empty() || (after.starts_with('=') && !after.starts_with("==")))
        .then_some((ty, after))
}

/// A local declaration: `float_declaration`, or outside a struct also `TYPE a, b, ...`
/// (several names in one statement, which `float_declaration` leaves to struct members).
fn local_declaration(stmt: &str, in_struct: bool) -> bool {
    if float_declaration(stmt).is_some() {
        return true;
    }
    if in_struct {
        return false;
    }
    let mut words = stmt.splitn(2, char::is_whitespace);
    let (Some(ty), Some(rest)) = (words.next(), words.next()) else {
        return false;
    };
    let rest = rest.trim_start();
    let name_len = rest.find(|c: char| !is_ident(c)).unwrap_or(rest.len());
    FLOAT_TYPES.contains(&ty)
        && name_len > 0
        && rest.starts_with(is_ident_start)
        && rest[name_len..].trim_start().starts_with(',')
}

/// The condition clause of a `for (init; condition; step)` header on `line`, if any.
fn for_condition(line: &str) -> Option<&str> {
    let i = line.find("for (").or_else(|| line.find("for("))?;
    let header = &line[i..];
    let open = header.find('(')?;
    let mut parts = header[open + 1..].splitn(3, ';');
    parts.next()?;
    let cond = parts.next()?;
    parts.next()?;
    Some(cond.trim())
}

/// B1: the condition of an `if` / `while` statement, when it contains a multiply that
/// an add or a subtract reads directly: `a * b + c`, `c + a * b`, `a + (b * c)` (bare
/// parentheses do not stop a contraction into `fma`), `(a * b).x + c` (a swizzle or an
/// index of a product is a product of components). Not `(a - b) * c` or
/// `k * (px + r)` (the add is inside the multiply's operand), nor `a + f(b * c)` (a call
/// boundary); a call's arguments are read as expressions of their own. The operands
/// must be precise locals first.
fn condition_multiply_adds(stmt: &str) -> bool {
    let t = stmt.trim_start_matches('}').trim_start();
    let t = t.strip_prefix("else").map_or(t, str::trim_start);
    let rest = if let Some(r) = t.strip_prefix("if") {
        r
    } else if let Some(r) = t.strip_prefix("while") {
        r
    } else {
        return false;
    };
    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return false;
    }
    let mut depth = 0usize;
    let mut end = rest.len();
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    let tokens = tokenize(&rest[1..end]);
    let mut p = Parser {
        tokens: &tokens,
        at: 0,
        fused: false,
    };
    // sides of comparisons, logical operators and `?:` are expressions of their own
    while p.at < tokens.len() {
        p.expr();
        if p.at < tokens.len() {
            p.at += 1; // a separator, or a token the grammar does not read
        }
    }
    p.fused
}

#[derive(Clone, Copy, PartialEq)]
enum Tok<'a> {
    Atom(&'a str),
    Op(char),
    Open,
    Close,
    OpenIndex,
    CloseIndex,
    Comma,
    Sep,
}

fn tokenize(s: &str) -> Vec<Tok<'_>> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
            let start = i;
            while i < b.len() {
                let d = b[i] as char;
                let exponent_sign = (d == '-' || d == '+')
                    && matches!(b[i - 1] as char, 'e' | 'E')
                    && s[start..i - 1]
                        .chars()
                        .all(|x| x.is_ascii_digit() || x == '.');
                if d.is_ascii_alphanumeric() || d == '_' || d == '.' || exponent_sign {
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(Tok::Atom(&s[start..i]));
        } else {
            out.push(match c {
                '(' => Tok::Open,
                ')' => Tok::Close,
                '[' => Tok::OpenIndex,
                ']' => Tok::CloseIndex,
                ',' => Tok::Comma,
                '+' | '-' | '*' | '/' | '%' => Tok::Op(c),
                _ => Tok::Sep, // < > = ! & | ? :
            });
            i += 1;
        }
    }
    out
}

/// What an expression is at its top, seen through bare parentheses.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Mul,
    Other,
}

struct Parser<'a, 'b> {
    tokens: &'b [Tok<'a>],
    at: usize,
    fused: bool,
}

impl Parser<'_, '_> {
    fn peek(&self) -> Option<Tok<'_>> {
        self.tokens.get(self.at).copied()
    }

    /// `.member` and `[index]` after a primary; an index is an expression of its own
    fn suffixes(&mut self) {
        loop {
            match self.peek() {
                Some(Tok::Atom(a)) if a.starts_with('.') => self.at += 1,
                Some(Tok::OpenIndex) => {
                    self.at += 1;
                    while !matches!(self.peek(), None | Some(Tok::CloseIndex)) {
                        let before = self.at;
                        self.expr();
                        if self.at == before {
                            self.at += 1;
                        }
                    }
                    if self.peek() == Some(Tok::CloseIndex) {
                        self.at += 1;
                    }
                }
                _ => return,
            }
        }
    }

    /// expr := term (('+' | '-') term)*
    fn expr(&mut self) -> Kind {
        let mut kind = self.term();
        let mut operands = vec![kind];
        while let Some(Tok::Op('+' | '-')) = self.peek() {
            self.at += 1;
            operands.push(self.term());
            kind = Kind::Other;
        }
        if operands.len() > 1 && operands.contains(&Kind::Mul) {
            self.fused = true;
        }
        kind
    }

    /// term := unary (('*' | '/' | '%') unary)*; a multiply as the last operator is `Mul`
    fn term(&mut self) -> Kind {
        let mut kind = self.unary();
        while let Some(Tok::Op(c @ ('*' | '/' | '%'))) = self.peek() {
            self.at += 1;
            self.unary();
            kind = if c == '*' { Kind::Mul } else { Kind::Other };
        }
        kind
    }

    /// unary := '-' unary | primary
    fn unary(&mut self) -> Kind {
        if let Some(Tok::Op('-' | '+')) = self.peek() {
            self.at += 1;
            return self.unary();
        }
        self.primary()
    }

    /// primary := atom [call] | '(' expr ')'; a call's arguments are expressions of their
    /// own and the call is `Other`
    fn primary(&mut self) -> Kind {
        match self.peek() {
            Some(Tok::Open) => {
                self.at += 1;
                let kind = self.expr();
                if self.peek() == Some(Tok::Close) {
                    self.at += 1;
                }
                // `.x` / `.xy` / `[i]` of a product is still a product (per component)
                self.suffixes();
                kind
            }
            Some(Tok::Atom(_)) => {
                self.at += 1;
                if self.peek() == Some(Tok::Open) {
                    self.at += 1;
                    loop {
                        match self.peek() {
                            None => break,
                            Some(Tok::Close) => {
                                self.at += 1;
                                break;
                            }
                            Some(Tok::Comma) => self.at += 1,
                            Some(_) => {
                                let before = self.at;
                                self.expr();
                                if self.at == before {
                                    self.at += 1;
                                }
                            }
                        }
                    }
                }
                self.suffixes();
                Kind::Other
            }
            _ => Kind::Other,
        }
    }
}

/// An `if (...)`, `else if (...)`, `else`, `for (...)` or `while (...)` line that
/// opens no brace and ends no statement: only the next statement is guarded.
fn is_unbraced_guard(line: &str) -> bool {
    let t = line.trim().trim_start_matches('}').trim_start();
    let header = t.starts_with("if ")
        || t.starts_with("if(")
        || t.starts_with("else")
        || t.starts_with("for ")
        || t.starts_with("for(")
        || t.starts_with("while ")
        || t.starts_with("while(");
    header && (t.ends_with(')') || t == "else")
}

/// The names of the `out` / `inout` parameters of a function signature line.
fn out_parameters(signature: &str) -> Vec<String> {
    let Some(params) = signature.split_once('(').map(|(_, p)| p) else {
        return Vec::new();
    };
    let params = params.split(')').next().unwrap_or("");
    params
        .split(',')
        .filter_map(|p| {
            let words: Vec<&str> = p.split_whitespace().collect();
            (words.len() >= 3 && matches!(words[0], "out" | "inout"))
                .then(|| words[words.len() - 1].to_string())
        })
        .collect()
}

/// The statement after `if (...)` / `else if (...)` / `else` on the same line, if any.
fn one_line_guard_body(stmt: &str) -> Option<&str> {
    let rest = if let Some(r) = stmt.strip_prefix("else") {
        let r = r.trim_start();
        if r.starts_with("if") {
            r
        } else {
            return (!r.is_empty() && r.starts_with(is_ident_start)).then_some(r);
        }
    } else {
        stmt
    };
    let rest = rest.strip_prefix("if")?.trim_start();
    if !rest.starts_with('(') {
        return None;
    }
    let mut depth = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let body = rest[i + 1..].trim_start();
                    return (!body.is_empty()).then_some(body);
                }
            }
            _ => {}
        }
    }
    None
}

fn scan(text: &str, qualifier: &str, label: &str, out: &mut Scan) {
    let mut returns_float = false;
    let mut previous = "";
    // `out` parameters of the function being read: a computed value assigned to one is
    // not a precise local either
    let mut outs: Vec<String> = Vec::new();
    let mut in_struct = false;
    for (n, line) in text.lines().enumerate() {
        let at = |what: &str| format!("{label} line {}: {what}: `{}`", n + 1, line.trim());
        // A declaration as the only statement of an unbraced guard: whatever
        // follows it (the `return` that reads it) runs unconditionally. A
        // rewrite of `if (c)` + `return a * b;` into a precise local made this.
        let stmt = line.trim();
        let declares = stmt
            .strip_prefix(qualifier)
            .filter(|s| s.starts_with(' '))
            .map_or(stmt, str::trim_start);
        if is_unbraced_guard(previous) && float_declaration(declares).is_some() {
            out.misses.push(at(
                "a declaration as the whole body of an unbraced if / else / loop",
            ));
        }
        if !stmt.is_empty() {
            previous = line;
        }
        // a function at column 0: does it return a float or a vector?
        if line.starts_with(is_ident_start) && line.contains('(') && !line.trim_end().ends_with(';')
        {
            let head = line.split('(').next().unwrap_or("");
            let mut words = head.split_whitespace();
            let ty = words.next().unwrap_or("");
            if words.next().is_some() {
                returns_float = FLOAT_TYPES.contains(&ty);
                outs = out_parameters(line);
            }
        }
        if line.starts_with('}') {
            returns_float = false;
            outs.clear();
            in_struct = false;
        }
        if line.trim_start().starts_with("struct ") {
            in_struct = true;
        }
        // the condition of a `for` is read like an `if`: the split below cuts it out
        // of its header
        if let Some(cond) = for_condition(line) {
            if condition_multiply_adds(&format!("if ({cond}) {{")) {
                out.misses
                    .push(at("a multiply and an add in a for condition"));
            }
        }
        for segment in line.split([';', '{', '}']) {
            let mut stmt = segment.trim();
            for prefix in ["for (", "for("] {
                if let Some(s) = stmt.strip_prefix(prefix) {
                    stmt = s.trim_start();
                }
            }
            // `if (c) float x = a * b` / `else float x = ...` on one line: the declaration is
            // the whole body of the guard, whatever reads `x` after it runs unconditionally
            if let Some(body) = one_line_guard_body(stmt) {
                let decl = body
                    .strip_prefix(qualifier)
                    .filter(|s| s.starts_with(' '))
                    .map_or(body, str::trim_start);
                if float_declaration(decl).is_some() {
                    out.misses.push(at(
                        "a declaration as the whole body of a one-line if / else",
                    ));
                }
            }
            if stmt.starts_with("const ") || stmt.starts_with("static const ") {
                continue;
            }
            if condition_multiply_adds(stmt) {
                out.misses
                    .push(at("a multiply and an add in an if / while condition"));
            }
            if let Some(s) = stmt.strip_prefix(qualifier).filter(|s| s.starts_with(' ')) {
                if local_declaration(s.trim_start(), in_struct) {
                    out.declarations += 1;
                }
            } else if local_declaration(stmt, in_struct) && returns_float_scope(line) {
                out.declarations += 1;
                out.misses.push(at(&format!(
                    "a float or vector local without `{qualifier}`"
                )));
            }
            if let Some((lhs, rhs)) = stmt.split_once('=') {
                let lhs = lhs.trim();
                if outs.iter().any(|o| o == lhs) && !rhs.starts_with('=') && !is_plain_value(rhs) {
                    out.misses
                        .push(at("a computed value assigned to an out parameter"));
                }
            }
            if returns_float {
                if let Some(i) = stmt.find("return ") {
                    if i == 0 || !is_ident(stmt[..i].chars().last().unwrap_or(' ')) {
                        out.returns += 1;
                        let expr = &stmt[i + "return ".len()..];
                        if !is_plain_value(expr) {
                            out.misses.push(at("a computed value returned directly"));
                        }
                    }
                }
            }
        }
    }
}

/// A local is a statement indented inside a function body; column-0 lines are
/// globals and signatures.
fn returns_float_scope(line: &str) -> bool {
    line.starts_with(char::is_whitespace)
}

#[test]
fn the_scanner_sees_a_missing_qualifier_and_a_computed_return() {
    let bad = "float f(float a, float b, float c) {\n    float t = a * b;\n    return t + c;\n}\n";
    let mut s = Scan::default();
    scan(bad, "precise", "bad", &mut s);
    assert_eq!(s.misses.len(), 2, "{:?}", s.misses);
    let good = "float f(float a, float b, float c) {\n    precise float t = a * b + c;\n    return t;\n}\n";
    let mut s = Scan::default();
    scan(good, "precise", "good", &mut s);
    assert!(s.misses.is_empty(), "{:?}", s.misses);
    assert_eq!((s.declarations, s.returns), (1, 1));
    // integer helpers and struct members are not floating-point locals
    let other = "uint h(uint x) {\n    uint y = x * 3u + 1u;\n    return y ^ x;\n}\nstruct P {\n    float x, y, z, _pad;\n};\n";
    let mut s = Scan::default();
    scan(other, "precise", "other", &mut s);
    assert!(s.misses.is_empty() && s.declarations == 0, "{:?}", s.misses);
    // an unbraced `if` around a precise local: the return after it is unconditional
    let guard = "float f(float a, float b) {\n    if (a > b)\n        precise float r = a * b;\n        return r;\n    precise float s = a - b;\n    return s;\n}\n";
    let mut s = Scan::default();
    scan(guard, "precise", "guard", &mut s);
    assert_eq!(s.misses.len(), 1, "{:?}", s.misses);
    assert!(s.misses[0].contains("unbraced"), "{:?}", s.misses);
    // the same with braces is fine, and so is an unbraced guard of a plain return
    let braced = "float f(float a, float b) {\n    if (a > b)\n        { precise float r = a * b; return r; }\n    if (a < b)\n        return a;\n    return b;\n}\n";
    let mut s = Scan::default();
    scan(braced, "precise", "braced", &mut s);
    assert!(s.misses.is_empty(), "{:?}", s.misses);
    // the same on one line: `if (c) float x = a * b;` and `else precise float y = ...;`
    let one_line = "float f(float a, float b) {\n    if (a > b) precise float r = a * b;\n    else precise float t = a;\n    precise float s = a - b;\n    return s;\n}\n";
    let mut s = Scan::default();
    scan(one_line, "precise", "one_line", &mut s);
    assert_eq!(s.misses.len(), 2, "{:?}", s.misses);
    assert!(
        s.misses.iter().all(|m| m.contains("one-line")),
        "{:?}",
        s.misses
    );
    assert_eq!(one_line_guard_body("if (a < b) return a"), Some("return a"));
    assert_eq!(one_line_guard_body("ifx = 1"), None);
    // a computed value assigned to an out parameter, and the same through a precise local
    let outp = "void g(float3 p, out float d, out float3 n) {\n    d = p.x * p.y + p.z;\n    precise float3 m = p * 2.0;\n    n = m;\n}\n";
    let mut s = Scan::default();
    scan(outp, "precise", "outp", &mut s);
    assert_eq!(s.misses.len(), 1, "{:?}", s.misses);
    assert!(s.misses[0].contains("out parameter"), "{:?}", s.misses);
    // B2: several names in one declaration are locals outside a struct, members inside
    let multi = "float f(float a) {\n    float m, n;\n    precise float x, y;\n    x = a;\n    return x;\n}\nstruct S {\n    float u, v;\n};\n";
    let mut s = Scan::default();
    scan(multi, "precise", "multi", &mut s);
    assert_eq!(s.misses.len(), 1, "{:?}", s.misses);
    assert!(s.misses[0].contains("float m, n"), "{:?}", s.misses);
    // B1: a multiply and an add in a condition; a comparison of products, an exponent and a
    // multiply alone are not
    for (cond, flagged) in [
        ("if (q.x * b - q.y * a > d * max(b - q.y, 0.0)) {", true),
        ("while (t * 2.0 + 1.0 < 5.0) {", true),
        ("} else if (a * b < c) {", false),
        ("if (x * 1e-6 > y) {", false),
        ("if (a - b > c * d) {", false),
        ("if (c + a * b > d) {", true),
        ("while (t + d * k < m) {", true),
        ("if ((a - b) * c > d) {", false),
        ("} else if (k * (px + r) < py) {", false),
        ("if (a * b > c && d - e < f) {", false),
        ("if (r + q.x * h < alice_rhs) {", true),
        ("if ((q.y - b) * d > q.x * b) {", false),
        ("if (a + (b * c) > d) {", true),
        ("if (a + f(b * c) > d) {", false),
        ("if (f(a * b + c) > d) {", true),
        ("if ((a * b) + c > d) {", true),
        ("if ((a * b).x + c > d) {", true),
        ("if ((a * b)[i] + c > d) {", true),
        ("if (v[i * 2] > d) {", false),
    ] {
        assert_eq!(condition_multiply_adds(cond), flagged, "{cond}");
    }
    // the condition clause of a for loop
    let looped = "float f(float d) {\n    for (int alice_i = 0; alice_i * 2.0 + 1.0 < d; alice_i++) {}\n    for (int j = 0; j < 4; j++) {}\n    return d;\n}\n";
    let mut s = Scan::default();
    scan(looped, "precise", "looped", &mut s);
    assert_eq!(s.misses.len(), 1, "{:?}", s.misses);
    assert!(s.misses[0].contains("for condition"), "{:?}", s.misses);
}

#[cfg(any(feature = "glsl", feature = "hlsl"))]
/// `texts`: (emitter, label, text). Every emitter must have compared something.
fn check(backend: &str, texts: &[(&str, String, String)], qualifier: &str) {
    let mut total = Scan::default();
    let mut per_emitter: Vec<(&str, usize, usize)> = Vec::new();
    for (emitter, label, text) in texts {
        let (d, r) = (total.declarations, total.returns);
        scan(text, qualifier, label, &mut total);
        let (dd, rr) = (total.declarations - d, total.returns - r);
        match per_emitter.iter_mut().find(|e| e.0 == *emitter) {
            Some(e) => {
                e.1 += dd;
                e.2 += rr;
            }
            None => per_emitter.push((emitter, dd, rr)),
        }
    }
    for (emitter, d, r) in &per_emitter {
        assert!(
            d + r > 0,
            "{backend} {emitter}: compared 0 declarations and 0 returns"
        );
    }
    assert!(
        total.declarations > 0 && total.returns > 0,
        "{backend}: compared {} declarations and {} returns",
        total.declarations,
        total.returns
    );
    assert!(
        total.misses.is_empty(),
        "{backend}: {} misses, first 40:\n{}",
        total.misses.len(),
        total
            .misses
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    eprintln!(
        "{backend}: {} texts, {} precise declarations, {} plain returns; per emitter (declarations, returns): {:?}",
        texts.len(),
        total.declarations,
        total.returns,
        per_emitter
    );
}

#[cfg(feature = "glsl")]
#[test]
fn glsl_computes_every_float_into_a_precise_local() {
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
    let mut texts = Vec::new();
    for (name, node) in common::corpus::corpus() {
        for (mode, m) in [
            ("hardcoded", GlslTranspileMode::Hardcoded),
            ("dynamic", GlslTranspileMode::Dynamic),
        ] {
            let sh = GlslShader::transpile(&node, m);
            assert!(
                sh.source.contains("#define ALICE_PRECISE precise"),
                "GLSL {name} ({mode}): the source does not define ALICE_PRECISE"
            );
            // Every public output. Not `to_fragment_shader_full`: its render pipeline is
            // GLSL ES 3.00, where `ALICE_PRECISE` is empty (a residual of that version).
            #[allow(deprecated)]
            let (unity, graph) = (
                sh.to_unity_custom_function(),
                sh.export_unity_shader_graph(),
            );
            for (emitter, text) in [
                ("source", sh.source.clone()),
                ("to_compute_shader", sh.to_compute_shader()),
                ("to_vulkan_compute_shader", sh.to_vulkan_compute_shader()),
                ("to_fragment_shader", sh.to_fragment_shader()),
                ("to_unity_custom_function", unity),
                ("export_unity_shader_graph", graph),
            ] {
                texts.push((emitter, format!("GLSL {name} ({mode}) {emitter}"), text));
            }
        }
    }
    assert_scanned_list("GLSL", &texts, &GLSL_SCANNED);
    check("GLSL", &texts, "ALICE_PRECISE");
}

#[cfg(feature = "hlsl")]
#[test]
fn hlsl_computes_every_float_into_a_precise_local() {
    use alice_sdf::compiled::hlsl::{HlslShader, HlslTranspileMode};
    let mut texts = Vec::new();
    for (name, node) in common::corpus::corpus() {
        for (mode, m) in [
            ("hardcoded", HlslTranspileMode::Hardcoded),
            ("dynamic", HlslTranspileMode::Dynamic),
        ] {
            let sh = HlslShader::transpile(&node, m);
            // every public output
            for (emitter, text) in [
                ("source", sh.source.clone()),
                ("to_compute_shader", sh.to_compute_shader()),
                ("to_unity_custom_function", sh.to_unity_custom_function()),
                ("export_unity_shader_graph", sh.export_unity_shader_graph()),
                ("to_ue5_custom_node", sh.to_ue5_custom_node()),
                (
                    "export_ue5_material_function",
                    sh.export_ue5_material_function(),
                ),
            ] {
                texts.push((emitter, format!("HLSL {name} ({mode}) {emitter}"), text));
            }
        }
    }
    assert_scanned_list("HLSL", &texts, &HLSL_SCANNED);
    check("HLSL", &texts, "precise");
}

/// The public methods of `GlslShader` / `HlslShader` that return shader text and are
/// scanned above, and the ones left out with the reason. A new method that is in
/// neither list fails `every_public_emitter_is_scanned`, so the scan list cannot fall
/// behind the API.
const GLSL_SCANNED: [&str; 7] = [
    "get_eval_function", // `&self.source`, scanned as `source`
    "to_compute_shader",
    "to_vulkan_compute_shader",
    "to_fragment_shader",
    "to_unity_custom_function",
    "export_unity_shader_graph",
    "transpile",
];
const HLSL_SCANNED: [&str; 7] = [
    "get_eval_function", // `&self.source`, scanned as `source`
    "to_compute_shader",
    "to_unity_custom_function",
    "export_unity_shader_graph",
    "to_ue5_custom_node",
    "export_ue5_material_function",
    "transpile",
];

/// Every module under `src/` with a public function that emits or takes shader text,
/// and for each function whether the scan above reads its output or why it is left out.
/// A function found in a file that is not here, or not listed for its file, fails
/// `every_public_emitter_is_scanned`; so does a listed name that no longer exists.
/// (file, functions whose output is scanned, functions left out with the reason)
type Emitter = (
    &'static str,
    &'static [&'static str],
    &'static [(&'static str, &'static str)],
);
const EMITTERS: [Emitter; 13] = [
    (
        "src/compiled/glsl/transpiler.rs",
        &GLSL_SCANNED,
        &[(
            "to_fragment_shader_full",
            "render_pipeline: residual (its template locals are not qualified)",
        )],
    ),
    ("src/compiled/hlsl/transpiler.rs", &HLSL_SCANNED, &[]),
    ("src/texture/shader.rs", &["generate_shader"], &[]),
    ("src/io/nanite.rs", &["export_nanite_hlsl_material"], &[]),
    (
        "src/npr/shader_glue.rs",
        &[],
        &[("full_helpers_for", "npr: residual, no determinism claim")],
    ),
    (
        "src/npr/scene_composer.rs",
        &[],
        &[
            ("build", "npr: residual, no determinism claim"),
            ("try_build", "npr: residual, no determinism claim"),
        ],
    ),
    (
        "src/compiled/glsl/render_pipeline.rs",
        &[],
        &[(
            "build_full_shader",
            "render_pipeline: residual (its template locals are not qualified)",
        )],
    ),
    (
        "src/compiled/transpiler_common.rs",
        &[],
        &[
            (
                "generate_shader",
                "assembles the transpilers' `source`, which is scanned",
            ),
            (
                "lit",
                "a literal of the transpilers' `source`, which is scanned",
            ),
            (
                "next_var",
                "a local name of the transpilers' `source`, which is scanned",
            ),
            (
                "param",
                "a parameter of the transpilers' `source`, which is scanned",
            ),
            (
                "transpile_node",
                "a statement of the transpilers' `source`, which is scanned",
            ),
            (
                "transpile_node_inner",
                "a statement of the transpilers' `source`, which is scanned",
            ),
            (
                "shader_unsupported_nodes",
                "a list of node names, no shader text",
            ),
        ],
    ),
    (
        "src/compiled/wgsl/transpiler.rs",
        &[],
        &[
            ("transpile", "WGSL: G3 on WGSL is the MSL / Metal path"),
            (
                "transpile_material",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
            (
                "get_eval_function",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
            (
                "to_compute_shader",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
            (
                "to_compute_shader_with_normals",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
            (
                "to_volume_shader",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
        ],
    ),
    (
        "src/compiled/wgsl/gpu_eval.rs",
        &[],
        &[
            ("device_info", "a device description, no shader text"),
            ("from_glsl_compute", "takes shader text, emits none"),
            ("from_shader", "takes shader text, emits none"),
        ],
    ),
    (
        "src/mesh/gpu_mc_shaders.rs",
        &[],
        &[
            (
                "generate_classify_shader",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
            (
                "generate_prefix_sum_propagate_shader",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
            (
                "generate_prefix_sum_scan_shader",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
            (
                "generate_sdf_grid_shader",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
            (
                "generate_vertex_shader",
                "WGSL: G3 on WGSL is the MSL / Metal path",
            ),
        ],
    ),
    (
        "src/mesh/gpu_marching_cubes.rs",
        &[],
        &[(
            "gpu_marching_cubes_from_shader",
            "takes shader text, emits none",
        )],
    ),
    (
        "src/volume/gpu_bake.rs",
        &[],
        &[(
            "gpu_bake_volume_from_shader",
            "takes shader text, emits none",
        )],
    ),
];

/// A name that says shader text: a word `hlsl`, `glsl`, `shader`, `usf` or `ush`.
fn shader_name(name: &str) -> bool {
    name.split('_')
        .any(|w| matches!(w, "hlsl" | "glsl" | "shader" | "usf" | "ush"))
}

/// `pub fn` of `text`: (name, the signature up to its body).
fn public_functions(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("pub fn ") {
        let after = &rest[i + "pub fn ".len()..];
        let name: String = after.chars().take_while(|c| is_ident(*c)).collect();
        let sig_end = after.find('{').unwrap_or(after.len());
        out.push((name, after[..sig_end].to_string()));
        rest = &after[sig_end.min(after.len())..];
    }
    out
}

fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn every_public_emitter_is_scanned() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    let mut found_total = 0;
    let mut problems = Vec::new();
    for path in &files {
        let rel = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(path).unwrap();
        let entry = EMITTERS.iter().find(|e| e.0 == rel);
        let found: Vec<String> = public_functions(&text)
            .into_iter()
            .filter(|(name, sig)| {
                let returns_text = sig.contains("-> String")
                    || sig.contains("-> &str")
                    || sig.contains("-> Option<String>");
                shader_name(name) || (entry.is_some() && (returns_text || name == "transpile"))
            })
            .map(|(name, _)| name)
            .collect();
        let Some((_, scanned, left_out)) = entry else {
            if let Some(name) = found.first() {
                problems.push(format!(
                    "{rel}: `{name}` emits or takes shader text and the file is not in EMITTERS"
                ));
            }
            continue;
        };
        found_total += found.len();
        for name in &found {
            if !scanned.contains(&name.as_str()) && !left_out.iter().any(|(n, _)| n == name) {
                problems.push(format!(
                    "{rel}: `{name}` returns shader text and is neither scanned nor left out with a reason"
                ));
            }
        }
        for name in scanned.iter().chain(left_out.iter().map(|(n, _)| n)) {
            if !found.iter().any(|f| f == name) {
                problems.push(format!("{rel}: `{name}` is listed but no longer exists"));
            }
        }
    }
    for (rel, _, _) in &EMITTERS {
        if !files.iter().any(|p| {
            p.strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
                == *rel
        }) {
            problems.push(format!("{rel}: listed in EMITTERS but not under src/"));
        }
    }
    assert!(
        found_total > 0,
        "0 shader-text functions enumerated under src/"
    );
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The emitters a corpus test scanned must be exactly its list (`transpile` and
/// `get_eval_function` are both the `source` text).
#[cfg(any(feature = "glsl", feature = "hlsl"))]
fn assert_scanned_list(backend: &str, texts: &[(&str, String, String)], list: &[&str]) {
    let mut scanned: Vec<&str> = texts.iter().map(|t| t.0).collect();
    scanned.sort_unstable();
    scanned.dedup();
    let mut expected: Vec<&str> = list
        .iter()
        .map(|n| match *n {
            "transpile" | "get_eval_function" => "source",
            other => other,
        })
        .collect();
    expected.sort_unstable();
    expected.dedup();
    assert_eq!(
        scanned, expected,
        "{backend}: the scanned emitters differ from the list"
    );
}

/// A fit result with a plain and a rotated octave, so the generator writes both branches.
#[cfg(feature = "texture-fit")]
fn fitted() -> alice_sdf::texture::TextureFitResult {
    use alice_sdf::texture::{FittedOctave, TextureFitResult};
    let octave = |rotation: f32| FittedOctave {
        amplitude: 0.5,
        frequency: 4.0,
        phase: [0.1, 0.2],
        seed: 7,
        rotation,
    };
    TextureFitResult {
        width: 8,
        height: 8,
        channels: 1,
        bias: vec![0.25],
        octaves: vec![vec![octave(0.0), octave(0.7)]],
        psnr_db: 30.0,
        nmse: 0.01,
    }
}

/// The procedural texture shaders (`texture::generate_shader`) in HLSL and GLSL.
#[cfg(feature = "texture-fit")]
#[test]
fn texture_shaders_compute_every_float_into_a_precise_local() {
    use alice_sdf::texture::{generate_shader, ShaderLanguage};
    let r = fitted();
    let hlsl = generate_shader(&r, ShaderLanguage::Hlsl, "fixture");
    let glsl = generate_shader(&r, ShaderLanguage::Glsl, "fixture");
    assert!(
        hlsl.contains("ruv") && glsl.contains("ruv"),
        "the rotated octave was not written"
    );
    check(
        "texture HLSL",
        &[("generate_shader", "texture HLSL".into(), hlsl)],
        "precise",
    );
    check(
        "texture GLSL",
        &[("generate_shader", "texture GLSL".into(), glsl)],
        "ALICE_PRECISE",
    );
}

/// The Nanite material function (`io::export_nanite_hlsl_material`) of every corpus node.
#[cfg(feature = "hlsl")]
#[test]
fn nanite_material_computes_every_float_into_a_precise_local() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("precise_nanite");
    std::fs::create_dir_all(&dir).unwrap();
    let texts: Vec<(&str, String, String)> = common::corpus::corpus()
        .into_iter()
        .map(|(name, node)| {
            let path = dir.join(format!("{name}.usf"));
            alice_sdf::io::nanite::export_nanite_hlsl_material(&node, &path).unwrap();
            (
                "export_nanite_hlsl_material",
                format!("Nanite {name}"),
                std::fs::read_to_string(&path).unwrap(),
            )
        })
        .collect();
    check("Nanite HLSL", &texts, "precise");
}
