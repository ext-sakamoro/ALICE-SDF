//! No test sends more than `MAX_POINTS_PER_SUBMIT` points to the GPU in one direct submit.
//!
//! A source check, so it needs no GPU and every test run has it. In a test file that names
//! `GpuEvaluator`, and in a `#[cfg(test)]` module of `src/` in a file that does, each `.eval_batch*(`
//! call is a direct submit. Its point count must be readable from the source: a literal array, or a
//! `let` of the argument's name built by `points(N)`, `(0..N)` or `.take(N)`. A count that cannot be
//! read is refused like one over the limit: such a test calls `gpu_batches::eval_in_batches`.
//! Reading no site at all fails.
//!
//! Points are a proxy for work. A test whose size is what it checks is listed in [`OVER_LIMIT`] with
//! its exact point count and the node count of its field (which the test itself asserts), and the
//! product must stay within [`MAX_LISTED_WORK`]: a large field at that size is still refused.
//!
//! Author: Moroya Sakamoto

#[path = "common/gpu_batches.rs"]
mod gpu_batches;

use glam::Vec3;
use gpu_batches::{in_batches, MAX_POINTS_PER_SUBMIT};
use std::path::Path;

/// A submit past the limit because its size is what it checks
struct Listed {
    file: &'static str,
    method: &'static str,
    points: usize,
    /// `SdfNode::node_count` of the field, asserted by the test
    nodes: usize,
    reason: &'static str,
}

/// The most points times field nodes a listed submit may have
const MAX_LISTED_WORK: usize = 1_000_000;

const OVER_LIMIT: &[Listed] = &[
    Listed {
        file: "test_gpu_eval_api_oracle.rs",
        method: "eval_batch_auto",
        points: 300_000,
        nodes: 2,
        reason: "crosses the 262,144-point threshold where eval_batch_auto splits",
    },
    Listed {
        file: "gpu_eval.rs",
        method: "eval_batch",
        points: 10_000,
        nodes: 1,
        reason: "a batch larger than a workgroup dispatch returns every point",
    },
    Listed {
        file: "test_gpu_workgroup_size.rs",
        method: "eval_batch",
        points: 70_000,
        nodes: 0,
        reason: "past the dispatch limit at size 1: refused, nothing is submitted",
    },
    Listed {
        file: "test_gpu_workgroup_size.rs",
        method: "eval_batch_async",
        points: 70_000,
        nodes: 0,
        reason: "past the dispatch limit at size 1: refused, nothing is submitted",
    },
    Listed {
        file: "test_gpu_workgroup_size.rs",
        method: "eval_batch_pooled",
        points: 70_000,
        nodes: 0,
        reason: "past the dispatch limit at size 1: refused, nothing is submitted",
    },
];

/// The number written by `prefix` + digits at the start of or inside `s`
fn number_after(s: &str, prefix: &str) -> Option<usize> {
    let at = s.find(prefix)? + prefix.len();
    let digits: String = s[at..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '_')
        .filter(|c| *c != '_')
        .collect();
    let close = s[at..].chars().find(|c| !c.is_ascii_digit() && *c != '_');
    if close == Some(')') {
        digits.parse().ok()
    } else {
        None
    }
}

/// The element count of the array literal `[a, b, ...]` opening `s`, commas inside brackets aside
fn array_len(s: &str) -> Option<usize> {
    let s = s.trim();
    if !s.starts_with('[') {
        return None;
    }
    let (mut depth, mut commas, mut last_nonspace) = (0i32, 0usize, '[');
    for c in s.chars() {
        match c {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(if last_nonspace == '[' {
                        0
                    } else {
                        commas + usize::from(last_nonspace != ',')
                    });
                }
            }
            ';' if depth == 1 => return None,
            ',' if depth == 1 => commas += 1,
            _ => {}
        }
        if !c.is_whitespace() {
            last_nonspace = c;
        }
    }
    None
}

/// The count a `let` right-hand side builds, if it can be read
fn count_of(rhs: &str) -> Option<usize> {
    array_len(rhs.trim().trim_start_matches("vec!"))
        .or_else(|| number_after(rhs, ".take("))
        .or_else(|| number_after(rhs, "points("))
        .or_else(|| number_after(rhs, "(0.."))
}

/// The right-hand side of the `let` opening `lines`, up to its `;`
fn statement(lines: &[&str]) -> String {
    let text = lines.join("\n");
    let Some((_, rhs)) = text.split_once('=') else {
        return String::new();
    };
    let mut depth = 0i32;
    for (i, c) in rhs.char_indices() {
        match c {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth -= 1,
            ';' if depth == 0 => return rhs[..i].to_string(),
            _ => {}
        }
    }
    rhs.to_string()
}

/// Each direct submit of the source: (line, method, point count if readable)
fn submits(src: &str) -> Vec<(usize, String, Option<usize>)> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(at) = line.find(".eval_batch") else {
            continue;
        };
        let rest = &line[at + 1..];
        let Some(open) = rest.find('(') else { continue };
        let method = rest[..open].to_string();
        let arg = rest[open + 1..].trim_start().trim_start_matches('&');
        let name: String = arg
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let count = if arg.starts_with('[') {
            array_len(arg)
        } else if name.is_empty() {
            None
        } else {
            let lets = [
                format!("let {name} "),
                format!("let mut {name} "),
                format!("let {name}:"),
                format!("let mut {name}:"),
            ];
            let start = lines[..i]
                .iter()
                .rposition(|l| lets.iter().any(|k| l.contains(k.as_str())));
            start.and_then(|at| count_of(&statement(&lines[at..i])))
        };
        out.push((i + 1, method, count));
    }
    out
}

/// What is wrong with the submits of one file
fn problems(file: &str, src: &str) -> Vec<String> {
    submits(src)
        .into_iter()
        .filter_map(|(line, method, count)| match count {
            Some(n) if n <= MAX_POINTS_PER_SUBMIT => None,
            Some(n) if OVER_LIMIT.iter().any(|l| l.file == file && l.method == method && l.points == n) => None,
            Some(n) => Some(format!("{file}:{line}: {method} of {n} points, past the limit: use gpu_batches::eval_in_batches")),
            None => Some(format!("{file}:{line}: {method} of a point count the source does not show: use gpu_batches::eval_in_batches")),
        })
        .collect()
}

/// The part of a source file `src/` keeps for tests: from a `#[cfg(test)]` directly above a `mod`
fn test_module(src: &str) -> Option<String> {
    let lines: Vec<&str> = src.lines().collect();
    let at = lines
        .windows(2)
        .position(|w| w[0].trim() == "#[cfg(test)]" && w[1].trim_start().starts_with("mod "))?;
    Some(
        lines[at..]
            .iter()
            .filter(|l| !l.trim_start().starts_with("//"))
            .copied()
            .collect::<Vec<_>>()
            .join(
                "
",
            ),
    )
}

/// Every `.rs` file under `dir`, sorted
fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("a source directory") {
        let path = entry.expect("an entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
    out.sort();
}

#[test]
fn no_test_dispatches_past_the_limit() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (mut read, mut read_src, mut bad, mut seen) = (0usize, 0usize, Vec::new(), Vec::new());
    let mut check = |file: String, text: &str, from_src: bool| {
        let found = submits(text);
        if from_src {
            read_src += found.len();
        }
        read += found.len();
        for (_, method, count) in &found {
            seen.push((file.clone(), method.clone(), *count));
        }
        bad.extend(problems(&file, text));
    };
    let mut tests = Vec::new();
    rust_files(&root.join("tests"), &mut tests);
    for path in tests {
        let name = path
            .file_name()
            .expect("a name")
            .to_string_lossy()
            .to_string();
        let src = std::fs::read_to_string(&path).expect("a test file");
        if name != "test_gpu_submit_limit.rs"
            && name != "gpu_batches.rs"
            && src.contains("GpuEvaluator")
        {
            check(name, &src, false);
        }
    }
    let mut sources = Vec::new();
    rust_files(&root.join("src"), &mut sources);
    for path in sources {
        let src = std::fs::read_to_string(&path).expect("a source file");
        if src.contains("GpuEvaluator") {
            if let Some(tests) = test_module(&src) {
                check(
                    path.file_name()
                        .expect("a name")
                        .to_string_lossy()
                        .to_string(),
                    &tests,
                    true,
                );
            }
        }
    }
    println!("{read} direct GPU submits read, {read_src} of them in src");
    assert!(
        read > 0 && read_src > 0,
        "read {read} submits, {read_src} in src: the check checked too little"
    );
    assert!(
        bad.is_empty(),
        "{}",
        bad.join(
            "
"
        )
    );
    for l in OVER_LIMIT {
        assert!(!l.reason.is_empty(), "{}: no reason", l.file);
        assert!(
            l.points > MAX_POINTS_PER_SUBMIT,
            "{}: {} points needs no entry",
            l.file,
            l.points
        );
        assert!(
            l.points * l.nodes <= MAX_LISTED_WORK,
            "{}: {} points of {} nodes is too much work",
            l.file,
            l.points,
            l.nodes
        );
        assert!(
            seen.iter()
                .any(|(f, m, n)| f == l.file && m == l.method && *n == Some(l.points)),
            "{}: the listed {} of {} points is gone",
            l.file,
            l.method,
            l.points
        );
    }
}

#[test]
fn the_check_refuses_what_it_should() {
    let over = "let pts = points(2049);\nlet a = gpu.eval_batch(&pts).unwrap();\n";
    assert_eq!(problems("x.rs", over).len(), 1, "one past the limit");
    let at = "let pts = points(2_048);\nlet a = gpu.eval_batch(&pts).unwrap();\n";
    assert!(problems("x.rs", at).is_empty(), "at the limit");
    let grid = "let pts = points(lo, hi, 24);\nlet a = gpu.eval_batch(&pts).unwrap();\n";
    assert_eq!(
        problems("x.rs", grid).len(),
        1,
        "a grid's count is not read as its last argument"
    );
    let unknown = "let mut pts = Vec::new();\nlet a = gpu.eval_batch(&pts).unwrap();\n";
    assert_eq!(problems("x.rs", unknown).len(), 1, "an unreadable count");
    let shadow =
        "let pts = points(8);\nlet pts = points(9000);\nlet a = gpu.eval_batch_async(&pts);\n";
    assert_eq!(
        problems("x.rs", shadow).len(),
        1,
        "the nearest let decides, any eval_batch method"
    );
    let array = "let pts = [a(), b(1, 2)];\nlet a = gpu.eval_batch(&pts).unwrap();\nlet b = gpu.eval_batch(&[p]).unwrap();\nlet c = gpu.eval_batch(&[]).unwrap();\n";
    let found = submits(array);
    assert_eq!(
        found.iter().map(|s| s.2).collect::<Vec<_>>(),
        [Some(2), Some(1), Some(0)]
    );
    assert!(problems("x.rs", array).is_empty());
    let listed =
        "let pts = points(300_000);\nlet a = gpu.eval_batch_auto(&pts, &mut pool).unwrap();\n";
    assert!(
        problems("test_gpu_eval_api_oracle.rs", listed).is_empty(),
        "the listed size"
    );
    let other =
        "let pts = points(300_001);\nlet a = gpu.eval_batch_auto(&pts, &mut pool).unwrap();\n";
    assert_eq!(
        problems("test_gpu_eval_api_oracle.rs", other).len(),
        1,
        "another size of a listed submit"
    );
    assert_eq!(
        problems("elsewhere.rs", listed).len(),
        1,
        "the listed size in another file"
    );
    let range = "let points: Vec<Vec3> = (0..2049).map(f).collect();\nlet a = gpu.eval_batch(&points).unwrap();\n";
    assert_eq!(problems("x.rs", range).len(), 1, "a range past the limit");
    let module = "fn live() { gpu.eval_batch(&x); }\n#[cfg(test)]\nfn helper() {}\n#[cfg(test)]\nmod tests {\n    // gpu.eval_batch(&y)\n    let a = gpu.eval_batch(&z);\n}\n";
    assert_eq!(
        submits(&test_module(module).expect("a module")).len(),
        1,
        "only the test module, comments aside"
    );
    assert!(test_module("fn live() {}\n").is_none());
    let multi = "let pts = vec![\n    a,\n    b,\n];\nlet a = gpu.eval_batch(&pts).unwrap();\n";
    assert_eq!(submits(multi)[0].2, Some(2), "an array over several lines");
    let later =
        "let pts = Vec::new();\nlet c = other.take(3);\nlet a = gpu.eval_batch(&pts).unwrap();\n";
    assert_eq!(
        problems("x.rs", later).len(),
        1,
        "a count in a later statement is not this one's"
    );
    let chain = "let pts: Vec<Vec3> = points(4000)\n    .into_iter()\n    .collect();\nlet a = gpu.eval_batch(&pts).unwrap();\n";
    assert_eq!(
        problems("x.rs", chain).len(),
        1,
        "a chain over several lines"
    );
    let repeat = "let a = gpu.eval_batch(&[p; 9000]).unwrap();\n";
    assert_eq!(
        problems("x.rs", repeat).len(),
        1,
        "a repeat array's count is not read as one element"
    );
    assert!(
        problems("x.rs", "let a = eval_in_batches(&gpu, &pts).unwrap();\n").is_empty(),
        "the helper is not a direct submit"
    );
}

#[test]
fn batches_keep_the_order_and_the_limit() {
    let pts: Vec<Vec3> = (0..=(2 * MAX_POINTS_PER_SUBMIT))
        .map(|i| Vec3::splat(i as f32))
        .collect();
    let (mut submits_seen, mut largest) = (0usize, 0usize);
    let got = in_batches::<_, ()>(&pts, |c| {
        submits_seen += 1;
        largest = largest.max(c.len());
        Ok(c.iter().map(|p| p.x).collect())
    })
    .expect("no error");
    assert_eq!((submits_seen, largest), (3, MAX_POINTS_PER_SUBMIT));
    assert_eq!(got, pts.iter().map(|p| p.x).collect::<Vec<_>>());
    assert_eq!(MAX_POINTS_PER_SUBMIT, 2048);

    let mut none = 0;
    let empty = in_batches::<f32, ()>(&[], |_| {
        none += 1;
        Ok(Vec::new())
    });
    assert_eq!((empty, none), (Ok(Vec::new()), 0));

    let mut calls = 0;
    let failed = in_batches::<f32, &str>(&pts, |_| {
        calls += 1;
        Err("no")
    });
    assert_eq!((failed, calls), (Err("no"), 1), "the first error stops");
}
