//! No test sends more than `MAX_POINTS_PER_SUBMIT` points to the GPU in one direct submit.
//!
//! A source check, so it needs no GPU and every test run has it. In a test file that names
//! `GpuEvaluator`, each `.eval_batch*(` call is a direct submit. Its point count must be readable from
//! the source: a literal array, or a `let` of the argument's name built by `points(N)` or
//! `.take(N)`. A count that cannot be read is refused like one over the limit: such a test calls
//! `gpu_batches::eval_in_batches`. Reading no site at all fails.
//!
//! Author: Moroya Sakamoto

#[path = "common/gpu_batches.rs"]
mod gpu_batches;

use glam::Vec3;
use gpu_batches::{in_batches, MAX_POINTS_PER_SUBMIT};
use std::path::Path;

/// Submits that must exceed the limit because the size is what they test: (file, method, reason)
const OVER_LIMIT: &[(&str, &str, &str)] = &[(
    "test_gpu_eval_api_oracle.rs",
    "eval_batch_auto",
    "300,000 points cross the 262,144-point threshold where it splits",
)];

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
    array_len(rhs)
        .or_else(|| number_after(rhs, ".take("))
        .or_else(|| number_after(rhs, "points("))
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
            lines[..i]
                .iter()
                .rev()
                .find_map(|l| lets.iter().find_map(|k| l.find(k.as_str()).map(|_| *l)))
                .and_then(|l| count_of(l.split_once('=')?.1))
        };
        out.push((i + 1, method, count));
    }
    out
}

/// What is wrong with the submits of one file
fn problems(file: &str, src: &str) -> Vec<String> {
    submits(src)
        .into_iter()
        .filter(|(_, method, _)| !OVER_LIMIT.iter().any(|(f, m, _)| *f == file && m == method))
        .filter_map(|(line, method, count)| match count {
            Some(n) if n <= MAX_POINTS_PER_SUBMIT => None,
            Some(n) => Some(format!("{file}:{line}: {method} of {n} points, past the limit: use gpu_batches::eval_in_batches")),
            None => Some(format!("{file}:{line}: {method} of a point count the source does not show: use gpu_batches::eval_in_batches")),
        })
        .collect()
}

#[test]
fn no_test_dispatches_past_the_limit() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let (mut read, mut bad) = (0usize, Vec::new());
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .expect("tests")
        .map(|e| e.expect("entry").path())
        .collect();
    names.sort();
    for path in names {
        if path.extension().is_none_or(|x| x != "rs")
            || path
                .file_name()
                .is_some_and(|n| n == "test_gpu_submit_limit.rs")
        {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("a test file");
        if !src.contains("GpuEvaluator") {
            continue;
        }
        let file = path
            .file_name()
            .expect("a name")
            .to_string_lossy()
            .to_string();
        read += submits(&src).len();
        bad.extend(problems(&file, &src));
    }
    println!("{read} direct GPU submits read");
    assert!(read > 0, "no GPU submit read: the check checked nothing");
    assert!(bad.is_empty(), "{}", bad.join("\n"));
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
    let pts: Vec<Vec3> = (0..2 * MAX_POINTS_PER_SUBMIT + 1)
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
