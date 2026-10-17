#!/usr/bin/env python3
"""No build configuration sets a flag that changes floating-point results.

ALICE-Law G14 (float-environment-invariants): a Law-bearing crate is not
built with a fast-math-style flag (reassociation, contraction to fma,
flushing subnormals to zero, assuming no NaN / infinity / signed zero) or
with `target-cpu=native` (which selects instructions for whichever machine
builds it). This is the static half of the check: it reads every file a flag
can come from and reports every occurrence outside a comment, whatever
surrounds it. scripts/build_flags_check.py is the dynamic half: it reads the
flags cargo actually passes to rustc. The file is generic and is kept
byte-identical across the repositories that carry it.

Files read, anywhere under the root (target/, node_modules/ and .git/ are
skipped), by category:

    cargo-config   .cargo/config, .cargo/config.toml
    toml           every other *.toml (Cargo.toml, pyproject.toml, Makefile.toml,
                   Cross.toml, ...); every TOML file must parse
    build-rs       build.rs
    rust           every other *.rs
    yaml           *.yml, *.yaml (workflows, actions, other CI and compose files)
    json           *.json (devcontainer, editor and tool settings)
    script         *.sh, *.bash, *.zsh, *.py, *.ps1, *.psm1, *.nix, Makefile,
                   makefile, GNUmakefile, *.mk, justfile, Justfile, Dockerfile*,
                   *.dockerfile, .envrc, .env, .env.*, *.env, *.bat, *.cmd
    wrapper        a file named as RUSTC_WRAPPER / RUSTC_WORKSPACE_WRAPPER / RUSTC
                   (env) or rustc-wrapper / rustc-workspace-wrapper / rustc (cargo
                   config) that exists in the tree and is not read already

Comments are blanked first: `#` comments in TOML, YAML and the script kinds
(a `#` that starts the line or follows whitespace, outside quotes), `//` and
`/* */` in Rust and JSON, `rem` / `::` lines in batch files. The value of a
YAML `name:` key is blanked too (a step or job label configures nothing).
String contents are kept, since flags live in strings. For TOML and JSON the
decoded string values are checked as well (an escape such as \\u0069 is
decoded first); a decoded value is reported at line 0 when it does not appear
verbatim in the file.

Closure: the value after `target-cpu=` (or target_cpu), `target-feature=` and
`llvm-args=` must be a plain literal: characters from [A-Za-z0-9_.-] (plus
`+` and `,` for a feature list, `=`, `:`, `/` and `,` for llvm-args), ending
at whitespace, the end of the line, `,` `]` `)` `;` `|` `&` `<` `>`, or a
closing quote (with `#` allowed after it, for a Rust raw string) that is not
followed by more of the word or by `+` / `%` / `.` (concatenation or
formatting). Anything else is unresolved: an empty value, a
quote right after `=`, a backslash escape, `$`, `%`, `{`, a quote splicing
the word, a continuation on the next line. An unresolved value is a finding,
because the guard cannot know what it becomes.

Rules, one per occurrence:

    target-cpu-native          target-cpu=native
    target-cpu-unresolved      target-cpu= with a value that is not a plain literal
    target-feature-unresolved  the same for target-feature=
    llvm-args-unresolved       the same for llvm-args=
    rustflags-unresolved       the value of a RUSTFLAGS / RUSTDOCFLAGS /
                               CARGO_*RUSTFLAGS / rustflags assignment, key or call
                               argument taken from a command or a file: $( ),
                               backticks, a < redirect, cat, open(, read_text(,
                               read_to_string
    llvm-fast-math             LLVM options that relax IEEE 754: enable-unsafe-fp-math,
                               enable-no-nans-fp-math, enable-no-infs-fp-math,
                               enable-no-signed-zeros-fp-math, fp-contract=fast,
                               denormal-fp-math (any value)
    c-fast-math                C / C++ flags of the same kind: -ffast-math, -Ofast,
                               -funsafe-math-optimizations, -fassociative-math,
                               -freciprocal-math, -ffinite-math-only,
                               -fno-signed-zeros, -ffp-contract=fast,
                               -ffp-model=fast|aggressive, -fdenormal-fp-math,
                               -mdaz-ftz, /fp:fast, -fp-model fast, -march=native,
                               -mcpu=native, -mfma (a C compiler contracts a * b + c
                               into fma by default once fma is available), -mno-sse,
                               -mno-sse2, -mfpmath=387
    x87-float                  a plain target-feature list that disables sse or sse2
                               (x86 then computes in x87 extended precision)
    fast-intrinsic             (*.rs under a src/ directory) fadd_fast / fsub_fast /
                               fmul_fast / fdiv_fast / frem_fast, the *_algebraic
                               intrinsics, the algebraic_add / _sub / _mul / _div /
                               _rem methods, #![feature(core_intrinsics)] and
                               #![feature(float_algebraic)]
    unparseable                a TOML file that does not parse
    unreadable                 a file read here that is not UTF-8

Not findings, by policy: a plain Rust target-feature list that adds or
removes instructions other than sse / sse2 (+avx2, +fma, +dotprod, -avx2,
-fma, -sse4.1), a plain target-cpu other than native (x86-64-v2),
#[target_feature(enable = ...)] in source, and a Rust cfg predicate
`target_feature = "avx2"` (spaces around `=` and a quoted value: the form of a
cfg, which a rustc flag never takes). Rust never contracts a * b + c
into an fma, mul_add is fused whether or not fma is enabled, and the x86_64
float ABI is SSE2 throughout, so these select different instructions for the
same IEEE 754 results. The tests pin this policy.

Allowlist (per repository, optional): scripts/build_config_allowlist.toml
(this file itself is not scanned)

    [[allow]]          a deliberate occurrence: it covers the findings in `path`
    path = "..."       whose comment-blanked line, stripped of surrounding
    line = "..."       whitespace (or decoded value, for a line-0 finding), is
    count = 1          exactly `line`; `count` (default 1) must equal the number
    reason = "..."     of lines it covers, so a new occurrence is a finding
    [[exclude]]        a file that is not read: only a file no build reads (the
    path = "..."       two G14 scripts and their tests, documents and data that
                       quote flags).
    reason = "..."     An exclude is rejected for .cargo/config*, any *.toml in a
                       .cargo directory, any Cargo.toml or pyproject.toml,
                       build.rs, a rustc wrapper, a workflow or action file, and
                       a script named in a workflow or action file (other than
                       this script, scripts/build_flags_check.py and their tests):
                       those take [[allow]] entries

An entry without a non-empty reason, with an unknown key or a wrong type, an
allow whose count differs from what it covers (zero included), an exclude of
a file this script does not read or of a rejected kind, and an allowlist that
is not valid TOML are problems (rule `allowlist`). Each entry is printed with
what it covered.

YAML is not parsed here (the standard library has no YAML parser; the
workflow files are parsed by the repository's actionlint job), so a YAML
escape that spells a whole flag is left to the dynamic half.

Output: `path:line: rule: text` per problem, the allowed occurrences and the
entries, then the counts per category. Exit 0 clean, 1 problems, 2 when
nothing was read or the root has no Cargo.toml.

    python3 scripts/build_config_guard.py [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

SKIP_DIRS = {"target", "node_modules", ".git"}
ALLOWLIST = "scripts/build_config_allowlist.toml"
SELF = {"scripts/build_config_guard.py", "scripts/test_build_config_guard.py",
        "scripts/build_flags_check.py", "scripts/test_build_flags_check.py"}
CATEGORIES = ["cargo-config", "toml", "build-rs", "rust", "yaml", "json", "script", "wrapper"]

# key -> (rule for an unresolved value, characters of a plain value)
CLOSED_KEYS = [
    (re.compile(r"target[-_]cpu\s*=\s*"), "target-cpu", r"[A-Za-z0-9_.-]"),
    (re.compile(r"target[-_]feature\s*=\s*"), "target-feature", r"[A-Za-z0-9_.+,-]"),
    (re.compile(r"llvm[-_]args\s*=\s*"), "llvm-args", r"[A-Za-z0-9_.=+,:/-]"),
]
DELIMS = set(" \t,])};|&<>")
QUOTES = "\"'`"
LLVM_FAST = re.compile(
    r"enable-(?:unsafe|no-nans|no-infs|no-signed-zeros)-fp-math"
    r"|(?<![\w-])-?fp-contract=fast|(?<![\w-])-?denormal-fp-math")
PLAIN_RULES = [
    ("llvm-fast-math", LLVM_FAST),
    ("c-fast-math", re.compile(
        r"(?<![\w-])-(?:ffast-math|Ofast\b|funsafe-math-optimizations|fassociative-math"
        r"|freciprocal-math|ffinite-math-only|fno-signed-zeros|ffp-contract=fast"
        r"|ffp-model=(?:fast|aggressive)\b|fdenormal-fp-math|mdaz-ftz|march=native\b"
        r"|mcpu=native\b|mfma\b|mno-sse2?\b|mfpmath=387\b)"
        r"|/fp:fast|-fp-model[ =]fast")),
]
# a RUSTFLAGS-family key followed by its value (assignment, YAML / TOML key, call argument)
FLAGS_KEY = re.compile(
    r"(?i)\b(?:CARGO_\w*RUSTFLAGS|RUSTDOCFLAGS|RUSTFLAGS|rustflags)\b[\"']?\]?\s*(?:\+?=|:|,)\s*")
FROM_COMMAND = re.compile(r"\$\(|`|<\s*[\"'$\w./]|\bcat\s|\bopen\s*\(|read_text\s*\(|read_to_string|\$\(<")
SOURCE_RULES = [
    ("fast-intrinsic", re.compile(
        r"\b(?:f(?:add|sub|mul|div|rem)_(?:fast|algebraic)|algebraic_(?:add|sub|mul|div|rem))\b"
        r"|feature\s*\(\s*(?:core_intrinsics|float_algebraic)\b")),
]
YAML_NAME = re.compile(r"^(\s*(?:-\s+)?)name\s*:")
WRAPPER_REF = re.compile(
    r"\b(?:RUSTC_WORKSPACE_WRAPPER|RUSTC_WRAPPER|RUSTC|rustc-workspace-wrapper|rustc-wrapper|rustc)"
    r"\s*[:=]\s*[\"']?([^\s\"'$]+)")
SCRIPT_SUFFIXES = {".sh", ".bash", ".zsh", ".py", ".ps1", ".psm1", ".nix", ".mk",
                   ".dockerfile", ".env", ".bat", ".cmd"}
SCRIPT_NAMES = {"Makefile", "makefile", "GNUmakefile", "justfile", "Justfile", ".envrc", ".env"}


def category(rel: str) -> str | None:
    """The category of a file this guard reads, or None to skip it."""
    p = PurePosixPath(rel)
    name, parts, suffix = p.name, p.parts, p.suffix.lower()
    if rel == ALLOWLIST:
        return None
    if name in ("config", "config.toml") and len(parts) >= 2 and parts[-2] == ".cargo":
        return "cargo-config"
    if suffix == ".toml":
        return "toml"
    if name == "build.rs":
        return "build-rs"
    if suffix == ".rs":
        return "rust"
    if suffix in (".yml", ".yaml"):
        return "yaml"
    if suffix == ".json":
        return "json"
    if name in SCRIPT_NAMES or suffix in SCRIPT_SUFFIXES or name.startswith(".env.") \
            or name.startswith("Dockerfile"):
        return "script"
    return None


def is_ci_file(rel: str) -> bool:
    p = PurePosixPath(rel)
    return (len(p.parts) >= 3 and p.parts[0] == ".github" and p.parts[1] in ("workflows", "actions")
            and p.suffix.lower() in (".yml", ".yaml")) or p.name in ("action.yml", "action.yaml")


def exclude_refusal(rel: str, cat: str, ci_text: str) -> str | None:
    """Why `rel` may not be excluded, or None when it may."""
    p = PurePosixPath(rel)
    if cat in ("cargo-config", "build-rs", "wrapper"):
        return f"{cat} is read by the build"
    if p.name in ("Cargo.toml", "pyproject.toml"):
        return f"{p.name} is read by the build"
    if p.suffix.lower() == ".toml" and ".cargo" in p.parts[:-1]:
        return "a .cargo directory is read by cargo"
    if is_ci_file(rel):
        return "a workflow or action file sets the CI build"
    if cat == "script" and rel not in SELF and re.search(r"(?<![\w./-])" + re.escape(rel) + r"(?![\w-])", ci_text):
        return "a script named in a workflow or action file runs in CI"
    return None


def blank_hash_comments(line: str) -> str:
    """Drop a `#` comment that starts the line or follows whitespace, outside quotes."""
    quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\" and quote == '"':
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
        i += 1
    return line


def blank_slash_comments(text: str) -> str:
    """Replace // and /* */ comments with spaces, keeping strings and line numbers."""
    out = []
    i, n = 0, len(text)
    depth = 0
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if depth:
            if c == "/" and nxt == "*":
                depth += 1
                out.append("  ")
                i += 2
            elif c == "*" and nxt == "/":
                depth -= 1
                out.append("  ")
                i += 2
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
            continue
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and nxt == "*":
            depth = 1
            out.append("  ")
            i += 2
        elif c == "r" and re.match(r'r#*"', text[i:]) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'r(#*)"', text[i:])
            close = '"' + m.group(1)
            j = text.find(close, i + len(m.group(0)))
            j = n if j < 0 else j + len(close)
            out.append(text[i:j])
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(text[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def code_lines(rel: str, text: str, cat: str) -> list[str]:
    if cat in ("build-rs", "rust", "json"):
        return blank_slash_comments(text).split("\n")
    lines = text.split("\n")
    if PurePosixPath(rel).suffix.lower() in (".bat", ".cmd"):
        return ["" if re.match(r"\s*(?:@?rem\b|::)", l, re.I) else l for l in lines]
    lines = [blank_hash_comments(l) for l in lines]
    if cat == "yaml":
        lines = [m.group(1) if (m := YAML_NAME.match(l)) else l for l in lines]
    return lines


def closed_value(line: str, start: int, chars: str) -> str | None:
    """The plain literal starting at `start`, or None when the value is not plain."""
    m = re.match(chars + "*", line[start:])
    value = m.group(0)
    end = start + len(value)
    if not value:
        return None
    if end >= len(line) or line[end] in DELIMS:
        return value
    if line[end] in QUOTES:
        rest = line[end + 1:]
        if not rest or rest[0] in DELIMS or rest[0] in QUOTES or rest[0] == "#":
            after = rest.lstrip(" \t")
            if after[:1] in ("+", "%", "."):
                return None
            return value
        return None
    return None


def check_line(line: str, rules_src: bool, rust: bool = False) -> list[tuple[str, str]]:
    """(rule, matched text) for every occurrence on one comment-blanked line."""
    hits = []
    for pattern, key, chars in CLOSED_KEYS:
        for m in pattern.finditer(line):
            if rust and key == "target-feature" and line[m.end():m.end() + 1] == '"' \
                    and "=" in line[m.start():m.end()] and re.search(r"\s=|=\s", line[m.start():m.end()]):
                continue  # cfg(target_feature = "avx2"): a cfg predicate in Rust, not a flag
            value = closed_value(line, m.end(), chars)
            text = line[m.start():m.end()] + (value or line[m.end():m.end() + 12])
            if value is None:
                hits.append((f"{key}-unresolved", text))
            elif key == "target-cpu" and value == "native":
                hits.append(("target-cpu-native", text))
            elif key == "target-feature" and any(f in ("-sse", "-sse2") for f in value.split(",")):
                hits.append(("x87-float", text))
    for m in FLAGS_KEY.finditer(line):
        rest = line[m.end():]
        if rest[:1] in "\"'":
            q = rest[0]
            j = 1
            while j < len(rest) and rest[j] != q:
                j += 2 if rest[j] == "\\" else 1
            value = rest[:j + 1]
        else:
            value = re.match(r"[^\s;&|]*", rest).group(0)
        if FROM_COMMAND.search(value):
            hits.append(("rustflags-unresolved", (line[m.start():m.end()] + value)[:80]))
    for rule, pattern in PLAIN_RULES + (SOURCE_RULES if rules_src else []):
        for m in pattern.finditer(line):
            hits.append((rule, m.group(0)))
    return hits


def decoded_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from decoded_strings(k)
            yield from decoded_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from decoded_strings(v)


def walk(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]) or not path.is_file():
            continue
        yield rel.as_posix(), path


def load_allowlist(root: Path) -> tuple[list[dict], list[dict], list[str]]:
    """The allow and exclude entries, and the problems in the file itself."""
    path = root / ALLOWLIST
    if not path.is_file():
        return [], [], []
    if tomllib is None:
        return [], [], [f"{ALLOWLIST}:0: allowlist: Python 3.11+ (tomllib) is needed to read it"]
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        return [], [], [f"{ALLOWLIST}:0: allowlist: not readable as TOML ({e.__class__.__name__})"]
    problems = []
    unknown = set(data) - {"allow", "exclude"}
    if unknown:
        problems.append(f"{ALLOWLIST}:0: allowlist: unknown table {sorted(unknown)}")
    entries = {}
    spec = {"allow": (("path", "line", "reason"), ("count",)), "exclude": (("path", "reason"), ())}
    for table, (required, optional) in spec.items():
        rows = data.get(table, [])
        if not isinstance(rows, list):
            problems.append(f"{ALLOWLIST}:0: allowlist: [[{table}]] must be an array of tables")
            rows = []
        good = []
        for i, row in enumerate(rows):
            why = []
            bad = [k for k in required if not isinstance(row.get(k), str) or not row[k].strip()]
            if bad:
                why.append(f"empty or missing {', '.join(bad)}")
            extra = sorted(set(row) - set(required) - set(optional))
            if extra:
                why.append(f"unknown key {', '.join(extra)}")
            if "count" in row and (not isinstance(row["count"], int) or isinstance(row["count"], bool)
                                   or row["count"] < 1):
                why.append("count must be a positive integer")
            if why:
                problems.append(f"{ALLOWLIST}:0: allowlist: {table} entry {i + 1}: {'; '.join(why)}")
                continue
            good.append(row)
        entries[table] = good
    return entries["allow"], entries["exclude"], problems


def wrapper_targets(rel: str, text: str, root: Path) -> set[str]:
    """Files in the tree named as a rustc wrapper in this file."""
    found = set()
    base = PurePosixPath(rel).parent
    bases = [PurePosixPath(""), base]
    if base.name == ".cargo":
        bases.append(base.parent)
    for m in WRAPPER_REF.finditer(text):
        ref = m.group(1).strip(",;)")
        if ref.startswith("/"):
            continue
        for b in bases:
            cand = (b / ref).as_posix()
            while cand.startswith("./"):
                cand = cand[2:]
            if (root / cand).is_file():
                found.add(cand)
    return found


def scan(root: Path):
    allow, exclude, problems = load_allowlist(root)
    counts = {c: 0 for c in CATEGORIES}
    root_manifest = (root / "Cargo.toml").is_file()
    findings = []
    files = [(rel, path, category(rel)) for rel, path in walk(root)]
    readable = {rel: cat for rel, _, cat in files if cat}
    ci_text = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                        for rel, path, cat in files if cat and is_ci_file(rel))
    # wrappers first: they are build-read files, so they cannot be excluded either
    queue = [(rel, path, cat) for rel, path, cat in files if cat]
    pending = list(queue)
    wrappers: dict[str, Path] = {}
    while pending:
        rel, path, cat = pending.pop(0)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for w in sorted(wrapper_targets(rel, "\n".join(code_lines(rel, text, cat)), root)):
            if w not in readable and w not in wrappers:
                wrappers[w] = root / w
                pending.append((w, root / w, "wrapper"))
    for w, path in wrappers.items():
        readable[w] = "wrapper"
        queue.append((w, path, "wrapper"))
    excluded = {}
    for e in exclude:
        cat = readable.get(e["path"])
        if cat is None:
            problems.append(f"{ALLOWLIST}:0: allowlist: exclude {e['path']!r} is not a file this script reads")
            continue
        why = exclude_refusal(e["path"], cat, ci_text)
        if why:
            problems.append(f"{ALLOWLIST}:0: allowlist: exclude {e['path']!r} is refused ({why}): use [[allow]]")
            continue
        excluded[e["path"]] = e["reason"]
    for rel, path, cat in queue:
        if rel in excluded:
            continue
        counts[cat] += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            problems.append(f"{rel}:0: unreadable: {e.__class__.__name__}")
            continue
        decoded = None
        suffix = PurePosixPath(rel).suffix.lower()
        if suffix == ".toml":
            if tomllib is None:
                problems.append(f"{rel}:0: unparseable: Python 3.11+ (tomllib) is needed")
            else:
                try:
                    decoded = tomllib.loads(text)
                except tomllib.TOMLDecodeError as e:
                    problems.append(f"{rel}:0: unparseable: {e}")
        lines = code_lines(rel, text, cat)
        if suffix == ".json":
            try:
                decoded = json.loads("\n".join(lines))
            except ValueError:
                decoded = None  # JSON with comments / trailing commas: lines are still read
        src = cat == "rust" and "src" in PurePosixPath(rel).parts[:-1]
        for lineno, line in enumerate(lines, start=1):
            for rule, hit in check_line(line, src, cat in ("rust", "build-rs")):
                findings.append((rel, lineno, rule, hit, line.strip()))
        if decoded is not None:
            for s in decoded_strings(decoded):
                if s in text:
                    continue
                for rule, hit in check_line(s, False):
                    findings.append((rel, 0, rule, hit + " (decoded value)", s.strip()))
    allowed = []
    lines_covered = [set() for _ in allow]
    for rel, lineno, rule, hit, line in findings:
        cover = [i for i, a in enumerate(allow) if a["path"] == rel and a["line"].strip() == line]
        msg = f"{rel}:{lineno}: {rule}: {hit}"
        if cover:
            lines_covered[cover[0]].add((lineno, line))
            allowed.append(f"allowed {msg} (entry {cover[0] + 1})")
        else:
            problems.append(msg)
    entries = []
    for i, a in enumerate(allow):
        want = a.get("count", 1)
        got = len(lines_covered[i])
        entries.append(f"allow entry {i + 1}: {a['path']}: {got} of {want} lines ({a['reason']})")
        if got != want:
            problems.append(f"{ALLOWLIST}:0: allowlist: allow entry {i + 1} ({a['path']}: {a['line']!r}) "
                            f"covers {got} lines, count is {want}")
    for path, reason in excluded.items():
        entries.append(f"exclude: {path}: not read ({reason})")
    return problems, allowed, entries, counts, root_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=Path(__file__).resolve().parent.parent, type=Path)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    problems, allowed, entries, counts, root_manifest = scan(root)
    read = sum(counts.values())
    for line in problems + allowed + entries:
        print(line)
    per = ", ".join(f"{c} {counts[c]}" for c in CATEGORIES)
    print(f"build_config_guard: {read} files read ({per}), {len(allowed)} allowed, {len(problems)} problems")
    if read == 0 or not root_manifest:
        print("build_config_guard: nothing to check (no root Cargo.toml or no file read)", file=sys.stderr)
        return 2
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
