#!/usr/bin/env python3
"""No deterministic path of the crate draws OS entropy or reads the wall clock.

An estimate this crate computes (a quantile, a cardinality, a noisy count) is
meant to compare equal given the same input, on any machine, at any time, so
no part of production code may depend on the wall clock or on OS randomness:
either one would make the same input produce a different estimate on a
different run. Randomness a caller wants belongs to the caller, passed in
explicitly (`XorShift64::new(seed)` / `with_seed` / `with_probability` /
`with_seed_params`); this crate's own production code never reaches for it on
its own, except the one deprecated path documented below.

Checked in every src/**/*.rs file, outside `#[cfg(test)]` / `#[cfg(all(test,
...))]` modules, with the code of a file joined (a call split over lines still
matches), `use .. as` aliases of `Instant` / `SystemTime` resolved, and a
fully-qualified-self call (`<std::time::Instant>::now()`) recognised the same
as the plain form:

    getrandom  OsRng  thread_rng  rand::random  from_entropy(  try_from_entropy(
    SystemTime::now  Instant::now  UNIX_EPOCH  RandomState (any use, not only
    `::new`: `::default()` reaches the same OS-seeded hasher)  Uuid::new_v4

clippy's `disallowed-methods` in clippy.toml is the first layer (compiled,
`cargo clippy --all-targets -D warnings`); this script is the second, readable
without compiling and run on `src/` only (clippy's `--all-targets` also covers
tests/benches/examples/fuzz, where a scoped `#[allow(clippy::disallowed_methods,
reason = "...")]` carries the same per-call-site accounting this script does).

Deliberately not scanned: `HashMap` / `HashSet` (their default hasher also
seeds from OS randomness, but that is a separate gate -- iteration order, not
entropy in a value -- this crate has no HashMap/HashSet in src/ today).

A definition (`fn from_entropy`) is not a call. The call sites that stay are
this repository's own data, listed in scripts/entropy_guard.toml (this script
is shared byte-for-byte between repositories, so it carries none):

    [[allow]]
    path = "src/bin/main.rs"
    pattern = "Instant::now"
    count = 14
    reason = "CLI timing printed to the terminal, not stored"

Each entry gives the path, a pattern name above, the exact count and a reason:
a count that drifts either way fails, so a new call cannot hide behind an old
entry and a removed one must be removed there too; an entry that matches
nothing fails as stale; the same site listed twice, an unknown pattern or key,
or a file that does not parse fails. No file means no allowance (every call is
a finding), never a skip. Scanning no file, or no pattern, fails.

    python3 scripts/entropy_guard.py
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A qualified-self call wraps the type in `<...>` and may add `as Trait`
# before the closing `>` (`<std::time::Instant>::now()`, `<std::time::Instant
# as SomeTrait>::now()`): the type name's own `\b` boundary does not care about
# the angle brackets, so only the `::method` suffix needs to tolerate them.
_QUALIFIED_SELF = r"(?:\s+as\s+\w+)?\s*>?\s*"
PATTERNS = {
    "getrandom": r"\bgetrandom\b",
    "OsRng": r"\bOsRng\b",
    "thread_rng": r"\bthread_rng\b",
    "rand::random": r"\brand\s*::\s*random\b",
    "from_entropy(": r"(?<!fn )\b(?:try_)?from_entropy\s*\(",
    "SystemTime::now": r"\bSystemTime\b" + _QUALIFIED_SELF + r"::\s*now\b",
    "Instant::now": r"\bInstant\b" + _QUALIFIED_SELF + r"::\s*now\b",
    "UNIX_EPOCH": r"\bUNIX_EPOCH\b",
    # Not `::new` only: `RandomState::default()` seeds the same OS-randomness
    # hasher and clippy's `disallowed-methods` cannot ban a trait method by
    # the implementing type alone, so this guard is the only layer that can
    # cover it (both: `\bRandomState\b` matches a bare `use` of the type too,
    # which is fine -- a `use` with no call is harmless, but it is so rare in
    # practice that the simpler, safer-by-default bare-word match is kept).
    "RandomState": r"\bRandomState\b",
    "Uuid::new_v4": r"\bUuid\b" + _QUALIFIED_SELF + r"::\s*new_v4\b",
}
# `use std::time::Instant as Clock;` makes `Clock::now()` the same call
ALIASABLE = {
    "Instant": ("Instant::now", r"\b{}\b" + _QUALIFIED_SELF + r"::\s*now\b"),
    "SystemTime": ("SystemTime::now", r"\b{}\b" + _QUALIFIED_SELF + r"::\s*now\b"),
}
ALIAS_RE = re.compile(r"\b(Instant|SystemTime)\s+as\s+(\w+)")
# this repository's allowances: (file, pattern) -> (count, reason), read by load_allowed
ALLOW_FILE = "scripts/entropy_guard.toml"
ALLOW_KEYS = {"path", "pattern", "count", "reason"}


def load_allowed(root: Path) -> tuple[dict[tuple[str, str], tuple[int, str]], list[str]]:
    """(allowances, problems) from ALLOW_FILE under `root`; an absent file is no allowance"""
    f = root / ALLOW_FILE
    if not f.is_file():
        return {}, []
    try:
        data = tomllib.loads(f.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        return {}, [f"{ALLOW_FILE} is not TOML: {e}"]
    problems: list[str] = []
    if set(data) - {"allow"} or not isinstance(data.get("allow", []), list):
        problems.append(f"{ALLOW_FILE}: only [[allow]] entries are read (found {sorted(data)})")
    allowed: dict[tuple[str, str], tuple[int, str]] = {}
    for i, e in enumerate(data.get("allow", []) if isinstance(data.get("allow", []), list) else [], 1):
        where = f"{ALLOW_FILE} entry {i}"
        if not isinstance(e, dict):
            problems.append(f"{where}: not a table")
            continue
        extra = set(e) - ALLOW_KEYS
        if extra:
            problems.append(f"{where}: unknown key {sorted(extra)}")
        path, name, count, reason = e.get("path"), e.get("pattern"), e.get("count"), e.get("reason", "")
        if not isinstance(path, str) or not path:
            problems.append(f"{where}: path must be a file under src/")
            continue
        if name not in PATTERNS:
            problems.append(f"{where}: unknown pattern {name!r} (one of {sorted(PATTERNS)})")
            continue
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            problems.append(f"{where}: count must be a positive integer (got {count!r})")
            continue
        if not isinstance(reason, str):
            reason = ""
        if (path, name) in allowed:
            problems.append(f"{where}: {name} in {path} listed twice")
            continue
        allowed[(path, name)] = (count, reason)
    return allowed, problems
# A char literal ('{', '\n', '\u{1F600}', also after a `b` prefix) but not a
# lifetime ('a, 'static: those have no closing quote, so they never match.
CHAR_RE = re.compile(r"'(?:\\u\{[0-9a-fA-F_]+\}|\\x[0-9a-fA-F]{2}|\\.|[^'\\\n])'")
RAW_RE = re.compile(r'b?r(#*)"')


def sanitize(text: str) -> str:
    """One pass over the source, in order: string (escapes), raw string
    (`r#".."#`, `br".."`), char literal, `//` comment, `/* */` comment
    (nested). A string becomes `""`, a char `''`, a comment nothing; newlines
    are kept so line numbers stay. Because every construct is read where it
    starts, a `/*` inside a string or a line comment, or a quote inside a
    comment, opens nothing. Without this, a char literal that happens to spell
    a brace (`'{'`) throws off the brace-depth count a `#[cfg(test)] mod { .. }`
    skip relies on, hiding whatever production code follows in the same file."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif two == "/*":
            depth, i = 1, i + 2
            while i < n and depth:
                if text[i:i + 2] == "/*":
                    depth, i = depth + 1, i + 2
                elif text[i:i + 2] == "*/":
                    depth, i = depth - 1, i + 2
                else:
                    if text[i] == "\n":
                        out.append("\n")
                    i += 1
        elif c == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    out.append("\n" if text[i + 1] == "\n" else "")
                    i += 1
                elif text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 1
            out.append('""')
        elif (c in "rb") and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")) and (m := RAW_RE.match(text, i)):
            end = text.find('"' + m.group(1), m.end())
            end = n if end < 0 else end + 1 + len(m.group(1))
            out.append("\n" * text.count("\n", i, end) + '""')
            i = end
        elif c == "'" and (m := CHAR_RE.match(text, i)):
            out.append("''")
            i = m.end()
        else:
            out.append(c)
            i += 1
    return "".join(out)


def production_lines(text: str) -> list[tuple[int, str]]:
    """(line number, code) outside `#[cfg(test)]` modules, strings, chars and comments removed."""
    out = []
    skip_depth = None  # brace depth at which the test module started
    pending_test = False
    depth = 0
    for n, code in enumerate(sanitize(text).splitlines(), 1):
        stripped = code.strip()
        if skip_depth is None and (stripped == "#[cfg(test)]" or re.match(r"#\[cfg\(all\(test\s*,", stripped)):
            pending_test = True
            continue
        opens, closes = code.count("{"), code.count("}")
        if pending_test and stripped:
            if stripped.startswith("#["):
                continue  # more attributes on the test module (`#[allow(..)]`)
            if re.match(r"(pub(\([^)]*\))?\s+)?mod\s+\w+\s*\{", stripped) and opens > closes:
                skip_depth = depth
            pending_test = False
        depth += opens - closes
        if skip_depth is not None:
            if depth <= skip_depth:
                skip_depth = None
            continue
        out.append((n, code))
    return out


def matches(lines: list[tuple[int, str]]) -> dict[str, list[int]]:
    """Line numbers of each pattern in the production lines of one file, with
    the code joined so a call split over lines still matches, and aliases
    (`use std::time::Instant as Clock`) resolved"""
    text = "\n".join(code for _, code in lines)
    starts = []
    pos = 0
    for _, code in lines:
        starts.append(pos)
        pos += len(code) + 1
    patterns = {name: [re.compile(rx)] for name, rx in PATTERNS.items()}
    for original, alias in ALIAS_RE.findall(text):
        name, rx = ALIASABLE[original]
        patterns[name].append(re.compile(rx.format(re.escape(alias))))
    found: dict[str, list[int]] = {}
    for name, rxs in patterns.items():
        hits = set()
        for rx in rxs:
            for m in rx.finditer(text):
                idx = max(i for i, st in enumerate(starts) if st <= m.start())
                hits.add(lines[idx][0])
        if hits:
            found[name] = sorted(hits)
    return found


def check(root: Path) -> tuple[list[str], int, int]:
    allowed_sites, problems = load_allowed(root)
    counts: dict[tuple[str, str], list[int]] = {}
    files = sorted((root / "src").rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        for name, hits in matches(production_lines(path.read_text(encoding="utf-8"))).items():
            counts[(rel, name)] = hits
    for key, lines in sorted(counts.items()):
        allowed = allowed_sites.get(key)
        where = ", ".join(f"{key[0]}:{n}" for n in lines)
        if allowed is None:
            problems.append(f"{key[1]} in production code: {where}")
        elif allowed[0] != len(lines):
            problems.append(f"{key[1]} in {key[0]}: {len(lines)} call sites, {ALLOW_FILE} allows {allowed[0]} ({where})")
    for key, (count, reason) in sorted(allowed_sites.items()):
        if not reason.strip():
            problems.append(f"{ALLOW_FILE} entry {key[1]} in {key[0]} has no reason")
        if key not in counts:
            problems.append(f"{ALLOW_FILE} entry {key[1]} in {key[0]} ({count}) matches nothing: remove it")
    return problems, len(files), len(PATTERNS)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    problems, files, patterns = check(ROOT)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if files == 0 or patterns == 0:
        print("error: compared nothing (no src file or no pattern)", file=sys.stderr)
        return 1
    print(f"entropy_guard: {files} files, {patterns} patterns, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
