#!/usr/bin/env python3
"""No deterministic path of the crate draws OS entropy or reads the wall clock.

The CPU evaluators promise the same distance bits for the same tree and point
on every platform, and the bounds, meshes, heightmaps and fits built from them
inherit that promise only if nothing upstream of them depends on timing or on
OS randomness: either one would make the same tree evaluate, mesh or export
differently on a different run or a different machine.

Checked in every src/**/*.rs file, outside `#[cfg(test)]` modules, with the
code of a file joined (a call split over lines still matches), `use .. as`
aliases of `Instant` / `SystemTime` resolved, and a fully-qualified-self call
(`<std::time::Instant>::now()`) recognised the same as the plain form:

    getrandom  OsRng  thread_rng  rand::random  from_entropy(  try_from_entropy(
    SystemTime::now  Instant::now  UNIX_EPOCH  RandomState (any use, not only
    `::new`: `::default()` reaches the same OS-seeded hasher)  Uuid::new_v4

clippy's `disallowed-methods` in clippy.toml is the first layer (compiled,
`cargo clippy --all-targets -D warnings`); this script is the second, readable
without compiling and run on `src/` only (clippy's `--all-targets` also covers
tests/benches/examples, where a scoped `#[allow(clippy::disallowed_methods,
reason = "...")]` carries the same per-call-site accounting this script does).

Deliberately not scanned: `HashMap` / `HashSet` (their default hasher also
seeds from OS randomness, but that is tracked as its own gate -- iteration
order, not entropy in a value -- not here).

A definition (`fn from_entropy`) is not a call. ALLOWED lists the call sites
that stay, each with its reason and exact count: a count that drifts either
way fails, so a new call cannot hide behind an old entry and a removed one
must be removed here too. Scanning no file, or no pattern, fails.

    python3 scripts/entropy_guard.py
"""
from __future__ import annotations

import re
import sys
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
# (file, pattern) -> (count, reason)
ALLOWED: dict[tuple[str, str], tuple[int, str]] = {
    # The CLI binary's own stdout timing diagnostics (compile/mesh/export
    # elapsed ms, printed with `println!`): never stored, never fed back into
    # the SDF tree, mesh or export it measures, so no computed output depends
    # on them. This is the `alice-sdf` bin target, not the library surface.
    # The 14 logical call sites (cmd_print / cmd_bench / cmd_texture_fit) all
    # go through one local `diagnostic_clock()` wrapper (carrying the single
    # scoped `#[allow(clippy::disallowed_methods, reason = "...")]` clippy
    # needs, since clippy has no file-level scoping), so only one textual
    # `Instant::now()` call site remains for this script to find.
    ("src/bin/main.rs", "Instant::now"): (
        1, "CLI timing diagnostics only (14 call sites via one diagnostic_clock() wrapper), printed not stored, not part of library output",
    ),
}
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
        if skip_depth is None and stripped == "#[cfg(test)]":
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
    problems: list[str] = []
    counts: dict[tuple[str, str], list[int]] = {}
    files = sorted((root / "src").rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        for name, hits in matches(production_lines(path.read_text(encoding="utf-8"))).items():
            counts[(rel, name)] = hits
    for key, lines in sorted(counts.items()):
        allowed = ALLOWED.get(key)
        where = ", ".join(f"{key[0]}:{n}" for n in lines)
        if allowed is None:
            problems.append(f"{key[1]} in production code: {where}")
        elif allowed[0] != len(lines):
            problems.append(f"{key[1]} in {key[0]}: {len(lines)} call sites, ALLOWED says {allowed[0]} ({where})")
    for key, (count, reason) in sorted(ALLOWED.items()):
        if not reason.strip():
            problems.append(f"ALLOWED entry {key[1]} in {key[0]} has no reason")
        if key not in counts:
            problems.append(f"ALLOWED entry {key[1]} in {key[0]} ({count}) matches nothing: remove it")
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
