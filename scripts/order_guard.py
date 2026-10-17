#!/usr/bin/env python3
"""No Law-bearing output depends on an unspecified order (ALICE-Law G13).

Reads the production code of src/ (strings, chars, comments and `#[cfg(test)]`
modules removed, by the same reader as scripts/entropy_guard.py) and reports
every use of a collection whose iteration order is unspecified (HashMap,
HashSet and their hasher-parametrised and third-party forms) and every source
of a combination order that depends on thread count or scheduling (rayon,
par_iter / par_chunks, thread::spawn / thread::scope, available_parallelism,
crossbeam) or on addresses (a pointer cast to an integer), and every normal
dependency in ORDER_DEPS (rayon, crossbeam, hashbrown, indexmap, ...). The
type-resolved half is clippy.toml (disallowed-types HashMap / HashSet and
their iterators, std::thread::Builder / JoinHandle / Scope; disallowed-methods
available_parallelism), which aliases and glob imports cannot evade.

Each hit must be listed in scripts/order_allowlist.toml as
`"<file>::<pattern>" = { count = N, lines = ["<sha256>", ...], reason = "..." }`:
the exact number of lines with that pattern in that file, the SHA-256 of each
of those lines (the source line with surrounding whitespace removed, in file
order) and why the order cannot reach an output (membership or keyed lookup
only, a sorted copy before output, ...). The hashes pin what was reviewed:
rewriting an allowed line (an ordered collect turned into appends in
completion order on the same line) changes its hash and fails until the list
is updated next to the reason. An unlisted hit, a count that differs, a line
whose hash differs, a stale entry, a reason shorter than MIN_REASON, or
reading no file fails. A Cargo.toml dependency entry has no lines.

usage: scripts/order_guard.py [--root DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entropy_guard import production_lines  # noqa: E402

MIN_REASON = 12
PATTERNS = {
    "HashMap": r"\b(?:HashMap|FxHashMap|AHashMap|IndexMap|DashMap)\b|\bhash_map\s*::",
    "HashSet": r"\b(?:HashSet|FxHashSet|AHashSet|IndexSet|DashSet)\b|\bhash_set\s*::",
    # every rayon parallel method (par_iter, par_windows, par_chunks_exact, par_sort_*,
    # par_drain, par_split, ...) and every rayon trait (ParallelIterator, ParallelSlice,
    # IntoParallelRefIterator, IndexedParallelIterator, FromParallelIterator, ...), called
    # with a dot or through the trait (UFCS): what an allowed `use rayon::prelude::*` brings in
    "parallel": r"\brayon\b|\bpar_\w+|\binto_par_\w+|\b\w*Parallel[A-Z]\w*|\bcrossbeam\b",
    # any path through std::thread, a braced or glob import of it, or a glob of std
    # (sleep / yield_now / current / park only wait or name the calling thread)
    "threads": r"\bthread\s*::\s*(?!(?:sleep|yield_now|current|park|park_timeout)\b)[\w{*]"
               r"|\buse\s+(?:::)?std\s*::\s*\{[^}]*\bthread\b|\buse\s+(?:::)?std\s*::\s*\*"
               r"|\bstd\s*::\s*thread\b(?!\s*::\s*(?:sleep|yield_now|current|park|park_timeout)\b)"
               r"|\bavailable_parallelism\b",
    "address order": r"\bas\s+\*\s*(?:const|mut)\s+\w+\s+as\s+usize\b|\.as_ptr\(\)\s+as\s+usize\b|\baddr\(\)",
}


def line_hash(line: str) -> str:
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()


def scan(root: Path) -> tuple[dict[str, int], int, dict[str, list[str]]]:
    """Hit counts per `<file>::<pattern>`, files read, and the hit source lines (stripped)."""
    counts: dict[str, int] = {}
    texts: dict[str, list[str]] = {}
    files = sorted((root / "src").rglob("*.rs")) if (root / "src").exists() else []
    rxs = {name: re.compile(rx) for name, rx in PATTERNS.items()}
    for path in files:
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        raw = text.splitlines()
        for n, code in production_lines(text):
            for name, rx in rxs.items():
                if rx.search(code):
                    key = f"{rel}::{name}"
                    counts[key] = counts.get(key, 0) + 1
                    texts.setdefault(key, []).append(raw[n - 1].strip())
    return counts, len(files), texts


# dependencies whose presence means a parallel combination or an unspecified
# iteration order is available to the library
ORDER_DEPS = {"rayon", "rayon-core", "crossbeam", "crossbeam-utils", "crossbeam-channel", "crossbeam-deque",
              "hashbrown", "indexmap", "dashmap", "ahash", "fxhash", "rustc-hash"}


def order_deps(root: Path) -> list[str]:
    """Names in ORDER_DEPS among the normal dependencies of Cargo.toml (also
    target-specific ones and renamed ones by their package name)."""
    cargo = root / "Cargo.toml"
    if not cargo.exists():
        return []
    doc = tomllib.loads(cargo.read_text(encoding="utf-8"))
    tables = [doc.get("dependencies", {})]
    tables += [t.get("dependencies", {}) for t in doc.get("target", {}).values()]
    found = []
    for deps in tables:
        for key, spec in deps.items():
            name = spec.get("package", key) if isinstance(spec, dict) else key
            if name in ORDER_DEPS:
                found.append(name)
    return sorted(found)


def check(root: Path) -> tuple[list[str], int, int]:
    allow_path = root / "scripts" / "order_allowlist.toml"
    allowed = tomllib.loads(allow_path.read_text(encoding="utf-8")).get("allow", {}) if allow_path.exists() else {}
    counts, files, texts = scan(root)
    problems = []
    if files == 0:
        return ["no src file read (compared nothing)"], 0, 0
    for key, n in sorted(counts.items()):
        entry = allowed.get(key)
        if entry is None:
            problems.append(f"{key}: {n} line(s) in production code, not in scripts/order_allowlist.toml")
        elif entry.get("count") != n:
            problems.append(f"{key}: {n} line(s), scripts/order_allowlist.toml says {entry.get('count')}")
        else:
            got = [line_hash(t) for t in texts[key]]
            want = entry.get("lines")
            if not isinstance(want, list) or want != got:
                shown = "; ".join(f"{h} {t}" for h, t in zip(got, texts[key]))
                problems.append(f"{key}: the allowed lines are not the reviewed ones "
                                f"(lines in scripts/order_allowlist.toml differ; now: {shown})")
    for dep in order_deps(root):
        key = f"Cargo.toml::{dep}"
        counts[key] = 1
        if key not in allowed:
            problems.append(f"{key}: a dependency that provides parallel combination or unspecified order, "
                            f"not in scripts/order_allowlist.toml")
    for key, entry in sorted(allowed.items()):
        if key not in counts:
            problems.append(f"{key}: listed but not found (stale entry)")
        if len(str(entry.get("reason", "")).strip()) < MIN_REASON:
            problems.append(f"{key}: reason shorter than {MIN_REASON} characters")
    return problems, files, sum(counts.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    root = Path(ap.parse_args(argv).root)
    problems, files, hits = check(root)
    for p in problems:
        print(f"order_guard: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"order_guard: {files} files, {len(PATTERNS)} patterns, {hits} allowed line(s), 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
