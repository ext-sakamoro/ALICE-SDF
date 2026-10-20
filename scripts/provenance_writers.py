#!/usr/bin/env python3
"""Every public writer of a durable format stores the semantics identifier (ALICE-Law G8).

Candidates come from two scans, and every candidate must be classified:

  * the text scan (every OS, no index): a `pub fn` under src/ (any fn under src/bin/, the command-line
    tool, whose fns are private), outside `#[cfg(test)]` modules, that has a writer's name (`export*`, `save*`, `write*`, `bake_to*`, `encode*`, `serialize*`,
    `persist*`, `dump*`, `to_bytes`, `to_vec`), a write sink (a `Write`, `BufWriter` or `File`
    parameter; `File::create` / `create_new` / `options`, `fs::write` / `copy` / `rename` or
    `OpenOptions` in the body), a stored form as its return type (`Vec<u8>`, `[u8; N]`, `&[u8]`,
    `Box<[u8]>`, or `String` for a `to_*` / `as_*` fn, alone or in a Result / Option), or the
    `_with_provenance` / `_with_stamp` suffix, which also makes the fn it extends a candidate
    whatever its name;
  * with `--index DIR` (the SCIP index of scripts/scip_index.sh, built from this commit): every pub
    item and every trait-impl method of src/, and every fn of src/bin/, whose call graph
    (scripts/scip_reach.py) reaches a write sink of std (file creation and writes of std::fs, any `Write` method). This sees what text
    cannot: a trait-impl method has no `pub`, a writer may have any name.

Two fns of one file with one name (methods of two impls) share the key `<file>::<fn>`. The
`*_with_provenance` / `*_with_stamp` variants are counted with the writer they extend: each must
extend a writer of the same file and be called in a test under tests/. Each candidate must be
exactly one of

  - covered: named as `"<file>::<fn>"` in a table of a test under tests/, which writes it, reads the
    identifier back and checks a mismatch,
  - pending: listed in scripts/provenance-pending.txt (writers that do not store it yet), or
  - not a writer: listed in scripts/provenance-not-writers.txt with its reason. A reason
    `delegates to <fn>` (`delegates to <a> (and to <b> for ...)` for a fn with branches) must name
    covered or pending writers, each after a `to`, and every body of the key must call each of them (through a `use ... as` alias is enough); any other reason has at least five words. A line
    for an entry only the call graph finds is checked, without an index, to name a fn of that file
    that is no plain `pub fn`.

The pending file also holds `expected-writers <N>`, the number of writers found (candidates that
are not "not a writer"): a new writer, or one removed, changes the count and fails until the
number is updated on purpose.

Fails when: no writer is found; the count differs from expected-writers; a candidate is in none of
the three lists or in two; a pending, covered or not-a-writer line names no candidate; a
`delegates to` reason names no writer or a body does not call it; any other reason is short; with
--index, an index of another commit or a dirty tree, or one that holds no write sink.

What the scans do not see (known residuals, so a pass is not a proof):
  * a text producer with a neutral name (`pub fn render() -> String`) that reaches no write sink:
    `String` counts as a stored form only for a `to_*` / `as_*` fn
  * in text mode, a `pub fn` written on the line of its `impl` header (`impl X { pub fn f() ... }`);
    rustfmt never writes that form, and `cargo fmt --check` rejects it
  * in --index mode, the code of the bindings (src/python/, src/ffi/, src/godot/, src/wasm.rs): the
    call graph keeps their references as roots, not edges, so their fns are found by the text scan
    only (src/bin/ is the exception: each of its fns starts from what its body references)
  * in --index mode, what scripts/api_boundary_parsers.py lists for the call graph: calls through a
    function pointer or a stored closure, bodies a macro generates (a `write!` expands to no
    reference), and dynamic dispatch to an impl whose self type is never reached

usage: scripts/provenance_writers.py [ROOT] [--index DIR]
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIR = "src"
# the command-line tool: its fns are private, yet each writes a file the user keeps
BIN_DIR = "src/bin/"
TEST_FILE = "tests/test_provenance.rs"
PENDING_FILE = "scripts/provenance-pending.txt"
NOT_WRITERS_FILE = "scripts/provenance-not-writers.txt"

PUB_FN = re.compile(r"^\s*pub\s+(?:(?:const|async|unsafe|extern\s*(?:\"[^\"]*\")?)\s+)*fn\s+(\w+)", re.M)
ANY_FN = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|async|unsafe|extern\s*(?:\"[^\"]*\")?)\s+)*fn\s+(\w+)", re.M
)
# a fn definition in the index: `.../<name>().` (a free fn or a method)
FN_SYM = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(\)\.$")
WRITER_NAME = re.compile(r"^(?:(?:export|save|write|bake_to|encode|serialize|persist|dump)\w*|to_bytes|to_vec)$")
# a write sink in the signature (a `Write` parameter) or the body (a file created or written)
SINK_SIG = re.compile(
    r"\b(?:impl|dyn)\s+(?:std::)?(?:io::)?Write\b|\w+\s*:\s*(?:std::)?(?:io::)?Write\b"
    r"|:\s*&?\s*(?:mut\s+)?(?:std::)?(?:io::)?BufWriter\s*<|:\s*&?\s*(?:mut\s+)?(?:std::)?(?:fs::)?File\b"
)
SINK_BODY = re.compile(r"\bFile::(?:create|create_new|options)\b|\bfs::(?:write|copy|rename)\b|\bOpenOptions\b")
# a stored form as the return type: bytes (`Vec<u8>`, `[u8; N]`, `&[u8]`, `Box<[u8]>`), or text for a
# `to_*` / `as_*` fn, alone or in a Result / Option
_WRAP = r"^\s*(?:(?:(?:std::)?(?:io::)?Result|Option|PyResult|\w*Result)\s*<\s*)?"
RET_BYTES = re.compile(
    _WRAP + r"(?:Vec\s*<\s*u8\s*>|\[\s*u8\s*;[^\]]*\]|&\s*(?:'\w+\s+)?(?:mut\s+)?\[\s*u8\s*\]|Box\s*<\s*\[\s*u8\s*\]\s*>)"
)
RET_TEXT = re.compile(_WRAP + r"String\b")
# the write sinks of the call graph (--index), as `scripts/api_boundary_parsers.py` names symbols:
# file creation and writes of std::fs, and every `Write` method (a write into memory is followed
# too; such byte producers are candidates by their return type anyway)
SINK_SYM = re.compile(
    r"^std (?:fs/(?:write|copy|rename|hard_link)\(\)\.|fs/impl#\[File\](?:create|create_new|options)\(\)\."
    r"|fs/impl#\[OpenOptions\]open\(\)\.|io/(?:\S*\[Write\]|Write#)(?:write|write_all|write_fmt|write_vectored|flush)\(\)\.)"
)
TEST_MODULE = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*(?:#\s*\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{")
COVERED = re.compile(r'\(\s*"(src/[\w/]+\.rs::\w+)"\s*,')


def blank_comments_and_strings(text: str) -> str:
    """The text with comments, string and char literals replaced by spaces (newlines kept)"""
    out = []
    i, n = 0, len(text)

    def blank(a: int, b: int) -> None:
        out.append("".join(c if c == "\n" else " " for c in text[a:b]))

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif (m := re.match(r'b?r(#*)"', text[i:])) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            close = '"' + m.group(1)
            j = text.find(close, i + m.end())
            j = n if j < 0 else j + len(close)
            blank(i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        elif c == "'" and (m := re.match(r"'(?:\\.[^']*|[^'\\\n])'", text[i:])):
            blank(i, i + m.end())
            i += m.end()
        else:
            out.append(c)
            i += 1
    return "".join(out)


def outside_test_modules(text: str) -> str:
    """The source with every `#[cfg(test)] mod name { ... }` block removed (comments and strings blanked)"""
    clean = blank_comments_and_strings(text)
    while (m := TEST_MODULE.search(clean)) is not None:
        depth, j = 1, m.end()
        while j < len(clean) and depth:
            depth += {"{": 1, "}": -1}.get(clean[j], 0)
            j += 1
        clean = clean[: m.start()] + clean[j:]
    return clean


def writer_files(root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(os.path.join(root, SRC_DIR)):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        # keys use "/" on every OS: they are compared with the names in the test and the lists
        out += [f"{rel_dir}/{f}" for f in files if f.endswith(".rs")]
    return sorted(out)


VARIANT = re.compile(r"_with_(?:provenance|stamp)$")


def signature_end(text: str, i: int) -> int | None:
    """The index of the `{` that opens the body of the fn whose signature starts at `i`, or None
    when a `;` ends it first; a `;` inside brackets (an array type `[u8; 32]`) does not"""
    depth = 0
    while i < len(text):
        c = text[i]
        if c in "([<":
            depth += 1
        elif c in ")]>" and depth and not (c == ">" and text[i - 1] == "-"):
            depth -= 1
        elif c == "{" and depth == 0:
            return i
        elif c == ";" and depth == 0:
            return None
        i += 1
    return None


def fn_defs(text: str, pattern: re.Pattern[str]) -> dict[str, list[tuple[str, str]]]:
    """{name: [(signature, body), ...]} of every fn `pattern` finds in `text` that has a body"""
    fns: dict[str, list[tuple[str, str]]] = {}
    for m in pattern.finditer(text):
        brace = signature_end(text, m.end())
        if brace is None:
            continue  # a declaration without a body (trait or extern item)
        depth, end = 1, brace + 1
        while end < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
        fns.setdefault(m.group(1), []).append((text[m.end():brace], text[brace:end]))
    return fns


def bodies(root: str, key: str) -> list[str]:
    """The bodies of every fn named by `key` in its file, whatever its visibility (an entry only
    the call graph finds has no body in the text scan)"""
    rel, _, name = key.partition("::")
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        text = outside_test_modules(fh.read())
    return [body for _, body in fn_defs(text, ANY_FN).get(name, [])]


def candidates(root: str) -> dict[str, list[str]]:
    """{"<file>::<fn>": [body, ...]} of every public fn with a writer's name, a write sink, a stored
    form as its return type, or the `_with_provenance` / `_with_stamp` suffix (and the fn that
    suffix extends); two fns of one file with one name (methods of two impls) share a key"""
    out: dict[str, list[str]] = {}
    for rel in writer_files(root):
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            text = outside_test_modules(fh.read())
        fns = fn_defs(text, ANY_FN if rel.startswith(BIN_DIR) else PUB_FN)
        picked = set()
        for name, defs in fns.items():
            for sig, body in defs:
                ret = sig.split("->", 1)[1] if "->" in sig else ""
                if (
                    WRITER_NAME.match(name)
                    or VARIANT.search(name)
                    or SINK_SIG.search(sig)
                    or SINK_BODY.search(body)
                    or RET_BYTES.match(ret)
                    or (RET_TEXT.match(ret) and re.match(r"(?:to|as)_", name))
                ):
                    picked.add(name)
        # a variant counts its base, whatever the base's name
        picked |= {VARIANT.sub("", n) for n in picked if VARIANT.search(n) and VARIANT.sub("", n) in fns}
        for name in sorted(picked):
            out[f"{rel}::{name}"] = [body for _, body in fns[name]]
    return out


def reach_hits(entries: dict[str, set[str]], reach, sinks: set[str]) -> dict[str, set[str]]:
    """Pure core of the --index mode: {entry key: the sinks it reaches} for each entry whose call
    graph reaches one (`reach(set of symbols)` returns every symbol reachable from them)"""
    hits = {}
    for key, syms in entries.items():
        got = reach(set(syms)) & sinks
        if got:
            hits[key] = got
    return hits


def bin_fn_key(path: str, sym: str) -> str | None:
    """`src/bin/x.rs::name` for a fn defined in the command-line tool, else None"""
    f = FN_SYM.search(sym) if path.startswith(BIN_DIR) else None
    return f"{path}::{f.group(1)}" if f else None


def bin_entries(path: str, occ: list[dict]) -> dict[str, set[str]]:
    """{`src/bin/x.rs::name`: its symbol and every symbol referenced inside its body (the innermost
    enclosing fn), and those of the tool's fns it calls} for the fns of a document of the
    command-line tool; {} for any other document.
    `occ` holds SCIP occurrences: `symbol`, `roles` (bit 0 = definition), `range`, `enc` (the
    enclosing range of a definition), each `[line, char, (line,) char]`"""
    if not path.startswith(BIN_DIR):
        return {}

    def span(r: list[int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if len(r) == 4:
            return (r[0], r[1]), (r[2], r[3])
        return ((r[0], r[1]), (r[0], r[2])) if len(r) == 3 else None

    defs = []  # (start, end, symbol, key)
    for o in occ:
        sym = o["symbol"]
        if o["roles"] & 1 and sym and (key := bin_fn_key(path, sym)) and (enc := span(o.get("enc") or [])):
            defs.append((enc[0], enc[1], sym, key))
    out = {key: {sym} for _, _, sym, key in defs}
    for o in occ:
        sym = o["symbol"]
        at = span(o["range"])
        if o["roles"] & 1 or not sym or sym.startswith("local ") or at is None:
            continue
        inside = [d for d in defs if d[0] <= at[0] <= d[1]]
        if inside:
            out[max(inside, key=lambda d: d[0])[3]].add(sym)
    # the call graph holds no edge out of a fn of the tool, so a call to another one of its fns
    # takes that fn's references along (to a fixpoint)
    by_sym = {sym: key for _, _, sym, key in defs}
    changed = True
    while changed:
        changed = False
        for key, syms in out.items():
            more = set().union(*(out[by_sym[x]] for x in syms if x in by_sym)) - syms
            if more:
                syms |= more
                changed = True
    return out


def legacy(key: str) -> str:
    """`src/x.rs::Type::name` -> `src/x.rs::name`, the key form of the lists and the test table"""
    rel, _, rest = key.partition("::")
    return f"{rel}::{rest.rsplit('::', 1)[-1]}"


def reach_candidates(root: str, index: str) -> tuple[dict[str, set[str]], str | None, str | None]:
    """({legacy key: sinks} of every pub item and trait-impl method of src/ whose call graph
    reaches a write sink, the index commit, a problem). The index must be of HEAD and a clean tree."""
    sys.path.insert(0, os.path.join(root, "scripts"))
    import api_boundary_parsers as abp  # noqa: E402
    import scip_reach  # noqa: E402
    from pathlib import Path

    problem, commit = abp.index_commit(Path(root), Path(index))
    if problem:
        return {}, None, problem
    scips = [Path(index) / "native.scip", Path(index) / "wasm.scip"]
    for sp in scips:
        if not sp.is_file() or sp.stat().st_size == 0:
            return {}, None, f"{sp} is missing or empty"
    an = scip_reach.analyze(Path(root), scips, keep_graph=True)
    entries = {k: set(v) for k, v in an.items.items()}
    every: set[str] = set()
    for sp in scips:
        for doc in scip_reach.load_scip(sp):
            for o in doc["occ"]:
                sym = o["symbol"]
                if not sym or sym.startswith("local "):
                    continue
                every.add(sym)
                # a trait-impl method has no `pub` and is no item of the analysis, yet a caller
                # of the trait reaches it: an entry of its own
                m = scip_reach.IMPL_METHOD_RE.search(sym)
                if m and o["roles"] & 1 and doc["path"].startswith("src/"):
                    entries.setdefault(f"{doc['path']}::{m.group(1)}::{m.group(3)}", set()).add(sym)
            # a fn of the command-line tool is private and no item of the analysis; src/bin/ is one
            # of its roots, so its references are roots, not edges: each fn starts from what its
            # own body references
            for key, syms in bin_entries(doc["path"], doc["occ"]).items():
                entries.setdefault(key, set()).update(syms)
    sinks = {sym for sym in every if (sh := abp.short(sym)) and SINK_SYM.match(sh)}
    if not sinks:
        return {}, commit, "the index holds no write sink (compared nothing)"
    out: dict[str, set[str]] = {}
    for key, got in reach_hits(entries, an.reach, sinks).items():
        out.setdefault(legacy(key), set()).update(abp.short(g) for g in got)
    return out, commit, None


def variants(root: str) -> list[tuple[str, str]]:
    """(file, name) of every `*_with_provenance` / `*_with_stamp` writer variant"""
    return [tuple(k.split("::", 1)) for k in candidates(root) if VARIANT.search(k)]


def not_writers(root: str) -> list[tuple[str, str]]:
    """(key, reason) of scripts/provenance-not-writers.txt"""
    path = os.path.join(root, NOT_WRITERS_FILE)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, reason = line.partition(" ")
            out.append((key, reason.strip()))
    return out


def writers(root: str, cand: dict[str, list[str]] | None = None) -> list[str]:
    skip = {k for k, _ in not_writers(root)}
    return [k for k in (candidates(root) if cand is None else cand) if not VARIANT.search(k) and k not in skip]


def test_texts(root: str) -> dict[str, str]:
    """{path: text} of every integration test (tests/*.rs): a writer's round-trip table and the
    calls of its variants may live in the test file of its format"""
    tdir = os.path.join(root, "tests")
    if not os.path.isdir(tdir):
        return {}
    out = {}
    for f in sorted(os.listdir(tdir)):
        if f.endswith(".rs"):
            with open(os.path.join(tdir, f), encoding="utf-8") as fh:
                out[f"tests/{f}"] = fh.read()
    return out


def covered(root: str) -> set[str]:
    return {k for text in test_texts(root).values() for k in COVERED.findall(text)}


def pending(root: str) -> tuple[int | None, list[str]]:
    expected, lines = None, []
    with open(os.path.join(root, PENDING_FILE), encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("expected-writers "):
                expected = int(line.split()[1])
            else:
                lines.append(line)
    return expected, lines


def fn_exists(root: str, key: str) -> bool:
    """Whether the file of `key` defines a fn of its name that is no plain `pub fn` (a trait-impl
    method or a `pub(crate)` / private helper, which only the call graph makes a candidate); a
    `pub fn` the text scan does not pick is not a candidate, so its line is stale"""
    rel, _, name = key.partition("::")
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as fh:
        text = outside_test_modules(fh.read())
    if any(m.group(1) == name for m in PUB_FN.finditer(text)):
        return False
    return re.search(r"\bfn\s+" + re.escape(name) + r"\b", text) is not None


def calls(body: str, target: str) -> bool:
    """Whether `body` calls `target`, directly or through a `use ... target as alias` in the body"""
    names = [target] + re.findall(r"\b" + re.escape(target) + r"\s+as\s+(\w+)", body)
    return any(re.search(r"\b" + re.escape(n) + r"\s*\(", body) for n in names)


def check(root: str, reached: dict[str, set[str]] | None = None) -> tuple[list[str], int]:
    cand = candidates(root)
    # with an index, an entry the call graph shows writing is a candidate whatever its text says
    for key in reached or {}:
        cand.setdefault(key, [])
    found = writers(root, cand)
    errors = []
    if not found:
        return ["no writer found (compared nothing)"], 0
    expected, todo = pending(root)
    cov = covered(root)
    nots = not_writers(root)
    have = set(found)
    if expected is None:
        errors.append(f"{PENDING_FILE}: no `expected-writers <N>` line")
    elif expected != len(found):
        errors.append(
            f"{len(found)} writers found, {PENDING_FILE} expects {expected}: a writer was added or "
            "removed; give a new one a round-trip entry in tests/test_provenance.rs, then update the number"
        )
    dup = sorted({t for t in todo if todo.count(t) > 1})
    errors += [f"{PENDING_FILE}: listed twice: {t}" for t in dup]
    keys = [k for k, _ in nots]
    errors += [f"{NOT_WRITERS_FILE}: listed twice: {k}" for k in sorted({k for k in keys if keys.count(k) > 1})]
    for w in sorted(have):
        if w not in cov and w not in todo:
            errors.append(f"{w}: neither covered by a table in tests/ nor pending nor in {NOT_WRITERS_FILE}")
    for t in sorted(set(todo)):
        if t in cov:
            errors.append(f"{PENDING_FILE}: {t} is covered by a table in tests/; remove the line")
        if t not in have:
            errors.append(f"{PENDING_FILE}: {t} is not a writer")
    for c in sorted(cov):
        if c not in have:
            errors.append(f"tests/: {c} is not a writer")
    writer_names = {w.split("::", 1)[1] for w in have}
    for key, reason in nots:
        # without an index, a line for an entry only the call graph finds (a trait-impl method, a
        # pub(crate) helper) is checked to name a fn of that file; with one, it must be a candidate
        known = key in cand or (reached is None and fn_exists(root, key))
        if not known or VARIANT.search(key):
            errors.append(f"{NOT_WRITERS_FILE}: {key} is not a candidate (remove the line)")
            continue
        if key in cov or key in todo:
            errors.append(f"{NOT_WRITERS_FILE}: {key} is also covered or pending")
        if reason.startswith("delegates to "):
            # every fn the reason names after a `to` (`delegates to a (and to b for ...)`) is checked
            bs = cand.get(key) or bodies(root, key)
            for target in re.findall(r"\bto\s+(\w+)", reason):
                if target not in writer_names:
                    errors.append(f"{NOT_WRITERS_FILE}: {key} delegates to `{target}`, which is no covered or pending writer")
                elif not bs or not all(calls(b, target) for b in bs):
                    errors.append(f"{NOT_WRITERS_FILE}: {key} delegates to `{target}`, but its body does not call it")
        elif len(reason.split()) < 5:
            errors.append(f"{NOT_WRITERS_FILE}: {key}: the reason needs at least five words")
    test_text = "\n".join(blank_comments_and_strings(t) for t in test_texts(root).values())
    for rel, name in (tuple(k.split("::", 1)) for k in cand if VARIANT.search(k)):
        base = f"{rel}::{VARIANT.sub('', name)}"
        if base not in have:
            errors.append(f"{rel}::{name}: extends no writer ({base} is not one)")
        if not re.search(r"\b" + re.escape(name) + r"\s*\(", test_text):
            errors.append(f"{rel}::{name}: not called in tests/")
    return errors, len(found)


def main(argv: list[str]) -> int:
    """`provenance_writers.py [ROOT] [--index DIR]`: without --index the text scan alone (every
    OS); with it, also the entries the SCIP index (scripts/scip_index.sh) shows reaching a write sink"""
    args = argv[1:]
    index = None
    if "--index" in args:
        i = args.index("--index")
        index = args[i + 1]
        args = args[:i] + args[i + 2:]
    root = args[0] if args else ROOT
    reached, commit = None, None
    if index is not None:
        reached, commit, problem = reach_candidates(root, index)
        if problem:
            print(f"provenance-writers: {problem}", file=sys.stderr)
            return 1
    errors, n = check(root, reached)
    if errors:
        for e in errors:
            print(f"provenance-writers: {e}", file=sys.stderr)
        return 1
    _, todo = pending(root)
    where = f"; index {commit}: {len(reached)} entries reach a write sink" if reached is not None else ""
    print(f"provenance-writers: ok ({n} writers: {n - len(todo)} covered, {len(todo)} pending; "
          f"{len(not_writers(root))} candidates not writers{where})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
