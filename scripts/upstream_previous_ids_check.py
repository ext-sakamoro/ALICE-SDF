#!/usr/bin/env python3
"""upstream_previous_ids_check.py — another crate's PREVIOUS_*_SEMANTICS_IDS is never read by position

A crate's list of previous semantics identifiers is the crate's own data: on a major fold it goes back to `&[]`,
which is a correct state. A downstream crate that reads `upstream::PREVIOUS_X_SEMANTICS_IDS[0]` (or `.first()`,
`.get(0)`, ...) then panics or loses its value the moment it resolves the new upstream. Read the list as a whole:
iterate it, `contains()`, `len()`, `is_empty()`, pass it on; to name one identifier, use the upstream's named
constant.

This is a lint on a coding pattern: another crate's list may only be used whole (contains / len / is_empty, an
iterator into a consumer of all of it, a for-loop over it, the list passed on as a whole, a binding or a whole copy
of it, which is then followed); every other use refuses. It catches the forms it can parse; it is not a proof that no
code depends on the list's length.

Another crate: the first path segment is not one of the repository's own crates (every [package] name of a
Cargo.toml in the tree, `-` read as `_`; `crate`, `self`, `super`). A crate's own list is checked by its own CI.
Comments and string literals are left out; doc comments are kept (a doc example runs as a doctest).

usage: upstream_previous_ids_check.py [--repo DIR]
exit 0: no finding (the count of files scanned is printed), 1: a finding (path:line and the fix), 2: no .rs file
scanned (nothing compared)
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LIST_NAME = r"PREVIOUS_[A-Z0-9_]*SEMANTICS_IDS"
PATH_LIST_RE = re.compile(r"(?<![A-Za-z0-9_])((?:::)?(?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)+)(" + LIST_NAME + r")\b")
FIX = ("use the list as a whole (iterate it, contains(), len(), is_empty()) or name the identifier you mean with the "
       "upstream's constant: the upstream may empty the list on a major version")


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    text: str


def own_modules(texts: dict[str, str]) -> set[str]:
    """the module names the repository declares (`mod ids;`, `pub mod law {`): a path through one of them is the
    repository's own (a `use ids::X` path may start at a module of the same crate)"""
    return {m.group(1) for t in texts.values() for m in re.finditer(r"\bmod\s+([A-Za-z_][A-Za-z0-9_]*)\s*[;{]", t)}


def own_crates(root: Path) -> set[str]:
    """the [package] names of every tracked Cargo.toml, `-` read as `_`"""
    out = set()
    files = subprocess.run(["git", "-C", str(root), "ls-files", "--", "Cargo.toml", "*/Cargo.toml"],
                           capture_output=True, text=True).stdout.split()
    for f in files:
        text = (root / f).read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^\[package\][^\[]*?^name\s*=\s*\"([^\"]+)\"", text, re.M | re.S)
        if m:
            out.add(m.group(1).replace("-", "_"))
    return out


def strip(text: str) -> str:
    """the text with `//` and `/* */` comments and string / char literals blanked (newlines kept); `///` and `//!`
    doc comments are kept, without their marker (a doc example is code)"""
    out, i, n = [], 0, len(text)
    fence = False
    while i < n:
        if text.startswith("///", i) or text.startswith("//!", i):
            # a doc comment: only the code of a ``` block (a doctest) is code; prose that names the list is not
            j = text.find("\n", i)
            j = n if j < 0 else j
            body = text[i + 3:j]
            if body.strip().startswith("```"):
                fence = not fence
                out.append(" " * (j - i))
            elif fence:
                out.append("   " + body)
            else:
                out.append(" " * (j - i))
            i = j
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        if text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
            continue
        m = re.match(r'b?r(#*)"', text[i:])
        if m:
            end = text.find('"' + m.group(1), i + m.end())
            j = n if end < 0 else end + 1 + len(m.group(1))
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
            continue
        if text[i] == '"' or text.startswith('b"', i):
            j = i + (2 if text[i] == "b" else 1)
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(re.sub(r"[^\n]", " ", text[i:j + 1]))
            i = j + 1
            continue
        if text[i] == "'" and re.match(r"'(\\.|[^\\'])'", text[i:]):
            m = re.match(r"'(\\.|[^\\'])'", text[i:])
            out.append(" " * m.end())
            i += m.end()
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _foreign(prefix: str, own: set[str]) -> bool:
    first = prefix.lstrip(":").split("::")[0].strip()
    return bool(first) and first not in own and first not in ("crate", "self", "super", "Self")


def reexports(texts: dict[str, str], own: set[str]) -> set[str]:
    """the names under which the repository re-exports another crate's list (`pub use x::PREVIOUS_.. as P`): a path
    through the own crate that ends in one of them still reaches the other crate's list"""
    out = set()
    for text in texts.values():
        code = strip(text)
        for m in re.finditer(r"\bpub(?:\s*\([^)]*\))?\s+use\s+([^;]+);", code):
            body = re.sub(r"\s+", "", m.group(1))
            if not _foreign(body.lstrip(":").split("::")[0] + "::", own):
                continue
            for lm in re.finditer(r"(" + LIST_NAME + r")(?:as([A-Za-z_][A-Za-z0-9_]*))?", body):
                out.add(lm.group(2) or lm.group(1))
    return out


def _skip_parens(t: str, i: int) -> int:
    """the index after the parenthesised group that starts at t[i] == "(" (-1 if unbalanced)"""
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "(":
            depth += 1
        elif t[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


ADAPTERS = {"copied", "cloned", "map", "filter", "filter_map", "enumerate", "zip", "chain", "flat_map", "inspect",
            "rev", "by_ref"}
CONSUMERS = {"any", "all", "count", "collect", "for_each", "find", "position", "sum", "max", "min", "fold", "eq",
             "cmp", "partial_cmp"}
WHOLE_METHODS = {"contains", "len", "is_empty"}
BINDING = {"to_vec", "to_owned", "clone"}  # a copy of the whole list: allowed, and the copy is followed


def classify(after: str, before: str) -> str | None:
    """None when the list is used whole; else why it is refused: an accessor's name, `pattern`, or `use`

    whole: `.contains(..)` / `.len()` / `.is_empty()`; `.iter()` / `.into_iter()` through adapters (copied, cloned,
    map, filter, enumerate, zip, rev ...) into a consumer of all of it (any, all, count, collect, for_each, find,
    position, sum, fold ...); a for-loop over all of it; the list passed as a whole (followed by `)` `,` `]` `;` or
    `}`) -- a `let` binding or a copy (`to_vec` / `collect` into a `let`) is followed by the caller"""
    if re.search(r"\blet\s*\[[^\]=]*\]\s*=\s*[&*\s]*$", before) or (
            re.search(r"\bmatch\s+[&*\s]*$", before) and re.match(r"^\s*\{", after)):
        return "pattern"
    t = after
    m = re.match(r"^\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()", t)
    if m is None:
        if re.match(r"^\s*[),\];}]", t) or (re.search(r"\bfor\s+[^;{]*\bin\s+[&*\s]*$", before)
                                           and re.match(r"^\s*\{", t)):
            return None
        if re.match(r"^\s*\[", t):
            return "index"
        return "use"
    name = m.group(1)
    if name in WHOLE_METHODS or name in BINDING:
        return None
    if name not in ("iter", "into_iter"):
        return {"get": "get", "first": "first", "split_first": "first", "last": "last", "split_last": "last",
                "split_at": "split", "get_unchecked": "split"}.get(name, name)
    # an iterator: adapters, then a consumer of all of it, or handed on whole
    pos = m.end()
    while True:
        end = _skip_parens(t, pos)
        if end < 0:
            return "use"
        rest = t[end:]
        nm = re.match(r"^\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()", rest)
        if nm is None:
            if re.match(r"^\s*[),\];}]", rest) or (re.search(r"\bfor\s+[^;{]*\bin\s+[&*\s]*$", before)
                                                  and re.match(r"^\s*\{", rest)):
                return None
            return "use"
        step = nm.group(1)
        if step in CONSUMERS:
            return None
        if step not in ADAPTERS:
            return step  # next / nth / last / take / skip / step_by / peekable ...: by position
        t, pos = rest, nm.end()


def scan_text(path: str, text: str, own: set[str], reexported: set[str] = frozenset()) -> list[Finding]:
    """every use of another crate's list in one file that is not a use of the whole list (`reexported`: names the
    repository re-exports from another crate, followed through own paths)"""
    code = strip(text)
    # `use` statements and const / static re-definitions name the list, they do not use it
    masked = re.sub(r"\b(?:pub(?:\s*\([^)]*\))?\s+)?use\s+[^;]+;", lambda m: re.sub(r"[^\n]", " ", m.group(0)), code)
    names: set[str] = set()   # local names that stand for another crate's list
    defined = set(re.findall(r"\b(?:const|static)\s+(" + LIST_NAME + r")\s*:", code))
    for m in re.finditer(r"\buse\s+([^;]+);", code):
        body = re.sub(r"\s+", "", m.group(1))
        root = body.lstrip(":").split("::")[0]
        if not _foreign(root + "::", own):
            continue
        for lm in re.finditer(r"(" + LIST_NAME + r")(?:as([A-Za-z_][A-Za-z0-9_]*))?", body):
            names.add(lm.group(2) or lm.group(1))
        if body.endswith("::*") or "::*," in body or "{*" in body:
            names |= set(re.findall(r"\b(" + LIST_NAME + r")\b", code)) - defined
    path_re = re.compile(r"(?<![A-Za-z0-9_])((?:::)?(?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)+)(" + LIST_NAME
                         + ("|" + "|".join(map(re.escape, sorted(reexported))) if reexported else "") + r")\b")

    def foreign_ref(expr: str) -> bool:
        pm = path_re.fullmatch(expr.strip())
        return bool(pm and (_foreign(pm.group(1), own) or pm.group(2) in reexported)) or expr.strip() in names

    # bindings of the list or of a whole copy of it: `let p = x::LIST;`, `let v = x::LIST.to_vec();`,
    # `let v: Vec<_> = x::LIST.iter().copied().collect();`, `const P: &[..] = x::LIST;` (repeated: a binding of a
    # binding)
    for _ in range(3):
        for m in re.finditer(r"\b(?:let\s+(?:mut\s+)?|(?:pub(?:\s*\([^)]*\))?\s+)?(?:const|static)\s+)"
                             r"([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+?)?=\s*[&*\s]*([^;]+);", masked):
            rhs = re.sub(r"\s+", "", m.group(2))
            base = re.split(r"\.(?:to_vec|to_owned|clone|iter|into_iter)\(", rhs)[0]
            if foreign_ref(base) and m.group(1) not in names:
                names.add(m.group(1))
    out = []
    lines_at = [0] + [i + 1 for i, ch in enumerate(masked) if ch == "\n"]

    def line_of(pos: int) -> int:
        import bisect
        return bisect.bisect_right(lines_at, pos)

    def check(start: int, end: int, label: str) -> None:
        before = masked[max(0, start - 120):start]
        if re.search(r"\b(?:let\s+(?:mut\s+)?|const\s+|static\s+|as\s+)$", before):
            return  # the name being bound, not a use
        kind = classify(masked[end:end + 400], before)
        if kind:
            out.append(Finding(path, line_of(start), kind, label))

    for m in path_re.finditer(masked):
        if _foreign(m.group(1), own) or m.group(2) in reexported:
            check(m.start(), m.end(), (m.group(1) + m.group(2)).replace(" ", ""))
    for name in sorted(names):
        for m in re.finditer(r"(?<![A-Za-z0-9_:.])" + re.escape(name) + r"\b(?!\s*::)", masked):
            check(m.start(), m.end(), name)
    return sorted(out, key=lambda f: (f.line, f.kind))


def render(f: Finding) -> str:
    return f"{f.path}:{f.line}: {f.text} read by position ({f.kind}): {FIX}"


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    root = Path(a.repo).resolve()
    own = own_crates(root)
    files = subprocess.run(["git", "-C", str(root), "ls-files", "--", "*.rs"], capture_output=True,
                           text=True).stdout.split()
    if not files:
        print("upstream-previous-ids: 0 files scanned (no tracked .rs): nothing compared", file=sys.stderr)
        return 2
    texts = {f: (root / f).read_text(encoding="utf-8", errors="replace") for f in files if (root / f).is_file()}
    own = own | own_modules({f: strip(t) for f, t in texts.items()})
    reexported = reexports(texts, own)
    found = []
    for f, text in texts.items():
        found += scan_text(f, text, own, reexported)
    for f in found:
        print(f"upstream-previous-ids: {render(f)}", file=sys.stderr)
    print(f"upstream-previous-ids: {len(files)} file{'s' if len(files) != 1 else ''} scanned, {len(found)} "
          f"finding{'s' if len(found) != 1 else ''}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
