#!/usr/bin/env python3
"""The clippy configuration for the library code, and a check that it fires.

`clippy.toml` (repository root) forbids the platform libm for every target. The library
code additionally may not call the methods (and name the types) listed in
`scripts/clippy_lib_gate.toml`, for example `f32::max` / `f32::min`, which leave the sign
of an equal pair of zeros unspecified and differ between targets. Tests, examples and
benches keep the std functions, so the extra entries live in a second configuration,
`.github/clippy-lib/clippy.toml`, that CI and preflight pass to `cargo clippy` with
`CLIPPY_CONF_DIR`. That file is generated from the root one (the root lists plus the extra
entries, inserted at the top of `disallowed-methods` / `disallowed-types`), so the libm
list cannot drift between the two.

This script is the same file in every repository that carries it; what differs per
repository is in `scripts/clippy_lib_gate.toml`:

    header = "..."                    # comment block at the top of the generated file
    method-reason = "..."             # reason of a [[method]] without its own
    type-reason = "..."               # reason of a [[type]] without its own
    [[method]]  path = "f32::max"  reason = "..."    (reason optional)
    [[type]]    path = "std::collections::HashMap"  reason = "..."
    [[form]]    text = "a.max(b)"  path = "f32::max"  also = ["..."]
    [[clean]]   text = "later.duration_since(earlier)"

  * check   fails when `.github/clippy-lib/clippy.toml` differs from what the root file
            generates (`--write` regenerates it), when a root list holds two entries on
            one line, or when an extra entry is already in the root file
  * canary  runs clippy with that configuration on `scripts/clippy-lib-canary` (a crate
            that uses every forbidden form once) and judges each [[form]] by the path:
            the form's line must carry a disallowed-method / disallowed-type diagnostic
            naming that form's `path`. A diagnostic naming another path does not count
            for the form; every diagnostic must be the path of a form on its line or one
            of that form's `also` paths (a second diagnostic the same expression emits,
            e.g. the type of `RandomState::new()`), and a [[clean]] line must have none.
            It fails on a form without its diagnostic, on an unaccounted diagnostic, on a
            form text that does not occur exactly once, on no forms at all, when clippy
            exits 0, and when clippy reports a configured path that refers to nothing.
            Every library-only entry must be named by a form (clippy is silent about a
            method path that names no method, so only a form shows the entry fires). A
            judge that only asked "is there some diagnostic on this line" let a removed
            entry pass when another entry still fired on that line.

    python3 scripts/clippy_lib_gate.py            check + canary
    python3 scripts/clippy_lib_gate.py --check    check only
    python3 scripts/clippy_lib_gate.py --write    regenerate the file
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROOT_CONFIG = os.path.join(ROOT, "clippy.toml")
DATA = os.path.join(HERE, "clippy_lib_gate.toml")
LIB_DIR = os.path.join(ROOT, ".github", "clippy-lib")
LIB_CONFIG = os.path.join(LIB_DIR, "clippy.toml")
CANARY = os.path.join(HERE, "clippy-lib-canary")

DIAG = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):\d+: (?:warning|error): "
                  r"use of a disallowed (?P<kind>method|type) `(?P<path>[^`]+)`", re.M)
# clippy's warning for a configured path that names nothing: that entry forbids nothing
UNRESOLVED = re.compile(r"(?:warning|error): `(?P<path>[^`]+)` does not refer to (?:a reachable|an existing) \w+")
LISTS = ("disallowed-methods", "disallowed-types")


@dataclass
class Form:
    text: str
    path: str
    also: list[str] = field(default_factory=list)


@dataclass
class Data:
    header: str
    methods: list[tuple[str, str]]  # (path, reason)
    types: list[tuple[str, str]]
    forms: list[Form]
    clean: list[str]


def load_data(text: str) -> Data:
    d = tomllib.loads(text)
    unknown = set(d) - {"header", "method-reason", "type-reason", "method", "type", "form", "clean"}
    if unknown:
        raise ValueError(f"clippy_lib_gate.toml: unknown keys {sorted(unknown)}")

    def entries(key: str, default_key: str) -> list[tuple[str, str]]:
        out = []
        for e in d.get(key, []):
            if not isinstance(e, dict) or not isinstance(e.get("path"), str) or not e["path"]:
                raise ValueError(f"clippy_lib_gate.toml: every [[{key}]] needs a path")
            reason = e.get("reason", d.get(default_key))
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"clippy_lib_gate.toml: [[{key}]] {e['path']} has no reason and no {default_key}")
            out.append((e["path"], reason))
        paths = [p for p, _ in out]
        dup = sorted({p for p in paths if paths.count(p) > 1})
        if dup:
            raise ValueError(f"clippy_lib_gate.toml: [[{key}]] lists {', '.join(dup)} more than once")
        return out

    forms = []
    for e in d.get("form", []):
        if not isinstance(e, dict) or not isinstance(e.get("text"), str) or not isinstance(e.get("path"), str) \
                or not e["text"] or not e["path"]:
            raise ValueError("clippy_lib_gate.toml: every [[form]] needs a text and the path its diagnostic names")
        also = e.get("also", [])
        if not isinstance(also, list) or not all(isinstance(a, str) and a for a in also):
            raise ValueError(f"clippy_lib_gate.toml: [[form]] `{e['text']}` also must be a list of paths")
        forms.append(Form(e["text"], e["path"], also))
    texts = [f.text for f in forms]
    dup = sorted({t for t in texts if texts.count(t) > 1})
    if dup:
        raise ValueError(f"clippy_lib_gate.toml: [[form]] text {', '.join(repr(t) for t in dup)} declared twice")
    clean = []
    for e in d.get("clean", []):
        if not isinstance(e, dict) or not isinstance(e.get("text"), str) or not e["text"]:
            raise ValueError("clippy_lib_gate.toml: every [[clean]] needs a text")
        clean.append(e["text"])
    header = d.get("header", "")
    if not isinstance(header, str):
        raise ValueError("clippy_lib_gate.toml: header must be a string")
    data = Data(header, entries("method", "method-reason"), entries("type", "type-reason"), forms, clean)
    # every library-only entry is exercised: clippy says nothing about a method path that names no
    # method (`f32::no_such_method`), so only a form naming it shows that the entry fires
    named = {f.path for f in forms} | {a for f in forms for a in f.also}
    bare = [p for p, _ in data.methods + data.types if p not in named]
    if bare:
        raise ValueError(f"clippy_lib_gate.toml: {len(bare)} library-only entr{'y has' if len(bare) == 1 else 'ies have'} "
                         f"no [[form]] naming it: {', '.join(bare)}")
    return data


def list_span(text: str, key: str) -> tuple[int, int] | None:
    """(start, end) of the `key = [` ... `]` block, or None when the file has no such list."""
    marker = f"{key} = [\n"
    n = text.count(marker)
    if n == 0:
        return None
    if n != 1:
        raise ValueError(f"clippy.toml: expected at most one `{key} = [` line, found {n}")
    i = text.index(marker)
    j = text.index("\n]", i)
    return i, j


def count_entries(text: str, key: str = "disallowed-methods") -> int:
    span = list_span(text, key)
    return 0 if span is None else len(re.findall(r'path\s*=\s*"', text[span[0]:span[1]]))


def crowded(text: str) -> list[str]:
    """Lines of a root list holding more than one entry (each entry is one line)."""
    out = []
    for key in LISTS:
        span = list_span(text, key)
        if span is None:
            continue
        for line in text[span[0]:span[1]].split("\n")[1:]:
            if len(re.findall(r'path\s*=\s*"', line)) > 1:
                out.append(f"{key}: {line.strip()[:60]}")
    return out


def render(root_text: str, data: Data) -> str:
    """The library configuration generated from the root one."""
    out = root_text
    for key, extra in (("disallowed-methods", data.methods), ("disallowed-types", data.types)):
        if not extra:
            continue
        for path, _ in extra:
            if f'path = "{path}"' in root_text:
                raise ValueError(f"clippy.toml already lists {path}: it belongs in clippy_lib_gate.toml only")
        span = list_span(out, key)
        if span is None:
            raise ValueError(f"clippy.toml: no `{key} = [` list to add the library entries to")
        at = span[0] + len(f"{key} = [\n")
        entries = "".join(f'    {{ path = "{p}", reason = "{r}" }},\n' for p, r in extra)
        out = out[:at] + entries + out[at:]
    return data.header + "\n" + out


def check(root_text: str, lib_text: str | None, data: Data) -> list[str]:
    many = crowded(root_text)
    if many:
        return [f"clippy.toml: {len(many)} line(s) hold more than one entry: " + "; ".join(many)]
    want = render(root_text, data)
    rel = os.path.relpath(LIB_CONFIG, ROOT)
    if lib_text is None:
        return [f"{rel} is missing (python3 scripts/clippy_lib_gate.py --write)"]
    if lib_text != want:
        return [f"{rel} differs from what clippy.toml generates (python3 scripts/clippy_lib_gate.py --write)"]
    # the generated file equals the render, so only the root list's own size is left to check:
    # an empty root list would leave the libm unforbidden while everything else matched
    if count_entries(root_text) == 0:
        return ["clippy.toml: disallowed-methods lists nothing (compared nothing)"]
    return []


def line_of(lines: list[str], text: str, what: str) -> int:
    hits = [n for n, l in enumerate(lines, 1) if text in l and not l.lstrip().startswith("//")]
    if len(hits) != 1:
        raise ValueError(f"canary: {what} `{text}` occurs {len(hits)} times, expected once")
    return hits[0]


def canary_lines(source: str, data: Data) -> tuple[dict[int, list[Form]], dict[int, str]]:
    """Forms by line, and clean lines; each text must occur exactly once."""
    lines = source.split("\n")
    forms: dict[int, list[Form]] = {}
    for f in data.forms:
        forms.setdefault(line_of(lines, f.text, "form"), []).append(f)
    clean = {line_of(lines, t, "clean line"): t for t in data.clean}
    both = sorted(set(forms) & set(clean))
    if both:
        raise ValueError(f"canary: line(s) {both} hold both a form and a clean text")
    return forms, clean


def judge_canary(rc: int, output: str, forms: dict[int, list[Form]],
                 clean: dict[int, str]) -> tuple[list[str], int, int]:
    """Errors, the number of forms flagged by their own path, and the number of diagnostics."""
    diags = [(int(m["line"]), m["path"]) for m in DIAG.finditer(output) if m["file"].endswith("lib.rs")]
    seen = set(diags)
    errors = [f"clippy.toml entry `{m['path']}` does not refer to anything: it forbids nothing"
              for m in UNRESOLVED.finditer(output)]
    expected = sum(len(v) for v in forms.values())
    if expected == 0:
        errors.append("no canary form to judge (expected 0, compared nothing)")
    if rc == 0:
        errors.append("clippy exited 0 on the canary: the configuration did not forbid anything")
    flagged = 0
    for line, fs in sorted(forms.items()):
        for f in fs:
            if (line, f.path) in seen:
                flagged += 1
            else:
                other = sorted(p for l, p in seen if l == line)
                errors.append(f"canary line {line} (`{f.text}`) has no diagnostic naming `{f.path}`"
                              + (f" (only {', '.join(other)})" if other else ""))
    for line, text in sorted(clean.items()):
        on = sorted(p for l, p in seen if l == line)
        if on:
            errors.append(f"canary line {line} (`{text}`) must stay clean but has {', '.join(on)}")
    accounted = {(line, p) for line, fs in forms.items() for f in fs for p in [f.path, *f.also]}
    for line, path in sorted(seen - accounted):
        if line not in clean:
            errors.append(f"unaccounted diagnostic: line {line} `{path}` (no [[form]] on that line names it)")
    return errors, flagged, len(diags)


def run_canary(data: Data) -> list[str]:
    src = os.path.join(CANARY, "src", "lib.rs")
    with open(src, encoding="utf-8") as f:
        forms, clean = canary_lines(f.read(), data)
    env = dict(os.environ, CLIPPY_CONF_DIR=LIB_DIR, CARGO_TERM_COLOR="never")
    target = os.environ.get("CARGO_TARGET_DIR") or os.path.join(tempfile.gettempdir(), "clippy-lib-canary-target")
    cmd = ["cargo", "clippy", "--quiet", "--manifest-path", os.path.join(CANARY, "Cargo.toml"),
           "--target-dir", os.path.join(target, "clippy-lib-canary"), "--message-format=short",
           "--", "-D", "clippy::disallowed_methods", "-D", "clippy::disallowed_types"]
    # touch the source so clippy re-lints it (a cached clean check prints nothing)
    os.utime(src)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    errors, flagged, n = judge_canary(r.returncode, r.stdout + r.stderr, forms, clean)
    expected = sum(len(v) for v in forms.values())
    print(f"clippy-lib canary: {flagged}/{expected} forms flagged by their own path, {n} diagnostics, "
          f"{len(clean)} clean lines (clippy exit {r.returncode})")
    if errors:
        errors.append("clippy output:\n" + (r.stdout + r.stderr)[-3000:])
    return errors


def main(argv: list[str]) -> int:
    try:
        return run(argv)
    except ValueError as e:  # a refused data file, configuration or canary: one line, red
        print(f"error: {e}", file=sys.stderr)
        return 1


def run(argv: list[str]) -> int:
    with open(DATA, encoding="utf-8") as f:
        data = load_data(f.read())
    with open(ROOT_CONFIG, encoding="utf-8") as f:
        root_text = f.read()
    if "--write" in argv:
        os.makedirs(LIB_DIR, exist_ok=True)
        with open(LIB_CONFIG, "w", encoding="utf-8") as f:
            f.write(render(root_text, data))
        print(f"wrote {os.path.relpath(LIB_CONFIG, ROOT)}")
        return 0
    lib_text = None
    if os.path.exists(LIB_CONFIG):
        with open(LIB_CONFIG, encoding="utf-8") as f:
            lib_text = f.read()
    errors = check(root_text, lib_text, data)
    if not errors:
        print(f"clippy-lib config: root {count_entries(root_text)} disallowed methods + {len(data.methods)}, "
              f"root {count_entries(root_text, 'disallowed-types')} disallowed types + {len(data.types)}")
    if not errors and "--check" not in argv:
        errors = run_canary(data)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
