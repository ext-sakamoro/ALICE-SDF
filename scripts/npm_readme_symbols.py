#!/usr/bin/env python3
"""The npm package's README names only what the wasm crate exports.

`@alice-sdf/wasm` is built by wasm-pack from `examples/wasm-demo`, so its API is exactly the
`#[wasm_bindgen]` items of `examples/wasm-demo/src/lib.rs`. A hand-written page drifts from that
(the old `npm/index.d.ts` declared 17 functions that exist nowhere). This check reads

- the JavaScript code blocks of `npm/README.md` and the JavaScript examples in the crate's `///`
  doc comments: imported names, `new X(`, `SdfType.Variant`, bare calls of imported functions, and
  method calls on a variable bound by `new X(` (checked against the methods of X);
- the API section of `npm/README.md`: under a `### \\`Class\\`` heading the backticked lower-case
  names are methods of that class, under `### \\`Enum\\`` the backticked names are its variants,
  under `### Functions` the backticked `name(` are free functions;

and fails when a name is not exported, or when nothing was compared.

usage: python3 scripts/npm_readme_symbols.py [--readme npm/README.md] [--lib examples/wasm-demo/src/lib.rs]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# the default export of a wasm-pack `--target web` build (it loads the module); not a crate item
LOADER = "init"

# JavaScript globals a usage example constructs (views over wasm memory, image data)
JS_CLASSES = {
    "Array",
    "Float32Array",
    "Float64Array",
    "ImageData",
    "Int32Array",
    "Uint32Array",
    "Uint8Array",
    "Uint8ClampedArray",
}

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


class Exports:
    def __init__(self) -> None:
        self.classes: dict[str, set[str]] = {}
        # associated functions without `self` (called as `Class.name(`)
        self.statics: dict[str, set[str]] = {}
        self.enums: dict[str, set[str]] = {}
        self.functions: set[str] = set()

    def names(self) -> set[str]:
        return set(self.classes) | set(self.enums) | self.functions


def _strip_comments(src: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in src.splitlines())


def _block(src: str, start: int) -> tuple[str, int]:
    """The text between the `{` at or after `start` and its matching `}`"""
    i = src.index("{", start)
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1 : j], j + 1
    raise ValueError("unbalanced braces")


def parse_exports(lib: str) -> Exports:
    ex = Exports()
    src = _strip_comments(lib)
    # an item preceded by #[wasm_bindgen] or #[wasm_bindgen(...)] (other attributes may follow)
    attr = re.compile(r"#\[wasm_bindgen(\((?P<args>[^)]*)\))?\]")
    inside_until = 0
    for m in attr.finditer(src):
        # attributes on the methods of an impl block are read with the block
        if m.start() < inside_until:
            continue
        args = m.group("args") or ""
        rest = src[m.end() :]
        # skip further attributes
        k = 0
        while True:
            a = re.match(r"\s*#\[[^\]]*\]", rest[k:])
            if not a:
                break
            k += a.end()
        tail = rest[k:]
        pos = m.end() + k
        if s := re.match(rf"\s*pub struct ({IDENT})", tail):
            ex.classes.setdefault(s.group(1), set())
        elif e := re.match(rf"\s*pub enum ({IDENT})", tail):
            body, _ = _block(src, pos)
            ex.enums[e.group(1)] = set(re.findall(rf"^\s*({IDENT})\s*(?:=[^,]*)?,?\s*$", body, re.M))
        elif f := re.match(rf"\s*pub fn ({IDENT})", tail):
            # `#[wasm_bindgen(start)]` runs on load and is not exported by name
            if "start" not in args:
                ex.functions.add(f.group(1))
        elif i := re.match(rf"\s*impl ({IDENT})", tail):
            body, inside_until = _block(src, pos)
            fns = re.findall(rf"pub fn ({IDENT})\s*\(([^)]*)\)", body)
            # the constructor is called as `new X(`, not as a method
            ctor = set(re.findall(rf"#\[wasm_bindgen\(constructor\)\]\s*pub fn ({IDENT})", body))
            cls = i.group(1)
            ex.classes.setdefault(cls, set()).update(
                n for n, params in fns if re.search(r"\bself\b", params) and n not in ctor
            )
            ex.statics.setdefault(cls, set()).update(
                n for n, params in fns if not re.search(r"\bself\b", params) and n not in ctor
            )
        elif re.match(r"\s*extern", tail):
            continue
    return ex


def js_blocks_readme(md: str) -> list[str]:
    return re.findall(r"^```(?:js|javascript|ts|typescript)\s*\n(.*?)^```", md, re.M | re.S)


def js_blocks_lib(lib: str) -> list[str]:
    blocks = []
    lines = lib.splitlines()
    i = 0
    while i < len(lines):
        t = lines[i].strip()
        if re.match(r"^///\s*```(js|javascript|ts|typescript)\s*$", t):
            body = []
            i += 1
            while i < len(lines) and not re.match(r"^///\s*```\s*$", lines[i].strip()):
                body.append(re.sub(r"^\s*/// ?", "", lines[i]))
                i += 1
            blocks.append("\n".join(body))
        i += 1
    return blocks


def check_code(code: str, ex: Exports, where: str, errors: list[str]) -> int:
    compared = 0
    code = "\n".join(line.split("//", 1)[0] for line in code.splitlines())
    imported: set[str] = set()
    for m in re.finditer(r"import\s+(?:(" + IDENT + r")\s*,?\s*)?(?:\{([^}]*)\})?\s*from", code):
        if m.group(1) and m.group(1) != LOADER:
            errors.append(f"{where}: default import `{m.group(1)}` (the default export is `{LOADER}`)")
        compared += 1 if m.group(1) else 0
        for name in re.findall(IDENT, m.group(2) or ""):
            imported.add(name)
            compared += 1
            if name not in ex.names():
                errors.append(f"{where}: imports `{name}`, which the crate does not export")
    bound: dict[str, str] = {}
    for m in re.finditer(rf"(?:const|let|var)\s+({IDENT})\s*=\s*new\s+({IDENT})\s*\(", code):
        bound[m.group(1)] = m.group(2)
    for cls in ex.classes:
        for m in re.finditer(rf"(?<![.\w]){cls}\.({IDENT})\s*\(", code):
            compared += 1
            if m.group(1) not in ex.statics.get(cls, set()):
                errors.append(f"{where}: `{cls}.{m.group(1)}(`, which is not a static function of {cls}")
        # a variable bound by a static call is taken to be of that class
        for m in re.finditer(rf"(?:const|let|var)\s+({IDENT})\s*=\s*{cls}\.{IDENT}\s*\(", code):
            bound[m.group(1)] = cls
    for m in re.finditer(rf"\bnew\s+({IDENT})\s*\(", code):
        compared += 1
        if m.group(1) in JS_CLASSES:
            compared -= 1
            continue
        if m.group(1) not in ex.classes:
            errors.append(f"{where}: `new {m.group(1)}(`, which is not an exported class")
    for enum, variants in ex.enums.items():
        for m in re.finditer(rf"\b{enum}\.({IDENT})", code):
            compared += 1
            if m.group(1) not in variants:
                errors.append(f"{where}: `{enum}.{m.group(1)}`, which is not a variant")
    for m in re.finditer(rf"(?<![.\w])({IDENT})\s*\(", code):
        name = m.group(1)
        if name in imported and name not in ex.classes and name != LOADER:
            compared += 1
            if name not in ex.functions:
                errors.append(f"{where}: calls `{name}(`, which is not an exported function")
    for m in re.finditer(rf"(?<![.\w])({IDENT})\.({IDENT})\s*\(", code):
        var, method = m.groups()
        if var in bound and bound[var] in ex.classes:
            compared += 1
            if method not in ex.classes[bound[var]]:
                errors.append(f"{where}: `{var}.{method}(` on a {bound[var]}, which has no such method")
    return compared


def check_api_section(md: str, ex: Exports, errors: list[str]) -> int:
    compared = 0
    api = md.split("\n## API", 1)
    if len(api) < 2:
        errors.append("npm/README.md: no `## API` section")
        return 0
    section = re.split(r"\n## ", api[1], maxsplit=1)[0]
    # drop code blocks (checked as code)
    section = re.sub(r"```.*?```", "", section, flags=re.S)
    for head, body in re.findall(r"^### (.+?)\n(.*?)(?=^### |\Z)", section, re.M | re.S):
        h = re.fullmatch(r"`(" + IDENT + r")`", head.strip())
        # the names a section lists: the first cell of a table row, and every name on a line
        # that lists them (`Members: ...`, or an enum's variants); other prose and the meaning
        # column mention arguments, not exports
        ticks = []
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if cells and not re.fullmatch(r"-+", cells[0]):
                    ticks += re.findall(r"`([^`]+)`", cells[0])
            elif line.startswith("Members:") or (h and h.group(1) in ex.enums):
                ticks += re.findall(r"`([^`]+)`", line)
        if h and h.group(1) in ex.classes:
            cls = h.group(1)
            for t in ticks:
                if t.startswith("new "):
                    compared += 1
                    if not re.match(rf"new {cls}\(", t):
                        errors.append(f"npm/README.md ### {cls}: `{t}` is not this class's constructor")
                    continue
                n = re.match(rf"({IDENT})(\(|$)", t)
                if n and n.group(1)[0].islower() and n.group(1) == n.group(1).lower():
                    compared += 1
                    if n.group(1) not in ex.classes[cls]:
                        errors.append(f"npm/README.md ### {cls}: `{t}` is not a method of {cls}")
        elif h and h.group(1) in ex.enums:
            enum = h.group(1)
            for t in ticks:
                if re.fullmatch(IDENT, t):
                    compared += 1
                    if t not in ex.enums[enum]:
                        errors.append(f"npm/README.md ### {enum}: `{t}` is not a variant")
        elif head.strip() == "Functions":
            for t in ticks:
                n = re.match(rf"({IDENT})\(", t)
                if n:
                    compared += 1
                    if n.group(1) not in ex.functions:
                        errors.append(f"npm/README.md ### Functions: `{t}` is not an exported function")
        elif h:
            errors.append(f"npm/README.md: heading `{h.group(1)}` names nothing the crate exports")
            compared += 1
    return compared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", type=Path, default=ROOT / "npm" / "README.md")
    ap.add_argument("--lib", type=Path, default=ROOT / "examples" / "wasm-demo" / "src" / "lib.rs")
    a = ap.parse_args(argv)
    md = a.readme.read_text(encoding="utf-8")
    lib = a.lib.read_text(encoding="utf-8")
    ex = parse_exports(lib)
    errors: list[str] = []
    if not ex.names():
        errors.append(f"{a.lib}: no #[wasm_bindgen] exports found")
    compared = 0
    for n, b in enumerate(js_blocks_readme(md), 1):
        compared += check_code(b, ex, f"npm/README.md code block {n}", errors)
    for n, b in enumerate(js_blocks_lib(lib), 1):
        compared += check_code(b, ex, f"{a.lib.name} doc example {n}", errors)
    compared += check_api_section(md, ex, errors)
    print(
        f"npm README symbols: {compared} names compared against {len(ex.names())} exports "
        f"({len(ex.classes)} classes, {len(ex.enums)} enums, {len(ex.functions)} functions)"
    )
    if compared == 0:
        errors.append("0 names compared: the check found nothing to check")
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
