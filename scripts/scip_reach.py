#!/usr/bin/env python3
"""Integration ledger from rust-analyzer SCIP: which `pub` items are reached, and from where.

`scripts/wiring_guard.py` counts an item as wired when its *name* appears in any
non-test code, including `examples/`. Two consequences: (1) an item that only an
example calls passes, so an API that no other part of the crate or binding uses
looks wired; (2) names are not resolved, so `impl Foo {}` alone wires `Foo`, and
`.eval(` wires every `eval` in the crate.

This script resolves references with rust-analyzer's SCIP output
(`scripts/scip_index.sh`), so each reference points at one definition, and
classifies every `pub` / `pub(crate)` item defined in `src/`:

  L0  unreached       no non-test code reaches it, examples included
  L1  example-only    reached only when examples / benches / fuzz count as roots
  live                reached without examples: from crate-internal roots or a binding

Roots without examples: the bindings (src/ffi/, src/python/, src/godot/,
src/wasm.rs), module-level code that is not a `use` statement, and trait-impl
methods of traits defined outside the crate (Display, Default, Drop, ... are
called implicitly). A trait method that is reached also reaches every impl of
it in the crate, so calls through `dyn Trait` and generic bounds are followed.

The trait-impl links come from the impl symbol names (`impl#[X][Tr]run().`
implements `Tr#run().`), because rust-analyzer's SCIP output carries no
`is_implementation` relationships (measured with rust-analyzer 1.98.1: 0
relationships of any kind in 88580 symbol infos). Relationships are still read
when present. Until 2026-10-04 they were the only source, so no impl body was
ever reached and their callees were reported as L0.

References inside comments, strings, `#[cfg(test)]` code and `use` statements
are dropped with the same preprocessing wiring_guard.py uses, so the two
checkers agree on what a reference is and differ only in resolution.

L1 is a label, not a defect: a library API that users call directly is expected
to be example-only inside this crate.

Usage:
  python3 scripts/scip_reach.py [--scip target/scip] [--write docs/integration-status.md]
  python3 scripts/scip_reach.py --check-baseline     # ratchet: no new L0, no stale entry
  python3 scripts/scip_reach.py --write-baseline     # after an intended change
Exit 1 when an index is missing, when the analysis compared nothing
(0 items, 0 references from examples / bindings / the fuzz crate, or 0 trait-impl links), or, with --check-baseline,
when an L0 item is not in scripts/integration-baseline.txt (a new public item
that nothing reaches) or a baseline entry is no longer L0 (remove the line).
"""

from __future__ import annotations

import argparse
import re
import sys
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiring_guard import USE_RE, remove_cfg_test, strip_rust  # noqa: E402

ROOT_DIRS = ("examples/", "benches/", "fuzz/")
# A binding is a file or a directory (a trailing `/`); every file under it is a root.
BINDINGS = ("src/ffi/", "src/python/", "src/godot/", "src/wasm.rs")


def binding_of(rel: str) -> str | None:
    """The binding `rel` belongs to (its key in `roots_by_binding`), or None."""
    for b in BINDINGS:
        if rel == b or (b.endswith("/") and rel.startswith(b)):
            return b
    return None
PUB_DEF = r'\bpub(?:\([^)]*\))?\s+(?:(?:const|unsafe|async|extern(?:\s+"[^"]*")?|default)\s+)*(?:fn|struct|enum|const|static|trait|type|union)\s+'


# --- SCIP (protobuf) decoding, stdlib only ----------------------------------

def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _fields(b: bytes):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield f, wt, v


def _packed(v, wt) -> list[int]:
    if wt != 2:
        return [v]
    out, i = [], 0
    while i < len(v):
        x, i = _varint(v, i)
        out.append(x)
    return out


def _span(r: list[int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """SCIP range: [startLine, startChar, endLine, endChar] or [line, startChar, endChar]."""
    if len(r) == 4:
        return (r[0], r[1]), (r[2], r[3])
    if len(r) == 3:
        return (r[0], r[1]), (r[0], r[2])
    return None


def load_scip(path: Path) -> list[dict]:
    """Documents with occurrences and symbol relationships."""
    docs = []
    for f, _wt, v in _fields(path.read_bytes()):
        if f != 2:
            continue
        doc = {"path": "", "occ": [], "impl": []}
        for f2, _w2, v2 in _fields(v):
            if f2 == 1:
                doc["path"] = v2.decode()
            elif f2 == 2:
                occ = {"range": [], "symbol": "", "roles": 0, "enc": []}
                for f3, w3, v3 in _fields(v2):
                    if f3 == 1:
                        occ["range"] = _packed(v3, w3)
                    elif f3 == 2:
                        occ["symbol"] = v3.decode()
                    elif f3 == 3:
                        occ["roles"] = v3
                    elif f3 == 7:
                        occ["enc"] = _packed(v3, w3)
                doc["occ"].append(occ)
            elif f2 == 3:
                sym, targets = "", []
                for f3, _w3, v3 in _fields(v2):
                    if f3 == 1:
                        sym = v3.decode()
                    elif f3 == 4:
                        rel_sym, is_impl = "", False
                        for f4, _w4, v4 in _fields(v3):
                            if f4 == 1:
                                rel_sym = v4.decode()
                            elif f4 == 3:
                                is_impl = bool(v4)
                        if is_impl and rel_sym:
                            targets.append(rel_sym)
                for t in targets:
                    doc["impl"].append((sym, t))
        docs.append(doc)
    return docs


# --- analysis ----------------------------------------------------------------

class Analysis:
    def __init__(self) -> None:
        self.items: dict[str, set[str]] = {}       # key "src/x.rs::name" -> symbols
        self.level: dict[str, str] = {}            # key -> L0 / L1 / live
        self.example_refs = 0
        self.binding_refs = 0
        self.unindexed: list[str] = []  # pub items in the source with no index definition
        self.fuzz_refs = 0        # references from the fuzz crate's index to this crate
        self.fuzz_docs = 0
        self.impl_links = 0       # trait-impl method -> in-crate trait method
        self.external_impls = 0   # trait-impl methods of traits defined outside the crate
        self.generated_items = 0  # items of types a macro invocation generates


IMPL_HEADER_RE = re.compile(r"\bimpl\b[^{;]*\{")
PUB_NAME_RE = re.compile(PUB_DEF + r"([A-Za-z_][A-Za-z0-9_]*)")

# fuzz/ is a separate crate: its index is read last, and its paths get this prefix
FUZZ_INDEX = "fuzz.scip"
FUZZ_PREFIX = "fuzz/"

# rust-analyzer names a trait-impl method `<module>/impl#[<Type>][<Trait>]<method>().`
# The type part may itself contain brackets (`[T; 3]`), so the greedy first group
# backs off to the last `][` before the method. The trait part is bare (`TgsHooks`)
# or backticked with generics and a path (`` `From<crate::x::Y>` ``).
IMPL_METHOD_RE = re.compile(r"impl#\[(.*)\]\[([^\[\]]+)\]([A-Za-z_][A-Za-z0-9_]*)\(\)\.$")
# a type (struct / enum / union / trait) definition: `<module>/<Name>#`
TYPE_DEF_RE = re.compile(r"(?:^|[/ ])([A-Za-z_][A-Za-z0-9_]*)#$")
# a symbol that names a type: the type itself or one of its fields / variants
TYPE_OR_MEMBER_RE = re.compile(r"(?:^|[/ ])([A-Za-z_][A-Za-z0-9_]*)#(?:[A-Za-z_][A-Za-z0-9_]*\.)?$")
# a method of an inherent or trait impl: `impl#[<Type>]m().` / `impl#[<Type>][<Trait>]m().`
INHERENT_OR_IMPL_RE = re.compile(r"impl#\[([^\]]*(?:\[[^\]]*\][^\]]*)*)\]")
# a field or variant: `<module>/<Name>#<member>.` (not a method `...().`); the
# match starts at the `#`, so the owner type symbol is everything before it plus `#`
FIELD_RE = re.compile(r"#[A-Za-z_][A-Za-z0-9_]*\.$")
# a method defined inside a trait or type body: `<module>/<Name>#<method>().`, or
# `<Name>#<method>().` right after the package version for an item at the crate root
MEMBER_METHOD_RE = re.compile(r"(?:^|[/ ])([A-Za-z_][A-Za-z0-9_]*)#([A-Za-z_][A-Za-z0-9_]*)\(\)\.$")


def _trait_name(raw: str) -> str:
    """`TgsHooks` / `` `From<crate::a::B>` `` / `` `core::ops::Add<Self>` `` -> bare trait name."""
    name = raw.strip("`").split("<", 1)[0]
    return name.rsplit("::", 1)[-1]


def link_trait_impls(defined: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    """Trait-impl methods recovered from symbol names.

    rust-analyzer's SCIP output carries no `is_implementation` relationships
    (rust-analyzer 1.98.1 emits none of any kind), so a call through a trait never
    reached the impl bodies. The impl symbol names its trait, so the link is rebuilt
    here: `impl#[X][Tr]run().` implements every in-crate `.../Tr#run().`. Returns
    (trait method -> impl methods, impl methods of traits not defined in the crate).

    Matching is by trait *name*: two in-crate traits with the same name and method
    would both receive the impl, which can only make more items reached (never fewer).
    A trait counts as external only when no in-crate symbol carries its name, so an
    impl of an in-crate trait never becomes a root by a missed match.
    """
    members: dict[tuple[str, str], set[str]] = {}
    impls: list[tuple[str, str, str]] = []
    for s in defined:
        m = IMPL_METHOD_RE.search(s)
        if m:
            impls.append((s, _trait_name(m.group(2)), m.group(3)))
            continue
        m = MEMBER_METHOD_RE.search(s)
        if m and not m.group(1).startswith("impl"):
            members.setdefault((m.group(1), m.group(2)), set()).add(s)
    in_crate = {name for name, _ in members}
    linked: dict[str, set[str]] = {}
    external: set[str] = set()
    for sym, trait, method in impls:
        targets = members.get((trait, method))
        if targets:
            for t in targets:
                linked.setdefault(t, set()).add(sym)
        elif trait not in in_crate:
            external.add(sym)
        # an in-crate trait without that method links nowhere and is no root
    return linked, external


def _keep_mask(text: str) -> tuple[list[str], list[str]]:
    """Per line, the code with comments / strings / cfg(test) / use statements /
    impl headers blanked. An `impl Foo {` header is not a use of `Foo`: it sits at
    module level, so counting it would make every type with an impl block reached
    (the same blind spot as the name-based guard's limit 3)."""
    code = remove_cfg_test(strip_rust(text))
    blank = lambda m: re.sub(r"[^\n]", " ", m.group(0))  # noqa: E731
    code = USE_RE.sub(blank, code)
    code = IMPL_HEADER_RE.sub(lambda m: blank(m)[:-1] + "{", code)
    return text.split("\n"), code.split("\n")


def unindexed_items(root: Path, indexed: set[str]) -> list[str]:
    """`src/x.rs::name` for every `pub` definition in the source (comments, strings,
    cfg(test) and use statements removed) that no index definition accounts for."""
    out = set()
    for p in sorted((Path(root) / "src").rglob("*.rs")):
        rel = p.relative_to(root).as_posix()
        if binding_of(rel):
            continue
        _raw, code = _keep_mask(p.read_text(encoding="utf-8", errors="replace"))
        for line in code:
            for m in PUB_NAME_RE.finditer(line):
                key = f"{rel}::{m.group(1)}"
                if key not in indexed:
                    out.add(key)
    return sorted(out)


def item_key(rel: str, name: str, sym: str) -> str:
    """`src/x.rs::name`, or `src/x.rs::Type::name` for an item of an impl block.

    Keyed by name alone, `Triangle::closest_point` and `TriMesh::closest_point` in
    one file were one item, reached as soon as either was: 21 keys hid an
    unreached method behind a reached one of the same name (2026-10-04)."""
    m = INHERENT_OR_IMPL_RE.search(sym)
    return f"{rel}::{_trait_name(m.group(1))}::{name}" if m else f"{rel}::{name}"


def legacy_key(key: str) -> str:
    """`src/x.rs::Type::name` -> `src/x.rs::name`, the form wiring-baseline.txt uses."""
    rel, _, rest = key.partition("::")
    return f"{rel}::{rest.rsplit('::', 1)[-1]}"


def _visible(code_lines: list[str], pos: tuple[int, int]) -> bool:
    line, ch = pos
    return line < len(code_lines) and ch < len(code_lines[line]) and not code_lines[line][ch].isspace()


MACRO_RULES_RE = re.compile(r"\bmacro_rules!\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{")
MACRO_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)!\s*[\(\[\{]")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


MEMBER_DEF_RE = re.compile(r"\b(?:fn|const)\s+([A-Za-z_][A-Za-z0-9_]*)")
META_PATH_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)\s*::\s*([A-Za-z_][A-Za-z0-9_]*)")
META_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[a-z]+")


def macro_params(body: str) -> list[str]:
    """The metavariables of the first rule's matcher, in order (`($n:ident, $c:ident) => ...`)."""
    head = body.split("=>", 1)[0]
    return META_PARAM_RE.findall(head)


def split_members(body: str) -> tuple[dict[str, str], str]:
    """A macro body split into the text of each `fn` / `const` it defines (its
    signature through the end of its block or `;`) and the remaining text."""
    parts: dict[str, str] = {}
    rest, last = [], 0
    for m in MEMBER_DEF_RE.finditer(body):
        if m.start() < last:
            continue  # inside a member already taken
        i, depth, opened = m.end(), 0, False
        while i < len(body):
            c = body[i]
            if c == "{":
                depth, opened = depth + 1, True
            elif c == "}":
                depth -= 1
                if opened and depth == 0:
                    i += 1
                    break
            elif c == ";" and not opened:
                i += 1
                break
            i += 1
        rest.append(body[last:m.start()])
        parts[m.group(1)] = parts.get(m.group(1), "") + body[m.start():i]
        last = i
    rest.append(body[last:])
    return parts, "".join(rest)


def macro_bodies(text: str) -> dict[str, str]:
    """`macro_rules! name { ... }` -> its body text (brace-matched, comments and
    strings already blanked by the caller)."""
    out: dict[str, str] = {}
    for m in MACRO_RULES_RE.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out[m.group(1)] = text[m.end():i - 1]
    return out


def analyze(root: Path, scip_paths: list[Path], keep_graph: bool = False) -> Analysis:
    """Classify every pub item. With `keep_graph=True` the result also carries the
    reachability function and the root sets (`a.reach`, `a.roots_binding`,
    `a.roots_by_binding` per binding file, `a.roots_example`, `a.roots_core`) so callers can ask what a different set of
    roots reaches (scripts/integration_levels.py). The default leaves the result
    exactly as before: levels, counts and baseline keys do not depend on the flag."""
    root = Path(root)
    a = Analysis()
    edges: dict[object, set[str]] = {}
    roots_core: set[str] = set()
    roots_binding: set[str] = set()  # also in roots_core; kept apart only for keep_graph
    roots_by_binding: dict[str, set[str]] = {}
    roots_example: set[str] = set()
    implementers: dict[str, set[str]] = {}
    defined: set[str] = set()
    impl_targets: dict[str, set[str]] = {}
    src_defined: set[str] = set()
    texts: dict[str, tuple[list[str], list[str]]] = {}
    seen: set[tuple] = set()
    # types a macro invocation defines: (file, macro name) -> {type name: type symbol}
    generated: dict[tuple[str, str], dict[str, str]] = {}
    # the arguments of that invocation, in order: (file, macro, type name) -> [arg, ...]
    gen_args: dict[tuple[str, str, str], list[str]] = {}

    def lines_of(rel: str) -> tuple[list[str], list[str]]:
        if rel not in texts:
            p = root / rel
            texts[rel] = _keep_mask(p.read_text(encoding="utf-8", errors="replace")) if p.exists() else ([], [])
        return texts[rel]

    # the fuzz index last: its references are counted against src_defined
    scip_paths = sorted(scip_paths, key=lambda sp: Path(sp).name == FUZZ_INDEX)
    for sp in scip_paths:
        # the fuzz crate is indexed from fuzz/, so its paths are relative to it
        prefix = FUZZ_PREFIX if Path(sp).name == FUZZ_INDEX else ""
        for doc in load_scip(sp):
            doc["path"] = prefix + doc["path"]
            rel = doc["path"]
            if prefix:
                a.fuzz_docs += 1
            for sym, target in doc["impl"]:
                implementers.setdefault(target, set()).add(sym)
                impl_targets.setdefault(sym, set()).add(target)
            is_src = rel.startswith("src/")
            is_root_file = rel.startswith(ROOT_DIRS)
            if not (is_src or is_root_file):
                continue  # tests/ and anything else never root or carry references
            raw, code = lines_of(rel)
            defs = []
            refs = []
            for o in doc["occ"]:
                s = o["symbol"]
                sp_ = _span(o["range"])
                if not s or s.startswith("local ") or sp_ is None:
                    continue
                if o["roles"] & 1:
                    defined.add(s)
                    if is_src:
                        src_defined.add(s)
                    enc = _span(o["enc"]) if o["enc"] else None
                    if enc is not None:
                        defs.append((enc[0], enc[1], s))
                    if is_src and not binding_of(rel) and _visible(code, sp_[0]):
                        line = raw[sp_[0][0]] if sp_[0][0] < len(raw) else ""
                        name = line[sp_[0][1]:sp_[1][1]]
                        if name and re.search(PUB_DEF + re.escape(name) + r"\b", line):
                            a.items.setdefault(item_key(rel, name, s), set()).add(s)
                        else:
                            # a type defined at a macro invocation (`impl_x!(Name, 10);`):
                            # rust-analyzer puts the definition on the argument token
                            call = MACRO_CALL_RE.search(code[sp_[0][0]] if sp_[0][0] < len(code) else "")
                            tm = TYPE_DEF_RE.search(s)
                            if call and tm and tm.group(1) == name:
                                generated.setdefault((rel, call.group(1)), {})[name] = s
                                line = code[sp_[0][0]]
                                inner = line[call.end():].split(")")[0].split("]")[0].split("}")[0]
                                gen_args[(rel, call.group(1), name)] = [x.strip() for x in inner.split(",")]
                else:
                    if is_src and not _visible(code, sp_[0]):
                        continue
                    refs.append((sp_[0], s))
            # innermost enclosing definition for every reference (ranges nest)
            defs.sort(key=lambda d: (d[0], (-d[1][0], -d[1][1])))
            starts = [d[0] for d in defs]
            # a reached type reaches its fields and variants (`Type#field.`, `Enum#Variant.`),
            # so the types *they* name are reached too; methods (`...().`) are not members
            # in this sense: using a type or trait does not call every method on it
            open_defs: list[tuple] = []
            for st, en, ds in defs:
                while open_defs and open_defs[-1][1] < st:
                    open_defs.pop()
                if open_defs and open_defs[-1][2].endswith("#") and not ds.endswith(")."):
                    edges.setdefault(open_defs[-1][2], set()).add(ds)
                open_defs.append((st, en, ds))
            for pos, s in refs:
                key = (rel, pos, s)
                if key in seen:
                    continue
                seen.add(key)
                ctx = None
                j = bisect_right(starts, pos) - 1
                while j >= 0:
                    st, en, ds = defs[j]
                    if st <= pos <= en:
                        ctx = ds
                        break
                    j -= 1
                if is_root_file:
                    roots_example.add(s)
                    if not prefix:
                        a.example_refs += 1  # the fuzz crate has its own guard below
                    elif s in src_defined:
                        a.fuzz_refs += 1  # libfuzzer-sys / arbitrary / std do not count
                elif binding_of(rel):
                    roots_core.add(s)
                    roots_binding.add(s)
                    roots_by_binding.setdefault(binding_of(rel), set()).add(s)
                    a.binding_refs += 1
                elif ctx is None:
                    roots_core.add(s)  # module-level code that is not a `use`
                else:
                    edges.setdefault(ctx, set()).add(s)

    # Items a macro_rules body defines. rust-analyzer emits the generated types at
    # the invocation (collected above) but no definition for the members the body
    # writes, and it records no reference from inside the expanded body. So:
    #   * each generated type is an item (`file::Type`), and each `pub fn` /
    #     `pub const` of the body is an item per generated type
    #     (`file::Type::name`), keyed by the symbols references use
    #     (`impl#[Type]name().` / `impl#[Type]NAME.`): reached when referenced;
    #   * each member reaches what its own text names: same-module items, the other
    #     members of the same generated type (`self.x()` / `Self::X`), and through
    #     `$param::name` the member `name` of the type that invocation passed as
    #     `$param` (resolved from the matcher and the invocation's arguments); the
    #     type reaches what the text outside the members names.
    defined_by_name: dict[tuple[str, str], set[str]] = {}
    for d in defined:
        head, _, tail = d.rpartition("/")
        for suffix in ("#", "().", "."):
            if tail.endswith(suffix) and not tail[:-len(suffix)].count("#"):
                defined_by_name.setdefault((head, tail[:-len(suffix)]), set()).add(d)
    for (rel, macro), types in sorted(generated.items()):
        body = macro_bodies("\n".join(lines_of(rel)[1])).get(macro)
        if body is None:
            continue
        members = sorted(set(PUB_NAME_RE.findall(body)))
        member_text, outside = split_members(body)
        params = macro_params(body)
        for tname, tsym in sorted(types.items()):
            base = tsym[:-len(tname) - 1]  # "...<module>/"
            gen_syms = {tsym}
            a.items.setdefault(f"{rel}::{tname}", set()).add(tsym)
            for n in members:
                syms = {f"{base}impl#[{tname}]{n}().", f"{base}impl#[{tname}]{n}."}
                a.items.setdefault(f"{rel}::{tname}::{n}", set()).update(syms)
                gen_syms |= syms
            mod = base.rstrip("/")
            bound = dict(zip(params, gen_args.get((rel, macro, tname), [])))

            def reach(text: str, own: str | None) -> set[str]:
                out: set[str] = set()
                for ident in set(IDENT_RE.findall(text)) - ({own} if own else set()):
                    out |= defined_by_name.get((mod, ident), set())
                    if ident in member_text:  # another member of this generated type
                        out |= {f"{base}impl#[{tname}]{ident}().", f"{base}impl#[{tname}]{ident}."}
                for param, name in META_PATH_RE.findall(text):
                    other = bound.get(param)
                    if other:
                        out |= {f"{base}impl#[{other}]{name}().", f"{base}impl#[{other}]{name}."}
                return out

            edges.setdefault(tsym, set()).update(reach(outside, None) - {tsym})
            for n, text in member_text.items():
                own = {f"{base}impl#[{tname}]{n}().", f"{base}impl#[{tname}]{n}."}
                for g in own:  # private members too: a pub member may reach others through them
                    edges.setdefault(g, set()).update(reach(text, n) - own)
            a.generated_items += len(members) + 1

    # pub items the index does not define and no macro invocation accounts for
    # (a macro_rules body nothing invokes, or a pattern this analysis does not
    # model). They have no level, so they are listed and ratcheted separately
    a.unindexed = unindexed_items(root, {legacy_key(k) for k in a.items})

    # Trait impls. An impl method runs only when (1) the trait method it implements
    # is reached, or the trait is defined outside the crate (Display, Default, Drop,
    # ... are called implicitly), and (2) a value of its self type exists, which is
    # approximated as "the type, or any member of it, is reached" (rapid type
    # analysis). Without (2) a reached trait method would reach every impl in the
    # crate, including impls of types nothing constructs, and every external-trait
    # impl would be a root (measured: 6 superseded TGS hook types and 2 unused
    # physics2d types became reached that way).
    external: set[str] = {s for s, ts in impl_targets.items() if any(t not in defined for t in ts)}
    a.impl_links = sum(len(v) for v in implementers.values())
    linked, ext_by_name = link_trait_impls(defined)
    for trait_method, impl_methods in linked.items():
        implementers.setdefault(trait_method, set()).update(impl_methods)
        a.impl_links += len(impl_methods)
    external |= ext_by_name
    a.external_impls = len(external)
    type_defs: dict[str, list[str]] = {}
    for d in defined:
        m = TYPE_DEF_RE.search(d)
        if m:
            type_defs.setdefault(m.group(1), []).append(d)

    def self_type(impl_sym: str) -> str | None:
        """Name of an impl method's self type, or None when it is not a crate type
        (`impl Tr for f32`, `impl Tr for Vec<T>`): values of those always exist."""
        m = IMPL_METHOD_RE.search(impl_sym)
        name = _trait_name(m.group(1)) if m else ""
        return name if name in type_defs else None

    def type_of(sym: str) -> str | None:
        """The crate type a live symbol shows to exist: `X#` (named), `X#field.`
        (accessed), `impl#[X]new().` / `impl#[X][Tr]m().` (called)."""
        m = INHERENT_OR_IMPL_RE.search(sym)
        if m:
            return _trait_name(m.group(1))
        m = TYPE_OR_MEMBER_RE.search(sym)
        return m.group(1) if m else None

    def reach(start: set[str]) -> set[str]:
        live = set(start)
        stack = list(start)
        pending = set(external)  # waiting for their self type
        while True:
            while stack:
                s = stack.pop()
                for t in edges.get(s, ()):
                    if t not in live:
                        live.add(t)
                        stack.append(t)
                # a field or variant that is used shows a value of its type exists
                m = FIELD_RE.search(s)
                if m:
                    owner = s[:m.start() + 1]
                    if owner not in live and owner in defined:
                        live.add(owner)
                        stack.append(owner)
                for t in implementers.get(s, ()):
                    if t not in live:
                        pending.add(t)
            reached_types = {ty for ty in map(type_of, live) if ty}
            ready = set()
            for t in pending:
                if t in live:
                    ready.add(t)
                    continue
                ty = self_type(t)
                if ty is None or ty in reached_types:
                    ready.add(t)
                    live.add(t)
                    stack.append(t)
            pending -= ready
            if not stack:
                return live

    live_core = reach(roots_core)
    live_all = reach(roots_core | roots_example)
    for key, syms in a.items.items():
        if syms & live_core:
            a.level[key] = "live"
        elif syms & live_all:
            a.level[key] = "L1"
        else:
            a.level[key] = "L0"
    if keep_graph:
        a.reach = reach
        a.roots_core = set(roots_core)
        a.roots_binding = set(roots_binding)
        a.roots_by_binding = {k: set(v) for k, v in roots_by_binding.items()}
        a.roots_example = set(roots_example)
    return a


def baseline_unwired(root: Path) -> set[str]:
    p = root / "scripts" / "wiring-baseline.txt"
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "unwired":
            out.add(parts[1])
    return out


def report(a: Analysis, baseline: set[str]) -> str:
    l0 = sorted(k for k, v in a.level.items() if v == "L0")
    l1 = sorted(k for k, v in a.level.items() if v == "L1")
    live = sum(1 for v in a.level.values() if v == "live")
    # wiring-baseline.txt names items by file and name only
    by_legacy: dict[str, list[str]] = {}
    for k in a.level:
        by_legacy.setdefault(legacy_key(k), []).append(k)
    missed = [k for k in l0 if legacy_key(k) not in baseline]
    resolved = sorted(b for b in baseline
                      if by_legacy.get(b) and all(a.level[k] in ("live", "L1") for k in by_legacy[b]))
    out = [
        "# ALICE-SDF Integration Status",
        "",
        "_Generated by `scripts/scip_reach.py` from rust-analyzer SCIP indexes (`scripts/scip_index.sh`); no timestamp, the file changes only when its content does._",
        "",
        "Every `pub` / `pub(crate)` item defined in `src/` (binding files excluded), classified by what reaches it.",
        "References are resolved to one definition each, so items that share a name are told apart.",
        "",
        "| Level | Meaning | Count |",
        "|-------|---------|------:|",
        f"| L0 | not reached by any non-test code, examples included | {len(l0)} |",
        f"| L1 | reached only from `examples/` / `benches/` / `fuzz/` | {len(l1)} |",
        f"| live | reached without examples (crate-internal roots or a binding) | {live} |",
        f"| | **total** | **{len(a.level)}** |",
        "",
        "L1 is a label, not a defect: a module users call directly is example-only inside this crate.",
        "It does mean the item is not reached from another module or a binding.",
        "",
        "## Compared with the wiring guard",
        "",
        f"`scripts/wiring-baseline.txt` lists {len(baseline)} unwired items.",
        "",
        f"### L0 here but not in the baseline ({len(missed)})",
        "",
        "The name-based guard counts these as wired; resolved references find no caller.",
        "",
    ]
    out += [f"- `{k}`" for k in missed] or ["- (none)"]
    out += [
        "",
        f"### In the baseline but reached here ({len(resolved)})",
        "",
        "The guard lists these as unwired; a resolved reference reaches them (level in brackets).",
        "",
    ]
    out += [f"- `{k}` ({'/'.join(sorted({a.level[x] for x in by_legacy[k]}))})" for k in resolved] or ["- (none)"]
    out += ["", f"## L0 — unreached ({len(l0)})", ""]
    out += [f"- `{k}`" for k in l0] or ["- (none)"]
    out += ["", f"## Not indexed ({len(a.unindexed)})", "",
            "`pub` items in the source that the SCIP index has no definition for (items inside a "
            "`macro_rules` body). Their reach is not checked; the baseline lists them so the set cannot grow unnoticed.", ""]
    out += [f"- `{k}`" for k in a.unindexed] or ["- (none)"]
    out += [
        "",
        "## Limits",
        "",
        "- A pattern in a `match` arm counts as a reference: a type that is only matched on, never constructed, is reached.",
        "- Code inside macro expansions is resolved as far as rust-analyzer resolves it.",
        "- Calls through a trait (`dyn Tr`, `T: Tr`) reach the impls of that method whose self type is reached (the type, a field, or one of its methods is live); an impl of a type nothing reaches stays unreached.",
        "- Trait-impl links come from the impl symbol names; rust-analyzer's SCIP output has no implementation relationships.",
        "- Methods are listed as `file::Type::method`, so same-named methods of different types in one file are told apart.",
        "- Items in the bindings (`src/ffi/`, `src/python/`, `src/godot/`, `src/wasm.rs`) are roots and are not listed.",
    ]
    out += ["", f"## L1 — example-only ({len(l1)})", ""]
    by_file: dict[str, list[str]] = {}
    for k in l1:
        f, n = k.split("::", 1)
        by_file.setdefault(f, []).append(n)
    if not by_file:
        out.append("- (none)")
    for f in sorted(by_file):
        out.append(f"- `{f}`: " + ", ".join(f"`{n}`" for n in by_file[f]))
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--scip", default="target/scip")
    ap.add_argument("--write", help="write the markdown ledger to this path")
    ap.add_argument("--baseline", default="scripts/integration-baseline.txt")
    ap.add_argument("--check-baseline", action="store_true", help="fail on new L0 items or stale baseline entries")
    ap.add_argument("--write-baseline", action="store_true", help="record the current L0 items as the baseline")
    args = ap.parse_args(argv)
    root = Path(args.root)
    sdir = Path(args.scip) if Path(args.scip).is_absolute() else root / args.scip
    paths = [sdir / "native.scip", sdir / "wasm.scip", sdir / FUZZ_INDEX]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"error: SCIP index missing: {missing} (run scripts/scip_index.sh)", file=sys.stderr)
        return 1
    a = analyze(root, paths)
    errors = []
    if not a.items:
        errors.append("0 pub items found in src/ (the analysis looked at nothing)")
    if a.example_refs == 0:
        errors.append("0 references from examples/benches (index or path filter is wrong)")
    if a.binding_refs == 0:
        errors.append("0 references from binding files (feature-gated modules were not indexed)")
    if a.fuzz_docs == 0 or a.fuzz_refs == 0:
        errors.append(f"fuzz index: {a.fuzz_docs} documents, {a.fuzz_refs} references to the crate "
                      "(fuzz targets would not count as callers)")
    if a.impl_links == 0:
        errors.append("0 trait-impl links resolved (calls through a trait would never reach an impl)")
    counts = {lv: sum(1 for v in a.level.values() if v == lv) for lv in ("L0", "L1", "live")}
    print(f"compared: items {len(a.items)}, example refs {a.example_refs}, binding refs {a.binding_refs}, "
          f"fuzz refs {a.fuzz_refs}, trait-impl links {a.impl_links}, external-trait impls {a.external_impls}, "
          f"unindexed {len(a.unindexed)}, "
          f"L0 {counts['L0']}, L1 {counts['L1']}, live {counts['live']}")
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if errors:
        return 1
    if args.write:
        Path(args.write).write_text(report(a, baseline_unwired(root)), encoding="utf-8")
    bpath = Path(args.baseline) if Path(args.baseline).is_absolute() else root / args.baseline
    l0 = sorted(k for k, v in a.level.items() if v == "L0")
    if args.write_baseline:
        bpath.write_text(BASELINE_HEADER + "".join(f"{k}\n" for k in l0)
                         + "".join(f"{UNINDEXED}{k}\n" for k in a.unindexed), encoding="utf-8")
        print(f"wrote {len(l0)} L0 entries and {len(a.unindexed)} unindexed entries to {bpath}")
    if args.check_baseline:
        if not bpath.exists():
            print(f"error: baseline {bpath} missing (run with --write-baseline)", file=sys.stderr)
            return 1
        lines = {ln.strip() for ln in bpath.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#")}
        base = {ln for ln in lines if not ln.startswith(UNINDEXED)}
        base_unindexed = {ln[len(UNINDEXED):] for ln in lines if ln.startswith(UNINDEXED)}
        new_unindexed = [k for k in a.unindexed if k not in base_unindexed]
        stale_unindexed = sorted(base_unindexed - set(a.unindexed))
        for k in new_unindexed:
            print(f"error: new unindexed item {k}: a pub item the SCIP index has no definition for "
                  "(a macro_rules body?), so its reach cannot be checked", file=sys.stderr)
        for k in stale_unindexed:
            print(f"error: stale unindexed entry {k}: remove the line", file=sys.stderr)
        new = [k for k in l0 if k not in base]
        stale = sorted(k for k in base if a.level.get(k) != "L0")
        for k in new:
            print(f"error: new L0 item {k}: no non-test code reaches it (add a caller or an example, "
                  "or remove it)", file=sys.stderr)
        for k in stale:
            lv = a.level.get(k, "gone")
            print(f"error: stale baseline entry {k} (now {lv}): remove the line", file=sys.stderr)
        print(f"baseline: {len(base)} entries, new L0 {len(new)}, stale {len(stale)}; "
              f"unindexed {len(base_unindexed)} entries, new {len(new_unindexed)}, stale {len(stale_unindexed)}")
        if new or stale or new_unindexed or stale_unindexed:
            return 1
    return 0


UNINDEXED = "unindexed: "
BASELINE_HEADER = """# L0 items (no non-test code reaches them) that existed when the ratchet was
# introduced. scripts/scip_reach.py --check-baseline fails on any L0 item not
# listed here, and on any line here that is no longer L0. Shrink this file;
# regenerate with --write-baseline only for an intended change.
# Lines starting with "unindexed: " are pub items the SCIP index has no
# definition for (macro_rules bodies); they have no level, the same ratchet
# applies to them.
"""


if __name__ == "__main__":
    sys.exit(main())
