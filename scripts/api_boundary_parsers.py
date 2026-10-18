#!/usr/bin/env python3
"""ALICE-Law G7 / G8: a `local` boundary entry must not reach a parser of outside bytes or text.

`scripts/api_boundary.py` lists every entry point that takes bytes, text or a reader, with a
decision written by hand in `scripts/api_boundary.toml`: `handshake` (checks the producer's
identifier), `pending` (must, and cannot yet) or `local` (no Law-bearing value from another
producer crosses it). Nothing checked that a `local` decision is consistent with what the function
does: `MslShader::from_wgsl` compiles arbitrary WGSL with naga and was once decided `local` by an
ordering mistake of the rule that wrote the list, and every check still passed.

This script follows the call graph of rust-analyzer's SCIP index (the graph of
`scripts/scip_reach.py`: references inside each definition, trait methods to their impls) from each
`fn::`, `abi::` (extern "C"), `binding::` (the Rust function a pyo3 wrapper calls) and `target::`
(wasm32-only) entry of the manifest, and fails when an entry decided `local` reaches a parser, unless
the manifest exempts that entry with a reason (`[parser_exempt]`, for a function that parses only
what this crate wrote, or only reads the identifier itself). The parsers are the manifest's
`parsers` list (those outside serde / bincode) plus every stored-form reader of the library clippy
configuration (.github/clippy-lib/clippy.toml, the set scripts/clippy_lib_hidden_check.py reads:
serde_json's from_* and Deserializer constructors, serde's Deserialize::deserialize /
DeserializeSeed::deserialize and their impls, bincode's decoders), matched by crate, type and name.
`impl::` (serde / FromStr implementations, mostly derive expansions) and `reexport::` entries have
no body of this crate in the index; they are counted in the report as not followed.

Manifest (`scripts/api_boundary.toml`):
  parsers = ["<crate> <descriptor>", ...]   the parse entry points outside serde / bincode, in SCIP
                                            form without the version (`naga front/wgsl/parse_str().`);
                                            each must be in the index (a name it cannot find is a
                                            typo, not a skip)
  [parser_exempt]    "<fn key>" = "<reason>"  a `local` entry that reaches a parser on purpose
  [parser_unindexed] "<fn key>" = "<reason>"  an entry the index has no definition for
Absent `parsers` fails: this repository adopted the rule, so dropping the list is not allowed (the
other repositories that share api_boundary.py do not run this script).

Fails on: a `local` entry reaching a parser without an exemption; an exemption that is stale (the
entry is gone, is not `local`, or no longer reaches a parser); a listed parser the index does not
contain; an empty parser list; a clippy configuration whose stored-form readers match nothing in
the index; an entry the index cannot resolve that is not in `[parser_unindexed]`,
and an unindexed entry that now resolves or is gone; a reason shorter than 12 characters; an index
built from another commit than HEAD or from a dirty tree (`<index>/commit`, written by
scripts/scip_index.sh); no resolved entry (compared nothing). The report names the index commit.

What the walk cannot see (known residuals, so a pass is not a proof):
  * calls through a function pointer or a closure stored in a value: the reference is to the
    variable, not to the function it holds (test_a_call_through_a_function_pointer_is_not_seen)
  * dynamic dispatch reaches the impls of a trait method only when the impl's self type is reached
    (scip_reach.reach), so an impl constructed in another crate is not followed
  * bodies a macro generates: rust-analyzer records no reference from inside an expansion (the
    entries of such functions are listed in `[parser_unindexed]`)
  * async bodies are followed as written; the state machine the compiler builds adds no edges
  * a parser reached only through another crate's code (a dependency that parses internally) is
    seen only if that dependency's entry point is itself in `parsers`

usage: scripts/api_boundary_parsers.py [--root DIR] [--index DIR]   (default index: target/scip)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

MIN_REASON = 12
SYM_RE = re.compile(r"^rust-analyzer cargo (\S+) \S+ (.*)$")
TRAIT_KEY_RE = re.compile(r"^<(\w+)(?:<[^>]*>)? for (\w+)(?:<[^>]*>)?>$")
IMPL_RE = re.compile(r"impl#\[([^\]]*)\](?:\[([^\]]*)\])?(\w+)\(\)\.$")
MEMBER_RE = re.compile(r"(\w+)#(\w+)\(\)\.$")
FREE_RE = re.compile(r"(?:^|/)(\w+)\(\)\.$")


def short(sym: str) -> str | None:
    """`rust-analyzer cargo <crate> <version> <descriptor>` -> `<crate> <descriptor>`"""
    m = SYM_RE.match(sym)
    return f"{m.group(1)} {m.group(2)}" if m else None


def _bare(t: str) -> str:
    return t.strip("`").split("<")[0]


def definitions(symbols, crate: str) -> dict[tuple[str | None, str], set[str]]:
    """(owner, name) -> symbols of this crate's functions: owner is the self type, `Trait for Type`
    for a trait impl method, the trait or type for a member, None for a free function"""
    out: dict[tuple[str | None, str], set[str]] = {}
    for s in symbols:
        sh = short(s)
        if not sh or not sh.startswith(crate + " ") or not sh.endswith(")."):
            continue
        d = sh[len(crate) + 1:]
        m = IMPL_RE.search(d)
        if m:
            ty = _bare(m.group(1))
            out.setdefault((ty, m.group(3)), set()).add(s)
            if m.group(2):
                out.setdefault((f"{_bare(m.group(2))} for {ty}", m.group(3)), set()).add(s)
            continue
        m = MEMBER_RE.search(d)
        if m:
            out.setdefault((m.group(1), m.group(2)), set()).add(s)
            continue
        m = FREE_RE.search(d)
        if m:
            out.setdefault((None, m.group(1)), set()).add(s)
    return out


def owner_and_name(key: str) -> tuple[str | None, str]:
    """`fn::a::b::f` -> (None, f), `fn::Type::m` -> (Type, m), `fn::<Tr for Ty>::m` -> (Tr for Ty, m)"""
    parts = key.split("::")[1:]
    name = parts[-1].split("#")[0]
    if len(parts) < 2:
        return None, name
    m = TRAIT_KEY_RE.match(parts[-2])
    if m:
        return f"{m.group(1)} for {m.group(2)}", name
    return (parts[-2] if parts[-2][:1].isupper() else None), name


FOLLOWED_KINDS = ("fn", "abi", "binding", "target")


def resolve(key: str, defs: dict, by_desc: dict[str, set[str]]) -> set[str]:
    """The index symbols of a manifest entry: `fn::a::b::f` / `abi::a::b::f` /
    `target::<target>::a::b::f` by their module path first (`a/b/f().`), then by owner and name;
    `binding::pyo3::f` / `binding::pyo3::T::m` by owner and name (the wrapped Rust function)"""
    kind, _, rest = key.partition("::")
    if kind == "target":
        rest = rest.partition("::")[2]
    elif kind == "binding":
        rest = rest.partition("::")[2]
    parts = rest.split("::")
    if kind != "binding":
        exact = by_desc.get("/".join(parts) + "().", set())
        if exact:
            return exact
    owner, name = owner_and_name("x::" + rest)
    syms = defs.get((owner, name), set())
    if not syms and owner is None:
        # rustdoc names a trait's method without its trait (`fn::decl_float`): every method of
        # that name in the crate (its declarations and impls) is followed
        syms = set().union(*(v for (o, n), v in defs.items() if n == name and o is not None))
    return syms


def match_path(path: str, index_shorts: set[str]) -> set[str]:
    """The index symbols a Rust path of the clippy configuration names: same crate, same item
    name, and the same type or trait when the path names one (`serde_json::Deserializer::new`
    matches `serde_json impl#[Deserializer<R>]new().`; a re-export such as `serde_json::from_str`
    matches `serde_json de/from_str().`). A trait method also matches its impls in any crate
    (`serde::de::Deserialize::deserialize` matches `x impl#[T][Deserialize]deserialize().`)."""
    segs = path.split("::")
    crate, name = segs[0], segs[-1]
    owner = segs[-2] if len(segs) >= 2 and segs[-2][:1].isupper() else None
    out = set()
    for sh in index_shorts:
        pkg, _, d = sh.partition(" ")
        if not d.endswith(f"{name}()."):
            continue
        m = IMPL_RE.search(d)
        if m and m.group(3) == name:
            if owner and m.group(2) and _bare(m.group(2)) == owner:
                out.add(sh)  # an impl of the trait method, in any crate
                continue
            if pkg.replace("-", "_") == crate and owner and _bare(m.group(1)) == owner:
                out.add(sh)
            continue
        if pkg.replace("-", "_") != crate:
            continue
        m = MEMBER_RE.search(d)
        if m and m.group(2) == name:
            if owner and m.group(1) == owner:
                out.add(sh)
            continue
        m = FREE_RE.search(d)
        if m and m.group(1) == name and owner is None:
            out.add(sh)
    return out


def judge(listed: dict[str, str], parsers: list[str] | None, exempt: dict[str, str],
          unindexed: dict[str, str], defs: dict, reach, index_shorts: set[str],
          stored_paths: list[str] = (), by_desc: dict[str, set[str]] | None = None):
    """Pure core (the tests drive it with a small graph). `reach(set of symbols)` returns every
    symbol reached from them. `parsers` is the hand list (SCIP form, each must be in the index);
    `stored_paths` are the Rust paths of the library clippy configuration's stored-form readers,
    matched against the index (match_path). Returns (problems, stats)."""
    by_desc = by_desc or {}
    problems: list[str] = []
    if parsers is None:
        return ["`parsers` is missing from scripts/api_boundary.toml (the rule was adopted here; "
                "dropping the list is not allowed)"], {}
    if not parsers:
        problems.append("`parsers` is empty (compared nothing)")
    for p in parsers:
        if p not in index_shorts:
            problems.append(f"parser `{p}` is not in the index (a typo, or the dependency changed)")
    derived: set[str] = set()
    for path in stored_paths:
        derived |= match_path(path, index_shorts)
    wanted = set(parsers) | derived
    for table, entries in (("parser_exempt", exempt), ("parser_unindexed", unindexed)):
        for key, reason in entries.items():
            if len(reason.strip()) < MIN_REASON:
                problems.append(f"[{table}] {key}: reason shorter than {MIN_REASON} characters")
            if key not in listed:
                problems.append(f"[{table}] {key}: not an entry of [boundary] (stale)")
    resolved = local_hits = exempted = 0
    not_followed: dict[str, int] = {}
    for key, decision in sorted(listed.items()):
        kind = key.split("::", 1)[0]
        if kind not in FOLLOWED_KINDS:
            # impl:: (a derive's generated body) and reexport:: (another crate's item) have no
            # body of this crate in the index: counted in the report, not followed
            not_followed[kind] = not_followed.get(kind, 0) + 1
            continue
        syms = resolve(key, defs, by_desc)
        if not syms:
            if key not in unindexed:
                problems.append(f"{key}: the index has no definition for it (add it to [parser_unindexed] "
                                "with the reason, e.g. a macro-generated body)")
            continue
        if key in unindexed:
            problems.append(f"[parser_unindexed] {key}: the index resolves it now (stale)")
        resolved += 1
        hit = sorted({short(s) for s in reach(syms)} & wanted)
        local = decision.split(":", 1)[0].strip() == "local"
        if key in exempt:
            if not local:
                problems.append(f"[parser_exempt] {key}: the entry is not `local` (stale)")
            elif not hit:
                problems.append(f"[parser_exempt] {key}: reaches no listed parser any more (stale)")
            else:
                exempted += 1
            continue
        if local and hit:
            local_hits += 1
            problems.append(f"{key}: decided `local` but reaches {', '.join(f'`{h}`' for h in hit)} "
                            "(decide handshake or pending, or exempt it with the reason)")
    if resolved == 0:
        problems.append("no entry resolved in the index (compared nothing)")
    if stored_paths and not derived:
        problems.append("no stored-form reader of the clippy configuration matched the index "
                        "(the matching or the configuration is broken)")
    return problems, {"resolved": resolved, "unindexed": len(unindexed), "exempt": exempted,
                      "parsers": len(parsers), "stored_paths": len(stored_paths),
                      "stored_matched": len(derived), "local_hits": local_hits,
                      "not_followed": not_followed}


def stored_form_readers(root: Path) -> tuple[list[str], str | None]:
    """(paths, problem): the stored-form readers the library clippy configuration bans, read with
    scripts/clippy_lib_hidden_check.py's rule (entries whose reason starts with its prefix); a
    configuration that names none is a problem (the serde / bincode half of the parsers would be
    empty and every local serde reader would pass)"""
    sys.path.insert(0, str(root / "scripts"))
    import clippy_lib_hidden_check  # noqa: E402

    config = root / ".github" / "clippy-lib" / "clippy.toml"
    if not config.is_file():
        return [], f"{config}: missing (the stored-form readers are read from it)"
    stored = sorted(clippy_lib_hidden_check.stored_form_paths(config.read_text(encoding="utf-8")))
    if not stored:
        return [], "the library clippy configuration names no stored-form reader"
    return stored, None


def index_commit(root: Path, index: Path) -> tuple[str | None, str | None]:
    """(problem, commit of the index)"""
    f = index / "commit"
    if not f.is_file():
        return f"{f}: missing (rebuild the index with scripts/scip_index.sh)", None
    recorded = f.read_text(encoding="utf-8").split()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if not recorded or recorded[0] != head:
        return (f"the index was built from {recorded[0] if recorded else '?'}, HEAD is {head} "
                "(rebuild it from the checked commit)"), (recorded[0] if recorded else None)
    if "dirty" in recorded[1:]:
        return "the index was built from a tree with uncommitted changes", recorded[0]
    return None, recorded[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--index", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    index = Path(a.index) if a.index else root / "target" / "scip"
    cfg = tomllib.loads((root / "scripts" / "api_boundary.toml").read_text(encoding="utf-8"))
    crate = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["package"]["name"]
    stored, problem = stored_form_readers(root)
    if problem:
        print(f"api_boundary_parsers: {problem}", file=sys.stderr)
        return 1
    problem, commit = index_commit(root, index)
    if problem:
        print(f"api_boundary_parsers: {problem}", file=sys.stderr)
        return 1
    scips = [index / "native.scip", index / "wasm.scip"]
    for p in scips:
        if not p.is_file() or p.stat().st_size == 0:
            print(f"api_boundary_parsers: {p} is missing or empty", file=sys.stderr)
            return 1
    sys.path.insert(0, str(root / "scripts"))
    import scip_reach  # noqa: E402

    an = scip_reach.analyze(root, scips, keep_graph=True)
    # every symbol of the index: the definitions (a trait's provided method nobody calls is
    # still an entry) and every referenced symbol (the parsers of other crates)
    every: set[str] = set()
    for p in scips:
        for doc in scip_reach.load_scip(p):
            every.update(o["symbol"] for o in doc["occ"] if o["symbol"] and not o["symbol"].startswith("local "))
    defs = definitions(every, crate)
    by_desc: dict[str, set[str]] = {}
    for sym in every:
        sh = short(sym)
        if sh and sh.startswith(crate + " "):
            by_desc.setdefault(sh[len(crate) + 1:], set()).add(sym)
    problems, stats = judge(cfg.get("boundary", {}), cfg.get("parsers"), cfg.get("parser_exempt", {}),
                            cfg.get("parser_unindexed", {}), defs, an.reach,
                            {s for s in map(short, every) if s}, stored, by_desc)
    for p in problems:
        print(f"api_boundary_parsers: {p}", file=sys.stderr)
    if problems:
        return 1
    nf = ", ".join(f"{v} {k}" for k, v in sorted(stats["not_followed"].items()))
    print(f"api_boundary_parsers: index {commit}: {stats['resolved']} entries followed (fn / abi / binding / "
          f"target), not followed: {nf or 'none'}; parsers: {stats['parsers']} listed + "
          f"{stats['stored_paths']} stored-form readers of the clippy configuration "
          f"({stats['stored_matched']} index symbols); {stats['exempt']} exempt, "
          f"{stats['unindexed']} unindexed, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
