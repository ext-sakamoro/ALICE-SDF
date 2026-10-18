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
`fn::` entry of the manifest, and fails when an entry decided `local` reaches one of the parsers the
manifest declares, unless the manifest exempts that entry with a reason (`[parser_exempt]`, for a
function that parses only what this crate wrote, or only reads the identifier itself).

Manifest (`scripts/api_boundary.toml`):
  parsers = ["<crate> <descriptor>", ...]   the parse entry points, in SCIP form without the
                                            version (`serde_json de/from_slice().`); each must be
                                            in the index (a name it cannot find is a typo, not a skip)
  [parser_exempt]    "<fn key>" = "<reason>"  a `local` entry that reaches a parser on purpose
  [parser_unindexed] "<fn key>" = "<reason>"  an entry the index has no definition for
Absent `parsers` fails: this repository adopted the rule, so dropping the list is not allowed (the
other repositories that share api_boundary.py do not run this script).

Fails on: a `local` entry reaching a parser without an exemption; an exemption that is stale (the
entry is gone, is not `local`, or no longer reaches a parser); a listed parser the index does not
contain; an empty parser list; an entry the index cannot resolve that is not in `[parser_unindexed]`,
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


def judge(listed: dict[str, str], parsers: list[str] | None, exempt: dict[str, str],
          unindexed: dict[str, str], defs: dict, reach, index_shorts: set[str]):
    """Pure core (the tests drive it with a small graph). `reach(set of symbols)` returns every
    symbol reached from them. Returns (problems, stats)."""
    problems: list[str] = []
    if parsers is None:
        return ["`parsers` is missing from scripts/api_boundary.toml (the rule was adopted here; "
                "dropping the list is not allowed)"], {}
    if not parsers:
        problems.append("`parsers` is empty (compared nothing)")
    for p in parsers:
        if p not in index_shorts:
            problems.append(f"parser `{p}` is not in the index (a typo, or the dependency changed)")
    wanted = set(parsers)
    for table, entries in (("parser_exempt", exempt), ("parser_unindexed", unindexed)):
        for key, reason in entries.items():
            if len(reason.strip()) < MIN_REASON:
                problems.append(f"[{table}] {key}: reason shorter than {MIN_REASON} characters")
            if key not in listed:
                problems.append(f"[{table}] {key}: not an entry of [boundary] (stale)")
    resolved = local_hits = exempted = 0
    for key, decision in sorted(listed.items()):
        if not key.startswith("fn::"):
            continue
        owner, name = owner_and_name(key)
        syms = defs.get((owner, name), set())
        if not syms and owner is None:
            # rustdoc names a trait's method without its trait (`fn::decl_float`): every method of
            # that name in the crate (its declarations and impls) is followed
            syms = set().union(*(v for (o, n), v in defs.items() if n == name and o is not None))
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
        problems.append("no fn entry resolved in the index (compared nothing)")
    return problems, {"resolved": resolved, "unindexed": len(unindexed), "exempt": exempted,
                      "parsers": len(parsers), "local_hits": local_hits}


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
    problems, stats = judge(cfg.get("boundary", {}), cfg.get("parsers"), cfg.get("parser_exempt", {}),
                            cfg.get("parser_unindexed", {}), defs, an.reach,
                            {s for s in map(short, every) if s})
    for p in problems:
        print(f"api_boundary_parsers: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"api_boundary_parsers: index {commit}: {stats['resolved']} fn entries followed, "
          f"{stats['parsers']} parsers, {stats['exempt']} exempt, {stats['unindexed']} unindexed, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
