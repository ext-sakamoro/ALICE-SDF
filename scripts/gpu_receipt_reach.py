#!/usr/bin/env python3
"""GPU receipt reach: the GPU receipt rule (scripts/gpu-receipt-suites.txt) must cover every crate file the code under
its hand-kept paths reaches.

The rule's hand-kept paths name the sheet field, its tests and examples, the dependency versions and the rule with
its runner. Every other crate file those Rust files reach, transitively, is listed in the rule's generated reach
block, and this check fails when the block differs from the walk in either direction (a reached file missing would
land without the GPU receipt, the rule failing open; a listed file no longer reached is stale), or when the walk
reads no file. `--write-reach` rewrites the block from the walk.

The walk follows `use` trees and inline paths (`crate::`, `self::`, `super::`, and the crate's own name from tests
and examples), `pub use` re-exports to the defining file, and `include_str!` / `include_bytes!`. Comments and string
literals are not uses. The crate root (src/lib.rs) is never a reached file: it declares every module, so its
re-exports are followed instead. A path that names no file of the crate (std, another crate) is not followed.

run: python3 scripts/gpu_receipt_reach.py [--root .] [--write-reach]
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

GPU_RECEIPT_RULE = 'scripts/gpu-receipt-suites.txt'


class RuleError(ValueError):
    """the rule's reach block cannot be read"""


def rule_glob_match(pattern: str, path: str) -> bool:
    """the receipt rule's glob (the landing tool's GPU receipt rule): `**` spans directories, `*` does not"""
    rx, i = '', 0
    while i < len(pattern):
        if pattern.startswith('**', i):
            rx, i = rx + '.*', i + 2
        elif pattern[i] == '*':
            rx, i = rx + '[^/]*', i + 1
        else:
            rx, i = rx + re.escape(pattern[i]), i + 1
    return re.fullmatch(rx, path) is not None


def _rust_code(src: str) -> str:
    """the source with comments and string literals blanked (a path in a doc or a message is not a use)"""
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    src = re.sub(r'//[^\n]*', ' ', src)
    return re.sub(r'r#*"(?:.|\n)*?"#*|"(?:\\.|[^"\\])*"', '""', src)


def _split_top(text: str) -> list[str]:
    out, buf, depth = [], '', 0
    for ch in text:
        depth += (ch == '{') - (ch == '}')
        if ch == ',' and depth == 0:
            out.append(buf)
            buf = ''
        else:
            buf += ch
    return [b for b in (out + [buf]) if b.strip()]


def _use_paths(text: str) -> list[tuple[list[str], str]]:
    """(segments, the name it brings in) for each leaf of a use tree such as `crate::a::{b, c::{d as e, *}}`"""
    text = re.sub(r'\s+', ' ', text).strip()
    if '{' in text:
        i = text.index('{')
        head = [x for x in text[:i].replace(' ', '').split('::') if x]
        inner = text[i + 1:text.rindex('}')]
        return [(head + segs, name) for part in _split_top(inner) for segs, name in _use_paths(part)]
    m = re.fullmatch(r'(.*?)(?: as (\w+))?', text)
    segs = [x for x in m.group(1).replace(' ', '').split('::') if x]
    if segs and segs[-1] == 'self':
        segs = segs[:-1]
    return [(segs, m.group(2) or (segs[-1] if segs else ''))]


class _Reach:
    """the files a Rust source file reaches through `crate::` / `self::` / `super::` paths (and, from tests and
    examples, the crate's own name), following `pub use` re-exports to the defining file and `include_str!` /
    `include_bytes!` to the included file. The crate root (src/lib.rs) is never a reached file: it declares every
    module, so its re-exports are followed instead. A path that names no file of the crate (std, another crate) is
    not followed."""

    def __init__(self, root: Path, crate: str):
        self.root, self.src, self.crate = root, root / 'src', crate
        self.lib = self.src / 'lib.rs'
        self._code: dict[Path, str] = {}

    def code(self, f: Path) -> str:
        if f not in self._code:
            self._code[f] = _rust_code(f.read_text(encoding='utf-8'))
        return self._code[f]

    def mod_dir(self, f: Path) -> Path:
        return f.parent if f.name in ('mod.rs', 'lib.rs') else f.parent / f.stem

    def module_file(self, d: Path) -> Path | None:
        if d == self.src:
            return self.lib
        for c in (d.parent / f'{d.name}.rs', d / 'mod.rs'):
            if c.is_file():
                return c
        return None

    def base(self, f: Path, segs: list[str], in_src: bool) -> tuple[Path, list[str]] | None:
        if not segs:
            return None
        if segs[0] == self.crate or (in_src and segs[0] == 'crate'):
            return self.src, segs[1:]
        if not in_src:
            return None
        if segs[0] == 'self':
            return self.mod_dir(f), segs[1:]
        if segs[0] == 'super':
            d = self.mod_dir(f)
            while segs and segs[0] == 'super':
                d, segs = d.parent, segs[1:]
            return d, segs
        return None

    def resolve(self, d: Path, segs: list[str], seen: set) -> set[Path]:
        key = (d, tuple(segs))
        if key in seen:
            return set()
        seen.add(key)
        out: set[Path] = set()
        cur, i = d, 0
        while i < len(segs):
            nxt = next((c for c in (cur / f'{segs[i]}.rs', cur / segs[i] / 'mod.rs') if c.is_file()), None)
            if nxt is None:
                break
            out.add(nxt)
            cur, i = cur / segs[i], i + 1
        if i < len(segs) and segs[i] != '*':
            m = self.module_file(cur)
            if m is not None:
                out |= self.reexport(m, segs[i], seen)
        out.discard(self.lib)
        return out

    def reexport(self, m: Path, name: str, seen: set) -> set[Path]:
        out: set[Path] = set()
        for stmt in re.finditer(r'\bpub(?:\s*\([^)]*\))?\s+use\s+([^;]+);', self.code(m)):
            for segs, brought in _use_paths(stmt.group(1)):
                if brought == name or (segs and segs[-1] == '*'):
                    full = segs[:-1] + [name] if segs and segs[-1] == '*' else segs
                    b = self.base(m, full, True)
                    if b is None and full:                     # a 2018 path relative to the module (`pub use child::X`)
                        b = (self.mod_dir(m), full)
                    if b is not None:
                        out |= self.resolve(b[0], b[1], seen)
        return out

    def of(self, f: Path) -> set[Path]:
        in_src = f.is_relative_to(self.src)
        code = self.code(f)
        out: set[Path] = set()
        names = 'crate|self|super|' + re.escape(self.crate) if in_src else re.escape(self.crate)
        for m in re.finditer(r'\buse\s+([^;]+);', code):
            for segs, _ in _use_paths(m.group(1)):
                b = self.base(f, segs, in_src)
                if b is not None:
                    out |= self.resolve(b[0], b[1], set())
        for m in re.finditer(r'(?<![\w:])(%s)((?:\s*::\s*\w+)+)' % names, code):
            segs = [m.group(1)] + [x for x in re.sub(r'\s+', '', m.group(2)).split('::') if x]
            b = self.base(f, segs, in_src)
            if b is not None:
                out |= self.resolve(b[0], b[1], set())
        raw = f.read_text(encoding='utf-8')
        for m in re.finditer(r'include_(?:str|bytes)!\s*\(\s*"([^"]+)"\s*\)', raw):
            inc = (f.parent / m.group(1)).resolve()
            if inc.is_file() and inc.is_relative_to(self.root.resolve()):
                out.add(self.root / inc.relative_to(self.root.resolve()))
        return out


REACH_BEGIN = '# reach: generated by `python3 scripts/gpu_receipt_reach.py --write-reach`, do not edit by hand'
REACH_END = '# end reach'


def _rule_paths(text: str) -> tuple[list[str], list[str] | None]:
    """(hand-kept path globs, generated reach paths or None when the rule has no reach block)

    A marker line that is not exactly REACH_BEGIN / REACH_END, a second block, or a block left open is a RuleError:
    the block's lines would otherwise be read as hand-kept paths and nothing would be compared."""
    seeds: list[str] = []
    block: list[str] | None = None
    inside = False
    for n, raw in enumerate(text.splitlines(), 1):
        mark = raw.strip()
        if mark == REACH_BEGIN:
            if block is not None:
                raise RuleError(f'{GPU_RECEIPT_RULE}:{n}: a second reach block')
            inside, block = True, []
            continue
        if mark == REACH_END:
            if not inside:
                raise RuleError(f'{GPU_RECEIPT_RULE}:{n}: {REACH_END!r} without {REACH_BEGIN!r}')
            inside = False
            continue
        if re.match(r'#\s*(end\s+)?reach\b', mark):
            raise RuleError(f'{GPU_RECEIPT_RULE}:{n}: not the reach marker: {mark!r} (want {REACH_BEGIN!r} or '
                            f'{REACH_END!r})')
        line = raw.split('#', 1)[0].strip()
        parts = line.split()
        if len(parts) == 2 and parts[0] == 'path':
            (block if inside else seeds).append(parts[1])
    if inside:
        raise RuleError(f'{GPU_RECEIPT_RULE}: the reach block is not closed by {REACH_END!r}')
    return seeds, block


def reach_closure(root: Path, crate: str, seeds: list[str]) -> tuple[list[str], int, list[str]]:
    """(the crate files reached transitively from the Rust files under the seed paths and not under a seed path,
    the number of seed files walked, problems of the walk itself)"""
    def under(rel: str) -> bool:
        return any(rule_glob_match(g, rel) for g in seeds)

    start = sorted(p for d in ('src', 'tests', 'examples') for p in (root / d).rglob('*.rs')
                   if under(p.relative_to(root).as_posix()))
    reach = _Reach(root, crate)
    seen, todo, extra = set(start), list(start), set()
    while todo:
        cur = todo.pop()
        if cur.suffix != '.rs':
            continue
        for r in reach.of(cur):
            rel = r.relative_to(root).as_posix()
            if not under(rel):
                extra.add(rel)
            if r not in seen:
                seen.add(r)
                todo.append(r)
    problems = [] if start else [f'gpu_receipt_reach: no Rust file under the paths of {GPU_RECEIPT_RULE} '
                                 f'(the walk read nothing)']
    return sorted(extra), len(start), problems


def gpu_receipt_reach(root: Path, crate: str) -> tuple[list[str], int]:
    """(problems, files walked): the GPU receipt rule must cover every crate file reached, transitively, from the
    Rust files under its hand-kept paths (`use crate::…`, `self::`, `super::`, the crate's name from tests and
    examples, `pub use` re-exports to the defining file, `include_str!` / `include_bytes!`). Those reached files
    are listed in the rule's generated reach block, which must equal the walk exactly: a reached file missing from
    it would land without the GPU receipt (the rule failing open), and a listed file no longer reached is stale.
    Without a reach block every reached file must be under a hand-kept path. 0 files walked with a rule present is
    a problem."""
    f = root / GPU_RECEIPT_RULE
    if not f.is_file():
        return [], 0
    try:
        seeds, block = _rule_paths(f.read_text(encoding='utf-8'))
    except RuleError as e:
        return [f'gpu_receipt_reach: {e}'], 0
    extra, walked, problems = reach_closure(root, crate, seeds)
    have = set(block or [])
    for rel in extra:
        if rel not in have:
            problems.append(f'gpu_receipt_reach: {rel} is reached from the files under {GPU_RECEIPT_RULE} but is '
                            f'under none of its paths (a change there would land without the GPU receipt); '
                            f'run --write-reach')
    for rel in sorted(have - set(extra)):
        problems.append(f'gpu_receipt_reach: {rel} is in the reach block of {GPU_RECEIPT_RULE} but is no longer '
                        f'reached (stale); run --write-reach')
    return problems, walked


def write_reach(root: Path, crate: str) -> int:
    """rewrite the rule's reach block from the walk; returns the number of listed files"""
    f = root / GPU_RECEIPT_RULE
    text = f.read_text(encoding='utf-8')
    seeds, _ = _rule_paths(text)
    extra, walked, problems = reach_closure(root, crate, seeds)
    if problems:
        raise SystemExit('\n'.join(problems))
    lines, out, inside = text.splitlines(), [], False
    for raw in lines:
        if raw.strip() == REACH_BEGIN:
            inside = True
            continue
        if raw.strip() == REACH_END:
            inside = False
            continue
        if not inside:
            out.append(raw)
    k = max((n for n, ln in enumerate(out) if ln.split('#', 1)[0].strip().startswith('path ')), default=len(out) - 1)
    out[k + 1:k + 1] = [REACH_BEGIN] + [f'path {rel}' for rel in extra] + [REACH_END]
    f.write_text('\n'.join(out) + '\n', encoding='utf-8')
    return len(extra)


def crate_name(root: Path) -> str:
    with open(root / 'Cargo.toml', 'rb') as f:
        return tomllib.load(f)['package']['name'].replace('-', '_')


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--write-reach', action='store_true',
                    help=f'rewrite the reach block of {GPU_RECEIPT_RULE} from the walk, then check')
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if not (root / GPU_RECEIPT_RULE).is_file():
        print(f'gpu-receipt-reach: no {GPU_RECEIPT_RULE}, nothing to check')
        return 0
    crate = crate_name(root)
    if a.write_reach:
        n = write_reach(root, crate)
        print(f'gpu-receipt-reach: reach block of {GPU_RECEIPT_RULE} rewritten, {n} file(s)')
    problems, walked = gpu_receipt_reach(root, crate)
    for p in problems:
        print(p, file=sys.stderr)
    print(f'gpu-receipt-reach: {walked} file(s) under the hand-kept paths walked, {len(problems)} problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
