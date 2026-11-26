#!/usr/bin/env python3
"""CI の `cargo test` が tests/*.rs の feature 付き部分を実際に compile して走らせているかを検査する.

`#![cfg(feature = "x")]` を先頭に持つ test file は、feature x を付けずに `cargo test --tests`
すると空の binary になり、`running 0 tests` のまま green になる (2026-10-04 実測:
`npr_shader_validate` 19 本と `test_physics_bridge_determinism` 9 本が 3 OS すべてで 0 本だった).
関数単位の `#[cfg(feature = "x")]` も同じで、test 数が変わらず中の比較 arm だけが消える.

検査内容:
  1. tests/*.rs の各 file を走らせる `cargo test` が CI に 1 つ以上ある
     (file 先頭の `#![cfg(...)]` を満たす feature 集合で)
  2. file 内の `#[cfg(...)]` のうち feature を含むものそれぞれに、それを満たす feature 集合で
     その file を走らせる `cargo test` が 1 つ以上ある

CI 側は push / pull_request で起動する .github/workflows/*.yml の `run:` から `cargo test` を読む.
`working-directory` が repo 直下でない step と、`-p` / `--manifest-path` で別 package を指す
step は repo 直下の tests/ を走らせないので数えない.

検査対象 (test file / cargo test 呼び出し) が 0 件なら fail する (空振りを green にしない).

run: python3 scripts/ci_test_coverage_check.py [--root .]
"""
from __future__ import annotations

import argparse
import re
import shlex
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- cfg 式


def _tokenize_cfg(s: str) -> list[str]:
    toks = re.findall(r'"(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*|[(),=]', s)
    return toks


def parse_cfg(expr: str):
    """`all(feature = "a", not(unix))` → ('all', [('feature', 'a'), ('not', ('atom', 'unix'))])."""
    toks = _tokenize_cfg(expr)
    pos = 0

    def node():
        nonlocal pos
        name = toks[pos]
        pos += 1
        if name in ('all', 'any', 'not') and pos < len(toks) and toks[pos] == '(':
            pos += 1
            kids = []
            while toks[pos] != ')':
                kids.append(node())
                if toks[pos] == ',':
                    pos += 1
            pos += 1
            return ('not', kids[0]) if name == 'not' else (name, kids)
        if pos < len(toks) and toks[pos] == '=':
            value = toks[pos + 1].strip('"')
            pos += 2
            return ('feature', value) if name == 'feature' else ('kv', name, value)
        return ('atom', name)

    tree = node()
    if pos != len(toks):
        raise ValueError(f'cfg 式を読み切れない: {expr!r}')
    return tree


def eval_cfg(tree, features: set[str]):
    """True / False / None (feature 以外の述語 = OS など、満たしうるとみなす)."""
    kind = tree[0]
    if kind == 'feature':
        return tree[1] in features
    if kind in ('atom', 'kv'):
        return None
    if kind == 'not':
        v = eval_cfg(tree[1], features)
        return None if v is None else not v
    vals = [eval_cfg(k, features) for k in tree[1]]
    if kind == 'all':
        if False in vals:
            return False
        return True if all(v is True for v in vals) else None
    # any
    if True in vals:
        return True
    return None if None in vals else False


def mentions_gpu(tree) -> bool:
    """Whether a cfg tree asks for the `gpu` feature (a test or arm that needs a device); `not(feature = "gpu")` is
    the arm for builds without it and does not count."""
    if tree[0] == 'feature':
        return tree[1] == 'gpu'
    if tree[0] == 'not':
        return False
    if tree[0] in ('all', 'any'):
        return any(mentions_gpu(k) for k in tree[1])
    return False


REQUIRE_GPU = 'ALICE_SDF_REQUIRE_GPU'
# the variables that make a test fail (not skip) without its device: any adapter, or the Metal one
REQUIRE_VARS = (REQUIRE_GPU, 'ALICE_SDF_REQUIRE_METAL')


def mentions_feature(tree) -> bool:
    if tree[0] == 'feature':
        return True
    if tree[0] == 'not':
        return mentions_feature(tree[1])
    if tree[0] in ('all', 'any'):
        return any(mentions_feature(k) for k in tree[1])
    return False


# ---------------------------------------------------------------- tests/*.rs


@dataclass
class TestFile:
    path: Path
    file_cfgs: list = field(default_factory=list)       # [(line, expr, tree)]
    item_cfgs: list = field(default_factory=list)       # [(line, expr, tree)]


_CFG_RE = re.compile(r'^\s*#(!?)\[cfg\((.*)\)\]\s*$', re.S)


def scan_test_file(path: Path) -> TestFile:
    tf = TestFile(path)
    lines = path.read_text(encoding='utf-8').split('\n')
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('//') or not stripped.startswith(('#[cfg(', '#![cfg(')):
            i += 1
            continue
        start = i
        buf = stripped
        while buf.count('[') > buf.count(']') and i + 1 < len(lines):
            i += 1
            buf += ' ' + lines[i].strip()
        i += 1
        m = _CFG_RE.match(buf)
        if not m:
            continue
        expr = m.group(2)
        tree = parse_cfg(expr)
        if m.group(1) == '!':
            tf.file_cfgs.append((start + 1, expr, tree))
        elif mentions_feature(tree):
            tf.item_cfgs.append((start + 1, expr, tree))
    return tf


# ---------------------------------------------------------------- Cargo.toml


@dataclass
class Package:
    name: str
    features: dict[str, list[str]]

    def expand(self, names) -> set[str]:
        out: set[str] = set()
        stack = list(names)
        while stack:
            f = stack.pop()
            if f in out or f.startswith('dep:') or '/' in f:
                continue
            out.add(f)
            stack.extend(self.features.get(f, []))
        return out


def load_package(cargo_toml: Path) -> Package:
    data = tomllib.loads(cargo_toml.read_text(encoding='utf-8'))
    return Package(data['package']['name'], data.get('features', {}))


# ---------------------------------------------------------------- workflows


@dataclass
class Invocation:
    where: str
    args: list[str]

    no_default: bool = False
    all_features: bool = False
    features: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    any_tests: bool = False        # --tests / --all-targets
    other_target: bool = False     # --lib / --bin(s) / --example(s) / --bench(es)
    doc: bool = False
    other_package: bool = False
    no_run: bool = False           # --no-run: compiles the tests, runs none
    require_gpu: bool = False      # a REQUIRE_VARS variable = 1 in the step's effective env (job, step or command)


def parse_invocation(where: str, args: list[str], package: Package, require_gpu: bool = False) -> Invocation:
    inv = Invocation(where, args)
    inv.require_gpu = require_gpu
    it = iter(args)
    for a in it:
        if a == '--':
            break
        key, _, val = a.partition('=')
        if key in ('--features', '-F'):
            v = val if val else next(it, '')
            inv.features += [f for f in re.split(r'[\s,]+', v) if f]
        elif a == '--all-features':
            inv.all_features = True
        elif a == '--no-default-features':
            inv.no_default = True
        elif key == '--test':
            inv.tests.append(val if val else next(it, ''))
        elif a in ('--tests', '--all-targets'):
            inv.any_tests = True
        elif a in ('--lib', '--bins', '--examples', '--benches'):
            inv.other_target = True
        elif key in ('--bin', '--example', '--bench'):
            inv.other_target = True
            if not val:
                next(it, '')
        elif a == '--doc':
            inv.doc = True
        elif a == '--no-run':
            inv.no_run = True
        elif key in ('-p', '--package'):
            if (val if val else next(it, '')) != package.name:
                inv.other_package = True
        elif key == '--manifest-path':
            p = val if val else next(it, '')
            if Path(p).as_posix() not in ('Cargo.toml', './Cargo.toml'):
                inv.other_package = True
    return inv


def runs_file(inv: Invocation, stem: str) -> bool:
    # `cargo test --no-run` builds the test binaries and executes none of them: it runs no file
    if inv.other_package or inv.doc or inv.no_run:
        return False
    if inv.tests:
        return stem in inv.tests
    if inv.any_tests:
        return True
    return not inv.other_target


def enabled_features(inv: Invocation, package: Package) -> set[str]:
    if inv.all_features:
        return package.expand(package.features.keys())
    base = [] if inv.no_default else ['default']
    return package.expand(base + inv.features)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(' '))


def _block(lines: list[str], i: int, parent_indent: int) -> tuple[list[str], int]:
    """lines[i:] のうち parent_indent より深い行 (空行を含む) を返す."""
    out = []
    while i < len(lines) and (not lines[i].strip() or _indent(lines[i]) > parent_indent):
        out.append(lines[i])
        i += 1
    while out and not out[-1].strip():
        out.pop()
    return out, i


def _scalar(lines: list[str], i: int, key_indent: int, value: str) -> tuple[str, int]:
    """`key: value` の value (block scalar `|` / `>` は続く行を畳む)."""
    value = value.strip()
    if value[:1] in ('|', '>'):
        body, j = _block(lines, i + 1, key_indent)
        texts = [ln.strip() for ln in body]
        if value.startswith('>'):
            # 折り畳み: 空行以外は空白で繋ぐ
            return ' '.join(t for t in texts if t), j
        return '\n'.join(texts), j
    if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
        value = value[1:-1]
    return value, i + 1


def _env_of(lines: list[str], key_indent: int) -> dict[str, str]:
    """The `env:` mapping written at `key_indent` in `lines` (a job's or a step's own keys)."""
    for i, ln in enumerate(lines):
        if _indent(ln) == key_indent and re.match(r'^\s*env:\s*$', ln):
            body, _ = _block(lines, i + 1, key_indent)
            out = {}
            for b in body:
                m = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$', b)
                if m:
                    out[m.group(1)] = m.group(2).strip().strip('"\'')
            return out
    return {}


def _working_dir_of(lines: list[str]) -> str | None:
    for ln in lines:
        m = re.match(r'^\s*working-directory:\s*(.*)$', ln)
        if m:
            return m.group(1).strip().strip('"\'')
    return None


def _is_root(wd: str | None) -> bool:
    return wd is None or wd.rstrip('/') in ('', '.', './', '${{ github.workspace }}')


def _triggered_on_push_or_pr(lines: list[str]) -> bool:
    for i, ln in enumerate(lines):
        m = re.match(r'^(on|"on"|\'on\'|true):\s*(.*)$', ln)
        if not m:
            continue
        rest = m.group(2).strip()
        if rest:
            return bool(re.search(r'\b(push|pull_request)\b', rest))
        body, _ = _block(lines, i + 1, 0)
        return any(re.match(r'^\s{2}(push|pull_request)\s*:', b) for b in body)
    return False


def _commands(script: str) -> list[str]:
    script = re.sub(r'\\\n\s*', ' ', script)       # bash の行継続
    script = re.sub(r'`\n\s*', ' ', script)         # pwsh の行継続
    out = []
    for line in script.split('\n'):
        out += [c.strip() for c in re.split(r'&&|\|\||;', line) if c.strip()]
    return out


def scan_workflow(path: Path, package: Package) -> list[Invocation]:
    lines = path.read_text(encoding='utf-8').split('\n')
    if not _triggered_on_push_or_pr(lines):
        return []
    top_wd = None
    jobs_at = None
    for i, ln in enumerate(lines):
        if re.match(r'^defaults:\s*$', ln):
            body, _ = _block(lines, i + 1, 0)
            top_wd = _working_dir_of(body)
        if re.match(r'^jobs:\s*$', ln):
            jobs_at = i
    if jobs_at is None:
        return []

    invs: list[Invocation] = []
    jobs_body, _ = _block(lines, jobs_at + 1, 0)
    offset = jobs_at + 1
    job_indent = min(_indent(l) for l in jobs_body if l.strip())
    j = 0
    while j < len(jobs_body):
        ln = jobs_body[j]
        if not ln.strip() or _indent(ln) != job_indent:
            j += 1
            continue
        job_lines, k = _block(jobs_body, j + 1, job_indent)
        job_start = offset + j + 1
        job_wd = top_wd
        job_key = min((_indent(l) for l in job_lines if l.strip()), default=job_indent + 2)
        job_env = _env_of(job_lines, job_key)
        steps_at = None
        for n, jl in enumerate(job_lines):
            if re.match(r'^\s*defaults:\s*$', jl):
                body, _ = _block(job_lines, n + 1, _indent(jl))
                job_wd = _working_dir_of(body) or job_wd
            if re.match(r'^\s*steps:\s*$', jl) and steps_at is None:
                steps_at = n
        if steps_at is not None:
            steps_indent = _indent(job_lines[steps_at])
            steps, _ = _block(job_lines, steps_at + 1, steps_indent - 1)
            item_indent = None
            s = 0
            while s < len(steps):
                sl = steps[s]
                if re.match(r'^\s*-\s', sl) and (item_indent is None or _indent(sl) == item_indent):
                    item_indent = _indent(sl)
                    body, s2 = _block(steps, s + 1, item_indent)
                    first = ' ' * (item_indent + 2) + sl.strip()[1:].lstrip()
                    step = [first] + body
                    step_line = job_start + steps_at + 1 + s
                    wd = _working_dir_of(step) or job_wd
                    env = {**job_env, **_env_of(step, item_indent + 2)}
                    _collect_runs(step, item_indent + 2, wd, f'{path.name}:{step_line}', package, invs,
                                  any(env.get(v) == '1' for v in REQUIRE_VARS))
                    s = s2
                else:
                    s += 1
        j = k
    return invs


def _collect_runs(step: list[str], key_indent: int, wd, where: str, package: Package, invs: list,
                  require_gpu: bool = False) -> None:
    i = 0
    while i < len(step):
        ln = step[i]
        m = re.match(r'^(\s*)run:\s*(.*)$', ln)
        if not m or _indent(ln) != key_indent:
            i += 1
            continue
        script, i = _scalar(step, i, key_indent, m.group(2))
        if not _is_root(wd):
            continue
        for cmd in _commands(script):
            try:
                toks = shlex.split(cmd)
            except ValueError:
                toks = cmd.split()
            for t in range(len(toks) - 1):
                if toks[t] == 'cargo' and toks[t + 1] == 'test':
                    # a VAR=value prefix on the command sets the env for this command only
                    prefix = {a.split('=', 1)[0]: a.split('=', 1)[1] for a in toks[:t] if re.match(r'^[A-Za-z_]\w*=', a)}
                    req = require_gpu or any(prefix.get(v) == '1' for v in REQUIRE_VARS)
                    invs.append(parse_invocation(where, toks[t + 2:], package, req))
                    break


# ---------------------------------------------------------------- 突合


GPU_RECEIPT_RULE = 'scripts/gpu-receipt-suites.txt'


def gpu_receipt_suites(root: Path) -> tuple[dict[str, int], list[str]]:
    """({suite: min}, problems) from the repository's GPU receipt rule (absent: no suites, no problems).

    A suite listed there runs on a real GPU machine and is held at landing by the GPU receipt rule of the CI evidence
    receipt (the landing tool's GPU receipt rule, which reads the same file): the landing of a change under the
    file's paths needs a receipt for the landing tree with every listed suite at its min. A listed suite therefore
    counts here as run with every feature of the crate and the device required. The file must be one that rule can
    apply: at least one path line and one suite line, each min a positive integer; otherwise it is a finding."""
    f = root / GPU_RECEIPT_RULE
    if not f.is_file():
        return {}, []
    paths, suites, problems = 0, {}, []
    for n, raw in enumerate(f.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == 'path' and len(parts) == 2:
            paths += 1
        elif parts[0] == 'suite' and len(parts) == 3 and re.fullmatch(r'[1-9][0-9]*', parts[2]) and parts[1] not in suites:
            suites[parts[1]] = int(parts[2])
        else:
            problems.append(f'gpu_receipt_rule: {GPU_RECEIPT_RULE}:{n}: not a `path <glob>` or `suite <name> <min>` line '
                            f'with a positive min and a suite listed once: {raw.strip()!r}')
    if not paths:
        problems.append(f'gpu_receipt_rule: {GPU_RECEIPT_RULE}: no path line (the receipt rule would hold no change)')
    if not suites:
        problems.append(f'gpu_receipt_rule: {GPU_RECEIPT_RULE}: no suite line')
    for name in suites:
        if not (root / 'tests' / f'{name}.rs').is_file():
            problems.append(f'gpu_receipt_rule: {GPU_RECEIPT_RULE}: suite {name} has no tests/{name}.rs')
    return ({} if problems else suites), problems


def check(root: Path) -> tuple[list[str], dict]:
    package = load_package(root / 'Cargo.toml')
    files = [scan_test_file(p) for p in sorted((root / 'tests').glob('*.rs'))]
    invs: list[Invocation] = []
    wf_dir = root / '.github' / 'workflows'
    for wf in sorted(list(wf_dir.glob('*.yml')) + list(wf_dir.glob('*.yaml'))):
        invs += scan_workflow(wf, package)

    stats = {'files': len(files), 'invocations': len(invs),
             'cfgs': sum(len(f.file_cfgs) + len(f.item_cfgs) for f in files)}
    violations: list[str] = []
    if not files:
        violations.append('empty_scan: tests/*.rs が 0 件 (tests/ が無いか、検査器が何も見ていない)')
    if not invs:
        violations.append('empty_scan: push / pull_request で走る `cargo test` が workflow に 0 件')
    if violations:
        return violations, stats
    receipt_suites, rule_problems = gpu_receipt_suites(root)
    violations += rule_problems
    receipt = Invocation(where=f'the GPU receipt ({GPU_RECEIPT_RULE})', args=[], all_features=True, require_gpu=True)
    stats['receipt_suites'] = len(receipt_suites)

    gpu_runs = gpu_files = 0
    for tf in files:
        rel = tf.path.relative_to(root).as_posix()
        stem = tf.path.stem
        runners = [(inv, enabled_features(inv, package)) for inv in invs if runs_file(inv, stem)]
        if stem in receipt_suites:
            runners.append((receipt, enabled_features(receipt, package)))
        # a file with gpu tests (or gpu arms) must run at least once with the gpu feature where its device must be
        # present: a test that skips without an adapter reports ok, so only a run with a REQUIRE_VARS variable set
        # turns a missing device into a failure (skip-tolerant runs on device-less runners do not count)
        if any(mentions_gpu(t) for _, _, t in tf.file_cfgs + tf.item_cfgs):
            gpu_files += 1
            runs = [inv for inv, feats in runners if 'gpu' in feats and not inv.no_run]
            gpu_runs += len(runs)
            if runs and not any(inv.require_gpu for inv in runs):
                violations.append(
                    f'gpu_unrequired: {rel}: every run with the gpu feature ({", ".join(i.where for i in runs)}) '
                    f'lacks {REQUIRE_GPU}=1 (or ALICE_SDF_REQUIRE_METAL=1): a test that skips without a device would pass')

        def satisfied(feats, extra=None):
            trees = [t for _, _, t in tf.file_cfgs] + ([extra] if extra else [])
            return eval_cfg(('all', trees), feats) is not False

        if not any(satisfied(feats) for _, feats in runners):
            need = ' かつ '.join(f'cfg({e})' for _, e, _ in tf.file_cfgs) or '(cfg なし)'
            violations.append(
                f'unrun: {rel}: {need} を満たす feature で {stem} を走らせる `cargo test` が CI に無い '
                f'(この file の test は CI で 0 本実行になる)')
            continue
        for line, expr, tree in tf.item_cfgs:
            if not any(satisfied(feats, tree) for _, feats in runners):
                violations.append(
                    f'unrun_item: {rel}:{line}: cfg({expr}) を満たす feature で {stem} を走らせる '
                    f'`cargo test` が CI に無い (この arm は CI で compile されない)')
    stats['gpu_runs'] = gpu_runs
    # gpu files that no step runs with the gpu feature are already unrun findings; with none of those, 0 gpu runs
    # means the check itself read nothing
    if gpu_files and gpu_runs == 0 and not any(v.startswith('unrun') for v in violations):
        violations.append('empty_scan: gpu test files exist and no CI step runs one with the gpu feature (the device check read nothing)')
    return violations, stats


def main() -> int:
    # Windows の stdout は locale 既定 (cp1252) で、日本語の集計行を print すると
    # UnicodeEncodeError で落ちる (2026-10-04 windows-latest で実測)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    violations, stats = check(root)
    for v in violations:
        print(v, file=sys.stderr)
    print(f"ci-test-coverage: test file {stats['files']} / feature cfg {stats['cfgs']} / "
          f"cargo test {stats['invocations']} を突合、{len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == '__main__':
    sys.exit(main())
