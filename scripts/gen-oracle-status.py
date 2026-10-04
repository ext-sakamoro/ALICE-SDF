#!/usr/bin/env python3
"""
Generate ALICE-SDF oracle status report.

Scans tests/ for oracle tests and classifies them by implementation status:
- 🟢 Implemented: green (no #[ignore], implementation exists)
- 🟡 Partial: red or pending (implementation exists but incomplete)
- 🔴 Pending: not implemented (#[ignore] or placeholder)
"""

import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"

DOCS_DIR.mkdir(exist_ok=True)


def _strip_line_comment(line):
    """`//` 以降を落とす (コメントに書かれた #[test] / fn を数えない)."""
    return line.split('//', 1)[0] if line.lstrip().startswith('//') else line


def _read_attribute(lines, i):
    """lines[i] から始まる属性 `#[...]` を、閉じる `]` まで連結して返す (行継続 `\\` も畳む).

    返り値: (属性の本文, 次に読む行の index).  実物の `#[ignore = "…… \\\n ……"]` は複数行に
    またがり、1 行ずつ読むと理由文が落ちて「pending」と誤読される.
    """
    buf = lines[i].strip()
    j = i + 1
    while buf.count('[') > buf.count(']') or buf.count('"') % 2 == 1:
        if j >= len(lines):
            break
        nxt = lines[j].strip()
        buf = (buf[:-1] if buf.endswith('\\') else buf + ' ') + nxt
        j += 1
    return buf, j


def _ignore_reason(attr):
    """`#[ignore]` / `#[ignore = "…"]` の理由 (空の ignore は 'pending')."""
    m = re.match(r'#\[ignore\s*=\s*"(.*)"\s*\]\s*$', attr, flags=re.S)
    return re.sub(r'\s+', ' ', m.group(1)).strip() if m else 'pending'


def extract_test_metadata(test_file):
    """Extract test functions and their ignore status from a test file.

    属性は `#[test]` の前後どちらに `#[ignore]` があっても拾い、`#[test]` から `fn` までの
    距離に上限を設けない (長い doc comment / 複数行の `#[should_panic]` を落とさない).
    """
    tests = []
    lines = Path(test_file).read_text(encoding='utf-8').split('\n')

    pending_attrs = []  # 直前までに読んだ属性 (空行・コメントでは切れない)
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith('//') or not stripped:
            i += 1
            continue
        if stripped.startswith('#['):
            attr, i = _read_attribute(lines, i)
            pending_attrs.append(attr)
            continue
        fn_match = re.match(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(', stripped)
        if fn_match:
            is_test = any(re.match(r'#\[(?:\w+::)?test\]$', a) for a in pending_attrs)
            ignores = [a for a in pending_attrs if a.startswith('#[ignore')]
            if is_test:
                tests.append({
                    'name': fn_match.group(1),
                    'file': Path(test_file).name,
                    'is_ignored': bool(ignores),
                    'ignore_reason': _ignore_reason(ignores[0]) if ignores else '',
                })
        pending_attrs = []
        i += 1

    return tests


def classify_ignored(reason):
    """`#[ignore]` の理由から 3 分類する.

    - red:     意図して red のまま残している oracle (「the red is correct」/「src gap」/「known defect: AUD-…」= 監査台帳の欠陥)
               実装側が追いつけば #[ignore] を外す  実装を足す対象であって、期待値を緩めない
    - gated:   実行が長い / 診断表を出すだけ / 手動 (runtime / diagnostic / manual / run with --release)
    - pending: 理由の無い bare な #[ignore]
    """
    r = reason.strip().lower()
    if r == 'pending' or not r:
        return 'pending'
    if r.startswith(('the red is correct', 'src gap', 'known defect')):
        return 'red'
    return 'gated'


def run_tests_and_categorize():
    """Run tests and categorize by status."""
    all_tests = defaultdict(list)

    for test_file in sorted(TESTS_DIR.glob('*.rs')):
        tests = extract_test_metadata(test_file)
        for test in tests:
            all_tests[test_file.name].append(test)

    # Categorize
    implemented = []
    partial = []
    pending = []

    for test_file_name, tests in sorted(all_tests.items()):
        for test in tests:
            test_info = {
                'test_name': test['name'],
                'file': test['file'],
                'ignore_reason': test['ignore_reason'],
            }

            if test['is_ignored']:
                test_info['reason'] = test['ignore_reason'] or 'pending implementation'
                pending.append(test_info)
            else:
                # Not ignored = assume implemented
                implemented.append(test_info)

    return {
        'implemented': implemented,
        'partial': partial,
        'pending': pending,
    }


def _line(test):
    reason = test['ignore_reason']
    if reason and reason != 'pending':
        short = reason[:110] + ('…' if len(reason) > 110 else '')
        return f"- `{test['test_name']}` ({test['file']}) — {short}\n"
    return f"- `{test['test_name']}` ({test['file']})\n"


def generate_markdown_report(categorized):
    """Generate markdown report (a pure function: no timestamp, so it changes only when the tests do)."""
    ignored = categorized['pending']
    by_class = {'red': [], 'gated': [], 'pending': []}
    for t in ignored:
        by_class[classify_ignored(t['ignore_reason'])].append(t)
    total = sum(len(v) for v in categorized.values())

    report = f"""# ALICE-SDF Oracle Status

_Generated from `tests/*.rs` (no timestamp: the file changes only when its content does)._

## Summary

| Category | Count |
|----------|-------|
| 🟢 Not ignored (run by CI) | {len(categorized['implemented'])} |
| 🔴 Red by design | {len(by_class['red'])} |
| ⏱ Gated (runtime / diagnostic / manual) | {len(by_class['gated'])} |
| ⚪ Pending (bare `#[ignore]`) | {len(by_class['pending'])} |
| **Total** | **{total}** |

`Not ignored` means only that the test carries no `#[ignore]`: this report does not run it.
CI's `cargo test` is what says whether it passes.

"""

    if by_class['red']:
        report += f"""## 🔴 Red by design ({len(by_class['red'])})

Oracles kept red on purpose: the implementation is not there yet, and a companion test pins
today's behaviour so CI coverage is not lost. The fix is in `src/`; the expected value is never loosened.

"""
        for t in sorted(by_class['red'], key=lambda x: x['test_name']):
            report += _line(t)
        report += "\n"

    if by_class['gated']:
        report += f"""## ⏱ Gated ({len(by_class['gated'])})

Correct tests that are too slow for every push, or that print a measurement table.
Run them with `python3 scripts/run_ignored.py` or `cargo test --release -- --ignored`.

"""
        for t in sorted(by_class['gated'], key=lambda x: x['test_name']):
            report += _line(t)
        report += "\n"

    if by_class['pending']:
        report += f"""## ⚪ Pending ({len(by_class['pending'])})

`#[ignore]` with no reason: not yet implemented, or forgotten.

"""
        for t in sorted(by_class['pending'], key=lambda x: x['test_name']):
            report += _line(t)
        report += "\n"

    if categorized['partial']:
        report += f"""## 🟡 Partial ({len(categorized['partial'])})

"""
        for t in sorted(categorized['partial'], key=lambda x: x['test_name']):
            report += f"- `{t['test_name']}` ({t['file']}) — {t.get('reason', 'partial implementation')}\n"
        report += "\n"

    report += f"""## 🟢 Not ignored ({len(categorized['implemented'])})

Per-file counts (the test names are in `tests/`):

| File | Tests |
|------|-------|
"""
    per_file = defaultdict(int)
    for t in categorized['implemented']:
        per_file[t['file']] += 1
    for name, n in sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0])):
        report += f"| `{name}` | {n} |\n"

    report += """
---

## How to Contribute

When an oracle goes green:
1. Remove `#[ignore]` from the test (and the companion test that pins the old behaviour, if the reason says so)
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify

For details: [CLAUDE.md](../CLAUDE.md)
"""

    return report


def main():
    import sys
    print("Scanning ALICE-SDF oracle tests...", file=sys.stderr)

    categorized = run_tests_and_categorize()
    total = sum(len(v) for v in categorized.values())
    if total == 0:
        # 検査対象 0 件を green と読ませない (tests/ の場所や属性の読み違いで空振りする)
        print("ERROR: oracle test が 1 件も見つからない (走査の空振り)", file=sys.stderr)
        sys.exit(1)
    report = generate_markdown_report(categorized)

    output_file = DOCS_DIR / "oracle-status.md"
    output_file.write_text(report, encoding='utf-8')

    print(f"✅ Generated {output_file}", file=sys.stderr)
    print(f"   Implemented: {len(categorized['implemented'])}", file=sys.stderr)
    print(f"   Partial: {len(categorized['partial'])}", file=sys.stderr)
    print(f"   Pending: {len(categorized['pending'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
