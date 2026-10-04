#!/usr/bin/env python3
"""
Generate ALICE-SDF wiring status report.

Scans baseline and wiring_guard output to report permitted vs new violations.
"""

import re
import subprocess
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
WIRING_GUARD = PROJECT_ROOT / "scripts" / "wiring_guard.py"
BASELINE_FILE = PROJECT_ROOT / "scripts" / "wiring-baseline.txt"

DOCS_DIR.mkdir(exist_ok=True)


def load_baseline():
    """Load baseline (permitted violations)."""
    baseline = {
        'dead_code': [],
        'unwired': [],
    }

    if not BASELINE_FILE.exists():
        return baseline

    with open(BASELINE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                violation_type = parts[0]
                if violation_type in baseline:
                    baseline[violation_type].append(line)

    return baseline


def run_wiring_guard():
    """Execute wiring_guard and capture output."""
    try:
        result = subprocess.run(
            ["python3", str(WIRING_GUARD)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running wiring_guard: {e}\n"


# wiring_guard が stderr に出す違反の行: `<kind>: <key>: <message>` (scripts/wiring_guard.py の
# `print(f"{v.kind}: {v.key}: {v.message}")`)  最後の `wiring-guard: N violation(s)` は集計行で
# 違反ではない  以前はここを `wiring-guard:` で始まる行と取り違えていて、違反を 1 件も拾えず
# 新しい違反があっても必ず「違反なし」と報告していた (2026-10-03)
KINDS = ('dead_code', 'unwired', 'stale_baseline', 'unbalanced_braces')


def parse_violations(output):
    """Parse wiring_guard error output into {kind: [line, ...]}."""
    violations = {kind: [] for kind in KINDS}
    for line in output.split('\n'):
        kind, sep, _ = line.partition(': ')
        if sep and kind in violations:
            violations[kind].append(line)
    return violations


def exit_code(violations):
    """違反があっても report は書く: 違反の gate は CI の wiring_guard step の仕事で、
    ここで落ちると台帳が更新されず古いままになる (本末転倒)."""
    return 0


def per_file_counts(baseline):
    """baseline の行数を file ごとに数える (dead_code は 1 行 = 1 file の件数集約)."""
    counts = Counter()
    for line in baseline['dead_code'] + baseline['unwired']:
        m = re.search(r'(src/\S+?\.rs)', line)
        if m:
            counts[m.group(1)] += 1
    return counts


def generate_markdown_report(baseline, violations):
    """Generate markdown report combining baseline and current violations."""
    # Check if there are any NEW (non-baseline) violations
    has_new_violations = bool(violations['dead_code'] or violations['unwired'] or
                               violations['stale_baseline'] or violations['unbalanced_braces'])

    total_baseline = len(baseline['dead_code']) + len(baseline['unwired'])

    if has_new_violations:
        status = "❌ **NEW violations detected** — Must be resolved or added to baseline"
    elif total_baseline > 0:
        status = f"🟡 **{total_baseline} baseline items** — Permitted violations, ratchet in place"
    else:
        status = "✅ **All clear** — No violations"

    report = f"""# ALICE-SDF Wiring Status

_Generated from `scripts/wiring-baseline.txt` and `scripts/wiring_guard.py` (no timestamp: the file changes only when its content does)._

## Status

{status}

---

"""

    if has_new_violations:
        if violations['dead_code']:
            report += f"""## 🔴 NEW: Dead Code Guard ({len(violations['dead_code'])})

```
{chr(10).join(violations['dead_code'])}
```

"""

        if violations['unwired']:
            report += f"""## 🔴 NEW: Unwired Items ({len(violations['unwired'])})

```
{chr(10).join(violations['unwired'])}
```

"""

        if violations['stale_baseline']:
            report += f"""## 🔴 NEW: Stale Baseline ({len(violations['stale_baseline'])})

```
{chr(10).join(violations['stale_baseline'])}
```

"""

        if violations['unbalanced_braces']:
            report += f"""## 🔴 NEW: Syntax Errors ({len(violations['unbalanced_braces'])})

```
{chr(10).join(violations['unbalanced_braces'])}
```

"""

    if total_baseline > 0:
        report += f"""## 📋 Baseline ({total_baseline} permitted)

Violations explicitly allowed via `scripts/wiring-baseline.txt`.
Must resolve or remove from baseline to reduce ratchet.

### By file

| File | Baseline lines |
|------|----------------|
"""
        for name, n in sorted(per_file_counts(baseline).items(), key=lambda kv: (-kv[1], kv[0])):
            report += f"| `{name}` | {n} |\n"
        report += "\n"

        if baseline['dead_code']:
            report += f"""### Dead Code ({len(baseline['dead_code'])})

```
{chr(10).join(baseline['dead_code'])}
```

"""

        if baseline['unwired']:
            report += f"""### Unwired Items ({len(baseline['unwired'])})

```
{chr(10).join(baseline['unwired'])}
```

"""

    report += """---

## What is the Wiring Guard?

The wiring guard ensures that all public items in `src/` are actually called from production code:

- **Dead Code Guard**: Verifies that `#[allow(dead_code)]` has a documented reason
- **Unwired Items**: Detects public functions, structs, etc. that are never called (except in tests)
- **Stale Baseline**: Ensures baseline entries are still needed
- **Brace Balance**: Checks syntax integrity

### Resolving Violations

1. **New violations**: Either implement/wire the item, or add to `scripts/wiring-baseline.txt`
2. **Baseline cleanup**: Remove lines from baseline as violations are resolved
3. **Comments**: Add `// ALLOW-DEAD:` or `// ALLOW-UNWIRED:` with reason (12+ chars)

For details: see `scripts/wiring_guard.py`
"""

    return report


def main():
    import sys

    print("Scanning wiring status...", file=sys.stderr)
    baseline = load_baseline()
    guard_output = run_wiring_guard()
    violations = parse_violations(guard_output)

    report = generate_markdown_report(baseline, violations)

    output_file = DOCS_DIR / "wiring-status.md"
    output_file.write_text(report, encoding='utf-8')

    print(f"✅ Generated {output_file}", file=sys.stderr)
    print(f"   Baseline: {len(baseline['dead_code']) + len(baseline['unwired'])}", file=sys.stderr)
    print(f"   New violations: {len(violations['dead_code']) + len(violations['unwired']) + len(violations['stale_baseline']) + len(violations['unbalanced_braces'])}", file=sys.stderr)

    return exit_code(violations)


if __name__ == "__main__":
    import sys
    sys.exit(main())
