#!/usr/bin/env python3
"""4.0.1 soundness guard: no raw `Interval { lo, hi }` literal in `src/`
without a written reason.

`Interval::new` / `Interval::outward` round the bounds outward by one ulp, so
every arm that *computes* a new bound stays a valid enclosure even though Rust
gives us no rounding-mode control. That property is an invariant over the whole
evaluator, and until now nothing enforced it: an arm written as
`Interval { lo: a, hi: b }` skips the widening silently, and the inclusion
oracle keeps reporting 0.000e0 because it samples points — it cannot see an
enclosure that is one ulp too narrow at a corner it never sampled. The drift
this cost before the outward rounding landed was 2.086e-4 relative.

A literal is allowed only where the operation *selects* existing values instead
of computing new ones (`min` / `max` / `hull` / `clamp` / `abs` / `neg` /
`intersect`), or where a bound is exact by construction (`point`, `ZERO`,
`EVERYTHING`, the 0 lower bound of `sqr`). Those sites carry

    // ALLOW-RAW-INTERVAL: <why this bound needs no widening>

on the literal's opening line or within the three lines above it. The marker is
deliberately noisy: it turns "skip the rounding" from a default into a line a
reviewer sees in the diff.

Stale markers (a marker with no raw literal under it) are a failure too, so the
allowlist cannot drift away from the code it justifies.
"""

import re
import subprocess
import sys

DIRS = ["src/"]

# `Interval {` / `Self {` — `Vec3Interval {` is excluded by the lookbehind, and
# its fields are `Interval` values, so it cannot bypass the rounding anyway.
OPEN = re.compile(r"(?<![A-Za-z0-9_])(?:Interval|Self)\s*\{")
# a field *named* `lo`, written either as `lo: <expr>` or as the field-init
# shorthand `Interval { lo, hi }` (the shorthand has no colon; matching only
# `lo:` would leave the guard a hole). Matched in field position so that a
# field whose *value* is a local called `lo` (`Self { x: lo, y: hi }`) is not
# mistaken for one.
LO_FIELD = re.compile(r"^lo(\s*:|$)")
MARKER = re.compile(r"ALLOW-RAW-INTERVAL:[ \t]*(\S.*)")
MIN_REASON = 12


def mask(src: str) -> str:
    """Blank out comments and string / char literals, keeping offsets intact.

    Brace counting and field detection run on this; the marker lookup runs on
    the original text.
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            for _ in range(2):
                if i < n:
                    out[i] = " "
                    i += 1
        elif c == "r" and i + 1 < n and src[i + 1] in '#"':
            j = i + 1
            hashes = 0
            while j < n and src[j] == "#":
                hashes += 1
                j += 1
            if j < n and src[j] == '"':
                close = '"' + "#" * hashes
                end = src.find(close, j + 1)
                end = n if end == -1 else end + len(close)
                for k in range(i, end):
                    if src[k] != "\n":
                        out[k] = " "
                i = end
            else:
                i += 1
        elif c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, n)):
                if src[k] != "\n":
                    out[k] = " "
            i = j
        elif c == "'":
            m = re.match(r"'(\\.|[^'\\])'", src[i:])
            if m:
                for k in range(i, i + m.end()):
                    out[k] = " "
                i += m.end()
            else:
                i += 1  # lifetime
        else:
            i += 1
    return "".join(out)


def match_brace(masked: str, open_idx: int) -> int:
    depth = 0
    for k in range(open_idx, len(masked)):
        if masked[k] == "{":
            depth += 1
        elif masked[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    return len(masked) - 1


def has_lo_field(body: str) -> bool:
    """True if the literal body declares a field named `lo` at its top level.

    Splits on the commas at nesting depth 0, so a nested literal's own `lo`
    (or a call argument) does not count as this literal's field.
    """
    fields, depth, cur = [], 0, []
    for ch in body:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == "," and depth == 0:
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    fields.append("".join(cur))
    return any(LO_FIELD.match(f.strip()) for f in fields)


def main() -> int:
    # tracked plus not-yet-added files: a new arm in a brand-new module is the
    # case this guard exists for, and `git ls-files` alone would not see it
    # until after `git add`.
    files = sorted(
        set(
            subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"] + DIRS,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        )
    )

    violations: list[str] = []
    stale: list[str] = []
    allowed = 0

    for f in files:
        if not f.endswith(".rs"):
            continue
        src = open(f, encoding="utf-8").read()
        masked = mask(src)
        lines = src.split("\n")
        starts = [0]
        for line in lines[:-1]:
            starts.append(starts[-1] + len(line) + 1)

        def line_of(off: int) -> int:
            lo, hi = 0, len(starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if starts[mid] <= off:
                    lo = mid
                else:
                    hi = mid - 1
            return lo  # 0-based

        marked_lines: set[int] = set()

        for m in OPEN.finditer(masked):
            ln = line_of(m.start())
            if "struct " in masked[starts[ln] : starts[ln] + len(lines[ln])]:
                continue  # struct definition, not a literal
            brace = m.end() - 1
            close = match_brace(masked, brace)
            if not has_lo_field(masked[brace + 1 : close]):
                continue  # not an Interval-shaped literal

            reason = None
            for probe in range(ln, max(-1, ln - 4), -1):
                mm = MARKER.search(lines[probe])
                if mm:
                    reason = mm.group(1).strip()
                    marked_lines.add(probe)
                    break
            if reason is None:
                violations.append(
                    f"{f}:{ln + 1}: raw `Interval {{ lo, hi }}` with no "
                    f"ALLOW-RAW-INTERVAL reason — use Interval::new / "
                    f"Interval::outward, or state why the bound is exact\n"
                    f"    {lines[ln].strip()}"
                )
            elif len(reason) < MIN_REASON:
                violations.append(
                    f"{f}:{ln + 1}: ALLOW-RAW-INTERVAL reason is too short "
                    f"({len(reason)} < {MIN_REASON} chars): {reason!r}"
                )
            else:
                allowed += 1

        for i, line in enumerate(lines):
            if MARKER.search(line) and i not in marked_lines:
                stale.append(
                    f"{f}:{i + 1}: ALLOW-RAW-INTERVAL with no raw literal in the "
                    f"next 3 lines (stale allowance)\n    {line.strip()}"
                )

    if violations or stale:
        if violations:
            print("❌ raw Interval literal bypassing the outward rounding:")
            print("\n".join(violations))
        if stale:
            print("❌ stale ALLOW-RAW-INTERVAL marker:")
            print("\n".join(stale))
        return 1

    print(
        f"✓ No raw Interval literal bypassing the outward rounding "
        f"({allowed} allowed sites, {len(files)} files)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
