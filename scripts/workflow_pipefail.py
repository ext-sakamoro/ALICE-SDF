#!/usr/bin/env python3
"""Every `run:` step of a workflow that pipes must run with pipefail.

  workflow_pipefail.py [<workflows dir>]     (default .github/workflows)

GitHub runs a step that names no shell as `bash -e {0}`: without pipefail,
`script | tee log` takes tee's exit status and a failing script goes green.
A step passes when its shell sets pipefail: `bash` (GitHub runs it as `bash
--noprofile --norc -eo pipefail {0}`) or a command line with `-o pipefail`. The
nearest setting wins: the step's `shell:`, else its job's `defaults.run.shell`,
else the workflow's. A step also passes when its script runs `set -o pipefail`
before its first pipe, or reads every pipe's status from `PIPESTATUS` right after
it (the same line or the next command line).

A pipe is a `|` outside quotes that is not part of `||` and not the `|` between
the patterns of a `case` branch. Steps in a shell that is not bash (pwsh,
powershell, python, cmd) are not checked: pipefail is a bash setting. The workflows are read
line by line (no YAML library): mappings by indentation, a step from its `- `
line, `run: |` / `run: >` block scalars by indentation.

Exit 0 clean; 1 on a step that pipes without pipefail (file:line); 2 when no
workflow file was read.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PIPE = re.compile(r"(?<!\|)\|(?![|&])")
QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")


def indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


CASE_OPEN = re.compile(r"\bcase\b.*?\bin\b")
CASE_PATTERN = re.compile(r"(^|;;|\bin\b)\s*\(?[^()|;]+(\|[^()|;]+)*\)")


def bare_lines(script: str) -> list[str]:
    """The script's lines without quoted text, and inside `case … esac` without the
    branch patterns (`0 | 2 | 3)` is an alternation, not a pipe)."""
    out = []
    depth = 0
    for line in script.splitlines():
        bare = QUOTED.sub("", line)
        opens = len(CASE_OPEN.findall(bare))
        depth += opens
        if depth > 0:
            bare = CASE_PATTERN.sub(lambda m: m.group(1) + " ", bare)
        depth -= len(re.findall(r"\besac\b", bare))
        depth = max(depth, 0)
        out.append(bare)
    return out


def pipes(script: str) -> bool:
    return any(PIPE.search(line) for line in bare_lines(script))


def defaults_shell(lines: list[str], at: int) -> str | None:
    """the `run: shell:` the `defaults:` mapping on line `at` sets, or None"""
    base = indent(lines[at])
    run_at = None
    for m in lines[at + 1:]:
        if not m.strip() or m.lstrip().startswith("#"):
            continue
        if indent(m) <= base:
            break
        if m.strip() == "run:":
            run_at = indent(m)
        elif run_at is not None and indent(m) <= run_at:
            run_at = None
        elif run_at is not None and (sm := re.fullmatch(r"shell:\s*(.+?)\s*", m.strip())):
            return sm.group(1)
    return None


def sets_pipefail(shell: str) -> bool:
    """GitHub runs `shell: bash` as `bash --noprofile --norc -eo pipefail {0}`; a custom
    command line sets it only when it says so"""
    return shell == "bash" or bool(re.search(r"(^|\s)-[a-z]*o\s+pipefail\b|(^|\s)-o\s+pipefail\b", shell))


def status_checked(script: str) -> bool:
    """every pipe's status is read from PIPESTATUS right after it: later on the same line
    or on the next command line (bash keeps PIPESTATUS for the last pipeline only)"""
    # the pipe is found on the line without quotes; PIPESTATUS is read on the raw line ("${PIPESTATUS[0]}" is quoted)
    pairs = [(r, b) for r, b in zip(script.splitlines(), bare_lines(script)) if r.strip() and not r.lstrip().startswith("#")]
    found = False
    for i, (raw, bare) in enumerate(pairs):
        m = PIPE.search(bare)
        if not m:
            continue
        found = True
        bar = len(raw) - len(raw.split("|")[-1]) if "|" in raw else 0
        tail = raw[bar:]
        nxt = pairs[i + 1][0] if i + 1 < len(pairs) else ""
        if "PIPESTATUS" not in tail and "PIPESTATUS" not in nxt:
            return False
    return found


def pipefail_before_first_pipe(script: str) -> bool:
    for bare in bare_lines(script):
        if re.search(r"\bset\s+-[a-z]*o\s+pipefail\b", bare):
            return True
        if PIPE.search(bare):
            return False
    return False


def check(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    problems: list[str] = []
    wf_shell = next((defaults_shell(lines, k) for k, l in enumerate(lines) if l == "defaults:"), None)
    job_shell = None
    job_indent = None
    in_jobs = False
    i = 0
    while i < len(lines):
        l = lines[i]
        s = l.strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        ind = indent(l)
        if ind == 0:
            in_jobs = s == "jobs:"
            job_indent = None
        elif in_jobs and (job_indent is None or ind == job_indent) and s.endswith(":") and not s.startswith("- "):
            job_indent = ind
            # this job's own defaults: scan its block for `defaults:` one level in
            job_shell = None
            for j in range(i + 1, len(lines)):
                m = lines[j]
                if m.strip() and not m.lstrip().startswith("#") and indent(m) <= ind:
                    break
                if m.strip() == "defaults:":
                    job_shell = defaults_shell(lines, j)
                    break
        if in_jobs and s.startswith("- "):
            dash = ind
            key_indent = dash + 2
            step_shell = None
            script_parts: list[str] = []
            run_line = None
            j = i
            first = True
            while j < len(lines):
                m = lines[j]
                ms = m.strip()
                if not first and ms and not ms.startswith("#") and indent(m) <= dash:
                    break
                body = m[key_indent:] if first else m[key_indent:] if indent(m) >= key_indent else ms
                if first:
                    body = m[dash + 2:]
                first = False
                km = re.match(r"(run|shell):\s*(.*)$", body) if (indent(m) == key_indent or j == i) else None
                if km and km.group(1) == "shell":
                    step_shell = km.group(2).strip()
                elif km and km.group(1) == "run":
                    run_line = j + 1
                    val = km.group(2)
                    if re.fullmatch(r"[|>][+-]?\s*", val):
                        k = j + 1
                        while k < len(lines) and (not lines[k].strip() or indent(lines[k]) > key_indent):
                            script_parts.append(lines[k])
                            k += 1
                        j = k
                        continue
                    script_parts.append(val)
                j += 1
            script = "\n".join(script_parts)
            if run_line is not None and pipes(script):
                # the nearest setting wins: the step's, else its job's defaults, else the workflow's
                shell = step_shell if step_shell is not None else job_shell if job_shell is not None else wf_shell
                # pipefail is a bash setting: a pwsh / python / cmd step is not this check's
                not_bash = shell is not None and re.match(r"(pwsh|powershell|python|cmd)\b", shell)
                ok = (not_bash or (shell is not None and sets_pipefail(shell)) or pipefail_before_first_pipe(script)
                      or status_checked(script))
                if not ok:
                    problems.append(f"{path.name}:{run_line}: pipes without pipefail (set `shell: bash` on the step, "
                                    f"the job or the workflow's `defaults.run`)")
            i = j
            continue
        i += 1
    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(".github/workflows")
    files = sorted(list(root.glob("*.yml")) + list(root.glob("*.yaml")))
    if not files:
        print(f"error: no workflow file in {root} (nothing was checked)")
        return 2
    problems = [p for f in files for p in check(f)]
    for p in problems:
        print(p)
    print(f"workflow pipefail: {len(files)} workflow(s) read, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
