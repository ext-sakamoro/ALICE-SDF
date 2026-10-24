#!/usr/bin/env python3
"""Every job on the self-hosted runner runs only for the owner's pushes in this repository.

usage: self_hosted_guard.py [workflows dir]

A job whose `runs-on` names `self-hosted` runs on one machine the owner controls, so it may
start only from the owner's own push in this repository (a pull request from elsewhere, a re-run
by someone else or a fork must not reach it). For every such job this fails unless:
  - its `if:` requires all of REQUIRED (repository, actor and triggering actor) as conjuncts of
    the whole expression: the top level is an `&&` chain with each of them as one operand, and no
    `||` or `!` at the top level (an `||` inside parentheses is fine); a string with an embedded
    `${{ }}` is always true and fails;
  - its `runs-on` is resolved: a `${{ matrix.<key> }}` is read from the job's matrix (any value that
    names self-hosted makes it a self-hosted job), any other expression is a finding;
  - its workflow has no `workflow_dispatch` trigger (a manual run on this host needs required
    reviewers first);
  - its workflow's concurrency never cancels a run (`cancel-in-progress: false`; one machine);
and it fails when no self-hosted job was found (compared nothing). Workflows are read with the
strict reader of scripts/changed_paths.py (a file outside its YAML subset is a finding).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import changed_paths as cp  # noqa: E402

REPOSITORY = "ext-sakamoro/ALICE-SDF"
OWNER = "ext-sakamoro"
REQUIRED = (
    f"github.repository == '{REPOSITORY}'",
    f"github.actor == '{OWNER}'",
    f"github.triggering_actor == '{OWNER}'",
)


class Unresolved(ValueError):
    pass


def split_top(expr: str, op: str) -> list[str]:
    """expr split at `op` outside parentheses and quotes"""
    parts, depth, cur, i, q = [], 0, "", 0, None
    while i < len(expr):
        c = expr[i]
        if q:
            cur += c
            if c == q:
                # a doubled quote is an escaped quote in GitHub expressions
                if expr[i + 1:i + 2] == q:
                    cur += q
                    i += 2
                    continue
                q = None
            i += 1
            continue
        if c == "'":
            q = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                raise Unresolved(f"unbalanced parentheses in `{expr}`")
        if depth == 0 and expr.startswith(op, i):
            parts.append(cur)
            cur = ""
            i += len(op)
            continue
        cur += c
        i += 1
    if depth or q:
        raise Unresolved(f"unbalanced parentheses or quotes in `{expr}`")
    parts.append(cur)
    return [x.strip() for x in parts]


def strip_parens(expr: str) -> str:
    """the expression without parentheses that wrap all of it"""
    while expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1]
        try:
            split_top(inner, "&&")  # balanced inside: the outer pair wraps everything
        except Unresolved:
            break
        depth, wraps = 0, True
        for j, c in enumerate(expr):
            depth += c == "("
            depth -= c == ")"
            if depth == 0 and j < len(expr) - 1:
                wraps = False
                break
        if not wraps:
            break
        expr = inner.strip()
    return expr


def conjuncts(cond: str) -> list[str]:
    """the top-level `&&` operands of a job `if:` (Unresolved when it is not one && chain)"""
    s = cond.strip()
    if s.startswith("${{") and s.endswith("}}") and s.count("${{") == 1:
        s = s[3:-2].strip()
    elif "${{" in s:
        raise Unresolved("an `if:` with an embedded `${{ }}` is a string, always true")
    s = strip_parens(s)
    if len(split_top(s, "||")) > 1:
        raise Unresolved("`||` at the top level of the `if:`")
    parts = [strip_parens(x) for x in split_top(s, "&&")]
    return [" ".join(x.split()) for x in parts]


def runs_on_values(job: dict) -> list:
    """the values `runs-on` takes (a matrix expression read from the job's matrix)"""
    ro = job.get("runs-on")
    if isinstance(ro, str) and "${{" in ro:
        import re as _re
        m = _re.fullmatch(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}", ro.strip())
        matrix = ((job.get("strategy") or {}).get("matrix") or {}) if isinstance(job.get("strategy"), dict) else {}
        if not m or not isinstance(matrix, dict):
            raise Unresolved(f"runs-on `{ro}`")
        key = m.group(1)
        vals = []
        if isinstance(matrix.get(key), list):
            vals += matrix[key]
        for inc in matrix.get("include") or []:
            if isinstance(inc, dict) and key in inc:
                vals.append(inc[key])
        if not vals or any(isinstance(v, str) and "${{" in v for v in vals):
            raise Unresolved(f"runs-on `{ro}`")
        return vals
    return [ro]


def self_hosted(runs_on) -> bool:
    if isinstance(runs_on, str):
        return "self-hosted" in runs_on.split(",") or runs_on.strip() == "self-hosted"
    if isinstance(runs_on, list):
        return "self-hosted" in runs_on
    if isinstance(runs_on, dict):  # runs-on: {group: ..., labels: [...]}
        labels = runs_on.get("labels")
        return labels == "self-hosted" or (isinstance(labels, list) and "self-hosted" in labels)
    return False


def check(root: Path) -> tuple[list[str], int]:
    errs: list[str] = []
    found = 0
    for f in sorted(list(root.glob("*.yml")) + list(root.glob("*.yaml"))):
        try:
            data = cp.load_yaml(f.read_text(encoding="utf-8"))
            on = cp.triggers(data)
        except cp.FormError as e:
            errs.append(f"{f.name}: {e}")
            continue
        jobs = data.get("jobs") if isinstance(data, dict) else None
        hosted = []
        for jid, job in (jobs or {}).items():
            if not isinstance(job, dict):
                continue
            try:
                vals = runs_on_values(job)
            except Unresolved as e:
                errs.append(f"{f.name}: job `{jid}`: cannot resolve {e} (a self-hosted runner could hide behind it)")
                continue
            if any(self_hosted(v) for v in vals):
                hosted.append((jid, job))
        if not hosted:
            continue
        if "workflow_dispatch" in on:
            errs.append(f"{f.name}: a workflow with self-hosted jobs has a workflow_dispatch trigger")
        conc = data.get("concurrency")
        if not (isinstance(conc, dict) and conc.get("cancel-in-progress") is False):
            errs.append(f"{f.name}: a workflow with self-hosted jobs must set `cancel-in-progress: false`")
        for jid, job in hosted:
            found += 1
            cond = job.get("if")
            if not isinstance(cond, str):
                errs.append(f"{f.name}: self-hosted job `{jid}` has no `if:`, so it does not require {list(REQUIRED)}")
                continue
            try:
                parts = conjuncts(cond)
            except Unresolved as e:
                errs.append(f"{f.name}: self-hosted job `{jid}`: {e}")
                continue
            missing = [r for r in REQUIRED if r not in parts]
            if missing:
                errs.append(f"{f.name}: self-hosted job `{jid}` does not require {missing} as conjuncts of its `if:`")
    return errs, found


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else HERE.parent / ".github" / "workflows"
    errs, found = check(root)
    for e in errs:
        print(f"self_hosted_guard: {e}")
    print(f"self_hosted_guard: {found} self-hosted job(s) checked, {len(errs)} finding(s)")
    if found == 0:
        print("self_hosted_guard: no self-hosted job found (compared nothing)")
        return 1
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
