#!/usr/bin/env python3
"""Every job on the self-hosted runner runs only for the owner's pushes in this repository.

usage: self_hosted_guard.py [workflows dir]

A job whose `runs-on` names `self-hosted` runs on one machine the owner controls, so it may
start only from the owner's own push in this repository (a pull request from elsewhere, a re-run
by someone else or a fork must not reach it). For every such job this fails unless:
  - its `if:` requires all of REQUIRED (repository, actor and triggering actor);
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
        hosted = [(jid, job) for jid, job in (jobs or {}).items()
                  if isinstance(job, dict) and self_hosted(job.get("runs-on"))]
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
            missing = [r for r in REQUIRED if not (isinstance(cond, str) and r in cond)]
            if missing:
                errs.append(f"{f.name}: self-hosted job `{jid}` does not require {missing} in its `if:`")
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
