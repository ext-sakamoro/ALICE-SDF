#!/usr/bin/env python3
"""Every workflow run by a push to a branch has a top-level `concurrency` group that
cancels superseded runs on a branch but never a run on main (per-commit group on main).

Without one, each push to a branch queues a full run of its own and a superseded run
holds a shared runner until it finishes, delaying every other run in the account.
A workflow whose push trigger names only tags (a release) is exempt, as is one with
no push trigger. Fails when such a workflow lacks the key, and when no workflow was read.

A workflow whose jobs ALL run on `self-hosted` runners (one machine) may keep one group
per ref on main instead of one per commit (the `changes` job that runs `changed_paths.py
decide` and the job that runs `changed_paths.py gate` may run on a hosted runner: they only
decide and report, so the machine still runs one workflow's jobs; each is recognised by its
steps, not by text: every `run:` step of the `changes` job is exactly the decide call, and the
gate job has exactly one `run:` step, which is exactly the gate call; the changes job may also
have the one policy step, POLICY_RUN, that ANDs decide with a condition): a per-commit group
would let a burst of pushes queue without bound on that machine, while a per-ref group
keeps at most one pending run (the latest main commit is always run; an older pending one may be
superseded). It must say so in a comment line starting with SELF_HOSTED_REASON and
giving the reason, its group must still name `github.ref`, and it may never cancel
an in-progress run on main (the cancel rule applies to it unchanged).

Workflows are read with the strict reader of scripts/changed_paths.py (block and flow style,
comments anywhere); a file outside its YAML subset is a finding, and without that script the
check fails (it does not fall back to reading lines).

usage: workflow_concurrency.py [workflows dir]

Shared byte-identical by the ALICE crates that carry it (this file and its test,
ALICE-Law's shared table); change every copy together.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import changed_paths as cp
except ImportError:  # the reader is required: no fallback to reading lines
    cp = None


# the forms this repository uses (a different spelling of the same rule fails and must
# be added here deliberately)
MAIN_EXCEPTED = "github.ref != 'refs/heads/main'"
PER_SHA_ON_MAIN = "github.ref == 'refs/heads/main' && github.sha"
# the comment that claims the single self-hosted runner exemption, followed by a reason
SELF_HOSTED_REASON = "# concurrency: single self-hosted runner"
MIN_REASON = 12
# the whole run of a changes job's decide step / of the gate step (no other command may ride along)
DECIDE_RUN = re.compile(r"\s*python3?\s+scripts/changed_paths\.py\s+decide(?:\s+[^;&|`\n]*)?\s*")
GATE_RUN = re.compile(r'\s*python3?\s+scripts/changed_paths\.py\s+gate\s+--needs-json\s+"\$NEEDS"\s*')
# the one other step a changes job may have: run = decide AND a policy (a self-hosted workflow that
# runs only for some pushes), so a skip by policy is run=false and the gate expects a skipped job
POLICY_RUN = (
    'if [ "$DECIDED" = true ] && [ "$ALLOWED" = true ]; then run=true; else run=false; fi\n'
    'echo "decided=$DECIDED allowed=$ALLOWED run=$run"\n'
    'echo "run=$run" >> "$GITHUB_OUTPUT"'
)


def run_steps(job: dict) -> list[str]:
    steps = job.get("steps") if isinstance(job, dict) else None
    return [s.get("run") for s in steps or [] if isinstance(s, dict) and "run" in s]


def self_hosted(runs_on) -> bool:
    if isinstance(runs_on, str):
        return "self-hosted" in runs_on
    if isinstance(runs_on, list):
        return any(isinstance(x, str) and "self-hosted" in x for x in runs_on)
    if isinstance(runs_on, dict):
        return self_hosted(runs_on.get("labels"))
    return False


def all_jobs_self_hosted(data) -> bool:
    """whether the workflow has a self-hosted job and every job runs on `self-hosted`, apart from
    changed_paths.py's `changes` job (decide only) and its gate job (the gate call only)"""
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, dict):
        return False
    hosted = 0
    for jid, job in jobs.items():
        if not isinstance(job, dict):
            return False
        if self_hosted(job.get("runs-on")):
            hosted += 1
            continue
        runs = run_steps(job)
        if not runs or not all(isinstance(r, str) for r in runs):
            return False
        if jid == "changes":
            decides = [r for r in runs if DECIDE_RUN.fullmatch(r) and "$(" not in r]
            others = [r for r in runs if r not in decides]
            if len(decides) == 1 and all(r.strip() == POLICY_RUN for r in others) and len(others) <= 1:
                continue
        if len(runs) == 1 and GATE_RUN.fullmatch(runs[0]):
            continue
        return False
    return hosted > 0


def self_hosted_reason(text: str) -> bool:
    """whether a comment line claims the exemption and gives a reason after it"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(SELF_HOSTED_REASON):
            reason = stripped[len(SELF_HOSTED_REASON):].strip(" :-—")
            if len(reason) >= MIN_REASON:
                return True
    return False


def push_to_branches(data) -> bool:
    """whether `on:` has a push trigger that is not limited to tags"""
    if "push" not in cp.triggers(data):
        return False
    push = cp.triggers(data)["push"]
    if not isinstance(push, dict):
        return True
    keys = set(push)
    return not (keys and keys <= {"tags", "tags-ignore"})


def main(argv: list[str]) -> int:
    if cp is None:
        print("error: scripts/changed_paths.py (the workflow reader) is missing", file=sys.stderr)
        return 1
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / ".github" / "workflows"
    files = sorted(list(root.glob("*.yml")) + list(root.glob("*.yaml")))
    missing, checked = [], 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        try:
            data = cp.load_yaml(text)
            if not push_to_branches(data):
                continue
        except cp.FormError as e:
            missing.append(f"{f.name}: {e} (read with the strict reader of scripts/changed_paths.py)")
            continue
        checked += 1
        conc = data.get("concurrency")
        if isinstance(conc, str):
            conc = {"group": conc}
        if not isinstance(conc, dict) or "group" not in conc:
            missing.append(f"{f.name}: runs on a push to a branch but has no top-level concurrency group")
            continue
        group = str(conc.get("group"))
        cancel = conc.get("cancel-in-progress")
        # main is never cancelled: a cancelled main commit has no conclusion (unverified)
        cancel_ok = cancel is None or cancel is False or (isinstance(cancel, str) and MAIN_EXCEPTED in cancel)
        # on main each commit has a group of its own: a group keeps one pending run and
        # cancels an older pending one whatever cancel-in-progress says
        group_ok = PER_SHA_ON_MAIN in group
        if not group_ok and all_jobs_self_hosted(data):
            # one machine: a per-ref group (at most one pending run) with a stated reason
            if "github.ref" in group and self_hosted_reason(text):
                group_ok = True
            else:
                missing.append(
                    f"{f.name}: all jobs are self-hosted but the group is not per ref, or the "
                    f"'{SELF_HOSTED_REASON} <reason>' comment is missing"
                )
        if not cancel_ok:
            missing.append(f"{f.name}: cancel-in-progress can cancel a run on main")
        if not group_ok:
            missing.append(f"{f.name}: the concurrency group is not per commit on main")
    for m in missing:
        print(m)
    if not files:
        print("error: no workflow read (compared nothing)", file=sys.stderr)
        return 1
    if missing:
        return 1
    print(f"ok: {checked} of {len(files)} workflows run on a branch push, each has a concurrency group")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
