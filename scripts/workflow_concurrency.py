#!/usr/bin/env python3
"""Every workflow run by a push to a branch has a top-level `concurrency` group that
cancels superseded runs on a branch but never a run on main (per-commit group on main).

Without one, each push to a branch queues a full run of its own and a superseded run
holds a shared runner until it finishes, delaying every other run in the account.
A workflow whose push trigger names only tags (a release) is exempt, as is one with
no push trigger. Fails when such a workflow lacks the key, and when no workflow was read.

A workflow whose jobs ALL run on `self-hosted` runners (one machine) may keep one group
per ref on main instead of one per commit: a per-commit group would let a burst of
pushes queue without bound on that machine, while a per-ref group keeps at most one
pending run (the latest main commit is always run; an older pending one may be
superseded). It must say so in a comment line starting with SELF_HOSTED_REASON and
giving the reason, its group must still name `github.ref`, and it may never cancel
an in-progress run on main (the cancel rule applies to it unchanged).

usage: workflow_concurrency.py [workflows dir]

Kept byte-identical in ALICE-LOL, ALICE-SDF and ALICE-DetMath (this file and its
test); change every copy together.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# the forms this repository uses (a different spelling of the same rule fails and must
# be added here deliberately)
MAIN_EXCEPTED = "github.ref != 'refs/heads/main'"
PER_SHA_ON_MAIN = "github.ref == 'refs/heads/main' && github.sha"
# the comment that claims the single self-hosted runner exemption, followed by a reason
SELF_HOSTED_REASON = "# concurrency: single self-hosted runner"
MIN_REASON = 12


def all_jobs_self_hosted(text: str) -> bool:
    """whether the workflow has jobs and every `runs-on` names `self-hosted`"""
    runs = re.findall(r"^\s+runs-on:\s*(.+)$", text, re.M)
    return bool(runs) and all("self-hosted" in r for r in runs)


def self_hosted_reason(text: str) -> bool:
    """whether a comment line claims the exemption and gives a reason after it"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(SELF_HOSTED_REASON):
            reason = stripped[len(SELF_HOSTED_REASON):].strip(" :-—")
            if len(reason) >= MIN_REASON:
                return True
    return False


def push_to_branches(text: str) -> bool:
    """whether `on:` has a push trigger that is not limited to tags"""
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if re.match(r"^on:\s*$", l))
    except StopIteration:
        return bool(re.search(r"^on:.*\bpush\b", text, re.M))
    block = []
    for l in lines[start + 1:]:
        if re.match(r"^\S", l):
            break
        block.append(l)
    push = [i for i, l in enumerate(block) if re.match(r"^  push:", l)]
    if not push:
        return False
    body = []
    for l in block[push[0] + 1:]:
        if re.match(r"^  \S", l):
            break
        body.append(l)
    keys = {m.group(1) for l in body if (m := re.match(r"^    ([a-z-]+):", l))}
    return not (keys and keys <= {"tags", "tags-ignore"})


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / ".github" / "workflows"
    files = sorted(list(root.glob("*.yml")) + list(root.glob("*.yaml")))
    missing, checked = [], 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        if not push_to_branches(text):
            continue
        checked += 1
        m = re.search(r"^concurrency:\n((?:[ \t]+.*\n?)*)", text, re.M)
        if not m:
            missing.append(f"{f.name}: runs on a push to a branch but has no top-level concurrency group")
            continue
        body = m.group(1)
        group = re.search(r"^\s+group:\s*(.*)$", body, re.M)
        cancel = re.search(r"^\s+cancel-in-progress:\s*(.*)$", body, re.M)
        # main is never cancelled: a cancelled main commit has no conclusion (unverified)
        cancel_ok = cancel is None or cancel.group(1).strip() == "false" or MAIN_EXCEPTED in cancel.group(1)
        # on main each commit has a group of its own: a group keeps one pending run and
        # cancels an older pending one whatever cancel-in-progress says
        group_ok = group is not None and PER_SHA_ON_MAIN in group.group(1)
        if not group_ok and group is not None and all_jobs_self_hosted(text):
            # one machine: a per-ref group (at most one pending run) with a stated reason
            if "github.ref" in group.group(1) and self_hosted_reason(text):
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
