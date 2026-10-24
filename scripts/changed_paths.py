#!/usr/bin/env python3
"""Decide whether a path-filtered workflow runs, from the files a push or pull request changed.

GitHub's `on.push.paths` diffs only the commits that are new to the repository when a push
creates a branch. A `ci/**` branch created at a SHA another ref already has (its `work/`
twin) gets an empty diff, and the workflow is skipped without a run. This script replaces
that filter: the workflow has no `paths` on `on.push`, a `changes` job runs this script,
and the heavy jobs depend on its `run` output.

Subcommands
-----------
decide   which files changed, and whether any matches the patterns
    --event NAME        github.event_name (push / pull_request / schedule / workflow_dispatch)
    --before SHA        github.event.before (push)
    --after SHA         github.event.after or github.sha (push)
    --created BOOL      github.event.created (push; true when the push created the branch)
    --forced BOOL       github.event.forced (push)
    --base SHA          github.event.pull_request.base.sha (pull_request)
    --head SHA          github.event.pull_request.head.sha (pull_request)
    --main-ref REF      the ref a new branch is compared with (default origin/main)
    --patterns-file F   one pattern per line; `#` comments and blank lines are ignored
    --pattern P         a pattern (repeatable; appended after the file's patterns)
    --repo DIR          the repository (default: current directory)
  The diff:
    schedule, workflow_dispatch          always run (no diff)
    push, created / forced / before=0…0  merge-base(main-ref, after)..after
    push, otherwise                      before..after (the same as the old filter, also on main)
    pull_request                         base...head
  Fail-closed (run=true, the reason is logged): a git failure, a missing merge-base, an
  unreachable `before`, an unknown event, an unsupported pattern, a non-empty diff of which
  no file was compared, and an empty diff on a new / force-pushed branch (the change cannot
  be known). An empty diff on an existing branch skips.
  Output: `run`, `count` (changed files), `matched` and `reason` in $GITHUB_OUTPUT; the
  decision and the file list in the log and in $GITHUB_STEP_SUMMARY. Exit 0. File names
  are read NUL-separated (`git diff -z`) and printed JSON-escaped inside a
  `::stop-commands::` block, so a name cannot be misread or run as a workflow command.

gate     the final job of a workflow: did the jobs that should have run succeed?
    --needs-json JSON   ${{ toJSON(needs) }} (needs the `changes` job and the heavy jobs)
  changes did not succeed: fail. run=true: every other job must be `success`. run=false:
  every other job must be `skipped`. Fails when no heavy job was named. Exit 0 / 1.
  Do NOT make the Gate job a required status check: `paths` stays on `pull_request`, so a
  pull request that touches no listed path gets no run, and a required check would stay
  pending forever.

sync     the trigger lint: no workflow filters `on.push` by paths unless the push runs on
         main only (main is never created by a push, so before..after holds), and the `pull_request`
         paths of a workflow equal its pattern file `.github/paths/<workflow>.txt`, which the
         workflow passes to this script; in such a workflow, the job that runs `gate` needs
         `changes` and every other job (a job added later cannot run outside the gate)
    [ROOT]              the repository root (default: this script's parent's parent)
  Fails when a pattern file was not compared with any workflow, or a workflow with a pattern
  file has no gate job or no job besides `changes` and the gate. Exit 0 / 1.

Patterns (a subset of GitHub's filter pattern syntax)
-----------------------------------------------------
  `*`   any characters except `/`
  `**`  any characters including `/`; `**/` at the start or after `/` also matches no
        directory (`**/src/**` matches `src/a.rs`)
  `?`   one character except `/`
  `!P`  negation: a path is selected when the last pattern that matches it is not negated
        (patterns are read in order, as GitHub does)
  Anything else is literal. `[`, `]`, `+`, `{`, `}` and a leading `\\` are rejected
  (fail-closed run) instead of being matched with a meaning GitHub may not share.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

ZERO = "0" * 40
UNSUPPORTED = set("[]+{}")
SUMMARY_FILES = 200


class PatternError(ValueError):
    pass


def compile_pattern(pat: str) -> re.Pattern[str]:
    """one glob (without its `!`) as an anchored regex"""
    if not pat:
        raise PatternError("empty pattern")
    bad = sorted(set(pat) & UNSUPPORTED)
    if bad or pat.startswith("\\"):
        raise PatternError(f"unsupported character in pattern {pat!r}: {''.join(bad) or chr(92)}")
    out, i, n = [], 0, len(pat)
    while i < n:
        if pat.startswith("**", i):
            at_seg_start = i == 0 or pat[i - 1] == "/"
            if at_seg_start and pat.startswith("**/", i):
                out.append("(?:.*/)?")  # zero or more directories
                i += 3
            else:
                out.append(".*")
                i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    # DOTALL: a file name may contain a newline, and `**` must still cross it
    return re.compile("".join(out) + r"\Z", re.S)


def compile_patterns(pats: list[str]) -> list[tuple[bool, re.Pattern[str]]]:
    rules = []
    for p in pats:
        neg = p.startswith("!")
        rules.append((neg, compile_pattern(p[1:] if neg else p)))
    if not rules:
        raise PatternError("no pattern")
    if all(neg for neg, _ in rules):
        raise PatternError("only negated patterns (nothing can be selected)")
    return rules


def selected(path: str, rules: list[tuple[bool, re.Pattern[str]]]) -> bool:
    keep = False
    for neg, rx in rules:
        if rx.match(path):
            keep = not neg
    return keep


def read_patterns(file: str | None, extra: list[str]) -> list[str]:
    pats: list[str] = []
    if file:
        for line in Path(file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pats.append(line)
    return pats + list(extra)


class GitError(RuntimeError):
    pass


def git(repo: str, *args: str) -> str:
    """stdout as text, without newline translation (a file name may contain CR)"""
    try:
        p = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=False)
    except OSError as e:
        raise GitError(f"git {' '.join(args)}: {e}") from e
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace").strip()[:300]
        raise GitError(f"git {' '.join(args)}: exit {p.returncode}: {err}")
    return p.stdout.decode("utf-8", "surrogateescape")


def diff_names(repo: str, spec: str) -> list[str]:
    # -z: names are NUL-separated and never C-quoted (without it git quotes a name with a
    # tab, quote, backslash, newline or control character, and `src/**` misses `"src/a\tb"`)
    # --no-renames: a rename lists both the old and the new name (with renames on,
    # src/x.rs -> docs/x.rs lists docs/x.rs only and `src/**` misses it)
    out = git(repo, "diff", "-z", "--name-only", "--no-renames", spec)
    return [n for n in out.split("\0") if n]


def truthy(v: str | None) -> bool:
    return (v or "").strip().lower() == "true"


def is_zero(sha: str | None) -> bool:
    return not sha or set(sha.strip()) == {"0"}


class Decision:
    def __init__(self, run: bool, reason: str, files: list[str] | None = None,
                 matched: list[str] | None = None, diff: str = ""):
        self.run, self.reason, self.diff = run, reason, diff
        self.files = files or []
        self.matched = matched or []


def decide(a: argparse.Namespace) -> Decision:
    ev = (a.event or "").strip()
    if ev in ("schedule", "workflow_dispatch"):
        return Decision(True, f"{ev}: always runs")
    try:
        rules = compile_patterns(read_patterns(a.patterns_file, a.pattern or []))
    except (PatternError, OSError) as e:
        return Decision(True, f"fail-closed: patterns: {e}")
    new_branch = False
    try:
        if ev == "push":
            if is_zero(a.after):
                return Decision(True, "fail-closed: push without an after SHA")
            after = git(a.repo, "rev-parse", "--verify", f"{a.after}^{{commit}}").strip()
            if truthy(a.created) or truthy(a.forced) or is_zero(a.before):
                new_branch = True
                why = ("created" if truthy(a.created) else "forced" if truthy(a.forced)
                       else "before is 0")
                base = git(a.repo, "merge-base", a.main_ref, after).strip()
                if not base:
                    return Decision(True, f"fail-closed: no merge-base of {a.main_ref} and {after}")
                spec = f"{base}..{after}"
                label = f"push ({why}): merge-base({a.main_ref})..after = {spec}"
            else:
                before = git(a.repo, "rev-parse", "--verify", f"{a.before}^{{commit}}").strip()
                spec = f"{before}..{after}"
                label = f"push: before..after = {spec}"
        elif ev == "pull_request":
            if is_zero(a.base) or is_zero(a.head):
                return Decision(True, "fail-closed: pull_request without base / head SHA")
            spec = f"{a.base}...{a.head}"
            label = f"pull_request: base...head = {spec}"
        else:
            return Decision(True, f"fail-closed: unknown event {ev!r}")
        files = diff_names(a.repo, spec)
    except GitError as e:
        return Decision(True, f"fail-closed: {e}")
    if not files:
        if new_branch:
            return Decision(True, f"fail-closed: empty diff on a new / force-pushed branch ({label})",
                            diff=label)
        return Decision(False, f"skip: no file changed ({label})", diff=label)
    compared = 0
    matched = []
    for f in files:
        compared += 1
        if selected(f, rules):
            matched.append(f)
    if compared == 0:  # unreachable with a non-empty list; kept as the stated invariant
        return Decision(True, "fail-closed: changed files but none compared", files, diff=label)
    if matched:
        return Decision(True, f"run: {len(matched)} of {len(files)} changed files match ({label})",
                        files, matched, label)
    return Decision(False, f"skip: none of {len(files)} changed files matches ({label})",
                    files, matched, label)


def one_line(text: str) -> str:
    """a log line that cannot carry a workflow command on a following line"""
    return json.dumps(text)[1:-1]


def report(d: Decision) -> None:
    head = f"changed_paths: run={'true' if d.run else 'false'} count={len(d.files)} matched={len(d.matched)}"
    print(head)
    print(f"reason: {one_line(d.reason)}")
    # a file name is attacker-controlled text: print it JSON-escaped (the line starts with
    # `+ "` or `- "`, newlines and control characters are escaped), and stop the runner from
    # reading workflow commands (`::warning::` and the like) while the list is printed
    token = secrets.token_hex(16)
    print(f"::stop-commands::{token}")
    for f in d.files:
        print(f"{'+' if f in d.matched else '-'} {json.dumps(f)}")
    print(f"::{token}::")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        reason = one_line(d.reason)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"run={'true' if d.run else 'false'}\ncount={len(d.files)}\n"
                     f"matched={len(d.matched)}\nreason={reason}\n")
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(f"### changed paths\n\n`{head}`\n\n{d.reason}\n\n")
            if d.files:
                fh.write("| match | file |\n|---|---|\n")
                for f in d.files[:SUMMARY_FILES]:
                    fh.write(f"| {'yes' if f in d.matched else ''} | `{json.dumps(f)}` |\n")
                if len(d.files) > SUMMARY_FILES:
                    fh.write(f"\n... {len(d.files) - SUMMARY_FILES} more\n")
            fh.write("\n")


def gate(needs_json: str) -> tuple[bool, str]:
    try:
        needs = json.loads(needs_json)
    except json.JSONDecodeError as e:
        return False, f"needs is not JSON: {e}"
    ch = needs.get("changes") if isinstance(needs, dict) else None
    if not isinstance(ch, dict):
        return False, "no `changes` job in needs"
    if ch.get("result") != "success":
        return False, f"changes job did not succeed ({ch.get('result')})"
    run = (ch.get("outputs") or {}).get("run")
    if run not in ("true", "false"):
        return False, f"changes job has no run output ({run!r})"
    heavy = {k: (v or {}).get("result") for k, v in needs.items() if k != "changes"}
    if not heavy:
        return False, "no heavy job in needs (compared nothing)"
    want = "success" if run == "true" else "skipped"
    bad = {k: r for k, r in heavy.items() if r != want}
    listing = ", ".join(f"{k}={r}" for k, r in sorted(heavy.items()))
    if bad:
        return False, f"run={run}: expected every job {want}, got {listing}"
    return True, f"run={run}: {listing}"


# --- sync: the trigger lint -------------------------------------------------------------

def on_block(text: str) -> list[str]:
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if re.match(r"^(on|\"on\"|'on'):\s*$", l))
    except StopIteration:
        return []
    block = []
    for l in lines[start + 1:]:
        if re.match(r"^\S", l):
            break
        block.append(l)
    return block


def sub_block(block: list[str], key: str, indent: int) -> list[str] | None:
    pre = " " * indent
    for i, l in enumerate(block):
        if re.match(rf"^{pre}{re.escape(key)}:\s*$", l):
            body = []
            for m in block[i + 1:]:
                if m.strip() and not m.startswith(pre + " "):
                    break
                body.append(m)
            return body
    return None


def list_items(body: list[str]) -> list[str]:
    out = []
    for l in body:
        m = re.match(r"^\s*-\s*(.+?)\s*$", l)
        if m:
            v = m.group(1)
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            out.append(v)
    return out


def push_branches(push: list[str]) -> list[str] | None:
    """the `branches` of an on.push block (None when it has none: every branch)"""
    for i, l in enumerate(push):
        m = re.match(r"^    branches:\s*(.*?)\s*$", l)
        if not m:
            continue
        if m.group(1).startswith("["):
            return [x.strip().strip("'\"") for x in m.group(1).strip("[]").split(",") if x.strip()]
        body = []
        for b in push[i + 1:]:
            if b.strip() and not b.startswith("      "):
                break
            body.append(b)
        return list_items(body)
    return None


def jobs_needs(text: str) -> dict[str, tuple[list[str], bool]]:
    """job id -> (its `needs`, whether it runs `changed_paths.py gate`), from the `jobs:` block"""
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if re.match(r"^jobs:\s*$", l))
    except StopIteration:
        return {}
    out: dict[str, tuple[list[str], bool]] = {}
    cur: str | None = None
    body: list[str] = []

    def close() -> None:
        if cur is None:
            return
        needs: list[str] = []
        for i, l in enumerate(body):
            m = re.match(r"^    needs:\s*(.*?)\s*$", l)
            if not m:
                continue
            v = m.group(1)
            if v.startswith("["):
                needs = [x.strip().strip("'\"") for x in v.strip("[]").split(",") if x.strip()]
            elif v:
                needs = [v.strip("'\"")]
            else:
                sub = []
                for b in body[i + 1:]:
                    if b.strip() and not b.startswith("      "):
                        break
                    sub.append(b)
                needs = list_items(sub)
        gate_job = any(re.search(r"changed_paths\.py\s+gate\b", l) for l in body)
        out[cur] = (needs, gate_job)

    for l in lines[start + 1:]:
        if re.match(r"^\S", l):
            break
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", l)
        if m:
            close()
            cur, body = m.group(1), []
        elif cur is not None:
            body.append(l)
    close()
    return out


def gate_cover(name: str, text: str) -> list[str]:
    """the gate job needs `changes` and every other job of the workflow"""
    jobs = jobs_needs(text)
    gates = [j for j, (_, g) in jobs.items() if g]
    if len(gates) != 1:
        return [f"{name}: {len(gates)} jobs run `changed_paths.py gate` (want exactly 1)"]
    gate_id = gates[0]
    others = [j for j in jobs if j not in (gate_id, "changes")]
    if "changes" not in jobs:
        return [f"{name}: no `changes` job"]
    if not others:
        return [f"{name}: no job besides `changes` and the gate (the gate checks nothing)"]
    need = set(jobs[gate_id][0])
    missing = [j for j in ["changes", *others] if j not in need]
    if missing:
        return [f"{name}: the gate job `{gate_id}` does not need {missing} "
                f"(a job outside the gate's needs can fail or be skipped unseen)"]
    return []


def sync(root: Path) -> tuple[list[str], int]:
    wf_dir, pat_dir = root / ".github" / "workflows", root / ".github" / "paths"
    errs: list[str] = []
    for f in sorted(list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))):
        push = sub_block(on_block(f.read_text(encoding="utf-8")), "push", 2)
        if push is not None and push_branches(push) != ["main"]:
            for key in ("paths", "paths-ignore"):
                if sub_block(push, key, 4) is not None:
                    errs.append(f"{f.name}: on.push has `{key}` (a new branch at an existing SHA "
                                f"skips the run); use a changes job with scripts/changed_paths.py")
    compared = 0
    for p in sorted(pat_dir.glob("*.txt")) if pat_dir.is_dir() else []:
        wf = wf_dir / f"{p.stem}.yml"
        rel = f".github/paths/{p.name}"
        if not wf.is_file():
            errs.append(f"{rel}: no workflow {wf.name}")
            continue
        text = wf.read_text(encoding="utf-8")
        compared += 1
        if f"--patterns-file {rel}" not in text:
            errs.append(f"{wf.name}: does not pass `--patterns-file {rel}` to changed_paths.py")
        errs.extend(gate_cover(wf.name, text))
        try:
            pats = read_patterns(str(p), [])
            compile_patterns(pats)
        except (PatternError, OSError) as e:
            errs.append(f"{rel}: {e}")
            continue
        if rel not in pats:
            errs.append(f"{rel}: does not list itself (a change to the patterns must run the workflow)")
        pr = sub_block(on_block(text), "pull_request", 2)
        if pr is not None:
            paths = sub_block(pr, "paths", 4)
            if paths is not None and list_items(paths) != pats:
                errs.append(f"{wf.name}: on.pull_request.paths differ from {rel} "
                            f"({list_items(paths)} != {pats})")
    return errs, compared


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide")
    for k in ("event", "before", "after", "created", "forced", "base", "head", "patterns-file"):
        d.add_argument(f"--{k}", default="")
    d.add_argument("--pattern", action="append", default=[])
    d.add_argument("--main-ref", default="origin/main")
    d.add_argument("--repo", default=".")
    g = sub.add_parser("gate")
    g.add_argument("--needs-json", required=True)
    s = sub.add_parser("sync")
    s.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args(argv)
    if a.cmd == "decide":
        report(decide(a))
        return 0
    if a.cmd == "gate":
        ok, msg = gate(a.needs_json)
        print(f"gate: {'pass' if ok else 'FAIL'}: {one_line(msg)}")
        summ = os.environ.get("GITHUB_STEP_SUMMARY")
        if summ:
            with open(summ, "a", encoding="utf-8") as fh:
                fh.write(f"gate: {'pass' if ok else 'FAIL'}: {msg}\n")
        return 0 if ok else 1
    errs, compared = sync(Path(a.root))
    for e in errs:
        print(e)
    if compared == 0:
        print("error: no pattern file compared with a workflow (compared nothing)", file=sys.stderr)
        return 1
    if errs:
        return 1
    print(f"ok: {compared} pattern files match their workflows, each gate needs every job; "
          f"no on.push paths filter on a branch other than main")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
