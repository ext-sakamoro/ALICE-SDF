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
steps, not by text: every `run:` step of the `changes` job is a decide call (at least one), and
the gate job has exactly one `run:` step, which is exactly the gate call; the changes job may also
have the one policy step, POLICY_RUN, that ANDs decide with a condition, and at most one step of
each command in DECISION_SCRIPTS, each an exact command shape that only reads the diff or the
environment and writes $GITHUB_OUTPUT; any other step voids the exemption): a per-commit group
would let a burst of pushes queue without bound on that machine, while a per-ref group
keeps at most one pending run (the latest main commit is always run; an older pending one may be
superseded). It must say so in a comment line starting with SELF_HOSTED_REASON and
giving the reason, its group must still name `github.ref`, and it may never cancel
an in-progress run on main (the cancel rule applies to it unchanged).

A run of one event never shares a group with a run of another event while either may cancel:
the group and cancel-in-progress are evaluated (GitHub's expression rules: `&&` / `||` return an
operand, `==` ignores case, `format`) for a push to a ci/** branch, a push to main, a manual run
(workflow_dispatch) on each, and a schedule run on main, all on the same commit, for the events the
workflow has (push counts only when it reaches a branch). Without this a manual run of a workflow
cancels the push run of the same commit (one group per ref off main), and the push run's result is
lost. A context the evaluator does not know is a finding (it does not guess); `inputs.*` is empty
outside a manual run. Fails when no workflow was read.

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
# decision-only steps a changes job may also have, at most one each: the exact command shape (the
# arguments are environment variables, so no command can ride along); each script only reads the
# diff or the environment and writes $GITHUB_OUTPUT. A new entry is a deliberate change here.
DECISION_SCRIPTS = (
    # which installed Unreal engines a run builds (ALICE-SDF's Unreal workflow)
    re.compile(
        r'\s*python3?\s+scripts/unreal_engines\.py\s+--roots\s+"\$ROOTS"\s+--event\s+"\$EVENT"'
        r'\s+--ref\s+"\$REF"\s+--specific\s+"\$SPECIFIC"\s*'
    ),
)


# the events whose runs are compared, on one commit; a manual run has its own run id
SHA_CI, SHA_MAIN = "c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1", "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0"


class Event:
    def __init__(self, name: str, trigger: str, ref: str, sha: str, run_id: str):
        self.name, self.trigger, self.ref, self.sha, self.run_id = name, trigger, ref, sha, run_id


EVENTS = (
    Event("push ci", "push", "refs/heads/ci/x", SHA_CI, "101"),
    Event("push main", "push", "refs/heads/main", SHA_MAIN, "102"),
    Event("workflow_dispatch ci", "workflow_dispatch", "refs/heads/ci/x", SHA_CI, "103"),
    Event("workflow_dispatch main", "workflow_dispatch", "refs/heads/main", SHA_MAIN, "104"),
    Event("schedule main", "schedule", "refs/heads/main", SHA_MAIN, "105"),
)


class EvalError(ValueError):
    pass


TOKEN = re.compile(r"\s*(?:(?P<str>'(?:[^']|'')*')"
                   r"|(?P<num>-?(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))|(?P<op>==|!=|&&|\|\||[!(),])"
                   r"|(?P<name>[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*))")


def tokens(expr: str) -> list[tuple[str, str]]:
    out, i = [], 0
    while i < len(expr):
        if expr[i:].strip() == "":
            break
        m = TOKEN.match(expr, i)
        if not m:
            raise EvalError(f"cannot read {expr[i:]!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        i = m.end()
    return out


def truthy(v) -> bool:
    return v not in ("", 0, False, None)


LITERALS = {"true": True, "false": False, "null": None}
HEX = re.compile(r"[+-]?0[xX][0-9a-fA-F]+")
DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def kind_of(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    return "object"


def to_number(v) -> float:
    """GitHub's conversion for a loose comparison: null 0, true 1, false 0, a text read as a number
    ('' 0, hex 0x.., decimal and exponent forms), anything else NaN"""
    k = kind_of(v)
    if k == "null":
        return 0.0
    if k in ("bool", "number"):
        return float(v)
    if k == "string":
        t = v.strip()
        if t == "":
            return 0.0
        if HEX.fullmatch(t):
            return float(int(t, 16))
        if DECIMAL.fullmatch(t):
            return float(t)
    return float("nan")


def loose_equal(a, b) -> bool:
    ka, kb = kind_of(a), kind_of(b)
    if ka == kb == "string":
        return a.casefold() == b.casefold()
    if ka == kb and ka != "number":
        return a is b if ka == "object" else a == b
    return to_number(a) == to_number(b)   # NaN (an object, a text that is no number) equals nothing


def as_text(v) -> str:
    """how format() writes a value: null as '', booleans as true / false, whole numbers without a point"""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v == v and v.is_integer():
        return str(int(v))
    return str(v)


def format_text(fmt: str, args: list) -> str:
    """format(): {N} takes argument N, {{ and }} are literal braces, anything else is an error"""
    out, i = [], 0
    while i < len(fmt):
        c = fmt[i]
        if c == "{" and fmt.startswith("{{", i):
            out.append("{")
            i += 2
        elif c == "}" and fmt.startswith("}}", i):
            out.append("}")
            i += 2
        elif c == "{":
            m = re.match(r"\{(\d+)\}", fmt[i:])
            if not m:
                raise EvalError(f"format: a lone or malformed brace in {fmt!r}")
            n = int(m.group(1))
            if n >= len(args):
                raise EvalError(f"format: {{{n}}} has no argument in {fmt!r}")
            out.append(as_text(args[n]))
            i += m.end()
        elif c == "}":
            raise EvalError(f"format: a lone closing brace in {fmt!r}")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def evaluate(expr: str, ctx: dict):
    """an expression of GitHub's language over the named contexts: ||, &&, ==, !=, !, (), 'text',
    numbers (decimal, exponent, hex, negative), true / false / null, format() with {{ }} escapes;
    == and != read mixed types as numbers; a name outside ctx is an error"""
    toks, pos = tokens(expr), [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else (None, None)

    def take(value=None):
        kind, v = peek()
        if kind is None or (value is not None and v != value):
            raise EvalError(f"expected {value or 'a value'} in {expr!r}")
        pos[0] += 1
        return kind, v

    def primary():
        kind, v = take()
        if kind == "str":
            return v[1:-1].replace("''", "'")
        if kind == "num":
            body = v.lstrip("-")
            n = int(body, 16) if body[:2] in ("0x", "0X") else (float(body) if any(c in body for c in ".eE")
                                                                else int(body))
            return -n if v.startswith("-") else n
        if kind == "op" and v == "(":
            r = disj()
            take(")")
            return r
        if kind == "op" and v == "!":
            return not truthy(primary())
        if kind == "name" and v == "format":
            take("(")
            args = [disj()]
            while peek() == ("op", ","):
                take(",")
                args.append(disj())
            take(")")
            fmt = args[0]
            if not isinstance(fmt, str):
                raise EvalError("format needs a text")
            return format_text(fmt, args[1:])
        if kind == "name" and v in LITERALS:
            return LITERALS[v]
        if kind == "name":
            if v in ctx:
                return ctx[v]
            if v.startswith("inputs.") and "inputs" in ctx:
                return ctx["inputs"]
            raise EvalError(f"unknown context {v}")
        raise EvalError(f"unexpected {v!r} in {expr!r}")

    def comparison():
        left = primary()
        kind, v = peek()
        if kind == "op" and v in ("==", "!="):
            take()
            right = primary()
            same = loose_equal(left, right)
            return same if v == "==" else not same
        return left

    def conj():
        left = comparison()
        while peek() == ("op", "&&"):
            take()
            right = comparison()
            left = right if truthy(left) else left
        return left

    def disj():
        left = conj()
        while peek() == ("op", "||"):
            take()
            right = conj()
            left = left if truthy(left) else right
        return left

    result = disj()
    if pos[0] != len(toks):
        raise EvalError(f"left over {toks[pos[0]][1]!r} in {expr!r}")
    return result


MARKER = re.compile(r"\$\{\{((?:(?!\}\}).)*)\}\}", re.S)


def interpolate(text: str, ctx: dict):
    """the value of a YAML scalar with ${{ }} markers: the whole value when it is one marker"""
    whole = MARKER.fullmatch(text.strip())
    if whole:
        return evaluate(whole.group(1), ctx)
    if "${{" in MARKER.sub("", text):
        raise EvalError(f"an unclosed marker in {text!r}")
    return MARKER.sub(lambda m: str(evaluate(m.group(1), ctx)), text)


def resolve(conc: dict, data, event: Event) -> tuple[str, bool]:
    """(group, may cancel) of a run of the event"""
    ctx = {"github.workflow": str(data.get("name") or "w"), "github.ref": event.ref, "github.sha": event.sha,
           "github.event_name": event.trigger, "github.run_id": event.run_id,
           "github.ref_name": event.ref.removeprefix("refs/heads/"),
           "inputs": "manual-input" if event.trigger == "workflow_dispatch" else ""}
    group = str(interpolate(str(conc.get("group")), ctx))
    cancel = conc.get("cancel-in-progress", False)
    may_cancel = truthy(interpolate(cancel, ctx)) if isinstance(cancel, str) else bool(cancel)
    return group, may_cancel


def shared_groups(name: str, data, conc: dict) -> list[str]:
    """findings: pairs of runs of different events that share a group while either may cancel"""
    trig = cp.triggers(data)
    have = {e for e in ("workflow_dispatch", "schedule") if e in trig}
    if push_to_branches(data):
        have.add("push")
    events = [e for e in EVENTS if e.trigger in have]
    if len(events) < 2:
        return []
    try:
        resolved = [(e, *resolve(conc, data, e)) for e in events]
    except EvalError as err:
        return [f"{name}: the concurrency group cannot be evaluated ({err})"]
    out = []
    for i, (a, ga, ca) in enumerate(resolved):
        for b, gb, cb in resolved[i + 1:]:
            if ga == gb and (ca or cb):
                out.append(f"{name}: a {a.name} run and a {b.name} run share the group {ga!r} and one may cancel "
                           "the other (key the group by the event, e.g. a manual run's own run id)")
    return out


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


def decision_only(runs: list[str]) -> bool:
    """whether a changes job's run steps only decide: at least one decide call, at most one
    POLICY_RUN, at most one step of each DECISION_SCRIPTS shape, and nothing else"""
    decides = [r for r in runs if DECIDE_RUN.fullmatch(r) and "$(" not in r]
    policies = [r for r in runs if r.strip() == POLICY_RUN]
    scripts = [[r for r in runs if shape.fullmatch(r)] for shape in DECISION_SCRIPTS]
    known = len(decides) + len(policies) + sum(len(s) for s in scripts)
    return (len(decides) >= 1 and len(policies) <= 1 and all(len(s) <= 1 for s in scripts)
            and known == len(runs))


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
        if jid == "changes" and decision_only(runs):
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
            branch_push = push_to_branches(data)
        except cp.FormError as e:
            missing.append(f"{f.name}: {e} (read with the strict reader of scripts/changed_paths.py)")
            continue
        conc = data.get("concurrency") if isinstance(data, dict) else None
        if isinstance(conc, str):
            conc = {"group": conc}
        if isinstance(conc, dict) and "group" in conc:
            missing.extend(shared_groups(f.name, data, conc))
        if not branch_push:
            continue
        checked += 1
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
