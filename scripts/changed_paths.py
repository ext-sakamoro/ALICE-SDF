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
  Each workflow is read by a strict reader of the YAML subset workflows use (block and flow
  style, comments anywhere), not by its lines; a file outside the subset is a finding.
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

# A strict reader for the YAML subset GitHub workflow files use (stdlib only), so sync reads the
# file's structure and not its lines (a comment at any indentation and a flow-style value are what
# they are in YAML, not a line that ends a block or a key that is not seen).
#
# It returns the tree yaml.safe_load builds (YAML 1.1 implicit types: the key `on` is True) for what
# it accepts, and raises FormError for anything outside the subset instead of guessing. The owner
# repository compares it with PyYAML over the workflows of the ALICE repositories in CI.
# Accepted: block mappings and sequences (`- key: v` items included), comments, `---` at the top,
# one-line plain / single-quoted / double-quoted scalars, literal and folded block scalars
# (`|` `|-` `|+` `>` `>-` `>+`), flow sequences and mappings that close on their line.
# Refused: tabs in indentation, anchors / aliases / tags / merge keys, multi-line plain, quoted or
# flow values, explicit indentation indicators, complex keys, duplicate keys, more than one
# document, and plain scalars YAML 1.1 reads as a type this reader does not build (octal / hex /
# binary / sexagesimal ints, exponent / inf / nan floats, timestamps, `=`).

class FormError(ValueError):
    pass


# what PyYAML's reader refuses, and the line breaks it reads besides LF / CRLF (refused here)
NON_PRINTABLE = re.compile("[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]")
OTHER_BREAKS = re.compile("[\r\x85\u2028\u2029]")
NULLS = {"", "~", "null", "Null", "NULL"}
BOOLS = {**{w: True for w in "yes Yes YES true True TRUE on On ON".split()},
         **{w: False for w in "no No NO false False FALSE off Off OFF".split()}}
INT_DEC = re.compile(r"[-+]?(?:0|[1-9][0-9_]*)\Z")
FLOAT_SIMPLE = re.compile(r"[-+]?(?:[0-9][0-9_]*)\.[0-9_]*\Z")
# the other implicit types of YAML 1.1 (PyYAML's resolver): refused, not built
REFUSED_SCALARS = [
    (re.compile(r"[-+]?0b[0-1_]+\Z"), "a binary int"),
    (re.compile(r"[-+]?0[0-7_]+\Z"), "an octal int"),
    (re.compile(r"[-+]?0x[0-9a-fA-F_]+\Z"), "a hex int"),
    (re.compile(r"[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+(?:\.[0-9_]*)?\Z"), "a sexagesimal number"),
    (re.compile(r"[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+][0-9]+\Z"), "a float with an exponent"),
    (re.compile(r"[-+]?\.[0-9_]+(?:[eE][-+][0-9]+)?\Z"), "a float without an integer part"),
    (re.compile(r"[-+]?\.(?:inf|Inf|INF)\Z|\.(?:nan|NaN|NAN)\Z"), "an inf / nan float"),
    (re.compile(r"[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}"), "a timestamp"),
    (re.compile(r"=\Z"), "a value key"),
    (re.compile(r"<<\Z"), "a merge key"),
]


def resolve(plain: str, where: str):
    """the value of a plain scalar, as yaml.safe_load builds it"""
    if plain in NULLS:
        return None
    if plain in BOOLS:
        return BOOLS[plain]
    if INT_DEC.match(plain):
        return int(plain.replace("_", ""))
    if FLOAT_SIMPLE.match(plain):
        return float(plain.replace("_", ""))
    for rx, what in REFUSED_SCALARS:
        if rx.match(plain):
            raise FormError(f"{where}: `{plain}` is {what}")
    return plain


def strip_comment(s: str) -> str:
    """a plain value without its trailing ` # comment`"""
    m = re.search(r"(?:^|[ \t])#", s)
    return (s[:m.start()] if m else s).rstrip()


DQ_ESC = {"0": "\0", "a": "\a", "b": "\b", "t": "\t", "\t": "\t", "n": "\n", "v": "\v", "f": "\f",
          "r": "\r", "e": "\x1b", " ": " ", '"': '"', "/": "/", "\\": "\\", "N": "\x85",
          "_": "\xa0", "L": " ", "P": " "}


def quoted(s: str, where: str) -> tuple[str, str]:
    """(the value of the quoted scalar at the start of s, the rest of s after it)"""
    q = s[0]
    out, i = [], 1
    while i < len(s):
        c = s[i]
        if q == "'" and c == "'":
            if s[i + 1:i + 2] == "'":
                out.append("'")
                i += 2
                continue
            return "".join(out), s[i + 1:]
        if q == '"' and c == '"':
            return "".join(out), s[i + 1:]
        if q == '"' and c == "\\":
            e = s[i + 1:i + 2]
            if e in DQ_ESC:
                out.append(DQ_ESC[e])
                i += 2
                continue
            n = {"x": 2, "u": 4, "U": 8}.get(e)
            if n and re.fullmatch(r"[0-9a-fA-F]+", s[i + 2:i + 2 + n] or "-"):
                out.append(chr(int(s[i + 2:i + 2 + n], 16)))
                i += 2 + n
                continue
            raise FormError(f"{where}: an escape this reader does not read (\\{e})")
        out.append(c)
        i += 1
    raise FormError(f"{where}: a quoted scalar that does not close on its line")


def scalar(s: str, where: str):
    """a one-line scalar value (quoted, flow sequence or plain), comments allowed after it"""
    if not s:
        return None
    c = s[0]
    if c in "'\"":
        v, rest = quoted(s, where)
        if strip_comment(rest):
            raise FormError(f"{where}: text after a quoted scalar")
        return v
    if c in "[{":
        return Flow(s, where).whole()
    if c in "&*!%@`|>]},":
        raise FormError(f"{where}: `{c}` starts a form this reader does not read")
    if re.match(r"[-?:](?:[ \t]|$)", s):
        raise FormError(f"{where}: `{s[0]} ` starts a form this reader does not read")
    v = strip_comment(s)
    if re.search(r":(?:[ \t]|$)", v):
        raise FormError(f"{where}: `: ` inside a plain scalar")
    return resolve(v, where)


class Flow:
    """a flow collection (`[...]` / `{...}`, nesting allowed) that closes on its line"""

    def __init__(self, s: str, where: str):
        self.s, self.i, self.where = s, 0, where

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def node(self, in_map_key: bool = False):
        self.ws()
        if self.i >= len(self.s):
            raise FormError(f"{self.where}: a flow collection that does not close on its line")
        c = self.s[self.i]
        if c == "[":
            return self.seq()
        if c == "{":
            return self.mapping()
        if c in "'\"":
            v, rest = quoted(self.s[self.i:], self.where)
            self.i = len(self.s) - len(rest)
            return v
        if c in "&*!%@`|>?#":
            raise FormError(f"{self.where}: `{c}` in a flow collection")
        if c == "-" and self.s[self.i + 1:self.i + 2] in ("", " ", "\t", ",", "]", "}"):
            raise FormError(f"{self.where}: `-` as an indicator in a flow collection")
        if c == ":":
            raise FormError(f"{self.where}: a flow scalar that starts with `:`")
        # a plain scalar ends at `,` `[` `]` `{` `}` or at `: ` / `:` before a flow indicator
        j = self.i
        while j < len(self.s):
            d = self.s[j]
            if d in ",[]{}":
                break
            if d == ":" and (j + 1 == len(self.s) or self.s[j + 1] in " \t,[]{}"):
                break
            if d == "#" and self.s[j - 1] in " \t":
                raise FormError(f"{self.where}: a comment inside a flow collection")
            if d == "?":
                raise FormError(f"{self.where}: `?` inside a flow collection")
            j += 1
        v = self.s[self.i:j].strip()
        self.i = j
        if not v:
            raise FormError(f"{self.where}: an empty entry in a flow collection")
        return resolve(v, self.where)

    def seq(self) -> list:
        self.i += 1
        out = []
        self.ws()
        if self.s[self.i:self.i + 1] == "]":
            self.i += 1
            return out
        while True:
            v = self.node()
            self.ws()
            if self.s[self.i:self.i + 1] == ":":
                raise FormError(f"{self.where}: a mapping entry inside a flow sequence")
            out.append(v)
            c = self.s[self.i:self.i + 1]
            self.i += 1
            if c == "]":
                return out
            if c != ",":
                raise FormError(f"{self.where}: a flow sequence this reader does not read")
            self.ws()
            if self.s[self.i:self.i + 1] == "]":
                raise FormError(f"{self.where}: a trailing comma in a flow sequence")

    def mapping(self) -> dict:
        self.i += 1
        out: dict = {}
        self.ws()
        if self.s[self.i:self.i + 1] == "}":
            self.i += 1
            return out
        while True:
            k = self.node(in_map_key=True)
            if isinstance(k, (list, dict)):
                raise FormError(f"{self.where}: a complex key in a flow mapping")
            self.ws()
            if self.s[self.i:self.i + 1] != ":":
                raise FormError(f"{self.where}: a flow mapping entry without a value")
            self.i += 1
            if k in out:
                raise FormError(f"{self.where}: a duplicate key {k!r}")
            self.ws()
            out[k] = None if self.s[self.i:self.i + 1] in (",", "}") else self.node()
            self.ws()
            c = self.s[self.i:self.i + 1]
            self.i += 1
            if c == "}":
                return out
            if c != ",":
                raise FormError(f"{self.where}: a flow mapping this reader does not read")
            self.ws()
            if self.s[self.i:self.i + 1] == "}":
                raise FormError(f"{self.where}: a trailing comma in a flow mapping")

    def whole(self):
        v = self.node()
        if strip_comment(self.s[self.i:]):
            raise FormError(f"{self.where}: text after a flow collection")
        return v


class Reader:
    def __init__(self, text: str):
        # a byte order mark at the start is not content (PyYAML drops it too)
        if text.startswith("\ufeff"):
            text = text[1:]
        # a CRLF checkout (Windows) is the same file: YAML reads CRLF as one line break
        text = text.replace("\r\n", "\n")
        m = NON_PRINTABLE.search(text)
        if m:
            raise FormError(f"the character U+{ord(m.group(0)):04X}, which YAML does not allow")
        m = OTHER_BREAKS.search(text)
        if m:
            raise FormError(f"the line break U+{ord(m.group(0)):04X}, which this reader does not read")
        self.lines = text.split("\n")
        self.final_break = bool(self.lines) and self.lines[-1] == ""
        if self.final_break:
            self.lines.pop()

    @staticmethod
    def blank(l: str) -> bool:
        s = l.strip()
        return not s or s.startswith("#")

    def next_content(self, i: int) -> int:
        while i < len(self.lines) and self.blank(self.lines[i]):
            if "\t" in self.lines[i]:
                raise FormError(f"line {i + 1}: a tab outside a block scalar")
            i += 1
        return i

    @staticmethod
    def indent(l: str) -> int:
        return len(l) - len(l.lstrip(" "))

    def read(self):
        i = self.next_content(0)
        if i < len(self.lines) and self.lines[i].rstrip() == "---":
            i = self.next_content(i + 1)
        if i >= len(self.lines):
            return None
        if self.indent(self.lines[i]):
            raise FormError(f"line {i + 1}: the document does not start at column 0")
        v, i = self.block(i, 0)
        i = self.next_content(i)
        if i < len(self.lines):
            raise FormError(f"line {i + 1}: content after the document ({self.lines[i].strip()[:40]!r})")
        return v

    def block(self, i: int, n: int, first: str | None = None):
        """the block node whose first line is i at indent n (`first`: that line's text when it
        follows a `- ` on the same line)"""
        if "\t" in self.lines[i]:
            raise FormError(f"line {i + 1}: a tab outside a block scalar")
        text = first if first is not None else self.lines[i][n:]
        if text == "-" or text.startswith("- "):
            return self.seq(i, n, first)
        return self.mapping(i, n, first)

    def seq(self, i: int, n: int, first: str | None):
        out = []
        while True:
            if "\t" in self.lines[i]:
                raise FormError(f"line {i + 1}: a tab outside a block scalar")
            text = first if first is not None else self.lines[i][n:]
            first = None
            if not (text == "-" or text.startswith("- ")):
                break
            rest = text[1:]
            body = rest.lstrip(" ")
            where = f"line {i + 1}"
            if not body or body.startswith("#"):
                j = self.next_content(i + 1)
                if j < len(self.lines) and self.indent(self.lines[j]) > n:
                    v, i = self.block(j, self.indent(self.lines[j]))
                else:
                    v, i = None, i + 1
            elif self.is_key(body) or body.startswith("- "):
                v, i = self.block(i, n + 1 + (len(rest) - len(body)), body)
            elif body[0] in "|>":
                v, i = self.block_scalar(i, n, body, where)
            else:
                v, i = scalar(body, where), i + 1
                self.no_continuation(i, n, where)
            out.append(v)
            i = self.next_content(i)
            if i >= len(self.lines) or self.indent(self.lines[i]) != n:
                break
        if i < len(self.lines) and self.indent(self.lines[i]) > n:
            raise FormError(f"line {i + 1}: an indentation this reader does not read")
        return out, i

    KEY = re.compile(r"""('(?:[^']|'')*'|"(?:[^"\\]|\\.)*"|[^\s'"#\[\]{},&*!|>%@`?:][^#]*?):(?:[ \t]|$)""")

    def is_key(self, text: str) -> bool:
        return bool(self.KEY.match(text))

    def key_of(self, text: str, where: str):
        m = self.KEY.match(text)
        if not m:
            raise FormError(f"{where}: a line that is not `key: value` ({text.strip()[:40]!r})")
        raw = m.group(1).rstrip()
        if raw[0] in "'\"":
            k, rest = quoted(raw, where)
            if rest.strip():
                raise FormError(f"{where}: a complex key")
        else:
            k = resolve(raw, where)
        return k, text[m.end():].strip()

    def mapping(self, i: int, n: int, first: str | None):
        out: dict = {}
        while True:
            if "\t" in self.lines[i]:
                raise FormError(f"line {i + 1}: a tab outside a block scalar")
            text = first if first is not None else self.lines[i][n:]
            first = None
            where = f"line {i + 1}"
            if text.startswith("- ") or text == "-":
                raise FormError(f"{where}: a sequence item inside a mapping")
            if text.startswith("? "):
                raise FormError(f"{where}: a complex key")
            k, val = self.key_of(text, where)
            if isinstance(k, (list, dict)):
                raise FormError(f"{where}: a complex key")
            if k in out:
                raise FormError(f"{where}: a duplicate key {k!r}")
            if not val or val.startswith("#"):
                j = self.next_content(i + 1)
                if j < len(self.lines) and self.indent(self.lines[j]) > n:
                    v, i = self.block(j, self.indent(self.lines[j]))
                elif j < len(self.lines) and self.indent(self.lines[j]) == n and \
                        (self.lines[j][n:] == "-" or self.lines[j][n:].startswith("- ")):
                    v, i = self.seq(j, n, None)
                else:
                    v, i = None, i + 1
            elif val[0] in "|>":
                v, i = self.block_scalar(i, n, val, where)
            else:
                v, i = scalar(val, where), i + 1
                self.no_continuation(i, n, where)
            out[k] = v
            i = self.next_content(i)
            if i >= len(self.lines) or self.indent(self.lines[i]) != n:
                break
        if i < len(self.lines) and self.indent(self.lines[i]) > n:
            raise FormError(f"line {i + 1}: an indentation this reader does not read")
        return out, i

    def no_continuation(self, i: int, n: int, where: str) -> None:
        """a one-line scalar must not continue on a more indented line (a multi-line scalar)"""
        j = self.next_content(i)
        if j < len(self.lines) and self.indent(self.lines[j]) > n:
            raise FormError(f"{where}: a scalar that continues on the next line")

    def block_scalar(self, i: int, n: int, head: str, where: str):
        m = re.fullmatch(r"([|>])([-+]?)(?:[ \t]+#.*)?", head.rstrip())
        if not m:
            raise FormError(f"{where}: a block scalar header this reader does not read ({head!r})")
        style, chomp = m.groups()
        j = i + 1
        lines = []
        while j < len(self.lines):
            l = self.lines[j]
            if l.strip() and self.indent(l) <= n:
                break
            if not l.strip() and "\t" in l:
                raise FormError(f"line {j + 1}: a tab on a blank line of a block scalar")
            lines.append(l)
            j += 1
        # trailing lines that are only spaces belong to the scalar's trailing breaks
        content = [l for l in lines if l.strip()]
        if not content:
            return ("" if chomp != "+" else "\n" * len(lines)), j
        base = self.indent(content[0])
        if base <= n:
            raise FormError(f"{where}: a block scalar without indented content")
        for l in lines:
            if l.strip():
                break
            if len(l) > base:
                raise FormError(f"{where}: a leading blank line longer than the block scalar's indentation")
        body = []
        for l in lines:
            if l.strip() and self.indent(l) < base:
                raise FormError(f"{where}: a block scalar line less indented than its first")
            if not l.strip() and len(l) > base:
                body.append(l[base:])  # a whitespace-only line keeps its extra spaces
            else:
                body.append(l[base:] if l.strip() else "")
        # trailing empty lines
        k = len(body)
        while k and body[k - 1] == "":
            k -= 1
        main, trail = body[:k], body[k:]
        if style == "|":
            text = "\n".join(main)
        else:
            text = ""
            prev_more = False
            empties = 0
            started = False
            for x in main:
                if x == "":
                    empties += 1
                    continue
                more = x[0] in " \t"
                if not started:
                    text = "\n" * empties + x if empties else x
                    started = True
                elif empties:
                    text += ("\n" if (prev_more or more) else "") + "\n" * empties + x
                elif prev_more or more:
                    text += "\n" + x
                else:
                    text += " " + x
                empties = 0
                prev_more = more
        # the line breaks after the last content line (none after a last line that ends the file
        # without one)
        breaks = len(trail) + (0 if j == len(self.lines) and not self.final_break else 1)
        if chomp == "-":
            return text, j
        if chomp == "+":
            return text + "\n" * breaks, j
        return text + ("\n" if breaks else ""), j


def load_yaml(text: str):
    """the tree of a workflow file, as yaml.safe_load builds it, or FormError"""
    return Reader(text).read()

GATE_RE = re.compile(r"changed_paths\.py\s+gate\b")


def name_list(v, what: str) -> list[str]:
    """a list of strings (a single string is a list of one, as GitHub reads `needs` and `branches`)"""
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return v
    raise FormError(f"{what} is {v!r}, not a string or a list of strings")


def triggers(data) -> dict:
    """`on` as a mapping (YAML 1.1 reads the key `on` as True; a quoted 'on' stays a string)"""
    if not isinstance(data, dict):
        raise FormError("the file is not a mapping")
    on = data[True] if True in data else data.get("on")
    if on is None:
        return {}
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list) and all(isinstance(x, str) for x in on):
        return {x: None for x in on}
    if isinstance(on, dict):
        return on
    raise FormError(f"`on` is {on!r}")


def runs(job) -> list[str]:
    steps = job.get("steps") if isinstance(job, dict) else None
    return [s["run"] for s in steps or [] if isinstance(s, dict) and isinstance(s.get("run"), str)]


def jobs_needs(data) -> dict[str, tuple[list[str], bool]]:
    """job id -> (its `needs`, whether a step runs `changed_paths.py gate`)"""
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, dict):
        return {}
    out = {}
    for jid, job in jobs.items():
        needs = job.get("needs") if isinstance(job, dict) else None
        out[str(jid)] = ([] if needs is None else name_list(needs, f"jobs.{jid}.needs"),
                         any(GATE_RE.search(r) for r in runs(job)))
    return out


def gate_cover(name: str, data) -> list[str]:
    """the gate job needs `changes` and every other job of the workflow"""
    try:
        jobs = jobs_needs(data)
    except FormError as e:
        return [f"{name}: {e}"]
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
    trees: dict[str, object] = {}
    for f in sorted(list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))):
        try:
            data = load_yaml(f.read_text(encoding="utf-8"))
            trees[f.name] = data
            push = triggers(data).get("push")
            if push is None:
                continue
            if not isinstance(push, dict):
                raise FormError(f"on.push is {push!r}")
            br = push.get("branches")
            if br is not None and name_list(br, "on.push.branches") == ["main"]:
                continue
            for key in ("paths", "paths-ignore"):
                if key in push:
                    errs.append(f"{f.name}: on.push has `{key}` (a new branch at an existing SHA "
                                f"skips the run); use a changes job with scripts/changed_paths.py")
        except FormError as e:
            errs.append(f"{f.name}: {e} (this lint reads a strict subset of YAML and fails on the rest)")
    compared = 0
    for p in sorted(pat_dir.glob("*.txt")) if pat_dir.is_dir() else []:
        wf = wf_dir / f"{p.stem}.yml"
        rel = f".github/paths/{p.name}"
        if not wf.is_file():
            errs.append(f"{rel}: no workflow {wf.name}")
            continue
        if wf.name not in trees:
            continue  # unreadable: already reported
        data = trees[wf.name]
        compared += 1
        jobs = data.get("jobs") if isinstance(data, dict) else None
        passes = rf"--patterns-file\s+{re.escape(rel)}(?!\S)"
        if not any(re.search(passes, r) for j in (jobs or {}).values() for r in runs(j)):
            errs.append(f"{wf.name}: does not pass `--patterns-file {rel}` to changed_paths.py")
        errs.extend(gate_cover(wf.name, data))
        try:
            pats = read_patterns(str(p), [])
            compile_patterns(pats)
        except (PatternError, OSError) as e:
            errs.append(f"{rel}: {e}")
            continue
        if rel not in pats:
            errs.append(f"{rel}: does not list itself (a change to the patterns must run the workflow)")
        pr = triggers(data).get("pull_request")
        if isinstance(pr, dict) and "paths" in pr:
            try:
                pr_paths = name_list(pr["paths"], "on.pull_request.paths")
            except FormError as e:
                errs.append(f"{wf.name}: {e}")
                continue
            if pr_paths != pats:
                errs.append(f"{wf.name}: on.pull_request.paths differ from {rel} "
                            f"({pr_paths} != {pats})")
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
