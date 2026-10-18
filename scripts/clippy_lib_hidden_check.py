#!/usr/bin/env python3
"""No f32 / f64 / f32x8 max or min hides under an allow written for the libm ban.

`clippy::disallowed_methods` is one lint for every entry of
`.github/clippy-lib/clippy.toml`: an `#[allow(clippy::disallowed_methods)]` put on a
function so that it may call the platform libm (a rendering helper, say) also lets
that function call `f32::max` / `f32::min` unseen. This script copies the tracked
tree to a temporary directory, turns every such allow outside `src/raw_minmax.rs`
into an allow of an unrelated lint (so the code still compiles and is linted with
nothing hidden), runs the library clippy there, and fails on any max / min
diagnostic outside `src/raw_minmax.rs`.

The same lint carries the ALICE-Law G8 ban on bare reads of a stored form (serde_json /
bincode): in the rewritten tree every such diagnostic must fall inside a site marked
`#[allow(clippy::disallowed_methods, reason = "stored form <status>: ...")]` (listed by
`scripts/bare_serde_reads.py`), and every such site must still contain one; a read hidden
under an allow written for libm, or a stored-form site that now covers some other call, fails.

It also fails when it compared nothing: no allow was rewritten, or the rewritten
tree produced no `disallowed_methods` diagnostic at all (the allows exist for libm
calls, so unhiding them must show those).

  python3 scripts/clippy_lib_hidden_check.py [--features F]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import bare_serde_reads

ROOT = Path(__file__).resolve().parent.parent
KEEP = "src/raw_minmax.rs"
LINT = "clippy::disallowed_methods"
STAND_IN = "clippy::disallowed_types"
# serde_json / bincode entry points, and serde's own trait methods (`T::deserialize` on a
# serde_json::Value or another Deserializer)
STORED_FORM = re.compile(r"`((?:serde_json|bincode|serde)::[^`]+)`")
MINMAX = re.compile(r"`((?:f32|f64)::(?:max|min)|wide::f32x8::(?:max|min))`")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def neutralise(text: str) -> tuple[str, int]:
    return text.replace(LINT, STAND_IN), text.count(LINT)


def judge(messages: list[dict]) -> tuple[list[str], int]:
    """(max / min diagnostics outside KEEP, number of disallowed_methods diagnostics)"""
    hidden, seen = [], 0
    for msg in messages:
        if (msg.get("code") or {}).get("code") != LINT:
            continue
        seen += 1
        spans = [s for s in msg.get("spans", []) if s.get("is_primary")]
        where = f"{spans[0]['file_name']}:{spans[0]['line_start']}" if spans else "?"
        if MINMAX.search(msg.get("message", "")) and not where.startswith(KEEP):
            hidden.append(f"{where}: {msg['message']}")
    return hidden, seen


def judge_stored(messages: list[dict], sites: dict[str, list[tuple[int, int]]]) -> tuple[list[str], int]:
    """(errors, number of stored-form diagnostics): each must lie in a reviewed site of its file,
    and each reviewed site must contain one"""
    errors, n, used = [], 0, set()
    for msg in messages:
        if (msg.get("code") or {}).get("code") != LINT or not STORED_FORM.search(msg.get("message", "")):
            continue
        n += 1
        spans = [s for s in msg.get("spans", []) if s.get("is_primary")]
        file, line = (spans[0]["file_name"], spans[0]["line_start"]) if spans else ("?", 0)
        hit = [(file, r) for r in sites.get(file, []) if r[0] <= line <= r[1]]
        if hit:
            used.update(hit)
        else:
            errors.append(f"{file}:{line}: {msg['message']} outside a stored-form site (hidden under another allow)")
    for file, ranges in sorted(sites.items()):
        for r in ranges:
            if (file, r) not in used:
                errors.append(f"{file}:{r[0]}: stored-form site with no stored-form read in it")
    return errors, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default="")
    ap.add_argument("--dump", help="write the max / min diagnostics found (JSON lines) here")
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="clippy-lib-hidden-") as tmp:
        dst = Path(tmp)
        rewritten = 0
        stored_sites: dict[str, list[tuple[int, int]]] = {}
        for rel in tracked_files():
            src = ROOT / rel
            if not src.is_file():
                continue
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if rel.startswith("src/") and rel.endswith(".rs") and rel != KEEP:
                original = src.read_text(encoding="utf-8")
                found = [(s["first"], s["last"]) for s in bare_serde_reads.sites(original)]
                if found:
                    stored_sites[rel] = found
                text, n = neutralise(original)
                rewritten += n
                out.write_text(text, encoding="utf-8")
            else:
                shutil.copyfile(src, out)
        if (ROOT / ".cargo").is_dir():  # a local [patch] (never committed)
            shutil.copytree(ROOT / ".cargo", dst / ".cargo")
        if rewritten == 0:
            print("error: no allow of the lint outside src/raw_minmax.rs was found (compared nothing)", file=sys.stderr)
            return 1
        env = dict(os.environ, CLIPPY_CONF_DIR=str(dst / ".github" / "clippy-lib"), CARGO_TERM_COLOR="never")
        env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))
        cmd = ["cargo", "clippy", "--lib", "--message-format=json", "--quiet"]
        if a.features:
            cmd += ["--features", a.features]
        r = subprocess.run(cmd, cwd=dst, env=env, capture_output=True, text=True)
        messages = []
        for line in r.stdout.splitlines():
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("reason") == "compiler-message":
                messages.append(m["message"])
        hidden, seen = judge(messages)
        stored_errors, n_stored = judge_stored(messages, stored_sites)
        if a.dump:
            with open(a.dump, "w", encoding="utf-8") as f:
                for m in messages:
                    if (m.get("code") or {}).get("code") == LINT and MINMAX.search(m.get("message", "")):
                        f.write(json.dumps(m) + "\n")
    if seen == 0:
        print(f"error: {rewritten} allow(s) rewritten but clippy reported no disallowed_methods at all "
              f"(compared nothing; clippy exit {r.returncode})\n{r.stderr[-2000:]}", file=sys.stderr)
        return 1
    if not stored_sites or n_stored == 0:
        print(f"error: {len(stored_sites)} file(s) with stored-form sites, {n_stored} stored-form diagnostic(s) "
              "(compared nothing)", file=sys.stderr)
        return 1
    for h in hidden:
        print(f"error: hidden under an allow: {h}", file=sys.stderr)
    for e in stored_errors:
        print(f"error: {e}", file=sys.stderr)
    if hidden or stored_errors:
        return 1
    print(f"clippy-lib hidden check: {rewritten} allow(s) lifted, {seen} disallowed-method diagnostic(s), "
          f"0 max / min outside {KEEP}, {n_stored} stored-form read(s) all in "
          f"{sum(map(len, stored_sites.values()))} reviewed site(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
