#!/usr/bin/env python3
"""No forbidden call hides under an allow written for the libm ban.

`clippy::disallowed_methods` is one lint for every entry of a clippy.toml: an
`#[allow(clippy::disallowed_methods)]` put on a function or a whole test file so
that it may call the platform libm also lets it call every other forbidden
method unseen. This script copies the tracked tree to a temporary directory,
turns every such allow in the files a mode checks (except the files that mode
keeps) into an allow of an unrelated lint (so the code still compiles and is
linted with nothing hidden), runs clippy there, and fails on any diagnostic of
the mode's pattern outside the kept files.

Modes:

  lib       (default) `cargo clippy --lib` with `.github/clippy-lib/clippy.toml`;
            pattern `f32::max` / `f32::min` / `f64::max` / `f64::min` /
            `wide::f32x8::max` / `wide::f32x8::min`; rewrites src/**.rs;
            keeps src/raw_minmax.rs
  temp-dir  `cargo clippy --all-targets` with the root clippy.toml; pattern
            `std::env::temp_dir` (a fixed name under the temporary directory
            collides between concurrent runs); rewrites src/**.rs and
            tests/**.rs; keeps src/test_dir.rs and tests/common/test_dir.rs,
            the two places every test directory is made. Examples and benches
            are not tests and are not rewritten (an example says per call why
            it writes there)

It also fails when it compared nothing: no allow was rewritten, or the rewritten
tree produced no `disallowed_methods` diagnostic at all (the allows exist for libm
calls, so unhiding them must show those).

  python3 scripts/clippy_lib_hidden_check.py [--mode lib|temp-dir] [--features F]
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

ROOT = Path(__file__).resolve().parent.parent
KEEP = "src/raw_minmax.rs"
LINT = "clippy::disallowed_methods"
STAND_IN = "clippy::disallowed_types"
MINMAX = re.compile(r"`((?:f32|f64)::(?:max|min)|wide::f32x8::(?:max|min))`")
TEMP_DIR = re.compile(r"`std::env::temp_dir`")

MODES = {
    "lib": {
        "pattern": MINMAX,
        "what": "max / min",
        "dirs": ("src/",),
        "keep": (KEEP,),
        "conf_dir": ".github/clippy-lib",
        "targets": ["--lib"],
    },
    "temp-dir": {
        "pattern": TEMP_DIR,
        "what": "std::env::temp_dir",
        "dirs": ("src/", "tests/"),
        "keep": ("src/test_dir.rs", "tests/common/test_dir.rs"),
        "conf_dir": None,
        "targets": ["--all-targets"],
    },
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def neutralise(text: str) -> tuple[str, int]:
    return text.replace(LINT, STAND_IN), text.count(LINT)


def rewrites(rel: str, mode: str = "lib") -> bool:
    """Whether the mode lifts the allows in this tracked file."""
    m = MODES[mode]
    return rel.endswith(".rs") and rel.startswith(m["dirs"]) and rel not in m["keep"]


def judge(messages: list[dict], mode: str = "lib") -> tuple[list[str], int]:
    """(pattern diagnostics outside the kept files, number of disallowed_methods diagnostics)"""
    m = MODES[mode]
    hidden, seen = [], 0
    for msg in messages:
        if (msg.get("code") or {}).get("code") != LINT:
            continue
        seen += 1
        spans = [s for s in msg.get("spans", []) if s.get("is_primary")]
        where = f"{spans[0]['file_name']}:{spans[0]['line_start']}" if spans else "?"
        if m["pattern"].search(msg.get("message", "")) and not where.startswith(m["keep"]):
            hidden.append(f"{where}: {msg['message']}")
    return hidden, seen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=sorted(MODES), default="lib")
    ap.add_argument("--features", default="")
    ap.add_argument("--dump", help="write the pattern diagnostics found (JSON lines) here")
    a = ap.parse_args(argv)
    m = MODES[a.mode]
    with tempfile.TemporaryDirectory(prefix="clippy-lib-hidden-") as tmp:
        dst = Path(tmp)
        rewritten = 0
        for rel in tracked_files():
            src = ROOT / rel
            if not src.is_file():
                continue
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if rewrites(rel, a.mode):
                text, n = neutralise(src.read_text(encoding="utf-8"))
                rewritten += n
                out.write_text(text, encoding="utf-8")
            else:
                shutil.copyfile(src, out)
        if (ROOT / ".cargo").is_dir():  # a local [patch] (never committed)
            shutil.copytree(ROOT / ".cargo", dst / ".cargo")
        if rewritten == 0:
            print(f"error: no allow of the lint outside {', '.join(m['keep'])} was found (compared nothing)",
                  file=sys.stderr)
            return 1
        env = dict(os.environ, CARGO_TERM_COLOR="never")
        if m["conf_dir"]:
            env["CLIPPY_CONF_DIR"] = str(dst / m["conf_dir"])
        else:
            env.pop("CLIPPY_CONF_DIR", None)
        env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))
        cmd = ["cargo", "clippy", *m["targets"], "--message-format=json", "--quiet"]
        if a.features:
            cmd += ["--features", a.features]
        r = subprocess.run(cmd, cwd=dst, env=env, capture_output=True, text=True)
        messages = []
        for line in r.stdout.splitlines():
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("reason") == "compiler-message":
                messages.append(msg["message"])
        hidden, seen = judge(messages, a.mode)
        if a.dump:
            with open(a.dump, "w", encoding="utf-8") as f:
                for msg in messages:
                    if (msg.get("code") or {}).get("code") == LINT and m["pattern"].search(msg.get("message", "")):
                        f.write(json.dumps(msg) + "\n")
    if seen == 0:
        print(f"error: {rewritten} allow(s) rewritten but clippy reported no disallowed_methods at all "
              f"(compared nothing; clippy exit {r.returncode})\n{r.stderr[-2000:]}", file=sys.stderr)
        return 1
    for h in hidden:
        print(f"error: hidden under an allow: {h}", file=sys.stderr)
    if hidden:
        return 1
    print(f"clippy hidden check ({a.mode}): {rewritten} allow(s) lifted, {seen} disallowed-method diagnostic(s), "
          f"0 {m['what']} outside {', '.join(m['keep'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
