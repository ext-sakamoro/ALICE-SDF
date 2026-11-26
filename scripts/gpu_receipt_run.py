#!/usr/bin/env python3
"""Run the GPU receipt suites (scripts/gpu-receipt-suites.txt) on this machine's GPU and write the GPU receipt.

One command on the GPU machine, in a checkout or an export of the tree to land:

    python3 scripts/gpu_receipt_run.py --tree <40-hex tree> --repo ALICE-SDF --run-id <id> --notebook <id> --out <dir>

`--tree` is the tree the files are from (`git rev-parse <commit>^{tree}` where the export was made; the landing check
compares it with the landing commit's tree). The adapter's speed row is the last row of the file named by
ALICE_SDF_ADAPTER_SPEEDS (the measured row the field is admitted with). Every listed suite runs with
`cargo test --release --features asdf,gpu --test <suite>` and ALICE_SDF_REQUIRE_GPU=1 (and
ALICE_SDF_VOLUME_READBACK=1), and its passed count is read from libtest's result lines. The receipt is the directory
<out>/receipts/<repo>-<tree>-<run id>/ with run.log (every suite's command and output, in order) and result.json:
repo, tree, run_id, notebook, adapter (the speed row), run_log_sha256 (of run.log's bytes) and per suite its passed
count and exit code. It is written only when every suite exited 0 with at least its min passed; otherwise nothing
is written and the exit code is 1, with each shortfall printed. A rule file with no suite or no path, or a suite
with a min that is not a positive integer, exits with its reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

RULE = "scripts/gpu-receipt-suites.txt"
RESULT = re.compile(r"^test result: (?:ok|FAILED)\. (\d+) passed;", re.M)


def suites(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    paths = 0
    for n, raw in enumerate((root / RULE).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "path" and len(parts) == 2:
            paths += 1
        elif parts[0] == "suite" and len(parts) == 3 and re.fullmatch(r"[1-9][0-9]*", parts[2]) and parts[1] not in out:
            out[parts[1]] = int(parts[2])
        else:
            raise SystemExit(f"gpu-receipt: {RULE}:{n}: not a path line or a suite line with a positive min: {raw!r}")
    if not out or not paths:
        raise SystemExit(f"gpu-receipt: {RULE} lists no suite or no path")
    return out


def passed(log: str) -> int:
    """the passed count of the test binary's result line (one binary per --test)"""
    return sum(int(m) for m in RESULT.findall(log))


def judge(want: dict[str, int], got: dict[str, dict]) -> list[str]:
    """the shortfalls (empty: a receipt)"""
    why = []
    for name, least in want.items():
        g = got.get(name)
        if g is None:
            why.append(f"{name}: not run")
        elif g["rc"] != 0:
            why.append(f"{name}: rc {g['rc']}")
        elif g["passed"] < least:
            why.append(f"{name}: {g['passed']} passed, under its min {least}")
    return why


def adapter_row() -> str:
    f = os.environ.get("ALICE_SDF_ADAPTER_SPEEDS", "")
    rows = [ln for ln in Path(f).read_text(encoding="utf-8").splitlines() if ln.startswith("0x")] if f else []
    if not rows:
        raise SystemExit("gpu-receipt: ALICE_SDF_ADAPTER_SPEEDS names no file with a measured row")
    return rows[-1]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", a.tree):
        raise SystemExit(f"gpu-receipt: {a.tree!r} is not a tree id")
    root = Path(a.root)
    want = suites(root)
    row = adapter_row()
    env = dict(os.environ, ALICE_SDF_REQUIRE_GPU="1", ALICE_SDF_VOLUME_READBACK="1")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.run_id):
        raise SystemExit(f"gpu-receipt: run id {a.run_id!r} is not a plain name")
    got: dict[str, dict] = {}
    log: list[str] = []
    for name in want:
        cmd = ["cargo", "test", "--release", "--features", "asdf,gpu", "--test", name]
        p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        got[name] = {"passed": passed(p.stdout + p.stderr), "rc": p.returncode}
        log.append(f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}rc={p.returncode}\n")
        print(f"gpu-receipt: {name}: rc {p.returncode}, {got[name]['passed']} passed (min {want[name]})")
    why = judge(want, got)
    if why:
        for w in why:
            print(f"gpu-receipt: no receipt: {w}", file=sys.stderr)
        return 1
    write_receipt(Path(a.out), a.repo, a.tree, a.run_id, a.notebook, row, got, "".join(log))
    return 0


def write_receipt(out: Path, repo: str, tree: str, run_id: str, notebook: str, adapter: str, got: dict,
                  log: str) -> Path:
    """<out>/receipts/<repo>-<tree>-<run id>/ with run.log and result.json (run_log_sha256 = sha256 of run.log)"""
    d = out / "receipts" / f"{repo}-{tree}-{run_id}"
    d.mkdir(parents=True, exist_ok=False)
    data = log.encode("utf-8")
    (d / "run.log").write_bytes(data)
    result = {"repo": repo, "tree": tree, "run_id": run_id, "notebook": notebook, "adapter": adapter,
              "run_log_sha256": hashlib.sha256(data).hexdigest(), "suites": got}
    (d / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"gpu-receipt: wrote {d} ({len(got)} suites)")
    return d


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
