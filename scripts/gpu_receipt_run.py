#!/usr/bin/env python3
"""Run the GPU receipt suites (scripts/gpu-receipt-suites.txt) on this machine's GPU and write the GPU receipt.

One command on the GPU machine, in a checkout or an export of the tree to land:

    python3 scripts/gpu_receipt_run.py --tree <40-hex tree> --repo ALICE-SDF --run-id <id> --notebook <id> --out <dir>

`--tree` is the tree the files are from (`git rev-parse <commit>^{tree}` where the export was made; the landing check
compares it with the landing commit's tree). The adapter is measured here, on the adapter the suites get: the command
first runs `cargo run --release --features gpu --example sheet_field_adapter_speed -- <UTC date>` and takes its row
(vendor, device, backend, name, driver, ns per tick); a software adapter (lavapipe / llvmpipe, SwiftShader, the
Windows basic render driver: by vendor id 0x10005 / 0x1ae0 / 0x1414 or by name) gets no receipt. The row is written to
<out>/adapter-<run id>.tsv and passed to the suites as ALICE_SDF_ADAPTER_SPEEDS (the row the field is admitted with).
Every listed suite runs with
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
# libtest's result line; with colour on (CARGO_TERM_COLOR / --color always) `ok` / `FAILED` are wrapped in escape codes, which are
# removed before matching (scripts/cargo_output_parser_check.py)
CARGO_OUTPUT_PARSER = True
RESULT = re.compile(r"^test result: (?:ok|FAILED)\. (\d+) passed;", re.M)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


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
    return sum(int(m) for m in RESULT.findall(ANSI.sub("", log)))


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


SOFTWARE_VENDORS = {"0x10005", "0x1ae0", "0x1414"}   # Mesa (lavapipe / llvmpipe), Google SwiftShader, Microsoft WARP
SOFTWARE_NAMES = ("llvmpipe", "lavapipe", "swiftshader", "basic render", "software")


def software_adapter(row: str) -> str | None:
    """why the speed row names a software adapter, or None (vendor id in column 1, name in column 4)"""
    cols = row.split("\t")
    if len(cols) < 6 or not cols[0].startswith("0x"):
        return f"not a speed row: {row!r}"
    if cols[0].lower() in SOFTWARE_VENDORS:
        return f"vendor {cols[0]} is a software adapter"
    name = cols[3].lower()
    hit = next((w for w in SOFTWARE_NAMES if w in name), None)
    return f"adapter {cols[3]!r} is a software adapter ({hit})" if hit else None


def machine_type(run=subprocess.run) -> str:
    """the GPU machine type for the refusal message: the GPU name nvidia-smi reports, else the host's platform"""
    import platform
    try:
        p = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        names = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()] if p.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        names = []
    return ", ".join(dict.fromkeys(names)) if names else f"{platform.system()} {platform.machine()}, no nvidia-smi GPU"


def no_adapter(machine: str, rc: int, stderr: str) -> str:
    """the refusal when the probe gets no adapter row: the first step, before any suite is built (11-27: an A100-SXM4
    notebook cannot create a wgpu device, 'Parent device is lost', while the A100 PCIe and the A6000 can)"""
    tail = " | ".join(ln for ln in stderr.strip().splitlines()[-3:] if ln.strip())
    return (f"gpu-receipt: no hardware adapter on this machine ({machine}): the adapter probe created no GPU device "
            f"(rc {rc}); take the receipt on a machine type that does: {tail}")


def adapter_row(root: Path, env: dict) -> tuple[str, str]:
    """(the speed row the example measures on this machine's adapter, the command and its output for the log)"""
    import datetime
    cmd = ["cargo", "run", "--release", "--features", "gpu", "--example", "sheet_field_adapter_speed", "--",
           datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")]
    p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    rows = [ln for ln in p.stdout.splitlines() if ln.startswith("0x")]
    if p.returncode != 0 or not rows:
        raise SystemExit(no_adapter(machine_type(), p.returncode, p.stderr))
    why = software_adapter(rows[-1])
    if why:
        raise SystemExit(f"gpu-receipt: no receipt on this adapter ({machine_type()}): {why}")
    return rows[-1], f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}rc={p.returncode}\n"


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
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.run_id):
        raise SystemExit(f"gpu-receipt: run id {a.run_id!r} is not a plain name")
    want = suites(root)
    env = dict(os.environ, ALICE_SDF_REQUIRE_GPU="1", ALICE_SDF_VOLUME_READBACK="1")
    row, speed_log = adapter_row(root, env)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    speeds = out / f"adapter-{a.run_id}.tsv"
    speeds.write_text(f"# measured by gpu_receipt_run.py for run {a.run_id}\n{row}\n", encoding="utf-8")
    env["ALICE_SDF_ADAPTER_SPEEDS"] = str(speeds)
    print(f"gpu-receipt: adapter {row.split(chr(9))[3]} ({row.split(chr(9))[0]})")
    got: dict[str, dict] = {}
    log: list[str] = [speed_log]
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
