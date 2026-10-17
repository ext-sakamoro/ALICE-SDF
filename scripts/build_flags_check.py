#!/usr/bin/env python3
"""The flags cargo actually passes to rustc do not change floating-point results.

ALICE-Law G14 (float-environment-invariants), dynamic half. The static half,
scripts/build_config_guard.py, reads the files a flag can come from; a flag
can also be computed (a variable, an escape, a file read at run time), so this
script runs a cargo command with `-v` in a fresh target directory and reads
the rustc command line of every unit cargo compiles. RUSTFLAGS, .cargo/config
rustflags, profile rustflags and encoded rustflags all appear there as cargo
resolved them. The file is generic and is kept byte-identical across the
repositories that carry it.

    python3 scripts/build_flags_check.py [--crate NAME] -- cargo check --release --lib [...]

The command is run with `-v` and `--target-dir <a new empty directory>`
appended (a unit cargo already built prints `Fresh` instead of its command
line, so a reused directory would hide the flags). Each `Running` line is a
finding when it carries:

    target-cpu-native   -C target-cpu=native (or target_cpu)
    x87-float           a target-feature list that disables sse or sse2
    llvm-fast-math      an LLVM option that relaxes IEEE 754 (enable-unsafe-fp-math,
                        enable-no-nans-fp-math, enable-no-infs-fp-math,
                        enable-no-signed-zeros-fp-math, fp-contract=fast,
                        denormal-fp-math)

The check fails when the command fails, when no rustc command line was seen,
when none was for this crate (the package's library target, or --crate), and
when this crate's unit was `Fresh`. Output: one line per finding, then the
counts. Exit 0 clean, 1 findings or a failed run.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

CARGO_OUTPUT_PARSER = True

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][A-Za-z0-9]")
RUNNING = re.compile(r"^\s*Running\s+`(.*)`\s*$")
FRESH = re.compile(r"^\s*Fresh\s+(\S+)")
CRATE_NAME = re.compile(r"--crate-name[ =](\S+)")
RULES = [
    ("target-cpu-native", re.compile(r"target[-_]cpu=[\"']?native(?![\w-])")),
    ("llvm-fast-math", re.compile(
        r"enable-(?:unsafe|no-nans|no-infs|no-signed-zeros)-fp-math"
        r"|(?<![\w-])-?fp-contract=fast|(?<![\w-])-?denormal-fp-math")),
]
FEATURE_LIST = re.compile(r"target[-_]feature=[\"']?([^\s\"']+)")


def parse(output: str, crate: str) -> tuple[list[str], int, int, bool]:
    """Findings, rustc lines seen, of which for `crate`, and whether `crate` was Fresh."""
    findings = []
    seen = own = 0
    fresh = False
    for raw in output.splitlines():
        line = ANSI.sub("", raw)
        f = FRESH.match(line)
        if f and f.group(1).replace("-", "_") == crate:
            fresh = True
        m = RUNNING.match(line)
        if not m:
            continue
        cmd = m.group(1)
        seen += 1
        name = (CRATE_NAME.search(cmd) or [None, "?"])[1].strip("'\"")
        if name == crate:
            own += 1
        for rule, pattern in RULES:
            for hit in pattern.finditer(cmd):
                findings.append(f"{name}: {rule}: {hit.group(0)}")
        for flist in FEATURE_LIST.finditer(cmd):
            if any(feat in ("-sse", "-sse2") for feat in flist.group(1).split(",")):
                findings.append(f"{name}: x87-float: target-feature={flist.group(1)}")
    return findings, seen, own, fresh


def package_crate(root: Path) -> str:
    if tomllib is None:
        raise SystemExit("build_flags_check: Python 3.11+ (tomllib) is needed")
    data = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    name = data.get("lib", {}).get("name") or data["package"]["name"]
    return name.replace("-", "_")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: build_flags_check.py [--crate NAME] [--root DIR] -- <cargo command>", file=sys.stderr)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crate")
    ap.add_argument("--root", default=Path(__file__).resolve().parent.parent, type=Path)
    args = ap.parse_args(argv[:i])
    command = argv[i + 1:]
    if not command:
        print("build_flags_check: no command given", file=sys.stderr)
        return 2
    if any(a == "--target-dir" or a.startswith("--target-dir=") for a in command):
        print("build_flags_check: the target directory is chosen here (a fresh one); drop --target-dir",
              file=sys.stderr)
        return 2
    root = Path(args.root).resolve()
    crate = (args.crate or package_crate(root)).replace("-", "_")
    target = Path(tempfile.mkdtemp(prefix="build-flags-"))
    try:
        full = command + ["-v", "--target-dir", str(target)]
        print("build_flags_check: running " + " ".join(full), flush=True)
        r = subprocess.run(full, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace")
    finally:
        shutil.rmtree(target, ignore_errors=True)
    findings, seen, own, fresh = parse(r.stdout, crate)
    problems = list(findings)
    if r.returncode != 0:
        print("\n".join(r.stdout.splitlines()[-30:]))
        problems.append(f"the command failed (exit {r.returncode})")
    if seen == 0:
        problems.append("no rustc command line was seen")
    elif own == 0:
        problems.append(f"no rustc command line for {crate}")
    if fresh:
        problems.append(f"{crate} was Fresh: its command line was not shown")
    for p in problems:
        print(p)
    print(f"build_flags_check: {seen} rustc invocations ({own} for {crate}), {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
