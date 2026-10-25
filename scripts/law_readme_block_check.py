#!/usr/bin/env python3
"""The preflight half of .github/workflows/law-readme-block.yml: the README "ALICE-Law integration"
block is checked here exactly as the CI job checks it, so a change to alice-law-conformance.toml,
Cargo.toml or the README text cannot leave preflight green and turn that job red.

Same inputs as the job:
  - the ALICE-Law revision named in README.md (`law-integration:begin law=<40 hex>`, the first one),
    which must be the repository's ALICE-Law pin (the `rev` of .github/shared-files-pin), as must
    every block's revision in README.md and README_JP.md: the generator checks a block against the
    revision the block names, so a block left at an older revision would otherwise pass while
    stating that revision's facts,
  - ALICE-Law's scripts/gen-law-readme-block.py *of that revision*,
  - the `check` arguments of the job (`--crate`, `--law-main`, `--lint`), read from the workflow
    file of this repository, so this script carries no crate name of its own,
  - an ALICE-Law clone whose origin/main is current (the job clones afresh; here the clone is
    fetched before every run).

The ALICE-Law clone is $ALICE_LAW_DIR when set, otherwise a blob-less clone of the job's URL kept
under ~/.cache/alice-siblings/ (made on first use, as the job makes it).

Outcomes:
  - neither a workflow nor a block in README.md: prints "n/a" and exits 0 (nothing to check),
  - a workflow and no block, or the generator's check failing, or a pin the clone does not have,
    or a block whose revision is not the repository's ALICE-Law pin, or no readable pin,
    or a fetch / clone failing: exit 1,
  - the generator reporting 0 blocks compared, or no "<n> blocks compared" line: exit 1 (a check
    that compared nothing is not a pass),
  - a workflow whose `check` command cannot be read: exit 2.
The generator compares committed content (git HEAD of this repository), as the job does; when
README.md, README_JP.md, alice-law-conformance.toml or Cargo.toml has uncommitted changes, a pass
says so (those changes are not covered).

usage: scripts/law_readme_block_check.py [--root DIR]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WORKFLOW = Path(".github/workflows/law-readme-block.yml")
LAW_URL = "https://github.com/ext-sakamoro/ALICE-Law.git"
PIN_RE = re.compile(r"law-integration:begin law=([0-9a-f]{40})")
ANY_PIN_RE = re.compile(r"law-integration:begin law=(\S*)")
# the repository's ALICE-Law pin: the revision its shared files, G11 table and README block are checked at
LAW_PIN = Path(".github/shared-files-pin")
LAW_PIN_REV_RE = re.compile(r"(?m)^rev ([0-9a-f]{40})$")
CHECK_RE = re.compile(
    r'gen-law-readme-block\.py"?\s+check\s+--law\s+\S+\s+--crate\s+(?P<crate>\S+)\s*\\?\s*'
    r"--crate-repo\s+\S+\s+--law-main\s+(?P<law_main>\S+)\s+--lint\s+(?P<lint>\S+)"
)
# what the generator reads from this repository (at HEAD)
INPUTS = ("README.md", "README_JP.md", "alice-law-conformance.toml", "Cargo.toml")
COMPARED_RE = re.compile(r"(\d+) blocks? compared, (\d+) problems?")


def fail(msg: str, code: int = 1) -> int:
    print(f"law-readme-block-check: {msg}", file=sys.stderr)
    return code


def git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)


def law_clone() -> tuple[Path | None, str]:
    """(an ALICE-Law clone with a fetched origin, '') or (None, why)"""
    given = os.environ.get("ALICE_LAW_DIR")
    path = Path(given) if given else Path.home() / ".cache" / "alice-siblings" / "ALICE-Law"
    if not (path / ".git").exists() and not (path / "HEAD").exists():
        if given:
            return None, f"ALICE_LAW_DIR={given} is not a git clone"
        path.parent.mkdir(parents=True, exist_ok=True)
        r = git("clone", "--quiet", "--filter=blob:none", LAW_URL, str(path))
        if r.returncode:
            return None, f"cannot clone {LAW_URL}: {r.stderr.strip()}"
    r = git("fetch", "--quiet", "origin", cwd=path)
    if r.returncode:
        return None, f"cannot fetch {path}: {r.stderr.strip()} (the job reads the current origin/main)"
    return path, ""


def uncommitted(root: Path) -> list[str]:
    """the inputs of the block (READMEs, the manifest, Cargo.toml) with uncommitted changes"""
    r = git("status", "--porcelain", "--", *INPUTS, cwd=root)
    return sorted({line[3:].strip() for line in r.stdout.splitlines() if line.strip()}) if r.returncode == 0 else []


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="the repository (default: the current directory)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    wf = root / WORKFLOW
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    pin = PIN_RE.search(text)
    if not wf.exists():
        if pin:
            return fail(f"README.md has a law-integration block but there is no {WORKFLOW} to check it")
        print(f"law-readme-block-check: n/a (no {WORKFLOW} and no law-integration block in README.md)")
        return 0
    m = CHECK_RE.search(wf.read_text(encoding="utf-8"))
    if not m:
        return fail(f"{WORKFLOW}: no `gen-law-readme-block.py check --law .. --crate .. --crate-repo .. "
                    f"--law-main .. --lint ..` command to mirror", 2)
    if not pin:
        return fail("README.md has no law-integration block (the job fails the same way)")
    pin_file = root / LAW_PIN
    law_pin = LAW_PIN_REV_RE.search(pin_file.read_text(encoding="utf-8")) if pin_file.is_file() else None
    if law_pin is None:
        return fail(f"no `rev <40 hex>` line in {LAW_PIN}: the block's revision cannot be compared with "
                    f"the repository's ALICE-Law pin")
    for name in ("README.md", "README_JP.md"):
        f = root / name
        revs = ANY_PIN_RE.findall(f.read_text(encoding="utf-8")) if f.exists() else []
        off = [r for r in revs if r != law_pin.group(1)]
        if off:
            return fail(f"{name}: a block names ALICE-Law {off[0][:12]}, not the repository's pin "
                        f"{law_pin.group(1)[:12]} ({LAW_PIN}); regenerate: gen-law-readme-block.py write "
                        f"--law-ref {law_pin.group(1)[:12]} ...")
    law, why = law_clone()
    if law is None:
        return fail(why)
    rev = pin.group(1)
    if git("cat-file", "-e", f"{rev}^{{commit}}", cwd=law).returncode:
        return fail(f"ALICE-Law has no commit {rev} (the block's pin)")
    gen = git("show", f"{rev}:scripts/gen-law-readme-block.py", cwd=law)
    if gen.returncode:
        return fail(f"ALICE-Law {rev[:12]} has no scripts/gen-law-readme-block.py")
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "gen-law-readme-block.py"
        script.write_text(gen.stdout, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(script), "check", "--law", str(law), "--crate", m["crate"],
             "--crate-repo", str(root), "--law-main", m["law_main"], "--lint", m["lint"]],
            cwd=root, capture_output=True, text=True,
        )
    out = r.stdout + r.stderr
    sys.stdout.write(out)
    counts = COMPARED_RE.findall(out)
    if not counts:
        return fail("the generator printed no `<n> blocks compared` line: nothing is shown compared")
    compared = int(counts[-1][0])
    if compared == 0:
        return fail("0 blocks compared: a check that compared nothing is not a pass")
    if r.returncode:
        return fail(f"the README block differs from ALICE-Law {rev[:12]} (exit {r.returncode}); "
                    f"regenerate: gen-law-readme-block.py write --law-ref {rev[:12]} --crate {m['crate']} ...")
    print(f"law-readme-block-check: {m['crate']} at ALICE-Law {rev[:12]}: {compared} block(s) compared, ok")
    dirty = uncommitted(root)
    if dirty:
        print(f"law-readme-block-check: note: this checked committed HEAD; uncommitted changes to "
              f"{', '.join(dirty)} are not covered (commit, then run again)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
