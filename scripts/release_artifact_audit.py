#!/usr/bin/env python3
"""What a built release artifact gives away: none of the build machine's home paths, private
names, internal operations vocabulary or token-like strings may be in what is shipped.

An artifact is any file handed out (a library, an executable, a wasm module, a web bundle, a
wheel or a plugin archive); a directory is read as every file under it. Containers (zip, tar,
gzip, bzip2, xz, zstd) are opened and their members read, to a depth of 4; one that cannot be
read is a finding, never passed.

The words are ALICE-Law's: its scripts/public_text_check.py, at the revision
.github/public-text-pin names and refused unless its SHA-256 is the one written there (the same
pin scripts/public_text.sh uses), decides what a home path, a private name and an internal word
are (private names are held there as SHA-256 of token n-grams, never as text). On each file
this script runs:

- the checker's build-path reader (`file_findings`): home directories compiled into binary
  bytes, read as they are, as UTF-16 and inside compressed sections (a placeholder's or the CI
  runner's home is not a name);
- the checker's line scan (`scan_text`) on the file's text (for a binary file, its printable
  runs of 6 or more characters, also in UTF-16);
- the token patterns below on the same strings (a credential's shape: a known prefix and its
  length, a private key header, a JSON web token).

A file byte-identical to a file tracked at HEAD in this repository (a README shipped in a
package) is not read again: scripts/public_text.sh judges the tracked file, with the
repository's allowed mentions, and it is counted as such.

A finding is printed as the file, the member and the kind, never the text found. Exit 1 on a
finding, 2 when nothing could be compared (no artifact, no string read, a malformed pin, a
checker whose SHA-256 differs), 0 otherwise.

usage: scripts/release_artifact_audit.py [--checker FILE] [--repo DIR] ARTIFACT_OR_DIR...
  --checker: a copy of the checker (its SHA-256 is still checked against the pin); without it
  the checker is fetched from ALICE-Law at the pinned revision
  --repo: the repository whose tracked files are judged by the public text check (default: this one)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / ".github" / "public-text-pin"
LAW_URL = "https://github.com/ext-sakamoro/ALICE-Law"
DEPTH = 4
BUDGET = 1 << 30  # bytes read out of the containers of one artifact

# the shape of a credential: a provider's fixed prefix and the length of what follows it
TOKEN_PATTERNS = [
    ("github token", re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{36,}")),
    ("github token", re.compile(r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{60,}")),
    ("aws access key", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    ("slack token", re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}")),
    ("api key", re.compile(r"(?<![A-Za-z0-9])sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{32,}")),
    ("hugging face token", re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{34,}")),
    ("crates.io token", re.compile(r"(?<![A-Za-z0-9])cio[A-Za-z0-9]{32}(?![A-Za-z0-9])")),
    ("google api key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_\-]{35}")),
    ("json web token", re.compile(r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
]


class Unusable(Exception):
    """nothing can be compared: exit 2"""


def read_pin() -> tuple[str, str]:
    text = PIN.read_text(encoding="utf-8") if PIN.is_file() else ""
    rev = re.search(r"^rev ([0-9a-f]{40})\s*$", text, re.M)
    sha = re.search(r"^sha256 ([0-9a-f]{64})\s*$", text, re.M)
    if not rev or not sha:
        raise Unusable(f"{PIN.relative_to(ROOT)}: needs 'rev <40 hex>' and 'sha256 <64 hex>'")
    return rev.group(1), sha.group(1)


def fetch_checker(rev: str, into: Path) -> Path:
    repo = into / "law"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", LAW_URL, rev], check=True)
    out = into / "public_text_check.py"
    out.write_bytes(subprocess.run(["git", "-C", str(repo), "show", "FETCH_HEAD:scripts/public_text_check.py"],
                                   check=True, capture_output=True).stdout)
    return out


def load_checker(path: Path, sha: str) -> ModuleType:
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        raise Unusable(f"{path}: sha256 {got}, {PIN.relative_to(ROOT)} says {sha}")
    spec = importlib.util.spec_from_file_location("public_text_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact_files(args: list[str]) -> list[Path]:
    files: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file())
        elif p.is_file():
            files.append(p)
        else:
            raise Unusable(f"{a}: no such file or directory")
    return files


def tracked_blobs(repo: Path) -> set[str]:
    """the object ids of the files tracked at HEAD"""
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "HEAD"], check=True, capture_output=True,
                         text=True).stdout
    return {line.split()[2] for line in out.splitlines() if line.split()[1] == "blob"}


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Audit:
    def __init__(self, checker: ModuleType, tracked: set[str] | None = None) -> None:
        self.c = checker
        self.tracked = tracked or set()
        self.findings: list[str] = []
        self.files = 0
        self.strings = 0
        self.same_as_tracked = 0

    def strings_of(self, data: bytes) -> list[str]:
        return self.c.binary_runs(data) if self.c.is_binary(data) else data.decode("utf-8", errors="replace").splitlines()

    def leaf(self, where: str, name: str, data: bytes) -> None:
        self.files += 1
        if blob_id(data) in self.tracked:
            self.same_as_tracked += 1
            return
        self.findings += self.c.file_findings(where, data, None)
        strings = self.strings_of(data)
        self.strings += len(strings)
        self.findings += self.c.scan_text(where, self.c.text_of(data), None)
        for label, pattern in TOKEN_PATTERNS:
            n = sum(len(pattern.findall(s)) for s in strings)
            if n:
                self.findings.append(f"{where}: {n} {label}")

    def read(self, where: str, name: str, data: bytes, depth: int, budget: list[int]) -> None:
        kind = self.c.container_kind(data, name)
        if kind is None:
            self.leaf(where, name, data)
            return
        if depth >= DEPTH:
            self.findings.append(f"{where}: {kind}, nested too deep: unscanned container")
            return
        try:
            members = self.c._members(data, kind)
        except Exception:  # unknown, damaged, encrypted or unsupported: reported, never passed
            self.findings.append(f"{where}: {kind}: unscanned container")
            return
        if not members:
            self.findings.append(f"{where}: {kind}, empty: unscanned container")
        for member, inner in members:
            budget[0] -= len(inner)
            if budget[0] < 0:
                self.findings.append(f"{where}: {kind}, over {BUDGET >> 20} MiB: unscanned container")
                return
            self.read(f"{where}!{member or '(stream)'}", member, inner, depth + 1, budget)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checker", type=Path)
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("artifacts", nargs="*")
    a = ap.parse_args(argv)
    try:
        files = artifact_files(a.artifacts)
        if not files:
            raise Unusable("no artifact given: nothing compared")
        _, sha = read_pin()
        with tempfile.TemporaryDirectory() as tmp:
            path = a.checker if a.checker else fetch_checker(read_pin()[0], Path(tmp))
            checker = load_checker(path, sha)
        audit = Audit(checker, tracked_blobs(a.repo))
        for f in files:
            audit.read(str(f), f.name, f.read_bytes(), 0, [BUDGET])
        if audit.strings == 0:
            raise Unusable(f"{len(files)} artifacts, {audit.files} files read and no string compared")
    except Unusable as e:
        print(f"release artifact audit: {e}", file=sys.stderr)
        return 2
    for f in audit.findings:
        print(f)
    print(f"{len(files)} artifacts, {audit.files} files ({audit.same_as_tracked} same as a tracked file), "
          f"{audit.strings} strings compared, {len(audit.findings)} findings")
    return 1 if audit.findings else 0


if __name__ == "__main__":
    sys.exit(main())
