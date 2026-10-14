#!/usr/bin/env python3
"""No built binary is tracked in this repository outside an allow list.

A prebuilt library, executable or archive carries the paths of the machine that
built it, goes stale against the source, and nobody can review it. Such files are
built (scripts, CI) instead of committed. A tracked file is a built binary when
- its name ends (in any letter case) in .so / .dylib / .a / .aar / .dll / .lib /
  .exe / .wasm / .node, or it sits in an .xcframework bundle, or
- its first bytes are an ELF, Mach-O (32 / 64 bit, either byte order, fat and fat64),
  PE (`MZ` with `PE\\0\\0` at the offset the DOS header names), WebAssembly or `ar` header.
A file that must stay tracked is listed in `.github/tracked-binaries.txt`, one path
per line with a reason after `#`; an entry whose file is gone fails too, and so does
a tracked file that cannot be read (it is not shown to be text).

Exit 0 when every tracked file was read and none is a binary outside the list,
1 on a finding, 2 when no tracked file was read (nothing compared).

usage: tracked_binaries.py [repository root]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXTENSIONS = (".so", ".dylib", ".a", ".aar", ".dll", ".lib", ".exe", ".wasm", ".node")
BUNDLES = (".xcframework/",)
MAGIC = (b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe",
         b"\xca\xfe\xba\xbf", b"!<arch>\n", b"\x00asm")
FAT = b"\xca\xfe\xba\xbe"  # Mach-O fat, and also a Java class file: told apart by the next 4 bytes
ALLOW = ".github/tracked-binaries.txt"


def allow_list(root: Path) -> dict[str, str]:
    """path -> reason; a line without a reason is an error"""
    f = root / ALLOW
    out: dict[str, str] = {}
    if not f.exists():
        return out
    for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path, _, why = line.partition("#")
        if not why.strip():
            raise ValueError(f"{ALLOW}:{n}: an entry needs a reason after '#'")
        out[path.strip()] = why.strip()
    return out


def is_pe(data: bytes) -> bool:
    if not data.startswith(b"MZ") or len(data) < 0x40:
        return False
    off = int.from_bytes(data[0x3C:0x40], "little")
    return data[off:off + 4] == b"PE\x00\x00"


def is_binary(root: Path, path: str) -> bool:
    """raises OSError when the file cannot be read"""
    low = path.lower()
    if low.endswith(EXTENSIONS) or any(b in low + "/" for b in BUNDLES):
        return True
    p = root / path
    if p.is_symlink() or p.is_dir():
        return False  # (a link's target is checked as its own path; a directory is a submodule)
    with open(p, "rb") as fh:
        head = fh.read(64)
        if head.startswith(b"MZ") and len(head) >= 0x40:
            off = int.from_bytes(head[0x3C:0x40], "little")
            fh.seek(0)
            if is_pe(fh.read(off + 4)):
                return True
    if any(head.startswith(m) for m in MAGIC):
        return True
    # Mach-O fat: the architecture count (small); a Java class has its version there (45 and up)
    return head.startswith(FAT) and len(head) >= 8 and int.from_bytes(head[4:8], "big") < 40


def main(argv: list[str]) -> int:
    root = Path(argv[0] if argv else ".").resolve()
    files = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
                           ).stdout.decode("utf-8", "surrogateescape").split("\0")
    files = [f for f in files if f]
    if not files:
        print("error: no tracked file read (compared nothing)", file=sys.stderr)
        return 2
    try:
        allowed = allow_list(root)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    found, unreadable = [], []
    for f in files:
        try:
            binary = is_binary(root, f)
        except OSError:
            unreadable.append(f)
            continue
        if binary and f not in allowed:
            found.append(f)
    stale = sorted(set(allowed) - set(files))
    for f in found:
        print(f"error: {f}: a built binary is tracked (build it instead, or list it in {ALLOW} with a reason)")
    for f in unreadable:
        print(f"error: {f}: a tracked file could not be read (not shown to be text)")
    for f in stale:
        print(f"error: {ALLOW}: {f} is listed but not tracked")
    if found or unreadable or stale:
        return 1
    print(f"ok: {len(files)} tracked files read, no built binary outside {ALLOW} ({len(allowed)} listed)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
