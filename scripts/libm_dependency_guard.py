#!/usr/bin/env python3
"""A direct libm dependency must come with the libm ban.

The transcendental float methods (`f32::sin`, `f64::exp`, ...) are banned by
`disallowed-methods` in clippy.toml, because the platform libm behind them is
not bit-exact across targets. clippy only reaches a libm *free function*
(`libm::sin(x)`) when clippy.toml lists that function by its path. A crate
that does not depend on libm directly cannot name `libm::sin` at all, so the
hole is unreachable there; it reopens without notice the moment `libm` is
added as a direct dependency. This script fails as soon as that happens
without the ban machinery. The file is generic and is kept byte-identical
across the repositories that carry it.

Scope: direct dependencies only. A libm that arrives transitively (another
crate's `libm` feature, e.g. num-traits) is out of scope by design: this
crate cannot name it as `libm::...` without declaring it, so the hole stays
closed until a declaration appears, and then this script fails.

Manifests: every Cargo.toml under the root (root package, workspace members,
nested unpublished crates such as fuzz/). A manifest declares libm when it
names it in

    [dependencies] [dev-dependencies] [build-dependencies]
    (and the legacy dev_dependencies / build_dependencies spellings)
    [target.<cfg>.<any of the above>]  [workspace.dependencies]
    [patch.<registry>]  [replace]

either under the key `libm` (plain, inline table, dotted `libm.version`, or a
`[dependencies.libm]` table) or renamed through `package = "libm"`. The
manifest is parsed with tomllib (the authority) and located line by line; if
the two disagree the manifest is reported as unlocated rather than passed.

When any manifest declares libm, the repository must carry

    - a root clippy.toml, and every clippy.toml / .clippy.toml under the root,
      listing at least one `libm::` path under `disallowed-methods` (clippy
      reads the nearest clippy.toml, so a nested one without the ban would
      replace the root's list for its crate; a clippy.toml under a dot
      directory such as .github/, selected through CLIPPY_CONF_DIR, counts
      too), and
    - scripts/libm_ban_check.py (which checks the ban covers every libm
      function of the resolved version).

When no manifest declares libm, the repository passes.

Skipped directories: target/, node_modules/, .git/, and for manifests any
directory whose name starts with a dot. Output: `path:line: rule` per problem, then the counts.
Exit 0 clean, 1 problems, 2 when no manifest was read.

    python3 scripts/libm_dependency_guard.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    sys.exit("libm_dependency_guard: needs Python 3.11+ (tomllib)")

ROOT = Path(__file__).resolve().parents[1]
CRATE = "libm"
BAN_CHECK = "scripts/libm_ban_check.py"
SKIP_DIRS = {"target", "node_modules"}
DEP_TABLES = {
    "dependencies",
    "dev-dependencies",
    "dev_dependencies",
    "build-dependencies",
    "build_dependencies",
}
FIX = "(copy the libm ban block and scripts/libm_ban_check.py from ALICE-Zip)"


def walk(root: Path, pred, dot_dirs: bool = False) -> list[Path]:
    """Files under `root` matching `pred`; dot directories are entered only
    with `dot_dirs` (and .git never)."""
    out: list[Path] = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for p in entries:
            if p.is_dir():
                if p.name in SKIP_DIRS or p.name == ".git":
                    continue
                if p.name.startswith(".") and not dot_dirs:
                    continue
                stack.append(p)
            elif pred(p):
                out.append(p)
    return sorted(out)


# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------
_KEY_PART = r'(?:"[^"]*"|\'[^\']*\'|[A-Za-z0-9_-]+)'
_KEY_PATH = _KEY_PART + r"(?:\s*\.\s*" + _KEY_PART + r")*"
_HEADER = re.compile(r"^\s*\[\[?\s*(" + _KEY_PATH + r")\s*\]\]?")
_KEYLINE = re.compile(r"^\s*(" + _KEY_PATH + r")\s*=\s*(.*)$")
_PACKAGE_INLINE = re.compile(r"\bpackage\s*=\s*([\"'])" + CRATE + r"\1")
_STRING_VALUE = re.compile(r"^([\"'])" + CRATE + r"\1")


def _parts(path: str) -> list[str]:
    return [p.strip("\"'") for p in re.findall(_KEY_PART, path)]


def _dep(path: list[str]):
    """(kind, key, rest) when `path` reaches into one dependency entry."""
    if len(path) >= 2 and path[0] in DEP_TABLES:
        return path[0], tuple(path[:2]), path[2:]
    if len(path) >= 4 and path[0] == "target" and path[2] in DEP_TABLES:
        return "target-" + path[2], tuple(path[:4]), path[4:]
    if len(path) >= 3 and path[:2] == ["workspace", "dependencies"]:
        return "workspace-dependencies", tuple(path[:3]), path[3:]
    if len(path) >= 3 and path[0] == "patch":
        return "patch", tuple(path[:3]), path[3:]
    if len(path) >= 2 and path[0] == "replace":
        return "replace", tuple(path[:2]), path[2:]
    return None


def _name(key: tuple) -> str:
    return key[-1].split(":", 1)[0]  # [replace] keys are "name:version"


def _semantic(doc: dict) -> set[tuple]:
    """Dependency entries naming libm, from the parsed manifest."""
    hits: set[tuple] = set()

    def table(prefix: tuple, deps):
        if not isinstance(deps, dict):
            return
        for name, spec in deps.items():
            key = prefix + (name,)
            if _name(key) == CRATE:
                hits.add(key)
            elif isinstance(spec, dict) and spec.get("package") == CRATE:
                hits.add(key)

    for t in DEP_TABLES:
        table((t,), doc.get(t))
    for cfg, body in (doc.get("target") or {}).items():
        if isinstance(body, dict):
            for t in DEP_TABLES:
                table(("target", cfg, t), body.get(t))
    table(("workspace", "dependencies"), (doc.get("workspace") or {}).get("dependencies"))
    for reg, body in (doc.get("patch") or {}).items():
        table(("patch", reg), body)
    table(("replace",), doc.get("replace"))
    return hits


def _strip_toml_comment(line: str) -> str:
    quote = None
    escaped = False
    for i, ch in enumerate(line):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def _located(text: str) -> dict[tuple, int]:
    """Dependency entries naming libm, with the first line that names them."""
    found: dict[tuple, int] = {}
    table: list[str] = []
    in_multiline = None
    for n, raw in enumerate(text.splitlines(), 1):
        if in_multiline:
            if raw.count(in_multiline) % 2 == 1:
                in_multiline = None
            continue
        line = _strip_toml_comment(raw)
        for delim in ('"""', "'''"):
            if line.count(delim) % 2 == 1:
                in_multiline = delim
        m = _HEADER.match(line)
        if m:
            table = _parts(m.group(1))
            d = _dep(table)
            if d and not d[2] and _name(d[1]) == CRATE:
                found.setdefault(d[1], n)
            continue
        m = _KEYLINE.match(line)
        if not m:
            continue
        d = _dep(table + _parts(m.group(1)))
        if not d:
            continue
        _, key, rest = d
        value = m.group(2).strip()
        if _name(key) == CRATE:
            found.setdefault(key, n)
        elif not rest and _PACKAGE_INLINE.search(value):
            found.setdefault(key, n)
        elif rest == ["package"] and _STRING_VALUE.match(value):
            found.setdefault(key, n)
    return found


def declarations(path: Path, rel: str) -> tuple[list[int], list[str]]:
    """(lines declaring libm, problems) for one manifest; line 0 stands for a
    declaration the parser finds and the line scan does not."""
    text = path.read_text(encoding="utf-8")
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return [], [f"{rel}:0: unparsable manifest"]
    want = _semantic(doc)
    got = _located(text)
    lines = sorted(got[k] for k in want & set(got)) + [0 for _ in want - set(got)]
    problems = [f"{rel}:0: libm dependency unlocated (parser finds it, line scan does not)"
                for _ in want - set(got)]
    problems += [f"{rel}:{got[k]}: unlocated (line scan finds libm, parser does not)"
                 for k in set(got) - want]
    return lines, problems


# --------------------------------------------------------------------------
# ban machinery
# --------------------------------------------------------------------------
def bans_libm(path: Path) -> bool:
    """True when clippy.toml lists at least one `libm::` disallowed method."""
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    for entry in doc.get("disallowed-methods") or []:
        p = entry.get("path") if isinstance(entry, dict) else entry
        if isinstance(p, str) and p.startswith("libm::"):
            return True
    return False


def check(root: Path) -> tuple[list[str], int, int]:
    manifests = walk(root, lambda p: p.name == "Cargo.toml")
    problems: list[str] = []
    declared: list[str] = []
    for p in manifests:
        rel = p.relative_to(root).as_posix()
        lines, probs = declarations(p, rel)
        problems += probs
        declared += [f"{rel}:{n}" for n in lines]
    n_declare = len({d.rsplit(":", 1)[0] for d in declared})
    if not declared:
        return problems, len(manifests), 0

    clippy = walk(root, lambda p: p.name in ("clippy.toml", ".clippy.toml"), dot_dirs=True)
    root_clippy = root / "clippy.toml"
    if not root_clippy.is_file():
        root_clippy = root / ".clippy.toml"
    unbanned = [c for c in clippy if not bans_libm(c)]
    if not root_clippy.is_file() or unbanned:
        problems += [f"{d}: libm is a direct dependency but clippy.toml bans no "
                     f"libm:: path {FIX}" for d in declared]
        problems += [f"{c.relative_to(root).as_posix()}:0: no libm:: path under "
                     "disallowed-methods" for c in unbanned]
    if not (root / BAN_CHECK).is_file():
        problems += [f"{d}: libm is a direct dependency but {BAN_CHECK} is missing "
                     f"{FIX}" for d in declared]
    return problems, len(manifests), n_declare


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="repository root to check (default: this script's repository)")
    args = ap.parse_args(argv)
    problems, manifests, n_declare = check(args.root.resolve())
    for p in problems:
        print(p, file=sys.stderr)
    print(f"libm_dependency_guard: {manifests} manifests read, {n_declare} declare libm, "
          f"{len(problems)} problems")
    if manifests == 0:
        print("libm_dependency_guard: compared nothing (0 manifests)", file=sys.stderr)
        return 2
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
