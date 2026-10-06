#!/usr/bin/env python3
"""Check that README.md, README_JP.md and docs/MODULES.md agree with the code.

The READMEs avoid version numbers where they can (badges read crates.io,
installation uses `cargo add`). What they still have to state is compared
with its source here, so a change to `Cargo.toml` or `src/lib.rs` that the
docs do not follow fails CI instead of drifting:

  * features  the table after `<!-- readme-sync: features -->` lists exactly
              the `[features]` of Cargo.toml (minus `default`), in each README
              and in the crate-level doc of src/lib.rs (docs.rs shows that one)
  * msrv      every line carrying `<!-- readme-sync: msrv -->` states
              `rust-version` as `**X.Y**`
  * version   any `alice-sdf = "X"` / `version = "X"` dependency line is
              caret-compatible with the package version
  * example   the first ```rust block of each README is the crate-level
              doctest in src/lib.rs, so the README example is compiled
  * modules   docs/MODULES.md lists every `pub mod` of src/lib.rs exactly once
              and nothing else
  * links     every relative link in the three files points at a file that
              exists
  * sections  README_JP.md has the same number of `##` sections as README.md

Every check must compare at least one item; a check that compared nothing
fails, so a renamed marker or table cannot turn the gate into a no-op.

Usage: `python3 scripts/readme_sync.py --check` (exit 1 on any mismatch).
`--root DIR` runs against another tree (used by scripts/test_readme_sync.py).
"""

from __future__ import annotations

import os
import re
import sys

DOCS = ("README.md", "README_JP.md", "docs/MODULES.md")
READMES = ("README.md", "README_JP.md")


def read(root: str, rel: str) -> str:
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return f.read()


def cargo_package(toml: str) -> dict[str, str]:
    pkg = re.search(r"^\[package\]\s*$(.*?)^\[", toml, re.M | re.S)
    body = pkg.group(1) if pkg else ""
    out = {}
    for key in ("version", "rust-version"):
        m = re.search(rf'^{key}\s*=\s*"([^"]+)"', body, re.M)
        if m:
            out[key] = m.group(1)
    return out


def cargo_features(toml: str) -> set[str]:
    sec = re.search(r"^\[features\]\s*$(.*?)(?=^\[|\Z)", toml, re.M | re.S)
    if not sec:
        return set()
    names = re.findall(r'^([A-Za-z0-9_-]+)\s*=', sec.group(1), re.M)
    return {n for n in names if n != "default"}


def table_after(text: str, marker: str) -> list[str] | None:
    i = text.find(marker)
    if i < 0:
        return None
    rows = []
    started = False
    for line in text[i + len(marker):].splitlines():
        if line.startswith("|"):
            started = True
            rows.append(line)
        elif started:
            break
    return rows


def first_cell_names(rows: list[str]) -> list[str]:
    names = []
    for r in rows:
        m = re.match(r"\|\s*`([^`]+)`[^|]*\|", r)  # `std` or `std` (default)
        if m:
            names.append(m.group(1))
    return names


def caret_ok(req: str, version: str) -> bool:
    def parts(v: str) -> list[int]:
        return [int(x) for x in re.findall(r"\d+", v.split("-")[0])][:3]

    r, v = parts(req), parts(version)
    if not r:
        return False
    r += [0] * (3 - len(r))
    v += [0] * (3 - len(v))
    if v < r:
        return False
    if r[0] > 0:
        return v[0] == r[0]
    if r[1] > 0:
        return v[0] == 0 and v[1] == r[1]
    return v[:3] == r[:3]


def lib_doc(lib: str) -> str:
    lines = []
    for line in lib.splitlines():
        if line.startswith("//!"):
            lines.append(line[4:] if line.startswith("//! ") else line[3:])
        elif lines and line.strip():
            break
    return "\n".join(lines)


def first_rust_block(text: str) -> str | None:
    m = re.search(r"^```rust\n(.*?)^```", text, re.M | re.S)
    return m.group(1).rstrip("\n") if m else None


def pub_mods(lib: str) -> list[str]:
    return re.findall(r"^\s*pub mod (\w+)", lib, re.M)


def check(root: str) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    toml = read(root, "Cargo.toml")
    pkg = cargo_package(toml)
    lib = read(root, "src/lib.rs")
    docs = {rel: read(root, rel) for rel in DOCS if os.path.exists(os.path.join(root, rel))}
    for rel in DOCS:
        if rel not in docs:
            errors.append(f"{rel}: missing")

    # features
    want = cargo_features(toml)
    n = 0
    tables = {rel: docs.get(rel, "") for rel in READMES}
    tables["src/lib.rs"] = lib_doc(lib)
    for rel, text in tables.items():
        rows = table_after(text, "<!-- readme-sync: features -->")
        if rows is None:
            errors.append(f"{rel}: no `<!-- readme-sync: features -->` table")
            continue
        got = first_cell_names(rows)
        n += len(got)
        dup = sorted({g for g in got if got.count(g) > 1})
        if dup:
            errors.append(f"{rel}: features listed twice: {dup}")
        if set(got) != want:
            errors.append(
                f"{rel}: features table {sorted(set(got))} != Cargo.toml {sorted(want)}"
                f" (missing {sorted(want - set(got))}, extra {sorted(set(got) - want)})"
            )
    counts["features"] = n

    # msrv
    n = 0
    msrv = pkg.get("rust-version")
    if not msrv:
        errors.append("Cargo.toml: no rust-version")
    for rel in READMES:
        lines = [l for l in docs.get(rel, "").splitlines() if "readme-sync: msrv" in l]
        if not lines:
            errors.append(f"{rel}: no line marked `<!-- readme-sync: msrv -->`")
        for l in lines:
            m = re.search(r"\*\*(\d+\.\d+(?:\.\d+)?)\*\*", l)
            n += 1
            if not m or m.group(1) != msrv:
                errors.append(f"{rel}: MSRV line says {m.group(1) if m else '?'}, Cargo.toml rust-version is {msrv}")
    counts["msrv"] = n

    # version literals (optional: zero is fine, the docs prefer `cargo add`)
    n = 0
    version = pkg.get("version", "")
    pat = re.compile(r'alice-sdf\s*=\s*(?:"([^"]+)"|\{[^}\n]*?version\s*=\s*"([^"]+)")')
    for rel, text in docs.items():
        for m in pat.finditer(text):
            req = m.group(1) or m.group(2)
            n += 1
            if not caret_ok(req, version):
                errors.append(f"{rel}: `alice-sdf = \"{req}\"` does not match package version {version}")
    counts["version"] = n

    # example
    n = 0
    doc = lib_doc(lib)
    for rel in READMES:
        block = first_rust_block(docs.get(rel, ""))
        if block is None:
            errors.append(f"{rel}: no ```rust block")
            continue
        n += 1
        if block not in doc:
            errors.append(f"{rel}: first ```rust block is not the crate-level doctest in src/lib.rs")
    counts["example"] = n

    # modules
    mods = pub_mods(lib)
    listed = []
    if "docs/MODULES.md" in docs:
        for line in docs["docs/MODULES.md"].splitlines():
            m = re.match(r"\|\s*`(\w+)`\s*\|", line)
            if m:
                listed.append(m.group(1))
    dup = sorted({x for x in listed if listed.count(x) > 1})
    if dup:
        errors.append(f"docs/MODULES.md: listed twice: {dup}")
    missing = sorted(set(mods) - set(listed))
    extra = sorted(set(listed) - set(mods))
    if missing:
        errors.append(f"docs/MODULES.md: public modules not listed: {missing}")
    if extra:
        errors.append(f"docs/MODULES.md: listed but not a `pub mod` in src/lib.rs: {extra}")
    counts["modules"] = len(listed)

    # links
    n = 0
    for rel, text in docs.items():
        base = os.path.dirname(os.path.join(root, rel))
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            n += 1
            if not os.path.exists(os.path.normpath(os.path.join(base, path))):
                errors.append(f"{rel}: link target does not exist: {target}")
    counts["links"] = n

    # sections
    secs = {rel: len(re.findall(r"^## ", docs.get(rel, ""), re.M)) for rel in READMES}
    if secs["README.md"] != secs["README_JP.md"]:
        errors.append(f"README.md has {secs['README.md']} `##` sections, README_JP.md has {secs['README_JP.md']}")
    counts["sections"] = secs["README.md"]

    for name, c in counts.items():
        if c == 0 and name != "version":
            errors.append(f"check `{name}` compared nothing")
    return errors, counts


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "--root" in sys.argv:
        root = sys.argv[sys.argv.index("--root") + 1]
    errors, counts = check(root)
    print("compared: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if "--check" in sys.argv and errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
