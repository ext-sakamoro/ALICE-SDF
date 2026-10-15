#!/usr/bin/env python3
"""Lint the public CHANGELOG for structure.

The repository is public; its documents are read by people and by language
models judging whether the project is real. One class of problem is checked
mechanically:

  * changelog   version headings: no duplicates, `[Unreleased]` first,
                released versions in descending semver order, the Cargo.toml
                version either released or covered by `[Unreleased]`; inside
                `[Unreleased]`, each Keep a Changelog category at most once
                and no emoji status markers

The words a public text must not carry (how the work was done, record names,
device names, private addresses, private names held as SHA-256) are checked in
every tracked file and every pushed commit by the ALICE-Law public text checker
(scripts/public_text.sh, at the revision .github/public-text-pin names), not
here, so there is one list and it cannot drift apart from a copy.

Released sections are history and are not rewritten, so the per-category and
emoji checks apply to `[Unreleased]` only.

Every check must compare at least one item; a check that compared nothing
fails, so a renamed file or heading cannot turn the gate into a no-op.

Usage: `python3 scripts/docs_lint.py --check` (exit 1 on any finding).
`--root DIR` runs against another tree (used by scripts/test_docs_lint.py).
"""

from __future__ import annotations

import os
import re
import sys

CATEGORIES = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
EMOJI_RE = re.compile("[⚠✅❌⭐\U0001f300-\U0001faff]")
VERSION_RE = re.compile(r"^## \[([^\]]+)\]", re.M)


def read(root: str, rel: str) -> str:
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return f.read()


def semver_key(v: str) -> tuple:
    core, _, pre = v.partition("-")
    nums = tuple(int(x) for x in re.findall(r"\d+", core)[:3])
    # a pre-release sorts below the release it precedes
    return nums + ((1,) if not pre else (0, tuple(int(x) if x.isdigit() else x for x in re.split(r"[.]", pre))))


def cargo_version(root: str) -> str | None:
    m = re.search(r'^\[package\]\s*$(.*?)^\[', read(root, "Cargo.toml"), re.M | re.S)
    if not m:
        return None
    v = re.search(r'^version\s*=\s*"([^"]+)"', m.group(1), re.M)
    return v.group(1) if v else None


def unreleased_body(changelog: str) -> str | None:
    m = re.search(r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|\Z)", changelog, re.M | re.S)
    return m.group(1) if m else None


def check(root: str) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts = {"versions": 0, "categories": 0}

    if not os.path.exists(os.path.join(root, "CHANGELOG.md")):
        errors.append("CHANGELOG.md: missing")
    else:
        cl = read(root, "CHANGELOG.md")
        heads = VERSION_RE.findall(cl)
        counts["versions"] = len(heads)
        dup = sorted({h for h in heads if heads.count(h) > 1})
        if dup:
            errors.append(f"CHANGELOG.md: version headings appear twice: {dup}")
        if heads and heads[0] != "Unreleased":
            errors.append(f"CHANGELOG.md: the first version heading is [{heads[0]}], not [Unreleased]")
        released = [h for h in heads if h != "Unreleased"]
        for a, b in zip(released, released[1:]):
            if semver_key(a) <= semver_key(b):
                errors.append(f"CHANGELOG.md: [{a}] is listed above [{b}] but is not newer")
        cv = cargo_version(root)
        if cv and cv not in released and "Unreleased" not in heads:
            errors.append(f"CHANGELOG.md: Cargo.toml version {cv} has no section and there is no [Unreleased]")
        if cv and released and cv not in released and semver_key(cv) < semver_key(released[0]):
            errors.append(f"CHANGELOG.md: Cargo.toml version {cv} is older than the newest section [{released[0]}]")
        body = unreleased_body(cl)
        # an empty [Unreleased] is the normal state right after a release
        if body is not None and body.strip():
            cats = re.findall(r"^### (\w+)", body, re.M)
            counts["categories"] = len(cats)
            for c in sorted(set(cats)):
                if c not in CATEGORIES:
                    errors.append(f"CHANGELOG.md [Unreleased]: `### {c}` is not a Keep a Changelog category")
                elif cats.count(c) > 1:
                    errors.append(f"CHANGELOG.md [Unreleased]: `### {c}` appears {cats.count(c)} times (append to one list)")
            for i, line in enumerate(body.splitlines(), 1):
                if EMOJI_RE.search(line):
                    errors.append(f"CHANGELOG.md [Unreleased] line {i}: emoji status marker (use **Behavior change:** / **Breaking:**)")
        else:
            counts["categories"] = 1  # no unreleased work (or just released): nothing to compare, not an error

    for name, c in counts.items():
        if c == 0:
            errors.append(f"check `{name}` compared nothing")
    return errors, counts


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
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
