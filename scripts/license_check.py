#!/usr/bin/env python3
"""Every manifest in this repository declares the licence it actually ships under.

Every crate and package of this repository is `LicenseRef-Proprietary` (copies received
earlier keep the licence they were received under; that is history, not a current claim).
A path that must declare another value would be allow-listed explicitly, by exact (path,
declared value) pair -- an unexpected value at an allow-listed path still fails. The list
is empty: every manifest declares `LicenseRef-Proprietary`.

Checks:
- every tracked Cargo.toml / package.json / pyproject.toml's `license` field equals
  `LicenseRef-Proprietary`, except the allow-list below;
- no file of an earlier licence (`LICENSE-MIT`, `LICENSE-APACHE`, `LICENSE-COMMUNITY`)
  anywhere in the tree;
- no tracked file holds the MIT licence grant text itself (its opening sentence, or a
  `MIT License` heading on its own line) -- a *reference* to the historical dual licence
  in prose (`MIT OR Apache-2.0`) is not this, and is allowed anywhere;
- `scripts/license_copy_check.py` (every sub-package's copy byte-identical to the root
  LICENSE-Proprietary) is folded in here rather than re-implemented;
- 0 manifests read, 0 tracked files read, or 0 licence declarations found each fail
  (the check must not pass by comparing nothing).

usage: license_check.py [--self-test]
Exit 1 on a violation, 2 when nothing was read.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROPRIETARY = "LicenseRef-Proprietary"

# exact (tracked path, declared value) pairs that may differ from LicenseRef-Proprietary
ALLOWED: dict[str, str] = {}

# file names of the licences earlier versions were published under
OLD_LICENCE_FILES = ("LICENSE-MIT", "LICENSE-APACHE", "LICENSE-COMMUNITY")

# manifest kind -> (filename to match, regex over the file's text capturing the
# declared value in group 1); `pyproject.toml`'s PEP 621 table form is tried first,
# its legacy plain-string form second
MANIFEST_PATTERNS: dict[str, list[re.Pattern]] = {
    "Cargo.toml": [re.compile(r'^license\s*=\s*"([^"]+)"', re.M)],
    "package.json": [re.compile(r'"license"\s*:\s*"([^"]+)"')],
    "pyproject.toml": [
        re.compile(r'^license\s*=\s*\{\s*text\s*=\s*"([^"]+)"\s*\}', re.M),
        re.compile(r'^license\s*=\s*"([^"]+)"', re.M),
    ],
}

# the MIT grant text's opening sentence, or the bare "MIT License" title line -- never
# the word "MIT" alone, which also appears in allowed historical prose
# (split across two literals so this file does not hold the sentence itself)
MIT_TEXT = re.compile(r"Permission is hereby granted, free" r" of charge|^\s*MIT" r" License\s*$", re.M)

# tracked paths that hold someone else's code under its own licence (none today)
THIRD_PARTY: tuple[str, ...] = ()


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def declarations(paths: list[str]) -> list[tuple[str, str]]:
    """[(tracked path, declared licence value)] for every manifest that has one."""
    out = []
    for rel in paths:
        name = Path(rel).name
        pats = MANIFEST_PATTERNS.get(name)
        if not pats:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8")
        for pat in pats:
            m = pat.search(text)
            if m:
                out.append((rel, m.group(1)))
                break
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2

    paths = tracked()
    if not paths:
        print("error: read 0 tracked files", file=sys.stderr)
        return 2

    decls = declarations(paths)
    if not decls:
        print("error: read 0 licence declarations", file=sys.stderr)
        return 2

    bad: list[str] = []

    for rel, value in decls:
        want = ALLOWED.get(rel, PROPRIETARY)
        if value != want:
            bad.append(f"{rel}: license = {value!r}, expected {want!r}")

    for rel in paths:
        if Path(rel).name in OLD_LICENCE_FILES:
            bad.append(f"{rel} exists (the licence is {PROPRIETARY})")

    mit_text_hits = 0
    for rel in paths:
        if THIRD_PARTY and rel.startswith(THIRD_PARTY):
            continue
        p = ROOT / rel
        if not p.is_file():
            continue
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            continue  # binary, not a licence text
        if MIT_TEXT.search(data.decode("utf-8", "replace")):
            mit_text_hits += 1
            bad.append(f"{rel} holds the MIT licence text")

    try:
        import license_copy_check
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import license_copy_check
    copy_rc = license_copy_check.main()
    if copy_rc != 0:
        bad.append("scripts/license_copy_check.py failed (see its own output above)")

    print(
        f"licenses: read {len(paths)} tracked files, {len(decls)} declaration(s), "
        f"{mit_text_hits} licence-text scan hit(s), {len(bad)} violation(s)"
    )
    for b in bad:
        print(f"error: {b}", file=sys.stderr)
    return 1 if bad else 0


def self_test() -> int:
    """Fixed, fake (path, text) scenarios -- never touches this repo's real tree."""
    import tempfile

    def run_against(tmp: Path, manifests: dict[str, str], other_files: dict[str, str],
                     allowed: dict[str, str] | None = None) -> int:
        for rel, body in {**manifests, **other_files}.items():
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        # a trivial, genuinely matching copy so the folded-in license_copy_check call
        # exercises its real pass path instead of being faked around (its own 0-compared
        # guard would otherwise fire if COPIES were emptied for this fake tree)
        (tmp / "LICENSE-Proprietary").write_text("X\n", encoding="utf-8")
        (tmp / "fake-pkg").mkdir(parents=True, exist_ok=True)
        (tmp / "fake-pkg" / "LICENSE.md").write_text("X\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(tmp), "add", "-A"], check=True, capture_output=True)

        global ROOT, ALLOWED
        saved_root, saved_allowed = ROOT, dict(ALLOWED)
        try:
            ROOT = tmp
            ALLOWED = allowed if allowed is not None else {}
            # license_copy_check reads its own module-level ROOT/COPIES too; point it at
            # this fake tree's genuinely-matching copy
            import license_copy_check as lcc

            saved_lcc = (lcc.ROOT, lcc.MASTER, list(lcc.COPIES))
            lcc.ROOT, lcc.MASTER, lcc.COPIES = tmp, tmp / "LICENSE-Proprietary", ["fake-pkg/LICENSE.md"]
            try:
                return main([])
            finally:
                lcc.ROOT, lcc.MASTER, lcc.COPIES = saved_lcc
        finally:
            ROOT, ALLOWED = saved_root, saved_allowed

    base_manifests = {
        "Cargo.toml": 'license = "LicenseRef-Proprietary"\n',
        "crates/a/Cargo.toml": 'license     = "LicenseRef-Proprietary"\n',  # aligned spacing
        "bindings/npm/package.json": '{\n  "license": "LicenseRef-Proprietary"\n}\n',
        "pyproject.toml": 'license = "LicenseRef-Proprietary"\n',
    }

    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        assert run_against(t, base_manifests, {}) == 0, "an all-LicenseRef-Proprietary tree must pass"

    # mutation 0: each earlier licence, in each manifest kind, must fail
    for rel, body in [("Cargo.toml", 'license = "Apache-2.0"\n'),
                      ("crates/a/Cargo.toml", 'license = "AGPL-3.0-or-later"\n'),
                      ("bindings/npm/package.json", '{\n  "license": "SEE LICENSE IN LICENSE.md"\n}\n'),
                      ("pyproject.toml", 'license = { text = "Apache-2.0" }\n')]:
        with tempfile.TemporaryDirectory() as d:
            t = Path(d)
            m = dict(base_manifests)
            m[rel] = body
            assert run_against(t, m, {}) == 1, f"{body.strip()} in {rel} must fail"

    # mutation 1: a stray MIT in a Cargo.toml, aligned spacing (`license     =`)
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        m = dict(base_manifests)
        m["crates/a/Cargo.toml"] = 'license     = "MIT"\n'
        assert run_against(t, m, {}) == 1, "a stray MIT in an aligned Cargo.toml field must fail"

    # mutation 2: a stray MIT in package.json
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        m = dict(base_manifests)
        m["bindings/npm/package.json"] = '{\n  "license": "MIT"\n}\n'
        assert run_against(t, m, {}) == 1, "a stray MIT in package.json must fail"

    # mutation 3: a stray MIT in pyproject.toml (table form)
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        m = dict(base_manifests)
        m["pyproject.toml"] = 'license = { text = "MIT" }\n'
        assert run_against(t, m, {}) == 1, "a stray MIT in pyproject.toml must fail"

    # mutation 4: the MIT licence grant text appearing in README prose (not a mere
    # historical mention of "MIT OR Apache-2.0", the actual licence text) -- built by
    # concatenation so this file's own tracked source never holds the sentence whole
    # (otherwise a real run of this same check against this repo would flag itself)
    grant_sentence = "Permission is hereby granted, free" + " of charge, to any person..."
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        other = {"README.md": f"See LICENSE.\n\n{grant_sentence}\n"}
        assert run_against(t, base_manifests, other) == 1, "the MIT grant text in README prose must fail"

    # historical prose mentioning the old dual licence, by name only, must NOT fail
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        other = {"README.md": "Releases up to 4.1.1 remain available under their published terms "
                               "(`MIT OR Apache-2.0`).\n"}
        assert run_against(t, base_manifests, other) == 0, "a historical MIT OR Apache-2.0 mention must pass"

    # an allow-listed path with its expected other value passes
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        m = dict(base_manifests)
        m["vrchat-package/package.json"] = '{\n  "license": "SEE LICENSE IN LICENSE.md"\n}\n'
        assert run_against(t, m, {}, allowed={"vrchat-package/package.json": "SEE LICENSE IN LICENSE.md"}) == 0, \
            "an allow-listed value must pass"
        # but an unexpected value at that SAME allow-listed path must still fail
        m["vrchat-package/package.json"] = '{\n  "license": "MIT"\n}\n'
        assert run_against(t, m, {}, allowed={"vrchat-package/package.json": "SEE LICENSE IN LICENSE.md"}) == 1, \
            "an unexpected value at an allow-listed path must still fail"

    # a file of an earlier licence anywhere must fail even with every declaration correct
    for name in OLD_LICENCE_FILES:
        for where in ("", "sub/"):
            with tempfile.TemporaryDirectory() as d:
                t = Path(d)
                other = {f"{where}{name}": "old licence text\n"}
                assert run_against(t, base_manifests, other) == 1, f"a {where}{name} file must fail"

    # 0 tracked files / 0 manifests read must each fail, not vacuously pass
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        subprocess.run(["git", "-C", str(t), "init", "-q"], check=True)
        global ROOT
        saved = ROOT
        try:
            ROOT = t
            assert main([]) == 2, "0 tracked files must fail"
        finally:
            ROOT = saved
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        assert run_against(t, {}, {"README.md": "nothing to declare here\n"}) == 2, \
            "0 licence declarations must fail"

    print("license_check self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main())
