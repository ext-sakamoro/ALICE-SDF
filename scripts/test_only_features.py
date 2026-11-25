#!/usr/bin/env python3
"""The crate's test-only features (scripts/test_only_features.txt) are test-only: no production build turns them on.

scripts/scip_index.sh builds the SCIP index without these features, so the code behind them is not counted as
production code that nothing reaches. That is only sound when no production build can enable them, which this
check holds: a listed feature is refused when

- it is not a feature of the crate;
- the crate's own `[dev-dependencies]` entry on itself does not enable it (a test-only feature is the one this
  crate's tests turn on; a production feature named here by mistake is not in that entry);
- it is in the crate's `default` (directly or through another default feature);
- another feature of the crate enables it (`"test-hooks"`, `"alice-sdf/test-hooks"` or `"alice-sdf?/test-hooks"`);
- another manifest of this repository that depends on the crate enables it by default: in the dependency's
  `features`, or through a feature reachable from that manifest's own `default`. A feature the other manifest
  names only behind one of its own non-default features (an opt-in, such as the CI-only consumer's `guard`) is
  allowed and printed.

The crate's own `[dev-dependencies]` entry on itself is how its tests turn the features on, and is not a
dependent. An empty list, or a list read from no manifest, exits 2 (nothing compared is a failure).

usage: scripts/test_only_features.py [ROOT]          check
       scripts/test_only_features.py --names [ROOT]  print the listed names, one per line (scip_index.sh)
"""
import os
import subprocess
import sys
import tomllib

LIST = "scripts/test_only_features.txt"


def listed(root):
    names = []
    with open(os.path.join(root, LIST), encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(s)
    return names


def _enables(entry, crate, name):
    """whether a feature entry of a [features] list turns on `name` of `crate` (or of this package)"""
    return entry == name or entry in (f"{crate}/{name}", f"{crate}?/{name}")


def _closure(features, start):
    """the features reachable from `start` in one manifest's [features] table"""
    seen, todo = set(), list(start)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        todo.extend(e for e in features.get(f, []) if "/" not in e and not e.startswith("dep:"))
    return seen


def _manifests(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "*Cargo.toml", "Cargo.toml"], capture_output=True,
                         text=True, check=True).stdout.split()
    return sorted(set(out))


def _deps_on(manifest, crate):
    """the dependency tables' entries for `crate` (by package name or key) in a manifest"""
    found = []
    for table in ("dependencies", "dev-dependencies", "build-dependencies"):
        for key, spec in manifest.get(table, {}).items():
            pkg = spec.get("package", key) if isinstance(spec, dict) else key
            if pkg == crate:
                found.append((table, key, spec))
    for target in manifest.get("target", {}).values():
        for table in ("dependencies", "dev-dependencies", "build-dependencies"):
            for key, spec in target.get(table, {}).items():
                pkg = spec.get("package", key) if isinstance(spec, dict) else key
                if pkg == crate:
                    found.append((table, key, spec))
    return found


def check(root):
    """(problems, notes, compared): compared counts the (name, manifest) pairs read"""
    names = listed(root)
    problems, notes, compared = [], [], 0
    if not names:
        return [f"{LIST}: no feature listed (nothing to compare)"], notes, 0
    with open(os.path.join(root, "Cargo.toml"), "rb") as f:
        own = tomllib.load(f)
    crate = own["package"]["name"]
    features = own.get("features", {})
    default = _closure(features, features.get("default", []))
    self_dev = [spec for table, _, spec in _deps_on(own, crate) if table == "dev-dependencies"]
    self_features = {f for spec in self_dev if isinstance(spec, dict) for f in spec.get("features", [])}
    for name in names:
        if name not in features:
            problems.append(f"{name}: not a feature of {crate}")
            continue
        if name not in self_features:
            problems.append(f"{name}: {crate}'s [dev-dependencies] entry on itself does not enable it")
        if name in default:
            problems.append(f"{name}: in {crate}'s default features")
        for other, entries in features.items():
            if other != name and any(_enables(e, crate, name) for e in entries):
                problems.append(f"{name}: enabled by {crate}'s feature `{other}`")
    for rel in _manifests(root):
        if rel == "Cargo.toml":
            continue
        with open(os.path.join(root, rel), "rb") as f:
            m = tomllib.load(f)
        deps = _deps_on(m, crate)
        if not deps:
            continue
        mf = m.get("features", {})
        on_by_default = _closure(mf, mf.get("default", []))
        for name in names:
            compared += 1
            for table, key, spec in deps:
                if isinstance(spec, dict) and name in spec.get("features", []):
                    problems.append(f"{name}: {rel} enables it in its {table} entry `{key}`")
            for f_name, entries in mf.items():
                if any(_enables(e, crate, name) or any(_enables(e, k, name) for _, k, _ in deps) for e in entries):
                    if f_name in on_by_default:
                        problems.append(f"{name}: {rel} enables it through its default feature `{f_name}`")
                    else:
                        notes.append(f"{name}: {rel} enables it only behind its non-default feature `{f_name}`")
    compared += len(names)
    return problems, notes, compared


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    root = args[0] if args else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "--names" in argv:
        names = listed(root)
        if not names:
            print(f"{LIST}: no feature listed", file=sys.stderr)
            return 2
        print("\n".join(names))
        return 0
    problems, notes, compared = check(root)
    for n in notes:
        print(f"test-only-features: note: {n}")
    for p in problems:
        print(f"test-only-features: {p}")
    if compared == 0 or (problems and problems[0].endswith("(nothing to compare)")):
        print("test-only-features: compared nothing", file=sys.stderr)
        return 2
    print(f"test-only-features: {len(listed(root))} listed, {compared} compared, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
