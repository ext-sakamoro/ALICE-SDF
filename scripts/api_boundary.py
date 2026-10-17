#!/usr/bin/env python3
"""Enumerate the crate's boundary from its public API and compare it with a
reasoned manifest (ALICE-Law G7 / G8).

The crate is documented with rustdoc's JSON output (unstable, enabled with
RUSTC_BOOTSTRAP=1 on the pinned toolchain, the same way as
scripts/clippy_external_coverage.py of ALICE-DB / ALICE-Crypto), with
--document-hidden-items (a #[doc(hidden)] pub item is callable too) and the
features named in the manifest. The JSON format_version must be one this
script was written against (SUPPORTED_FORMATS) or it stops: a new toolchain is
a reason to re-check the walker, not to guess.

Enumerated, from the items of this crate:
  * fn: every function or method whose parameters, return type, generic
    parameters or where clause contain an element type of bytes, text or a
    reader: the primitives u8 / i8 / str / char, or a path ending in one of
    BYTE_PATHS (String, Read, BufRead, Path, File, ...), at any depth (inside
    references, slices, arrays, tuples, Box / Vec / Cow / Option / Result,
    impl Trait / dyn Trait bounds, associated-type constraints such as
    `Iterator<Item = u8>`, and through this crate's type aliases)
  * abi: every function with a non-Rust ABI (extern "C" and the like)
  * impl: every non-blanket trait implementation of a trait in SERDE_TRAITS
    (Serialize, Deserialize: derive or manual) or READ_TRAITS (Read, BufRead,
    FromStr), and of From / TryFrom when the trait's type argument contains a
    byte, text or reader element
  * reexport: every `pub use` of an item outside this crate (the item's own
    signature is not visible here, so the re-export itself needs a decision)

From a second rustdoc run with --document-private-items (an exported symbol can live
in a private module, which the public index omits):
  * abi: every function with a non-Rust ABI anywhere in the crate (this replaces the
    public pass's abi list, so `#[no_mangle] extern "C"` functions in private modules
    count)
  * binding: the entry points of the binding generators the manifest declares in
    `bindings` (their attributes are expanded before rustdoc runs, so the generated
    wrappers are what is visible): "pyo3" lists `__pyfunction_<f>` (key
    `binding::pyo3::<f>`) and `__pymethod_<m>__` on a type T (`binding::pyo3::T::<m>`),
    plus its module init. A declared generator that yields no entry point fails (its
    naming changed, or the feature is off): the list cannot go empty silently.
    Generated wrappers are not listed again as abi.

From one extra rustdoc run per entry of `targets` (a list of {target, features}, e.g.
wasm32-unknown-unknown with the wasm feature):
  * target: every public function that exists only on that target (its path is not in
    the native public run), key `target::<target>::<path>`. wasm-bindgen expands its
    exports into anonymous items rustdoc does not index, so the exported functions
    themselves, compiled only for the target, are what is listed. A target run that
    yields no such function fails.

Each must appear in the manifest (scripts/api_boundary.toml) as
`"<kind>::<key>" = "<decision>"`, the decision being `handshake: <test fn>`
(the entry point checks the producer's identifier; the test function must
exist in src/ or tests/), `local: <reason>` (no Law-bearing value from
another producer crosses it) or `pending: <reason>` (it must become a handshake
and cannot yet, e.g. the identifier it would check is not persisted yet). A
pending entry is allowed only while the crate's alice-law-conformance.toml does
not claim G7 pass (status unverified or fail-known): pending with G7 pass fails.
A second item with the same key gets `#2`, `#3`.

Host-only items: rustdoc documents the host it runs on, so an item compiled only for one OS
(e.g. a `#[cfg(target_os = "linux")] extern "C"` function a dependency's macro emits into this
crate) is enumerated only there. The manifest names such keys in `host = { "<key>" = "<os>" }`
(os: linux, macos or windows; each key also needs its decision in [boundary]). On that OS the
key is checked as any other (absent there = stale); on another OS it is not compared and a
`PREFLIGHT-SKIP: api_boundary host-only <key> (<os>)` line says so; a host key enumerated on
another OS, a host key without a decision and an unknown os fail.
Coverage limit: CI runs this check on Linux only, so an item that exists only on macOS or
Windows is never enumerated anywhere until a run on that OS is added.

The check fails on an enumerated item that is not listed, a listed item that
was not enumerated (stale), a missing handshake test, a reason shorter than
MIN_REASON, an unsupported format_version, a failed rustdoc, and when it
enumerated nothing (compared nothing).

The summary line gives the count per kind and per decision.

usage: scripts/api_boundary.py [--root DIR] [--json FILE] [--skip-missing-targets]
  --json FILE  read an existing rustdoc JSON instead of running rustdoc (tests; no
               private pass, no targets)
  --skip-missing-targets
               (preflight) a target whose rustup target is not installed is skipped,
               with a `PREFLIGHT-SKIP:` line naming it (run-preflight.sh records the line
               in its receipt); without the flag (CI) a missing target fails the run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import platform
import tomllib
from pathlib import Path

SUPPORTED_FORMATS = frozenset({60})
MIN_REASON = 12
BYTE_PRIMITIVES = frozenset({"u8", "i8", "str", "char"})
BYTE_PATHS = frozenset({"String", "Read", "BufRead", "Bytes", "BytesMut", "OsStr", "OsString", "Path",
                        "PathBuf", "File", "TcpStream", "UdpSocket", "Stdin"})
SERDE_TRAITS = frozenset({"Serialize", "Deserialize", "Deserializer", "Serializer"})
READ_TRAITS = frozenset({"Read", "BufRead", "FromStr"})
CONVERSION_TRAITS = frozenset({"From", "TryFrom"})
DECISIONS = ("handshake", "local", "pending")
HOSTS = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}  # platform.system() -> os name
BINDING_GENERATORS = {
    "pyo3": (re.compile(r"^__pyfunction_(\w+)$"), re.compile(r"^__pymethod_(\w+?)__$")),
}
PYO3_INIT = re.compile(r"^__pyo3_init$|^PyInit_\w+$")


def last(path: str) -> str:
    return path.split("::")[-1]


class Walker:
    def __init__(self, doc: dict):
        if doc.get("format_version") not in SUPPORTED_FORMATS:
            raise SystemExit(f"api_boundary: rustdoc JSON format_version {doc.get('format_version')} "
                             f"is not in {sorted(SUPPORTED_FORMATS)}; check the walker against the "
                             "new format, then add it")
        self.idx = doc["index"]
        self.seen_alias: set[str] = set()

    def item(self, i):
        return self.idx.get(str(i)) if i is not None else None

    def args_types(self, args):
        if not args:
            return
        ab = args.get("angle_bracketed")
        if ab:
            for a in ab.get("args", []):
                if isinstance(a, dict) and "type" in a:
                    yield a["type"]
            for c in ab.get("constraints", []):
                binding = c.get("binding", {})
                if "equality" in binding and isinstance(binding["equality"], dict) \
                        and "type" in binding["equality"]:
                    yield binding["equality"]["type"]
                for b in binding.get("constraint", []) or []:
                    yield {"__bound__": b}
        pa = args.get("parenthesized")
        if pa:
            yield from pa.get("inputs", [])
            if pa.get("output"):
                yield pa["output"]

    def bound_hits(self, bound) -> bool:
        tb = bound.get("trait_bound") if isinstance(bound, dict) else None
        if not tb:
            return False
        tr = tb["trait"]
        if last(tr["path"]) in BYTE_PATHS:
            return True
        return any(self.hits(t) for t in self.args_types(tr.get("args")))

    def hits(self, ty) -> bool:
        """Whether a rustdoc type contains a byte, text or reader element."""
        if ty is None or not isinstance(ty, dict):
            return False
        if "__bound__" in ty:
            return self.bound_hits(ty["__bound__"])
        k, v = next(iter(ty.items()))
        if k == "primitive":
            return v in BYTE_PRIMITIVES
        if k in ("slice",):
            return self.hits(v)
        if k == "array":
            return self.hits(v["type"])
        if k in ("borrowed_ref", "raw_pointer"):
            return self.hits(v["type"])
        if k == "tuple":
            return any(self.hits(t) for t in v)
        if k == "resolved_path":
            if last(v["path"]) in BYTE_PATHS:
                return True
            alias = self.item(v.get("id"))
            if alias and "type_alias" in alias.get("inner", {}) and str(v.get("id")) not in self.seen_alias:
                self.seen_alias.add(str(v.get("id")))
                try:
                    if self.hits(alias["inner"]["type_alias"]["type"]):
                        return True
                finally:
                    self.seen_alias.discard(str(v.get("id")))
            return any(self.hits(t) for t in self.args_types(v.get("args")))
        if k in ("impl_trait", "dyn_trait"):
            bounds = v if k == "impl_trait" else [{"trait_bound": {"trait": t["trait"]}} for t in v["traits"]]
            return any(self.bound_hits(b) for b in bounds)
        if k == "qualified_path":
            return self.hits(v.get("self_type")) or (
                v.get("trait") is not None and last(v["trait"]["path"]) in BYTE_PATHS)
        if k == "function_pointer":
            sig = v["sig"]
            return any(self.hits(t) for _, t in sig["inputs"]) or self.hits(sig.get("output"))
        return False

    def generics_hit(self, g) -> bool:
        for p in g.get("params", []):
            kind = p.get("kind", {})
            t = kind.get("type")
            if t:
                if any(self.bound_hits(b) for b in t.get("bounds", [])) or self.hits(t.get("default")):
                    return True
            c = kind.get("const")
            if c and self.hits(c.get("type")):
                return True
        for w in g.get("where_predicates", []):
            bp = w.get("bound_predicate")
            if bp and (self.hits(bp.get("type")) or any(self.bound_hits(b) for b in bp.get("bounds", []))):
                return True
        return False


def type_name(ty) -> str:
    if isinstance(ty, dict):
        k, v = next(iter(ty.items()))
        if k == "resolved_path":
            return last(v["path"])
        if k == "primitive":
            return v
        if k == "borrowed_ref":
            return type_name(v["type"])
        if k == "generic":
            return v
    return "?"


def _owners(w: Walker) -> dict[str, str]:
    owner: dict[str, str] = {}
    for it in w.idx.values():
        if it.get("crate_id") != 0:
            continue
        imp = it.get("inner", {}).get("impl")
        if not imp or imp.get("blanket_impl") is not None or imp.get("is_synthetic"):
            continue
        tr = imp.get("trait")
        for m in imp.get("items", []):
            owner[str(m)] = (f"<{last(tr['path'])} for {type_name(imp['for'])}>" if tr
                             else type_name(imp["for"]))
    return owner


def _fn_name(i, it, owner: dict[str, str], paths: dict) -> str:
    if str(i) in owner:
        return f"{owner[str(i)]}::{it.get('name')}"
    if str(i) in paths:
        return "::".join(paths[str(i)]["path"][1:])
    return it.get("name") or "?"


def _binding_key(gen: str, i, it, owner: dict[str, str]) -> str | None:
    name = it.get("name") or ""
    fpat, mpat = BINDING_GENERATORS[gen]
    m = fpat.match(name)
    if m:
        return f"binding::{gen}::{m.group(1)}"
    if mpat is not None and (m := mpat.match(name)):
        return f"binding::{gen}::{owner.get(str(i), '?')}::{m.group(1)}"
    return None


def enumerate_boundary(doc: dict, private: dict | None = None, bindings: tuple[str, ...] = (),
                       targets: tuple[dict, ...] = ()) -> list[str]:
    """The boundary keys: fn / impl / reexport from the public `doc`; abi from `private`
    (the --document-private-items run) when given, else from `doc`; binding from
    `private` for the declared `bindings`; target from each `targets` entry
    ({"name": <target>, "doc": <its rustdoc JSON>})"""
    w = Walker(doc)
    keys: list[str] = []
    owner = _owners(w)
    for i, it in w.idx.items():
        if it.get("crate_id") != 0:
            continue
        inner = it.get("inner", {})
        if "impl" in inner:
            imp = inner["impl"]
            if imp.get("blanket_impl") is not None or imp.get("is_synthetic"):
                continue
            tr = imp.get("trait")
            if tr:
                name = last(tr["path"])
                conv = name in CONVERSION_TRAITS and any(w.hits(t) for t in w.args_types(tr.get("args")))
                if name in SERDE_TRAITS or name in READ_TRAITS or conv:
                    keys.append(f"impl::{name} for {type_name(imp['for'])}")
    paths = doc.get("paths", {})
    for i, it in w.idx.items():
        if it.get("crate_id") != 0:
            continue
        inner = it.get("inner", {})
        if "function" in inner:
            f = inner["function"]
            name = _fn_name(i, it, owner, paths)
            abi = f["header"].get("abi")
            if abi != "Rust":
                if private is None:
                    keys.append(f"abi::{name}")
                continue
            sig = f["sig"]
            if any(w.hits(t) for _, t in sig["inputs"]) or w.hits(sig.get("output")) \
                    or w.generics_hit(f.get("generics", {})):
                keys.append(f"fn::{name}")
        elif "use" in inner and it.get("visibility") == "public":
            u = inner["use"]
            target = w.item(u.get("id"))
            if target is None or target.get("crate_id") != 0:
                keys.append(f"reexport::{u.get('source')}")
    if private is not None:
        keys += _private_keys(private, bindings)
    native = {tuple(v["path"]) for v in paths.values()}
    for tgt in targets:
        keys += _target_keys(tgt["name"], tgt["doc"], native)
    # number repeated keys in a stable order
    out: list[str] = []
    count: dict[str, int] = {}
    for k in sorted(keys):
        count[k] = count.get(k, 0) + 1
        out.append(k if count[k] == 1 else f"{k}#{count[k]}")
    return out


def _private_keys(doc: dict, bindings: tuple[str, ...]) -> list[str]:
    """abi (every non-Rust-ABI fn, any visibility) and the pyo3 bindings of the private run"""
    w = Walker(doc)
    owner = _owners(w)
    paths = doc.get("paths", {})
    keys: list[str] = []
    for i, it in w.idx.items():
        if it.get("crate_id") != 0 or "function" not in it.get("inner", {}):
            continue
        if "pyo3" in bindings:
            k = _binding_key("pyo3", i, it, owner)
            if k is not None:
                keys.append(k)
                continue
            if PYO3_INIT.match(it.get("name") or ""):
                keys.append(f"binding::pyo3::{it.get('name')}")
                continue
        if it["inner"]["function"]["header"].get("abi") != "Rust":
            keys.append(f"abi::{_fn_name(i, it, owner, paths)}")
    return keys


def _target_keys(target: str, doc: dict, native: set[tuple[str, ...]]) -> list[str]:
    """the public functions of a target's run that the native public run does not have"""
    keys: list[str] = []
    for i, it in doc.get("index", {}).items():
        if it.get("crate_id") != 0 or "function" not in it.get("inner", {}):
            continue
        if it.get("visibility") != "public":
            continue
        p = doc.get("paths", {}).get(str(i))
        if p and tuple(p["path"]) not in native:
            keys.append(f"target::{target}::{'::'.join(p['path'][1:])}")
    return keys


def rustdoc(root: Path, features: str, private: bool = False, target: str | None = None) -> dict:
    env = dict(os.environ)
    env["RUSTC_BOOTSTRAP"] = "1"
    env["RUSTDOCFLAGS"] = ""
    cmd = ["cargo", "rustdoc", "-q", "--lib"]
    if target:
        cmd += ["--target", target]
    if features:
        cmd += ["--features", features]
    cmd += ["--", "-Z", "unstable-options", "--output-format", "json", "--document-hidden-items"]
    if private:
        cmd += ["--document-private-items"]
    proc = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:])
        raise SystemExit("api_boundary: rustdoc failed")
    meta = json.loads(subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=root,
                                     stdout=subprocess.PIPE, text=True, check=True).stdout)
    target_dir = Path(meta["target_directory"])
    lib_kinds = {"lib", "rlib", "dylib", "cdylib", "staticlib"}
    lib = next(t["name"] for p in meta["packages"] for t in p["targets"] if lib_kinds & set(t["kind"]))
    doc_dir = target_dir / target / "doc" if target else target_dir / "doc"
    return json.loads((doc_dir / f"{lib.replace('-', '_')}.json").read_text(encoding="utf-8"))


def test_names(root: Path) -> set[str]:
    names: set[str] = set()
    for d in ("src", "tests"):
        if (root / d).exists():
            for path in (root / d).rglob("*.rs"):
                names.update(re.findall(r"\bfn\s+(\w+)", path.read_text(encoding="utf-8")))
    return names


def check(root: Path, doc: dict, listed: dict[str, str], private: dict | None = None,
          bindings: tuple[str, ...] = (), targets: tuple[dict, ...] = (),
          g7_status: str | None = None, host_only: dict[str, str] | None = None,
          this_host: str | None = None) -> tuple[list[str], list[str]]:
    found = enumerate_boundary(doc, private, bindings, targets)
    if not found:
        return ["enumerated no boundary item (compared nothing)"], found
    problems = []
    host_only = host_only or {}
    for key, os_name in sorted(host_only.items()):
        if os_name not in HOSTS.values():
            problems.append(f"host: {key}: unknown os {os_name!r} (known: {sorted(HOSTS.values())})")
        if key not in listed:
            problems.append(f"host: {key} has no decision in [boundary]")
        if os_name != this_host and key in found:
            problems.append(f"host: {key} is declared {os_name}-only but was enumerated on {this_host}")
    # a host key of another os is not compared here (main prints the PREFLIGHT-SKIP line)
    listed = {k: v for k, v in listed.items() if host_only.get(k, this_host) == this_host}
    for tgt in targets:
        if not any(k.startswith(f"target::{tgt['name']}::") for k in found):
            problems.append(f"targets: the {tgt['name']} run lists no function of its own "
                            "(the feature is off or the target's module is gone)")
    for gen in bindings:
        if gen not in BINDING_GENERATORS:
            problems.append(f"bindings: unknown generator {gen!r} (known: {sorted(BINDING_GENERATORS)})")
        elif not any(k.startswith(f"binding::{gen}::") for k in found):
            problems.append(f"bindings: {gen} is declared but no entry point was enumerated "
                            "(the generator's naming changed, the feature or target is off, or the "
                            "private run is missing)")
    tests = test_names(root)
    for key in sorted(set(found) - listed.keys()):
        problems.append(f"{key}: not in scripts/api_boundary.toml (decide handshake, local or pending)")
    for key in sorted(listed.keys() - set(found)):
        problems.append(f"{key}: listed but not enumerated (stale entry)")
    pending = 0
    for key, decision in sorted(listed.items()):
        kind, _, rest = decision.partition(":")
        rest = rest.strip()
        if kind == "handshake":
            if rest not in tests:
                problems.append(f"{key}: handshake test `fn {rest}` not found in src/ or tests/")
        elif kind in ("local", "pending"):
            if len(rest) < MIN_REASON:
                problems.append(f"{key}: reason shorter than {MIN_REASON} characters")
            pending += kind == "pending"
        else:
            problems.append(f"{key}: decision must start with `handshake:`, `local:` or `pending:`")
    if pending and g7_status not in ("unverified", "fail-known"):
        problems.append(f"{pending} pending entries while alice-law-conformance.toml G7 is "
                        f"{g7_status!r}: pending is allowed only with G7 unverified or fail-known")
    return problems, found


def g7_status(root: Path) -> str | None:
    path = root / "alice-law-conformance.toml"
    if not path.is_file():
        return None
    return tomllib.loads(path.read_text(encoding="utf-8")).get("G7", {}).get("status")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--json")
    ap.add_argument("--skip-missing-targets", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    cfg = tomllib.loads((root / "scripts" / "api_boundary.toml").read_text(encoding="utf-8"))
    features = cfg.get("features", "")
    bindings = tuple(cfg.get("bindings", []))
    if a.json:
        doc, private, targets = json.loads(Path(a.json).read_text(encoding="utf-8")), None, ()
    else:
        doc = rustdoc(root, features)
        private = rustdoc(root, features, private=True)
        wanted = cfg.get("targets", [])
        if a.skip_missing_targets and wanted:
            installed = set(subprocess.run(["rustup", "target", "list", "--installed"], stdout=subprocess.PIPE,
                                           text=True, check=False).stdout.split())
            for t in [t for t in wanted if t["target"] not in installed]:
                print(f"PREFLIGHT-SKIP: api_boundary target {t['target']}: the rustup target is not installed "
                      "(its `target::` entries are not checked here; CI runs them)")
            wanted = [t for t in wanted if t["target"] in installed]
        targets = tuple({"name": t["target"], "doc": rustdoc(root, t.get("features", ""), target=t["target"])}
                        for t in wanted)
    listed = cfg.get("boundary", {})
    skipped = {t["target"] for t in cfg.get("targets", [])} - {t["name"] for t in targets} if not a.json else set()
    if skipped:  # the entries of a skipped target are neither checked nor stale
        listed = {k: v for k, v in listed.items() if not any(k.startswith(f"target::{n}::") for n in skipped)}
    host_only = cfg.get("host", {})
    this_host = HOSTS.get(platform.system(), platform.system().lower())
    for key, os_name in sorted(host_only.items()):
        if os_name != this_host:
            print(f"PREFLIGHT-SKIP: api_boundary host-only {key} ({os_name}): not compiled on {this_host}, "
                  f"checked on {os_name}")
    problems, found = check(root, doc, listed, private, bindings, targets, g7_status(root), host_only, this_host)
    for p in problems:
        print(f"api_boundary: {p}", file=sys.stderr)
    if problems:
        return 1
    kinds: dict[str, int] = {}
    for k in found:
        kinds[k.split("::", 1)[0]] = kinds.get(k.split("::", 1)[0], 0) + 1
    decisions = {d: sum(1 for k in found if listed[k].startswith(d + ":")) for d in DECISIONS}
    print(f"api_boundary: {len(found)} items ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}): "
          + ", ".join(f"{v} {d}" for d, v in decisions.items()) + ", 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
