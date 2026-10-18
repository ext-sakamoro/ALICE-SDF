#!/usr/bin/env python3
"""The clippy configuration for the library code, and a check that it fires.

`clippy.toml` (repository root) forbids the platform libm for every target.
The library code additionally may not call `f32::max` / `f32::min` /
`f64::max` / `f64::min` / `wide::f32x8::max` / `wide::f32x8::min`: those
leave the sign of an equal pair of zeros
unspecified and the targets differ (aarch64 returns +0 from max and -0 from
min, x86_64 returns the argument), so a stored value could carry a different
bit on Linux x86 than on macOS ARM. The library uses the Law helpers
(`law_max_f32` / `law_min_f32`, and `Real::max` / `min` for `f32x8`), or, for a
value that never reaches a distance, bound, mesh or persisted byte, the
`raw_*` helpers in `src/raw_minmax.rs` (`scripts/raw_minmax_manifest.py` says
which files may call them; `scripts/clippy_lib_hidden_check.py` shows that no
other allow of the lint hides a max / min). Tests, examples and benches keep the std functions, so the extra
entries live in a second configuration, `.github/clippy-lib/clippy.toml`,
that CI and preflight pass to `cargo clippy --lib` with `CLIPPY_CONF_DIR`.

That file is generated from the root one (the root list plus EXTRA, inserted
at the top of `disallowed-methods`), so the libm list cannot drift between the
two. This script

  * check   fails when `.github/clippy-lib/clippy.toml` differs from what the
            root file generates (`--write` regenerates it)
  * canary  runs clippy with that configuration on `scripts/clippy-lib-canary`,
            a crate that uses every forbidden form once (method and path calls,
            a `fold` with `f64::max`, and one libm call), and fails unless
            clippy exits non-zero with a `disallowed_methods` diagnostic on
            each of those lines: a configuration that clippy did not read, or
            an entry it does not match, would otherwise pass every build

  python3 scripts/clippy_lib_gate.py            check + canary
  python3 scripts/clippy_lib_gate.py --check    check only
  python3 scripts/clippy_lib_gate.py --write    regenerate the file
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROOT_CONFIG = os.path.join(ROOT, "clippy.toml")
LIB_DIR = os.path.join(ROOT, ".github", "clippy-lib")
LIB_CONFIG = os.path.join(LIB_DIR, "clippy.toml")
CANARY = os.path.join(HERE, "clippy-lib-canary")

REASON = ("the sign of an equal pair of zeros is unspecified and differs between "
          "targets; use law_max_f32 / law_min_f32, or .law_max / .law_min / .law_clamp / "
          ".law_max_element / .law_min_element for glam vectors (or raw_* for values that are never stored)")
EXTRA = ["f32::max", "f32::min", "f64::max", "f64::min", "wide::f32x8::max", "wide::f32x8::min",
         # glam vectors: each lane goes through f32::max / f32::min (use LawMinMax / LawVec)
         "glam::Vec2::min",
         "glam::Vec2::max",
         "glam::Vec2::clamp",
         "glam::Vec2::min_element",
         "glam::Vec2::max_element",
         "glam::Vec3::min",
         "glam::Vec3::max",
         "glam::Vec3::clamp",
         "glam::Vec3::min_element",
         "glam::Vec3::max_element",
         "glam::Vec3A::min",
         "glam::Vec3A::max",
         "glam::Vec3A::clamp",
         "glam::Vec3A::min_element",
         "glam::Vec3A::max_element",
         "glam::Vec4::min",
         "glam::Vec4::max",
         "glam::Vec4::clamp",
         "glam::Vec4::min_element",
         "glam::Vec4::max_element",
         "glam::DVec2::min",
         "glam::DVec2::max",
         "glam::DVec2::clamp",
         "glam::DVec2::min_element",
         "glam::DVec2::max_element",
         "glam::DVec3::min",
         "glam::DVec3::max",
         "glam::DVec3::clamp",
         "glam::DVec3::min_element",
         "glam::DVec3::max_element",
         "glam::DVec4::min",
         "glam::DVec4::max",
         "glam::DVec4::clamp",
         "glam::DVec4::min_element",
         "glam::DVec4::max_element",
         ]

# ALICE-Law G13: the library may not iterate a collection whose order is unspecified
# (tests, examples and benches may). The types, their iterators, and the methods that
# iterate (method calls resolve through aliases and inference, which a type ban does not
# see); a membership-only use carries an allow with its reason and is pinned by
# scripts/order_guard.py
ORDER_REASON = ("iteration order is unspecified and differs between runs (ALICE-Law G13); use BTreeMap / "
                "BTreeSet, or allow a membership-only use with its reason and list it in scripts/order_allowlist.toml")
EXTRA_TYPES = ["std::collections::HashMap", "std::collections::HashSet",
               "std::collections::hash_map::Iter", "std::collections::hash_map::IterMut",
               "std::collections::hash_map::Keys", "std::collections::hash_map::Values",
               "std::collections::hash_map::ValuesMut", "std::collections::hash_map::IntoIter",
               "std::collections::hash_map::Drain", "std::collections::hash_set::Iter",
               "std::collections::hash_set::IntoIter", "std::collections::hash_set::Drain"]
EXTRA_ORDER_METHODS = ["std::collections::HashMap::iter", "std::collections::HashMap::iter_mut",
                       "std::collections::HashMap::keys", "std::collections::HashMap::values",
                       "std::collections::HashMap::values_mut", "std::collections::HashMap::into_keys",
                       "std::collections::HashMap::into_values", "std::collections::HashMap::drain",
                       "std::collections::HashMap::retain", "std::collections::HashSet::iter",
                       "std::collections::HashSet::drain", "std::collections::HashSet::retain"]

# ALICE-Law G8: the SDF tree types keep `Deserialize`, so a bare serde_json / bincode read of
# a tree would skip the stored semantics identifier. The library may call these only at a
# reviewed site that carries `#[allow(clippy::disallowed_methods, reason = "stored form
# <checked|other-type|pending>: ...")]` (scripts/bare_serde_reads.py lists them). clippy
# matches the resolved path, so an alias (`use serde_json as sj`), a brace import, a
# re-export (`serde_json::from_str` is `serde_json::de::from_str`) and
# `serde_json::value::from_value` are all found
STORED_FORM_REASON = ("a bare read of a stored form skips the semantics identifier (ALICE-Law G8); read "
                      "through a checked loader, or mark a reviewed site with #[allow(clippy::disallowed_methods)] "
                      "with a reason starting stored form checked: / other-type: / pending: (scripts/bare_serde_reads.py)")
EXTRA_STORED_FORM_METHODS = ["serde_json::from_str", "serde_json::from_slice", "serde_json::from_reader",
                             "serde_json::from_value", "serde_json::de::from_str", "serde_json::de::from_slice",
                             "serde_json::de::from_reader", "serde_json::value::from_value",
                             "serde_json::Deserializer::from_str", "serde_json::Deserializer::from_slice",
                             "serde_json::Deserializer::from_reader", "serde_json::Deserializer::new",
                             "serde_json::StreamDeserializer::new",
                             # the trait method: `T::deserialize(value)` with a serde_json::Value (itself a
                             # Deserializer) or a Deserializer built any other way; the derive expansions
                             # come from an external proc macro, which the lint does not see
                             "serde::de::Deserialize::deserialize", "serde::de::DeserializeSeed::deserialize",
                             "bincode::serde::decode_from_slice", "bincode::serde::decode_from_reader",
                             "bincode::serde::decode_from_std_read", "bincode::serde::borrow_decode_from_slice",
                             "bincode::decode_from_slice", "bincode::decode_from_reader",
                             "bincode::decode_from_std_read", "bincode::borrow_decode_from_slice"]

HEADER = """\
# Generated by scripts/clippy_lib_gate.py from the repository's clippy.toml:
# do not edit (python3 scripts/clippy_lib_gate.py --write)
#
# The library code (`cargo clippy --lib`, CLIPPY_CONF_DIR=.github/clippy-lib)
# also may not call f32 / f64 / f32x8 max and min; tests, examples and benches use the
# root configuration; nor name or iterate HashMap / HashSet (ALICE-Law G13); nor read a
# stored form with serde_json / bincode outside a reviewed site (ALICE-Law G8)
"""

# the lines of the canary that must each get a diagnostic, by the text on them
CANARY_FORMS = ["a.max(b)", "f32::min(a, b)", "a.min(b)", "f64::max)", "x.sin()", "v.max(w)", "v.min(w)",
                "Instant::now()", "SystemTime::now()", "i.elapsed()", "s.elapsed()", "RandomState::new()",
                "v2.min(o)", "v2.max(o)", "v2.clamp(lo, hi)", "v2.min_element()", "v2.max_element()", "v3.min(o)", "v3.max(o)", "v3.clamp(lo, hi)", "v3.min_element()", "v3.max_element()", "v3a.min(o)", "v3a.max(o)", "v3a.clamp(lo, hi)", "v3a.min_element()", "v3a.max_element()", "v4.min(o)", "v4.max(o)", "v4.clamp(lo, hi)", "v4.min_element()", "v4.max_element()", "d2.min(o)", "d2.max(o)", "d2.clamp(lo, hi)", "d2.min_element()", "d2.max_element()", "d3.min(o)", "d3.max(o)", "d3.clamp(lo, hi)", "d3.min_element()", "d3.max_element()", "d4.min(o)", "d4.max(o)", "d4.clamp(lo, hi)", "d4.min_element()", "d4.max_element()",
                "&std::collections::HashMap<u32, u32>)", "&std::collections::HashSet<u32>)", "hash_map::Iter<'_, u32, u32>)",
                "m.keys()", "m.values()", "m.iter_mut()", "s.iter()",
                "serde_json::from_str(js)", "sj::from_str(js)", "read_text::<Value>(js)",
                "serde_json::value::from_value(jv)", "serde_json::from_value(jv.clone())",
                "serde_json::from_slice(jb)", "serde_json::from_reader(jr)",
                "serde_json::Deserializer::from_str(js)", "bincode::serde::decode_from_slice(bb,",
                "bs::decode_from_slice(bb,", "bincode::serde::decode_from_reader(rd,",
                "bincode::serde::decode_from_std_read(br,", "bincode::decode_from_slice(bn,",
                "u32::deserialize(tv)", "serde_json::Deserializer::new(serde_json::de::StrRead::new(ts))",
                "<u32 as Deserialize>::deserialize(&mut td)", "u32::deserialize(&pv)",
                "std::marker::PhantomData::<u32>.deserialize(sv)"]

DIAG = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):\d+: (?:warning|error): use of a disallowed (?:method|type) `(?P<path>[^`]+)`",
                  re.M)

UNRESOLVED = re.compile(r"`(?P<path>[^`]+)` does not refer to (?:a|an) (?:reachable|existing)")


def render(root_text: str) -> str:
    """The library configuration generated from the root one."""
    marker = "disallowed-methods = [\n"
    if root_text.count(marker) != 1:
        raise ValueError("clippy.toml: expected exactly one `disallowed-methods = [` line")
    for path in EXTRA:
        if f'path = "{path}"' in root_text:
            raise ValueError(f"clippy.toml already lists {path}: it belongs in EXTRA only")
    entries = "".join(f'    {{ path = "{p}", reason = "{REASON}" }},\n' for p in EXTRA)
    entries += "".join(f'    {{ path = "{p}", reason = "{ORDER_REASON}" }},\n' for p in EXTRA_ORDER_METHODS)
    entries += "".join(f'    {{ path = "{p}", reason = "{STORED_FORM_REASON}" }},\n' for p in EXTRA_STORED_FORM_METHODS)
    i = root_text.index(marker) + len(marker)
    out = root_text[:i] + entries + root_text[i:]
    tmarker = "disallowed-types = [\n"
    if out.count(tmarker) != 1:
        raise ValueError("clippy.toml: expected exactly one `disallowed-types = [` line")
    for path in EXTRA_TYPES + EXTRA_ORDER_METHODS + EXTRA_STORED_FORM_METHODS:
        if f'path = "{path}"' in root_text:
            raise ValueError(f"clippy.toml already lists {path}: it belongs in the library list only")
    j = out.index(tmarker) + len(tmarker)
    types = "".join(f'    {{ path = "{p}", reason = "{ORDER_REASON}" }},\n' for p in EXTRA_TYPES)
    return HEADER + "\n" + out[:j] + types + out[j:]


def count_entries(text: str, key: str = "disallowed-methods") -> int:
    i = text.index(f"{key} = [")
    j = text.index("\n]", i)
    return len(re.findall(r'path\s*=\s*"', text[i:j]))


def check(root_text: str, lib_text: str | None) -> list[str]:
    want = render(root_text)
    if lib_text is None:
        return [f"{os.path.relpath(LIB_CONFIG, ROOT)} is missing (python3 scripts/clippy_lib_gate.py --write)"]
    if lib_text != want:
        return [f"{os.path.relpath(LIB_CONFIG, ROOT)} differs from what clippy.toml generates "
                "(python3 scripts/clippy_lib_gate.py --write)"]
    n_root, n_lib = count_entries(root_text), count_entries(lib_text)
    extra = len(EXTRA) + len(EXTRA_ORDER_METHODS) + len(EXTRA_STORED_FORM_METHODS)
    if n_root == 0 or n_lib != n_root + extra:
        return [f"disallowed-methods: root {n_root}, library {n_lib} (expected root + {extra}, root > 0)"]
    t_root, t_lib = count_entries(root_text, "disallowed-types"), count_entries(lib_text, "disallowed-types")
    if t_lib != t_root + len(EXTRA_TYPES):
        return [f"disallowed-types: root {t_root}, library {t_lib} (expected root + {len(EXTRA_TYPES)})"]
    return []


def canary_lines(source: str) -> dict[int, str]:
    """Line number of each form in the canary source (each must occur once)."""
    out = {}
    lines = source.split("\n")
    for form in CANARY_FORMS:
        hits = [n for n, l in enumerate(lines, 1) if form in l and not l.lstrip().startswith("//")]
        if len(hits) != 1:
            raise ValueError(f"canary: `{form}` occurs {len(hits)} times, expected once")
        out[hits[0]] = form
    return out


def judge_canary(rc: int, output: str, want: dict[int, str]) -> tuple[list[str], int]:
    """Errors, and the number of disallowed-method diagnostics on the canary."""
    hit = {int(m["line"]) for m in DIAG.finditer(output) if m["file"].endswith("lib.rs")}
    errors = []
    if rc == 0:
        errors.append("clippy exited 0 on the canary: the configuration did not forbid anything")
    for line, form in sorted(want.items()):
        if line not in hit:
            errors.append(f"canary line {line} (`{form}`) has no disallowed-method diagnostic")
    if not hit:
        errors.append("no disallowed-method diagnostic at all (compared nothing)")
    # clippy warns, without failing, on an entry that names nothing (a renamed or
    # removed function): that entry would forbid nothing
    for m in UNRESOLVED.finditer(output):
        errors.append(f"configuration entry `{m['path']}` does not refer to a function clippy can find")
    return errors, len(hit)


def run_canary() -> list[str]:
    with open(os.path.join(CANARY, "src", "lib.rs"), encoding="utf-8") as f:
        want = canary_lines(f.read())
    env = dict(os.environ, CLIPPY_CONF_DIR=LIB_DIR, CARGO_TERM_COLOR="never")
    target = os.environ.get("CARGO_TARGET_DIR") or os.path.join(tempfile.gettempdir(), "clippy-lib-canary-target")
    cmd = ["cargo", "clippy", "--quiet", "--manifest-path", os.path.join(CANARY, "Cargo.toml"),
           "--target-dir", os.path.join(target, "clippy-lib-canary"), "--message-format=short",
           "--", "-D", "clippy::disallowed_methods", "-D", "clippy::disallowed_types"]
    # touch the source so clippy re-lints it (a cached clean check prints nothing)
    os.utime(os.path.join(CANARY, "src", "lib.rs"))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    errors, n = judge_canary(r.returncode, r.stdout + r.stderr, want)
    if errors:
        errors.append("clippy output:\n" + (r.stdout + r.stderr)[-3000:])
    else:
        print(f"clippy-lib canary: {n} of {len(want)} forbidden forms flagged (clippy exit {r.returncode})")
    return errors


def main(argv: list[str]) -> int:
    with open(ROOT_CONFIG, encoding="utf-8") as f:
        root_text = f.read()
    if "--write" in argv:
        os.makedirs(LIB_DIR, exist_ok=True)
        with open(LIB_CONFIG, "w", encoding="utf-8") as f:
            f.write(render(root_text))
        print(f"wrote {os.path.relpath(LIB_CONFIG, ROOT)}")
        return 0
    lib_text = None
    if os.path.exists(LIB_CONFIG):
        with open(LIB_CONFIG, encoding="utf-8") as f:
            lib_text = f.read()
    errors = check(root_text, lib_text)
    if not errors:
        print(f"clippy-lib config: root {count_entries(root_text)} disallowed methods + {len(EXTRA)} "
              f"({', '.join(EXTRA)})")
    if not errors and "--check" not in argv:
        errors = run_canary()
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
