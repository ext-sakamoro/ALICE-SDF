#!/usr/bin/env bash
# Unreal plugin ABI / metadata gate that needs no Unreal (runs on the
# GitHub-hosted Linux lane and in scripts/preflight.sh). The real engine
# gate is scripts/unreal-ue5-ci.ps1 on the self-hosted `ue5` runner; this
# script catches, before that runner is even reached, the drift that made
# the plugin unbuildable from 1.7.2 to 3.1.0 without anyone noticing:
#
#   1. the header the plugin ships (ThirdParty/AliceSDF/include/alice_sdf.h)
#      is byte-identical to include/alice_sdf.h (it was 15 lines behind)
#   2. every `alice_sdf_*` function the header declares is exported by the
#      cdylib built with `--features unreal`, and every exported one is
#      declared (an undeclared export is unusable from C++, a declared
#      non-export is a link error inside Unreal)
#  2a. the header's argument TYPES match the `extern "C"` signatures in
#      src/ffi (step 2 compares name sets only, which let alice_sdf_mirror
#      ship as `float mx, float my, float mz` against `fn(_, u8, u8, u8)`
#      from 1.7.2 to 4.0.0 — it links, and the flags arrive in the wrong
#      register bank)
#   3. AliceSDF.uplugin is valid JSON, VersionName == the crate version,
#      EngineVersion names a 5.x engine (it said "6.0.0")
#   4. every `#include "/Plugin/AliceSDF/..."` in Shaders/ resolves to a file
#   5. the generated corpus files match tests/common/corpus.rs
#      (examples/unreal_corpus_oracle.rs, `git diff --exit-code`)
#
# usage: scripts/unreal-abi-check.sh   (from anywhere; builds the cdylib)
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { printf '\033[1;31mFAIL:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

# ── 1. shipped header == canonical header ──────────────────────────────────
step "unreal-plugin/ThirdParty header == include/alice_sdf.h"
diff -u include/alice_sdf.h unreal-plugin/ThirdParty/AliceSDF/include/alice_sdf.h \
    || fail "unreal-plugin/ThirdParty/AliceSDF/include/alice_sdf.h is not include/alice_sdf.h (cp include/alice_sdf.h unreal-plugin/ThirdParty/AliceSDF/include/)"

# ── 2. header declarations ⇔ cdylib exports ────────────────────────────────
step "cdylib exports ⇔ header declarations (--features unreal)"
cargo build --features unreal
case "$(uname -s)" in
    Linux*)  lib=${CARGO_TARGET_DIR:-target}/debug/libalice_sdf.so;    syms() { nm -D --defined-only "$lib" | awk '{print $3}'; } ;;
    Darwin*) lib=${CARGO_TARGET_DIR:-target}/debug/libalice_sdf.dylib; syms() { nm -gU "$lib" | awk '{sub(/^_/, "", $3); print $3}'; } ;;
    *)       lib=${CARGO_TARGET_DIR:-target}/debug/alice_sdf.dll;      syms() { dumpbin -exports "$lib" | tr -d '\r' | awk 'NF==4 && $1 ~ /^[0-9]+$/ {print $4}'; } ;;
esac
[[ -f "$lib" ]] || fail "no cdylib at $lib"
# Declarations: `<ret> alice_sdf_xxx(` at line start (prototypes only; the
# header has no macros or typedef'd function pointers named alice_sdf_*).
declared=$(grep -oE '^[A-Za-z_][A-Za-z0-9_ *]*[ *]alice_sdf_[a-z0-9_]+\(' include/alice_sdf.h \
    | grep -oE 'alice_sdf_[a-z0-9_]+' | sort -u)
exported=$(syms | grep -E '^alice_sdf_[a-z0-9_]+$' | sort -u)
[[ -n "$declared" ]] || fail "no alice_sdf_* prototypes found in include/alice_sdf.h (regex drift?)"
[[ -n "$exported" ]] || fail "no alice_sdf_* exports in $lib"
missing=$(comm -23 <(echo "$declared") <(echo "$exported") || true)
undeclared=$(comm -13 <(echo "$declared") <(echo "$exported") || true)
echo "declared: $(echo "$declared" | wc -l), exported: $(echo "$exported" | wc -l)"
[[ -z "$missing" ]]    || fail "declared in alice_sdf.h but not exported by the cdylib (link error in Unreal):"$'\n'"$missing"
[[ -z "$undeclared" ]] || fail "exported by the cdylib but not declared in alice_sdf.h (add the prototype):"$'\n'"$undeclared"

# ── 2a. header declarations ⇔ Rust signatures, by TYPE ─────────────────────
# Step 2 compares name sets only. A prototype whose argument TYPES disagree
# with the Rust export links fine and passes every name check, so it shipped
# undetected from 1.7.2 to 4.0.0: alice_sdf_mirror was declared
# `float mx, float my, float mz` against `fn(SdfHandle, u8, u8, u8)`. An
# integer argument travels in a different register bank than a float one on
# every ABI we support (AArch64 x0-x7 / v0-v7, x86-64 SysV rdi-r9 / xmm0-7),
# so the callee read unrelated registers instead of the flags — deterministically
# on a given call path, which is why a passing test never revealed it.
# Step 1 could not catch it either: it compares the two header copies to each
# other, so an error present in both is invisible.
#
# This step is about types only. Name-set drift stays step 2's job, which
# checks the real cdylib rather than parsing source, so a name present on one
# side alone is reported here but does not fail (a cfg-gated export is legal).
step "header argument types ⇔ src/ffi extern \"C\" signatures"
py=python3; python3 -c 'import sys' >/dev/null 2>&1 || py=python
"$py" - <<'EOF'
import pathlib, re, sys

NORM = {
    "uint8_t": "u8", "uint16_t": "u16", "uint32_t": "u32", "uint64_t": "u64",
    "int8_t": "i8", "int16_t": "i16", "int32_t": "i32", "int64_t": "i64",
    "float": "f32", "double": "f64", "size_t": "usize",
    "c_char": "char", "c_uint": "u32", "c_int": "i32", "c_float": "f32",
}


def norm(t, extra_ptr=False):
    t = re.sub(r"\b(const|mut)\b", " ", t).replace("*", " ptr ")
    parts = [NORM.get(p, p) for p in t.split()]
    ptrs = ["ptr"] * (parts.count("ptr") + (1 if extra_ptr else 0))
    # C writes `float *`, Rust writes `*const f32`; put ptr first on both.
    return " ".join(ptrs + [p for p in parts if p != "ptr"])


def header_sigs(path):
    text = re.sub(r"/\*.*?\*/", " ", re.sub(r"//[^\n]*", " ", path.read_text()), flags=re.S)
    out = {}
    for m in re.finditer(r"\b(\w[\w *]*?)\s+(alice_sdf_\w+)\s*\(([^;]*?)\)\s*;", text, flags=re.S):
        name, args = m.group(2), m.group(3)
        if args.strip() in ("void", ""):
            out[name] = []
            continue
        params = []
        for a in args.split(","):
            a = a.strip()
            # drop the parameter name; `float m[16]` decays to a pointer
            stripped = re.sub(r"\b\w+\s*(\[\s*\d*\s*\])?$", "", a).strip()
            params.append(norm(stripped or a, extra_ptr="[" in a))
        out[name] = params
    return out


def rust_sigs(root):
    out = {}
    pat = re.compile(
        r'(?:pub\s+)?(?:const\s+|unsafe\s+)*extern\s+"C"\s+fn\s+(alice_sdf_\w+)\s*\((.*?)\)\s*(?:->|\{)',
        re.S,
    )
    for f in sorted(root.rglob("*.rs")):
        for m in pat.finditer(f.read_text()):
            out[m.group(1)] = [
                norm(a.split(":", 1)[1]) for a in m.group(2).split(",") if ":" in a
            ]
    return out


header = header_sigs(pathlib.Path("include/alice_sdf.h"))
rust = rust_sigs(pathlib.Path("src/ffi"))
if not header:
    sys.exit("no alice_sdf_* prototypes parsed from include/alice_sdf.h (regex drift?)")
if not rust:
    sys.exit('no extern "C" fns parsed from src/ffi (regex drift?)')

problems = []
for name in sorted(set(header) & set(rust)):
    h, r = header[name], rust[name]
    if len(h) != len(r):
        problems.append(f"  {name}: header takes {len(h)} args, Rust takes {len(r)}\n"
                        f"    header={h}\n    rust  ={r}")
        continue
    for i, (a, b) in enumerate(zip(h, r)):
        if a != b:
            problems.append(f"  {name} arg{i}: header declares {a!r}, Rust takes {b!r}")

only_h = sorted(set(header) - set(rust))
only_r = sorted(set(rust) - set(header))
print(f"compared {len(set(header) & set(rust))} functions "
      f"(header {len(header)}, src/ffi {len(rust)})")
for label, names in (("header only", only_h), ("src/ffi only", only_r)):
    if names:
        print(f"note: {label} (types not compared, step 2 owns name drift): {names}")

if problems:
    sys.exit("argument types disagree between include/alice_sdf.h and src/ffi "
             "(the header is what C++/C# callers compile against, so the "
             "declaration must match the Rust export exactly):\n" + "\n".join(problems))
print("ok: every shared function has identical argument types")
EOF

# ── 2b. no native library in the repository ────────────────────────────────
step "unreal-plugin/ThirdParty: no committed binaries"
committed=$(git ls-files 'unreal-plugin/ThirdParty/AliceSDF/lib/**'     | grep -E '\.(dll|lib|dylib|so|a)$' || true)
[[ -z "$committed" ]] || fail "native libraries are committed:
$committed
They are build products; a committed copy goes stale without anything failing
(the one removed in 3.2.0 was seven months and one law change behind). Remove
them (git rm --cached) — the release zip ships the real ones and
scripts/build_ue5_plugin.sh fills the directory locally."
for d in Win64 Mac Linux; do
    [[ -f "unreal-plugin/ThirdParty/AliceSDF/lib/$d/README.txt" ]]         || fail "unreal-plugin/ThirdParty/AliceSDF/lib/$d/README.txt is missing (it tells a user where the library comes from)"
done
echo "ok: only README.txt in lib/{Win64,Mac,Linux}"

# ── 3. .uplugin metadata ───────────────────────────────────────────────────
step "AliceSDF.uplugin metadata"
crate_version=$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')
# python3 on a stock Windows box is the Store stub; fall back to `python`.
py=python3; python3 -c 'import sys' >/dev/null 2>&1 || py=python
"$py" - "$crate_version" <<'EOF'
import json, sys
crate = sys.argv[1]
with open("unreal-plugin/AliceSDF.uplugin", encoding="utf-8") as f:
    p = json.load(f)
problems = []
if p.get("VersionName") != crate:
    problems.append(f'VersionName {p.get("VersionName")!r} != crate version {crate!r}')
ev = str(p.get("EngineVersion", ""))
if not ev.startswith("5."):
    problems.append(f'EngineVersion {ev!r} is not a 5.x engine')
mods = p.get("Modules", [])
if not any(m.get("Name") == "AliceSDF" and m.get("Type") == "Runtime" for m in mods):
    problems.append("no Runtime module named AliceSDF")
if problems:
    sys.exit("uplugin: " + "; ".join(problems))
print(f'uplugin ok: VersionName {crate}, EngineVersion {ev}, {len(mods)} module(s)')
EOF

# ── 4. shader virtual includes resolve ─────────────────────────────────────
step "Shaders/: every /Plugin/AliceSDF/ include exists"
bad=0
while IFS=: read -r file inc; do
    target="unreal-plugin/Shaders/${inc#/Plugin/AliceSDF/}"
    if [[ ! -f "$target" ]]; then
        echo "$file: #include \"$inc\" → $target missing" >&2
        bad=1
    fi
done < <(grep -roE --include='*.usf' --include='*.ush' -H '#include "/Plugin/AliceSDF/[^"]+"' unreal-plugin/Shaders \
    | sed -E 's/:#include "([^"]+)"/:\1/')
[[ $bad -eq 0 ]] || fail "unresolved plugin shader includes"
echo "ok: $(grep -rlE --include='*.usf' --include='*.ush' '' unreal-plugin/Shaders | wc -l) shader files"

# ── 5. generated corpus files are current ──────────────────────────────────
step "generated corpus oracle files == tests/common/corpus.rs"
# --features unreal: a narrower set rebuilds the cdylib this script
# just inspected (and the one Unreal links) without the FFI.
cargo run --example unreal_corpus_oracle --features unreal -- --plugin unreal-plugin --golden target/unreal-golden
git diff --exit-code --stat -- unreal-plugin/Shaders/CorpusOracle unreal-plugin/Source/AliceSDF/Private/Generated \
    || fail "generated corpus files are stale — commit the regenerated unreal-plugin/Shaders/CorpusOracle + Source/AliceSDF/Private/Generated"
untracked=$(git ls-files --others --exclude-standard -- unreal-plugin/Shaders/CorpusOracle unreal-plugin/Source/AliceSDF/Private/Generated)
[[ -z "$untracked" ]] || fail "generated corpus files are not committed:"$'\n'"$untracked"

printf '\n\033[1;32munreal-abi-check: header / exports / uplugin / shader includes / generated files all consistent\033[0m\n'
