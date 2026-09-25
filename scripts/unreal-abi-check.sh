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
