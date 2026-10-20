#!/usr/bin/env bash
# Build the SCIP indexes that scripts/scip_reach.py reads.
#
# rust-analyzer only analyses the enabled features, so the default feature set
# leaves out every feature-gated module (ffi, python, godot, gpu, jit, the shader
# transpilers, the bridges, ...). The crate is indexed three times:
#   native  every feature except `wasm`, on x86_64-unknown-linux-gnu
#   wasm    `wasm` only, on wasm32-unknown-unknown (src/wasm.rs is compiled for
#           `target_arch = "wasm32"` only, so a native index leaves it out)
#   macos   every feature except `wasm`, on aarch64-apple-darwin (code under
#           `cfg(target_os = "macos")` is left out of the native index)
# rust-analyzer evaluates cfg for the configured target without compiling for it,
# so all three are built the same way on any host, Linux included.
#
# A fixed target is used for the native index too: rust-analyzer evaluates
# `cfg(target_arch = ...)` for the target it analyses, so without one an arm64
# host drops the x86_64-only SIMD items and the ledger depends on where it was
# generated. The other crates of the repository (fuzz, server, mobile, openxr)
# are indexed on their own.
#
# usage: scripts/scip_index.sh [--configs-only] [OUT_DIR]   (default: target/scip)
#   --configs-only  write the rust-analyzer configurations and stop (scripts/test_scip_reach.py)

set -euo pipefail
cd "$(dirname "$0")/.."

configs_only=0
if [ "${1:-}" = "--configs-only" ]; then
  configs_only=1
  shift
fi
out="${1:-target/scip}"
mkdir -p "$out"

native=$(python3 - <<'EOF'
import json, tomllib
features = tomllib.load(open("Cargo.toml", "rb"))["features"]
names = sorted(n for n in features if n not in ("default", "wasm"))
print(json.dumps({"cargo": {"features": names, "target": "x86_64-unknown-linux-gnu"}}))
EOF
)
printf '%s\n' "$native" > "$out/native.json"
printf '%s\n' "${native/x86_64-unknown-linux-gnu/aarch64-apple-darwin}" > "$out/macos.json"
printf '%s\n' '{"cargo":{"features":["wasm"],"noDefaultFeatures":true,"target":"wasm32-unknown-unknown"}}' > "$out/wasm.json"
[ "$configs_only" = 0 ] || exit 0

for set in native wasm macos; do
  rm -f "$out/$set.scip"
  rust-analyzer scip . --config-path "$out/$set.json" --output "$out/$set.scip"
  [ -s "$out/$set.scip" ] || { echo "scip_index: $out/$set.scip is empty" >&2; exit 1; }
done

# The crates of this repository that depend on this one through a path dependency
# are separate crates, so the indexes above do not contain them although
# scip_reach.py counts their code as callers (scripts/scip_reach.py SUBCRATES):
#   fuzz/                   fuzz targets (example-level callers)
#   server/                 REST server, mobile/uniffi-wrapper/  UniFFI bindings,
#   bindings/openxr/        OpenXR helpers (binding callers)
# Their references to this crate carry the same symbols as the crate's own index.
# Each with its default features, on the same fixed target.
printf '%s\n' '{"cargo":{"target":"x86_64-unknown-linux-gnu"}}' > "$out/sub.json"
written="$out/native.scip $out/wasm.scip $out/macos.scip"
for pair in fuzz:fuzz server:server mobile:mobile/uniffi-wrapper openxr:bindings/openxr; do
  name=${pair%%:*}
  dir=${pair#*:}
  rm -f "$out/$name.scip"
  rust-analyzer scip "$dir" --config-path "$out/sub.json" --output "$out/$name.scip"
  [ -s "$out/$name.scip" ] || { echo "scip_index: $out/$name.scip is empty" >&2; exit 1; }
  written="$written $out/$name.scip"
done
# The commit the indexes were built from, so a reader can refuse an index of another commit
# (scripts/api_boundary_parsers.py): `<sha> clean` or `<sha> dirty`
state=clean
[ -z "$(git status --porcelain)" ] || state=dirty
printf '%s %s\n' "$(git rev-parse HEAD)" "$state" > "$out/commit"
echo "scip_index: wrote $written" $out/commit
