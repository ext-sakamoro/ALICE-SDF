#!/usr/bin/env bash
# Build the SCIP indexes that scripts/scip_reach.py reads.
#
# rust-analyzer only analyses the enabled features, so the default feature set
# leaves out every feature-gated module (ffi, python, godot, gpu, jit, the shader
# transpilers, the bridges, ...). The crate is indexed twice:
#   native  every feature except `wasm`, on x86_64-unknown-linux-gnu
#   wasm    `wasm` only, on wasm32-unknown-unknown (src/wasm.rs is compiled for
#           `target_arch = "wasm32"` only, so a native index leaves it out)
#
# A fixed target is used for the native index too: rust-analyzer evaluates
# `cfg(target_arch = ...)` for the target it analyses, so without one an arm64
# host drops the x86_64-only SIMD items and the ledger depends on where it was
# generated. The fuzz crate is indexed on its own.
#
# usage: scripts/scip_index.sh [OUT_DIR]   (default: target/scip)

set -euo pipefail
cd "$(dirname "$0")/.."

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
printf '%s\n' '{"cargo":{"features":["wasm"],"noDefaultFeatures":true,"target":"wasm32-unknown-unknown"}}' > "$out/wasm.json"

for set in native wasm; do
  rm -f "$out/$set.scip"
  rust-analyzer scip . --config-path "$out/$set.json" --output "$out/$set.scip"
  [ -s "$out/$set.scip" ] || { echo "scip_index: $out/$set.scip is empty" >&2; exit 1; }
done

# fuzz/ is its own crate (path dependency on this one), so the indexes above do
# not contain it although scip_reach.py counts fuzz targets as callers. Its
# references to this crate carry the same symbols as the crate's own index.
# Default fuzz features, same fixed target.
printf '%s\n' '{"cargo":{"target":"x86_64-unknown-linux-gnu"}}' > "$out/fuzz.json"
rm -f "$out/fuzz.scip"
rust-analyzer scip fuzz --config-path "$out/fuzz.json" --output "$out/fuzz.scip"
[ -s "$out/fuzz.scip" ] || { echo "scip_index: $out/fuzz.scip is empty" >&2; exit 1; }
echo "scip_index: wrote $out/native.scip $out/wasm.scip $out/fuzz.scip"
