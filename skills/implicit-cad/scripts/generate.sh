#!/usr/bin/env bash
# ALICE-SDF `alice-implicit-cad` skill — validate SDF file (info dump).
#
# Usage:
#   scripts/generate.sh <input.asdf.json>
#   scripts/generate.sh <input.asdf>
#   scripts/generate.sh --help
#
# Runs `alice-sdf info` on the input, reporting the SDF tree structure,
# node count, and estimated bounds. Use this as the first check after
# authoring a new .asdf.json file.
#
# For actual mesh export, use scripts/export.sh (OBJ/GLB/FBX/USDA/ABC).
# For 3D print output, use scripts/print.sh (STL/3MF/OBJ).
#
# Shader transpile (GLSL/WGSL/HLSL) is NOT exposed via the alice-sdf CLI
# in the current version — it requires the library API:
#   use alice_sdf::glsl::to_glsl; let src = to_glsl(&node);
# Use the alice-lol-sdf skill for LLM-oriented DSL authoring with
# constrained decoding.
#
# Requires:
#   - Rust toolchain
#   - ALICE-SDF workspace at ../..

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
CRATE_DIR="$(cd "$SKILL_DIR/../.." && pwd)"

usage() {
    sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    usage 0
fi

INPUT="${1:?input SDF file required (.asdf or .asdf.json)}"

if [[ ! -f "$INPUT" ]]; then
    echo "error: input file not found: $INPUT" >&2
    exit 2
fi

cd "$CRATE_DIR"

echo "[alice-implicit-cad] Running alice-sdf info on $INPUT..." >&2
cargo run --release --features cli --bin alice-sdf -- info "$INPUT"
