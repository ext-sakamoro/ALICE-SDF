#!/usr/bin/env bash
# ALICE-SDF `alice-implicit-cad` skill — 3D print mesh from SDF.
#
# Usage:
#   scripts/print.sh <input.asdf.json> --output part.stl
#   scripts/print.sh <input.asdf.json> --output part.3mf --resolution 128
#   scripts/print.sh <input.asdf.json> --output part.obj --bounds 3.0
#   scripts/print.sh --help
#
# Format is auto-detected from the --output extension:
#   .stl — binary STL (universal 3D print)
#   .3mf — 3D Manufacturing Format (Bambu Studio, PrusaSlicer, OrcaSlicer)
#   .obj — Wavefront OBJ
#
# Default resolution is 128 (higher than export.sh's 64) since print output
# benefits from finer marching cubes. Cap at ~192 to keep triangle count
# tractable for slicers.
#
# For higher-level DSL authoring with LLM constrained decoding, use the
# alice-lol-sdf skill and its scripts/print.sh (which flows through
# alice_lol::print_export::{lol_to_stl, node_to_3mf} with automatic
# MeshRepair::repair_all).
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
    sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    usage 0
fi

INPUT="${1:?input .asdf/.asdf.json file required}"
shift

OUTPUT=""
RESOLUTION="128"
BOUNDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT="${2:?output path required}"; shift 2 ;;
        --resolution) RESOLUTION="${2:?resolution required}"; shift 2 ;;
        --bounds) BOUNDS="${2:?bounds required}"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

if [[ ! -f "$INPUT" ]]; then
    echo "error: input file not found: $INPUT" >&2
    exit 2
fi

if (( RESOLUTION > 192 )); then
    echo "warning: resolution $RESOLUTION exceeds 192 (parent crate rule for slicer safety)" >&2
fi

cd "$CRATE_DIR"

BOUNDS_ARG=""
if [[ -n "$BOUNDS" ]]; then
    BOUNDS_ARG="--bounds $BOUNDS"
fi

OUTPUT_ARG=""
if [[ -n "$OUTPUT" ]]; then
    OUTPUT_ARG="--output $OUTPUT"
fi

echo "[alice-implicit-cad] Printing $INPUT${OUTPUT:+ → $OUTPUT} (resolution=$RESOLUTION${BOUNDS:+, bounds=$BOUNDS})..." >&2
# shellcheck disable=SC2086
cargo run --release --features cli --bin alice-sdf -- \
    print "$INPUT" $OUTPUT_ARG --resolution "$RESOLUTION" $BOUNDS_ARG

echo "[alice-implicit-cad] Print export complete." >&2
