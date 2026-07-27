#!/usr/bin/env bash
# ALICE-SDF `alice-implicit-cad` skill — export SDF file to mesh artifact.
#
# Usage:
#   scripts/export.sh <input.asdf.json> --output part.glb
#   scripts/export.sh <input.asdf.json> --output part.obj --resolution 128
#   scripts/export.sh <input.asdf.json> --output part.fbx --bounds 3.0
#   scripts/export.sh --help
#
# Format is auto-detected from the --output extension:
#   .obj  — Wavefront OBJ (always available)
#   .glb  — glTF 2.0 binary
#   .fbx  — Autodesk FBX (requires --features fbx on parent crate)
#   .usda — USD ASCII (requires --features usd)
#   .abc  — Alembic (requires --features alembic)
#
# For STL / 3MF (3D print), use scripts/print.sh instead — the underlying
# alice-sdf CLI splits export (visualization formats) from print (fabrication).
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
    sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    usage 0
fi

INPUT="${1:?input .asdf/.asdf.json file required}"
shift

OUTPUT=""
RESOLUTION="64"
BOUNDS="2.0"

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

if [[ -z "$OUTPUT" ]]; then
    echo "error: --output required (extension determines format)" >&2
    exit 3
fi

cd "$CRATE_DIR"

echo "[alice-implicit-cad] Exporting $INPUT → $OUTPUT (resolution=$RESOLUTION, bounds=$BOUNDS)..." >&2
cargo run --release --features cli --bin alice-sdf -- \
    export "$INPUT" --output "$OUTPUT" --resolution "$RESOLUTION" --bounds "$BOUNDS"

echo "[alice-implicit-cad] Wrote $OUTPUT" >&2
