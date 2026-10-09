#!/usr/bin/env bash
# Teeth of examples/bake_assets: corrupt a baked directory in one way at a
# time and require `bake_assets verify` to exit non-zero. Each mutation
# refreshes the SHA-256 of the file it touches, so the semantic check (not
# the hash) has to catch it. The untouched copy must still verify, so the
# red comes from the mutation and not from copying.
#
# usage: scripts/bake_teeth.sh <baked-dir> [cargo features]
#   e.g. scripts/bake_teeth.sh bake-out gpu-mesh
set -euo pipefail

src=${1:?usage: scripts/bake_teeth.sh <baked-dir> [cargo features]}
feats=${2:-}
run() {
  if [ -n "$feats" ]; then
    cargo run -q --example bake_assets --features "$feats" -- "$@"
  else
    cargo run -q --example bake_assets -- "$@"
  fi
}

kinds=(drop-face shrink-aabb no-scenes)
gpu=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpu"]["status"])' "$src/manifest.json")
if [ "$gpu" = "ok" ]; then
  kinds+=(gpu-vertex)
else
  echo "SKIPPED: gpu-vertex mutation (this bake has no GPU mesh)"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cp -R "$src" "$tmp/clean"
run verify "$tmp/clean" > "$tmp/clean.log" 2>&1 \
  || { cat "$tmp/clean.log"; echo "the unmutated copy does not verify"; exit 1; }

# the check each mutation is aimed at (other checks may fail too)
expect() {
  case $1 in
    drop-face) echo 'boundary edges \(not watertight\)' ;;
    shrink-aabb) echo 'collider AABB .* does not contain' ;;
    no-scenes) echo 'no scene verified' ;;
    gpu-vertex) echo 'gpu-vs-cpu: vertex-to-surface distance' ;;
  esac
}

red=0
for k in "${kinds[@]}"; do
  rm -rf "$tmp/m"
  cp -R "$src" "$tmp/m"
  run mutate "$k" "$tmp/m"
  if run verify "$tmp/m" > "$tmp/log" 2>&1; then
    cat "$tmp/log"
    echo "TOOTHLESS: verify passed after the $k mutation"
    exit 1
  fi
  if ! grep -Eq "$(expect "$k")" "$tmp/log"; then
    cat "$tmp/log"
    echo "verify failed after $k, but not on the check aimed at it ($(expect "$k"))"
    exit 1
  fi
  echo "red as expected: $k"
  grep -E '^(FAIL|error)' "$tmp/log" | sed 's/^/    /'
  red=$((red + 1))
done

if [ "$red" -eq 0 ]; then
  echo "no mutation ran"
  exit 1
fi
echo "bake teeth: $red/${#kinds[@]} mutations red"
