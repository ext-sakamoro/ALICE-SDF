#!/usr/bin/env bash
# Whether `scripts/preflight.sh --quick` has to run the scip job (the reach ledger and the L0
# ratchet): yes when one of its inputs differs from origin/main, committed or not, and when
# there is no origin/main to compare with; no when the difference touches none of them.
# The ledger depends on callers, not only on the changed `pub` items: the library (src/), the
# callers it counts (examples/, benches/, fuzz/, and the crates it indexes on their own:
# server/, mobile/, bindings/), the features (Cargo.toml), the ledger and its baselines, and
# the scripts that build it. Renames are listed as a deletion and an addition (--no-renames),
# so a file moved out of an input directory still counts.
#
# Prints one line, "run: <reason>" or "skip: <reason>", and exits 0 (run) or 1 (skip).
# usage: scripts/scip_scope.sh   (from the repository root)
set -uo pipefail

base=$(git merge-base origin/main HEAD 2>/dev/null || true)
if [[ -z "$base" ]]; then
  echo "run: no merge-base with origin/main to compare with"
  exit 0
fi
changed=$( { git diff --no-renames --name-only "$base" HEAD; git diff --no-renames --name-only HEAD; git ls-files --others --exclude-standard; } | sort -u | grep . || true)
n_all=$(printf '%s\n' "$changed" | grep -c . || true)
inputs='^(src|examples|benches|fuzz|server|mobile|bindings)/|^Cargo\.toml$|^docs/integration-status\.md$|^scripts/(scip_index\.sh|scip_reach\.py|integration-baseline\.txt|wiring-baseline\.txt)$'
n_src=$(printf '%s\n' "$changed" | grep -cE "$inputs" || true)
if [[ "$n_src" -gt 0 ]]; then
  echo "run: $n_src scip input file(s) differ from origin/main (${base:0:12})"
  exit 0
fi
echo "skip: none of the $n_all file(s) that differ from origin/main (${base:0:12}) is a scip input"
exit 1
