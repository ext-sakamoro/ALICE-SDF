#!/usr/bin/env bash
# Whether `scripts/preflight.sh --quick` has to run the scip job (the reach ledger and the L0
# ratchet): yes when src/ differs from origin/main, committed or not, and when there is no
# origin/main to compare with; no when the difference is outside src/ only (docs, tests, CI).
# The reach ledger depends on callers, not only on the changed `pub` items, so any change
# under src/ can move it.
#
# Prints one line, "run: <reason>" or "skip: <reason>", and exits 0 (run) or 1 (skip).
# usage: scripts/scip_scope.sh   (from the repository root)
set -uo pipefail

base=$(git merge-base origin/main HEAD 2>/dev/null || true)
if [[ -z "$base" ]]; then
  echo "run: no merge-base with origin/main to compare with"
  exit 0
fi
changed=$( { git diff --name-only "$base" HEAD; git diff --name-only HEAD; git ls-files --others --exclude-standard; } | sort -u | grep . || true)
n_all=$(printf '%s\n' "$changed" | grep -c . || true)
n_src=$(printf '%s\n' "$changed" | grep -c '^src/' || true)
if [[ "$n_src" -gt 0 ]]; then
  echo "run: $n_src file(s) under src/ differ from origin/main (${base:0:12})"
  exit 0
fi
echo "skip: none of the $n_all file(s) that differ from origin/main (${base:0:12}) is under src/"
exit 1
