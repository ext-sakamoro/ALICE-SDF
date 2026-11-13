#!/usr/bin/env bash
# scripts/law_ratchets.sh — the ALICE-Law ratchets on this repository, before it lands.
#
# The same script in every crate; the repository's name is in its pin. Three of ALICE-Law's
# checkers, each run on this repository alone (`--only NAME=TREE`):
#   - scripts/no_panic_ratchet.py: production panic sites (unwrap / expect / panic! / todo! /
#     unimplemented! / unreachable! / assert! and their lint allows), counted by
#     tools/no-panic-scan, against the repository's row in conformance/no-panic-baseline.toml;
#   - scripts/structural_lint.py: the ast-grep rules (production prints, HashMap / HashSet
#     iteration, ast-grep-ignore markers) against conformance/structural-lint-baseline.toml;
#   - scripts/added_function_pins.py: a call of an alice-det-math function added after its
#     SEMANTICS_ID was cut is folded with that function's own pin (only for a repository whose
#     Cargo.toml depends on alice-det-math; the checker has nothing to compare otherwise).
# A count above the baseline is red, and so is one below it until ALICE-Law's baseline is
# lowered (the pin then moves to that revision): the counts only go down.
#
# ALICE-Law is fetched at the revision .github/law-ratchets-pin names (its whole tree: the
# checkers import their neighbours and read tools/ and lint/), and each checker and baseline
# is refused unless its SHA-256 is the one written there. A change in ALICE-Law never changes
# this repository's result by itself; the pin moves in a commit of its own.
#
# ast-grep is the version the pin names: the one on PATH when its version is that one, else the
# wheel installed with `pip --require-hashes` (the pin's hashes) into a cache directory.
# tools/no-panic-scan is built by the checker into a cache directory keyed by the revision.
#
# What is read is the committed tree (`git archive HEAD`), not the checkout. Each checker exits
# 2 when it compared nothing or could not read, and that fails the run like a finding does.
#
# usage: scripts/law_ratchets.sh [TREE]
#   TREE: a directory to check instead of the committed tree (the tests)
set -euo pipefail
cd "$(dirname "$0")/.."

py=$(command -v python3 || command -v python || true)
if [[ -z "$py" ]]; then
  echo "no python3 or python on PATH" >&2
  exit 2
fi

pin=.github/law-ratchets-pin
val() { awk -v k="$1" '$1 == k { print $2 }' "$pin"; }
name=$(val name)
rev=$(val rev)
astgrep=$(val ast-grep)
if [[ ! "$name" =~ ^ALICE-[A-Za-z0-9-]+$ || ! "$rev" =~ ^[0-9a-f]{40}$ || ! "$astgrep" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "$pin: needs 'name ALICE-<repository>', 'rev <40 hex>' and 'ast-grep <version>'" >&2
  exit 2
fi
declare -a checked=(
  "scripts/no_panic_ratchet.py:no-panic-checker-sha256"
  "conformance/no-panic-baseline.toml:no-panic-baseline-sha256"
  "scripts/structural_lint.py:structural-lint-checker-sha256"
  "conformance/structural-lint-baseline.toml:structural-lint-baseline-sha256"
  "scripts/added_function_pins.py:added-function-pins-checker-sha256"
)
wheels=$(awk '$1 == "ast-grep-wheel-sha256" { print $2 }' "$pin")
if [[ -z "$wheels" ]]; then
  echo "$pin: needs at least one 'ast-grep-wheel-sha256 <64 hex>'" >&2
  exit 2
fi
for w in $wheels; do
  if [[ ! "$w" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$pin: 'ast-grep-wheel-sha256 $w' is not 64 hex" >&2
    exit 2
  fi
done

tree=${1:-}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
law="$tmp/law"
git init -q "$law"
if ! git -C "$law" fetch -q --depth 1 https://github.com/ext-sakamoro/ALICE-Law "$rev"; then
  echo "ALICE-Law $rev: fetch failed" >&2
  exit 2
fi
git -C "$law" checkout -q --detach FETCH_HEAD
sum() { "$py" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }
for entry in "${checked[@]}"; do
  path=${entry%%:*}
  want=$(val "${entry##*:}")
  if [[ ! "$want" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$pin: needs '${entry##*:} <64 hex>'" >&2
    exit 2
  fi
  got=$(sum "$law/$path")
  if [[ "$got" != "$want" ]]; then
    echo "$path at $rev has sha256 $got, $pin says $want" >&2
    exit 2
  fi
done

cache=${XDG_CACHE_HOME:-$HOME/.cache}/alice-law-ratchets
mkdir -p "$cache"
# ast-grep at the pinned version
ag=$(command -v ast-grep || true)
if [[ -z "$ag" || "$("$ag" --version 2>/dev/null)" != "ast-grep $astgrep" ]]; then
  venv="$cache/ast-grep-$astgrep"
  if [[ ! -x "$venv/bin/ast-grep" || "$("$venv/bin/ast-grep" --version 2>/dev/null)" != "ast-grep $astgrep" ]]; then
    rm -rf "$venv"
    "$py" -m venv "$venv"
    req="$tmp/ast-grep.txt"
    {
      printf 'ast-grep-cli==%s' "$astgrep"
      for w in $wheels; do printf ' --hash=sha256:%s' "$w"; done
      printf '\n'
    } > "$req"
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$venv/bin/python" -m pip install -q --only-binary=:all: --require-hashes -r "$req"
  fi
  ag="$venv/bin/ast-grep"
fi
if [[ "$("$ag" --version)" != "ast-grep $astgrep" ]]; then
  echo "ast-grep: $("$ag" --version), the pin says $astgrep" >&2
  exit 2
fi
export AST_GREP="$ag"
# tools/no-panic-scan, built once per revision
export CARGO_TARGET_DIR="$cache/no-panic-scan-$rev"

if [[ -z "$tree" ]]; then
  tree="$tmp/tree"
  mkdir "$tree"
  git -c core.autocrlf=false -c core.eol=lf archive HEAD | tar -x -C "$tree"
  echo "law_ratchets: checking the committed tree (HEAD $(git rev-parse --short HEAD)) at ALICE-Law ${rev:0:12}; uncommitted changes are not checked"
fi

# every checker runs; the run exits with the worst code (2 unreadable > 1 finding > 0)
status=0
run() {
  local rc=0
  "$py" "$@" || rc=$?
  if (( rc > status )); then status=$rc; fi
}
run "$law/scripts/no_panic_ratchet.py" --only "$name=$tree"
run "$law/scripts/structural_lint.py" --only "$name=$tree"
if grep -Eq '^[[:space:]]*alice-det-math[[:space:]]*=' "$tree/Cargo.toml"; then
  run "$law/scripts/added_function_pins.py" --only "$name=$tree"
else
  echo "added_function_pins: $name does not depend on alice-det-math (Cargo.toml), nothing to fold"
fi
exit "$status"
