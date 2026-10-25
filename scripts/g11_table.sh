#!/usr/bin/env bash
# scripts/g11_table.sh — ALICE-Law G11 on this repository: every gate alice-law-conformance.toml
# declares pass has a row in conformance/g11.toml naming its checker, the canary it must flag, the
# test that fails when it compared nothing, and the shared-file rows that keep a shared checker identical.
#
# The checker (scripts/g11_table_check.py) is ALICE-Law's own, fetched at the revision
# .github/g11-table-pin names and refused unless its SHA-256 is the one written there; a change in
# ALICE-Law never changes this repository's result by itself. It exits 2 when it compared nothing
# or cannot read what it needs (a malformed pin, a fetch that failed), and that fails the run like a
# finding does.
#
# usage: scripts/g11_table.sh
set -euo pipefail
cd "$(dirname "$0")/.."

py=$(command -v python3 || command -v python || true)
if [[ -z "$py" ]]; then
  echo "no python3 or python on PATH" >&2
  exit 2
fi

pin=.github/g11-table-pin
name=$(awk '$1 == "name" { print $2 }' "$pin")
rev=$(awk '$1 == "rev" { print $2 }' "$pin")
checker_sum=$(awk '$1 == "checker-sha256" { print $2 }' "$pin")
if [[ ! "$name" =~ ^ALICE-[A-Za-z0-9-]+$ || ! "$rev" =~ ^[0-9a-f]{40}$ || ! "$checker_sum" =~ ^[0-9a-f]{64}$ ]]; then
  echo "$pin: needs 'name ALICE-<repository>', 'rev <40 hex>' and 'checker-sha256 <64 hex>'" >&2
  exit 2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git init -q "$tmp/law"
if ! git -C "$tmp/law" fetch -q --depth 1 https://github.com/ext-sakamoro/ALICE-Law "$rev"; then
  echo "ALICE-Law $rev: fetch failed" >&2
  exit 2
fi
git -C "$tmp/law" show "FETCH_HEAD:scripts/g11_table_check.py" > "$tmp/g11_table_check.py"
got=$("$py" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$tmp/g11_table_check.py")
if [[ "$got" != "$checker_sum" ]]; then
  echo "scripts/g11_table_check.py at $rev has sha256 $got, $pin says $checker_sum" >&2
  exit 2
fi
"$py" "$tmp/g11_table_check.py" --repo . --name "$name"
