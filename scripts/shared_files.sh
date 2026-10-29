#!/usr/bin/env bash
# scripts/shared_files.sh — ALICE-Law G11 (c) on this repository: the files this crate
# shares byte for byte with the other ALICE crates are the bytes ALICE-Law's table records.
#
# The same script in every crate; the repository's name is in its pin. The table
# (conformance/shared-files.toml) and its checker (scripts/shared_files_check.py)
# are ALICE-Law's own, fetched at the revision .github/shared-files-pin names and refused
# unless each SHA-256 is the one written there. The checker then compares this tree with
# the table: every file the table lists for this repository is present with the recorded
# bytes, and no shared path is present that the table neither lists nor exempts for it. It
# exits 2 when it compared nothing or cannot know what to compare (a malformed pin, a fetch
# that failed), and that fails the run like a difference does.
#
# The pin moves in a commit that also brings the copies to the bytes of the new revision's
# table; a change in ALICE-Law never changes this repository's result by itself.
#
# What is compared is the committed tree (`git archive HEAD` with no line-ending
# conversion), not the checkout: a checkout that converts line endings (Windows) holds
# other bytes than the ones every crate shares.
#
# A private repository (one the table, which lists public repositories only, never names) adds
# the pin line `visibility private`: the checker then runs in its private mode, comparing each
# table path present with its row. A repository the table lists cannot use it (the checker
# refuses), and a pin without the line is `visibility public`.
#
# usage: scripts/shared_files.sh [TREE]
#   TREE: a directory to check instead of the committed tree (the tests)
set -euo pipefail
cd "$(dirname "$0")/.."

py=$(command -v python3 || command -v python || true)
if [[ -z "$py" ]]; then
  echo "no python3 or python on PATH" >&2
  exit 2
fi

pin=.github/shared-files-pin
name=$(awk '$1 == "name" { print $2 }' "$pin")
rev=$(awk '$1 == "rev" { print $2 }' "$pin")
table_sum=$(awk '$1 == "table-sha256" { print $2 }' "$pin")
checker_sum=$(awk '$1 == "checker-sha256" { print $2 }' "$pin")
visibility=$(awk '$1 == "visibility" { print $2 }' "$pin")
# no line is public; a line with no value is malformed (it must not read as public)
if [[ $(awk '$1 == "visibility" { n++ } END { print n + 0 }' "$pin") -eq 0 ]]; then
  visibility=public
fi
if [[ ! "$name" =~ ^ALICE-[A-Za-z0-9-]+$ || ! "$rev" =~ ^[0-9a-f]{40}$ || ! "$table_sum" =~ ^[0-9a-f]{64}$ || ! "$checker_sum" =~ ^[0-9a-f]{64}$ ]]; then
  echo "$pin: needs 'name ALICE-<repository>', 'rev <40 hex>', 'table-sha256 <64 hex>' and 'checker-sha256 <64 hex>'" >&2
  exit 2
fi
if [[ "$visibility" != public && "$visibility" != private ]]; then
  echo "$pin: 'visibility' is 'public' or 'private', got '$visibility'" >&2
  exit 2
fi
mode=()
[[ "$visibility" == private ]] && mode=(--private)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git init -q "$tmp/law"
if ! git -C "$tmp/law" fetch -q --depth 1 https://github.com/ext-sakamoro/ALICE-Law "$rev"; then
  echo "ALICE-Law $rev: fetch failed" >&2
  exit 2
fi
sum() { "$py" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }
for f in conformance/shared-files.toml:table scripts/shared_files_check.py:checker; do
  path=${f%%:*}
  want=$table_sum
  [[ ${f##*:} == checker ]] && want=$checker_sum
  out="$tmp/$(basename "$path")"
  git -C "$tmp/law" show "FETCH_HEAD:$path" > "$out"
  got=$(sum "$out")
  if [[ "$got" != "$want" ]]; then
    echo "$path at $rev has sha256 $got, $pin says $want" >&2
    exit 2
  fi
done

tree=${1:-}
if [[ -z "$tree" ]]; then
  tree="$tmp/tree"
  mkdir "$tree"
  # core.autocrlf=false: git archive applies the checkout's line-ending conversion, which
  # on Windows would turn LF into CRLF and differ from the bytes every crate shares
  git -c core.autocrlf=false -c core.eol=lf archive HEAD | tar -x -C "$tree"
  echo "shared_files: checking the committed tree (HEAD $(git rev-parse --short HEAD)); uncommitted changes are not checked"
fi
"$py" "$tmp/shared_files_check.py" --table "$tmp/shared-files.toml" --repo "$tree" --name "$name" ${mode[@]+"${mode[@]}"}
