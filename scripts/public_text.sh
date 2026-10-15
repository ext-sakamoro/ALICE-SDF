#!/usr/bin/env bash
# scripts/public_text.sh — the ALICE-Law public text check on this repository.
#
# The checker is ALICE-Law's own scripts/public_text_check.py, fetched at the
# revision .github/public-text-pin names and refused unless its SHA-256 is the
# one written there (the file at a named revision, and byte-identical to it).
# It reads every tracked file at HEAD (path and text), then the commits in
# BASE..HEAD (each message, added line and added path). It exits 2 when it
# compared nothing or cannot know what to compare (a malformed pin, a base that
# is not a commit), and that fails the run like a finding does.
#
# The pin moves only in a commit of its own that passes this check at the new
# revision; a change in ALICE-Law never changes this repository's result by itself.
#
# A commit that changes .public-text.toml is always a finding in the commit range
# (every change of an exemption is shown to a reviewer): it lands alone, and the
# run for it is red by design. A release is cut only from a later commit whose
# run is green.
#
# usage: [PUBLIC_TEXT_BASE=<commit>] scripts/public_text.sh [BASE]
#   BASE: the argument, else PUBLIC_TEXT_BASE (preflight: the commit the change
#   starts from, as CI takes the previous head), else merge-base origin/main HEAD
set -euo pipefail
cd "$(dirname "$0")/.."

pin=.github/public-text-pin
rev=$(awk '$1 == "rev" { print $2 }' "$pin")
sum=$(awk '$1 == "sha256" { print $2 }' "$pin")
if [[ ! "$rev" =~ ^[0-9a-f]{40}$ || ! "$sum" =~ ^[0-9a-f]{64}$ ]]; then
  echo "$pin: needs 'rev <40 hex>' and 'sha256 <64 hex>'" >&2
  exit 2
fi

# the base first (before any network): one that is not a commit here would leave
# the commits unread, which is not "no new commit"
base=${1:-${PUBLIC_TEXT_BASE:-}}
if [[ -z "$base" ]]; then
  base=$(git merge-base origin/main HEAD)
fi
if ! git rev-parse -q --verify "$base^{commit}" >/dev/null; then
  echo "base $base is not a commit in this repository: the commits cannot be read" >&2
  exit 2
fi
n=$(git rev-list --count "$base..HEAD")

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git init -q "$tmp/law"
git -C "$tmp/law" fetch -q --depth 1 https://github.com/ext-sakamoro/ALICE-Law "$rev"
git -C "$tmp/law" show FETCH_HEAD:scripts/public_text_check.py > "$tmp/public_text_check.py"
got=$(python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "$tmp/public_text_check.py")
if [[ "$got" != "$sum" ]]; then
  echo "public_text_check.py at $rev has sha256 $got, $pin says $sum" >&2
  exit 2
fi

python3 "$tmp/public_text_check.py" --repo . --tree HEAD

if [[ "$n" -gt 0 ]]; then
  python3 "$tmp/public_text_check.py" --repo . --range "$base..HEAD"
else
  echo "no new commit (base $base is HEAD)"
fi
