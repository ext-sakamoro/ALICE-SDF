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
# This script's `new` needs a pin at ALICE-Law a5824065 or later (the checker's
# --new-since): taking these bytes moves the pin in the same commit, and an older pin
# exits 2 naming the revision.
#
# A commit that changes .public-text.toml is always a finding in the commit range
# (every change of an exemption is shown to a reviewer): it lands alone, and the
# run for it is red by design. A release is cut only from a later commit whose
# run is green.
#
# usage: [PUBLIC_TEXT_BASE=<commit>|new] scripts/public_text.sh [BASE|new] [--plan]
#   BASE: the argument, else PUBLIC_TEXT_BASE (preflight: the commit the change
#   starts from, as CI takes the previous head), else merge-base origin/main HEAD
#   new: the commits this push newly publishes, for a push with no previous head (the
#   first push of a branch, or of the repository): `git rev-list HEAD --not` every
#   other remote branch (origin/*, less the pushed one, PUBLIC_TEXT_BRANCH or the CI's
#   GITHUB_REF_NAME). A branch from main reads only its own commits, not main's
#   history; a repository's first push (the remote has no other branch) reads every
#   commit, the root included. When the remote has other branches but none of them
#   was fetched, the commits cannot be told apart: exit 2 (CI fetches every head)
#   --plan: print what would be read (the mode, the refs, the commit count) and exit
set -euo pipefail
cd "$(dirname "$0")/.."

pin=.github/public-text-pin
rev=$(awk '$1 == "rev" { print $2 }' "$pin")
sum=$(awk '$1 == "sha256" { print $2 }' "$pin")
if [[ ! "$rev" =~ ^[0-9a-f]{40}$ || ! "$sum" =~ ^[0-9a-f]{64}$ ]]; then
  echo "$pin: needs 'rev <40 hex>' and 'sha256 <64 hex>'" >&2
  exit 2
fi

plan=0
args=()
for a in "$@"; do
  if [[ "$a" == --plan ]]; then plan=1; else args+=("$a"); fi
done

# the base first (before any network): one that is not a commit here would leave
# the commits unread, which is not "no new commit"
base=${args[0]:-${PUBLIC_TEXT_BASE:-}}
others=()
if [[ "$base" == new ]]; then
  branch=${PUBLIC_TEXT_BRANCH:-${GITHUB_REF_NAME:-}}
  while IFS= read -r ref; do
    [[ -n "$ref" && "$ref" != refs/remotes/origin/HEAD && "$ref" != "refs/remotes/origin/$branch" ]] && others+=("$ref")
  done < <(git for-each-ref --format='%(refname)' refs/remotes/origin)
  if [[ ${#others[@]} -eq 0 ]]; then
    # no other branch here: the repository's first push, or the other branches were not fetched
    remote=$(git ls-remote --heads origin) || {
      echo "the remote's branches cannot be listed: a first push cannot be told from unfetched refs" >&2
      exit 2
    }
    remote_others=$(printf '%s\n' "$remote" | awk -v b="refs/heads/$branch" 'NF == 2 && $2 != b' | wc -l | tr -d ' ')
    if [[ "$remote_others" -gt 0 ]]; then
      echo "the remote has $remote_others other branch(es) and none was fetched: the new commits cannot be told apart" >&2
      exit 2
    fi
  fi
  n=$(git rev-list --count HEAD ${others[@]+"${others[@]/#/^}"})
  mode="new since ${#others[@]} ref(s)"
else
  if [[ -z "$base" ]]; then
    base=$(git merge-base origin/main HEAD)
  fi
  if ! git rev-parse -q --verify "$base^{commit}" >/dev/null; then
    echo "base $base is not a commit in this repository: the commits cannot be read" >&2
    exit 2
  fi
  n=$(git rev-list --count "$base..HEAD")
  mode="range $base..HEAD"
fi
if [[ "$plan" == 1 ]]; then
  echo "plan: $mode, $n commit(s)"
  for r in ${others[@]+"${others[@]}"}; do echo "plan: not $r"; done
  exit 0
fi

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

# the pinned checker must read what this script asks of it: `new` asks --new-since, which ALICE-Law's checker has
# from a5824065 on. These bytes and a pin at that revision or later land in one commit
checker_help=$(python3 "$tmp/public_text_check.py" --help 2>/dev/null || true)
if [[ "$base" == new && "$checker_help" != *--new-since* ]]; then
  echo "$pin names ALICE-Law $rev, whose public_text_check.py has no --new-since (this script's \`new\`): move $pin" \
    "to ALICE-Law a5824065c35f161c07d5a158f21198fb198494ee or later, in the same commit as these bytes" >&2
  exit 2
fi

python3 "$tmp/public_text_check.py" --repo . --tree HEAD

if [[ "$n" -eq 0 ]]; then
  if [[ "$base" == new ]]; then
    echo "no new commit ($mode)"
  else
    echo "no new commit (base $base is HEAD)"
  fi
  exit 0
fi
set +e
if [[ "$base" == new ]]; then
  out=$(python3 "$tmp/public_text_check.py" --repo . --new-since ${others[@]+"${others[@]}"})
else
  out=$(python3 "$tmp/public_text_check.py" --repo . --range "$base..HEAD")
fi
rc=$?
set -e
printf '%s\n' "$out"
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
# the checker says how many commits it read: none, where there are new commits, is a range that read nothing
read_n=$(printf '%s\n' "$out" | sed -n 's/.* in \([0-9][0-9]*\) commits.*/\1/p' | tail -1)
if [[ -z "$read_n" || "$read_n" -eq 0 ]]; then
  echo "the checker read ${read_n:-no} commits of $n new ($mode)" >&2
  exit 2
fi
