#!/usr/bin/env bash
# check_min_tests.sh — Enforce minimum test count per ALICE crate
# Usage: ./check_min_tests.sh [MIN_TESTS] [CRATE_DIR...]
# Default MIN_TESTS=50
# If no CRATE_DIRs given, scans ../ALICE-* directories

set -euo pipefail

MIN_TESTS="${1:-50}"
shift 2>/dev/null || true

# If no args, find all ALICE-* siblings
if [ $# -eq 0 ]; then
  CRATES=()
  for d in ../ALICE-*/; do
    if [ -f "$d/Cargo.toml" ]; then
      CRATES+=("$d")
    fi
  done
else
  CRATES=("$@")
fi

PASS=0
FAIL=0
SKIP=0
TOTAL_TESTS=0

printf "%-40s %8s %s\n" "Crate" "Tests" "Status"
printf "%-40s %8s %s\n" "-----" "-----" "------"

for crate_dir in "${CRATES[@]}"; do
  name=$(basename "$crate_dir")

  # Skip non-Rust crates
  if [ ! -f "$crate_dir/Cargo.toml" ]; then
    printf "%-40s %8s %s\n" "$name" "-" "SKIP (no Cargo.toml)"
    SKIP=$((SKIP + 1))
    continue
  fi

  # Count tests by listing them (much faster than running)
  # grep -c returns exit 1 when count=0; use `|| true` to absorb that under set -e
  count=$(cd "$crate_dir" && cargo test --lib -- --list 2>/dev/null | { grep -c ': test$' || true; })
  TOTAL_TESTS=$((TOTAL_TESTS + count))

  if [ "$count" -ge "$MIN_TESTS" ]; then
    printf "%-40s %8d %s\n" "$name" "$count" "PASS"
    PASS=$((PASS + 1))
  else
    printf "%-40s %8d %s\n" "$name" "$count" "FAIL (< $MIN_TESTS)"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "Summary: $PASS passed, $FAIL failed, $SKIP skipped, $TOTAL_TESTS total tests"

if [ "$FAIL" -gt 0 ]; then
  echo "ERROR: $FAIL crate(s) below minimum test threshold ($MIN_TESTS)"
  exit 1
fi

echo "All crates meet minimum test requirement."
exit 0
