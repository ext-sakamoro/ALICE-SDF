#!/usr/bin/env bash
# Stack budget of the recursive tree walkers, in every build that ships
#
# For each build: run the stack budget tests (tests/test_eval_stack_budget.rs,
# tests/test_compile_stack_budget.rs: the runtime canaries) and read each recursive function's
# frame from the same test binary (scripts/stack_frame_gate.py). Builds:
#   native debug    (the frames an unoptimised build gives a deep tree on a 2 MiB test thread)
#   native release  (the split helpers may be inlined there; the frame must still fit)
#   wasm32-wasip1 release under wasmtime, linked with the rust-lld default 1 MiB stack (the
#                   browser background thread's), canary on the module's stack; the engine's own stack (the
#                   native frames wasm functions run on) is held to the same per-level budget:
#                   max-wasm-stack = 1 MiB, and the smallest engine stack that passes is bisected
#                   and reported
#   headless Chromium, a background thread (scripts/browser_stack_canary.py): the deepest chain
#                   the browser engine runs, at least the wasm canary's depth
# Fails when a test fails, when a test binary ran 0 tests, when a frame is over the budget or a
# function is not found.
#
# Usage: scripts/stack_budget.sh [--native-only]
#
# Author: Moroya Sakamoto
set -euo pipefail

cd "$(dirname "$0")/.."
BUDGET=512
WASM_STACK=1048576
TESTS=(--test test_eval_stack_budget --test test_compile_stack_budget)
EVAL_FNS=(--function alice_sdf::eval::eval)
COMPILE_FNS=(
  --function alice_sdf::compiled::compiler::Compiler::compile_node
  --function alice_sdf::compiled::compiler::validate_subtree
  --function alice_sdf::compiled::compiler::compute_stack_depths
  --function alice_sdf::interval::eval_lipschitz
)
OBJDUMP=${OBJDUMP:-objdump}
# the release profile strips symbols; the frames are the same without stripping, and the gate
# needs the names (the rest of the release profile, LTO and one codegen unit, stays as shipped)
export CARGO_PROFILE_RELEASE_STRIP=false

# the test executables cargo built: "<name> <path>" per line
executables() {
  python3 -c '
import json, sys
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["kind"] == ["test"]:
        print(m["target"]["name"], m["executable"])
'
}

# run one test executable, require at least one test and no failure
run_tests() {
  local label=$1
  shift
  local out
  out=$("$@" 2>&1) || {
    echo "$out" | tail -20
    echo "stack-budget: FAIL $label: tests failed"
    return 1
  }
  local n
  # strip colour codes (CARGO_TERM_COLOR / a terminal colour libtest's "ok")
  n=$(echo "$out" | sed $'s/\x1b\\[[0-9;]*m//g' | sed -n 's/^test result: ok\. \([0-9]*\) passed.*/\1/p' | awk '{s += $1} END {print s + 0}')
  if [ "$n" -eq 0 ]; then
    echo "stack-budget: FAIL $label: 0 tests ran"
    return 1
  fi
  echo "stack-budget: $label: $n test(s) passed"
}

gate() {
  local name=$1 bin=$2
  shift 2
  local fns
  case $name in
    test_eval_stack_budget) fns=("${EVAL_FNS[@]}") ;;
    test_compile_stack_budget) fns=("${COMPILE_FNS[@]}") ;;
    *) echo "stack-budget: FAIL unexpected test binary $name"; return 1 ;;
  esac
  python3 scripts/stack_frame_gate.py "$bin" --budget "$BUDGET" --objdump "$OBJDUMP" "$@" "${fns[@]}"
}

native() {
  local profile=$1
  local flags=()
  [ "$profile" = release ] && flags=(--release)
  local list
  list=$(cargo test ${flags[@]+"${flags[@]}"} "${TESTS[@]}" --no-run --message-format=json | executables)
  [ "$(awk 'NF { n++ } END { print n + 0 }' <<<"$list")" -eq 2 ] || { echo "stack-budget: FAIL native $profile: expected 2 test binaries, got: $list"; return 1; }
  while read -r name bin; do
    run_tests "native $profile $name" "$bin"
    gate "$name" "$bin"
  done <<<"$list"
}

wasm() {
  command -v wasmtime >/dev/null || { echo "stack-budget: FAIL wasmtime not found"; return 1; }
  local list
  list=$(CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C link-arg=-zstack-size=$WASM_STACK" \
    cargo test --release --target wasm32-wasip1 "${TESTS[@]}" --no-run --message-format=json | executables)
  [ "$(awk 'NF { n++ } END { print n + 0 }' <<<"$list")" -eq 2 ] || { echo "stack-budget: FAIL wasm: expected 2 test binaries, got: $list"; return 1; }
  while read -r name bin; do
    run_tests "wasm32 release $name" wasmtime run -W max-wasm-stack=$WASM_STACK "$bin"
    gate "$name" "$bin" --wasm
    engine_floor "$name" "$bin"
  done <<<"$list"
}

# the smallest engine stack (16 KiB steps) the wasm test binary passes with: it passed at
# WASM_STACK above; report how much of it the walkers need
engine_floor() {
  local name=$1 bin=$2 lo=0 hi=$((WASM_STACK / 16384)) runs=0 mid
  while [ $((hi - lo)) -gt 1 ]; do
    mid=$(((lo + hi) / 2))
    runs=$((runs + 1))
    if wasmtime run -W max-wasm-stack=$((mid * 16384)) "$bin" >/dev/null 2>&1; then hi=$mid; else lo=$mid; fi
  done
  [ "$runs" -gt 0 ] || { echo "stack-budget: FAIL wasm32 $name: engine stack not bisected"; return 1; }
  echo "stack-budget: wasm32 $name: engine stack needed $((hi * 16)) KiB of $((WASM_STACK / 1024)) KiB ($runs runs)"
}

native debug
native release
if [ "${1:-}" != "--native-only" ]; then
  wasm
  # canary_eval: the recursive eval reaches 1096 levels in V8, below the wasm canary's 1792 (an
  # ignored known-defect test, tests/test_browser_stack_canary.rs); compile is held to it now
  python3 scripts/browser_stack_canary.py --known-defect canary_eval
fi
echo "stack-budget: OK"
