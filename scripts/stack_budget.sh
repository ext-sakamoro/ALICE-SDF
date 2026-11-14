#!/usr/bin/env bash
# Stack budget of the tree walkers, in every build that ships
#
# The walkers (eval, the bytecode compiler, its validation and stack depths, the Lipschitz bound)
# keep their pending work on a heap-backed stack instead of recursing. For each build: run the
# stack budget tests (tests/test_eval_stack_budget.rs, tests/test_compile_stack_budget.rs: deep
# chains on a stack of 512 B per level, regression guards) and check in the same test binary that
# no walker calls itself (scripts/stack_frame_gate.py --no-recursion). Builds:
#   native debug and release
#   wasm32-wasip1 release under wasmtime, linked with the rust-lld default 1 MiB stack (the
#                   browser background thread's), canary on the module's stack; the engine's own
#                   stack (the native frames wasm functions run on) at max-wasm-stack = 1 MiB, and
#                   the smallest engine stack that passes is bisected and reported
#   headless Chromium, a background thread (scripts/browser_stack_canary.py): the deepest chain
#                   the browser engine runs, at least the wasm canary's depth
# Fails when a test fails, when a test binary ran 0 tests, when a walker calls itself or is not
# found, or when the browser stays below the depth.
#
# Usage: scripts/stack_budget.sh [--native-only]
#
# Author: Moroya Sakamoto
set -euo pipefail

cd "$(dirname "$0")/.."
WASM_STACK=1048576
TESTS=(--test test_eval_stack_budget --test test_compile_stack_budget)
# Each tree walker recurses natively for its first K levels (K = *_NATIVE_LEVELS in the source)
# and hands deeper subtrees to a work-stack loop: the recursive driver's frame times K must fit the
# stack budget of one walk, and neither the loops nor the validation may call themselves
WALK_BUDGET=65536
native_levels() {
  local v
  v=$(grep -E "^pub(\(crate\))? const $1: u32 = [0-9]+;" "$2" | grep -oE '[0-9]+;' | tr -d ';')
  [ -n "$v" ] && [ "$v" -gt 0 ] || { echo "stack-budget: FAIL $1 not found in $2" >&2; exit 1; }
  echo "$v"
}
EVAL_K=$(native_levels EVAL_NATIVE_LEVELS src/eval/mod.rs)
COMPILE_K=$(native_levels COMPILE_NATIVE_LEVELS src/compiled/compiler.rs)
DEPTHS_K=$(native_levels DEPTHS_NATIVE_LEVELS src/compiled/compiler.rs)
LIPSCHITZ_K=$(native_levels LIPSCHITZ_NATIVE_LEVELS src/interval.rs)
echo "stack-budget: native levels eval $EVAL_K, compile $COMPILE_K, depths $DEPTHS_K, lipschitz $LIPSCHITZ_K (walk budget $WALK_BUDGET B)"
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
  if [ "${1:-}" = --wasm ]; then
    # the wasm module: the canaries on its stacks and the engine floor are the checks there
    return 0
  fi
  local g=(python3 scripts/stack_frame_gate.py "$bin" --objdump "$OBJDUMP")
  case $name in
    test_eval_stack_budget)
      "${g[@]}" --budget $((WALK_BUDGET / EVAL_K)) --function alice_sdf::eval::eval_rec
      "${g[@]}" --no-recursion --function alice_sdf::eval::eval_iter
      ;;
    test_compile_stack_budget)
      "${g[@]}" --budget $((WALK_BUDGET / COMPILE_K)) --function alice_sdf::compiled::compiler::Compiler::compile_rec
      "${g[@]}" --budget $((WALK_BUDGET / DEPTHS_K)) --function alice_sdf::compiled::compiler::depths_rec
      "${g[@]}" --budget $((WALK_BUDGET / LIPSCHITZ_K)) --function alice_sdf::interval::lipschitz_rec
      "${g[@]}" --no-recursion \
        --function alice_sdf::compiled::compiler::Compiler::compile_iter \
        --function alice_sdf::compiled::compiler::depths_iter \
        --function alice_sdf::interval::lipschitz_iter \
        --function alice_sdf::compiled::compiler::validate_for_compile
      ;;
    *) echo "stack-budget: FAIL unexpected test binary $name"; return 1 ;;
  esac
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
  python3 scripts/browser_stack_canary.py
fi
echo "stack-budget: OK"
