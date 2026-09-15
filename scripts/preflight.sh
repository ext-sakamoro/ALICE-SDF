#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push`: every hard gate of
# ci.yml, security-audit.yml and fuzz.yml (build only), with the commands CI
# runs. A step this script does not cover is a step that can only fail
# remotely — when a step is added to a workflow, add it here in the same
# commit.
#
# 2026-09-16: the semver-checks job went red on a removed optional dependency
# (`lazy_static` was an implicit public feature) after a push that had only
# been checked with cargo test + clippy locally. This file is the checklist
# so that cannot repeat; ~/.claude/hooks/pre-push-preflight.sh runs
# `--quick` before every push and blocks on failure.
#
# usage: scripts/preflight.sh [--quick]
#   --quick  skips the test suites, the release wasm build and cargo audit
#            (network); everything that judges the *source* still runs,
#            including semver-checks, deny, machete and the stub guard.
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

# Feature sets, verbatim from the workflows.
LINUX_ALL='glsl,hlsl,gpu,jit,svo,terrain,destruction,gi,ffi,volume,gpu-mesh,svo-gpu,openvdb,physics,codec,asp,sdf-cache,texture-fit'
DOCSRS='glsl,hlsl,jit,svo,terrain,destruction,gi,ffi'
BRIDGES='physics,codec,asp,sdf-cache'
MSRV=1.85

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }

need actionlint "brew install actionlint"
need cargo-semver-checks "cargo install cargo-semver-checks --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-machete "cargo install cargo-machete --locked"
rustup toolchain list | grep -q "^${MSRV}" || { echo "missing toolchain ${MSRV} (rustup toolchain install ${MSRV})" >&2; exit 1; }

# ── ci.yml ────────────────────────────────────────────────────────────────

step "actionlint (workflow YAML)"
actionlint .github/workflows/*.yml

step "fmt: cargo fmt --check (core)"
cargo fmt --check

step "fmt: cargo fmt --check (mobile/uniffi-wrapper)"
(cd mobile/uniffi-wrapper && cargo fmt --check)

step "clippy: strict (no-default-features)"
RUSTFLAGS="-Dwarnings" cargo clippy --lib --no-default-features

step "clippy: strict (default, all targets)"
RUSTFLAGS="-Dwarnings" cargo clippy --all-targets

step "clippy: strict (all features that build on Linux, all targets)"
RUSTFLAGS="-Dwarnings" cargo clippy --all-targets --features "$LINUX_ALL"

# Linux / Windows runners are x86_64; arch-gated bodies (SIMD dispatch) lint
# differently there (missing_const_for_fn, dead_code on aarch64-only paths).
step "clippy: strict on x86_64 (Linux / Windows runner arch, default + gpu)"
rustup target list --installed | grep -q x86_64-apple-darwin || rustup target add x86_64-apple-darwin
RUSTFLAGS="-Dwarnings" cargo clippy --all-targets --target x86_64-apple-darwin --features "glsl,hlsl,gpu,jit,ffi"

step "clippy: feature-gated examples build (gpu / glsl / hlsl)"
RUSTFLAGS="-Dwarnings" cargo build --examples --features "glsl,hlsl,gpu"

step "clippy-strict: mobile/uniffi-wrapper (path dep re-lint)"
(cd mobile/uniffi-wrapper && RUSTFLAGS="-Dwarnings" cargo clippy --lib --all-targets)

step "msrv: cargo +${MSRV} check --lib (default)"
cargo "+${MSRV}" check --lib

step "msrv: cargo +${MSRV} check --lib (docs.rs feature set)"
cargo "+${MSRV}" check --lib --features "$DOCSRS"

step "test job builds: no default / jit / unity / unreal / ffi+shaders"
cargo build --lib --no-default-features
cargo build --lib --no-default-features --features "jit"
cargo build --lib --no-default-features --features "unity"
cargo build --lib --no-default-features --features "unreal"
cargo build --features "ffi,hlsl,glsl"

step "wasm: build (wasm feature, wasm32 target)"
rustup target list --installed | grep -q wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
cargo build --lib --no-default-features --features wasm --target wasm32-unknown-unknown
cargo build --lib --no-default-features --features wasm

step "doc: cargo doc --lib --no-deps (RUSTDOCFLAGS=-Dwarnings)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps

step "bench: cargo bench --no-run"
cargo bench --no-run

# ── security-audit.yml ────────────────────────────────────────────────────

step "semver-checks: check-release vs crates.io (CI feature set)"
feats=$(cargo metadata --format-version 1 --no-deps \
  | jq -r '.packages[] | select(.name=="alice-sdf") | .features | keys[]' \
  | grep -vE '^(default|font|godot|python)$' | paste -sd, -)
cargo semver-checks check-release --package alice-sdf \
  --only-explicit-features --features "$feats"

step "deny: cargo deny --all-features check all"
cargo deny --all-features check all

step "machete: unused dependencies"
cargo machete

step "stub-guard: todo! / unimplemented! / panic!(STUB) / dbg!() in src/"
hits=$(grep -rnE 'todo!\(|unimplemented!\(|panic!\([^)]*STUB' src/ --include="*.rs" --exclude-dir=bin || true)
if [ -n "$hits" ]; then echo "$hits"; echo "stub in src/ (production path)" >&2; exit 1; fi
hits=$(grep -rn 'dbg!(' src/ --include="*.rs" || true)
if [ -n "$hits" ]; then echo "$hits"; echo "dbg!() in src/" >&2; exit 1; fi

# ── fuzz.yml (build only; the replay needs the nightly fuzz build) ────────

step "fuzz: cargo +nightly fuzz build (all targets)"
if cargo +nightly fuzz --version >/dev/null 2>&1; then
  (cd fuzz && cargo +nightly fuzz build)
else
  echo "skip: cargo +nightly fuzz not installed (cargo +nightly install cargo-fuzz)" >&2
fi

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test suites, release wasm, cargo audit skipped)"; exit 0
fi

# ── ci.yml test matrix (host = macOS ARM64 lane) ──────────────────────────

step "test: cargo test --lib"
cargo test --lib

step "test: cargo test --lib --features glsl,hlsl,gpu,texture-fit (shader transpilers + texture-fit)"
cargo test --lib --features "glsl,hlsl,gpu,texture-fit"

step "test: cargo test --tests (integration)"
cargo test --tests

step "test: feature-gated oracles (svo / texture-fit)"
cargo test --features "svo,texture-fit" --test test_svo_query_oracle --test test_texture_fit_oracle

step "test: cargo test --doc"
cargo test --doc

step "test: bridges (lib, no default)"
cargo test --lib --no-default-features --features "$BRIDGES"
RUSTFLAGS="-Dwarnings" cargo clippy --lib --features "$BRIDGES"

step "test: AAA meta"
cargo test --lib --no-default-features --features "aaa"

step "test: openvdb"
cargo build --lib --no-default-features --features openvdb
cargo test --lib --no-default-features --features openvdb vdb

step "gpu-parity: GPU <-> CPU law parity, shader validation, GPU marching cubes (Metal here, lavapipe in CI)"
ALICE_SDF_REQUIRE_GPU=1 cargo test --features "gpu,glsl,gpu-mesh" \
  --test test_gpu_law_parity --test test_gpu_noise_parity --test test_round_tie_parity \
  --test test_transpiler_naga_validate --test noise_shader_validate --test test_mesh_orientation

step "bevy: bindings/bevy/alice-sdf-bevy build + test"
(cd bindings/bevy/alice-sdf-bevy && cargo build --lib && cargo test --lib)

step "wasm-build: release wasm artifact"
cargo build --release --target wasm32-unknown-unknown --features wasm --no-default-features
test -f target/wasm32-unknown-unknown/release/alice_sdf.wasm

step "audit: cargo audit (RustSec)"
if command -v cargo-audit >/dev/null 2>&1; then
  cargo audit --deny yanked --ignore RUSTSEC-2025-0141 --ignore RUSTSEC-2024-0436
else
  echo "skip: cargo-audit not installed" >&2
fi

echo; echo "preflight OK"
