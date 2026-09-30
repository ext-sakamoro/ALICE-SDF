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
#   --quick  runs `cargo test --lib` but skips every other test step (the
#            integration suites, the feature-gated oracles — svo / texture-fit /
#            jit parity / MSL —, ffi + shaders, doctests, bridges and aaa), the
#            release wasm build and cargo audit (network); everything that
#            judges the *source* still runs, including semver-checks, deny,
#            machete and the stub guard.
#            2026-09-30: before this date `--quick` exited *before* every test
#            step, so the pre-push gate ran no `cargo test` at all. A green
#            `--quick` still does not mean the integration suites or the
#            oracles passed — drop `--quick` for that.
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
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing
# (2026-09-16: four clippy errors reached CI through a green preflight after a
# manual `cargo check`). Touching the crate root invalidates only this
# crate's fingerprint, so every clippy step below re-lints it.
relint() { touch src/lib.rs; }
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
relint; RUSTFLAGS="-Dwarnings" cargo clippy --lib --no-default-features

step "clippy: strict (default, all targets)"
relint; RUSTFLAGS="-Dwarnings" cargo clippy --all-targets

step "clippy: strict (all features that build on Linux, all targets)"
relint; RUSTFLAGS="-Dwarnings" cargo clippy --all-targets --features "$LINUX_ALL"

# Linux / Windows runners are x86_64; arch-gated bodies (SIMD dispatch) lint
# differently there (missing_const_for_fn, dead_code on aarch64-only paths).
step "clippy: strict on x86_64 (Linux / Windows runner arch, default + gpu)"
rustup target list --installed | grep -q x86_64-apple-darwin || rustup target add x86_64-apple-darwin
relint; RUSTFLAGS="-Dwarnings" cargo clippy --all-targets --target x86_64-apple-darwin --features "glsl,hlsl,gpu,jit,ffi"

step "clippy: feature-gated examples build (gpu / glsl / hlsl)"
RUSTFLAGS="-Dwarnings" cargo build --examples --features "glsl,hlsl,gpu"

step "clippy-strict: mobile/uniffi-wrapper (path dep re-lint)"
relint; (cd mobile/uniffi-wrapper && RUSTFLAGS="-Dwarnings" cargo clippy --lib --all-targets)

step "msrv: cargo +${MSRV} check --lib (default)"
cargo "+${MSRV}" check --lib

step "msrv: cargo +${MSRV} check --lib (docs.rs feature set)"
cargo "+${MSRV}" check --lib --features "$DOCSRS"

# ffi は 2026-09-30 に ci.yml 側が build -> test に格上げされたのでここからは外し、
# 下の test 群 ("test: ffi + shaders") に移した (ci.yml と逐語対応を保つ)。
step "test job builds: no default / jit / unity / unreal"
cargo build --lib --no-default-features
cargo build --lib --no-default-features --features "jit"
cargo build --lib --no-default-features --features "unity"
cargo build --lib --no-default-features --features "unreal"

step "wasm: build (wasm feature, wasm32 target)"
rustup target list --installed | grep -q wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
cargo build --lib --no-default-features --features wasm --target wasm32-unknown-unknown
cargo build --lib --no-default-features --features wasm

step "vrchat-host: 7 sample C# colliders vs alice_sdf goldens (dotnet)"
scripts/vrchat-host-parity.sh

# The engine itself (BuildPlugin + automation tests) only runs on the
# self-hosted ue5 runner (scripts/unreal-ue5-ci.ps1); this is the part of
# the Unreal contract that needs no Unreal.
step "unreal-abi: plugin header == include/alice_sdf.h, prototypes ⇔ cdylib exports, uplugin, shader includes, generated corpus"
scripts/unreal-abi-check.sh

step "doc: cargo doc --lib --no-deps (RUSTDOCFLAGS=-Dwarnings), default + docs.rs set + texture-fit"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$DOCSRS,texture-fit"

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

step "stub-guard: platform libm / mul_add in the evaluator and law directories (security-audit.yml, 3.1.0)"
python3 scripts/det_math_guard.py

step "stub-guard: raw Interval { lo, hi } bypassing the outward rounding (security-audit.yml, 4.0.1)"
python3 scripts/interval_outward_guard.py

step "stub-guard: tests/ の extern 宣言 ⇔ src/ffi の export (順序付き型列)"
# unreal-abi-check.sh step 2a は header / C# ⇔ export を見るので、tests/ の
# extern "C" 宣言は第 3 の宣言箇所として無検査だった。引数順の誤りは SysV
# (macOS / Linux) では整数と浮動小数でレジスタバンクが分かれ独立に採番される
# ため値が正しいレジスタに着いて通り、Microsoft x64 は位置でスロットを決める
# ので落ちる (2026-09-30 run 36680136653 が Windows だけ red)。
python3 scripts/abi_decl_check.py

# ── fuzz.yml (build only; the replay needs the nightly fuzz build) ────────

step "fuzz: cargo +nightly fuzz build (all targets)"
if cargo +nightly fuzz --version >/dev/null 2>&1; then
  (cd fuzz && cargo +nightly fuzz build)
else
  echo "skip: cargo +nightly fuzz not installed (cargo +nightly install cargo-fuzz)" >&2
fi

# --quick でも最低限の test は走らせる (2026-09-30)。それまでの --quick は
# test 群の手前で exit していたため、pre-push hook が走らせる gate が
# `cargo test` を 1 本も実行していなかった。default feature の --lib だけなら
# 数十秒で済み、「push 前に何も test していない」状態を避けられる。
if [[ $quick -eq 1 ]]; then
  step "test: cargo test --lib (--quick でも走らせる最小限)"
  cargo test --lib
  echo
  echo "preflight --quick OK"
  echo "  RAN     : fmt / clippy / MSRV / feature builds / wasm target / fuzz build / cargo test --lib"
  echo "  NOT RUN : cargo test --tests (integration), feature-gated oracles (svo / texture-fit /"
  echo "            jit parity / MSL), ffi + shaders, doctests, bridges, aaa, release wasm, cargo audit"
  echo "  => この green は「integration / oracle が通った」ことを意味しない。"
  echo "     それらを確認するには --quick を外して実行する。"
  exit 0
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

# JIT の parity arm は `#[cfg(feature = "jit")]` なので default の
# `cargo test --tests` では compile されない (2026-09-30 実測、CI でも一度も
# 走っていなかった)。test 本数では退行が見えない (relaxed_tracing だけ 8 -> 9、
# det_parity / evaluator_opcode_parity は本数不変で中の比較 arm だけ消える)。
step "test: JIT parity oracles (tree evaluator vs JitCompiledSdf / JitSimdSdf)"
cargo test --features jit \
  --test test_det_parity \
  --test test_evaluator_opcode_parity \
  --test test_relaxed_tracing

step "test: ffi + shaders (src/ffi の unit test 15 本、ci.yml と対)"
cargo test --lib --features "ffi,hlsl,glsl"

# MSL は WGSL emit を naga で翻訳したもの (compiled::msl)。翻訳と binding map の
# 誤りはこの oracle でしか出ない。Metal runtime compiler を使うので Xcode の
# Metal Toolchain component は不要。Linux / Windows には Metal が無いので skip。
if [ "$(uname -s)" = "Darwin" ]; then
  step "test: MSL -> Metal oracle (corpus compile + CPU parity, macOS のみ)"
  ALICE_SDF_REQUIRE_METAL=1 cargo test --features msl --test test_msl_metal_oracle -- --nocapture
else
  step "skip: MSL -> Metal oracle (Metal は macOS のみ)"
fi

step "test: cargo test --doc"
cargo test --doc

step "test: bridges (lib, no default)"
cargo test --lib --no-default-features --features "$BRIDGES"
relint; RUSTFLAGS="-Dwarnings" cargo clippy --lib --features "$BRIDGES"

step "test: AAA meta"
cargo test --lib --no-default-features --features "aaa"

step "test: openvdb"
cargo build --lib --no-default-features --features openvdb
cargo test --lib --no-default-features --features openvdb vdb

step "gpu-parity: GPU <-> CPU law parity, shader validation, GPU marching cubes (Metal here, lavapipe in CI)"
ALICE_SDF_REQUIRE_GPU=1 cargo test --features "gpu,glsl,gpu-mesh,texture-fit" \
  --test test_gpu_law_parity --test test_gpu_noise_parity --test test_round_tie_parity \
  --test test_transpiler_naga_validate --test noise_shader_validate --test test_mesh_orientation \
  --test test_texture_shader_gpu_parity

step "bevy: bindings/bevy/alice-sdf-bevy build + test"
(cd bindings/bevy/alice-sdf-bevy && cargo build --lib && cargo test --lib)

step "wasm-build: release wasm artifact"
cargo build --release --target wasm32-unknown-unknown --features wasm --no-default-features
test -f target/wasm32-unknown-unknown/release/alice_sdf.wasm

step "python-smoke: maturin develop --features python + python/tests/smoke.py"
if command -v maturin >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  venv=$(mktemp -d)/venv
  python3 -m venv "$venv"
  # shellcheck disable=SC1091
  . "$venv/bin/activate"
  pip install --quiet maturin numpy
  maturin develop --features python
  python python/tests/smoke.py
  deactivate
else
  echo "skip: maturin / python3 not installed" >&2
fi

step "audit: cargo audit (RustSec)"
if command -v cargo-audit >/dev/null 2>&1; then
  cargo audit --deny yanked --ignore RUSTSEC-2025-0141 --ignore RUSTSEC-2024-0436
else
  echo "skip: cargo-audit not installed" >&2
fi

echo; echo "preflight OK"
