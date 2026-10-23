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
# so that cannot repeat; run it (at least `--quick`) before every push.
#
# Not covered here, deliberately: `scripts/downstream_check.py` answers "which
# sibling repo does publishing this crate break, and whose `cargo publish` does
# it unblock". It reads the sibling checkouts under `~`, which do not exist on a
# CI runner and cannot be reconstructed from this repo, so it is a pre-publish
# step run by hand rather than a part of this script or of any workflow. Run it
# before `cargo publish`, not before every `git push`.
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

# The ci.yml scip job, same commands: the oracle, the indexes, no new L0 / stale baseline entry,
# and docs/integration-status.md regenerated with no diff. $1 = 1: rust-analyzer is required
# (the --quick run below calls this because src/ changed, so skipping is not an option there)
scip_reach_step() {
  python3 scripts/test_scip_reach.py
  python3 scripts/test_scip_scope.py
  if ! command -v rust-analyzer >/dev/null 2>&1; then
    if [[ "$1" -eq 1 ]]; then
      echo "rust-analyzer が無い (rustup component add rust-analyzer): src/ を変えた変更では scip reach を飛ばせない" >&2
      exit 1
    fi
    echo "skip: rust-analyzer が無い (rustup component add rust-analyzer)、CI の scip job が検査する" >&2
    return 0
  fi
  scripts/scip_index.sh
  python3 scripts/scip_reach.py --check-baseline
  python3 scripts/test_api_boundary_parsers.py
  python3 scripts/api_boundary_parsers.py
  python3 scripts/provenance_writers.py --index target/scip
  python3 scripts/scip_reach.py --write docs/integration-status.md
  git diff --exit-code docs/integration-status.md \
    || { echo "docs/integration-status.md が古い: 再生成した内容を commit する" >&2; exit 1; }
}

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

# Feature sets, verbatim from the workflows.
LINUX_ALL='glsl,hlsl,blinkscript,gpu,jit,svo,terrain,destruction,gi,ffi,volume,gpu-mesh,svo-gpu,openvdb,physics,codec,asp,sdf-cache,texture-fit,rust'
# the clippy job's LINT_FEATURES: every feature but wasm (linted on wasm32 below)
LINT_FEATURES='aaa,all-shaders,asp,blinkscript,cli,codec,destruction,ffi,font,gi,glsl,godot,gpu,gpu-mesh,hlsl,image,jit,msl,openvdb,physics,python,rust,sdf-cache,svo,svo-gpu,terrain,texture-fit,unity,unreal,volume'
DOCSRS='glsl,hlsl,jit,svo,terrain,destruction,gi,ffi,rust'
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
need cargo-semver-checks "cargo install cargo-semver-checks --version 0.50.0 --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-machete "cargo install cargo-machete --locked"
rustup toolchain list | grep -q "^${MSRV}" || { echo "missing toolchain ${MSRV} (rustup toolchain install ${MSRV})" >&2; exit 1; }

# ── ci.yml ────────────────────────────────────────────────────────────────

step "actionlint (workflow YAML)"
actionlint .github/workflows/*.yml

step "workflow concurrency: oracle + branch への push で走る全 workflow が main の run を取り消さない"
step "eol.yml / line endings (tests, then check)"
# text files are LF in the index and the working tree (.github/workflows/eol.yml, shared by the ALICE crates)
python3 scripts/test_check_eol.py
python3 scripts/check_eol.py
python3 scripts/test_workflow_concurrency.py
python3 scripts/workflow_concurrency.py

step "wiring-guard: oracle + 新規の未配線 / 理由の無い dead_code が無い"
python3 scripts/test_wiring_guard.py
python3 scripts/wiring_guard.py

step "license copies: oracle + sub-package の license が root LICENSE-COMMUNITY と byte 一致 (比較 0 件で fail)"
python3 scripts/license_copy_check.py --self-test
python3 scripts/license_copy_check.py

step "license declarations: oracle + 全 manifest の license が Apache-2.0 (Community subtree は許可リスト、比較 0 件で fail)"
python3 scripts/license_check.py --self-test
python3 scripts/license_check.py

step "libm dependency guard: oracle + libm の直接依存には clippy.toml の libm:: の禁止と scripts/libm_ban_check.py が要る (manifest 0 件で fail)"
python3 scripts/test_libm_dependency_guard.py
python3 scripts/libm_dependency_guard.py

step "default-build dependency licences (G10): oracle + permissive only, nothing reached fails"
python3 scripts/test_default_dependency_licenses.py
python3 scripts/default_dependency_licenses.py


step "order and entropy guards (G13 / G10 text layer): oracle + every listed line pinned by count and hash"
python3 scripts/test_order_guard.py
python3 scripts/order_guard.py
python3 scripts/entropy_guard.py

step "build configuration guard (G14, static): oracle + no native / fast-math / x87 flag (nothing read fails)"
python3 scripts/test_build_config_guard.py
python3 scripts/build_config_guard.py
python3 scripts/test_build_flags_check.py
step "provenance writers: oracle + 公開の書き出し関数はすべて意味の識別子を書くか、未対応の一覧にある (件数固定、0 件で fail)"
python3 scripts/test_provenance_writers.py
python3 scripts/provenance_writers.py
step "bare deserialization reads: oracle + src の serde / bincode の読み込みは library clippy が禁止し、確認済みの箇所 (stored form checked / other-type / pending の理由つき allow) だけにある (0 件で fail)"
python3 scripts/test_bare_serde_reads.py
python3 scripts/bare_serde_reads.py

step "api boundary (ALICE-Law G7): oracle + 一覧の照合 (wasm32 target が無ければ PREFLIGHT-SKIP 行で明示、CI は fail)"
python3 scripts/test_api_boundary.py
python3 scripts/api_boundary.py --skip-missing-targets

step "ci-test-coverage: oracle + feature 付き oracle を走らせる cargo test が CI にある"
python3 scripts/test_ci_test_coverage_check.py
python3 scripts/ci_test_coverage_check.py

step "raw max/min manifest (test, then check against origin/main) + library clippy config"
python3 scripts/test_raw_minmax_manifest.py
python3 scripts/raw_minmax_manifest.py --base origin/main
python3 scripts/test_clippy_lib_gate.py
python3 scripts/clippy_lib_gate.py --check
python3 scripts/test_shader_fma_guard.py
python3 scripts/shader_fma_guard.py
python3 scripts/test_glam_feature_check.py
python3 scripts/glam_feature_check.py

step "status generators: oracle + 走査件数 0 で fail (docs/wiring-status.md / docs/oracle-status.md)"
python3 scripts/test_gen_status.py

step "law-readme-block.yml: README ALICE-Law block (tests, then the check the job runs, at the block's ALICE-Law pin)"
python3 scripts/test_law_readme_block_check.py
python3 scripts/law_readme_block_check.py
python3 scripts/gen-wiring-status.py
python3 scripts/gen-oracle-status.py

step "public-text.yml / public-text: Public text (ALICE-Law checker at the pin; tree, then the commits)"
python3 scripts/test_public_text.py
scripts/public_text.sh

step "docs: oracle + CHANGELOG の構造 / README と code の一致"
python3 scripts/test_docs_lint.py
python3 scripts/docs_lint.py --check
# ci.yml wiring-guard: release artifact audit oracle
python3 scripts/test_release_artifact_audit.py
python3 scripts/test_tracked_binaries.py
python3 scripts/tracked_binaries.py
python3 scripts/test_readme_sync.py
python3 scripts/readme_sync.py --check
python3 scripts/test_version_sync.py
python3 scripts/version_sync.py --check
python3 scripts/test_shared_files.py
scripts/shared_files.sh

# ci.yml の env と同じ CARGO_TERM_COLOR=always で回す (cargo の出力を読む parser の試験)
step "cargo output parsers: oracle (色付き出力) + test に色付き sample がある、対象 0 件で fail"
CARGO_TERM_COLOR=always python3 scripts/test_ansi.py
CARGO_TERM_COLOR=always python3 scripts/test_libtest_count.py
CARGO_TERM_COLOR=always python3 scripts/test_cargo_output_parser_check.py
CARGO_TERM_COLOR=always python3 scripts/cargo_output_parser_check.py

step "fmt: cargo fmt --check (core)"
cargo fmt --check

step "fmt: cargo fmt --check (mobile/uniffi-wrapper)"
(cd mobile/uniffi-wrapper && cargo fmt --check)

step "clippy feature coverage: every Cargo feature is on in a clippy run (G1)"
python3 scripts/test_clippy_feature_coverage.py
python3 scripts/clippy_feature_coverage.py

step "clippy: strict (no-default-features)"
relint; RUSTFLAGS="-Dwarnings" cargo clippy --lib --no-default-features

step "clippy: strict (default, all targets)"
relint; RUSTFLAGS="-Dwarnings" cargo clippy --all-targets

step "clippy: strict (every feature but wasm, all targets)"
relint; RUSTFLAGS="-Dwarnings" cargo clippy --all-targets --features "$LINT_FEATURES"

# library code: f32 / f64 / f32x8 max and min forbidden (ci.yml "Clippy (library
# code, ...)"), here also for the x86_64 target the Linux / Windows runners use
step "clippy: library code with .github/clippy-lib (max / min forbidden), host and x86_64"
relint; CLIPPY_CONF_DIR=.github/clippy-lib RUSTFLAGS="-Dwarnings" cargo clippy --lib --features "$LINT_FEATURES"
rustup target list --installed | grep -q x86_64-apple-darwin || rustup target add x86_64-apple-darwin
relint; CLIPPY_CONF_DIR=.github/clippy-lib RUSTFLAGS="-Dwarnings" cargo clippy --lib --target x86_64-apple-darwin --features "glsl,hlsl,gpu,jit,ffi"
step "clippy: library code on wasm32 (max / min forbidden)"
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
CLIPPY_CONF_DIR=.github/clippy-lib RUSTFLAGS="-Dwarnings" cargo clippy --lib --target wasm32-unknown-unknown --features wasm

step "library clippy configuration (current + canary)"
python3 scripts/clippy_lib_gate.py
step "no max / min hides under an allow of the lint (scratch copy with the allows lifted)"
python3 scripts/test_clippy_lib_hidden_check.py
python3 scripts/clippy_lib_hidden_check.py --features "$LINT_FEATURES"

# Linux / Windows runners are x86_64; arch-gated bodies (SIMD dispatch) lint
# differently there (missing_const_for_fn, dead_code on aarch64-only paths).
step "clippy: strict on x86_64 (Linux / Windows runner arch, default + gpu)"
rustup target list --installed | grep -q x86_64-apple-darwin || rustup target add x86_64-apple-darwin
relint; RUSTFLAGS="-Dwarnings" cargo clippy --all-targets --target x86_64-apple-darwin --features "glsl,hlsl,gpu,jit,ffi"

step "clippy: feature-gated examples build (gpu / glsl / hlsl / blinkscript / msl / rust)"
RUSTFLAGS="-Dwarnings" cargo build --examples --features "glsl,hlsl,blinkscript,gpu,msl,rust"

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

step "semver-checks.yml / semver-checks: semver gate (tests)"
python3 scripts/test_semver_gate.py

step "semver-checks.yml / semver-checks: semver gate (latest crates.io release, every consumer feature set)"
python3 scripts/semver_gate.py --config .github/semver-gate.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver"

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
python3 scripts/test_det_math_guard.py
python3 scripts/det_math_guard.py

step "stub-guard: raw Interval { lo, hi } bypassing the outward rounding (security-audit.yml, 4.0.1)"
python3 scripts/interval_outward_guard.py

step "stub-guard: tests/ の extern 宣言 ⇔ src/ffi の export (順序付き型列)"
# unreal-abi-check.sh step 2a は header / C# ⇔ export を見るので、tests/ の
# extern "C" 宣言は第 3 の宣言箇所として無検査だった。引数順の誤りは SysV
# (macOS / Linux) では整数と浮動小数でレジスタバンクが分かれ独立に採番される
# ため値が正しいレジスタに着いて通り、Microsoft x64 は位置でスロットを決める
# ので落ちる (2026-09-30 run 36680136653 が Windows だけ red)。戻り値の型も比べる。
python3 scripts/test_abi_decl_check.py
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
  # the scip job (reach ledger and L0 ratchet) whenever one of its inputs differs from origin/main;
  # scripts/scip_scope.sh decides and says why when it skips
  if scope=$(scripts/scip_scope.sh); then
    step "scip: $scope"
    scip_reach_step 1
  else
    echo "skip scip reach (${scope#skip: })"
  fi
  echo
  echo "preflight --quick OK"
  echo "  RAN     : fmt / clippy / MSRV / feature builds / wasm target / fuzz build / cargo test --lib"
  echo "            (+ scip reach when a scip input differs from origin/main)"
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

step "test: the examples' whole text carries the semantics identifier (ci.yml と対、6 件未満で fail)"
cargo test --features "msl,glsl,hlsl,blinkscript,rust" --test test_g8_example_text 2>&1 | tee "${TMPDIR:-/tmp}/sdf-g8-example-text.log"
python3 scripts/libtest_count.py "${TMPDIR:-/tmp}/sdf-g8-example-text.log" --min 6 --what "example text provenance tests"

step "test: min/max signed-zero Law parity (tree / SIMD / JIT / SVO)"
cargo test --features "jit,svo" --test test_minmax_signed_zero_probe

step "test: provenance of the Nanite HLSL material (ci.yml の test job と対、1 件未満で fail)"
cargo test --features hlsl --test test_provenance nanite_hlsl_material 2>&1 | tee "${CARGO_TARGET_DIR:-target}/provenance-hlsl.log"
python3 scripts/libtest_count.py "${CARGO_TARGET_DIR:-target}/provenance-hlsl.log" --min 1 --what "Nanite HLSL material provenance tests (hlsl)"

step "test: provenance of the SVO chunk (ci.yml の test job と対、3 件未満で fail)"
cargo test --features svo --test test_provenance svo_chunk 2>&1 | tee "${CARGO_TARGET_DIR:-target}/provenance-svo.log"
python3 scripts/libtest_count.py "${CARGO_TARGET_DIR:-target}/provenance-svo.log" --min 3 --what "SVO chunk provenance tests (svo)"

step "test: svo-gated lib tests of the unordered map (min_key_by, count >= 2)"
cargo test --lib --features svo unordered::tests:: 2>&1 | tee "${CARGO_TARGET_DIR:-target}/unordered-svo.log"
python3 scripts/libtest_count.py "${CARGO_TARGET_DIR:-target}/unordered-svo.log" --min 2 --what "unordered map lib tests (svo)"

# JIT の parity arm は `#[cfg(feature = "jit")]` なので default の
# `cargo test --tests` では compile されない (2026-09-30 実測、CI でも一度も
# 走っていなかった)。test 本数では退行が見えない (relaxed_tracing だけ 8 -> 9、
# det_parity / evaluator_opcode_parity は本数不変で中の比較 arm だけ消える)。
step "test: JIT parity oracles (tree evaluator vs JitCompiledSdf / JitSimdSdf)"
cargo test --features jit \
  --test test_det_parity \
  --test test_evaluator_opcode_parity \
  --test test_relaxed_tracing \
  --test test_raycast_oracle \
  --test test_round_tie_parity \
  --test test_jit_dynamic_oracle \
  --test test_law_nan_operands

# file 先頭が `#![cfg(feature = …)]` の oracle (default の --tests では 0 本、ci.yml と対)
step "test: physics bridge determinism oracle"
cargo test --features physics --test test_physics_bridge_determinism

step "test: live SDF oracle (physics、ci.yml と対)"
cargo test --features physics --test test_live_sdf_oracle --test test_live_sdf_closed_loop

step "det-math is one version with physics, and the JIT matches it"
n=$(cargo tree --features "jit,physics" -i alice-det-math -e normal --prefix none | grep -cE '^alice-det-math v')
[ "$n" -eq 1 ] || { echo "alice-det-math versions with jit,physics: $n"; cargo tree --features "jit,physics" -i alice-det-math -e normal; exit 1; }
cargo test --features "jit,physics" --test test_det_parity --test test_jit_sincos_det --test test_det_near_zero_parity

step "test: NPR shader validation (glsl + gpu、naga のみで GPU adapter 不要)"
cargo test --features "glsl,gpu" --test npr_shader_validate

step "test: Rust source emit oracle (rustc で compile した出力 vs eval_compiled、ci.yml と対)"
cargo test --features rust --test test_rust_transpiler_oracle

step "test: SdfNode × backend の対応表 (docs/node-support.md と突合、jit + msl + glsl + hlsl + rust)"
cargo test --features jit,msl,rust,glsl,hlsl --test test_node_backend_matrix

step "test: shader text has no fused multiply-add (msl + glsl + hlsl + blinkscript、ci.yml と対)"
cargo test --features msl,glsl,hlsl,blinkscript --test test_shader_no_fused_multiply_add

step "test: HLSL / BlinkScript value parity + exports + round-tie hlsl arm (ci.yml の HLSL step と対)"
ALICE_SDF_REQUIRE_CXX=1 cargo test --features "hlsl,blinkscript" --test test_hlsl_blinkscript_parity \
  --test test_hlsl_export_oracle --test test_round_tie_parity --test test_shader_precise_temporaries

# HLSL の全出力を DirectX Shader Compiler で compile する gate (hlsl-dxc.yml と対)
# macOS 向けの公式 dxc は無いので、dxc が無い host では skip を明示して出す
# (test 自体も skip で green になるので、ここで出さないと空振りが見えない)
if command -v "${ALICE_SDF_DXC:-dxc}" >/dev/null 2>&1; then
  step "test: HLSL outputs compiled by dxc (hlsl-dxc.yml と対)"
  ALICE_SDF_REQUIRE_DXC=1 cargo test --features hlsl --test test_hlsl_dxc_compile -- --nocapture
else
  step "SKIPPED: HLSL dxc compile gate (dxc が無い、ALICE_SDF_DXC で指定可) — 0 件 compile、CI の hlsl-dxc.yml が検査する"
fi
# GLSL の全出力を glslang で SPIR-V にし、ALICE_PRECISE が NoContraction になることを確かめる (hlsl-dxc.yml の glslang の step と対)
if command -v "${ALICE_SDF_GLSLANG:-glslang}" >/dev/null 2>&1; then
  step "test: GLSL outputs compiled by glslang, NoContraction (hlsl-dxc.yml と対)"
  ALICE_SDF_REQUIRE_GLSLANG=1 cargo test --features glsl --test test_glsl_glslang_precise -- --nocapture
else
  step "SKIPPED: GLSL glslang gate (glslang が無い、ALICE_SDF_GLSLANG で指定可) — 0 件 compile、CI の hlsl-dxc.yml が検査する"
fi

step "test: ffi + shaders (src/ffi の unit test 15 本、ci.yml と対)"
cargo test --lib --features "ffi,hlsl,glsl"

# MSL は WGSL emit を naga で翻訳したもの (compiled::msl)。翻訳と binding map の
# 誤りはこの oracle でしか出ない。Metal runtime compiler を使うので Xcode の
# Metal Toolchain component は不要。Linux / Windows には Metal が無いので skip。
if [ "$(uname -s)" = "Darwin" ]; then
  step "test: MSL -> Metal oracle (corpus compile + CPU parity, macOS のみ)"
  ALICE_SDF_REQUIRE_METAL=1 cargo test --features msl --test test_msl_metal_oracle -- --nocapture
  step "test: shader text semantics identifier (msl、ci.yml の macOS step と対、9 件未満で fail)"
  cargo test --features msl --test test_shader_text_provenance -- --nocapture 2>&1 | tee "${TMPDIR:-/tmp}/sdf-shader-text-provenance-msl.log"
  python3 scripts/libtest_count.py "${TMPDIR:-/tmp}/sdf-shader-text-provenance-msl.log" --min 13 --what "shader text provenance tests (msl)"
  step "test: GpuEvaluator on Metal does not fuse a multiply-add (msl、ci.yml の macOS step と対、走った件数 0 で fail)"
  ALICE_SDF_REQUIRE_METAL=1 cargo test --features msl --test test_gpu_fma_contraction -- --nocapture 2>&1 | tee "${TMPDIR:-/tmp}/sdf-gpu-fma.log"
  python3 scripts/libtest_count.py "${TMPDIR:-/tmp}/sdf-gpu-fma.log" --min 1 --what "GpuEvaluator fma contraction tests (msl, Metal)"
else
  step "skip: MSL -> Metal oracle (Metal は macOS のみ)"
fi

step "test: cargo test --doc"
cargo test --doc

step "test: doctests, GpuEvalFuture lifetime (compile_fail, gpu feature)"
cargo test --doc --features gpu GpuEvalFuture 2>&1 | tee /tmp/alice-sdf-doc-gpu.$$.log
doc_rc=0
python3 scripts/libtest_count.py /tmp/alice-sdf-doc-gpu.$$.log --min 2 --what "GpuEvalFuture doctests" || doc_rc=$?
rm -f /tmp/alice-sdf-doc-gpu.$$.log
[ "$doc_rc" -eq 0 ] || exit "$doc_rc"

step "test: bridges (lib, no default)"
cargo test --lib --no-default-features --features "$BRIDGES"
relint; RUSTFLAGS="-Dwarnings" cargo clippy --lib --features "$BRIDGES"

step "test: bridge oracles (ci.yml の bridges job の bridge oracles と対)"
cargo test --features "$BRIDGES,gpu" --test test_codec_bridge_oracle --test test_asp_bridge_oracle --test test_g8_tree_text \
  --test test_sdf_eval_cache_oracle --test test_sim_bridge_oracle

step "test: AAA meta"
cargo test --lib --no-default-features --features "aaa"

step "test: aaa integration oracles (ci.yml の Test (integration, analytic oracles — aaa) と対)"
cargo test --features "aaa,image,codec,texture-fit" --test test_gi_volume_oracle --test test_terrain_destruction_oracle \
  --test test_svo_api_oracle --test test_volume_api_oracle --test test_terrain_api_oracle \
  --test test_gi_api_oracle --test test_destruction_api_oracle --test test_provenance

step "test: openvdb"
cargo build --lib --no-default-features --features openvdb
cargo test --lib --no-default-features --features openvdb vdb

step "test: io format oracle の openvdb + hlsl arm (ci.yml の openvdb job と対)"
cargo test --features "hlsl,openvdb" --test test_io_format_oracle
step "test: provenance of the ALICE-VDB grid (ci.yml の openvdb job と対、3 件未満で fail)"
cargo test --features openvdb --test test_provenance alice_vdb 2>&1 | tee "${CARGO_TARGET_DIR:-target}/provenance-vdb.log"
python3 scripts/libtest_count.py "${CARGO_TARGET_DIR:-target}/provenance-vdb.log" --min 3 --what "ALICE-VDB provenance tests (openvdb)"

step "test: .abc against the Alembic library (ci.yml の alembic-oracle job と対、3 件未満で fail、library が無ければ PREFLIGHT-SKIP 行)"
abc_inc=""
for d in /usr/include /opt/homebrew/include /usr/local/include; do
  [ -f "$d/Alembic/Abc/All.h" ] && abc_inc=$d && break
done
python3 scripts/test_alembic_library_check.py
if [ -n "$abc_inc" ]; then
  # the release the CI job builds (tests/oracle/alembic-library.txt); a local library may carry HDF5 too
  python3 scripts/alembic_library_check.py "$abc_inc"
  abc_out="${CARGO_TARGET_DIR:-target}/alembic-oracle"
  mkdir -p "$abc_out"
  for n in alembic_reader alembic_refwriter; do
    c++ -std=c++17 -O1 -I"$abc_inc" -I"$abc_inc/Imath" "tests/oracle/$n.cpp" -L"${abc_inc%/include}/lib" -lAlembic -o "$abc_out/$n"
  done
  ALICE_SDF_ALEMBIC_READER="$abc_out/alembic_reader" ALICE_SDF_ALEMBIC_REFWRITER="$abc_out/alembic_refwriter" \
    cargo test --test test_alembic_oracle -- --ignored 2>&1 | tee "$abc_out/alembic-oracle.log"
  python3 scripts/libtest_count.py "$abc_out/alembic-oracle.log" --min 3 --what "Alembic oracle tests"
else
  echo "PREFLIGHT-SKIP: Alembic oracle: no Alembic headers (brew alembic, or a build of tests/oracle/alembic-library.txt), checked by the alembic-oracle CI job"
fi

step "gpu-parity: GPU <-> CPU law parity, shader validation, GPU marching cubes (Metal here, lavapipe in CI)"
ALICE_SDF_REQUIRE_GPU=1 cargo test --features "gpu,glsl,gpu-mesh,texture-fit" \
  --test test_gpu_law_parity --test test_gpu_noise_parity --test test_round_tie_parity \
  --test test_transpiler_naga_validate --test noise_shader_validate --test test_mesh_orientation --test test_mc_shared_vertex_oracle \
  --test test_texture_shader_gpu_parity --test test_npr_bytecode_gpu_parity --test test_instanced_wgsl_gpu_parity \
  --test test_gpu_eval_api_oracle --test test_glsl_export_oracle --test test_shader_precise_temporaries

step "gpu-parity: shader text semantics identifier (gpu, glsl, hlsl, volume, blinkscript, texture-fit、ci.yml の gpu-parity job と対、14 件未満で fail)"
ALICE_SDF_REQUIRE_GPU=1 cargo test --features "gpu,glsl,hlsl,volume,blinkscript,texture-fit" --test test_shader_text_provenance -- --nocapture 2>&1 | tee "${TMPDIR:-/tmp}/sdf-shader-text-provenance.log"
python3 scripts/libtest_count.py "${TMPDIR:-/tmp}/sdf-shader-text-provenance.log" --min 14 --what "shader text provenance tests (gpu, glsl, hlsl, volume, blinkscript, texture-fit)"

step "gpu-parity: GPU unit tests ran, not skipped (6, ci.yml の GPU unit tests ran と対)"
python3 scripts/test_gpu_test_count.py
ALICE_SDF_REQUIRE_GPU=1 cargo test --lib --features "gpu,gpu-mesh" test_gpu_ -- --nocapture --test-threads=1 2>&1 | tee "${TMPDIR:-/tmp}/sdf-gpu-unit.log"
python3 scripts/gpu_test_count.py "${TMPDIR:-/tmp}/sdf-gpu-unit.log" --ran 6 --max-skipped 0

step "gpu-parity: aaa (volume gpu_bake, ci.yml の GPU ↔ CPU parity (aaa — volume gpu_bake) と対)"
ALICE_SDF_REQUIRE_GPU=1 cargo test --features "aaa" --test test_gi_volume_oracle --test test_volume_api_oracle

step "gpu-parity: physics (LiveSdf gpu_mesh、ci.yml の GPU ↔ CPU parity (physics — LiveSdf gpu_mesh) と対)"
ALICE_SDF_REQUIRE_GPU=1 cargo test --features "physics,gpu" --test test_live_sdf_gpu_parity

# examples/bake_assets: SDF -> mesh (CPU, and GPU where available) + physics
# manifest, verified from disk, then each corruption in scripts/bake_teeth.sh
# must make verify fail (bake.yml と対、CI は lavapipe で GPU 部分も走らせる)
bake_out=$(mktemp -d)
step "bake: CPU part (examples/bake_assets without gpu-mesh) + teeth"
cargo run -q --example bake_assets -- bake "$bake_out/cpu"
scripts/bake_teeth.sh "$bake_out/cpu"
if [ "$(uname -s)" = "Darwin" ]; then
  step "bake: GPU part (gpu-mesh on Metal) + teeth"
  ALICE_SDF_REQUIRE_GPU=1 cargo run -q --example bake_assets --features gpu-mesh -- bake "$bake_out/gpu"
  ALICE_SDF_REQUIRE_GPU=1 scripts/bake_teeth.sh "$bake_out/gpu" gpu-mesh
else
  step "SKIPPED: bake GPU part (Metal が無い host) — 0 scene を GPU で mesh 化、CI の bake.yml が lavapipe で検査する"
fi
rm -rf "$bake_out"

step "bevy: bindings/bevy/alice-sdf-bevy build + test"
(cd bindings/bevy/alice-sdf-bevy && cargo build --lib && cargo test --lib)

step "wasm-build: release wasm artifact"
cargo build --release --target wasm32-unknown-unknown --features wasm --no-default-features
test -f target/wasm32-unknown-unknown/release/alice_sdf.wasm

step "python-smoke: maturin develop --features python,openvdb + python/tests/smoke.py + vdb_roundtrip.py"
if command -v maturin >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  venv=$(mktemp -d)/venv
  python3 -m venv "$venv"
  # shellcheck disable=SC1091
  . "$venv/bin/activate"
  pip install --quiet maturin numpy
  maturin develop --features python,openvdb
  python python/tests/smoke.py
  python python/tests/vdb_roundtrip.py
  deactivate
else
  echo "skip: maturin / python3 not installed" >&2
fi

step "scip: L0 ratchet + docs/integration-status.md (rust-analyzer SCIP、約 4 分)"
scip_reach_step 0

step "release-profile.yml: goldens and the float environment in the release profile (G4 / G5)"
for spec in test_det_golden:3: test_det_parity:1:jit test_det_near_zero_parity:2:jit test_det_near_zero_probe:3:jit float_environment:2:; do
  name=${spec%%:*}
  rest=${spec#*:}
  min=${rest%%:*}
  feat=${rest#*:}
  cargo test --release --features "$feat" --test "$name" 2>&1 | tee "${CARGO_TARGET_DIR:-target}/release-$name.log"
  python3 scripts/libtest_count.py "${CARGO_TARGET_DIR:-target}/release-$name.log" --min "$min" --what "$name (release)"
done
cargo test --release --lib -- law_minmax_signed_zero_table tree_csg_ops_carry_the_law_on_signed_zero 2>&1 | tee "${CARGO_TARGET_DIR:-target}/release-lib-law.log"
python3 scripts/libtest_count.py "${CARGO_TARGET_DIR:-target}/release-lib-law.log" --min 2 --what "Law max / min unit tests (release)"

step "release-profile.yml: effective rustc flags (G14, dynamic)"
python3 scripts/build_flags_check.py -- cargo check --release --lib
python3 scripts/build_flags_check.py -- cargo check --release --lib --no-default-features
python3 scripts/build_flags_check.py -- cargo check --release --lib --features "$LINUX_ALL"

step "audit: cargo audit (RustSec)"
if command -v cargo-audit >/dev/null 2>&1; then
  cargo audit --deny yanked --ignore RUSTSEC-2025-0141 --ignore RUSTSEC-2024-0436
else
  echo "skip: cargo-audit not installed" >&2
fi

echo; echo "preflight OK"
