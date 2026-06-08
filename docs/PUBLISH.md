# ALICE-SDF Distribution Strategy

This document describes how each ALICE-SDF artifact is distributed today and
what the roadmap looks like.

---

## Current state (as of v1.7.2)

| Artifact | Channel | Status |
|----------|---------|--------|
| `alice_sdf` (Rust crate, core) | `path = "..."` only (private workspace) | ❌ not on crates.io |
| `alice-sdf-mobile` (UniFFI wrapper) | source build | ❌ not on crates.io |
| `alice-sdf-openxr` | source build | ❌ not on crates.io |
| `alice-sdf-bevy` | source build | ❌ not on crates.io |
| `alice-sdf-server` | source build (`publish = false`) | ❌ not on crates.io |
| Python wheel (`alice-sdf`) | GitHub Releases (`.whl` per platform, abi3-py310) | ✅ from v1.7.2 |
| iOS / Android `.xcframework` / `.aar` | source build (`mobile/packaging/*/build-*.sh`) | ✅ scripted |
| Unreal Engine plugin | source tree (`unreal-plugin/`) | ✅ source distribution |
| WASM (`alice_sdf.wasm`) | source build (`cargo build --target wasm32-unknown-unknown --features wasm`) | ✅ scripted |
| Three.js npm package (`@alice-sdf/threejs`) | source tree | ❌ not on npmjs.com |

## Why crates.io publish is blocked

The `alice_sdf` core crate has **path dependencies** on sibling crates in the
private ALICE Eco-System workspace (`alice-physics`, `alice-codec`, `libasp`,
`alice-cache`, `alice-font`). `cargo publish` refuses path deps without a
`version = "..."` registry pin.

In CI we sidestep this with stub crates (`.github/actions/alice-stubs`) that
generate empty `Cargo.toml`s on the fly, but this is not acceptable for a
crates.io publish.

## Roadmap to crates.io

To publish `alice_sdf` (and the binding crates that depend on it):

1. **Decouple optional bridges** — make each bridge feature (`physics`, `codec`,
   `font`) **optional + registry-pinned**. Use `version = "0.x"` + `optional = true`
   so that consumers without the registry crate can still build the core.
   - Requires publishing `alice-physics`, `alice-codec`, etc. to crates.io
     first, OR splitting them off into stub crates that ship the same name on
     crates.io.

2. **Publish bridge crates first** — in dependency order:
   - `libasp` (codec primitives)
   - `alice-cache`
   - `alice-codec`
   - `alice-font`
   - `alice-physics`

3. **Publish `alice-sdf` core** — replace path deps with registry pins.

4. **Publish binding crates** — `alice-sdf-mobile`, `alice-sdf-openxr`,
   `alice-sdf-bevy`. These already only depend on `alice-sdf` core
   (with `default-features = false`), so they slot in after step 3.

5. **Publish `alice-sdf-server`** — set `publish = true` once the core is on
   crates.io; consumers can then `cargo install alice-sdf-server`.

## Why we don't rush this

The Eco-System bridge crates (`alice-physics`, `alice-codec` etc.) are still
under active API churn. Publishing them prematurely would either:

- Lock us into early APIs we want to evolve, or
- Force a `0.x → 0.y` major-version bump on every Eco-System refactor, which
  would block downstream releases of `alice-sdf` itself.

Until those bridges stabilise, the current strategy is:
- Distribute end-user-facing artifacts (Python wheels, iOS XCFramework, Android
  AAR, WASM, Unreal plugin) directly from GitHub Releases.
- Keep the Rust crates source-only.

## What this means for users today

- **Python**: `pip install alice-sdf` from GitHub Releases (v1.7.2+) — fully supported.
- **iOS / Android**: clone the repo, run `mobile/packaging/ios/build-xcframework.sh`
  or `mobile/packaging/android/build-aar.sh`. Pre-built XCFramework / AAR
  attachments to GitHub Releases are planned.
- **Rust**: clone the repo and use as a path dep, or wait for the crates.io
  publish above.
- **Unreal Engine**: copy `unreal-plugin/` into your project's `Plugins/`
  directory.
- **WASM**: `cargo build --release --target wasm32-unknown-unknown --features wasm`,
  pick up `target/wasm32-unknown-unknown/release/alice_sdf.wasm`.

## Pre-built wheels — channel details

Pre-built wheels are uploaded to GitHub Releases by the
`.github/workflows/release-wheels.yml` workflow when a tag `v*` is pushed. The
matrix is:

| Wheel name | Target | Runner | Wheel suffix |
|------------|--------|--------|--------------|
| `linux-x86_64` | `x86_64-unknown-linux-gnu` | `ubuntu-latest` | `manylinux2014_x86_64` |
| `linux-aarch64` | `aarch64-unknown-linux-gnu` | `ubuntu-latest` (cross) | `manylinux2014_aarch64` |
| `macos-arm64` | `aarch64-apple-darwin` | `macos-14` (Apple Silicon) | `macosx_11_0_arm64` |
| `macos-x86_64` | `x86_64-apple-darwin` | `macos-15-intel` | `macosx_10_15_x86_64` |
| `windows-x86_64` | `x86_64-pc-windows-msvc` | `windows-latest` | `win_amd64` |

All wheels are built with `pyo3 = { features = ["abi3-py310"] }`, so each wheel
covers **Python 3.10 / 3.11 / 3.12 / 3.13** in a single file.
