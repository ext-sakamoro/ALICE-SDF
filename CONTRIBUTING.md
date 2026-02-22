# Contributing to ALICE-SDF

## Build

```bash
cargo build
cargo build --no-default-features   # lib-only (no CLI)
```

## Test

```bash
cargo test
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **SDF semantics**: all primitives return the *signed distance* — negative inside, positive outside, zero on the surface.
- **CSG correctness**: `union = min`, `intersection = max`, `subtraction = max(a, -b)`. Smooth variants use polynomial blending.
- **Deterministic evaluation**: same SDF tree + same point = same distance on all platforms.
- **Rayon parallelism**: mesh generation and batch evaluation use work-stealing parallelism.
- **SIMD via `wide`**: portable AVX2/AVX-512/NEON paths for inner-loop evaluation.
- **File format**: `.asdf` binary (bincode + CRC32) and `.asdf.json` (serde_json) formats.
- **Reciprocal constants**: pre-compute `1.0 / N` to avoid division in hot paths.
