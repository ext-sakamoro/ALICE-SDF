# fuzz/seeds

Committed regression inputs for the fuzz targets: every crash a target has
found is kept here (raw libFuzzer input, one file per finding) and replayed
before random exploration by `.github/workflows/fuzz.yml`.

- `fuzz_eval_parity/` — 7 findings from 2026-09-15 (round tie-break sign,
  Scale(ExpSmoothUnion) leaf-scaling, exp smooth underflow, SIMD atan2 sector
  tie, `p / s` vs `p * (1 / s)` tie, glam vs generic rotation at the pyramid
  base, four nested polar repeats)

Replay one locally: `cd fuzz && cargo +nightly fuzz run fuzz_eval_parity seeds/fuzz_eval_parity/<file>`
