# Wiring debt

Public items that no production code of this crate calls yet, each marked `// ALLOW-UNWIRED: wiring debt <key>, ...`
in the source. A key lists what is not wired and the condition that resolves it; the marker and its row go away
together.

| Key | Not wired | Resolved when |
|---|---|---|
| `kink-point-evaluator` | `sheet_field::kink::KinkSheet::from_geom` and `KinkSheet::eval_all`: the exact point evaluator (`s`, `d_b`, unsigned distance) a sheet builder uses for its node values, the same code the march runs in a flagged cell. Checked by hand values in `src/sheet_field/kink_build.rs::derived_normals_by_hand_and_the_point_evaluator`. | A builder in this crate writes node values from a sheet's mesh through it. |
