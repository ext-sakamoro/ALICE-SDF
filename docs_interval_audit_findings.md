# Interval-evaluation looseness audit: repeat / transform nodes

Measured on main (73e24c0), git 2.50, release.

## Method
- (A) `eval_interval` width vs the true SDF range over a 9^3 grid of 1-unit boxes in [-14,14]^3; soundness = interval contains the true range on every box.
- (B) `compute_tight_aabb_with_config(preset_large)` reach vs the true zero-level-set reach (dense scan, |sdf| < 0.6).
- child: `box3d(3,3,3).translate(8,0,0)` (off-centre); `box3d(3,3,8)` for twist/bend.
- harness: `examples/interval_audit.rs`.

## Table
| node | med x | p95 x | sound | aabb_reach | true_reach | verdict |
|---|---|---|---|---|---|---|
| control translate(box) | 1.00 | 1.00 | yes | 9.5 | 10.0 | tight |
| polar_repeat(6) | 2.15 | 4.51 | yes | 500 | 10.0 | LOOSE (aabb bug) |
| repeat_finite([3,1,1]) | 1.00 | 1.28 | yes | 18.5 | 19.0 | tight |
| repeat_infinite(16) | 6.41 | 8.48 | yes | 500 | (space-filling) | interval-loose; aabb correct |
| mirror(x) | 1.00 | 1.00 | yes | 9.5 | 10.0 | tight |
| twist(0.3) | 3.56 | 6.56 | yes | 500 | 4.0 | LOOSE (aabb bug) |
| bend(0.2) | 9.36 | 12.93 | yes | 500 | 4.0 | LOOSE (aabb bug) |

Soundness holds on every box (`eval_interval` contains the true range) — the invariant is intact; the defect is looseness, not unsoundness.

## Findings
- **polar_repeat / twist / bend** produce a bounded shape but a `preset_large` AABB of +/-500, so preview cells are far larger than the feature and marching cubes yields 0 triangles (e.g. `polar_repeat(6, translate(8,0,0, box))`). Root: the interval evaluation maps a box across the full rotated / angular range instead of folding it into one sector or a bounded local range.
- **repeat_infinite** is genuinely space-filling, so a large AABB is correct (the `preset_large` bound is the practical clamp); its interval width is still ~6x loose and could be tightened, lower priority.
- **repeat_finite** and **mirror** are tight.

## Proposed fix
- **polar_repeat**: before evaluating the child over a box, fold the box's angular range (atan2 over the box corners) into one sector `[0, 2*pi/count)`, map the box into sector-local coordinates, and evaluate the child there, so the child sees one sector rather than the full circle.
- **twist / bend**: bound the rotation angle over the box's extent (e.g. strength * z-range) and evaluate the child over the bounded-rotation box rather than the worst-case full rotation.

## Oracle (soundness + tightness), per node over a box set
- soundness: `eval_interval(node, box).lo <= true_min` and `.hi >= true_max` (must hold; currently holds).
- tightness: `interval_width <= k * true_width`, k ~ 3 (currently fails for polar_repeat / twist / bend).
