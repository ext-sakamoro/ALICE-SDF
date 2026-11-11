# Third-party notices

## Origin

ALICE-SDF is our own implementation, which grew out of the principle
"Don't send data, send the law" (ALICE-Zip's procedural laws).

The same methods are also widely known from published raymarching and SDF
literature, including Inigo Quilez's articles (iquilezles.org), hg_sdf by
Mercury and Ken Perlin's "Improving Noise" (2002). Where a doc comment names
one of these, it says where the method is *also known from*; it does not mean
the code was taken from there.

## Third-party code

No third-party source code is included in `src/`. The sources named in doc
comments were checked file by file:

- Distance functions that name Inigo Quilez's articles: our own Rust code; the
  articles are cited for the method only.
- Operators that name hg_sdf (stairs, columns, chamfer, engrave, groove, pipe,
  tongue, octant folding, GDF normals): our own Rust code; none of hg_sdf's
  functions is called or reproduced, and several forms deliberately differ
  from hg_sdf (see `src/operations/stairs.rs` and `src/operations/chamfer.rs`).
- Gradient noise (`src/modifiers/noise.rs`, `src/npr/noise.rs`): our own hash
  and our own encoding of the 12 cube-edge gradient directions; Perlin's
  reference permutation table is not used.

If a function is ever ported from someone else's code, it will be listed here
individually with its licence.

If you are an author named here and want the wording changed, please open an
issue.
