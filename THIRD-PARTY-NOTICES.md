# Third-party notices

ALICE-SDF is an independent Rust implementation, but a number of its distance
functions follow **mathematical forms published by others**. This file is the
single place where those sources are named; the per-file doc comments point
back here.

Nothing in this repository is a copy of third-party source code: each law was
re-implemented in Rust (and re-derived where the published form was only a
bound — see `CHANGELOG.md` for the Ellipsoid and Rounded Cone cases, which were
replaced with exact forms in 1.10.3 / 1.11.0).

## Inigo Quilez — distance function forms

Signed-distance formulas for a large part of the primitive set follow the
articles and Shadertoy demos of **Inigo Quilez** (`sdEgg`, `sdRoundCone`,
`sdCappedCone`, `sdCappedTorus`, `sdArc`, `sdMoon`, `sdCross`, `sdBlobbyCross`,
`sdRhombus`, `sdVesica`, `sdBezier`, the exact triangle/quad distances used by
the mesh BVH, and others).

- Files that name him in a doc comment: 34 (`grep -rl "Inigo Quilez" src/`)
- Layer in the architecture: `src/primitives/` (`docs/USAGE.md` §Layer 2)
- Also used as a convention: `iTime` and the Shadertoy-style `mainImage`
  wrapper emitted by the GLSL transpiler (`docs/ROADMAP.md` ADR)

## Mercury (hg_sdf) — operator forms

The stepped / columned / chamfered boolean operators and the octant mirror
follow the forms in **hg_sdf** by Mercury (`fOpUnionStairs`, `fOpUnionColumns`,
`fOpUnionChamfer`, `pR45` / octant folding).

- Files that name it in a doc comment: 15 (`grep -rl "hg_sdf\|Mercury" src/`)

## Perlin gradient table

`src/modifiers/noise.rs` uses a gradient table derived from Ken Perlin's
original improved-noise gradient set.

## License status of the upstream sources

The forms above are mathematics, and the implementations here are our own, but
the **upstream terms have not been verified inside this repository yet**. That
verification is tracked in the ALICE-* backlog together with the license census
(2026-09-27); until it lands, treat this file as attribution, not as a
statement about upstream licensing.

If you are an author listed here and want the wording or the attribution
changed, please open an issue.
