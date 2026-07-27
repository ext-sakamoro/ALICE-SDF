# ALICE-SDF Syntax Reference

126-construct SDF library. See parent crate `~/ALICE-SDF/API.md` for full argument documentation with types.

## Primitives (72)

3D: `sphere(r)`, `box3d(hx, hy, hz)`, `rounded_box(hx, hy, hz, r)`, `cylinder(r, h)`, `torus(R, r)`, `cone(r, h)`, `capsule(r, h)`, `ellipsoid(rx, ry, rz)`, `plane(nx, ny, nz, d)`, `octahedron(s)`, `rounded_cone(r1, r2, h)`, `pyramid(h)`, `hex_prism(r, h)`, `link(l, r1, r2)`, `capped_cone(h, r1, r2)`, `capped_torus(R, r, angle)`, `rounded_cylinder(r, rr, h)`, `tube(r_out, t, h)`, `barrel(r, h, b)`, `heart(s)`, `egg(ra, rb)`, `helix(R, r, pitch, h)`, `tetrahedron(s)`, `box_frame(hx, hy, hz, e)`, `diamond(r, h)`, `star_polygon(r, n, m, h)`, `cross_shape(l, t, r, h)`, `triangle(...)`, `bezier(...)`, `triangular_prism(...)`, `cut_sphere(...)`, `cut_hollow_sphere(...)`, `death_star(...)`, `solid_angle(...)`, `rhombus(...)`, `horseshoe(...)`, `vesica(...)`, `infinite_cylinder(...)`, `infinite_cone(...)`, `superellipsoid(...)`, `rounded_x(...)`, `pie(...)`, `trapezoid(...)`, `parallelogram(...)`, `tunnel(...)`, `uneven_capsule(...)`, `arc_shape(...)`, `moon(...)`, `blobby_cross(...)`, `parabola_segment(...)`, `regular_polygon(...)`, `stairs_prim(...)`, `dodecahedron(s)`, `icosahedron(s)`, `truncated_octahedron(s)`, `truncated_icosahedron(s)`.

TPMS (Triply Periodic Minimal Surfaces): `gyroid`, `schwarz_p`, `diamond_surface`, `neovius`, `lidinoid`, `iwp`, `frd`, `fischer_koch_s`, `pmy`. Great for lightweight infill and metamaterial lattices.

2D (for `extrude` / `revolution` upstream): `circle_2d`, `rect_2d`, `segment_2d`, `rounded_rect_2d`, `annular_2d`.

Chamfered variant: `chamfered_cube`.

## CSG Operations (24)

Sharp: `union`, `intersection`, `subtract`, `xor`, `pipe`, `engrave`, `groove`, `tongue`.

Smooth (organic blend): `smooth_union(k, a, b, ...)`, `smooth_intersection`, `smooth_subtract`, `exp_smooth_union`, `exp_smooth_intersection`, `exp_smooth_subtraction`.

Chamfered (hard bevel): `chamfer_union(r, a, b, ...)`, `chamfer_intersection`, `chamfer_subtraction`.

Stepped (staircase): `stairs_union(r, n, a, b, ...)`, `stairs_intersection`, `stairs_subtraction`.

Columnar: `columns_union(r, n, a, b, ...)`, `columns_intersection`, `columns_subtraction`.

## Transforms (7)

`translate(x, y, z, child)`, `rotate(rx, ry, rz, child)` (Euler degrees), `scale(s, child)`, `scale_non_uniform(sx, sy, sz, child)`, `mirror(axis, child)`, `polar_repeat(n, child)`, `shear(...)`.

## Modifiers (23)

Surface: `round(r, child)`, `onion(t, child)`, `shell(t, child)`, `surface_roughness(...)`, `noise(...)`, `displacement(...)`.

Deformation: `twist(angle, child)`, `bend(angle, child)`, `taper(...)`, `elongate(...)`.

Repetition: `repeat(...)`, `repeat_finite(...)`, `octant_mirror(child)`, `icosahedral_symmetry(child)`.

Generation: `revolution(child_2d)`, `extrude(h, child_2d)`, `sweep_bezier(...)`.

3D Print structural intent: `lattice_infill(...)`, `diamond_infill(...)`, `schwarz_infill(...)`. Use these instead of raw `intersection` + TPMS to avoid non-manifold mesh (see ALICE-LOL `CLAUDE.md` STL output rules).

Material tagging: `with_material(mat_id, child)`.

## Composition rules

1. All arguments are `f32` unless stated otherwise.
2. Operations take 2+ children: `union(sphere(1.0), box3d(0.5, 0.5, 0.5))`.
3. `subtract(a, b)` is asymmetric — `a` minus `b`. Chain nested `subtract` for sequential carving; do **not** union-collect cutters (produces non-manifold edges).
4. `intersection` + TPMS is unsafe for mesh export — use `*_infill` primitives instead.
5. Smooth booleans have a blend radius `k` (world units) as first arg.

## Evaluation modes (7)

| Mode | When to use |
|--|--|
| `interpret` | LLM-generated JSON tree, iterative editing |
| `vm` | Compiled `SdfNode` bytecode, faster than interpret |
| `simd` | 8-wide SIMD batch, best for mesh generation and dense sampling |
| `bvh` | Bounding volume hierarchy, best for large scenes |
| `soa` | Struct-of-Arrays batch, cache-friendly |
| `jit` | Cranelift JIT, best for hot loops |
| `gpu` | wgpu compute, best for volume baking / GPU marching cubes |

Default is `simd` for `to_mesh()` and `gpu` for `export_glb()` when `--features gpu` is enabled.

## Shader targets (3)

See `shader-targets.md` for GLSL / WGSL / HLSL emit differences.
