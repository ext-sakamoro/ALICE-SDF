"""Python binding smoke test — analytic oracle, run by CI (`python-smoke` job)
after `maturin develop --features python`.

Every number asserted here has a closed-form answer (unit sphere, box), so
the test is a check of the binding *and* of the law behind it, not of the
binding against itself. Plain asserts, no pytest dependency.

Author: Moroya Sakamoto
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np

import alice_sdf as sdf

EPS = 1e-5


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def signed_volume(vertices: np.ndarray, indices: np.ndarray) -> float:
    """Σ (a · (b × c)) / 6 over triangles: positive for outward winding."""
    tri = vertices[indices.reshape(-1, 3)]
    a, b, c = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def main() -> None:
    v = sdf.version()
    check(isinstance(v, str) and v.count(".") >= 2, f"version(): {v!r}")

    # ── point evaluation: unit sphere ────────────────────────────────────
    s = sdf.SdfNode.sphere(1.0)
    check(abs(s.eval(0.5, 0.0, 0.0) - (-0.5)) < EPS, "sphere inside")
    check(abs(s.eval(2.0, 0.0, 0.0) - 1.0) < EPS, "sphere outside")
    check(abs(s.eval(0.0, 0.0, 0.0) - (-1.0)) < EPS, "sphere centre")

    # ── batch evaluation == analytic |p| − 1 ─────────────────────────────
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2.0, 2.0, size=(4096, 3)).astype(np.float32)
    got = sdf.eval_batch(s, pts)
    want = np.linalg.norm(pts, axis=1) - 1.0
    check(got.shape == (4096,), f"eval_batch shape {got.shape}")
    check(np.max(np.abs(got - want)) < 1e-4, "eval_batch vs analytic")

    # ── compiled path ≡ tree path ────────────────────────────────────────
    c = sdf.compile_sdf(s)
    check(abs(c.eval(0.5, 0.0, 0.0) - (-0.5)) < EPS, "compiled eval")
    got_c = sdf.eval_compiled_batch(c, pts)
    check(np.max(np.abs(got_c - want)) < 1e-4, "eval_compiled_batch vs analytic")
    check(np.max(np.abs(got_c - got)) < 1e-5, "compiled vs tree")

    # ── CSG + transform laws ─────────────────────────────────────────────
    moved = s.translate(1.0, 0.0, 0.0)
    check(abs(moved.eval(0.0, 0.0, 0.0)) < EPS, "translated sphere surface at origin")
    u = s.union(moved)
    check(abs(u.eval(0.5, 0.0, 0.0) - (-0.5)) < EPS, "union takes the min")
    b = sdf.SdfNode.box3d(2.0, 2.0, 2.0)  # full extents (half = 1)
    check(abs(b.eval(0.0, 0.0, 0.0) - (-1.0)) < EPS, "box centre")
    check(abs(b.eval(2.0, 0.0, 0.0) - 1.0) < EPS, "box face distance")
    check(abs(b.eval(2.0, 2.0, 2.0) - math.sqrt(3.0)) < 1e-4, "box corner distance")

    # ── mesh: vertices on the surface, outward winding, analytic volume ──
    res = 32
    verts, idx = sdf.to_mesh(s, (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), res)
    check(verts.ndim == 2 and verts.shape[1] == 3, f"vertices shape {verts.shape}")
    check(idx.ndim == 1 and idx.size % 3 == 0 and idx.size > 0, f"indices shape {idx.shape}")
    check(int(idx.max()) < verts.shape[0], "index out of range")
    radii = np.linalg.norm(verts, axis=1)
    cell = 3.0 / res
    check(np.max(np.abs(radii - 1.0)) < cell, f"vertices off the sphere by {np.max(np.abs(radii - 1.0))}")
    vol = signed_volume(verts.astype(np.float64), idx.astype(np.int64))
    want_vol = 4.0 / 3.0 * math.pi
    check(vol > 0.0, f"mesh is wound inward (signed volume {vol})")
    check(abs(vol - want_vol) / want_vol < 0.05, f"volume {vol} vs {want_vol}")

    # ── serialization round trips ────────────────────────────────────────
    j = sdf.to_json(u)
    check(isinstance(j, str) and json.loads(j), "to_json")
    back = sdf.from_json(j)
    for p in ((0.0, 0.0, 0.0), (0.3, 0.2, -0.7), (1.5, 0.0, 0.0)):
        check(abs(back.eval(*p) - u.eval(*p)) < EPS, f"json round trip at {p}")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "u.asdf")
        sdf.save_sdf(u, path)
        check(os.path.getsize(path) > 0, "asdf file empty")
        loaded = sdf.load_sdf(path)
        for p in ((0.0, 0.0, 0.0), (0.3, 0.2, -0.7), (1.5, 0.0, 0.0)):
            check(abs(loaded.eval(*p) - u.eval(*p)) < EPS, f"asdf round trip at {p}")

    print(f"alice_sdf {v}: python smoke OK (mesh {verts.shape[0]} verts, volume {vol:.4f})")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
