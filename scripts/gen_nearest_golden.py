#!/usr/bin/env python3
"""Reference float64 distances for tests/test_sheet_field_nearest.rs (`GOLDEN`), independent of the crate.

Two per-face distances from a point p to a triangle a b c, written from their textbook definitions in IEEE double
(Python floats; no fused multiply-add, sqrt correctly rounded, the same as Rust's f64):

- plane-or-edge: the barycentric coordinates (u, v) of p's projection from the normal equations of the edges ab, ac;
  inside (u, v >= 0, u + v <= 1) the distance to the plane |ap . n| / |n|, else the smallest distance to the three
  edge segments (each the clamped projection onto the segment)
- Ericson's closest point on a triangle (Real-Time Collision Detection, 5.1.5): the Voronoi regions of the vertices,
  then the edges, then the face, tested in that order, and the distance to the closest point

Degenerate guards: a denominator clamped to 1e-30 (max(x, 1e-30) for the squared lengths and the Gram determinant;
x if |x| > 1e-30 else 1e-30 for the Ericson ratios).

The rows' inputs (synthetic triangles and query points) are data in the test file; this script re-derives both
distances of every row from them. Usage:
  gen_nearest_golden.py --check <test file>   recompute every GOLDEN row's two distances and compare bit for bit;
                                              0 rows read fails
"""
import math
import re
import struct
import sys


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def unbits(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def maxf(x, m):
    return x if x > m else m


def guard(x):
    return x if abs(x) > 1e-30 else 1e-30


def seg(p, p0, p1):
    e = sub(p1, p0)
    t = dot(sub(p, p0), e) / maxf(dot(e, e), 1e-30)
    t = min(max(t, 0.0), 1.0)
    return norm(sub(p, (p0[0] + t * e[0], p0[1] + t * e[1], p0[2] + t * e[2])))


def plane_seg(p, a, b, c):
    ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    n = cross(ab, ac)
    nn = dot(n, n)
    ab2, ac2, abac = dot(ab, ab), dot(ac, ac), dot(ab, ac)
    det = maxf(ab2 * ac2 - abac * abac, 1e-30)
    u = (ac2 * d1 - abac * d2) / det
    v = (ab2 * d2 - abac * d1) / det
    if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
        return abs(dot(ap, n)) / math.sqrt(maxf(nn, 1e-30))
    return min(min(seg(p, a, b), seg(p, b, c)), seg(p, c, a))


def ericson(p, a, b, c):
    ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = sub(p, b)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = sub(p, c)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    if d1 <= 0.0 and d2 <= 0.0:
        w = (1.0, 0.0, 0.0)
    elif d3 >= 0.0 and d4 <= d3:
        w = (0.0, 1.0, 0.0)
    elif d6 >= 0.0 and d5 <= d6:
        w = (0.0, 0.0, 1.0)
    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        t = d1 / guard(d1 - d3)
        w = (1.0 - t, t, 0.0)
    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        t = d2 / guard(d2 - d6)
        w = (1.0 - t, 0.0, t)
    elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        t = (d4 - d3) / guard((d4 - d3) + (d5 - d6))
        w = (0.0, 1.0 - t, t)
    else:
        den = guard(va + vb + vc)
        v, ww = vb / den, vc / den
        w = (1.0 - v - ww, v, ww)
    q = tuple(w[0] * a[i] + w[1] * b[i] + w[2] * c[i] for i in range(3))
    return norm(sub(p, q))


ROW = re.compile(r"\(\[((?:0x[0-9a-f]+,\s*){11}0x[0-9a-f]+)\],\s*(0x[0-9a-f]+),\s*(0x[0-9a-f]+)\)")


def check(path: str) -> int:
    text = open(path, encoding="utf-8").read()
    start = text.index("const GOLDEN")
    rows = ROW.findall(text[start:text.index("];", start)])
    if not rows:
        print("gen-nearest-golden: 0 GOLDEN rows read")
        return 1
    bad = 0
    for k, (xs, ps, er) in enumerate(rows):
        v = [unbits(int(x, 16)) for x in xs.split(",")]
        p, a, b, c = v[0:3], v[3:6], v[6:9], v[9:12]
        got = (bits(plane_seg(p, a, b, c)), bits(ericson(p, a, b, c)))
        if got != (int(ps, 16), int(er, 16)):
            bad += 1
            print(f"gen-nearest-golden: row {k}: plane_seg {got[0]:#018x} vs {ps}, ericson {got[1]:#018x} vs {er}")
    print(f"gen-nearest-golden: {len(rows)} rows recomputed, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        sys.exit(check(sys.argv[2]))
    print(__doc__)
    sys.exit(2)
