"""ALICE-VDB bytes from the Python binding — run by CI (`python-smoke` job), which builds the
binding with the `openvdb` feature.

`alice_sdf.bake_to_vdb` returns the bytes of the Rust `io::vdb::bake_to_vdb`: `ALICEVDB2`, the
32-byte meaning identifier, then the grid. This test reads them with an independent parser and
checks every distance of a unit sphere against the closed form |p| - 1 at the sample positions
`min + i * (max - min) / (res - 1)` (both ends included, x fastest; one sample at `min` for res 1). The identifier must be
the meaning identifier of the ALICE-VDB form computed independently (meaning_id.py) from the
build's semantics identifier, and the binding's JSON writer must stamp the JSON form's one. A
binding built without the feature fails here, it is not skipped.

Author: Moroya Sakamoto
"""

from __future__ import annotations

import json
import math
import struct
import sys

import alice_sdf as sdf
from meaning_id import meaning_id

TOL = 1e-5


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def parse(data: bytes) -> tuple[bytes, int, float, float, list[float]]:
    check(data[:9] == b"ALICEVDB2", f"magic: {data[:9]!r}")
    check(len(data) >= 9 + 32 + 12, f"header cut short: {len(data)} bytes")
    ident = data[9:41]
    res, bmin, bmax = struct.unpack_from("<Iff", data, 41)
    n = res * res * res
    check(len(data) == 53 + 4 * n, f"length {len(data)}, expected {53 + 4 * n} for res {res}")
    values = list(struct.unpack_from(f"<{n}f", data, 53))
    return ident, res, bmin, bmax, values


def main() -> int:
    check(hasattr(sdf, "bake_to_vdb"), "alice_sdf.bake_to_vdb is missing: build with --features python,openvdb")
    node = sdf.sphere(1.0)
    semantics = sdf.semantics_id()
    doc = json.loads(sdf.to_json(node))
    check("alice:semantics_id" not in doc, "the JSON writer still stamps the raw identifier")
    check(doc["alice:meaning_id"] == meaning_id("json", semantics), "the JSON meaning identifier")
    stamp = meaning_id("alice-vdb", semantics)
    check(stamp != doc["alice:meaning_id"], "two forms share a meaning identifier")
    compared = 0
    for res, bounds in [(5, (-1.5, 1.5)), (9, (-2.0, 2.0)), (2, (-1.0, 1.0)), (1, (-1.0, 1.0))]:
        ident, r, bmin, bmax, values = parse(sdf.bake_to_vdb(node, bounds, res))
        check(ident.hex() == stamp, f"identifier {ident.hex()} != the ALICE-VDB meaning identifier {stamp}")
        check((r, bmin, bmax) == (res, bounds[0], bounds[1]), f"grid header {(r, bmin, bmax)}")
        step = (bounds[1] - bounds[0]) / max(res - 1, 1)
        for k in range(res):
            for j in range(res):
                for i in range(res):
                    x, y, z = (bounds[0] + c * step for c in (i, j, k))
                    want = math.sqrt(x * x + y * y + z * z) - 1.0
                    got = values[(k * res + j) * res + i]
                    check(abs(got - want) <= TOL, f"res {res} ({i},{j},{k}): {got} != {want}")
                    compared += 1
    for bad in [((-1.0, 1.0), 0), ((1.0, 1.0), 4)]:
        try:
            sdf.bake_to_vdb(node, bad[0], bad[1])
        except ValueError:
            pass
        else:
            raise AssertionError(f"bake_to_vdb{bad} did not raise")
    check(compared == 5**3 + 9**3 + 2**3 + 1, f"compared {compared} samples")
    print(f"vdb_roundtrip: ok ({compared} samples compared, identifier {stamp[:16]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
