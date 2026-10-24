#!/usr/bin/env python3
"""Assert every sub-package that ships its own copy of the licence text is byte-identical to the
root LICENSE-Proprietary (so each package carries the full, current text and cannot drift).
Fails on: a missing copy, a byte difference, or nothing compared (0 copies => fail)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "LICENSE-Proprietary"
# sub-package licence files that MUST equal the root LICENSE-Proprietary byte-for-byte
COPIES = [
    "bindings/bevy/alice-sdf-bevy/LICENSE-Proprietary",
    "bindings/openxr/LICENSE-Proprietary",
    "bindings/threejs/LICENSE-Proprietary",
    "examples/wasm-demo/LICENSE-Proprietary",
    "mobile/uniffi-wrapper/LICENSE-Proprietary",
    "npm/LICENSE-Proprietary",
    "scripts/clippy-lib-canary/LICENSE-Proprietary",
    "server/LICENSE-Proprietary",
    "skills/implicit-cad/LICENSE-Proprietary",
    "unity-sdf-universe/LICENSE-Proprietary",
    "vrchat-package/LICENSE-Proprietary",
]
def check(root: Path, copies: list[str]) -> int:
    master_path = root / MASTER.name
    if not master_path.is_file():
        print(f"error: {MASTER.name} missing", file=sys.stderr); return 1
    master = master_path.read_bytes()
    errs, compared = [], 0
    for rel in copies:
        p = root / rel
        if not p.is_file():
            errs.append(f"{rel}: missing (must ship a byte-identical copy of {MASTER.name})"); continue
        compared += 1
        if p.read_bytes() != master:
            errs.append(f"{rel}: differs from {MASTER.name} (byte-identical copy required)")
    if compared == 0:
        errs.append("0 license copies compared (check is vacuous — list is empty or all missing)")
    for e in errs:
        print(f"error: {e}", file=sys.stderr)
    print(f"license-copy check: {compared} copy/copies compared against {MASTER.name}", file=sys.stderr)
    return 1 if errs else 0
def main() -> int:
    return check(ROOT, COPIES)
def self_test() -> int:
    # runs check() in-process on a scratch tree: no rewritten copy of this file, no
    # subprocess, so nothing depends on the platform's newline or path handling
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        (r/MASTER.name).write_bytes(b"X\n"); (r/"vrchat-package").mkdir(); (r/"server").mkdir()
        (r/"vrchat-package/LICENSE.md").write_bytes(b"X\n")
        (r/"server/LICENSE-Proprietary").write_bytes(b"X\n")
        assert check(r, ["vrchat-package/LICENSE.md", "server/LICENSE-Proprietary"]) == 0, "identical copies should pass"
        (r/"server/LICENSE-Proprietary").write_bytes(b"X\r\n")
        assert check(r, ["server/LICENSE-Proprietary"]) == 1, "a newline difference must fail"
        (r/"vrchat-package/LICENSE.md").write_bytes(b"X tampered\n")
        assert check(r, ["vrchat-package/LICENSE.md"]) == 1, "a differing copy must fail"
        (r/"vrchat-package/LICENSE.md").unlink()
        assert check(r, ["vrchat-package/LICENSE.md"]) == 1, "a missing copy must fail"
        assert check(r, []) == 1, "0 copies compared must fail (not vacuously pass)"
        (r/MASTER.name).unlink()
        assert check(r, ["server/LICENSE-Proprietary"]) == 1, "a missing master must fail"
    print("license_copy_check self-test ok"); return 0
if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main())
