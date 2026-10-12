#!/usr/bin/env python3
"""Assert every sub-package that ships its own copy of the ALICE Community License is byte-identical
to the root LICENSE-COMMUNITY (so a published package carries the full, current text and cannot drift).
Fails on: a missing copy, a byte difference, or nothing compared (0 copies => fail)."""
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "LICENSE-COMMUNITY"
# sub-package license files that MUST equal the root Community License byte-for-byte
COPIES = ["vrchat-package/LICENSE.md", "unity-sdf-universe/LICENSE"]
def main() -> int:
    if not MASTER.is_file():
        print(f"error: {MASTER.name} missing", file=sys.stderr); return 1
    master = MASTER.read_bytes()
    errs, compared = [], 0
    for rel in COPIES:
        p = ROOT / rel
        if not p.is_file():
            errs.append(f"{rel}: missing (must ship a byte-identical copy of LICENSE-COMMUNITY)"); continue
        compared += 1
        if p.read_bytes() != master:
            errs.append(f"{rel}: differs from LICENSE-COMMUNITY (byte-identical copy required)")
    if compared == 0:
        errs.append("0 license copies compared (check is vacuous — list is empty or all missing)")
    for e in errs:
        print(f"error: {e}", file=sys.stderr)
    print(f"license-copy check: {compared} copy/copies compared against LICENSE-COMMUNITY", file=sys.stderr)
    return 1 if errs else 0
def self_test() -> int:
    import tempfile, subprocess
    with tempfile.TemporaryDirectory() as d:
        r = Path(d); (r/"LICENSE-COMMUNITY").write_text("X\n"); (r/"vrchat-package").mkdir(); (r/"unity-sdf-universe").mkdir()
        me = Path(__file__)
        def run(copies_body):
            src = re.sub(r'^COPIES = \[.*\]$', copies_body, me.read_text(), count=1, flags=re.M).replace(
                'ROOT = Path(__file__).resolve().parent.parent', f'ROOT = Path(r"{d}")')
            t = r/"chk.py"; t.write_text(src)
            return subprocess.run([sys.executable, str(t)], capture_output=True).returncode
        (r/"vrchat-package/LICENSE.md").write_text("X\n")
        (r/"unity-sdf-universe/LICENSE").write_text("X\n")
        assert run('COPIES = ["vrchat-package/LICENSE.md", "unity-sdf-universe/LICENSE"]') == 0, "identical copies should pass"
        (r/"vrchat-package/LICENSE.md").write_text("X tampered\n")
        assert run('COPIES = ["vrchat-package/LICENSE.md"]') == 1, "a differing copy must fail"
        (r/"vrchat-package/LICENSE.md").unlink()
        assert run('COPIES = ["vrchat-package/LICENSE.md"]') == 1, "a missing copy must fail"
        assert run('COPIES = []') == 1, "0 copies compared must fail (not vacuously pass)"
    print("license_copy_check self-test ok"); return 0
if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main())
