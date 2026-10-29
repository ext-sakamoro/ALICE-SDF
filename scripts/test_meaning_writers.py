#!/usr/bin/env python3
"""Oracle of scripts/meaning_writers.py: on small trees whose answer is known, the checker passes
the meaning writers, fails each kind of raw write, fails a tree with no source or no meaning writer
site and a count that differs, and ignores a raw write inside a `#[cfg(test)]` module.

Author: Moroya Sakamoto
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import meaning_writers as mw  # noqa: E402

GOOD = """pub fn write(p: &Provenance, out: &mut Vec<u8>) {
    out.extend_from_slice(&p.slot_for(StoredForm::Vdb));
}
"""


def tree(src: str, sites: int | None) -> pathlib.Path:
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "src").mkdir()
    (root / "scripts").mkdir()
    (root / "src" / "lib.rs").write_text(src, encoding="utf-8")
    if sites is not None:
        (root / mw.COUNT_FILE).write_text(f"expected-sites {sites}\n", encoding="utf-8")
    return root


def problems(src: str, sites: int | None = 1) -> list[str]:
    return mw.check(tree(src, sites))[0]


def test_a_raw_write_fails() -> int:
    """the canary: each kind of raw write is flagged, named with its line"""
    raw = [
        "    out.extend_from_slice(&p.semantics_id);",
        "    w.write_all(&provenance.semantics_id)?;",
        "    for c in p.semantics_id.chunks_exact(4) {}",
        "    let h = Header { semantics_id: Some(provenance.semantics_id) };",
        "    let d = Doc { semantics_id: Some(provenance.hex()) };",
        "    let t = p.tagged();",
        '    writeln!(w, "// {KEY}={}", p.hex())?;',
        "    enc.add_text_chunk(crate::provenance::KEY.to_string(), p.hex());",
    ]
    for line in raw:
        src = GOOD.replace("}\n", line + "\n}\n", 1)
        found = problems(src)
        assert len(found) == 1 and "src/lib.rs:" in found[0], (line, found)
    return len(raw)


def test_no_meaning_writer_site_fails() -> int:
    """the compared-nothing test: no meaning writer site or no source file fails"""
    assert any("no meaning writer site" in p for p in problems("pub fn f() {}\n", 0))
    empty = pathlib.Path(tempfile.mkdtemp())
    (empty / "scripts").mkdir()
    (empty / mw.COUNT_FILE).write_text("expected-sites 0\n", encoding="utf-8")
    assert any("no source file" in p for p in mw.check(empty)[0])
    return 2


def main() -> int:
    ran = test_a_raw_write_fails() + test_no_meaning_writer_site_fails()
    # the meaning writer passes, and is counted
    assert problems(GOOD) == [], problems(GOOD)
    assert mw.check(tree(GOOD, 1))[2] == 1
    ran += 1
    # inside a test module a raw write is not a writer
    in_test = GOOD + "#[cfg(test)]\nmod tests {\n    fn t(p: &Provenance) { let _ = p.tagged(); }\n}\n"
    assert problems(in_test) == [], problems(in_test)
    ran += 1
    # in a comment it is not a write
    assert problems(GOOD + "// p.tagged() was the earlier form\n") == []
    ran += 1
    # a count that differs, no count file: each fails
    assert any("records 2" in p for p in problems(GOOD, 2))
    assert any("is missing" in p for p in problems(GOOD, None))
    ran += 2
    print(f"test_meaning_writers: ok ({ran} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
