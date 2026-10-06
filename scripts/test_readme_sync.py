#!/usr/bin/env python3
"""Tests for scripts/readme_sync.py.

Each case builds a small crate in a temporary directory, breaks exactly one
thing, and asserts that the checker reports it. The first case runs the
checker against this repository, so a change that makes the real docs
disagree with the code fails here as well as in the CI step.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readme_sync as rs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CARGO = """[package]
name = "alice-sdf"
version = "1.5.0"
rust-version = "1.85"

[features]
default = ["std"]
std = []
ffi = ["std"]

[dependencies]
"""

LIB = """//! Crate doc.
//!
//! <!-- readme-sync: features -->
//! | Feature | Description |
//! |---------|-------------|
//! | `std` (default) | std |
//! | `ffi` | ffi |
//!
//! ```rust
//! let x = 1;
//! assert_eq!(x, 1);
//! ```

pub mod alpha;
#[cfg(feature = "std")]
pub mod beta;
"""

EXAMPLE = "```rust\nlet x = 1;\nassert_eq!(x, 1);\n```\n"

README = """# t

## Install

{example}
See [modules](docs/MODULES.md).

## Features

<!-- readme-sync: features -->
| Feature | Description |
|---------|-------------|
| `std` | std |
| `ffi` | ffi |

## MSRV

Minimum supported Rust version: **1.85** <!-- readme-sync: msrv -->
"""

MODULES = """# Modules

| Module | Summary |
|--------|---------|
| `alpha` | a |
| `beta` | b |
"""


def crate(overrides: dict[str, str] | None = None) -> str:
    files = {
        "Cargo.toml": CARGO,
        "src/lib.rs": LIB,
        "README.md": README.format(example=EXAMPLE),
        "README_JP.md": README.format(example=EXAMPLE),
        "docs/MODULES.md": MODULES,
    }
    files.update(overrides or {})
    d = tempfile.mkdtemp()
    for rel, text in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
    return d


def errors(overrides: dict[str, str] | None = None) -> list[str]:
    return rs.check(crate(overrides))[0]


class RealRepo(unittest.TestCase):
    def test_this_repository_is_in_sync(self):
        errs, counts = rs.check(ROOT)
        self.assertEqual(errs, [])
        self.assertGreater(counts["modules"], 40)
        self.assertGreater(counts["links"], 0)


class Checks(unittest.TestCase):
    def test_clean_crate_passes(self):
        errs, counts = rs.check(crate())
        self.assertEqual(errs, [])
        self.assertEqual(counts["features"], 6)
        self.assertEqual(counts["modules"], 2)

    def test_feature_added_to_cargo_but_not_to_readme(self):
        e = errors({"Cargo.toml": CARGO.replace('ffi = ["std"]', 'ffi = ["std"]\nsimd = []')})
        self.assertTrue(any("simd" in x and "missing" in x for x in e), e)

    def test_feature_removed_from_cargo_but_still_in_readme(self):
        e = errors({"Cargo.toml": CARGO.replace('ffi = ["std"]\n', "")})
        self.assertTrue(any("extra ['ffi']" in x for x in e), e)

    def test_feature_missing_from_the_lib_rs_doc_table(self):
        e = errors({"src/lib.rs": LIB.replace("//! | `ffi` | ffi |\n", "")})
        self.assertTrue(any(x.startswith("src/lib.rs: features table") and "missing ['ffi']" in x for x in e), e)

    def test_lib_rs_first_cell_may_carry_a_note(self):
        # `std` (default): the feature name is the backticked part of the first cell
        self.assertEqual(errors(), [])

    def test_features_table_marker_removed(self):
        r = README.format(example=EXAMPLE).replace("<!-- readme-sync: features -->\n", "")
        e = errors({"README_JP.md": r})
        self.assertTrue(any("README_JP.md: no `<!-- readme-sync: features -->`" in x for x in e), e)

    def test_msrv_bumped_in_cargo_only(self):
        e = errors({"Cargo.toml": CARGO.replace('rust-version = "1.85"', 'rust-version = "1.88"')})
        self.assertTrue(any("MSRV line says 1.85" in x for x in e), e)

    def test_msrv_marker_removed_compares_nothing(self):
        r = README.format(example=EXAMPLE).replace(" <!-- readme-sync: msrv -->", "")
        e = errors({"README.md": r, "README_JP.md": r})
        self.assertTrue(any("compared nothing" in x and "msrv" in x for x in e), e)

    def test_stale_dependency_version(self):
        r = README.format(example=EXAMPLE) + '\n```toml\nalice-sdf = "0.14.0-preview.6"\n```\n'
        e = errors({"README.md": r})
        self.assertTrue(any("0.14.0-preview.6" in x for x in e), e)

    def test_compatible_dependency_version_passes(self):
        r = README.format(example=EXAMPLE) + '\n```toml\nalice-sdf = { version = "1.2", features = ["ffi"] }\n```\n'
        self.assertEqual(errors({"README.md": r}), [])

    def test_caret_rules(self):
        self.assertTrue(rs.caret_ok("1", "1.5.0"))
        self.assertTrue(rs.caret_ok("1.5", "1.5.0"))
        self.assertFalse(rs.caret_ok("1.6", "1.5.0"))
        self.assertFalse(rs.caret_ok("2", "1.5.0"))
        self.assertFalse(rs.caret_ok("0.14", "1.5.0"))
        self.assertTrue(rs.caret_ok("0.14.1", "0.14.3"))
        self.assertFalse(rs.caret_ok("0.13", "0.14.0"))

    def test_readme_example_diverges_from_doctest(self):
        r = README.format(example=EXAMPLE.replace("x, 1", "x, 2"))
        e = errors({"README_JP.md": r})
        self.assertTrue(any("README_JP.md: first ```rust block" in x for x in e), e)

    def test_new_module_not_listed(self):
        e = errors({"src/lib.rs": LIB + "pub mod gamma;\n"})
        self.assertTrue(any("not listed: ['gamma']" in x for x in e), e)

    def test_removed_module_still_listed(self):
        e = errors({"src/lib.rs": LIB.replace("pub mod alpha;\n", "")})
        self.assertTrue(any("not a `pub mod`" in x and "alpha" in x for x in e), e)

    def test_module_listed_twice(self):
        e = errors({"docs/MODULES.md": MODULES + "| `alpha` | again |\n"})
        self.assertTrue(any("listed twice: ['alpha']" in x for x in e), e)

    def test_broken_relative_link(self):
        r = README.format(example=EXAMPLE) + "\n[x](examples/gone.rs)\n"
        e = errors({"README.md": r})
        self.assertTrue(any("examples/gone.rs" in x for x in e), e)

    def test_links_in_docs_resolve_relative_to_docs(self):
        m = MODULES + "\n[up](../README.md) [here](MODULES.md)\n"
        self.assertEqual(errors({"docs/MODULES.md": m}), [])

    def test_external_and_anchor_links_are_not_files(self):
        r = README.format(example=EXAMPLE) + "\n[a](https://example.org/x) [b](#install)\n"
        self.assertEqual(errors({"README.md": r}), [])

    def test_japanese_readme_missing_a_section(self):
        r = README.format(example=EXAMPLE).replace("## MSRV\n", "")
        e = errors({"README_JP.md": r})
        self.assertTrue(any("`##` sections" in x for x in e), e)

    def test_missing_modules_file(self):
        d = crate()
        os.remove(os.path.join(d, "docs/MODULES.md"))
        errs = rs.check(d)[0]
        self.assertTrue(any("docs/MODULES.md: missing" in x for x in errs), errs)
        self.assertTrue(any("compared nothing" in x and "modules" in x for x in errs), errs)


if __name__ == "__main__":
    unittest.main()
