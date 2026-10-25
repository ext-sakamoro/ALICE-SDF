#!/usr/bin/env python3
"""Tests of scripts/shader_fma_guard.py: a canary per shader language, the forms that must not
count (comments, test modules, identifiers that only contain the name), the negations of a whole
multiply-add (direct and through a name) and the operand forms that are not one, and the
compared-nothing failure."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shader_fma_guard as g  # noqa: E402


def hits(src: str) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "lib.rs").write_text(src, encoding="utf-8")
        found, _, _ = g.scan(Path(d), ["lib.rs"])
        return found


class Canary(unittest.TestCase):
    # one emitted line per backend, each written the way that language spells it
    def test_wgsl(self):
        self.assertEqual(len(hits('fn e() -> String { format!("let d = -fma({a}, {b}, {c});") }')), 1)

    def test_glsl(self):
        self.assertEqual(len(hits('const S: &str = "float d = -fma(a, b, c);";')), 1)

    def test_hlsl_mad(self):
        self.assertEqual(len(hits('fn e(s: &mut String) { s.push_str("float d = -mad(a, b, c);"); }')), 1)

    def test_msl_namespaced(self):
        self.assertEqual(len(hits('const S: &str = "float d = -metal::precise::fma(a, b, c);";')), 1)

    def test_blinkscript_fmaf_with_space(self):
        self.assertEqual(len(hits('const S: &str = "float d = -fmaf (a, b, c);";')), 1)

    def test_raw_string_template(self):
        self.assertEqual(len(hits('const S: &str = r#"\nfn f() -> f32 {\n  return -fma(a, b, c);\n}\n"#;')), 1)

    def test_line_number(self):
        found = hits('// x\n\nconst S: &str = "fma(a, b, c)";\n')
        self.assertEqual(len(found), 1)
        self.assertIn("lib.rs:3:", found[0])


class NotCounted(unittest.TestCase):
    def test_comment(self):
        self.assertEqual(hits('// fma(a, b, c) is not used\n/* mad(a, b, c) */\nconst S: &str = "a * b + c";'), [])

    def test_quoted_text_inside_comments(self):
        src = '// the emitter never writes "fma(a, b, c)"\n/* nor "mad(a, b, c)" */\nconst S: &str = "a * b + c";'
        self.assertEqual(hits(src), [])

    def test_rust_call_outside_a_literal(self):
        # clippy holds Rust calls; this script reads shader text only
        self.assertEqual(hits("fn f(a: f32) -> f32 { a.mul_add(1.0, 2.0) + fma(a, a, a) }"), [])

    def test_identifier_containing_the_name(self):
        self.assertEqual(hits('const S: &str = "nomad(x) + my_fma(a) + fmadd(b) + madness(c)";'), [])

    def test_test_module(self):
        src = 'const A: &str = "x";\n#[cfg(test)]\nmod tests {\n    const B: &str = "fma(a, b, c)";\n    fn f() { let _ = "}"; }\n}\n'
        self.assertEqual(hits(src), [])

    def test_code_after_a_test_module_is_read(self):
        src = '#[cfg(test)]\nmod tests {\n    fn f() {}\n}\nconst B: &str = "fma(a, b, c)";\n'
        self.assertEqual(len(hits(src)), 1)

    def test_char_literal_and_lifetime(self):
        src = "fn f<'a>(s: &'a str) -> char { let _ = '\"'; 'x' }\nconst B: &str = \"fma(a, b, c)\";\n"
        self.assertEqual(len(hits(src)), 1)

    def test_escaped_quote(self):
        self.assertEqual(len(hits('const S: &str = "say \\"hi\\" then fma(a, b, c)";')), 1)


class NegatedMultiplyAdd(unittest.TestCase):
    # a negation of a whole multiply-add: after a driver contracts the sum, the outer fneg of an fma
    def test_parenthesised(self):
        self.assertEqual(len(hits('const S: &str = "let d = -(p.x * p.y + p.z);";')), 1)

    def test_zero_minus_and_minus_one_times(self):
        self.assertEqual(len(hits('const S: &str = "float d = 0.0 - (a * b + c);";')), 1)
        self.assertEqual(len(hits('const S: &str = "float d = -1.0 * (a * b - c);";')), 1)

    def test_through_a_name(self):
        src = 'const S: &str = r#"\nlet p = ky - kx * kx;\nlet z = sqrt(-p);\n"#;'
        self.assertEqual(len(hits(src)), 1)

    def test_through_a_typed_name(self):
        self.assertEqual(len(hits('const S: &str = "precise float p = ky - kx * kx; float z = sqrt(-p);";')), 1)

    def test_operand_forms_do_not_count(self):
        for text in ["let d = -p.x * p.y + p.z;", "float d = c - a * b;", "let z = sqrt(kx * kx - ky);",
                     "float t = a * b + c; return -t * 2.0;", "x = y - (a * b + c);", "float d = -(a * b);",
                     "float e = -(x + 1e-5);", "float v = clamp(q, -1.0, 1.0);", "float s = -sin(a * b + c);"]:
            self.assertEqual(hits(f'const S: &str = "{text}";'), [], text)

    def test_a_reassigned_name_is_judged_by_its_last_assignment(self):
        src = 'const S: &str = "float h = q * q + 4.0 * p3; h = sqrt(h); vec2 x = vec2(h, -h);";'
        self.assertEqual(hits(src), [])
        src = 'const S: &str = "float h = sqrt(q); h = q * q + p; float y = -h;";'
        self.assertEqual(len(hits(src)), 1)


class ComparedNothing(unittest.TestCase):
    def _repo(self, d: Path, files: dict[str, str]) -> None:
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        for rel, text in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(text, encoding="utf-8")
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)

    def test_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            self._repo(Path(d), {"README.md": "x"})
            self.assertEqual(g.main(["--root", d]), 2)

    def test_no_literal(self):
        with tempfile.TemporaryDirectory() as d:
            self._repo(Path(d), {"src/lib.rs": "pub fn f() {}\n"})
            self.assertEqual(g.main(["--root", d]), 2)

    def test_a_shipped_shader_file_is_read(self):
        # a whole .cginc / .ush file, at a path with a space, as the shipped samples have
        with tempfile.TemporaryDirectory() as d:
            self._repo(Path(d), {"src/lib.rs": 'pub const S: &str = "a * b + c";\n',
                                 "pkg/Samples~/SDF Gallery/x.cginc": "float p = ky - kx * kx;\nfloat z = sqrt(kx * kx - ky);\n"})
            self.assertEqual(g.main(["--root", d]), 0)
            (Path(d) / "pkg/Samples~/SDF Gallery/x.cginc").write_text(
                "float p = ky - kx * kx;\nfloat z = sqrt(-p);\n", encoding="utf-8")
            subprocess.run(["git", "-C", d, "add", "-A"], check=True)
            self.assertEqual(g.main(["--root", d]), 1)

    def test_hit_fails_and_clean_passes(self):
        with tempfile.TemporaryDirectory() as d:
            self._repo(Path(d), {"src/lib.rs": 'pub const S: &str = "a * b + c";\n',
                                 "src/wgsl/t.rs": 'pub const T: &str = "x";\n'})
            self.assertEqual(g.main(["--root", d]), 0)
            (Path(d) / "src/wgsl/t.rs").write_text('pub const T: &str = "-fma(a, b, c)";\n', encoding="utf-8")
            subprocess.run(["git", "-C", d, "add", "-A"], check=True)
            self.assertEqual(g.main(["--root", d]), 1)


if __name__ == "__main__":
    unittest.main()
