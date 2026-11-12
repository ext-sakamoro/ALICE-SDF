#!/usr/bin/env python3
"""Tests of scripts/hlsl_emitters.py: every HLSL-producing file is in the both-engines list.

Each case builds a small tree (src/ + .github/unreal-engines-both.txt) in a temporary
directory; the expectations follow the rule in the module doc, not the script's output.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hlsl_emitters as he  # noqa: E402


def tree(files: dict[str, str], patterns: list[str]) -> tempfile.TemporaryDirectory:
    d = tempfile.TemporaryDirectory()
    root = Path(d.name)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    lst = root / ".github" / "unreal-engines-both.txt"
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text("# header\n" + "".join(p + "\n" for p in patterns), encoding="utf-8")
    return d


def run(root: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(HERE / "hlsl_emitters.py"), root],
                       capture_output=True, text=True, check=False)
    return p.returncode, p.stdout + p.stderr


# one line per kind of HLSL production the rule names
PRODUCERS = {
    "ctor": 'let s = "float3(0.0, 1.0, 2.0)";',
    "matrix": 'let s = "float4x4(m)";',
    "asfloat": 'out.push_str("asfloat(0x7fc00000u)");',
    "asuint": 'let s = "asuint(x)";',
    "numthreads": 'let s = "[numthreads(64, 1, 1)]";',
    "cbuffer": 'let s = "cbuffer Params : register(b0)";',
    "rwbuffer": 'let s = "RWStructuredBuffer<float> o;";',
    "semantic": 'let s = "float4 main() : SV_Target";',
    "decl": 'let s = "float3 n = normalize(p);";',
    "typename": '"hlsl" => ("float", "int", "float2", "float3", "float4"),',
    "dialect": 'render(t, "hlsl")',
    "fn_suffix": "pub fn smooth_helper_hlsl() -> String { String::new() }",
    "fn_prefix": "fn hlsl_header() {}",
    "const": "pub const NOISE_HLSL: &str = \"\";",
    "static": "static BLEND_HLSL: &str = \"\";",
    "impl": "impl ShaderLang for HlslLang {}",
    "generic": "fn emit<L: ShaderLang>() {}",
    "template": "let s = render_template(&t, d);",
    "tag": "let s = tagged_shader(&body);",
    "tag_with": "let s = tagged_shader_with(&p, &body);",
    "usf": 'let f = "Corpus.usf";',
    "ush": 'let f = "Common.ush";',
}


class Detection(unittest.TestCase):
    def test_each_kind_of_producer_is_detected(self):
        for name, line in PRODUCERS.items():
            with self.subTest(kind=name):
                self.assertTrue(he.emits(f"// x\n{line}\n"), line)

    def test_a_cfg_naming_the_feature_is_not_a_producer(self):
        self.assertFalse(he.emits('#[cfg(any(feature = "glsl", feature = "hlsl"))]\npub mod x;\n'))
        self.assertFalse(he.emits('#[cfg(feature="hlsl")]\n'))

    def test_plain_rust_and_other_dialects_are_not_producers(self):
        for text in ["let v = Vec3::new(0.0, 1.0, 2.0);", 'let s = "vec3(0.0)";',
                     "// HLSL round ties away from zero", "fn lerp(a: f32) -> f32 { a }",
                     'let s = "vec3<f32>(0.0)";']:
            with self.subTest(text=text):
                self.assertFalse(he.emits(text + "\n"))


class Check(unittest.TestCase):
    def test_an_emitter_outside_the_list_fails_and_is_named(self):
        with tree({"src/a.rs": PRODUCERS["asfloat"], "src/b.rs": "fn f() {}"},
                  ["src/other.rs"]) as d:
            code, out = run(d)
            self.assertEqual(code, 1, out)
            self.assertIn("src/a.rs: produces HLSL text", out)
            self.assertNotIn("src/b.rs", out)

    def test_an_emitter_in_a_nested_directory_is_found(self):
        with tree({"src/deep/er/x.rs": PRODUCERS["generic"]}, ["src/a.rs"]) as d:
            code, out = run(d)
            self.assertEqual(code, 1, out)
            self.assertIn("src/deep/er/x.rs", out)

    def test_every_emitter_listed_passes(self):
        with tree({"src/a.rs": PRODUCERS["asfloat"], "src/hlsl/b.rs": PRODUCERS["impl"],
                   "src/c.rs": "fn f() {}"}, ["src/a.rs", "src/hlsl/**"]) as d:
            code, out = run(d)
            self.assertEqual(code, 0, out)
            self.assertIn("2 emitter file(s), 0 finding(s)", out)

    def test_no_emitter_found_fails(self):
        with tree({"src/a.rs": "fn f() {}"}, ["src/a.rs"]) as d:
            code, out = run(d)
            self.assertEqual(code, 1, out)
            self.assertIn("compared nothing", out)

    def test_an_empty_list_fails(self):
        with tree({"src/a.rs": PRODUCERS["asfloat"]}, []) as d:
            code, out = run(d)
            self.assertEqual(code, 1, out)
            # refused with its reason, not a crash (a traceback also exits 1)
            self.assertIn("hlsl_emitters: .github/unreal-engines-both.txt: no pattern", out)
            self.assertNotIn("Traceback", out)

    def test_this_repository_passes(self):
        code, out = run(str(HERE.parent))
        self.assertEqual(code, 0, out)


if __name__ == "__main__":
    unittest.main(verbosity=1)
