#!/usr/bin/env python3
"""Tests for scripts/npm_readme_symbols.py: each runs the script on a small crate source and README
and checks the exit status and the report. One planted name that the crate does not export must
fail the check, in each place the check reads.

usage: python3 scripts/test_npm_readme_symbols.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "npm_readme_symbols.py"

LIB = '''
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen(start)]
pub fn init() {}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum Shape {
    Ball = 0,
    Cube = 1,
}

#[wasm_bindgen]
pub struct Evaluator {
    s: Shape,
}

#[wasm_bindgen]
impl Evaluator {
    #[wasm_bindgen(constructor)]
    pub fn new(s: Shape) -> Evaluator {
        Evaluator { s }
    }

    pub fn unit() -> Evaluator {
        Evaluator { s: Shape::Ball }
    }

    pub fn eval(&self, x: f32) -> f32 {
        x
    }
}

/// ```js
/// const e = new Evaluator(Shape.Ball);
/// const d = e.eval(1.0);
/// ```
#[wasm_bindgen]
pub fn shader(s: Shape) -> String {
    String::new()
}
'''

README = '''# pkg

```javascript
import init, { Evaluator, Shape, shader } from 'pkg';

await init();
const e = new Evaluator(Shape.Cube);
const u = Evaluator.unit();
const d = e.eval(0.5) + u.eval(0.5);
const view = new Float32Array(4);
const text = shader(Shape.Ball);
```

## API

### `Shape`

`Ball`, `Cube`.

### `Evaluator`

| Member | Meaning |
|---|---|
| `new Evaluator(s)` | an evaluator for `s` |
| `eval(x)` | the distance at `x` |

Members: `eval`.

### Functions

| Function | Meaning |
|---|---|
| `shader(s)` | the shader text of `s` |

## License
'''


def run(lib: str, readme: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as d:
        lp, rp = Path(d) / "lib.rs", Path(d) / "README.md"
        lp.write_text(lib, encoding="utf-8")
        rp.write_text(readme, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--readme", str(rp), "--lib", str(lp)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )


def planted(text: str, old: str, new: str) -> str:
    assert text.count(old) >= 1, old
    return text.replace(old, new, 1)


class NpmReadmeSymbols(unittest.TestCase):
    def assert_fails(self, proc: subprocess.CompletedProcess[str], needle: str) -> None:
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn(needle, proc.stderr)

    def test_consistent_tree_passes_and_counts(self) -> None:
        p = run(LIB, README)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        # README code: 3 imports + the default import + 1 new (Float32Array is not counted)
        # + 2 variants + 1 static + 2 method calls + 1 free call = 11; the doc example: 1 new
        # + 1 variant + 1 method = 3; the API section: 2 variants + 1 constructor + 1 table
        # method + 1 listed member + 1 function = 6
        self.assertIn("20 names compared against 3 exports", p.stdout)

    def test_planted_import_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "shader }", "shader, to_mesh }")), "`to_mesh`")

    def test_planted_method_call_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "e.eval(0.5)", "e.to_wgsl()")), "`e.to_wgsl(`")

    def test_planted_static_call_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "Evaluator.unit()", "Evaluator.sphere(1.0)")), "`Evaluator.sphere(`")

    def test_method_on_a_variable_bound_by_a_static_is_checked(self) -> None:
        self.assert_fails(run(LIB, planted(README, "u.eval(0.5)", "u.subtract(e)")), "`u.subtract(`")

    def test_planted_class_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "new Evaluator(Shape.Cube)", "new Mesher(Shape.Cube)")), "`new Mesher(`")

    def test_planted_variant_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "shader(Shape.Ball)", "shader(Shape.Torus)")), "`Shape.Torus`")

    def test_planted_free_function_call_fails(self) -> None:
        r = planted(README, "shader }", "shader, compile }")
        self.assert_fails(run(LIB, planted(r, "shader(Shape.Ball)", "compile(Shape.Ball)")), "calls `compile(`")

    def test_start_function_is_not_an_export(self) -> None:
        # `#[wasm_bindgen(start)] pub fn init` runs on load: importing it by name is an error
        self.assert_fails(run(LIB, planted(README, "shader }", "shader, init }")), "imports `init`")

    def test_constructor_is_not_a_static(self) -> None:
        self.assert_fails(run(LIB, planted(README, "Evaluator.unit()", "Evaluator.new(Shape.Ball)")), "`Evaluator.new(`")

    def test_planted_doc_example_method_fails(self) -> None:
        self.assert_fails(run(planted(LIB, "e.eval(1.0)", "e.x_buffer()"), README), "`e.x_buffer(`")

    def test_planted_api_table_method_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "| `eval(x)` |", "| `to_mesh(n)` |")), "`to_mesh(n)` is not a method")

    def test_planted_members_list_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "Members: `eval`.", "Members: `eval`, `x_buffer`.")), "`x_buffer` is not a method")

    def test_planted_api_variant_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "`Ball`, `Cube`.", "`Ball`, `Cube`, `Torus`.")), "`Torus` is not a variant")

    def test_planted_api_function_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "| `shader(s)` |", "| `to_glsl(s)` |")), "`to_glsl(s)` is not an exported function")

    def test_heading_naming_nothing_fails(self) -> None:
        self.assert_fails(run(LIB, planted(README, "### Functions", "### `Mesher`\n\nA mesher.\n\n### Functions")), "heading `Mesher`")

    def test_meaning_column_arguments_are_not_names(self) -> None:
        # `s` and `x` in the meaning column are arguments; they must not be read as methods
        p = run(LIB, README)
        self.assertNotIn("`s`", p.stderr)
        self.assertNotIn("`x`", p.stderr)

    def test_nothing_compared_fails(self) -> None:
        no_doc = LIB.split("/// ```js")[0] + "#[wasm_bindgen]\npub fn shader(s: Shape) -> String { String::new() }\n"
        self.assert_fails(run(no_doc, "# pkg\n\nNo code.\n\n## API\n\n## License\n"), "0 names compared")

    def test_no_exports_fails(self) -> None:
        self.assert_fails(run("// nothing\n", README), "no #[wasm_bindgen] exports")


if __name__ == "__main__":
    unittest.main()
