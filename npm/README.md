# @alice-sdf/wasm

WebAssembly bindings for [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) — Mathematical 3D shape compression via Signed Distance Functions.

> "Don't send polygons. Send the law of shapes."

The package is built with `wasm-pack` from `examples/wasm-demo` (`npm run build`). Its typings are the ones `wasm-pack` generates (`dist/alice_sdf.d.ts`), so this page lists only what that crate exports.

## Install

```bash
npm install @alice-sdf/wasm
```

## Quick start

```javascript
import init, { SdfEvaluator, SdfType } from '@alice-sdf/wasm';

await init();

// A sphere of radius 1, moved to x = 0.5
const sphere = new SdfEvaluator(SdfType.Sphere);
sphere.set_params(1.0, 0.0, 0.0, 0.0);
sphere.set_translation(0.5, 0.0, 0.0);

// Signed distance at one point, and at many (x, y, z, x, y, z, ...)
const d = sphere.eval(0.0, 0.0, 0.0);
const ds = sphere.eval_batch(new Float32Array([0, 0, 0, 2, 0, 0]));
```

## API

### `SdfType`

`Sphere`, `Box`, `Cylinder`, `Torus`.

### `SdfEvaluator`

| Member | Meaning |
|---|---|
| `new SdfEvaluator(type)` | an evaluator for one shape of `SdfType`, parameters `(1, 1, 1, 0)` |
| `set_params(p0, p1, p2, p3)` | Sphere: radius `p0`; Box: half extents `p0, p1, p2`; Cylinder: radius `p0`, half height `p1`; Torus: major radius `p0`, minor radius `p1` |
| `set_translation(x, y, z)` | moves the shape |
| `eval(x, y, z)` | the signed distance at a point |
| `eval_batch(points)` | the signed distances at `points` (a `Float32Array` of x, y, z triples) |

### `Raymarcher`

```javascript
import init, { SdfEvaluator, SdfType, Raymarcher } from '@alice-sdf/wasm';

await init();
const shape = new SdfEvaluator(SdfType.Torus);
shape.set_params(1.0, 0.3, 0.0, 0.0);
const rm = new Raymarcher(256, 256);
rm.set_camera(0.0, 0.0, 4.0);
const rgba = rm.render(shape); // RGBA bytes, rm.get_width() × rm.get_height()
```

### `SoABuffer`

Points in separate x, y, z arrays in WebAssembly memory, for uploading to a GPU buffer without a copy.

```javascript
import init, { SoABuffer } from '@alice-sdf/wasm';

const wasm = await init();
const soa = new SoABuffer(10000);
soa.set_point(0, 1.0, 0.0, 0.0);
soa.eval_sphere(1.0);
const d0 = soa.get_distance(0);
// views over the arrays (valid until the memory grows)
const xs = new Float32Array(wasm.memory.buffer, soa.x_ptr(), soa.len());
```

Members: `set_point`, `set_from_aos`, `translate_all`, `len`, `byte_length`, `x_ptr`, `y_ptr`, `z_ptr`, `distances_ptr`, `distances_mut_ptr`, `eval_sphere`, `eval_box`, `get_distance`.

### Functions

| Function | Meaning |
|---|---|
| `generate_instanced_wgsl(type)` | WGSL of a compute shader that evaluates one shape of `SdfType` against many instance positions in SoA layout |
| `benchmark_eval(iterations)` | milliseconds to evaluate a sphere `iterations` times |

## License

`LicenseRef-Proprietary`. The text is [LICENSE-Proprietary](LICENSE-Proprietary).

Enquiries: https://extoria.co.jp/
