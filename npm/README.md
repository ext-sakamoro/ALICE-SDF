# @alice-sdf/wasm

WebAssembly bindings for [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) — Mathematical 3D shape compression via Signed Distance Functions.

> "Don't send polygons. Send the law of shapes."

## Install

```bash
npm install @alice-sdf/wasm
```

## Quick Start

```javascript
import init, { SdfEvaluator } from '@alice-sdf/wasm';

await init();

// Create shapes
const sphere = SdfEvaluator.sphere(1.0);
const box = SdfEvaluator.box3d(0.5, 0.5, 0.5);
const result = sphere.subtract(box);

// Evaluate distance
const dist = result.eval(0.5, 0.0, 0.0);

// Convert to mesh
const { vertices, indices } = result.to_mesh(-2, -2, -2, 2, 2, 2, 64);

// Generate GPU shader
const wgsl = result.to_wgsl();
```

## Features

- 53 SDF primitives (Sphere, Box, Torus, Cylinder, etc.)
- CSG operations (union, intersection, subtraction, smooth variants)
- Marching Cubes mesh generation
- WGSL/GLSL shader transpilation
- SIMD-accelerated batch evaluation
- WebGPU compute shader support

## Browser Compatibility

| Browser | WebGPU | Fallback |
|---------|--------|----------|
| Chrome 113+ | Yes | Canvas2D |
| Edge 113+ | Yes | Canvas2D |
| Safari 18+ | Yes | Canvas2D |
| Firefox | Flag | Canvas2D |

## License

MIT - Moroya Sakamoto
