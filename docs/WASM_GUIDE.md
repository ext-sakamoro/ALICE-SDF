# ALICE-SDF WebAssembly Guide

Deploy ALICE-SDF in the browser with WebAssembly and WebGPU.

## Table of Contents

- [Overview](#overview)
- [Building](#building)
- [Browser Compatibility](#browser-compatibility)
- [JavaScript API](#javascript-api)
- [WebGPU Integration](#webgpu-integration)
- [Canvas2D Fallback](#canvas2d-fallback)
- [Deployment](#deployment)
- [Performance](#performance)
- [Examples](#examples)

---

## Overview

ALICE-SDF compiles to WebAssembly, enabling real-time SDF evaluation in the browser. Two rendering paths are available:

| Path | Technology | Performance | Compatibility |
|------|-----------|-------------|---------------|
| **WebGPU** | GPU compute shaders | Excellent | Chrome 113+, Edge 113+, Safari 18+ |
| **Canvas2D** | CPU raymarching | Good | All browsers |

---

## Building

### Prerequisites

```bash
# Install wasm-pack
cargo install wasm-pack

# Or via curl
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### Build the WASM Module

```bash
cd ALICE-SDF/examples/wasm-demo

# Build for web target
wasm-pack build --target web

# Build with optimized size
wasm-pack build --target web --release
```

### Output Structure

```
examples/wasm-demo/
  pkg/
    alice_sdf_wasm.js       # JavaScript bindings
    alice_sdf_wasm_bg.wasm  # WebAssembly binary
    alice_sdf_wasm.d.ts     # TypeScript definitions
  index.html                # Demo page
  src/
    lib.rs                  # Rust WASM bindings
```

### Serve Locally

```bash
# Python
python3 -m http.server 8080

# Node.js
npx serve .

# Open http://localhost:8080
```

> **Note**: WebAssembly requires HTTP serving. `file://` URLs will not work due to CORS restrictions.

---

## Browser Compatibility

| Browser | WebGPU | WASM | Canvas2D | Notes |
|---------|--------|------|----------|-------|
| Chrome 113+ | Yes | Yes | Yes | Full support |
| Edge 113+ | Yes | Yes | Yes | Full support |
| Firefox 120+ | No | Yes | Yes | WebGPU behind flag `dom.webgpu.enabled` |
| Safari 18+ | Yes | Yes | Yes | WebGPU from macOS Sonoma / iOS 18 |
| Chrome Android | Yes | Yes | Yes | Chrome 121+ |
| Safari iOS 18+ | Yes | Yes | Yes | WebGPU supported |
| Older browsers | No | Yes* | Yes | WASM supported in all modern browsers |

*WebAssembly is supported in 96%+ of browsers globally.

---

## JavaScript API

### Initialize

```html
<script type="module">
import init, { SdfEvaluator } from './pkg/alice_sdf_wasm.js';

async function main() {
    await init();

    // Create an SDF evaluator
    const evaluator = new SdfEvaluator("sphere");
    evaluator.set_params(1.0, 0.0);

    // Evaluate at a point
    const distance = evaluator.eval(0.5, 0.0, 0.0);
    console.log(`Distance: ${distance}`);
}

main();
</script>
```

### Available Shape Types

```javascript
// Built-in shapes
const sphere = new SdfEvaluator("sphere");       // params: radius
const box = new SdfEvaluator("box");             // params: hx, hy, hz
const torus = new SdfEvaluator("torus");         // params: major_r, minor_r
const cylinder = new SdfEvaluator("cylinder");   // params: radius, height
```

### Transform and Modify

```javascript
const evaluator = new SdfEvaluator("sphere");
evaluator.set_params(1.0, 0.0);

// Set translation
evaluator.set_translation(0.0, 1.0, 0.0);
```

### Batch Evaluation

```javascript
// Evaluate multiple points
const points = new Float32Array([
    0.0, 0.0, 0.0,  // point 0
    1.0, 0.0, 0.0,  // point 1
    2.0, 0.0, 0.0,  // point 2
]);

const distances = evaluator.eval_batch(points);
console.log(distances);  // Float32Array [-1.0, 0.0, 1.0]
```

---

## WebGPU Integration

### Check WebGPU Support

```javascript
async function checkWebGPU() {
    if (!navigator.gpu) {
        console.log("WebGPU not supported, falling back to Canvas2D");
        return null;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.log("No GPU adapter found");
        return null;
    }

    const device = await adapter.requestDevice();
    console.log("WebGPU ready:", adapter.name);
    return device;
}
```

### GPU Compute Shader Evaluation (Planned)

> **Note**: `GpuSdfEvaluator` is planned but not yet exposed to JavaScript. The WebGPU compute infrastructure exists in the Rust crate (`wgpu` dependency in `examples/wasm-demo/Cargo.toml`) but the JS binding is not yet implemented. For now, use the CPU-based `SdfEvaluator` for batch evaluation.

```javascript
// Future API (not yet available):
// import init, { GpuSdfEvaluator } from './pkg/alice_sdf_wasm.js';
// const gpuEval = new GpuSdfEvaluator(device, "sphere", 1.0);
// const distances = await gpuEval.eval_batch(points);
```

### Real-time Raymarching (Canvas)

```javascript
import init, { Raymarcher } from './pkg/alice_sdf_wasm.js';

async function setupRaymarcher() {
    await init();

    const canvas = document.getElementById('sdf-canvas');
    const ctx = canvas.getContext('2d');

    const raymarcher = new Raymarcher(canvas.width, canvas.height);

    function render() {
        const time = performance.now() / 1000.0;
        raymarcher.set_camera_position(
            Math.sin(time) * 3.0,
            1.5,
            Math.cos(time) * 3.0
        );

        const pixels = raymarcher.render();
        const imageData = new ImageData(
            new Uint8ClampedArray(pixels),
            canvas.width,
            canvas.height
        );
        ctx.putImageData(imageData, 0, 0);

        requestAnimationFrame(render);
    }

    render();
}
```

---

## Canvas2D Fallback

For browsers without WebGPU, use the `Raymarcher` class which performs CPU-based raymarching:

```javascript
import init, { Raymarcher } from './pkg/alice_sdf_wasm.js';

async function cpuRender() {
    await init();

    const canvas = document.getElementById('sdf-canvas');
    const ctx = canvas.getContext('2d');

    // Use Raymarcher for CPU rendering (same class as above)
    const raymarcher = new Raymarcher(canvas.width, canvas.height);

    function render() {
        const pixels = raymarcher.render();
        const imageData = new ImageData(
            new Uint8ClampedArray(pixels),
            canvas.width,
            canvas.height
        );
        ctx.putImageData(imageData, 0, 0);
        requestAnimationFrame(render);
    }

    render();
}
```

---

## Deployment

### Static Hosting

The WASM demo is a static site. Deploy to any static host:

```bash
# Build production
cd examples/wasm-demo
wasm-pack build --target web --release

# Deploy
cp index.html pkg/ dist/
# Upload dist/ to your hosting provider
```

### Hosting Requirements

- **MIME Type**: `.wasm` files must be served as `application/wasm`
- **CORS**: If loading cross-origin, set appropriate headers
- **HTTPS**: WebGPU requires secure context (HTTPS or localhost)

### Common Hosting Providers

| Provider | Config | Notes |
|----------|--------|-------|
| **Vercel** | No config needed | Auto-detects WASM MIME type |
| **Netlify** | No config needed | WASM supported out of the box |
| **GitHub Pages** | No config needed | WASM supported |
| **Cloudflare Pages** | No config needed | Edge-hosted |
| **Apache** | Add `AddType application/wasm .wasm` to `.htaccess` | Manual config |
| **Nginx** | Add `types { application/wasm wasm; }` | Manual config |

### CDN Optimization

```html
<!-- Preload WASM binary for faster startup -->
<link rel="preload" href="./pkg/alice_sdf_wasm_bg.wasm"
      as="fetch" type="application/wasm" crossorigin>
```

---

## Performance

### WASM vs Native

| Operation | Native (Rust) | WASM (Chrome) | Ratio |
|-----------|---------------|---------------|-------|
| Single eval | 6 ns | ~15 ns | 2.5x slower |
| Batch 1M | 3.3 ms | ~10 ms | 3x slower |
| Raymarching 512x512 | 5 ms | ~15 ms | 3x slower |

### WebGPU vs CPU (WASM)

| Points | CPU (WASM) | WebGPU | Speedup |
|--------|-----------|--------|---------|
| 10K | 0.3 ms | 2 ms | CPU faster (transfer overhead) |
| 100K | 3 ms | 3 ms | Equal |
| 1M | 30 ms | 5 ms | 6x GPU |
| 10M | 300 ms | 20 ms | 15x GPU |

### Tips

1. **Use WebGPU for large batches** (>100K points)
2. **Use CPU for small batches** (<100K points)
3. **Reduce resolution** for real-time rendering (256x256 for CPU, 1080p for GPU)
4. **Use `requestAnimationFrame`** for smooth rendering
5. **Profile with DevTools** Performance tab

---

## Examples

### Interactive SDF Editor

```html
<!DOCTYPE html>
<html>
<head>
    <title>ALICE-SDF Editor</title>
    <style>
        body { margin: 0; background: #1a1a2e; color: #fff; font-family: monospace; }
        canvas { display: block; margin: 20px auto; border: 1px solid #333; }
        .controls { text-align: center; padding: 10px; }
        input[type="range"] { width: 200px; }
    </style>
</head>
<body>
    <div class="controls">
        <label>Radius: <input type="range" id="radius" min="0.1" max="2.0" step="0.1" value="1.0"></label>
        <label>Smoothing: <input type="range" id="smooth" min="0.0" max="1.0" step="0.05" value="0.2"></label>
    </div>
    <canvas id="sdf-canvas" width="512" height="512"></canvas>

    <script type="module">
    import init, { SdfEvaluator } from './pkg/alice_sdf_wasm.js';

    async function main() {
        await init();

        const canvas = document.getElementById('sdf-canvas');
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;

        function render() {
            const radius = parseFloat(document.getElementById('radius').value);
            const evaluator = new SdfEvaluator("sphere");
            evaluator.set_params(radius, 0.0);

            const imageData = ctx.createImageData(W, H);
            const data = imageData.data;

            for (let y = 0; y < H; y++) {
                for (let x = 0; x < W; x++) {
                    const px = (x / W - 0.5) * 4.0;
                    const py = (y / H - 0.5) * 4.0;
                    const d = evaluator.eval(px, py, 0.0);

                    const idx = (y * W + x) * 4;
                    if (d < 0) {
                        data[idx] = 100; data[idx+1] = 150; data[idx+2] = 255;
                    } else {
                        const v = Math.max(0, 255 - d * 200);
                        data[idx] = v * 0.3; data[idx+1] = v * 0.3; data[idx+2] = v * 0.5;
                    }
                    data[idx+3] = 255;
                }
            }

            ctx.putImageData(imageData, 0, 0);
        }

        document.getElementById('radius').addEventListener('input', render);
        document.getElementById('smooth').addEventListener('input', render);
        render();
    }

    main();
    </script>
</body>
</html>
```

---

## Related Documentation

- [QUICKSTART](QUICKSTART.md) - Getting started
- [API Reference](API_REFERENCE.md) - Complete API
- [Architecture](ARCHITECTURE.md) - Technical design

---

Author: Moroya Sakamoto
