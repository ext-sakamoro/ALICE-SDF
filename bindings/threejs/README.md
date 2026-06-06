# @alice-sdf/threejs

TypeScript wrapper for the ALICE-SDF WebAssembly module with Three.js helpers and optional
[`@react-three/fiber`](https://docs.pmnd.rs/react-three-fiber) components.

## Install

```bash
npm install @alice-sdf/threejs three
# optional R3F integration
npm install @react-three/fiber react react-dom
```

You also need the ALICE-SDF wasm bundle, generated from the Rust core:

```bash
cd /path/to/ALICE-SDF
cargo install wasm-pack  # if not already
wasm-pack build --target web --release \
  --no-default-features --features wasm -- --target-dir pkg-out
```

Resulting files (`alice_sdf.js`, `alice_sdf_bg.wasm`, `alice_sdf.d.ts`) should be served from
your app and the JS path passed to `AliceSDF.load(...)`.

## Usage (plain Three.js)

```ts
import * as THREE from "three";
import { AliceSDF } from "@alice-sdf/threejs";

const sdf = await AliceSDF.load("/alice_sdf.js");
console.log("ALICE-SDF version:", sdf.version()); // "1.6.0"

const d = sdf.sdfSphere([1, 0, 0], [0, 0, 0], 1.0);   // ≈ 0
const blended = sdf.opSmoothUnion(0.5, 0.6, 0.1);     // < 0.5

// Render an SDF slice into a Three.js DataTexture
const tex = await sdf.createSliceTexture(512, 512, [0, 0, 0], 1.0, 2.5);
const plane = new THREE.Mesh(
  new THREE.PlaneGeometry(2, 2),
  new THREE.MeshBasicMaterial({ map: tex }),
);
scene.add(plane);
```

## Usage (React Three Fiber)

```tsx
import { Canvas } from "@react-three/fiber";
import { AliceSDFSlicePlane } from "@alice-sdf/threejs/r3f";

function App() {
  return (
    <Canvas>
      <AliceSDFSlicePlane
        wasmUrl="/alice_sdf.js"
        center={[0, 0, 0]}
        radius={1.0}
        halfRange={2.5}
        resolution={256}
      />
    </Canvas>
  );
}
```

## Usage (WebXR raymarching)

```ts
import { AliceSDF, type Vec3 } from "@alice-sdf/threejs";

const sdf = await AliceSDF.load("/alice_sdf.js");

// XRSession frame callback
function onXRFrame(_t: number, frame: XRFrame) {
  const session = frame.session;
  for (const source of session.inputSources) {
    if (!source.gripSpace) continue;
    const pose = frame.getPose(source.gripSpace, referenceSpace!);
    if (!pose) continue;
    const { position, orientation } = pose.transform;
    const origin: Vec3 = [position.x, position.y, position.z];
    // Quaternion → forward direction (-Z in local space)
    const dir = forwardFromQuaternion(orientation);
    const hit = sdf.raymarchSphere(origin, dir, [0, 1.5, -1], 0.3, 5.0);
    if (hit > 0) {
      // hand controller is pointing at the sphere
    }
  }
  session.requestAnimationFrame(onXRFrame);
}
```

## API

| Method | Description |
|--------|-------------|
| `AliceSDF.load(wasmJsUrl)` | Load + init the wasm module (async) |
| `AliceSDF.fromModule(wasm)` | Wrap an already-initialized module |
| `version()` | "1.6.0" |
| `sdfSphere(point, center, radius)` | Primitive eval |
| `sdfBox / sdfTorus / sdfCylinder / sdfPlane` | 〃 |
| `opUnion / opIntersection / opSubtraction` | Boolean ops |
| `opSmoothUnion / opSmoothIntersection / opSmoothSubtraction` | k-smooth blends |
| `renderSphereSlice(w, h, center, radius, halfRange)` | RGBA `Uint8Array` |
| `createSliceTexture(w, h, center, radius, halfRange)` | Three.js `DataTexture` |
| `raymarchSphere(origin, dir, center, radius, maxDist)` | WebXR hit test |
| `raymarchTwoSpheresSmooth(...)` | 2-sphere smooth scene hit test |
| `sphereBatch(pointsFlat, center, radius)` | Batch eval for hand-mesh vertices |

## License

MIT OR Apache-2.0
