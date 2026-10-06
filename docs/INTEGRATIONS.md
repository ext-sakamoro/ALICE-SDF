# Platform and tool integrations

Mobile, web, game-engine, DCC, XR and CAD integrations. The [README](../README.md#bindings)
lists them in one table; this page has the setup for each.

[日本語](INTEGRATIONS_JP.md)

## Mobile (iOS / Android)

ALICE-SDF ships a [UniFFI](https://mozilla.github.io/uniffi-rs/)-based mobile SDK so the Rust core can be called directly from **Swift** (iOS) and **Kotlin** (Android).

### Supported targets

| Platform | Architectures | Distribution |
|----------|---------------|--------------|
| **iOS** | `aarch64-apple-ios` (device), `aarch64-apple-ios-sim`, `x86_64-apple-ios` | `AliceSDF.xcframework` (static lib + Swift bindings) |
| **Android** | `arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86` | `libuniffi_alice_sdf.so` + Kotlin bindings |

### Verified on real devices (2026-06-06)

| Platform | Hardware | Result |
|----------|----------|--------|
| iOS | iPhone 17 Pro Simulator (iOS 26.0, Xcode 26.2) | ✅ Demo app boots, renders 2D SDF slice ([screenshot](../mobile/samples/ios-swiftui/screenshots/AliceSDF-demo.png)) |
| Android | Pixel 6 emulator (Android 14 / API 34, arm64-v8a) | ✅ Demo app boots, renders 2D SDF slice ([screenshot](../mobile/samples/android-compose/screenshots/AliceSDF-android-demo.png)) |

Both platforms produced identical numerical output (`sphere d = 0.2806`, `smooth_union(k=0.3) = 0.2056`) — proving binary parity of the Rust core across Apple Silicon and Android ARM.

### Quick start (Swift)

```swift
import AliceSDF

let d = sdfSphere(
    point:  Vec3(x: 1, y: 0, z: 0),
    center: Vec3(x: 0, y: 0, z: 0),
    radius: 1.0
)
// d ≈ 0 (point on sphere surface)

let blended = opSmoothUnion(a: 0.5, b: 0.6, k: 0.1)
// blended < 0.5 (smooth union pulls below min)
```

### Quick start (Kotlin)

```kotlin
import uniffi.alice_sdf.*

val d = sdfSphere(
    point  = Vec3(1f, 0f, 0f),
    center = Vec3(0f, 0f, 0f),
    radius = 1.0f
)
// d ≈ 0

val blended = opSmoothUnion(a = 0.5f, b = 0.6f, k = 0.1f)
```

### Build the SDK

```bash
# iOS XCFramework (device + simulator)
cd mobile/packaging/ios && ./build-xcframework.sh

# Android .so + Kotlin bindings (4 ABI)
export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk
cd mobile/packaging/android && ./build-aar.sh
```

Sample apps and the full integration guide live in [`mobile/`](../mobile/).

---

## Web (WebAssembly), VFX (OpenVDB), Bevy Engine

### `wasm` feature — WebAssembly bindings

Browser-side SDF evaluation + slice rendering via [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/).

```bash
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
```

JavaScript usage:

```js
import init, { sdf_sphere, op_smooth_union, render_sphere_slice_rgba } from './alice_sdf.js';
await init();
const d = sdf_sphere(1, 0, 0, /*center*/ 0, 0, 0, /*radius*/ 1.0);  // ≈ 0
const rgba = render_sphere_slice_rgba(256, 256, 0, 0, 0, 1.0, 2.5);  // Uint8Array for canvas
```

### `openvdb` feature — OpenVDB Float Grid I/O

Bake SDF trees into voxel grids for DCC tools (Houdini, Maya, Nuke, Blender).

```rust
use alice_sdf::io::vdb::{bake_to_vdb, load_dense_grid_from_vdb};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
let bytes = bake_to_vdb(&node, (-2.0, 2.0), 64).unwrap();
std::fs::write("sphere.vdb", &bytes).unwrap();
```

Backed by [`vdb-rs`](https://crates.io/crates/vdb-rs) 0.6 (pure Rust). Currently writes a compact custom container (`ALICEVDB1`); full OpenVDB binary parity is planned as `vdb-rs` exposes its write API.

### `alice-sdf-bevy` — Bevy 0.18 plugin

Drop-in plugin that turns `SdfShape` components into renderable `Mesh3d` assets automatically.

```rust
use bevy::prelude::*;
use alice_sdf_bevy::{AliceSdfPlugin, SdfShape};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(AliceSdfPlugin)
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(SdfShape::Sphere { radius: 1.0 });
        })
        .run();
}
```

See `bindings/bevy/alice-sdf-bevy/examples/sphere_demo.rs` for a full 3-shape demo with camera + light.

### 3D Gaussian Splatting (`.splat`)

Convert SDF surfaces into Inria 3DGS-compatible `.splat` files (32 bytes/splat: position + scale + RGBA + compressed quaternion). Drop-and-drop into any WebGL viewer (gsplat.tech / SuperSplat / antimatter15/splat).

```rust
use alice_sdf::io::splat::{sdf_to_splats, save_splat, SplatConfig};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
let cfg = SplatConfig { bounds: (-2.0, 2.0), resolution: 64, base_color: [220, 220, 240, 255] };
let splats = sdf_to_splats(&node, &cfg);
save_splat("sphere.splat", &splats).unwrap();
```

### DCC integrations are reference integrations

The Blender / Houdini / Maya / Nuke / Cinema 4D plugins below are **reference integrations** (200–700 lines each): they show the `.asdf` load path and a few primitive generators inside each host, and are the starting point for a production plugin rather than one. The Bevy (`bindings/bevy/`), Three.js (`bindings/threejs/`) and OpenXR bindings are the fuller integrations and are compiled / type-checked in CI.

### Blender Add-on (`bindings/blender/`)

Blender 4.0+ add-on. Imports `.asdf` directly and adds an "ALICE-SDF" N-panel with sphere/box/torus generators. Requires the `alice_sdf` Python module (built via `cargo build --release --features python`).

Install: zip the `alice_sdf_blender/` folder and load via `Edit > Preferences > Add-ons > Install...`.

### Houdini Python Plugin (`bindings/houdini/`)

SideFX Houdini 20+ Python module + Python SOP code for `.asdf` loading and primitive generation. Auto-installer detects `$HSITE` / `$HOUDINI_USER_PREF_DIR`.

```python
import alice_sdf_hou
sdf = alice_sdf_hou.sphere(1.0)
alice_sdf_hou.sdf_to_hou_geo(sdf, hou.pwd().geometry(), bounds=(-2.0, 2.0), resolution=64)
```

### MagicaVoxel `.vox` IO

Voxelize an SDF tree and save as a MagicaVoxel-compatible `.vox` file (v150 RIFF) — ideal for indie / voxel art pipelines.

```rust
use alice_sdf::io::vox::{sdf_to_vox, save_vox, VoxConfig};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
let cfg = VoxConfig { size: 64, bounds: (-1.5, 1.5), color_index: 79 };
save_vox("sphere.vox", &sdf_to_vox(&node, &cfg)).unwrap();
```

### `@alice-sdf/threejs` — Three.js / React Three Fiber wrapper

TypeScript npm package built on the `wasm` feature. Exposes `AliceSDF` class, Three.js `DataTexture` helper, optional `<AliceSDFSlicePlane>` for `@react-three/fiber`, and WebXR raymarching helpers.

```ts
import { AliceSDF } from "@alice-sdf/threejs";
const sdf = await AliceSDF.load("/alice_sdf.js");
const tex = await sdf.createSliceTexture(512, 512, [0, 0, 0], 1.0, 2.5);
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: tex })));
```

### Maya Python Plugin (`bindings/maya/`)

Autodesk Maya 2024+ Python module — adds "ALICE-SDF" top-level menu, builds polygon meshes through `MFnMesh`.

```python
import alice_sdf_maya
alice_sdf_maya.register_menu()
alice_sdf_maya.add_sphere(radius=1.5, resolution=64)
```

### Nuke Python Plugin (`bindings/nuke/`)

Foundry Nuke 15 / 16+ Python module — exports `.asdf` to volume binary and 2D RGBA slices for compositing pipelines.

```python
import alice_sdf_nuke
alice_sdf_nuke.export_asdf_as_volume("/path/to/model.asdf", out_path="/tmp/model.alicevdb")
```

### Cinema 4D Python Plugin (`bindings/cinema4d/`)

Maxon Cinema 4D 2024 / 2025 / 2026+ Python module — generates `PolygonObject` from SDF primitives and imports `.asdf` directly into the C4D scene.

```python
import alice_sdf_c4d
alice_sdf_c4d.add_sphere(radius=100.0, resolution=64)   # C4D units: cm
alice_sdf_c4d.import_asdf("/path/to/model.asdf", bounds=(-300.0, 300.0), resolution=128)
```

### CAD Interchange — STEP / IGES (FEM-style mesh export)

ALICE-SDF can tessellate any SDF tree (Marching Cubes) and emit either:

- **STEP AP214** (ISO 10303-21 ASCII, `AUTOMOTIVE_DESIGN`) — a **faceted BREP**: every triangle is an `ADVANCED_FACE` on a `PLANE`, bounded by an `EDGE_LOOP` of `ORIENTED_EDGE`/`EDGE_CURVE`, collected into a `CLOSED_SHELL` → `MANIFOLD_SOLID_BREP` → `ADVANCED_BREP_SHAPE_REPRESENTATION` with millimetre units. *Note:* the surfaces are planes, so a sphere arrives as a tessellation, not as a `SPHERICAL_SURFACE`. An independent read-back oracle (`tests/test_step_export_oracle.rs`) checks that no `#id` reference dangles, that the AP214 roots are present, that the shell is closed (every edge used twice) and that the reconstructed volume matches the analytic one within 5%. Before 2026-09-27 the writer emitted only points and loops with no shell, solid, representation or units (and referenced an undefined `#0`), which no CAD tool could open.
- **Exact boxes** — an origin-centred axis-aligned box (`SdfNode::Box3d`) skips tessellation entirely and is written as six quadrilateral `ADVANCED_FACE`s, so its dimensions are exact regardless of `resolution` (the oracle checks the volume against `w·h·d` to 1e-6). Spheres and cylinders still tessellate.
- **IGES** (Entity 134 Node + Entity 136 Finite Element) — FEM mesh entities. Intended for FEM solvers and viewers that understand these entity types. Standard CAD modeling tools usually expect Entity 144 (Trimmed Surface), which is not yet emitted.

Both files round-trip in the unit tests but real-world CAD interoperability is best verified per-tool.

```rust
use alice_sdf::io::step::{export_step, StepConfig};
use alice_sdf::io::iges::{export_iges, IgesConfig};
use alice_sdf::prelude::*;

let node = SdfNode::sphere(1.0);
export_step("sphere.step", &node, &StepConfig::default()).unwrap();
export_iges("sphere.igs",  &node, &IgesConfig::default()).unwrap();
```

### `alice-sdf-openxr` — Native VR / AR Helpers (`bindings/openxr/`)

A small Rust crate on top of the [`openxr`](https://crates.io/crates/openxr) bindings that turns ALICE-SDF distance queries into VR-friendly helpers. Works with Meta Quest standalone (Android APK), PC VR (SteamVR / Oculus PC), Microsoft Mixed Reality, and Apple Vision Pro via its OpenXR backend.

```rust
use alice_sdf_openxr::{XrPose, raymarch_sphere};
use glam::Vec3;

// inside an XRFrame callback
let head_pose: XrPose = openxr_pose.into();
let hit_dist = raymarch_sphere(head_pose, Vec3::new(0.0, 1.5, -1.0), 0.3, 5.0);
if hit_dist > 0.0 {
    // controller / head is looking at the sphere
}
```

### `AliceSDFVisionOS` — Apple Vision Pro SwiftPM Package (`mobile/swift-package-visionos/`)

A visionOS / RealityKit-flavoured Swift Package that wraps the same `AliceSDF.xcframework` used by the iOS / iPadOS / macOS targets and adds factory helpers for `ModelEntity`.

```swift
import SwiftUI
import RealityKit
import AliceSDFVisionOS

struct ImmersiveView: View {
    var body: some View {
        RealityView { content in
            let sphere = AliceSDFRealityKit.makeSphereEntity(radius: 0.1)
            sphere.position = SIMD3(0, 1.5, -1.0)
            content.add(sphere)
        }
    }
}
```

### REST API Server (`server/`)

An `axum` + `tokio` HTTP server that exposes ALICE-SDF primitive evaluation and operations over JSON — intended as the backend for cloud-served SDF UIs (e.g. `alicelaw.net/sdf-metaverse`).

```bash
cd server && cargo run --release
# → ALICE-SDF server listening on http://0.0.0.0:8787
```

```http
POST /eval
Content-Type: application/json

{ "shape": "sphere", "point": [1, 0, 0], "params": { "radius": 1.0, "center": [0, 0, 0] } }
```

Response: `{ "distance": 0.0 }`
