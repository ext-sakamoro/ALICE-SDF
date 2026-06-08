/**
 * ALICE-SDF Three.js TypeScript wrapper
 * ======================================
 *
 * Browser-side TypeScript wrapper for the ALICE-SDF WebAssembly module
 * (built via `cargo build --target wasm32-unknown-unknown --features wasm`).
 *
 * Provides:
 * - `AliceSDF` class that lazily loads the wasm module
 * - Typed methods for primitives, operations, raymarching
 * - `createSliceTexture()` helper that produces a Three.js DataTexture from an SDF slice
 *
 * @example
 * ```ts
 * import { AliceSDF } from '@alice-sdf/threejs';
 *
 * const sdf = await AliceSDF.load('/alice_sdf.js'); // emitted by wasm-bindgen
 * const d = sdf.sdfSphere([1,0,0], [0,0,0], 1.0); // ≈ 0
 * const rgba = sdf.renderSphereSlice(256, 256, [0,0,0], 1.0, 2.5);
 * ```
 */

import type { Texture } from "three";

/**
 * Raw wasm module shape (matches the wasm-bindgen output from src/wasm.rs).
 *
 * 実際の関数名は `wasm-bindgen` のスネークケース → JavaScript はそのまま、
 * このラッパは camelCase に変換して公開する。
 */
export interface AliceSdfWasm {
  sdf_sphere: (
    px: number,
    py: number,
    pz: number,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
  ) => number;
  sdf_box: (
    px: number,
    py: number,
    pz: number,
    cx: number,
    cy: number,
    cz: number,
    hx: number,
    hy: number,
    hz: number,
  ) => number;
  sdf_torus_w: (px: number, py: number, pz: number, R: number, r: number) => number;
  sdf_cylinder_w: (
    px: number,
    py: number,
    pz: number,
    radius: number,
    half_height: number,
  ) => number;
  sdf_plane_w: (
    px: number,
    py: number,
    pz: number,
    nx: number,
    ny: number,
    nz: number,
    distance: number,
  ) => number;
  op_union: (a: number, b: number) => number;
  op_intersection: (a: number, b: number) => number;
  op_subtraction: (a: number, b: number) => number;
  op_smooth_union: (a: number, b: number, k: number) => number;
  op_smooth_intersection: (a: number, b: number, k: number) => number;
  op_smooth_subtraction: (a: number, b: number, k: number) => number;
  render_sphere_slice_rgba: (
    width: number,
    height: number,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
    half_range: number,
  ) => Uint8Array;
  raymarch_sphere: (
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
    max_dist: number,
  ) => number;
  raymarch_two_spheres_smooth: (
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    c1x: number,
    c1y: number,
    c1z: number,
    r1: number,
    c2x: number,
    c2y: number,
    c2z: number,
    r2: number,
    k: number,
    max_dist: number,
  ) => number;
  sphere_batch_flat: (
    points_xyz: Float32Array,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
  ) => Float32Array;
  alice_sdf_version: () => string;
}

export type Vec3 = [number, number, number];

/**
 * Lazy-loading typed wrapper around the ALICE-SDF wasm module.
 */
export class AliceSDF {
  private constructor(private readonly wasm: AliceSdfWasm) {}

  /**
   * Dynamically import + initialize the wasm-bindgen JS glue.
   *
   * @param wasmJsUrl - URL to the JS file emitted by wasm-bindgen (e.g. `./alice_sdf.js`)
   */
  static async load(wasmJsUrl: string): Promise<AliceSDF> {
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const mod: any = await import(/* @vite-ignore */ wasmJsUrl);
    if (typeof mod.default === "function") {
      await mod.default();
    }
    return new AliceSDF(mod as AliceSdfWasm);
  }

  /**
   * Construct from an already-loaded wasm module (e.g. when bundler handled init).
   */
  static fromModule(wasm: AliceSdfWasm): AliceSDF {
    return new AliceSDF(wasm);
  }

  /** Version of the wasm core (e.g. "1.6.0"). */
  version(): string {
    return this.wasm.alice_sdf_version();
  }

  // === Primitives ===

  sdfSphere(point: Vec3, center: Vec3, radius: number): number {
    return this.wasm.sdf_sphere(
      point[0],
      point[1],
      point[2],
      center[0],
      center[1],
      center[2],
      radius,
    );
  }

  sdfBox(point: Vec3, center: Vec3, halfExtents: Vec3): number {
    return this.wasm.sdf_box(
      point[0],
      point[1],
      point[2],
      center[0],
      center[1],
      center[2],
      halfExtents[0],
      halfExtents[1],
      halfExtents[2],
    );
  }

  sdfTorus(point: Vec3, majorRadius: number, minorRadius: number): number {
    return this.wasm.sdf_torus_w(point[0], point[1], point[2], majorRadius, minorRadius);
  }

  sdfCylinder(point: Vec3, radius: number, halfHeight: number): number {
    return this.wasm.sdf_cylinder_w(point[0], point[1], point[2], radius, halfHeight);
  }

  sdfPlane(point: Vec3, normal: Vec3, distance: number): number {
    return this.wasm.sdf_plane_w(
      point[0],
      point[1],
      point[2],
      normal[0],
      normal[1],
      normal[2],
      distance,
    );
  }

  // === Operations ===

  opUnion(a: number, b: number): number {
    return this.wasm.op_union(a, b);
  }
  opIntersection(a: number, b: number): number {
    return this.wasm.op_intersection(a, b);
  }
  opSubtraction(a: number, b: number): number {
    return this.wasm.op_subtraction(a, b);
  }
  opSmoothUnion(a: number, b: number, k: number): number {
    return this.wasm.op_smooth_union(a, b, k);
  }
  opSmoothIntersection(a: number, b: number, k: number): number {
    return this.wasm.op_smooth_intersection(a, b, k);
  }
  opSmoothSubtraction(a: number, b: number, k: number): number {
    return this.wasm.op_smooth_subtraction(a, b, k);
  }

  // === Slice rendering ===

  /** Render a 2D z=0 slice of a single sphere as an RGBA Uint8Array. */
  renderSphereSlice(
    width: number,
    height: number,
    center: Vec3,
    radius: number,
    halfRange: number,
  ): Uint8Array {
    return this.wasm.render_sphere_slice_rgba(
      width,
      height,
      center[0],
      center[1],
      center[2],
      radius,
      halfRange,
    );
  }

  /**
   * Render an SDF slice and wrap into a Three.js `DataTexture`.
   * Lazy-loads the `three` import so the helper is tree-shakable for non-three consumers.
   */
  async createSliceTexture(
    width: number,
    height: number,
    center: Vec3,
    radius: number,
    halfRange: number,
  ): Promise<Texture> {
    const three = await import("three");
    const rgba = this.renderSphereSlice(width, height, center, radius, halfRange);
    // TypeScript 5.7+ は Uint8Array<ArrayBufferLike> を `BufferSource` (ArrayBuffer-backed)
    // と互換と認めない (SharedArrayBuffer 可能性のため)。明示的に ArrayBuffer に copy する。
    const ab = new ArrayBuffer(rgba.byteLength);
    new Uint8Array(ab).set(rgba);
    const buf = new Uint8Array(ab);
    const tex = new three.DataTexture(buf, width, height, three.RGBAFormat, three.UnsignedByteType);
    tex.needsUpdate = true;
    return tex;
  }

  // === WebXR helpers ===

  /** Ray-march a sphere SDF along a ray. Returns hit distance or -1.0 if no hit. */
  raymarchSphere(
    origin: Vec3,
    dir: Vec3,
    center: Vec3,
    radius: number,
    maxDist = 100.0,
  ): number {
    return this.wasm.raymarch_sphere(
      origin[0],
      origin[1],
      origin[2],
      dir[0],
      dir[1],
      dir[2],
      center[0],
      center[1],
      center[2],
      radius,
      maxDist,
    );
  }

  /** Ray-march a smooth-union of two spheres (demo scene for VR/AR). */
  raymarchTwoSpheresSmooth(
    origin: Vec3,
    dir: Vec3,
    c1: Vec3,
    r1: number,
    c2: Vec3,
    r2: number,
    k: number,
    maxDist = 100.0,
  ): number {
    return this.wasm.raymarch_two_spheres_smooth(
      origin[0],
      origin[1],
      origin[2],
      dir[0],
      dir[1],
      dir[2],
      c1[0],
      c1[1],
      c1[2],
      r1,
      c2[0],
      c2[1],
      c2[2],
      r2,
      k,
      maxDist,
    );
  }

  /**
   * Batch sphere SDF evaluation for many query points (e.g. WebXR hand-mesh vertices).
   *
   * @param points - Flat XYZ array: [x0, y0, z0, x1, y1, z1, ...]
   */
  sphereBatch(points: Float32Array, center: Vec3, radius: number): Float32Array {
    return this.wasm.sphere_batch_flat(points, center[0], center[1], center[2], radius);
  }
}

export default AliceSDF;
