/**
 * ALICE-SDF WebAssembly Bindings
 *
 * Mathematical 3D shape compression via Signed Distance Functions.
 * "Don't send polygons. Send the law of shapes."
 */

/** Initialize the WASM module. Must be called before any other function. */
export function init(): Promise<void>;

/** SDF Evaluator - create and evaluate signed distance functions */
export class SdfEvaluator {
  /** Create a sphere SDF */
  static sphere(radius: number): SdfEvaluator;
  /** Create a box SDF */
  static box3d(hx: number, hy: number, hz: number): SdfEvaluator;
  /** Create a cylinder SDF */
  static cylinder(radius: number, height: number): SdfEvaluator;
  /** Create a torus SDF */
  static torus(major_radius: number, minor_radius: number): SdfEvaluator;

  /** CSG union */
  union(other: SdfEvaluator): SdfEvaluator;
  /** CSG intersection */
  intersection(other: SdfEvaluator): SdfEvaluator;
  /** CSG subtraction */
  subtract(other: SdfEvaluator): SdfEvaluator;
  /** Smooth union */
  smooth_union(other: SdfEvaluator, k: number): SdfEvaluator;

  /** Transform: translate */
  translate(x: number, y: number, z: number): SdfEvaluator;
  /** Transform: rotate (radians) */
  rotate(rx: number, ry: number, rz: number): SdfEvaluator;
  /** Transform: uniform scale */
  scale(s: number): SdfEvaluator;

  /** Evaluate distance at a single point */
  eval(x: number, y: number, z: number): number;
  /** Evaluate distance at multiple points (Float32Array of [x,y,z,...]) */
  eval_batch(points: Float32Array): Float32Array;

  /** Convert to mesh (returns {vertices: Float32Array, indices: Uint32Array}) */
  to_mesh(
    min_x: number,
    min_y: number,
    min_z: number,
    max_x: number,
    max_y: number,
    max_z: number,
    resolution: number,
  ): { vertices: Float32Array; indices: Uint32Array };

  /** Generate WGSL shader code for GPU rendering */
  to_wgsl(): string;
  /** Generate GLSL shader code */
  to_glsl(): string;

  /** Load from JSON string (ALICE-SDF JSON format) */
  static from_json(json: string): SdfEvaluator;
  /** Export to JSON string */
  to_json(): string;

  /** Free WASM memory */
  free(): void;
}

/** Raymarcher for CPU-based rendering */
export class Raymarcher {
  constructor(width: number, height: number);
  /** Render an SDF to an ImageData-compatible buffer */
  render(
    sdf: SdfEvaluator,
    camera_x: number,
    camera_y: number,
    camera_z: number,
  ): Uint8ClampedArray;
  free(): void;
}

/** SoA (Structure-of-Arrays) buffer for zero-copy SIMD evaluation */
export class SoABuffer {
  constructor(capacity: number);
  /** Set point at index */
  set_point(index: number, x: number, y: number, z: number): void;
  /** Get results as Float32Array */
  get_results(): Float32Array;
  free(): void;
}

/** Benchmark utility */
export function benchmark_eval(
  sdf: SdfEvaluator,
  points: number,
): { throughput_mps: number; ns_per_point: number };
