/**
 * Optional @react-three/fiber components for ALICE-SDF.
 *
 * Import path: `@alice-sdf/threejs/r3f`
 *
 * @example
 * ```tsx
 * import { Canvas } from '@react-three/fiber';
 * import { AliceSDFSlicePlane } from '@alice-sdf/threejs/r3f';
 *
 * <Canvas>
 *   <AliceSDFSlicePlane wasmUrl="/alice_sdf.js" center={[0,0,0]} radius={1} />
 * </Canvas>
 * ```
 */

import { useEffect, useState } from "react";
import type { Texture } from "three";

import { AliceSDF, type Vec3 } from "./index";

export interface AliceSDFSlicePlaneProps {
  /** URL of the wasm-bindgen JS glue file. */
  wasmUrl: string;
  center?: Vec3;
  radius?: number;
  halfRange?: number;
  resolution?: number;
  size?: number;
}

/**
 * A textured plane component showing a 2D slice of an SDF sphere.
 * Useful as a quick smoke test in R3F scenes.
 */
export function AliceSDFSlicePlane(props: AliceSDFSlicePlaneProps) {
  const {
    wasmUrl,
    center = [0, 0, 0],
    radius = 1,
    halfRange = 2.5,
    resolution = 256,
    size = 2,
  } = props;
  const [texture, setTexture] = useState<Texture | null>(null);

  useEffect(() => {
    let cancelled = false;
    AliceSDF.load(wasmUrl)
      .then((sdf) => sdf.createSliceTexture(resolution, resolution, center, radius, halfRange))
      .then((tex) => {
        if (!cancelled) setTexture(tex);
      })
      .catch(() => {
        /* swallow load errors so the host scene keeps running */
      });
    return () => {
      cancelled = true;
    };
  }, [wasmUrl, center[0], center[1], center[2], radius, halfRange, resolution]);

  if (!texture) return null;
  return (
    <mesh>
      <planeGeometry args={[size, size]} />
      <meshBasicMaterial map={texture} toneMapped={false} />
    </mesh>
  );
}
