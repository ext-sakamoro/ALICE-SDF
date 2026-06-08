//
//  AliceSDFVisionOS.swift
//  ALICE-SDF Mobile — Apple Vision Pro / visionOS RealityKit ヘルパー
//
//  ALICE-SDF Rust コアを XCFramework 経由で呼び出し、RealityKit シーンに
//  そのまま投入できる ModelEntity ファクトリを提供する。
//
//  使い方:
//      import AliceSDFVisionOS
//      import RealityKit
//
//      let sphere = AliceSDFRealityKit.makeSphereEntity(radius: 0.1)
//      content.add(sphere)
//
//      // 2 球 smooth blob (Rust コア経由):
//      let blob = AliceSDFRealityKit.makeBlobEntity(
//          centerA: SIMD3(-0.05, 0, 0), radiusA: 0.06,
//          centerB: SIMD3(0.05, 0, 0), radiusB: 0.06,
//          smoothK: 0.05, resolution: 32)
//      content.add(blob)
//

import Foundation
#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif
#if canImport(RealityKit)
import RealityKit
#endif

#if canImport(AliceSDFFramework)
import AliceSDFFramework
#endif

#if canImport(RealityKit)
@available(visionOS 1.0, iOS 17.0, *)
public enum AliceSDFRealityKit {

    // MARK: - Primitive Entity factories

    /// 球 SDF を RealityKit `ModelEntity` として生成 (RealityKit 組込みメッシュ)
    public static func makeSphereEntity(
        radius: Float = 0.1,
        material: RealityKit.Material? = nil
    ) -> ModelEntity {
        let mesh = MeshResource.generateSphere(radius: radius)
        let mat = material ?? SimpleMaterial(color: defaultBlue(), isMetallic: false)
        return ModelEntity(mesh: mesh, materials: [mat])
    }

    /// 直方体 SDF を ModelEntity 化
    public static func makeBoxEntity(
        size: SIMD3<Float> = SIMD3(0.1, 0.1, 0.1),
        material: RealityKit.Material? = nil
    ) -> ModelEntity {
        let mesh = MeshResource.generateBox(size: size)
        let mat = material ?? SimpleMaterial(color: defaultRed(), isMetallic: false)
        return ModelEntity(mesh: mesh, materials: [mat])
    }

    // MARK: - Custom SDF mesh

    /// 任意 SDF を voxel grid で評価し ModelEntity を返す
    ///
    /// `evaluate(p)` は world-space 座標 `p` での符号付き距離。
    /// `bounds` を `resolution` 等分割し、各 voxel 中心で評価 < 0 の voxel に
    /// 立方体 face を立てた簡易メッシュを構築する (粗いが軽い)。
    public static func makeSDFMeshEntity(
        bounds: (min: SIMD3<Float>, max: SIMD3<Float>),
        resolution: Int = 24,
        material: RealityKit.Material? = nil,
        evaluate: (SIMD3<Float>) -> Float
    ) -> ModelEntity? {
        guard let mesh = buildVoxelMesh(bounds: bounds, resolution: resolution, evaluate: evaluate) else {
            return nil
        }
        let mat = material ?? SimpleMaterial(color: defaultBlue(), isMetallic: false)
        return ModelEntity(mesh: mesh, materials: [mat])
    }

    /// 2 球の smooth-union (blob) を ModelEntity 化。
    /// `AliceSDFFramework` の `opSmoothUnion` + `sdfSphere` を直呼びすることで Rust コアと一致。
    public static func makeBlobEntity(
        centerA: SIMD3<Float>,
        radiusA: Float,
        centerB: SIMD3<Float>,
        radiusB: Float,
        smoothK: Float = 0.04,
        resolution: Int = 32,
        material: RealityKit.Material? = nil
    ) -> ModelEntity? {
        let pad = max(radiusA, radiusB) + smoothK
        let minB = SIMD3<Float>(
            min(centerA.x, centerB.x) - pad,
            min(centerA.y, centerB.y) - pad,
            min(centerA.z, centerB.z) - pad
        )
        let maxB = SIMD3<Float>(
            max(centerA.x, centerB.x) + pad,
            max(centerA.y, centerB.y) + pad,
            max(centerA.z, centerB.z) + pad
        )
        return makeSDFMeshEntity(
            bounds: (minB, maxB),
            resolution: resolution,
            material: material
        ) { p in
            let dA = sphereDistance(point: p, center: centerA, radius: radiusA)
            let dB = sphereDistance(point: p, center: centerB, radius: radiusB)
            return smoothUnion(dA, dB, k: smoothK)
        }
    }

    // MARK: - SDF math (delegated to AliceSDFFramework when available)

    /// 任意の点で球 SDF 距離を評価。XCFramework が link されていれば Rust コアを呼ぶ。
    public static func sphereDistance(
        point: SIMD3<Float>,
        center: SIMD3<Float>,
        radius: Float
    ) -> Float {
        #if canImport(AliceSDFFramework)
        return Float(AliceSDFFramework.sdfSphere(
            point: simdToFFI(point),
            center: simdToFFI(center),
            radius: radius
        ))
        #else
        let dx = point.x - center.x
        let dy = point.y - center.y
        let dz = point.z - center.z
        return sqrt(dx * dx + dy * dy + dz * dz) - radius
        #endif
    }

    /// 2 距離 SDF の smooth union。framework 経由なら Rust と完全一致。
    public static func smoothUnion(_ a: Float, _ b: Float, k: Float) -> Float {
        #if canImport(AliceSDFFramework)
        return Float(AliceSDFFramework.opSmoothUnion(a: a, b: b, k: k))
        #else
        let h = max(k - abs(a - b), 0.0) / k
        return min(a, b) - h * h * k * 0.25
        #endif
    }

    /// ハンドメッシュ頂点配列に対するバッチ SDF 距離評価
    public static func sphereBatch(
        points: [SIMD3<Float>],
        center: SIMD3<Float>,
        radius: Float
    ) -> [Float] {
        #if canImport(AliceSDFFramework)
        let ffiPoints = points.map { simdToFFI($0) }
        return AliceSDFFramework.sphereBatch(
            points: ffiPoints,
            center: simdToFFI(center),
            radius: radius
        ).map { Float($0) }
        #else
        return points.map { sphereDistance(point: $0, center: center, radius: radius) }
        #endif
    }

    /// 接続されている ALICE-SDF Rust core の version 文字列
    public static func coreVersion() -> String {
        #if canImport(AliceSDFFramework)
        return AliceSDFFramework.aliceSdfVersion()
        #else
        return "framework-not-linked"
        #endif
    }

    // MARK: - Internal

    #if canImport(AliceSDFFramework)
    private static func simdToFFI(_ v: SIMD3<Float>) -> AliceSDFFramework.Vec3 {
        AliceSDFFramework.Vec3(x: v.x, y: v.y, z: v.z)
    }
    #endif

    /// 簡易 voxel ベースメッシュ構築 (内部 voxel ごとに 1 立方体を発行)。
    /// production で滑らかさが必要なら Marching Cubes に差し替える。
    private static func buildVoxelMesh(
        bounds: (min: SIMD3<Float>, max: SIMD3<Float>),
        resolution: Int,
        evaluate: (SIMD3<Float>) -> Float
    ) -> MeshResource? {
        let n = max(2, resolution)
        let extent = bounds.max - bounds.min
        let step = SIMD3<Float>(extent.x / Float(n), extent.y / Float(n), extent.z / Float(n))
        let half = step * 0.5

        var positions: [SIMD3<Float>] = []
        var indices: [UInt32] = []

        for iz in 0..<n {
            for iy in 0..<n {
                for ix in 0..<n {
                    let p = bounds.min + SIMD3<Float>(
                        Float(ix) * step.x + half.x,
                        Float(iy) * step.y + half.y,
                        Float(iz) * step.z + half.z
                    )
                    if evaluate(p) < 0.0 {
                        let base = UInt32(positions.count)
                        appendCube(at: p, half: half, into: &positions, indices: &indices, base: base)
                    }
                }
            }
        }
        guard !positions.isEmpty else { return nil }

        var desc = MeshDescriptor()
        desc.positions = MeshBuffers.Positions(positions)
        desc.primitives = .triangles(indices)
        do {
            return try MeshResource.generate(from: [desc])
        } catch {
            return nil
        }
    }

    private static func appendCube(
        at p: SIMD3<Float>,
        half: SIMD3<Float>,
        into positions: inout [SIMD3<Float>],
        indices: inout [UInt32],
        base: UInt32
    ) {
        let corners: [SIMD3<Float>] = [
            p + SIMD3(-half.x, -half.y, -half.z),
            p + SIMD3( half.x, -half.y, -half.z),
            p + SIMD3( half.x,  half.y, -half.z),
            p + SIMD3(-half.x,  half.y, -half.z),
            p + SIMD3(-half.x, -half.y,  half.z),
            p + SIMD3( half.x, -half.y,  half.z),
            p + SIMD3( half.x,  half.y,  half.z),
            p + SIMD3(-half.x,  half.y,  half.z),
        ]
        positions.append(contentsOf: corners)
        let tris: [UInt32] = [
            0,1,2, 0,2,3,   // -Z
            4,6,5, 4,7,6,   // +Z
            0,4,5, 0,5,1,   // -Y
            3,2,6, 3,6,7,   // +Y
            0,3,7, 0,7,4,   // -X
            1,5,6, 1,6,2,   // +X
        ]
        indices.append(contentsOf: tris.map { $0 + base })
    }

    // MARK: - Default colors

    #if canImport(UIKit)
    private static func defaultBlue() -> UIColor { UIColor.systemBlue }
    private static func defaultRed() -> UIColor { UIColor.systemRed }
    #elseif canImport(AppKit)
    private static func defaultBlue() -> NSColor { NSColor.systemBlue }
    private static func defaultRed() -> NSColor { NSColor.systemRed }
    #else
    private static func defaultBlue() -> RealityFoundation.Material.Color { .init(white: 0.5, alpha: 1.0) }
    private static func defaultRed() -> RealityFoundation.Material.Color { .init(white: 0.5, alpha: 1.0) }
    #endif
}
#endif
