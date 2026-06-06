//
//  AliceSDFVisionOS.swift
//  ALICE-SDF Mobile — Apple Vision Pro / visionOS RealityKit ヘルパー
//
//  RealityKit シーンの中で ALICE-SDF プリミティブをすぐ使えるよう、
//  ModelEntity 化するファクトリ関数群を提供する。
//
//  使い方:
//      import AliceSDFVisionOS
//      import RealityKit
//
//      let sphere = AliceSDFRealityKit.makeSphereEntity(radius: 0.1)
//      content.add(sphere)
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

    /// 球 SDF を RealityKit `ModelEntity` (MeshResource) として生成
    /// `material` は呼び出し側で SimpleMaterial / PhysicallyBasedMaterial 等を渡す
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

    /// クロスプラットフォーム color (visionOS / iOS = UIColor、macOS = NSColor 自動切替)
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

    /// 任意の点で SDF 距離を評価 (ALICE-SDF wasm 同様のシグネチャ)
    public static func sphereDistance(point: SIMD3<Float>, center: SIMD3<Float>, radius: Float) -> Float {
        let dx = point.x - center.x
        let dy = point.y - center.y
        let dz = point.z - center.z
        return sqrt(dx * dx + dy * dy + dz * dz) - radius
    }

    /// 2 球の smooth union (visionOS hand-tracking のジェスチャ判定で使うことを想定)
    public static func smoothUnion(_ a: Float, _ b: Float, k: Float) -> Float {
        let h = max(k - abs(a - b), 0.0) / k
        return min(a, b) - h * h * k * 0.25
    }

    /// ハンドメッシュ頂点配列に対するバッチ SDF 距離評価
    public static func sphereBatch(
        points: [SIMD3<Float>],
        center: SIMD3<Float>,
        radius: Float
    ) -> [Float] {
        points.map { sphereDistance(point: $0, center: center, radius: radius) }
    }
}
#endif
