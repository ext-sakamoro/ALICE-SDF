// swift-tools-version: 5.9
// ALICE-SDF Mobile — visionOS SwiftPM Package
//
// Apple Vision Pro 用 RealityKit ヘルパー付き Swift Package。
// 既存の `mobile/swift-package` を base にして visionOS 専用の RealityKit Component を追加。

import PackageDescription

let package = Package(
    name: "AliceSDFVisionOS",
    platforms: [
        .visionOS(.v1),
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "AliceSDFVisionOS", targets: ["AliceSDFVisionOS"]),
    ],
    targets: [
        // 1. ALICE-SDF Mobile XCFramework (path 参照、build-xcframework.sh 生成物)
        .binaryTarget(
            name: "AliceSDFFramework",
            path: "../uniffi-wrapper/target/xcframework/AliceSDF.xcframework"
        ),

        // 2. visionOS / RealityKit ヘルパー層
        .target(
            name: "AliceSDFVisionOS",
            dependencies: ["AliceSDFFramework"],
            path: "Sources/AliceSDFVisionOS"
        ),
    ]
)
