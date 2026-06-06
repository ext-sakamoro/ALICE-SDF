import SwiftUI

struct ContentView: View {
    @State private var sphereYOffset: Float = 0.8
    @State private var radius: Float = 1.0
    @State private var smoothK: Float = 0.3

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                header

                metricsCard

                SDFSliceView(
                    sphereYOffset: sphereYOffset,
                    radius: radius,
                    smoothK: smoothK
                )
                .frame(width: 280, height: 280)
                .background(Color.black)
                .cornerRadius(12)

                sliders

                Spacer(minLength: 24)
            }
            .padding(.horizontal)
        }
    }

    private var header: some View {
        VStack(spacing: 4) {
            Text("ALICE-SDF Mobile")
                .font(.title2.bold())
            Text("v\(aliceSdfVersion()) • iOS demo")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.top, 12)
    }

    private var metricsCard: some View {
        let p = Vec3(x: 1.0, y: 0.0, z: 0.0)
        let c1 = Vec3(x: 0.0, y: sphereYOffset, z: 0.0)
        let c2 = Vec3(x: 0.0, y: -sphereYOffset, z: 0.0)
        let d1 = sdfSphere(point: p, center: c1, radius: radius)
        let d2 = sdfSphere(point: p, center: c2, radius: radius)
        let dUnion = opUnion(a: d1, b: d2)
        let dSmooth = opSmoothUnion(a: d1, b: d2, k: smoothK)

        return VStack(alignment: .leading, spacing: 6) {
            metricRow("query point", "(1, 0, 0)")
            metricRow("sphere1 d", String(format: "%.4f", d1))
            metricRow("sphere2 d", String(format: "%.4f", d2))
            metricRow("union", String(format: "%.4f", dUnion))
            metricRow("smooth union", String(format: "%.4f (k=%.2f)", dSmooth, smoothK))
        }
        .font(.system(.body, design: .monospaced))
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    private func metricRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
                .frame(width: 130, alignment: .leading)
            Text(value)
                .foregroundColor(.primary)
        }
    }

    private var sliders: some View {
        VStack(alignment: .leading, spacing: 10) {
            sliderRow("Sphere Y offset", value: $sphereYOffset, range: 0.0...2.0)
            sliderRow("Radius", value: $radius, range: 0.1...2.0)
            sliderRow("Smooth K", value: $smoothK, range: 0.01...1.0)
        }
        .padding(12)
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    private func sliderRow(_ label: String, value: Binding<Float>, range: ClosedRange<Float>) -> some View {
        VStack(alignment: .leading) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            Slider(value: value, in: range)
        }
    }
}

private struct SDFSliceView: View {
    let sphereYOffset: Float
    let radius: Float
    let smoothK: Float

    var body: some View {
        Canvas { ctx, size in
            let w = Int(size.width)
            let h = Int(size.height)
            let step = 4
            let halfRange: Float = 2.5

            let c1 = Vec3(x: 0, y: sphereYOffset, z: 0)
            let c2 = Vec3(x: 0, y: -sphereYOffset, z: 0)

            for y in stride(from: 0, to: h, by: step) {
                for x in stride(from: 0, to: w, by: step) {
                    let px = (Float(x) / Float(w) * 2 - 1) * halfRange
                    let py = (1 - Float(y) / Float(h) * 2) * halfRange
                    let p = Vec3(x: px, y: py, z: 0)
                    let d1 = sdfSphere(point: p, center: c1, radius: radius)
                    let d2 = sdfSphere(point: p, center: c2, radius: radius)
                    let d = opSmoothUnion(a: d1, b: d2, k: smoothK)

                    let color: Color
                    if d < 0 {
                        let t = min(1.0, Double(-d))
                        color = Color(red: 0.2, green: 0.4 + 0.5 * t, blue: 0.9)
                    } else {
                        let t = min(1.0, Double(d) * 0.6)
                        color = Color(red: 0.05 + t * 0.4, green: 0.05 + t * 0.4, blue: 0.1 + t * 0.4)
                    }

                    ctx.fill(
                        Path(CGRect(x: x, y: y, width: step, height: step)),
                        with: .color(color)
                    )
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
