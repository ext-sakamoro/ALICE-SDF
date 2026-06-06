package net.alicelaw.alicesdf

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.shape.RoundedCornerShape
import uniffi.alice_sdf.Vec3
import uniffi.alice_sdf.aliceSdfVersion
import uniffi.alice_sdf.opSmoothUnion
import uniffi.alice_sdf.opUnion
import uniffi.alice_sdf.sdfSphere

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    AliceSdfDemoScreen()
                }
            }
        }
    }
}

@Composable
fun AliceSdfDemoScreen() {
    var sphereYOffset by remember { mutableStateOf(0.80f) }
    var radius by remember { mutableStateOf(1.0f) }
    var smoothK by remember { mutableStateOf(0.30f) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(12.dp))
        Text("ALICE-SDF Mobile", fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Text("v${aliceSdfVersion()} • Android demo", fontSize = 12.sp, color = Color.Gray)

        Spacer(Modifier.height(16.dp))
        MetricsCard(sphereYOffset, radius, smoothK)

        Spacer(Modifier.height(16.dp))
        SdfSliceCanvas(sphereYOffset, radius, smoothK)

        Spacer(Modifier.height(16.dp))
        SlidersCard(
            sphereYOffset = sphereYOffset,
            radius = radius,
            smoothK = smoothK,
            onYChange = { sphereYOffset = it },
            onRadiusChange = { radius = it },
            onKChange = { smoothK = it }
        )

        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun MetricsCard(sphereYOffset: Float, radius: Float, smoothK: Float) {
    val p = Vec3(1.0f, 0.0f, 0.0f)
    val c1 = Vec3(0.0f, sphereYOffset, 0.0f)
    val c2 = Vec3(0.0f, -sphereYOffset, 0.0f)
    val d1 = sdfSphere(p, c1, radius)
    val d2 = sdfSphere(p, c2, radius)
    val dUnion = opUnion(d1, d2)
    val dSmooth = opSmoothUnion(d1, d2, smoothK)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(Color(0xFFEEEEEE))
            .padding(12.dp)
    ) {
        MetricRow("query point", "(1, 0, 0)")
        MetricRow("sphere1 d", "%.4f".format(d1))
        MetricRow("sphere2 d", "%.4f".format(d2))
        MetricRow("union", "%.4f".format(dUnion))
        MetricRow("smooth union", "%.4f (k=%.2f)".format(dSmooth, smoothK))
    }
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
        Text(
            label,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(130.dp),
            color = Color.Gray
        )
        Text(value, fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun SdfSliceCanvas(sphereYOffset: Float, radius: Float, smoothK: Float) {
    Canvas(
        modifier = Modifier
            .size(280.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(Color.Black)
    ) {
        val w = size.width.toInt()
        val h = size.height.toInt()
        val step = 4
        val halfRange = 2.5f

        val c1 = Vec3(0f, sphereYOffset, 0f)
        val c2 = Vec3(0f, -sphereYOffset, 0f)

        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val px = (x.toFloat() / w.toFloat() * 2f - 1f) * halfRange
                val py = (1f - y.toFloat() / h.toFloat() * 2f) * halfRange
                val p = Vec3(px, py, 0f)
                val d1 = sdfSphere(p, c1, radius)
                val d2 = sdfSphere(p, c2, radius)
                val d = opSmoothUnion(d1, d2, smoothK)

                val color = if (d < 0f) {
                    val t = (-d).coerceAtMost(1f)
                    Color(0.2f, 0.4f + 0.5f * t, 0.9f, 1.0f)
                } else {
                    val t = (d * 0.6f).coerceAtMost(1f)
                    Color(0.05f + t * 0.4f, 0.05f + t * 0.4f, 0.1f + t * 0.4f, 1.0f)
                }

                drawRect(
                    color = color,
                    topLeft = Offset(x.toFloat(), y.toFloat()),
                    size = Size(step.toFloat(), step.toFloat())
                )
                x += step
            }
            y += step
        }
    }
}

@Composable
private fun SlidersCard(
    sphereYOffset: Float,
    radius: Float,
    smoothK: Float,
    onYChange: (Float) -> Unit,
    onRadiusChange: (Float) -> Unit,
    onKChange: (Float) -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(Color(0xFFEEEEEE))
            .padding(12.dp)
    ) {
        SliderRow("Sphere Y offset", sphereYOffset, 0.0f..2.0f, onYChange)
        SliderRow("Radius", radius, 0.1f..2.0f, onRadiusChange)
        SliderRow("Smooth K", smoothK, 0.01f..1.0f, onKChange)
    }
}

@Composable
private fun SliderRow(label: String, value: Float, range: ClosedFloatingPointRange<Float>, onChange: (Float) -> Unit) {
    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row {
            Text(label, modifier = Modifier.weight(1f))
            Text("%.2f".format(value), fontFamily = FontFamily.Monospace, color = Color.Gray)
        }
        Slider(value = value, onValueChange = onChange, valueRange = range)
    }
}
