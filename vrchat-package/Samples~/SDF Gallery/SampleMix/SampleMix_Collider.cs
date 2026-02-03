// ALICE-SDF Sample: Mix Collider (Fractal Planet + Ring + Onion Shell)
// Static snapshot (no animation) for collision detection.
using UnityEngine;

#if UDONSHARP
using VRC.SDKBase;
using UdonSharp;
#endif

namespace AliceSDF.Samples
{
#if UDONSHARP
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class SampleMix_Collider : AliceSDF_Collider
#else
    public class SampleMix_Collider : AliceSDF_Collider
#endif
    {
        [Header("Fractal Planet")]
        public float planetRadius = 6.0f;
        public float holeSize = 0.8f;
        public float repeatScale = 5.0f;

        [Header("Ring")]
        public float ringMajor = 10.0f;
        public float ringMinor = 0.3f;

        [Header("Onion Shell")]
        public float onionRadius = 3.0f;
        public int onionLayers = 3;
        public float onionThickness = 0.15f;
        public Vector3 onionPosition = new Vector3(16f, 0f, 0f);

        [Header("Blend")]
        public float smoothness = 0.8f;

        public
#if UDONSHARP
        new
#else
        override
#endif
        float Evaluate(Vector3 p)
        {
            // --- Fractal Planet (Sphere ∩ Menger Sponge) ---
            float planet = p.magnitude - planetRadius;

            // Infinite cross via repetition
            Vector3 rp = Sdf.RepeatInfinite(p, new Vector3(repeatScale, repeatScale, repeatScale));
            float inf = 1000f;
            float barX = Sdf.Box(rp, new Vector3(inf, holeSize, holeSize));
            float barY = Sdf.Box(rp, new Vector3(holeSize, inf, holeSize));
            float barZ = Sdf.Box(rp, new Vector3(holeSize, holeSize, inf));
            float cross = Mathf.Min(barX, Mathf.Min(barY, barZ));

            float fractalPlanet = Mathf.Max(-cross, planet);

            // --- Torus Ring ---
            float qx = new Vector2(p.x, p.z).magnitude - ringMajor;
            float ring = new Vector2(qx, p.y).magnitude - ringMinor;

            // --- Onion Shell (static position for collision) ---
            Vector3 op = p - onionPosition;
            float onion = op.magnitude - onionRadius;
            for (int i = 0; i < onionLayers; i++)
            {
                onion = Mathf.Abs(onion) - onionThickness;
            }

            // --- Combine with smooth union ---
            float d = fractalPlanet;
            float k = smoothness;
            float inv_k = 1f / k;

            // SmoothUnion(d, ring)
            float h = Mathf.Max(k - Mathf.Abs(d - ring), 0f) * inv_k;
            d = Mathf.Min(d, ring) - h * h * k * 0.25f;

            // SmoothUnion(d, onion) with tighter blend
            float k2 = k * 0.5f;
            float inv_k2 = 1f / k2;
            h = Mathf.Max(k2 - Mathf.Abs(d - onion), 0f) * inv_k2;
            d = Mathf.Min(d, onion) - h * h * k2 * 0.25f;

            return d;
        }
    }
}
