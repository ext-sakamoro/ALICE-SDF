// ALICE-SDF Sample: Cosmic Collider (Sun + Planet + Ring + Moon)
// Static snapshot (no orbit animation) for collision detection.
using UnityEngine;

#if UDONSHARP
using VRC.SDKBase;
using UdonSharp;
#endif

namespace AliceSDF.Samples
{
#if UDONSHARP
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class SampleCosmic_Collider : AliceSDF_Collider
#else
    public class SampleCosmic_Collider : AliceSDF_Collider
#endif
    {
        [Header("Cosmic Parameters")]
        public float sunRadius = 8.0f;
        public float planetRadius = 2.5f;
        public float planetDistance = 18.0f;
        public float smoothness = 1.5f;

        public
#if UDONSHARP
        new
#else
        override
#endif
        float Evaluate(Vector3 p)
        {
            // Sun
            float sun = p.magnitude - sunRadius;

            // Planet (static position for collision — shader handles animation)
            Vector3 planetPos = new Vector3(planetDistance, 0f, 0f);
            float planet = (p - planetPos).magnitude - planetRadius;

            // Ring (torus around planet)
            Vector3 ringP = p - planetPos;
            float qx = new Vector2(ringP.x, ringP.z).magnitude - planetRadius * 1.8f;
            float ring = new Vector2(qx, ringP.y).magnitude - 0.12f;

            // Moon
            Vector3 moonPos = planetPos + new Vector3(4f, 1.5f, 0f);
            float moon = (p - moonPos).magnitude - 0.6f;

            // Smooth union
            float d = sun;
            float k = smoothness;
            float inv_k = 1f / k;

            // SmoothUnion(d, planet)
            float h = Mathf.Max(k - Mathf.Abs(d - planet), 0f) * inv_k;
            d = Mathf.Min(d, planet) - h * h * k * 0.25f;

            // SmoothUnion(d, ring)
            float k2 = k * 0.5f;
            float inv_k2 = 1f / k2;
            h = Mathf.Max(k2 - Mathf.Abs(d - ring), 0f) * inv_k2;
            d = Mathf.Min(d, ring) - h * h * k2 * 0.25f;

            // SmoothUnion(d, moon)
            h = Mathf.Max(k - Mathf.Abs(d - moon), 0f) * inv_k;
            d = Mathf.Min(d, moon) - h * h * k * 0.25f;

            return d;
        }
    }
}
