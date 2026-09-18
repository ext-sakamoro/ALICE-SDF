// ALICE-SDF Sample: Cosmic Collider (Sun + orbiting Planet + tilted, twisted
// Ring + Moon + asteroid belt)
//
// The same law as SampleCosmic_Raymarcher.shader's map(), including the
// animation: animTime is what the shader's _Time.y is (the base collider
// sets it from Time.timeSinceLevelLoad every frame; 0 on a host), so the
// player collides with the planet where it is drawn. The Rust golden
// (examples/vrchat_cosmic_golden.rs) is the t = 0 snapshot.
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
        [Header("Cosmic Parameters (match the material)")]
        public float sunRadius = 8.0f;
        public float planetRadius = 2.5f;
        public float planetDistance = 18.0f;
        public float smoothness = 1.5f;
        [Tooltip("Twist of the ring around its axis, radians per metre of height (shader _RingTwist)")]
        public float ringTwist = 0.5f;

        // Ring tilt: 15 degrees about X (exact, the shader uses the same constants)
        private const float TiltCos = 0.96592582f;
        private const float TiltSin = 0.25881905f;

        public override float Evaluate(Vector3 p)
        {
            float time = animTime;

            // Sun
            float sun = p.magnitude - sunRadius;

            // Planet (orbiting)
            float orbitAngle = time * 0.15f;
            Vector3 planetPos = new Vector3(Mathf.Cos(orbitAngle) * planetDistance, 0f, Mathf.Sin(orbitAngle) * planetDistance);
            float planet = (p - planetPos).magnitude - planetRadius;

            // Ring: tilted 15 degrees about X, twisted about its axis, torus in the XZ plane
            Vector3 ringP = p - planetPos;
            ringP = new Vector3(ringP.x, TiltCos * ringP.y - TiltSin * ringP.z, TiltSin * ringP.y + TiltCos * ringP.z);
            if (ringTwist > 0.01f)
            {
                float ta = ringTwist * ringP.y;
                float ct = Mathf.Cos(ta);
                float st = Mathf.Sin(ta);
                ringP = new Vector3(ct * ringP.x - st * ringP.z, ringP.y, st * ringP.x + ct * ringP.z);
            }
            float qx = new Vector2(ringP.x, ringP.z).magnitude - planetRadius * 1.8f;
            float ring = new Vector2(qx, ringP.y).magnitude - 0.12f;

            // Moon (orbiting the planet)
            float moonOrbit = time * 0.4f;
            Vector3 moonPos = planetPos + new Vector3(Mathf.Cos(moonOrbit) * 4.0f, Mathf.Sin(moonOrbit) * 1.5f, Mathf.Sin(moonOrbit) * 4.0f);
            float moon = (p - moonPos).magnitude - 0.6f;

            // Asteroid belt: six spheres, hard union
            float asteroids = 1e10f;
            float beltR = planetDistance * 0.75f;
            for (int i = 0; i < 6; i++)
            {
                float angle = i * (Mathf.PI / 3f) + time * (0.1f + i * 0.02f);
                Vector3 aPos = new Vector3(Mathf.Cos(angle) * beltR, (i % 2 == 0) ? 0.5f : -0.5f, Mathf.Sin(angle) * beltR);
                asteroids = Mathf.Min(asteroids, (p - aPos).magnitude - (0.3f + i * 0.1f));
            }

            // Combine with smooth union, in the shader's order
            float d = sun;
            d = OpSmoothUnion(d, planet, smoothness);
            d = OpSmoothUnion(d, ring, smoothness * 0.5f);
            d = OpSmoothUnion(d, moon, smoothness);
            d = OpSmoothUnion(d, asteroids, smoothness * 0.3f);
            return d;
        }

        // Polynomial smooth minimum, identical to opSmoothUnion in the shader
        private float OpSmoothUnion(float d1, float d2, float k)
        {
            if (k < 0.0001f) return Mathf.Min(d1, d2);
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 - d2), 0f) * invK;
            return Mathf.Min(d1, d2) - h * h * k * 0.25f;
        }
    }
}
