// ALICE-SDF Sample: Fractal Collider (Menger Sponge Labyrinth)
// Players can walk INSIDE the infinite labyrinth.
using UnityEngine;

#if UDONSHARP
using VRC.SDKBase;
using UdonSharp;
#endif

namespace AliceSDF.Samples
{
#if UDONSHARP
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class SampleFractal_Collider : AliceSDF_Collider
#else
    public class SampleFractal_Collider : AliceSDF_Collider
#endif
    {
        [Header("Fractal Parameters")]
        public float boxSize = 50.0f;
        public float holeSize = 2.0f;
        public float repeatScale = 15.0f;
        public float twistAmount = 0.02f;

        public
#if UDONSHARP
        new
#else
        override
#endif
        float Evaluate(Vector3 p)
        {
            // Optional twist
            if (twistAmount > 0.001f)
            {
                float angle = p.y * twistAmount;
                float c = Mathf.Cos(angle);
                float s = Mathf.Sin(angle);
                float px = c * p.x - s * p.z;
                float pz = s * p.x + c * p.z;
                p = new Vector3(px, p.y, pz);
            }

            // Step 1: Big box
            float box = SdfBox(p, new Vector3(boxSize, boxSize, boxSize));

            // Step 2: Infinite cross via repetition
            Vector3 rp = RepeatInfinite(p, new Vector3(repeatScale, repeatScale, repeatScale));
            // Cross = union of 3 infinite bars
            float inf = 1000f;
            float barX = SdfBox(rp, new Vector3(inf, holeSize, holeSize));
            float barY = SdfBox(rp, new Vector3(holeSize, inf, holeSize));
            float barZ = SdfBox(rp, new Vector3(holeSize, holeSize, inf));
            float cross = Mathf.Min(barX, Mathf.Min(barY, barZ));

            // Step 3: Subtract = Menger Sponge
            return Mathf.Max(-cross, box);
        }

        // Inlined SDF helpers (UdonSharp compatible)
        private float SdfBox(Vector3 p, Vector3 half)
        {
            float qx = Mathf.Abs(p.x) - half.x;
            float qy = Mathf.Abs(p.y) - half.y;
            float qz = Mathf.Abs(p.z) - half.z;
            float ox = Mathf.Max(qx, 0f);
            float oy = Mathf.Max(qy, 0f);
            float oz = Mathf.Max(qz, 0f);
            float outside = Mathf.Sqrt(ox * ox + oy * oy + oz * oz);
            float inside = Mathf.Min(Mathf.Max(qx, Mathf.Max(qy, qz)), 0f);
            return outside + inside;
        }

        private Vector3 RepeatInfinite(Vector3 p, Vector3 s)
        {
            return new Vector3(
                Mod(p.x + s.x * 0.5f, s.x) - s.x * 0.5f,
                Mod(p.y + s.y * 0.5f, s.y) - s.y * 0.5f,
                Mod(p.z + s.z * 0.5f, s.z) - s.z * 0.5f
            );
        }

        private float Mod(float x, float y)
        {
            return x - y * Mathf.Floor(x / y);
        }
    }
}
