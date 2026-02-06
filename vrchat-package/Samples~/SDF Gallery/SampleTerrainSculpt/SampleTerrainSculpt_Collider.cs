// =============================================================================
// ALICE-SDF Sample: Terrain Sculpt Collider & Interaction (UdonSharp)
// =============================================================================
// Real-time terrain sculpting with VR hands.
// Left hand = add terrain (SmoothUnion), Right hand = dig (SmoothSubtraction).
//
// The key innovation: both rendering AND collision use the same SDF formula.
// Dig a hole → you actually fall in. Build a hill → you can climb it.
// This is impossible with VRChat's standard mesh-based colliders.
//
// Sculpt operations are stored in a circular buffer (max 48).
// When the buffer is full, the oldest operation is overwritten.
//
// Network: Local-only (each player sculpts their own terrain).
//   For shared sculpting, add [UdonSynced] to sculptData/sculptCount
//   and call RequestSerialization() on each sculpt.
//
// Author: Moroya Sakamoto
// =============================================================================

using UnityEngine;

#if UDONSHARP
using VRC.SDKBase;
using UdonSharp;
#endif

namespace AliceSDF.Samples
{
#if UDONSHARP
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class SampleTerrainSculpt_Collider : UdonSharpBehaviour
#else
    public class SampleTerrainSculpt_Collider : MonoBehaviour
#endif
    {
        [Header("Sculpting")]
        [Tooltip("Radius of the sculpt brush")]
        public float sculptRadius = 0.3f;
        [Tooltip("Hand must be within this distance of terrain to sculpt")]
        public float sculptDistance = 0.15f;
        [Tooltip("Minimum time between sculpt operations (sec)")]
        public float sculptCooldown = 0.12f;
        [Tooltip("SmoothUnion factor for adding terrain (match shader _AddSmooth)")]
        public float addSmooth = 0.25f;
        [Tooltip("SmoothSubtraction factor for digging (match shader _SubSmooth)")]
        public float subSmooth = 0.15f;

        [Header("Player Collision")]
        public float collisionMargin = 0.15f;
        [Range(0.5f, 1.5f)]
        public float pushStrength = 1.0f;

        // Sculpt operation buffer
        private const int MAX_SCULPTS = 48;
        private Vector4[] sculptData;
        private int sculptCount = 0;
        private int nextSlot = 0;

        // Hand cooldown
        private float lastLeftTime = -10f;
        private float lastRightTime = -10f;

        // Cached
        private Material mat;

#if UDONSHARP
        private VRCPlayerApi localPlayer;
#endif

        void Start()
        {
            sculptData = new Vector4[MAX_SCULPTS];
            for (int i = 0; i < MAX_SCULPTS; i++)
                sculptData[i] = Vector4.zero;

            MeshRenderer rend = GetComponent<MeshRenderer>();
            if (rend != null) mat = rend.material;

#if UDONSHARP
            localPlayer = Networking.LocalPlayer;
#endif

            SyncShader();
        }

#if UDONSHARP
        public override void PostLateUpdate()
        {
            if (localPlayer == null) return;

            // --- Left hand: Add terrain ---
            Vector3 lPos = localPlayer.GetTrackingData(
                VRCPlayerApi.TrackingDataType.LeftHand).position;
            TrySculpt(lPos, true);

            // --- Right hand: Dig terrain ---
            Vector3 rPos = localPlayer.GetTrackingData(
                VRCPlayerApi.TrackingDataType.RightHand).position;
            TrySculpt(rPos, false);

            // --- Player Collision ---
            Vector3 playerPos = localPlayer.GetPosition();
            Vector3 feetPos = playerPos + Vector3.down * 0.05f;
            float dist = EvaluateSdf(feetPos);

            if (dist < collisionMargin)
            {
                Vector3 normal = EstimateGradient(feetPos);
                float pen = Mathf.Min(collisionMargin - dist, 2.0f);
                localPlayer.TeleportTo(
                    playerPos + normal * pen * pushStrength,
                    localPlayer.GetRotation()
                );
            }

            // --- Shader Sync ---
            SyncShader();

            // --- Hand Cursor ---
            if (mat != null)
            {
                float lDist = EvaluateSdf(lPos);
                float rDist = EvaluateSdf(rPos);
                mat.SetVector("_LeftHand", new Vector4(
                    lPos.x, lPos.y, lPos.z,
                    Mathf.Abs(lDist) < sculptRadius * 2f ? 1f : 0f));
                mat.SetVector("_RightHand", new Vector4(
                    rPos.x, rPos.y, rPos.z,
                    Mathf.Abs(rDist) < sculptRadius * 2f ? 1f : 0f));
            }
        }
#endif

        // =================================================================
        // Sculpting
        // =================================================================
        private void TrySculpt(Vector3 handPos, bool isAdd)
        {
            float dist = EvaluateSdf(handPos);

            // Hand must be near the terrain surface
            if (Mathf.Abs(dist) > sculptDistance) return;

            // Cooldown
            float lastTime = isAdd ? lastLeftTime : lastRightTime;
            if (Time.time - lastTime < sculptCooldown) return;

            // Record the sculpt operation
            float r = isAdd ? sculptRadius : -sculptRadius;
            RecordSculpt(handPos, r);

            if (isAdd) lastLeftTime = Time.time;
            else lastRightTime = Time.time;
        }

        private void RecordSculpt(Vector3 pos, float radius)
        {
            sculptData[nextSlot] = new Vector4(pos.x, pos.y, pos.z, radius);
            nextSlot = (nextSlot + 1) % MAX_SCULPTS;
            if (sculptCount < MAX_SCULPTS) sculptCount++;
        }

        private void SyncShader()
        {
            if (mat == null) return;
            mat.SetVectorArray("_SculptData", sculptData);
            mat.SetFloat("_SculptCount", (float)sculptCount);
            mat.SetFloat("_SculptRadius", sculptRadius);
        }

        // =================================================================
        // SDF Evaluation (matches shader map() exactly)
        // =================================================================
        public float EvaluateSdf(Vector3 p)
        {
            float terrain = p.y;

            for (int i = 0; i < sculptCount; i++)
            {
                float rw = sculptData[i].w;
                if (rw > 0.001f)
                {
                    // Add: SmoothUnion
                    Vector3 sp = new Vector3(sculptData[i].x, sculptData[i].y, sculptData[i].z);
                    float hill = (p - sp).magnitude - rw;
                    terrain = OpSmoothUnion(terrain, hill, addSmooth);
                }
                else if (rw < -0.001f)
                {
                    // Dig: SmoothSubtraction
                    Vector3 sp = new Vector3(sculptData[i].x, sculptData[i].y, sculptData[i].z);
                    float hole = (p - sp).magnitude - (-rw);
                    terrain = OpSmoothSubtraction(terrain, hole, subSmooth);
                }
            }

            return terrain;
        }

        private Vector3 EstimateGradient(Vector3 p)
        {
            float e = 0.02f;
            float dx = EvaluateSdf(new Vector3(p.x + e, p.y, p.z))
                     - EvaluateSdf(new Vector3(p.x - e, p.y, p.z));
            float dy = EvaluateSdf(new Vector3(p.x, p.y + e, p.z))
                     - EvaluateSdf(new Vector3(p.x, p.y - e, p.z));
            float dz = EvaluateSdf(new Vector3(p.x, p.y, p.z + e))
                     - EvaluateSdf(new Vector3(p.x, p.y, p.z - e));
            Vector3 grad = new Vector3(dx, dy, dz);
            float len = grad.magnitude;
            return (len > 0.0001f) ? grad / len : Vector3.up;
        }

        // =================================================================
        // Inlined SDF Operations (UdonSharp compatible)
        // =================================================================
        private float OpSmoothUnion(float d1, float d2, float k)
        {
            if (k < 0.0001f) return Mathf.Min(d1, d2);
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 - d2), 0f) * invK;
            return Mathf.Min(d1, d2) - h * h * k * 0.25f;
        }

        private float OpSmoothSubtraction(float d1, float d2, float k)
        {
            // Subtracts d2 FROM d1 (ALICE-SDF convention)
            if (k < 0.0001f) return Mathf.Max(d1, -d2);
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 + d2), 0f) * invK;
            return Mathf.Max(d1, -d2) + h * h * k * 0.25f;
        }
    }
}
