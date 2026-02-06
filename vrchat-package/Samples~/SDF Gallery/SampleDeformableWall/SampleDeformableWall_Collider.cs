// =============================================================================
// ALICE-SDF Sample: DeformableWall Collider (UdonSharp)
// =============================================================================
// Detects hand proximity to wall, records impact points, and sends them
// to the shader for visual deformation. Also handles player collision.
//
// Features:
//   - VR hand tracking: left/right hand impact detection
//   - Up to 16 impacts with circular buffer recycling
//   - Time-based decay (old impacts auto-expire)
//   - Shader sync via Material.SetVectorArray each frame
//   - Player push-back via SDF gradient
//
// Network: Local-only (each player sees their own dents).
//   To sync dents across players, add [UdonSynced] to impactPoints
//   and call RequestSerialization() on impact.
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
    public class SampleDeformableWall_Collider : UdonSharpBehaviour
#else
    public class SampleDeformableWall_Collider : MonoBehaviour
#endif
    {
        [Header("Wall Dimensions (match shader properties)")]
        [Tooltip("Must match shader _WallWidth")]
        public float wallWidth = 5.0f;
        [Tooltip("Must match shader _WallHeight")]
        public float wallHeight = 2.5f;
        [Tooltip("Must match shader _WallThick")]
        public float wallThickness = 0.2f;

        [Header("Impact Settings")]
        [Tooltip("Hand must be within this distance of wall surface to register")]
        public float impactDistance = 0.08f;
        [Tooltip("Minimum time between impacts from the same hand (sec)")]
        public float impactCooldown = 0.15f;
        [Tooltip("Must match shader _DecaySpeed")]
        public float decaySpeed = 0.5f;
        [Tooltip("Must match shader _DentRadius")]
        public float dentRadius = 0.35f;

        [Header("Player Collision")]
        public float collisionMargin = 0.1f;
        [Range(0.5f, 1.5f)]
        public float pushStrength = 1.0f;
        public float maxPushDistance = 2.0f;

        // Impact circular buffer
        private Vector4[] impactPoints;
        private int impactCount = 0;

        // Hand cooldown state
        private float lastLeftTime = -10f;
        private float lastRightTime = -10f;

        // Cached references
        private Material mat;
        private Vector3 wallCenter;
        private Vector3 wallHalf;

#if UDONSHARP
        private VRCPlayerApi localPlayer;
#endif

        void Start()
        {
            wallCenter = new Vector3(0f, wallHeight, 0f);
            wallHalf = new Vector3(wallWidth, wallHeight, wallThickness);

            impactPoints = new Vector4[16];
            for (int i = 0; i < 16; i++)
                impactPoints[i] = Vector4.zero;

            MeshRenderer rend = GetComponent<MeshRenderer>();
            if (rend != null) mat = rend.material;

#if UDONSHARP
            localPlayer = Networking.LocalPlayer;
#endif
        }

#if UDONSHARP
        public override void PostLateUpdate()
        {
            if (localPlayer == null) return;

            // --- Hand Impact Detection ---
            // Left hand
            VRCPlayerApi.TrackingData lh =
                localPlayer.GetTrackingData(VRCPlayerApi.TrackingDataType.LeftHand);
            Vector3 lPos = lh.position;
            TryRegisterImpact(lPos, true);

            // Right hand
            VRCPlayerApi.TrackingData rh =
                localPlayer.GetTrackingData(VRCPlayerApi.TrackingDataType.RightHand);
            Vector3 rPos = rh.position;
            TryRegisterImpact(rPos, false);

            // --- Player Collision ---
            Vector3 playerPos = localPlayer.GetPosition();
            Vector3 feetPos = playerPos + Vector3.down * 0.05f;
            float dist = EvaluateSdf(feetPos);

            if (dist < collisionMargin)
            {
                Vector3 normal = EstimateGradient(feetPos);
                float pen = Mathf.Min(collisionMargin - dist, maxPushDistance);
                localPlayer.TeleportTo(
                    playerPos + normal * pen * pushStrength,
                    localPlayer.GetRotation()
                );
            }

            // --- Shader Update ---
            SyncShader();
        }
#endif

        private void TryRegisterImpact(Vector3 handPos, bool isLeft)
        {
            float handDist = SdfBox(handPos - wallCenter, wallHalf);

            if (handDist < impactDistance)
            {
                float lastTime = isLeft ? lastLeftTime : lastRightTime;
                if (Time.time - lastTime < impactCooldown) return;

                RecordImpact(handPos);

                if (isLeft) lastLeftTime = Time.time;
                else lastRightTime = Time.time;
            }
        }

        private void RecordImpact(Vector3 pos)
        {
            // Find best slot: prefer empty, then recycle oldest
            int slot = -1;
            float oldestAge = 0f;
            int oldestIdx = 0;

            for (int i = 0; i < 16; i++)
            {
                // Empty slot (never used or fully decayed)
                if (impactPoints[i].w < 0.001f)
                {
                    slot = i;
                    break;
                }
                float age = Time.time - impactPoints[i].w;
                if (age > oldestAge)
                {
                    oldestAge = age;
                    oldestIdx = i;
                }
            }

            if (slot < 0) slot = oldestIdx;

            impactPoints[slot] = new Vector4(pos.x, pos.y, pos.z, Time.time);
            if (impactCount < 16) impactCount++;
        }

        private void SyncShader()
        {
            if (mat == null) return;
            // Recalculate active count (reset fully decayed slots)
            int activeCount = 0;
            for (int i = 0; i < 16; i++)
            {
                if (impactPoints[i].w > 0.001f)
                {
                    float age = Time.time - impactPoints[i].w;
                    float r = dentRadius * Mathf.Exp(-age * decaySpeed);
                    if (r < 0.005f)
                    {
                        impactPoints[i] = Vector4.zero;
                    }
                    else
                    {
                        activeCount++;
                    }
                }
            }
            impactCount = activeCount;

            mat.SetVectorArray("_ImpactPoints", impactPoints);
            mat.SetFloat("_ImpactCount", (float)impactCount);
        }

        // =================================================================
        // SDF Evaluation (simplified - no dents for collision performance)
        // =================================================================
        public float EvaluateSdf(Vector3 p)
        {
            float ground = p.y;
            float wall = SdfBox(p - wallCenter, wallHalf);
            return Mathf.Min(ground, wall);
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
        // Inlined SDF (UdonSharp compatible - no static class calls)
        // =================================================================
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
    }
}
