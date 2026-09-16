// =============================================================================
// ALICE-SDF Sample: Mochi Collider & Interaction (UdonSharp)
// =============================================================================
// Manages mochi (rice cake) physics: grab, merge, split, and growth.
// Sends mochi state to shader for SmoothUnion blob rendering.
//
// Interaction model (VR):
//   - Hand enters a mochi sphere -> mochi sticks to hand (grab)
//   - Pull hand away -> mochi stretches (SmoothUnion neck) then splits
//   - Push two free mochis together -> they merge (volume conservation)
//   - Merged mochis grow: r_new = cbrt(r1^3 + r2^3)
//   - Release hand far from mochi -> mochi drops with soft gravity
//
// The SDF evaluated here (EvaluateSdf) is the same formula the shader
// renders: SmoothUnion(ground, SmoothUnion(mochi_i, blendK), groundK).
// blendK / groundK are pushed to the material every frame, so this script
// is the single source of truth for both rendering and collision.
//
// Network: Local-only (each player sees their own mochi state).
//   For multiplayer sync, add [UdonSynced] to mochiPos / mochiR / mochiCount
//   and call RequestSerialization() on state changes.
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
    public class SampleMochi_Collider : UdonSharpBehaviour
#else
    public class SampleMochi_Collider : MonoBehaviour
#endif
    {
        /// <summary>Upper bound of tracked mochis. Must match MOCHI_MAX in the shader.</summary>
        public const int MaxMochi = 16;

        // Hand slots for the per-hand state arrays
        private const int HandLeft = 0;
        private const int HandRight = 1;
        private const int HandCount = 2;

        // cbrt(0.5): radius of each half when a mochi splits with volume conserved
        private const float SplitRadiusScale = 0.7937005f;

        [Header("Mochi Settings")]
        [Tooltip("SmoothUnion blend factor between mochis (sent to shader _BlendK)")]
        public float blendK = 0.5f;
        [Tooltip("Ground SmoothUnion factor (sent to shader _GroundK)")]
        public float groundK = 0.15f;
        [Tooltip("Minimum mochi radius (won't split below this)")]
        public float minRadius = 0.1f;
        [Tooltip("Settle rate for free mochis dropping to the ground (1/sec)")]
        public float gravity = 3.0f;

        [Header("Interaction")]
        [Tooltip("Hand must be within this fraction of radius to grab")]
        public float grabThreshold = 0.8f;
        [Tooltip("Dwell time before grab activates (sec)")]
        public float grabDwellTime = 0.08f;
        [Tooltip("Pull distance (x radius) to trigger split")]
        public float splitDistance = 2.5f;
        [Tooltip("Distance (x radius) for auto-release")]
        public float releaseDistance = 4.0f;
        [Tooltip("Merge threshold: fraction of combined radii")]
        public float mergeThreshold = 0.7f;

        [Header("Player Collision")]
        public float collisionMargin = 0.1f;
        [Range(0.5f, 1.5f)]
        public float pushStrength = 1.0f;

        // Mochi state arrays (indices 0..mochiCount-1 are live)
        private Vector3[] mochiPos;
        private float[] mochiR;
        private int mochiCount = 0;

        // Per-hand state, indexed by HandLeft / HandRight
        private int[] grab;            // mochi index held, -1 = not grabbing
        private Vector3[] grabOrigin;  // where the held mochi was grabbed
        private bool[] splitDone;      // one split per grab
        private float[] dwell;         // seconds the hand has been inside dwellTarget
        private int[] dwellTarget;     // mochi the hand is dwelling in, -1 = none

        // Shader data
        private Vector4[] shaderData;
        private Material mat;

#if UDONSHARP
        private VRCPlayerApi localPlayer;
#endif

        void Start()
        {
            mochiPos = new Vector3[MaxMochi];
            mochiR = new float[MaxMochi];
            shaderData = new Vector4[MaxMochi];

            grab = new int[HandCount];
            grabOrigin = new Vector3[HandCount];
            splitDone = new bool[HandCount];
            dwell = new float[HandCount];
            dwellTarget = new int[HandCount];
            for (int h = 0; h < HandCount; h++)
            {
                grab[h] = -1;
                dwellTarget[h] = -1;
            }

            // Place initial mochis (same layout as mochi.asdf.json)
            SpawnMochi(new Vector3(-0.6f, 0.35f, 0.5f), 0.35f);
            SpawnMochi(new Vector3( 0.5f, 0.30f, 0.3f), 0.30f);
            SpawnMochi(new Vector3( 0.0f, 0.28f,-0.4f), 0.28f);
            SpawnMochi(new Vector3(-0.9f, 0.40f,-0.2f), 0.40f);
            SpawnMochi(new Vector3( 0.4f, 0.25f,-0.8f), 0.25f);

            MeshRenderer rend = GetComponent<MeshRenderer>();
            if (rend != null)
                mat = rend.material;
            else
                Debug.LogWarning("[ALICE-SDF] SampleMochi_Collider: No MeshRenderer found. Shader sync disabled.");

#if UDONSHARP
            localPlayer = Networking.LocalPlayer;
#endif

            SyncShader();
        }

#if UDONSHARP
        public override void PostLateUpdate()
        {
            if (localPlayer == null) return;

            // --- Hand Tracking (VR only) ---
            if (localPlayer.IsUserInVR())
            {
                Vector3 lPos = localPlayer.GetTrackingData(
                    VRCPlayerApi.TrackingDataType.LeftHand).position;
                Vector3 rPos = localPlayer.GetTrackingData(
                    VRCPlayerApi.TrackingDataType.RightHand).position;

                // Grab / Move / Split (only with valid tracking)
                if (IsTrackingValid(lPos))
                    ProcessHand(lPos, HandLeft);
                if (IsTrackingValid(rPos))
                    ProcessHand(rPos, HandRight);
            }

            // --- Auto-Merge free mochis ---
            CheckMerge();

            // --- Gravity for free mochis ---
            ApplyGravity();

            // --- Player Collision ---
            Vector3 playerPos = localPlayer.GetPosition();
            Vector3 feetPos = playerPos + Vector3.down * 0.05f;
            float dist = EvaluateSdf(feetPos);

            if (dist < collisionMargin)
            {
                Vector3 normal = EstimateGradient(feetPos);
                float pen = Mathf.Min(collisionMargin - dist, 2.0f);
                float smooth = Mathf.Min(Time.deltaTime * 10f, 1f);
                localPlayer.TeleportTo(
                    playerPos + normal * pen * pushStrength * smooth,
                    localPlayer.GetRotation()
                );
            }

            // --- Shader Sync ---
            SyncShader();
        }
#endif

        // =================================================================
        // Hand Interaction
        // =================================================================
        private void ProcessHand(Vector3 handPos, int hand)
        {
            int grabbed = grab[hand];

            if (grabbed >= 0)
            {
                if (grabbed >= mochiCount)
                {
                    // Held mochi no longer exists (merged away), release
                    grab[hand] = -1;
                    return;
                }

                // Currently grabbing: move mochi to hand
                mochiPos[grabbed] = handPos;

                float pullDist = (handPos - grabOrigin[hand]).magnitude;
                float radius = mochiR[grabbed];

                // Split: shrink the held piece, leave the other half at the grab origin
                if (!splitDone[hand] && pullDist > radius * splitDistance
                    && radius > minRadius * 1.5f && mochiCount < MaxMochi)
                {
                    radius *= SplitRadiusScale;
                    mochiR[grabbed] = radius;
                    SpawnMochi(grabOrigin[hand], radius);
                    splitDone[hand] = true;
                }

                // Release when the hand has pulled too far (radius may have just shrunk)
                if (pullDist > radius * releaseDistance)
                    grab[hand] = -1;
                return;
            }

            // Not grabbing: a hand resting inside a mochi for grabDwellTime grabs it
            int closest = FindClosestMochi(handPos);
            bool inside = closest >= 0
                && (handPos - mochiPos[closest]).magnitude < mochiR[closest] * grabThreshold;

            if (!inside)
            {
                dwellTarget[hand] = -1;
                dwell[hand] = 0f;
                return;
            }

            if (dwellTarget[hand] != closest)
            {
                // New target, restart the dwell timer
                dwellTarget[hand] = closest;
                dwell[hand] = 0f;
                return;
            }

            dwell[hand] += Time.deltaTime;
            if (dwell[hand] < grabDwellTime) return;

            // Don't grab a mochi the other hand already holds
            if (grab[1 - hand] == closest) return;

            grab[hand] = closest;
            grabOrigin[hand] = mochiPos[closest];
            splitDone[hand] = false;
            dwell[hand] = 0f;
        }

        // =================================================================
        // Merge Logic
        // =================================================================
        private void CheckMerge()
        {
            // Check all pairs of free mochis
            for (int i = 0; i < mochiCount; i++)
            {
                if (IsGrabbed(i)) continue;

                for (int j = i + 1; j < mochiCount; j++)
                {
                    if (IsGrabbed(j)) continue;

                    float dist = (mochiPos[i] - mochiPos[j]).magnitude;
                    float threshold = (mochiR[i] + mochiR[j]) * mergeThreshold;
                    if (dist >= threshold) continue;

                    // Merge j into i (volume conservation): r = cbrt(r_i^3 + r_j^3)
                    float vi = mochiR[i] * mochiR[i] * mochiR[i];
                    float vj = mochiR[j] * mochiR[j] * mochiR[j];
                    float totalV = vi + vj;
                    float newR = CubeRoot(totalV);
                    mochiR[i] = newR;

                    // Volume-weighted centre, kept on or above the ground so the
                    // bigger mochi does not spend a frame sunk into the floor
                    Vector3 c = (mochiPos[i] * vi + mochiPos[j] * vj) / totalV;
                    mochiPos[i] = new Vector3(c.x, Mathf.Max(c.y, newR), c.z);

                    RemoveMochi(j);
                    j--; // Re-check this index
                }
            }
        }

        // =================================================================
        // Gravity
        // =================================================================
        private void ApplyGravity()
        {
            // Exponential settle toward the resting height, frame-rate independent
            float settle = 1f - Mathf.Exp(-gravity * Time.deltaTime);

            for (int i = 0; i < mochiCount; i++)
            {
                if (IsGrabbed(i)) continue;

                // Resting height: centre at Y = radius (sphere touching the ground)
                float targetY = mochiR[i];
                float currentY = mochiPos[i].y;
                if (currentY > targetY + 0.001f)
                {
                    mochiPos[i] = new Vector3(
                        mochiPos[i].x,
                        Mathf.Lerp(currentY, targetY, settle),
                        mochiPos[i].z
                    );
                }
                else if (currentY < targetY)
                {
                    // Slightly below ground, push up
                    mochiPos[i] = new Vector3(mochiPos[i].x, targetY, mochiPos[i].z);
                }
            }
        }

        // =================================================================
        // SDF Evaluation (for player collision) — mirrors the shader map()
        // =================================================================
        public float EvaluateSdf(Vector3 p)
        {
            float ground = p.y;
            float mochi = 1e10f;

            for (int i = 0; i < mochiCount; i++)
            {
                float d = (p - mochiPos[i]).magnitude - mochiR[i];
                mochi = OpSmoothUnion(mochi, d, blendK);
            }

            return OpSmoothUnion(ground, mochi, groundK);
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
        // Helpers
        // =================================================================
        private void SpawnMochi(Vector3 pos, float radius)
        {
            if (mochiCount >= MaxMochi) return;
            mochiPos[mochiCount] = pos;
            mochiR[mochiCount] = radius;
            mochiCount++;
        }

        private void RemoveMochi(int index)
        {
            if (index < 0 || index >= mochiCount) return;

            // Shift everything after index
            for (int i = index; i < mochiCount - 1; i++)
            {
                mochiPos[i] = mochiPos[i + 1];
                mochiR[i] = mochiR[i + 1];
            }
            mochiCount--;

            // Fix hand references into the shifted array
            for (int h = 0; h < HandCount; h++)
            {
                if (grab[h] == index) grab[h] = -1;
                else if (grab[h] > index) grab[h]--;

                if (dwellTarget[h] == index) { dwellTarget[h] = -1; dwell[h] = 0f; }
                else if (dwellTarget[h] > index) dwellTarget[h]--;
            }
        }

        private int FindClosestMochi(Vector3 pos)
        {
            int closest = -1;
            float closestDist = 1e10f;
            for (int i = 0; i < mochiCount; i++)
            {
                float d = (pos - mochiPos[i]).magnitude;
                if (d < closestDist)
                {
                    closestDist = d;
                    closest = i;
                }
            }
            return closest;
        }

        private bool IsGrabbed(int index)
        {
            return grab[HandLeft] == index || grab[HandRight] == index;
        }

        private float CubeRoot(float x)
        {
            // cbrt via pow — works for positive values
            return Mathf.Pow(x, 1f / 3f);
        }

        private bool IsTrackingValid(Vector3 pos)
        {
            // Tracking returns exactly (0,0,0) when lost. The mochis sit around
            // the world origin, so only the exact zero is rejected (a radius
            // test would carve a dead zone out of the play area).
            return pos != Vector3.zero;
        }

        // Polynomial smooth minimum, identical to opSmoothUnion in the shader
        private float OpSmoothUnion(float d1, float d2, float k)
        {
            if (k < 0.0001f) return Mathf.Min(d1, d2);
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 - d2), 0f) * invK;
            return Mathf.Min(d1, d2) - h * h * k * 0.25f;
        }

        // =================================================================
        // Shader Sync
        // =================================================================
        private void SyncShader()
        {
            if (mat == null) return;

            for (int i = 0; i < MaxMochi; i++)
            {
                if (i < mochiCount)
                    shaderData[i] = new Vector4(mochiPos[i].x, mochiPos[i].y, mochiPos[i].z, mochiR[i]);
                else
                    shaderData[i] = Vector4.zero;
            }

            mat.SetVectorArray("_MochiData", shaderData);
            mat.SetFloat("_MochiCount", (float)mochiCount);

            // Blend factors live here, not in the material: what the player
            // collides with is exactly what is rendered
            mat.SetFloat("_BlendK", blendK);
            mat.SetFloat("_GroundK", groundK);
        }
    }
}
