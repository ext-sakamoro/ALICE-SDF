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
// Network: Local-only (each player sees their own mochi state).
//   For multiplayer sync, add [UdonSynced] to mochiData arrays
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
        [Header("Mochi Settings")]
        [Tooltip("SmoothUnion blend factor (match shader _BlendK)")]
        public float blendK = 0.5f;
        [Tooltip("Ground SmoothUnion factor (match shader _GroundK)")]
        public float groundK = 0.15f;
        [Tooltip("Minimum mochi radius (won't split below this)")]
        public float minRadius = 0.1f;
        [Tooltip("Gravity speed for free mochis settling to ground")]
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

        // Mochi state arrays
        private Vector3[] mochiPos;
        private float[] mochiR;
        private int mochiCount = 0;

        // Grab state: -1 = not grabbing
        private int grabLeft = -1;
        private int grabRight = -1;
        private Vector3 grabOriginLeft;
        private Vector3 grabOriginRight;
        private bool splitDoneLeft = false;
        private bool splitDoneRight = false;

        // Dwell timers (prevent accidental grabs)
        private float dwellLeft = 0f;
        private float dwellRight = 0f;
        private int dwellTargetLeft = -1;
        private int dwellTargetRight = -1;

        // Shader data
        private Vector4[] shaderData;
        private Material mat;

#if UDONSHARP
        private VRCPlayerApi localPlayer;
#endif

        void Start()
        {
            mochiPos = new Vector3[16];
            mochiR = new float[16];
            shaderData = new Vector4[16];

            // Place initial mochis
            SpawnMochi(new Vector3(-0.6f, 0.35f, 0.5f), 0.35f);
            SpawnMochi(new Vector3( 0.5f, 0.30f, 0.3f), 0.30f);
            SpawnMochi(new Vector3( 0.0f, 0.28f,-0.4f), 0.28f);
            SpawnMochi(new Vector3(-0.9f, 0.40f,-0.2f), 0.40f);
            SpawnMochi(new Vector3( 0.4f, 0.25f,-0.8f), 0.25f);

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

            // --- Hand Tracking ---
            Vector3 lPos = localPlayer.GetTrackingData(
                VRCPlayerApi.TrackingDataType.LeftHand).position;
            Vector3 rPos = localPlayer.GetTrackingData(
                VRCPlayerApi.TrackingDataType.RightHand).position;

            // --- Grab / Move / Split ---
            ProcessHand(lPos, true);
            ProcessHand(rPos, false);

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
                localPlayer.TeleportTo(
                    playerPos + normal * pen * pushStrength,
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
        private void ProcessHand(Vector3 handPos, bool isLeft)
        {
            int grabbed = isLeft ? grabLeft : grabRight;

            if (grabbed >= 0)
            {
                // Currently grabbing: move mochi to hand
                if (grabbed < mochiCount)
                {
                    mochiPos[grabbed] = handPos;

                    // Check split condition
                    Vector3 origin = isLeft ? grabOriginLeft : grabOriginRight;
                    bool splitDone = isLeft ? splitDoneLeft : splitDoneRight;
                    float pullDist = (handPos - origin).magnitude;
                    float radius = mochiR[grabbed];

                    if (!splitDone && pullDist > radius * splitDistance
                        && radius > minRadius * 1.5f && mochiCount < 16)
                    {
                        // Split: shrink grabbed piece, leave a piece at origin
                        float origR = radius;
                        float newR = origR * 0.7937f; // cbrt(0.5)
                        mochiR[grabbed] = newR;

                        // Spawn the remaining piece at origin
                        SpawnMochi(origin, newR);

                        if (isLeft) splitDoneLeft = true;
                        else splitDoneRight = true;
                    }

                    // Re-read radius (may have changed after split)
                    radius = mochiR[grabbed];

                    // Check release condition (hand too far)
                    if (pullDist > radius * releaseDistance)
                    {
                        if (isLeft) { grabLeft = -1; }
                        else { grabRight = -1; }
                    }
                }
                else
                {
                    // Invalid grab index (mochi was removed), release
                    if (isLeft) grabLeft = -1;
                    else grabRight = -1;
                }
            }
            else
            {
                // Not grabbing: check for new grab via dwell
                int closest = FindClosestMochi(handPos);
                if (closest >= 0)
                {
                    float dist = (handPos - mochiPos[closest]).magnitude;
                    if (dist < mochiR[closest] * grabThreshold)
                    {
                        // Hand is inside mochi — track dwell
                        int prevTarget = isLeft ? dwellTargetLeft : dwellTargetRight;
                        if (prevTarget == closest)
                        {
                            float dwell = (isLeft ? dwellLeft : dwellRight) + Time.deltaTime;
                            if (isLeft) dwellLeft = dwell;
                            else dwellRight = dwell;

                            if (dwell >= grabDwellTime)
                            {
                                // Grab!
                                // Don't grab if other hand already has it
                                int otherGrab = isLeft ? grabRight : grabLeft;
                                if (otherGrab != closest)
                                {
                                    if (isLeft)
                                    {
                                        grabLeft = closest;
                                        grabOriginLeft = mochiPos[closest];
                                        splitDoneLeft = false;
                                        dwellLeft = 0f;
                                    }
                                    else
                                    {
                                        grabRight = closest;
                                        grabOriginRight = mochiPos[closest];
                                        splitDoneRight = false;
                                        dwellRight = 0f;
                                    }
                                }
                            }
                        }
                        else
                        {
                            // New target, reset dwell
                            if (isLeft) { dwellTargetLeft = closest; dwellLeft = 0f; }
                            else { dwellTargetRight = closest; dwellRight = 0f; }
                        }
                    }
                    else
                    {
                        // Hand outside all mochis, reset dwell
                        if (isLeft) { dwellTargetLeft = -1; dwellLeft = 0f; }
                        else { dwellTargetRight = -1; dwellRight = 0f; }
                    }
                }
                else
                {
                    if (isLeft) { dwellTargetLeft = -1; dwellLeft = 0f; }
                    else { dwellTargetRight = -1; dwellRight = 0f; }
                }
            }
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

                    if (dist < threshold)
                    {
                        // Merge j into i (volume conservation)
                        float vi = mochiR[i] * mochiR[i] * mochiR[i];
                        float vj = mochiR[j] * mochiR[j] * mochiR[j];
                        float totalV = vi + vj;

                        // New radius: r = cbrt(v)
                        mochiR[i] = CubeRoot(totalV);

                        // Weighted average position
                        mochiPos[i] = (mochiPos[i] * vi + mochiPos[j] * vj) / totalV;

                        // Remove j
                        RemoveMochi(j);
                        j--; // Re-check this index
                    }
                }
            }
        }

        // =================================================================
        // Gravity
        // =================================================================
        private void ApplyGravity()
        {
            for (int i = 0; i < mochiCount; i++)
            {
                if (IsGrabbed(i)) continue;

                // Settle to ground: center should be at Y = radius
                float targetY = mochiR[i];
                float currentY = mochiPos[i].y;
                if (currentY > targetY + 0.001f)
                {
                    mochiPos[i] = new Vector3(
                        mochiPos[i].x,
                        Mathf.Lerp(currentY, targetY, Time.deltaTime * gravity),
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
        // SDF Evaluation (for player collision)
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
            if (mochiCount >= 16) return;
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

            // Fix grab references
            if (grabLeft == index) grabLeft = -1;
            else if (grabLeft > index) grabLeft--;

            if (grabRight == index) grabRight = -1;
            else if (grabRight > index) grabRight--;

            // Fix dwell target references
            if (dwellTargetLeft == index) { dwellTargetLeft = -1; dwellLeft = 0f; }
            else if (dwellTargetLeft > index) dwellTargetLeft--;

            if (dwellTargetRight == index) { dwellTargetRight = -1; dwellRight = 0f; }
            else if (dwellTargetRight > index) dwellTargetRight--;
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
            return (grabLeft == index || grabRight == index);
        }

        private float CubeRoot(float x)
        {
            // cbrt via pow — works for positive values
            return Mathf.Pow(x, 1f / 3f);
        }

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

            for (int i = 0; i < 16; i++)
            {
                if (i < mochiCount)
                    shaderData[i] = new Vector4(mochiPos[i].x, mochiPos[i].y, mochiPos[i].z, mochiR[i]);
                else
                    shaderData[i] = Vector4.zero;
            }

            mat.SetVectorArray("_MochiData", shaderData);
            mat.SetFloat("_MochiCount", (float)mochiCount);
        }
    }
}
