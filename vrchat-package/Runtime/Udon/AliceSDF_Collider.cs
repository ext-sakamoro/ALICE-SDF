// =============================================================================
// AliceSDF_Collider.cs - SDF Collision for VRChat (UdonSharp)
// =============================================================================
// Evaluates an SDF along the player's body and pushes the player out of
// solid geometry, so players collide with mathematical surfaces.
//
// How it works (the Mochi rule):
//   1. Sample the body from the feet to the eyes (bodySamples points)
//   2. Of the samples inside the margin, the deepest WALL contact decides
//      (the feet resting on an SDF floor would otherwise mask the chest)
//   3. A wall-like surface (gradient mostly horizontal) pushes sideways,
//      geometrically smoothed, with a dead band so the push stops exactly
//   4. A floor-like surface (gradient mostly up) is left alone: standing on
//      an SDF needs a Unity collider under the feet (see the TerrainSculpt
//      sample's TerrainSupport); pushing up every frame only fights gravity
//      and the player bobs
//
// Usage:
//   1. Attach a subclass to the GameObject that renders the SDF
//   2. Override Evaluate() with the SDF the shader draws (UdonSharp resolves
//      virtual calls to the most derived override)
//   3. Set the collision margin and push strength in the Inspector
//
// Cost: bodySamples evaluations + 6 for the gradient per frame.
//
// Requires: VRChat SDK + UdonSharp
// If the VRC SDK is not installed, this file compiles as a MonoBehaviour
// so the law can be checked on a host (HostTests~).
//
// Author: Moroya Sakamoto
// =============================================================================

using UnityEngine;

#if UDONSHARP
using VRC.SDKBase;
using VRC.Udon;
using UdonSharp;
#endif

namespace AliceSDF
{
#if UDONSHARP
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class AliceSDF_Collider : UdonSharpBehaviour
#else
    // Stub: compiles without VRC SDK for packaging / testing
    public class AliceSDF_Collider : MonoBehaviour
#endif
    {
        // Horizontal gradient share above which a contact is a wall
        private const float WallSlope = 0.7f;
        // Remaining penetration (m) under which the player is left alone
        private const float PushDeadBand = 0.005f;

        [Header("Collision Settings")]
        [Tooltip("Distance margin around the SDF surface. The player stops this far from a wall.")]
        public float collisionMargin = 0.1f;

        [Tooltip("How strongly to push the player out. 1.0 = exact correction.")]
        [Range(0.5f, 1.5f)]
        public float pushStrength = 1.0f;

        [Tooltip("Epsilon for gradient estimation (smaller = more precise, slower).")]
        public float gradientEps = 0.02f;

        [Tooltip("Maximum push distance per frame to prevent teleporting through walls.")]
        public float maxPushDistance = 2.0f;

        [Tooltip("Samples along the body axis, feet to eyes (a single foot sample only sees what is below the knees)")]
        public int bodySamples = 5;

        [Tooltip("Eye height used when the avatar's cannot be read (m)")]
        public float fallbackEyeHeight = 1.6f;

        [Header("Animation")]
        [Tooltip("Seconds fed to a time-varying SDF, set every frame from Time.timeSinceLevelLoad (what the shader's _Time.y is); 0 on a host")]
        public float animTime = 0f;

        [Header("Debug")]
        [Tooltip("Debug.Log one line per contact (push) as [SDF] ..., readable in the VRChat client output_log")]
        public bool logEvents = false;

        // Internal state
#if UDONSHARP
        private VRCPlayerApi _localPlayer;
        private bool _pushing;
#endif

        // =====================================================================
        // USER SDF DEFINITION
        // =====================================================================

        /// <summary>
        /// Evaluate the SDF at world position p.
        /// Returns signed distance: negative = inside, positive = outside.
        /// OVERRIDE THIS with your world's SDF formula.
        /// </summary>
        public virtual float Evaluate(Vector3 p)
        {
            // === DEFAULT DEMO ===
            // Ground plane at Y=0
            float ground = p.y;

            // Sphere at (0, 1.5, 0) with radius 1.5
            float sphere = (p - new Vector3(0f, 1.5f, 0f)).magnitude - 1.5f;

            // Union = ground and a sphere
            return Mathf.Min(ground, sphere);
        }

        // =====================================================================
        // Core Logic
        // =====================================================================

#if UDONSHARP
        void Start()
        {
            _localPlayer = Networking.LocalPlayer;
            _pushing = false;
        }

        public override void PostLateUpdate()
        {
            if (_localPlayer == null) return;

            animTime = Time.timeSinceLevelLoad;

            Vector3 playerPos = _localPlayer.GetPosition();
            float eyeHeight = _localPlayer.GetAvatarEyeHeightAsMeters();
            if (eyeHeight <= 0f) eyeHeight = fallbackEyeHeight;

            Vector3 push = PlayerPushOut(playerPos, eyeHeight, Time.deltaTime);
            if (push != Vector3.zero)
            {
                if (!_pushing && logEvents)
                    Debug.Log("[SDF] push at " + playerPos.ToString("F2") + " by " + push.ToString("F3"));
                _pushing = true;
                _localPlayer.TeleportTo(playerPos + push, _localPlayer.GetRotation());
            }
            else
            {
                _pushing = false;
            }
        }
#endif

        // Displacement to apply to the player this frame, zero when clear or
        // when every contact is a floor (see the header). Among the body
        // samples inside the margin, the deepest WALL contact decides: the
        // feet resting on an SDF floor (d = 0) would otherwise mask a chest
        // in a sphere
        public Vector3 PlayerPushOut(Vector3 playerPos, float eyeHeight, float dt)
        {
            int n = bodySamples < 2 ? 2 : bodySamples;
            float bestPen = 0f;
            Vector3 bestDir = Vector3.zero;
            for (int i = 0; i < n; i++)
            {
                Vector3 s = playerPos + Vector3.up * (eyeHeight * i / (n - 1));
                float d = Evaluate(s);
                if (d >= collisionMargin) continue;
                float pen = Mathf.Min(collisionMargin - d, maxPushDistance);
                if (pen <= bestPen) continue;
                Vector3 normal = EstimateGradient(s);
                Vector3 flat = new Vector3(normal.x, 0f, normal.z);
                float flatLen = flat.magnitude;
                if (flatLen <= WallSlope) continue;   // floor or ceiling: not ours
                bestPen = pen;
                bestDir = flat / flatLen;
            }
            if (bestPen < PushDeadBand) return Vector3.zero;

            float smooth = Mathf.Min(dt * 10f, 1f);
            return bestDir * bestPen * pushStrength * smooth;
        }

        // The body-axis sample (feet = playerPos, eyes = playerPos + eyeHeight)
        // deepest inside the SDF
        public Vector3 DeepestBodySample(Vector3 playerPos, float eyeHeight)
        {
            int n = bodySamples < 2 ? 2 : bodySamples;
            float minDist = 1e10f;
            Vector3 minP = playerPos;
            for (int i = 0; i < n; i++)
            {
                Vector3 s = playerPos + Vector3.up * (eyeHeight * i / (n - 1));
                float d = Evaluate(s);
                if (d < minDist)
                {
                    minDist = d;
                    minP = s;
                }
            }
            return minP;
        }

        /// <summary>
        /// Estimate the SDF gradient (surface normal) at point p
        /// using central finite differences.
        /// </summary>
        protected Vector3 EstimateGradient(Vector3 p)
        {
            float e = gradientEps;

            float dx = Evaluate(p + new Vector3(e, 0, 0)) - Evaluate(p - new Vector3(e, 0, 0));
            float dy = Evaluate(p + new Vector3(0, e, 0)) - Evaluate(p - new Vector3(0, e, 0));
            float dz = Evaluate(p + new Vector3(0, 0, e)) - Evaluate(p - new Vector3(0, 0, e));

            Vector3 grad = new Vector3(dx, dy, dz);
            float len = grad.magnitude;
            if (len < 0.0001f) return Vector3.up;
            return grad / len;
        }
    }
}
