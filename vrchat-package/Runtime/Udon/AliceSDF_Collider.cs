// =============================================================================
// AliceSDF_Collider.cs - SDF Collision for VRChat (UdonSharp)
// =============================================================================
// Evaluates SDF at the player's position and pushes them out of solid geometry.
// This makes players able to STAND ON and COLLIDE WITH mathematical surfaces.
//
// How it works:
//   1. Get player position P
//   2. Compute d = SDF(P)
//   3. If d < 0 (inside), compute gradient ∇SDF(P)
//   4. Push player along ∇SDF by |d| to reach surface
//
// Usage:
//   1. Attach this script to a GameObject in your VRChat world
//   2. Override the Evaluate() method with your SDF (or use ALICE-Baker)
//   3. Set collision margin and push strength in Inspector
//
// Performance:
//   - Runs at FixedUpdate (50Hz), not every frame
//   - Single SDF eval + 6 gradient samples = 7 evaluations per tick
//   - Typical cost: < 0.01ms per player
//
// Requires: VRChat SDK + UdonSharp
// If VRC SDK is not installed, this file compiles as a stub (MonoBehaviour).
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
        [Header("Collision Settings")]
        [Tooltip("Distance margin around the SDF surface. Player stops this far from the surface.")]
        public float collisionMargin = 0.1f;

        [Tooltip("How strongly to push the player out. 1.0 = exact correction.")]
        [Range(0.5f, 1.5f)]
        public float pushStrength = 1.0f;

        [Tooltip("Epsilon for gradient estimation (smaller = more precise, slower).")]
        public float gradientEps = 0.02f;

        [Tooltip("Maximum push distance per frame to prevent teleporting through walls.")]
        public float maxPushDistance = 2.0f;

        [Header("Performance")]
        [Tooltip("Only check collision every N fixed updates (1 = every tick, 2 = half rate).")]
        [Range(1, 4)]
        public int updateInterval = 1;

        [Header("Debug")]
        [Tooltip("Enable debug logging in console.")]
        public bool debugMode = false;

        // Internal state
#if UDONSHARP
        private VRCPlayerApi _localPlayer;
#endif
        private int _tickCounter = 0;

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

            // Union = stand on ground OR on sphere
            return Mathf.Min(ground, sphere);
        }

        // =====================================================================
        // Core Logic
        // =====================================================================

#if UDONSHARP
        void Start()
        {
            _localPlayer = Networking.LocalPlayer;
        }

        public override void PostLateUpdate()
        {
            if (_localPlayer == null) return;

            _tickCounter++;
            if (_tickCounter % updateInterval != 0) return;

            Vector3 playerPos = _localPlayer.GetPosition();

            // Evaluate SDF at player feet (slightly below center)
            Vector3 feetPos = playerPos + Vector3.down * 0.05f;
            float dist = Evaluate(feetPos);

            // Check if player is inside the SDF surface
            float threshold = collisionMargin;
            if (dist < threshold)
            {
                // Compute gradient (surface normal direction)
                Vector3 normal = EstimateGradient(feetPos);

                // Push distance: how far inside we are
                float penetration = threshold - dist;
                penetration = Mathf.Min(penetration, maxPushDistance);

                // Push player out along the gradient
                Vector3 correction = normal * penetration * pushStrength;
                Vector3 newPos = playerPos + correction;

                _localPlayer.TeleportTo(newPos, _localPlayer.GetRotation());

                if (debugMode)
                {
                    Debug.Log($"[AliceSDF] Push: d={dist:F3}, n={normal}, corr={correction.magnitude:F3}");
                }
            }
        }
#endif

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
