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
//   - Desktop: click (Use) on a mochi -> the point on the view ray nearest
//     its centre becomes a virtual right hand, so grab / drag / split /
//     merge run through the same ProcessHand; release the button to drop
//   - Grab button (right click on desktop, grip in VR) while holding ->
//     split it right there, no pull needed
//   - Walk into a mochi -> the player is pushed out sideways (body sampled
//     from the feet to the eyes, the deepest sample decides the direction),
//     the mochi gives way by the mass ratio and the shader dents it around
//     the player's body capsule
//
// The SDF evaluated here (EvaluateSdf) is the same formula the shader
// renders: SmoothUnion(ground, SmoothUnion(mochi_i, blendK), groundK).
// blendK / groundK are pushed to the material every frame, so this script
// is the single source of truth for both rendering and collision.
//
// Network: owner-authoritative, manual sync. mochiPos / mochiR / mochiCount
//   are [UdonSynced]; the owner runs gravity and merging and serializes at
//   10 Hz while anything changed. Grabbing or walking into a mochi takes
//   ownership (once per grab / contact, never every frame), so the last
//   player to act drives the state; everyone else renders what they
//   receive and is still pushed out by it. Late joiners spawn nothing and
//   wait for the owner's state. Each player's body dent is local.
//
// Author: Moroya Sakamoto
// =============================================================================

using UnityEngine;

#if UDONSHARP
using VRC.SDKBase;
using VRC.Udon.Common;
using UdonSharp;
#endif

namespace AliceSDF.Samples
{
#if UDONSHARP
    [UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
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

        // Remaining penetration (m) under which the player is left alone
        private const float PushDeadBand = 0.005f;

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

        [Header("Desktop")]
        [Tooltip("How far the view ray looks for a mochi on click (m)")]
        public float cursorMaxDist = 4.0f;

        [Header("Debug")]
        [Tooltip("Debug.Log one line per event (grab / split / release / merge / click / push) as [Mochi] ..., readable in the VRChat client output_log")]
        public bool logEvents = false;

        [Header("Player Collision")]
        public float collisionMargin = 0.1f;
        [Range(0.5f, 1.5f)]
        public float pushStrength = 1.0f;
        [Tooltip("Samples along the body axis, feet to eyes (a single foot sample only sees the underside of a mochi)")]
        public int bodySamples = 5;
        [Tooltip("Eye height used when the avatar's cannot be read (m)")]
        public float fallbackEyeHeight = 1.6f;

        [Header("Player Body")]
        [Tooltip("Radius of the body capsule the shader dents the mochis with (m)")]
        public float playerRadius = 0.3f;
        [Tooltip("Player mass for the push split with the mochi (kg)")]
        public float playerMass = 60f;
        [Tooltip("Mochi density for its mass, 4/3 pi r^3 * density (kg/m^3)")]
        public float mochiDensity = 1000f;
        [Tooltip("Smooth-subtraction factor of the body dent (sent to shader _PlayerDentK)")]
        public float dentK = 0.12f;

        // Mochi state arrays (indices 0..mochiCount-1 are live), the synced state
#if UDONSHARP
        [UdonSynced] private Vector3[] mochiPos;
        [UdonSynced] private float[] mochiR;
        [UdonSynced] private int mochiCount = 0;
#else
        private Vector3[] mochiPos;
        private float[] mochiR;
        private int mochiCount = 0;
#endif
        // Set by every state change; the owner serializes at SyncInterval
        private bool stateDirty;
        private const float SyncInterval = 0.1f;

        // Per-hand state, indexed by HandLeft / HandRight
        private int[] grab;            // mochi index held, -1 = not grabbing
        private Vector3[] grabOrigin;  // where the held mochi was grabbed
        private bool[] splitDone;      // one split per grab
        private float[] maxPull;       // farthest the hand went from the grab origin, for the release log
        private float[] dwell;         // seconds the hand has been inside dwellTarget
        private int[] dwellTarget;     // mochi the hand is dwelling in, -1 = none

#if UDONSHARP
        // Desktop cursor: Use button state and the fixed distance of the
        // virtual hand along the view ray (-1 = no cursor)
        private bool useHeld;
        private float cursorDist;
        private float lastSyncTime;
        private bool pushing;   // in contact this frame chain (ownership taken at its start)
        private int receivedCount;   // mochiCount at the last logged deserialization
        // Mochi the player is currently pushing (-1 = none) and when that was
        // last logged: one line per contact, and a change of target no more
        // than every PushLogInterval (standing in the neck between two mochis
        // flips the nearest one every frame)
        private int pushingMochi;
        private float pushLogTime;
        private const float PushLogInterval = 0.5f;
#endif

        // Shader data
        private Vector4[] shaderData;
        private Material mat;
        private Vector4 playerCapA;   // xyz = feet end of the body capsule, w = radius (0 = none)
        private Vector4 playerCapB;   // xyz = head end

#if UDONSHARP
        private VRCPlayerApi localPlayer;
#endif

        void Start()
        {
            // A late joiner may have received the owner's arrays already
            if (mochiPos == null) mochiPos = new Vector3[MaxMochi];
            if (mochiR == null) mochiR = new float[MaxMochi];
            shaderData = new Vector4[MaxMochi];
            stateDirty = false;

            grab = new int[HandCount];
            grabOrigin = new Vector3[HandCount];
            splitDone = new bool[HandCount];
            maxPull = new float[HandCount];
            dwell = new float[HandCount];
            dwellTarget = new int[HandCount];
            for (int h = 0; h < HandCount; h++)
            {
                grab[h] = -1;
                dwellTarget[h] = -1;
            }

#if UDONSHARP
            localPlayer = Networking.LocalPlayer;
            useHeld = false;
            cursorDist = -1f;
            pushingMochi = -1;
            pushLogTime = -1f;
            lastSyncTime = -1f;
            pushing = false;
            receivedCount = -1;
#endif

            // Place initial mochis (same layout as mochi.asdf.json). Only the
            // owner (the first player in the instance): a late joiner would
            // overwrite the state it is about to receive.
            if (IsAuthority())
            {
                SpawnMochi(new Vector3(-0.6f, 0.35f, 0.5f), 0.35f);
                SpawnMochi(new Vector3( 0.5f, 0.30f, 0.3f), 0.30f);
                SpawnMochi(new Vector3( 0.0f, 0.28f,-0.4f), 0.28f);
                SpawnMochi(new Vector3(-0.9f, 0.40f,-0.2f), 0.40f);
                SpawnMochi(new Vector3( 0.4f, 0.25f,-0.8f), 0.25f);
            }

            // No player yet: w = 0 tells the shader not to dent
            playerCapA = Vector4.zero;
            playerCapB = Vector4.zero;

            MeshRenderer rend = GetComponent<MeshRenderer>();
            if (rend != null)
                mat = rend.material;
            else
                Debug.LogWarning("[ALICE-SDF] SampleMochi_Collider: No MeshRenderer found. Shader sync disabled.");

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
            else
            {
                ProcessDesktopCursor();
            }

            // --- Auto-Merge + Gravity: the owner simulates, everyone else
            // receives the result ---
            if (IsAuthority())
            {
                CheckMerge();
                ApplyGravity();
            }

            // --- Player Collision ---
            Vector3 playerPos = localPlayer.GetPosition();
            float eyeHeight = localPlayer.GetAvatarEyeHeightAsMeters();
            if (eyeHeight <= 0f) eyeHeight = fallbackEyeHeight;
            Vector3 push = PlayerPushOut(playerPos, eyeHeight, Time.deltaTime);
            if (push != Vector3.zero)
            {
                // Pushing a mochi moves it for everyone: take ownership at the
                // start of the contact
                if (!pushing)
                {
                    pushing = true;
                    TakeAuthority();
                }
                // The mochi takes its share of the separation, the player the rest
                int target = FindClosestMochi(DeepestBodySample(playerPos, eyeHeight));
                float yielded = YieldMochi(playerPos, eyeHeight, push);
                if (target >= 0 && target != pushingMochi
                    && (pushingMochi < 0 || Time.time - pushLogTime >= PushLogInterval))
                {
                    LogEvent("push #" + target + " r=" + F(mochiR[target]) + " player at " + F(playerPos)
                             + " mochi share " + F(yielded));
                    pushingMochi = target;
                    pushLogTime = Time.time;
                }
                localPlayer.TeleportTo(playerPos + push * (1f - yielded), localPlayer.GetRotation());
            }
            else
            {
                pushing = false;
                pushingMochi = -1;
            }

            // Body capsule for the shader dent (feet to eyes, radius playerRadius)
            float capTop = eyeHeight - playerRadius;
            if (capTop < playerRadius) capTop = playerRadius;
            playerCapA = new Vector4(playerPos.x, playerPos.y + playerRadius, playerPos.z, playerRadius);
            playerCapB = new Vector4(playerPos.x, playerPos.y + capTop, playerPos.z, 0f);

            // --- Network: the owner sends what changed, at most every SyncInterval ---
            if (stateDirty && IsAuthority() && Time.time - lastSyncTime >= SyncInterval)
            {
                RequestSerialization();
                stateDirty = false;
                lastSyncTime = Time.time;
            }

            // --- Shader Sync ---
            SyncShader();
        }

        // A hand may hold an index that the owner's state no longer has
        public override void OnDeserialization()
        {
            for (int h = 0; h < HandCount; h++)
            {
                if (grab[h] >= mochiCount) grab[h] = -1;
                if (dwellTarget[h] >= mochiCount) { dwellTarget[h] = -1; dwell[h] = 0f; }
            }
            if (mochiCount != receivedCount)
            {
                VRCPlayerApi owner = Networking.GetOwner(gameObject);
                LogEvent("received " + mochiCount + " mochis (was " + receivedCount + ") from "
                         + (owner != null ? owner.displayName : "?"));
                receivedCount = mochiCount;
            }
        }
#endif

        // =================================================================
        // Authority: who simulates and serializes (the object's owner)
        // =================================================================
        public bool IsAuthority()
        {
#if UDONSHARP
            return Networking.IsOwner(localPlayer, gameObject);
#else
            return true;
#endif
        }

        private void TakeAuthority()
        {
#if UDONSHARP
            if (!Networking.IsOwner(localPlayer, gameObject))
                Networking.SetOwner(localPlayer, gameObject);
#endif
        }

        private void MarkDirty()
        {
            stateDirty = true;
        }

        // State changed since the owner last serialized
        public bool HasUnsentChanges()
        {
            return stateDirty;
        }

        // =================================================================
        // Desktop cursor (no hand tracking): the Use button and the view ray
        // =================================================================
#if UDONSHARP
        public override void InputUse(bool value, UdonInputEventArgs args)
        {
            useHeld = value;
            if (!value)
            {
                cursorDist = -1f;
                ReleaseHand(HandRight);
            }
        }

        // Grab button (right click on desktop, grip in VR): split what the
        // hand holds, no pull needed. In VR the grip of either hand splits
        // that hand's mochi; on desktop it is always the cursor hand.
        public override void InputGrab(bool value, UdonInputEventArgs args)
        {
            if (!value) return;
            int hand = HandRight;
            if (localPlayer != null && localPlayer.IsUserInVR() && args.handType == HandType.LEFT) hand = HandLeft;
            if (grab[hand] < 0 || grab[hand] >= mochiCount) return;
            int g = grab[hand];
            if (SplitHeld(hand, "grab button"))
                splitDone[hand] = true;
            else
                LogEvent("split refused #" + g + " r=" + F(mochiR[g]) + " (min r " + F(minRadius * 1.5f) + " or " + MaxMochi + " mochis)");
        }

        // While Use is held, the point on the view ray at the distance fixed
        // on the click is the right hand: ProcessHand grabs it after the
        // dwell, drags it as the view turns, splits it on a fast turn
        private void ProcessDesktopCursor()
        {
            if (!useHeld) return;
            VRCPlayerApi.TrackingData head = localPlayer.GetTrackingData(VRCPlayerApi.TrackingDataType.Head);
            Vector3 o = head.position;
            Vector3 dir = head.rotation * Vector3.forward;
            if (cursorDist < 0f)
            {
                cursorDist = CursorDistance(o, dir);
                if (cursorDist < 0f)
                {
                    // Clicked past every mochi: this press does nothing
                    LogEvent("click missed (view from " + F(o) + " toward " + F(dir) + ")");
                    useHeld = false;
                    return;
                }
                LogEvent("click hit, cursor " + F(cursorDist) + " m along the view ray");
            }
            Vector3 cursor = o + dir * cursorDist;
            // A steep view ray would drag the held mochi under the floor
            int held = grab[HandRight];
            if (held >= 0 && held < mochiCount && cursor.y < mochiR[held])
                cursor = new Vector3(cursor.x, mochiR[held], cursor.z);
            ProcessHand(cursor, HandRight);
        }
#endif

        // Distance along the ray of the virtual hand for a click: the point
        // nearest the centre of the mochi the ray hits (inside it, so the
        // grab threshold is met), or -1 when the ray misses. Clicking near a
        // mochi's rim (outside grabThreshold x r) grabs nothing, like a hand
        // resting on the rim would.
        public float CursorDistance(Vector3 o, Vector3 dir)
        {
            float t = RaymarchMochi(o, dir, cursorMaxDist);
            if (t < 0f) return -1f;
            int i = FindClosestMochi(o + dir * t);
            if (i < 0) return -1f;
            float along = Vector3.Dot(mochiPos[i] - o, dir);
            return along > 0f ? along : -1f;
        }

        // Sphere-traced hit distance of the ray against the mochis (no ground),
        // -1 beyond maxDist
        public float RaymarchMochi(Vector3 o, Vector3 dir, float maxDist)
        {
            float t = 0f;
            for (int i = 0; i < 64; i++)
            {
                float d = EvaluateMochiSdf(o + dir * t);
                if (d < 0.001f) return t;
                t += d;
                if (t > maxDist) return -1f;
            }
            return -1f;
        }

        private void ReleaseHand(int hand)
        {
            if (grab[hand] >= 0 && grab[hand] < mochiCount)
            {
                float r = mochiR[grab[hand]];
                string pull = splitDone[hand]
                    ? "already split"
                    : "max pull " + F(maxPull[hand]) + " m, split at " + F(r * splitDistance) + " m";
                LogEvent("release #" + grab[hand] + " hand " + hand + " (button up, " + pull + ")");
            }
            grab[hand] = -1;
            dwellTarget[hand] = -1;
            dwell[hand] = 0f;
        }

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
                MarkDirty();

                float pullDist = (handPos - grabOrigin[hand]).magnitude;
                if (pullDist > maxPull[hand]) maxPull[hand] = pullDist;
                float radius = mochiR[grabbed];

                // Split: pulled far enough (once per grab)
                if (!splitDone[hand] && pullDist > radius * splitDistance
                    && SplitHeld(hand, "pulled " + F(pullDist) + " m"))
                {
                    splitDone[hand] = true;
                    radius = mochiR[grabbed];
                }

                // Release when the hand has pulled too far (radius may have just shrunk)
                if (pullDist > radius * releaseDistance)
                {
                    grab[hand] = -1;
                    LogEvent("release #" + grabbed + " hand " + hand + " (pulled " + F(pullDist) + " m)");
                }
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

            TakeAuthority();
            grab[hand] = closest;
            grabOrigin[hand] = mochiPos[closest];
            splitDone[hand] = false;
            maxPull[hand] = 0f;
            dwell[hand] = 0f;
            LogEvent("grab #" + closest + " r=" + F(mochiR[closest]) + " hand " + hand + " at " + F(handPos));
        }

        // Split the mochi a hand holds: the held piece shrinks to r*cbrt(0.5)
        // and the other half is left at the grab origin (volume conserved).
        // False when it is already at the minimum size or the array is full.
        private bool SplitHeld(int hand, string how)
        {
            int grabbed = grab[hand];
            if (grabbed < 0 || grabbed >= mochiCount) return false;
            float radius = mochiR[grabbed];
            if (radius <= minRadius * 1.5f || mochiCount >= MaxMochi) return false;
            radius *= SplitRadiusScale;
            mochiR[grabbed] = radius;
            SpawnMochi(grabOrigin[hand], radius);
            LogEvent("split #" + grabbed + " -> #" + (mochiCount - 1) + " r=" + F(radius) + " each, hand " + hand
                     + " (" + how + ")");
            return true;
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
                    MarkDirty();

                    // Volume-weighted centre, kept on or above the ground so the
                    // bigger mochi does not spend a frame sunk into the floor
                    Vector3 c = (mochiPos[i] * vi + mochiPos[j] * vj) / totalV;
                    mochiPos[i] = new Vector3(c.x, Mathf.Max(c.y, newR), c.z);

                    LogEvent("merge #" + j + " into #" + i + " r=" + F(newR) + " at " + F(mochiPos[i])
                             + ", " + (mochiCount - 1) + " mochis left");
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
                    MarkDirty();
                }
                else if (currentY < targetY)
                {
                    // Slightly below ground, push up
                    mochiPos[i] = new Vector3(mochiPos[i].x, targetY, mochiPos[i].z);
                    MarkDirty();
                }
            }
        }

        // =================================================================
        // SDF Evaluation (for player collision) — mirrors the shader map()
        // =================================================================
        public float EvaluateSdf(Vector3 p)
        {
            return OpSmoothUnion(p.y, EvaluateMochiSdf(p), groundK);
        }

        // =================================================================
        // Player Collision — the mochis only, sampled along the body
        // =================================================================
        // The ground plane is part of the rendered SDF (EvaluateSdf) but the
        // player already stands on the world's floor collider at the same
        // height: colliding with it made every frame on the floor a
        // penetration and the player bobbed. The body is sampled from the
        // feet (playerPos) to the eyes: a single foot sample sits below every
        // mochi's centre, so walking in from the side barely registered and
        // what push there was pointed up (onto the mochi). The deepest sample
        // decides the direction; a downward component is dropped because the
        // floor is VRChat's and a downward teleport only fights it.
        // Returns the displacement to apply this frame, zero when clear.
        public Vector3 PlayerPushOut(Vector3 playerPos, float eyeHeight, float dt)
        {
            Vector3 minP = DeepestBodySample(playerPos, eyeHeight);
            float minDist = EvaluateMochiSdf(minP);
            if (minDist >= collisionMargin) return Vector3.zero;

            // The push shrinks geometrically as the margin is approached and
            // never reaches exactly zero: without a dead band the player is
            // teleported every frame forever (ClientSim log: same position,
            // hundreds of times)
            float pen = Mathf.Min(collisionMargin - minDist, 2.0f);
            if (pen < PushDeadBand) return Vector3.zero;

            // Sideways whenever the surface allows it: lifting the player only
            // hands them to gravity, which drops them back into the next push
            // (the bobbing loop). Straight up only on top of a mochi, never down.
            Vector3 normal = EstimateGradient(minP);
            Vector3 flat = new Vector3(normal.x, 0f, normal.z);
            float flatLen = flat.magnitude;
            if (flatLen > 0.3f) normal = flat / flatLen;
            else if (normal.y < 0f) normal = Vector3.up;

            float smooth = Mathf.Min(dt * 10f, 1f);
            return normal * pen * pushStrength * smooth;
        }

        // The body-axis sample (feet = playerPos, eyes = playerPos + eyeHeight)
        // deepest inside the mochis
        public Vector3 DeepestBodySample(Vector3 playerPos, float eyeHeight)
        {
            int n = bodySamples < 2 ? 2 : bodySamples;
            float minDist = 1e10f;
            Vector3 minP = playerPos;
            for (int i = 0; i < n; i++)
            {
                Vector3 s = playerPos + Vector3.up * (eyeHeight * i / (n - 1));
                float d = EvaluateMochiSdf(s);
                if (d < minDist)
                {
                    minDist = d;
                    minP = s;
                }
            }
            return minP;
        }

        // The mochi gives way: the separation the push asks for is split by
        // mass, the mochi (4/3 pi r^3 * density) slides on the floor by its
        // share and the player takes the rest. Returns the mochi's share in
        // [0, 1); 0 when nothing yields (no mochi, or it is held in a hand).
        public float YieldMochi(Vector3 playerPos, float eyeHeight, Vector3 push)
        {
            int i = FindClosestMochi(DeepestBodySample(playerPos, eyeHeight));
            if (i < 0 || IsGrabbed(i)) return 0f;
            float r = mochiR[i];
            float mochiMass = mochiDensity * 4.18879f * r * r * r;
            float share = playerMass / (playerMass + mochiMass);
            mochiPos[i] = mochiPos[i] - new Vector3(push.x, 0f, push.z) * share;
            MarkDirty();
            return share;
        }

        // The mochis alone (no ground plane): what the player collides with.
        // No body dent here: the dent is where the player already is.
        public float EvaluateMochiSdf(Vector3 p)
        {
            float mochi = 1e10f;

            for (int i = 0; i < mochiCount; i++)
            {
                float d = (p - mochiPos[i]).magnitude - mochiR[i];
                mochi = OpSmoothUnion(mochi, d, blendK);
            }

            return mochi;
        }

        private Vector3 EstimateGradient(Vector3 p)
        {
            float e = 0.02f;
            float dx = EvaluateMochiSdf(new Vector3(p.x + e, p.y, p.z))
                     - EvaluateMochiSdf(new Vector3(p.x - e, p.y, p.z));
            float dy = EvaluateMochiSdf(new Vector3(p.x, p.y + e, p.z))
                     - EvaluateMochiSdf(new Vector3(p.x, p.y - e, p.z));
            float dz = EvaluateMochiSdf(new Vector3(p.x, p.y, p.z + e))
                     - EvaluateMochiSdf(new Vector3(p.x, p.y, p.z - e));
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
            MarkDirty();
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
            MarkDirty();

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

        // One line per event in the client log (grep "[Mochi]"), off by default
        private void LogEvent(string what)
        {
            if (logEvents) Debug.Log("[Mochi] " + what);
        }

        private string F(float v)
        {
            return v.ToString("F2");
        }

        private string F(Vector3 v)
        {
            return "(" + F(v.x) + ", " + F(v.y) + ", " + F(v.z) + ")";
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

            // Body capsule the shader dents the mochis with (w = 0 until the
            // first PostLateUpdate: no player, no dent)
            mat.SetVector("_PlayerCapA", playerCapA);
            mat.SetVector("_PlayerCapB", playerCapB);
            mat.SetFloat("_PlayerDentK", dentK);
        }
    }
}
