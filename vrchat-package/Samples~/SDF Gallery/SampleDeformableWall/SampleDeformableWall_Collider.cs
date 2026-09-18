// =============================================================================
// ALICE-SDF Sample: DeformableWall Collider (UdonSharp)
// =============================================================================
// A wall that dents where it is hit and recovers over time. VR: touch or
// hit it with either hand. Desktop: hold the left button (Use) to punch
// the point you look at. Walk into it and your body presses a dent into it
// while it pushes you back.
//
// The SDF evaluated here (EvaluateSdf) is the same formula the shader
// renders: min(ground, SmoothSubtract(wall, dent_i)) with the dent radius
// dentRadius * strength_i. dentSmooth / dentRadius are pushed to the
// material every frame, so this script is the single source of truth for
// both rendering and collision, and a dent you can see is a dent you can
// stand in.
//
// Impacts: up to 16, each (position, strength); strength starts at 1 and
// decays by exp(-decaySpeed * t); a slot under 0.01 is free. A new impact
// takes a free slot, else the weakest.
//
// Placement: the law is anchored to this object. The ground plane and the
// wall's foot sit at transform.position + groundOffset and the wall faces
// the object's local Z axis (rotate the object to turn the wall); both are
// read once in Start. Dents and the body capsule stay in world space (a
// sphere / capsule is rotation invariant).
//
// Network: owner-authoritative, manual sync. impactPoints / impactCount
//   are [UdonSynced]; whoever hits the wall takes ownership (once per
//   contact, not every frame); the owner decays the strengths and
//   serializes at 10 Hz while any dent is alive. Everyone else decays what
//   they received locally between packets so the recovery looks smooth.
//   Each player's body dent is local.
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
    public class SampleDeformableWall_Collider : UdonSharpBehaviour
#else
    public class SampleDeformableWall_Collider : MonoBehaviour
#endif
    {
        /// <summary>Upper bound of tracked dents. Must match the shader's _ImpactPoints[16].</summary>
        public const int MaxImpacts = 16;

        // Hand slots (the desktop cursor punches with HandRight)
        private const int HandLeft = 0;
        private const int HandRight = 1;
        private const int HandCount = 2;

        // A dent weaker than this is gone (slot free, shader skips it)
        private const float DeadStrength = 0.01f;
        // Remaining penetration (m) under which the player is left alone
        private const float PushDeadBand = 0.005f;

        [Header("Look (pushed to the material every frame when Apply Colors is on)")]
        [Tooltip("Push the colours and textures below to the material every frame (off = the material's own values)")]
        public bool applyColors = true;
        public Color wallColor = new Color(0.82f, 0.78f, 0.72f, 1f);
        [Tooltip("Colour a fresh dent glows with, fading as it recovers")]
        public Color dentGlowColor = new Color(1.0f, 0.6f, 0.3f, 1f);
        public Color groundColor = new Color(0.35f, 0.42f, 0.3f, 1f);
        [Tooltip("Optional texture on the wall (triplanar in the wall's frame, tiles per metre * scale)")]
        public Texture2D wallTexture;
        public float wallTextureScale = 1.0f;
        [Range(0f, 1f)] public float wallTextureStrength = 1.0f;
        [Tooltip("Optional texture on the ground patch under the volume")]
        public Texture2D groundTexture;
        public float groundTextureScale = 0.5f;
        [Range(0f, 1f)] public float groundTextureStrength = 1.0f;

        [Header("Placement")]
        [Tooltip("Ground plane and wall foot relative to this object (the prefab's cube is 8 m tall with the ground at its bottom face). The wall faces the local Z axis")]
        public Vector3 groundOffset = new Vector3(0f, -4f, 0f);

        [Header("Wall Dimensions (sent to the shader)")]
        [Tooltip("Half-width of the wall (m)")]
        public float wallWidth = 5.0f;
        [Tooltip("Half-height of the wall (m); it stands on the ground")]
        public float wallHeight = 2.5f;
        [Tooltip("Half-thickness of the wall (m)")]
        public float wallThickness = 0.2f;

        [Header("Impact Settings")]
        [Tooltip("Hand / cursor must be within this distance of the wall surface to register")]
        public float impactDistance = 0.08f;
        [Tooltip("Minimum time between impacts from the same hand (sec)")]
        public float impactCooldown = 0.15f;
        [Tooltip("Dent recovery: strength decays as exp(-speed * t) (1/sec)")]
        public float decaySpeed = 0.15f;
        [Tooltip("Dent radius at full strength (sent to shader _DentRadius)")]
        public float dentRadius = 0.35f;
        [Tooltip("SmoothSubtraction factor of a dent (sent to shader _DentSmooth)")]
        public float dentSmooth = 0.08f;

        [Header("Desktop")]
        [Tooltip("How far the view ray looks for the wall on a click (m)")]
        public float cursorMaxDist = 4.0f;

        [Header("Debug")]
        [Tooltip("Debug.Log one line per event (impact / click / push / ownership / received) as [Wall] ..., readable in the VRChat client output_log")]
        public bool logEvents = false;

        [Header("Player Collision")]
        public float collisionMargin = 0.1f;
        [Range(0.5f, 1.5f)]
        public float pushStrength = 1.0f;
        [Tooltip("Samples along the body axis, feet to eyes")]
        public int bodySamples = 5;
        [Tooltip("Eye height used when the avatar's cannot be read (m)")]
        public float fallbackEyeHeight = 1.6f;

        [Header("Player Body")]
        [Tooltip("Radius of the body capsule the shader presses into the wall (m)")]
        public float playerRadius = 0.3f;
        [Tooltip("Smooth-subtraction factor of the body dent (sent to shader _PlayerDentK)")]
        public float bodyDentK = 0.12f;

        // Dents (xyz = position, w = strength 0..1), the synced state
#if UDONSHARP
        [UdonSynced] private Vector4[] impactPoints;
        [UdonSynced] private int impactCount = 0;
#else
        private Vector4[] impactPoints;
        private int impactCount = 0;
#endif
        // Set by every state change; the owner serializes at SyncInterval
        private bool stateDirty;
        private const float SyncInterval = 0.1f;

        // Per-hand cooldown and contact (ownership once per contact)
        private float[] lastImpactTime;
        private bool[] touching;

#if UDONSHARP
        private bool useHeld;
        private bool missLogged;
        private float lastSyncTime;
        private int receivedCount;
        private bool pushing;
#endif

        // Cached references
        private Material mat;
        private Vector3 origin;          // ground plane / wall foot, world
        private Quaternion toLocal;      // world -> wall frame (rotation only)
        private Matrix4x4 worldToWall;   // same, with the translation, for the shader
        private Vector3 wallCenter;      // in the wall frame
        private Vector3 wallHalf;
        private Vector4 playerCapA;   // xyz = feet end of the body capsule, w = radius (0 = none)
        private Vector4 playerCapB;   // xyz = head end

#if UDONSHARP
        private VRCPlayerApi localPlayer;
#endif

        void Start()
        {
            origin = transform.position + groundOffset;
            toLocal = Quaternion.Inverse(transform.rotation);
            worldToWall = Matrix4x4.TRS(origin, transform.rotation, Vector3.one).inverse;
            wallCenter = new Vector3(0f, wallHeight, 0f);
            wallHalf = new Vector3(wallWidth, wallHeight, wallThickness);

            // A late joiner may have received the owner's dents already
            if (impactPoints == null)
            {
                impactPoints = new Vector4[MaxImpacts];
                for (int i = 0; i < MaxImpacts; i++)
                    impactPoints[i] = Vector4.zero;
            }
            stateDirty = false;

            lastImpactTime = new float[HandCount];
            touching = new bool[HandCount];
            for (int h = 0; h < HandCount; h++)
            {
                lastImpactTime[h] = -10f;
                touching[h] = false;
            }

            playerCapA = Vector4.zero;
            playerCapB = Vector4.zero;

            MeshRenderer rend = GetComponent<MeshRenderer>();
            if (rend != null)
                mat = rend.material;
            else
                Debug.LogWarning("[ALICE-SDF] SampleDeformableWall_Collider: No MeshRenderer found. Shader sync disabled.");

#if UDONSHARP
            localPlayer = Networking.LocalPlayer;
            useHeld = false;
            missLogged = false;
            lastSyncTime = -1f;
            receivedCount = -1;
            pushing = false;
#endif

            SyncShader();
        }

#if UDONSHARP
        public override void PostLateUpdate()
        {
            if (localPlayer == null) return;

            // --- Impacts: VR hands, or the desktop cursor while Use is held ---
            if (localPlayer.IsUserInVR())
            {
                Vector3 lPos = localPlayer.GetTrackingData(
                    VRCPlayerApi.TrackingDataType.LeftHand).position;
                Vector3 rPos = localPlayer.GetTrackingData(
                    VRCPlayerApi.TrackingDataType.RightHand).position;
                if (IsTrackingValid(lPos))
                    TryImpact(lPos, HandLeft, "left hand");
                if (IsTrackingValid(rPos))
                    TryImpact(rPos, HandRight, "right hand");
            }
            else
            {
                ProcessDesktopCursor();
            }

            // --- Recovery: everyone decays locally (smooth between packets),
            // the owner's values are the ones that get sent ---
            Decay(Time.deltaTime);

            // --- Player Collision: pushed out of the dented wall, body pressed in ---
            Vector3 playerPos = localPlayer.GetPosition();
            float eyeHeight = localPlayer.GetAvatarEyeHeightAsMeters();
            if (eyeHeight <= 0f) eyeHeight = fallbackEyeHeight;
            Vector3 push = PlayerPushOut(playerPos, eyeHeight, Time.deltaTime);
            if (push != Vector3.zero)
            {
                if (!pushing)
                    LogEvent("push at " + F(playerPos) + " by " + F(push));
                pushing = true;
                localPlayer.TeleportTo(playerPos + push, localPlayer.GetRotation());
            }
            else
            {
                pushing = false;
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

        public override void OnDeserialization()
        {
            if (impactCount != receivedCount)
            {
                VRCPlayerApi owner = Networking.GetOwner(gameObject);
                LogEvent("received " + impactCount + " dents (was " + receivedCount + ") from "
                         + (owner != null ? owner.displayName : "?"));
                receivedCount = impactCount;
            }
        }

        // =================================================================
        // Desktop: hold Use (left click) to punch the wall where you look
        // =================================================================
        public override void InputUse(bool value, UdonInputEventArgs args)
        {
            if (localPlayer != null && localPlayer.IsUserInVR()) return;
            useHeld = value;
            if (value) missLogged = false;
            else touching[HandRight] = false;
        }

        private void ProcessDesktopCursor()
        {
            if (!useHeld) return;
            VRCPlayerApi.TrackingData head = localPlayer.GetTrackingData(VRCPlayerApi.TrackingDataType.Head);
            Vector3 o = head.position;
            Vector3 dir = head.rotation * Vector3.forward;
            float t = RaymarchWall(o, dir, cursorMaxDist);
            if (t < 0f)
            {
                if (!missLogged)
                {
                    LogEvent("click missed (no wall within " + F(cursorMaxDist) + " m of the view)");
                    missLogged = true;
                }
                return;
            }
            TryImpact(o + dir * t, HandRight, "left click");
        }
#endif

        // =================================================================
        // Impacts (VR hands and the desktop cursor)
        // =================================================================
        // Register a dent at pos if it is within impactDistance of the wall's
        // undented face, outside the hollow of every live dent, and the hand's
        // cooldown has passed. A hand (or the view cursor) following a fresh
        // dent inward would otherwise drill through the 0.4 m wall in half a
        // second. A hit within half a radius of a live dent refreshes that dent.
        // Returns true when a dent was recorded or refreshed.
        public bool TryImpact(Vector3 pos, int hand, string how)
        {
            float d = SdfBox(ToLocal(pos) - wallCenter, wallHalf);
            if (Mathf.Abs(d) > impactDistance)
            {
                touching[hand] = false;
                return false;
            }
            if (Time.time - lastImpactTime[hand] < impactCooldown) return false;

            // Inside the hollow of a live dent (but not at its centre, which
            // refreshes it): the hand has followed the dent in, no new dent
            for (int i = 0; i < impactCount; i++)
            {
                float w = impactPoints[i].w;
                if (w < DeadStrength) continue;
                Vector3 c = new Vector3(impactPoints[i].x, impactPoints[i].y, impactPoints[i].z);
                float dist = (pos - c).magnitude;
                if (dist >= dentRadius * 0.5f && dist <= dentRadius * w) return false;
            }

            // Denting the wall changes it for everyone: take ownership at
            // the start of the contact
            if (!touching[hand])
            {
                touching[hand] = true;
                if (!IsAuthority()) LogEvent("contact by " + how + ": taking ownership");
                TakeAuthority();
            }

            int slot = RecordImpact(pos);
            lastImpactTime[hand] = Time.time;
            LogEvent("impact #" + slot + " at " + F(pos) + " by " + how + " (" + impactCount + " dents)");
            return true;
        }

        // A live dent within half a radius is refreshed in place; else a free
        // slot, else the weakest dent (the one nearest to recovered)
        private int RecordImpact(Vector3 pos)
        {
            for (int i = 0; i < impactCount; i++)
            {
                if (impactPoints[i].w < DeadStrength) continue;
                Vector3 c = new Vector3(impactPoints[i].x, impactPoints[i].y, impactPoints[i].z);
                if ((pos - c).magnitude < dentRadius * 0.5f)
                {
                    impactPoints[i] = new Vector4(c.x, c.y, c.z, 1f);
                    MarkDirty();
                    return i;
                }
            }
            int slot = -1;
            float weakest = 2f;
            int weakestIdx = 0;
            for (int i = 0; i < MaxImpacts; i++)
            {
                if (impactPoints[i].w < DeadStrength)
                {
                    slot = i;
                    break;
                }
                if (impactPoints[i].w < weakest)
                {
                    weakest = impactPoints[i].w;
                    weakestIdx = i;
                }
            }
            if (slot < 0) slot = weakestIdx;

            impactPoints[slot] = new Vector4(pos.x, pos.y, pos.z, 1f);
            if (slot >= impactCount) impactCount = slot + 1;
            MarkDirty();
            return slot;
        }

        // Strengths decay; impactCount shrinks past trailing dead slots
        public void Decay(float dt)
        {
            if (impactCount == 0) return;
            float k = Mathf.Exp(-decaySpeed * dt);
            bool any = false;
            for (int i = 0; i < impactCount; i++)
            {
                float w = impactPoints[i].w;
                if (w < DeadStrength) continue;
                w *= k;
                if (w < DeadStrength) w = 0f;
                impactPoints[i] = new Vector4(impactPoints[i].x, impactPoints[i].y, impactPoints[i].z, w);
                any = true;
            }
            while (impactCount > 0 && impactPoints[impactCount - 1].w < DeadStrength)
                impactCount--;
            if (any) MarkDirty();
        }

        // Live dents (strength >= DeadStrength)
        public int LiveDents()
        {
            int n = 0;
            for (int i = 0; i < impactCount; i++)
                if (impactPoints[i].w >= DeadStrength) n++;
            return n;
        }

        // Sphere-traced hit distance of a ray against the undented wall (no
        // ground), -1 beyond maxDist: where a click lands, see TryImpact
        public float RaymarchWall(Vector3 o, Vector3 dir, float maxDist)
        {
            float t = 0f;
            for (int i = 0; i < 64; i++)
            {
                float d = SdfBox(ToLocal(o + dir * t) - wallCenter, wallHalf);
                if (d < 0.001f) return t;
                t += d;
                if (t > maxDist) return -1f;
            }
            return -1f;
        }

        // =================================================================
        // Authority: who serializes (the object's owner)
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
        // Player Collision — the dented wall only, sampled along the body
        // =================================================================
        // The ground is VRChat's floor collider; the body is sampled from the
        // feet to the eyes and the deepest sample decides the direction, which
        // is kept horizontal (a wall pushes sideways; lifting only hands the
        // player to gravity). Returns the displacement for this frame, zero
        // when clear.
        public Vector3 PlayerPushOut(Vector3 playerPos, float eyeHeight, float dt)
        {
            Vector3 minP = DeepestBodySample(playerPos, eyeHeight);
            float minDist = EvaluateWallSdf(minP);
            if (minDist >= collisionMargin) return Vector3.zero;

            float pen = Mathf.Min(collisionMargin - minDist, 2.0f);
            if (pen < PushDeadBand) return Vector3.zero;

            Vector3 normal = EstimateGradient(minP);
            Vector3 flat = new Vector3(normal.x, 0f, normal.z);
            float flatLen = flat.magnitude;
            if (flatLen > 0.3f) normal = flat / flatLen;
            else if (normal.y < 0f) normal = Vector3.up;

            float smooth = Mathf.Min(dt * 10f, 1f);
            return normal * pen * pushStrength * smooth;
        }

        public Vector3 DeepestBodySample(Vector3 playerPos, float eyeHeight)
        {
            int n = bodySamples < 2 ? 2 : bodySamples;
            float minDist = 1e10f;
            Vector3 minP = playerPos;
            for (int i = 0; i < n; i++)
            {
                Vector3 s = playerPos + Vector3.up * (eyeHeight * i / (n - 1));
                float d = EvaluateWallSdf(s);
                if (d < minDist)
                {
                    minDist = d;
                    minP = s;
                }
            }
            return minP;
        }

        // =================================================================
        // SDF Evaluation — mirrors the shader map()
        // =================================================================
        public float EvaluateSdf(Vector3 p)
        {
            return Mathf.Min(ToLocal(p).y, EvaluateWallSdf(p));
        }

        // The dented wall alone (no ground, no body dent): what the player
        // collides with and what a hand touches
        public float EvaluateWallSdf(Vector3 p)
        {
            float wall = SdfBox(ToLocal(p) - wallCenter, wallHalf);
            for (int i = 0; i < impactCount; i++)
            {
                float w = impactPoints[i].w;
                if (w < DeadStrength) continue;
                Vector3 c = new Vector3(impactPoints[i].x, impactPoints[i].y, impactPoints[i].z);
                float dent = (p - c).magnitude - dentRadius * w;
                wall = OpSmoothSubtraction(wall, dent, dentSmooth);
            }
            return wall;
        }

        private Vector3 EstimateGradient(Vector3 p)
        {
            float e = 0.02f;
            float dx = EvaluateWallSdf(new Vector3(p.x + e, p.y, p.z))
                     - EvaluateWallSdf(new Vector3(p.x - e, p.y, p.z));
            float dy = EvaluateWallSdf(new Vector3(p.x, p.y + e, p.z))
                     - EvaluateWallSdf(new Vector3(p.x, p.y - e, p.z));
            float dz = EvaluateWallSdf(new Vector3(p.x, p.y, p.z + e))
                     - EvaluateWallSdf(new Vector3(p.x, p.y, p.z - e));
            Vector3 grad = new Vector3(dx, dy, dz);
            float len = grad.magnitude;
            return (len > 0.0001f) ? grad / len : Vector3.up;
        }

        // =================================================================
        // Helpers (inlined SDF, UdonSharp compatible)
        // =================================================================
        private bool IsTrackingValid(Vector3 pos)
        {
            // Tracking returns exactly (0,0,0) when lost; only the exact zero
            // is rejected (a real hand is never bit-exactly at the world origin)
            return pos != Vector3.zero;
        }

        // One line per event in the client log (grep "[Wall]"), off by default
        private void LogEvent(string what)
        {
            if (logEvents) Debug.Log("[Wall] " + what);
        }

        private string F(float v)
        {
            return v.ToString("F2");
        }

        private string F(Vector3 v)
        {
            return "(" + F(v.x) + ", " + F(v.y) + ", " + F(v.z) + ")";
        }

        // World point -> the wall's frame (origin at the wall foot, wall along local X/Y)
        private Vector3 ToLocal(Vector3 p)
        {
            return toLocal * (p - origin);
        }

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

        // Subtracts d2 FROM d1, identical to opSmoothSubtraction in the shader
        private float OpSmoothSubtraction(float d1, float d2, float k)
        {
            if (k < 0.0001f) return Mathf.Max(d1, -d2);
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 + d2), 0f) * invK;
            return Mathf.Max(d1, -d2) + h * h * k * 0.25f;
        }

        // =================================================================
        // Shader Sync
        // =================================================================
        private void SyncShader()
        {
            if (mat == null) return;
            mat.SetVectorArray("_ImpactPoints", impactPoints);
            mat.SetFloat("_ImpactCount", (float)impactCount);

            // Geometry and blend factors live here, not in the material: what
            // the player collides with is exactly what is rendered
            mat.SetFloat("_WallWidth", wallWidth);
            mat.SetFloat("_WallHeight", wallHeight);
            mat.SetFloat("_WallThick", wallThickness);
            mat.SetFloat("_DentRadius", dentRadius);
            mat.SetFloat("_DentSmooth", dentSmooth);

            // Body capsule the shader presses into the wall (w = 0 until the
            // first PostLateUpdate: no player, no dent)
            mat.SetVector("_PlayerCapA", playerCapA);
            mat.SetVector("_PlayerCapB", playerCapB);
            mat.SetFloat("_PlayerDentK", bodyDentK);

            // Placement: the shader evaluates the wall in the same frame
            mat.SetMatrix("_WorldToWall", worldToWall);

            if (applyColors)
            {
                mat.SetColor("_WallColor", wallColor);
                mat.SetColor("_DentColor", dentGlowColor);
                mat.SetColor("_GroundColor", groundColor);
                if (wallTexture != null) mat.SetTexture("_WallTex", wallTexture);
                mat.SetFloat("_WallTexScale", wallTextureScale);
                mat.SetFloat("_WallTexStrength", wallTexture != null ? wallTextureStrength : 0f);
                if (groundTexture != null) mat.SetTexture("_GroundTex", groundTexture);
                mat.SetFloat("_GroundTexScale", groundTextureScale);
                mat.SetFloat("_GroundTexStrength", groundTexture != null ? groundTextureStrength : 0f);
            }
        }
    }
}
