// =============================================================================
// ALICE-SDF Sample: Terrain Sculpt Collider & Interaction (UdonSharp)
// =============================================================================
// Real-time terrain sculpting. VR: left hand adds terrain (SmoothUnion),
// right hand digs (SmoothSubtraction). Desktop: hold the left button (Use)
// to raise the ground under the view cursor, the right button (Drop) to dig.
//
// The key point: rendering AND collision use the same SDF formula. Dig a
// hole and you fall in; build a hill and you climb it. Mesh colliders
// cannot change shape at runtime, an SDF can.
//
// Standing on an SDF: VRChat's player controller needs a Unity collider
// under its feet to be grounded (gravity, jumping, the falling state).
// This script therefore drives a small invisible collider ("support",
// assigned in the Inspector or a scene object named TerrainSupport) that follows
// the player every frame: its top is placed on the SDF surface directly
// below the player (level, the highest point under the foot), so the
// ground the player stands on is always the terrain as it is now. The SDF itself handles
// the rest: buried feet (terrain built where you stand) are lifted onto
// the surface, and the steep flank of a hill pushes you back sideways.
//
// Sculpt operations are stored in a circular buffer (Sculpt Capacity, up to
// 128). When the buffer is full, the oldest operation is overwritten.
//
// Placement: the ground plane sits at transform.position + groundOffset
// (read once in Start); turning the object is fine (the plane is the
// same), tilting it is not supported. Player collision (support box, lift,
// wall push) acts only while the player is inside the volume cube's
// footprint, so several terrains and the world's own floor coexist.
//
// Network: owner-authoritative, manual sync. sculptData / sculptCount /
//   nextSlot are [UdonSynced]; whoever sculpts takes ownership (once per
//   stroke, not every frame) and serializes at 10 Hz while anything
//   changed. Everyone renders and collides with the same received terrain.
//   Late joiners keep the arrays they receive.
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
    public class SampleTerrainSculpt_Collider : UdonSharpBehaviour
#else
    public class SampleTerrainSculpt_Collider : MonoBehaviour
#endif
    {
        /// <summary>Array size of the sculpt buffer. Must match the shader's _SculptData[128].</summary>
        public const int MaxSculpts = 128;

        // Hand slots (VR hands; the desktop cursor uses HandLeft to add and HandRight to dig)
        private const int HandLeft = 0;
        private const int HandRight = 1;
        private const int HandCount = 2;

        // Terrain above the feet up to this is a step for the support collider
        // and the player controller; beyond it the SDF lifts or pushes (m)
        private const float StepLimit = 0.3f;
        // Horizontal gradient share above which a penetration is a wall, not a floor
        private const float WallSlope = 0.7f;
        // Remaining penetration (m) under which the player is left alone
        private const float PushDeadBand = 0.005f;

        [Header("Look (pushed to the material every frame when Apply Colors is on)")]
        [Tooltip("Push the colours and texture below to the material every frame (off = the material's own values)")]
        public bool applyColors = true;
        public Color grassColor = new Color(0.35f, 0.55f, 0.25f, 1f);
        public Color dirtColor = new Color(0.55f, 0.40f, 0.25f, 1f);
        public Color rockColor = new Color(0.40f, 0.38f, 0.35f, 1f);
        [Tooltip("Glow where the left hand / left click will add terrain")]
        public Color addCursorColor = new Color(0.3f, 0.6f, 1.0f, 1f);
        [Tooltip("Glow where the right hand / right click will dig")]
        public Color digCursorColor = new Color(1.0f, 0.3f, 0.2f, 1f);
        [Tooltip("Optional texture on the surface (triplanar, tiles per metre * scale); the rock colour stays underground")]
        public Texture2D groundTexture;
        public float groundTextureScale = 0.5f;
        [Range(0f, 1f)] public float groundTextureStrength = 1.0f;

        [Header("Placement")]
        [Tooltip("Ground plane relative to this object: the prefab's cube is 10 m tall with the ground 3 m under its centre (2 m of digging room below, 7 m of building room above)")]
        public Vector3 groundOffset = new Vector3(0f, -3f, 0f);

        [Header("Sculpting")]
        [Tooltip("Block brush (Minecraft-like): every operation is a cube of side 2 * Sculpt Radius snapped to a grid of that size; digging removes the block just inside the face you point at, building adds the block just outside. Off = spheres placed where you point")]
        public bool blockBrush = true;
        [Tooltip("Brush size: half-side of a block (Block Brush) or radius of the sphere (sent to shader _SculptRadius)")]
        public float sculptRadius = 0.5f;
        [Tooltip("Hand must be within this distance of the terrain surface to sculpt")]
        public float sculptDistance = 0.15f;
        [Tooltip("Minimum time between sculpt operations of one hand (sec)")]
        public float sculptCooldown = 0.12f;
        [Tooltip("SmoothUnion factor for adding terrain (sent to shader _AddSmooth)")]
        public float addSmooth = 0.03f;
        [Tooltip("SmoothSubtraction factor for digging (sent to shader _SubSmooth)")]
        public float subSmooth = 0.03f;
        [Tooltip("Operations kept (1-128); the oldest is overwritten beyond this. Every ray step and every collision sample folds them all, so this is the GPU / Udon cost knob")]
        [Range(1, 128)]
        public int sculptCapacity = 96;

        [Header("Desktop")]
        [Tooltip("How far the view ray looks for terrain (m)")]
        public float cursorMaxDist = 6.0f;

        [Header("Player Collision")]
        [Tooltip("Jump strength given to the player on start so 1 m blocks can be climbed (VRChat worlds start with no jump); 0 = leave the world's own setting")]
        public float jumpImpulse = 3f;
        [Tooltip("Invisible collider that follows the player on the SDF surface (scene object TerrainSupport if empty)")]
        public Transform support;
        [Tooltip("Height of the support box (its top is placed on the surface)")]
        public float supportHeight = 0.2f;
        [Range(0.5f, 1.5f)]
        public float pushStrength = 1.0f;
        [Tooltip("How far above the feet the surface search starts (m); terrain deeper than this over the player is climbed out of")]
        public float surfaceSearchUp = 0.6f;
        [Tooltip("Half-width of the foot (m): the support takes the highest surface under 5 points across it, so the player stays on a ridge until their feet fully leave it instead of the floor flicking at the edge")]
        public float footRadius = 0.12f;

        [Header("Debug")]
        [Tooltip("Debug.Log one line per event (sculpt / click / ownership / floor drop / rise / lift / wall push) as [Terrain] ..., readable in the VRChat client output_log")]
        public bool logEvents = false;

        // Sculpt buffer (xyz = position, w = radius: positive = add, negative = dig), the synced state
#if UDONSHARP
        [UdonSynced] private Vector4[] sculptData;
        [UdonSynced] private int sculptCount = 0;
        [UdonSynced] private int nextSlot = 0;
#else
        private Vector4[] sculptData;
        private int sculptCount = 0;
        private int nextSlot = 0;
#endif
        // Set by every state change; the owner serializes at SyncInterval
        private bool stateDirty;
        private const float SyncInterval = 0.1f;

        // Per-hand cooldown, indexed by HandLeft / HandRight
        private float[] lastSculptTime;
        // A stroke lasts while the hand / cursor stays near the surface;
        // ownership is taken at its start, not on every operation
        private bool[] stroking;

#if UDONSHARP
        // Desktop: button states and the view-ray cursor of this frame
        private bool addHeld;
        private bool digHeld;
        private bool[] missLogged;   // one "missed" line per press that never met terrain
        // Where each button last sculpted: while held, the cursor must move on
        // before it sculpts again (a VR hand stops by itself because the new
        // sphere puts it deeper than sculptDistance; the view cursor always
        // sits on the new surface and would stack a column toward the eyes)
        private Vector3[] lastSculptPos;
        // Operations recorded since a desktop button went down (both buttons
        // up: 0). The view cursor is cast against the terrain without them,
        // i.e. the surface as it was at the press: a held button paints on
        // that surface and cannot climb its own hill toward the eyes
        private int heldOps;
        private Vector3 cursorPos;
        private bool cursorValid;
        private float lastSyncTime;
        private int receivedCount;
        private bool lifting;      // feet were buried last frame (one log line per burial)
        private bool wallPushing;  // in a wall contact chain (one log line per contact)
#endif

        // Where the support top was last frame: the floor log, and the edge
        // rule of SupportHeight (a sample level with the last floor is a ridge
        // the player is still on, not a wall beside them)
        private float lastSupportY;
        private bool supportPlaced;

        // Cached references
        private Material mat;
        private Vector3 origin;      // ground plane, world (only y matters)
        private Bounds volume;       // the cube's world bounds: collision acts inside its footprint
        private bool hasVolume;
        private Vector4 leftCursor;    // sent to the shader as _LeftHand (xyz, w = visible)
        private Vector4 rightCursor;   // sent to the shader as _RightHand

#if UDONSHARP
        private VRCPlayerApi localPlayer;
#endif

        void Start()
        {
            // A late joiner may have received the owner's buffer already
            if (sculptData == null)
            {
                sculptData = new Vector4[MaxSculpts];
                for (int i = 0; i < MaxSculpts; i++)
                    sculptData[i] = Vector4.zero;
            }
            stateDirty = false;

            lastSculptTime = new float[HandCount];
            stroking = new bool[HandCount];
            for (int h = 0; h < HandCount; h++)
            {
                lastSculptTime[h] = -10f;
                stroking[h] = false;
            }

            leftCursor = Vector4.zero;
            rightCursor = Vector4.zero;

            if (support == null)
            {
                GameObject found = GameObject.Find("TerrainSupport");
                if (found != null) support = found.transform;
            }
            if (support == null)
                Debug.LogWarning("[ALICE-SDF] SampleTerrainSculpt_Collider: no support collider (assign Support or add a scene object TerrainSupport with a BoxCollider). Players cannot stand on the terrain.");

            origin = transform.position + groundOffset;
            MeshRenderer rend = GetComponent<MeshRenderer>();
            if (rend != null)
                mat = rend.material;
            else
                Debug.LogWarning("[ALICE-SDF] SampleTerrainSculpt_Collider: No MeshRenderer found. Shader sync disabled.");
            hasVolume = rend != null;
            if (hasVolume) volume = rend.bounds;

#if UDONSHARP
            localPlayer = Networking.LocalPlayer;
            if (localPlayer != null && jumpImpulse > 0f) localPlayer.SetJumpImpulse(jumpImpulse);
            addHeld = false;
            digHeld = false;
            missLogged = new bool[HandCount];
            lastSculptPos = new Vector3[HandCount];
            heldOps = 0;
            cursorValid = false;
            lastSyncTime = -1f;
            receivedCount = -1;
            lifting = false;
            wallPushing = false;
#endif
            supportPlaced = false;
            lastSupportY = 0f;

            SyncShader();
        }

#if UDONSHARP
        public override void PostLateUpdate()
        {
            if (localPlayer == null) return;

            leftCursor = Vector4.zero;
            rightCursor = Vector4.zero;

            if (localPlayer.IsUserInVR())
            {
                // --- Hand Sculpting: left adds, right digs (valid tracking only) ---
                Vector3 lPos = localPlayer.GetTrackingData(
                    VRCPlayerApi.TrackingDataType.LeftHand).position;
                Vector3 rPos = localPlayer.GetTrackingData(
                    VRCPlayerApi.TrackingDataType.RightHand).position;

                if (IsTrackingValid(lPos))
                {
                    TrySculpt(lPos, true, HandLeft, "left hand");
                    leftCursor = CursorFor(lPos);
                }
                if (IsTrackingValid(rPos))
                {
                    TrySculpt(rPos, false, HandRight, "right hand");
                    rightCursor = CursorFor(rPos);
                }
            }
            else
            {
                ProcessDesktopCursor();
            }

            // --- Player Collision: support under the feet, lift when buried, wall push ---
            Vector3 playerPos = localPlayer.GetPosition();
            Vector3 push = PlayerPushOut(playerPos, Time.deltaTime);
            if (push != Vector3.zero)
            {
                float feetDepth = -EvaluateSdf(playerPos);
                if (push.y > 0f && push.x == 0f && push.z == 0f)
                {
                    if (!lifting)
                        LogEvent("lift: feet " + F(feetDepth) + " m under the terrain at " + F(playerPos));
                    lifting = true;
                }
                else
                {
                    if (!wallPushing)
                        LogEvent("wall push at " + F(playerPos) + " by " + F(push));
                    wallPushing = true;
                }
                localPlayer.TeleportTo(playerPos + push, localPlayer.GetRotation());
            }
            else
            {
                lifting = false;
                wallPushing = false;
            }
            PlaceSupport(playerPos);

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
            if (sculptCount != receivedCount)
            {
                VRCPlayerApi owner = Networking.GetOwner(gameObject);
                LogEvent("received " + sculptCount + " sculpts (was " + receivedCount + ") from "
                         + (owner != null ? owner.displayName : "?"));
                receivedCount = sculptCount;
            }
        }

        // =================================================================
        // Desktop: the Use button (left click) raises, Drop (right click) digs,
        // both at the point where the view ray meets the terrain
        // =================================================================
        public override void InputUse(bool value, UdonInputEventArgs args)
        {
            if (localPlayer != null && localPlayer.IsUserInVR()) return;
            addHeld = value;
            if (value) missLogged[HandLeft] = false;
            else stroking[HandLeft] = false;
            if (!addHeld && !digHeld) heldOps = 0;
        }

        public override void InputDrop(bool value, UdonInputEventArgs args)
        {
            if (localPlayer != null && localPlayer.IsUserInVR()) return;
            digHeld = value;
            if (value) missLogged[HandRight] = false;
            else stroking[HandRight] = false;
            if (!addHeld && !digHeld) heldOps = 0;
        }

        private void ProcessDesktopCursor()
        {
            VRCPlayerApi.TrackingData head = localPlayer.GetTrackingData(VRCPlayerApi.TrackingDataType.Head);
            Vector3 o = head.position;
            Vector3 dir = head.rotation * Vector3.forward;
            if (!addHeld && !digHeld) heldOps = 0;
            float t = RaymarchTerrainSkipping(o, dir, cursorMaxDist, heldOps);
            cursorValid = t >= 0f;
            // The law's plane is infinite but this terrain lives in its cube:
            // a hit on the floor elsewhere is not this terrain (nothing is
            // drawn there and nothing would collide)
            if (cursorValid && !InFootprint(o + dir * t))
            {
                cursorValid = false;
                if (addHeld && !missLogged[HandLeft]) { LogEvent("click outside this terrain's footprint at " + F(o + dir * t)); missLogged[HandLeft] = true; }
                if (digHeld && !missLogged[HandRight]) { LogEvent("right click outside this terrain's footprint at " + F(o + dir * t)); missLogged[HandRight] = true; }
                return;
            }
            if (!cursorValid)
            {
                if (addHeld && !missLogged[HandLeft]) { LogEvent("click missed (no terrain within " + F(cursorMaxDist) + " m of the view)"); missLogged[HandLeft] = true; }
                if (digHeld && !missLogged[HandRight]) { LogEvent("right click missed (no terrain within " + F(cursorMaxDist) + " m of the view)"); missLogged[HandRight] = true; }
                return;
            }
            cursorPos = o + dir * t;
            if (addHeld && CursorMoved(HandLeft) && TrySculptSkipping(cursorPos, true, HandLeft, "left click", heldOps))
            {
                lastSculptPos[HandLeft] = cursorPos;
                heldOps++;
            }
            if (digHeld && CursorMoved(HandRight) && TrySculptSkipping(cursorPos, false, HandRight, "right click", heldOps))
            {
                lastSculptPos[HandRight] = cursorPos;
                heldOps++;
            }
            // The cursor glows in the colour of what a click would do
            Vector4 c = new Vector4(cursorPos.x, cursorPos.y, cursorPos.z, 1f);
            if (digHeld) rightCursor = c;
            else leftCursor = c;
        }

        // First operation of a press, or the cursor has moved on from the
        // last one (a stroke paints; staring at one spot sculpts once, click
        // again to stack)
        private bool CursorMoved(int hand)
        {
            if (!stroking[hand]) return true;
            return (cursorPos - lastSculptPos[hand]).magnitude >= sculptRadius * 0.75f;
        }
#endif

        // =================================================================
        // Sculpting (VR hands and the desktop cursor)
        // =================================================================
        // A sculpt at pos: add a hill (isAdd) or dig a hole, if the point is
        // within sculptDistance of the surface and the hand's cooldown has
        // passed. Returns true when an operation was recorded.
        public bool TrySculpt(Vector3 pos, bool isAdd, int hand, string how)
        {
            return TrySculptSkipping(pos, isAdd, hand, how, 0);
        }

        // As TrySculpt, but the surface test ignores the skipRecent most
        // recently recorded operations (the desktop press, see heldOps)
        public bool TrySculptSkipping(Vector3 pos, bool isAdd, int hand, string how, int skipRecent)
        {
            // Only inside this terrain's cube footprint (the plane is infinite,
            // the terrain is not)
            if (!InFootprint(pos))
            {
                stroking[hand] = false;
                return false;
            }

            float dist = EvaluateSdfSkipping(pos, skipRecent);

            // The hand / cursor must be near the terrain surface; leaving it ends the stroke
            if (Mathf.Abs(dist) > sculptDistance)
            {
                stroking[hand] = false;
                return false;
            }

            if (Time.time - lastSculptTime[hand] < sculptCooldown) return false;

            // Sculpting changes the terrain for everyone: take ownership at
            // the start of the stroke
            if (!stroking[hand])
            {
                stroking[hand] = true;
                if (!IsAuthority()) LogEvent("stroke by " + how + ": taking ownership");
                TakeAuthority();
            }

            float r = isAdd ? sculptRadius : -sculptRadius;
            if (blockBrush) pos = SnapToBlock(pos, isAdd);
            RecordSculpt(pos, r);
            lastSculptTime[hand] = Time.time;
            LogEvent((isAdd ? "add" : "dig") + " #" + ((nextSlot + Capacity() - 1) % Capacity()) + " r=" + F(sculptRadius)
                     + " at " + F(pos) + " by " + how + " (" + sculptCount + " stored)");
            return true;
        }

        // Block brush: the grid cell the point falls in, nudged along the
        // surface normal so digging takes the block just inside the face and
        // building the block just outside it (a hit point sits exactly on a
        // grid plane after the first block). Returns the cell centre.
        public Vector3 SnapToBlock(Vector3 pos, bool isAdd)
        {
            float cell = sculptRadius * 2f;
            if (cell < 0.001f) return pos;
            Vector3 n = EstimateGradient(pos);
            Vector3 q = isAdd ? pos + n * (cell * 0.1f) : pos - n * (cell * 0.1f);
            return new Vector3(
                (Mathf.Floor(q.x / cell) + 0.5f) * cell,
                (Mathf.Floor(q.y / cell) + 0.5f) * cell,
                (Mathf.Floor(q.z / cell) + 0.5f) * cell);
        }

        // The ring size in use (Inspector, clamped to the array)
        public int Capacity()
        {
            if (sculptCapacity < 1) return 1;
            if (sculptCapacity > MaxSculpts) return MaxSculpts;
            return sculptCapacity;
        }

        private void RecordSculpt(Vector3 pos, float radius)
        {
            sculptData[nextSlot] = new Vector4(pos.x, pos.y, pos.z, radius);
            nextSlot = (nextSlot + 1) % Capacity();
            if (sculptCount < Capacity()) sculptCount++;
            MarkDirty();
        }

        // Cursor glow for a VR hand: visible while the hand is near the surface
        // (and inside this terrain's footprint)
        private Vector4 CursorFor(Vector3 handPos)
        {
            float d = EvaluateSdf(handPos);
            float visible = (InFootprint(handPos) && Mathf.Abs(d) < sculptRadius * 2f) ? 1f : 0f;
            return new Vector4(handPos.x, handPos.y, handPos.z, visible);
        }

        // Sphere-traced hit distance of a ray against the terrain, -1 beyond
        // maxDist. Steps by |d| so a ray starting inside a hill (the head is
        // never inside, but a cursor test may be) still reaches the surface.
        public float RaymarchTerrain(Vector3 o, Vector3 dir, float maxDist)
        {
            return RaymarchTerrainSkipping(o, dir, maxDist, 0);
        }

        public float RaymarchTerrainSkipping(Vector3 o, Vector3 dir, float maxDist, int skipRecent)
        {
            float t = 0f;
            for (int i = 0; i < 96; i++)
            {
                float d = EvaluateSdfSkipping(o + dir * t, skipRecent);
                if (Mathf.Abs(d) < 0.002f) return t;
                t += Mathf.Max(Mathf.Abs(d), 0.002f);
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
        // Player Collision
        // =================================================================
        // The y of the terrain surface directly below (or, when buried,
        // above) the player's feet. Starts surfaceSearchUp above the feet:
        // if that point is inside the terrain the player is under more
        // terrain than that, so the search first climbs out (up to 4 m),
        // then sphere-traces straight down to the surface. Returns the feet
        // y itself only if no surface is found within 8 m below.
        public float SurfaceHeight(Vector3 feet)
        {
            float x = feet.x;
            float z = feet.z;
            float y = feet.y + surfaceSearchUp;
            float d = EvaluateSdf(new Vector3(x, y, z));
            int guard = 0;
            while (d < 0f && guard < 64)
            {
                y += Mathf.Max(-d, 0.02f);
                if (y > feet.y + 4f) return feet.y;
                d = EvaluateSdf(new Vector3(x, y, z));
                guard++;
            }
            // Downward trace: stop at the surface (d ~ 0), never overshoot
            for (int i = 0; i < 96; i++)
            {
                if (d < 0.001f) return y;
                y -= d;
                if (y < feet.y - 8f) return feet.y;
                d = EvaluateSdf(new Vector3(x, y, z));
            }
            return y;
        }

        // How the terrain meets the feet this frame:
        //   0 = clear, or shallow enough for the support collider and the
        //       controller's step to handle (depth <= StepLimit)
        //   1 = wall: the feet are inside a steep flank with more than
        //       StepLimit of terrain above them (a cliff, not a step)
        //   2 = buried: more than StepLimit of terrain above the feet under a
        //       mostly horizontal surface (built where the player stands)
        public int ContactKind(Vector3 feet)
        {
            float d = EvaluateSdf(feet);
            if (d >= 0f) return 0;
            float depth = SurfaceHeight(feet) - feet.y;
            if (depth <= StepLimit) return 0;
            Vector3 normal = EstimateGradient(feet);
            float flatLen = new Vector3(normal.x, 0f, normal.z).magnitude;
            return flatLen > WallSlope ? 1 : 2;
        }

        // Displacement to apply to the player this frame, zero when the
        // support collider alone handles the contact:
        //   - wall: sideways along the horizontal part of the gradient,
        //     geometrically smoothed like the Mochi push, with a dead band
        //   - buried: straight up onto the surface
        public Vector3 PlayerPushOut(Vector3 feet, float dt)
        {
            if (!InFootprint(feet)) return Vector3.zero;
            int kind = ContactKind(feet);
            if (kind == 0) return Vector3.zero;

            if (kind == 1)
            {
                Vector3 normal = EstimateGradient(feet);
                Vector3 flat = new Vector3(normal.x, 0f, normal.z);
                float pen = Mathf.Min(-EvaluateSdf(feet), 2.0f);
                if (pen < PushDeadBand) return Vector3.zero;
                float smooth = Mathf.Min(dt * 10f, 1f);
                return (flat / flat.magnitude) * pen * pushStrength * smooth;
            }

            float depth = SurfaceHeight(feet) - feet.y;
            return new Vector3(0f, depth + 0.01f, 0f);
        }

        // Where the support top goes: the highest surface under the foot
        // (centre and four points footRadius out, so a ridge carries the
        // player until their feet really leave it; sampled at the centre
        // only, the floor flicked between the ridge and the ground at every
        // edge), except against a wall (the sideways push is handling that)
        // where the surface above would put the box into the legs and the
        // controller would fight it; then the floor stays at the feet
        public float SupportHeight(Vector3 feet)
        {
            if (ContactKind(feet) == 1) return feet.y;
            float h = SurfaceHeight(feet);
            if (footRadius > 0f)
            {
                float r = footRadius;
                h = FootSample(h, new Vector3(feet.x + r, feet.y, feet.z));
                h = FootSample(h, new Vector3(feet.x - r, feet.y, feet.z));
                h = FootSample(h, new Vector3(feet.x, feet.y, feet.z + r));
                h = FootSample(h, new Vector3(feet.x, feet.y, feet.z - r));
            }
            return h;
        }

        // A foot-width sample raises the support only if it is a step above
        // the centre, or level with the floor of the last frame (the ridge the
        // player is walking off). A surface a wall's height above both is the
        // wall the push just cleared them from: taking it would lift the
        // player onto it.
        private float FootSample(float h, Vector3 p)
        {
            float hs = SurfaceHeight(p);
            if (hs <= h) return h;
            if (hs - h <= StepLimit) return hs;
            if (supportPlaced && Mathf.Abs(hs - lastSupportY) <= StepLimit) return hs;
            return h;
        }

        // Place the support box so its top face lies on the surface below
        // the feet
        private void PlaceSupport(Vector3 feet)
        {
            if (support == null) return;
            if (!InFootprint(feet))
            {
                // Outside this terrain: park the box out of the way and let
                // the world's own floor (or another terrain) carry the player
                if (supportPlaced) LogEvent("left the terrain footprint at " + F(feet));
                supportPlaced = false;
                support.position = new Vector3(feet.x, feet.y - 100f, feet.z);
                return;
            }
            float h = SupportHeight(feet);
            // The floor under the player moved by a step or more: a hole was dug
            // or a hill built under them, or they walked onto / off one
            if (supportPlaced && Mathf.Abs(h - lastSupportY) > 0.15f)
                LogEvent("floor " + (h > lastSupportY ? "rose" : "dropped") + " " + F(Mathf.Abs(h - lastSupportY)) + " m to y=" + F(h) + " under " + F(feet));
            lastSupportY = h;
            supportPlaced = true;
            // Level, never tilted to the surface normal: the box is re-placed
            // every frame so a slope is a staircase of millimetre steps the
            // controller walks up, whereas a box tilted at a ridge's edge
            // (steep normal) had the player sliding off it
            support.rotation = Quaternion.identity;
            support.position = new Vector3(feet.x, h - supportHeight * 0.5f, feet.z);
        }

        // =================================================================
        // SDF Evaluation (matches shader map() exactly)
        // =================================================================
        public float EvaluateSdf(Vector3 p)
        {
            return EvaluateSdfSkipping(p, 0);
        }

        // The terrain without the skipRecent most recently recorded
        // operations (the slots just before nextSlot, wrapping): what the
        // terrain was before a desktop press started
        public float EvaluateSdfSkipping(Vector3 p, int skipRecent)
        {
            float terrain = p.y - origin.y;
            if (skipRecent > sculptCount) skipRecent = sculptCount;
            // Skipped slots are [nextSlot - skipRecent, nextSlot) modulo the capacity
            int skipFrom = nextSlot - skipRecent;

            for (int i = 0; i < sculptCount; i++)
            {
                if (skipRecent > 0)
                {
                    int rel = ((i - skipFrom) % Capacity() + Capacity()) % Capacity();
                    if (rel < skipRecent) continue;
                }
                float rw = sculptData[i].w;
                if (rw > 0.001f)
                {
                    // Add: SmoothUnion
                    Vector3 sp = new Vector3(sculptData[i].x, sculptData[i].y, sculptData[i].z);
                    float hill = Brush(p - sp, rw);
                    terrain = OpSmoothUnion(terrain, hill, addSmooth);
                }
                else if (rw < -0.001f)
                {
                    // Dig: SmoothSubtraction
                    Vector3 sp = new Vector3(sculptData[i].x, sculptData[i].y, sculptData[i].z);
                    float hole = Brush(p - sp, -rw);
                    terrain = OpSmoothSubtraction(terrain, hole, subSmooth);
                }
            }

            return terrain;
        }

        // Inside the volume cube's footprint (x / z of its world bounds; the
        // height is the SDF's business). Without a renderer everything counts.
        public bool InFootprint(Vector3 p)
        {
            if (!hasVolume) return true;
            return p.x >= volume.min.x && p.x <= volume.max.x && p.z >= volume.min.z && p.z <= volume.max.z;
        }

        // One sculpt operation's distance: a cube of half-side r (block brush)
        // or a sphere of radius r, identical to the shader's brush()
        private float Brush(Vector3 d, float r)
        {
            if (!blockBrush) return d.magnitude - r;
            float qx = Mathf.Abs(d.x) - r;
            float qy = Mathf.Abs(d.y) - r;
            float qz = Mathf.Abs(d.z) - r;
            float ox = Mathf.Max(qx, 0f);
            float oy = Mathf.Max(qy, 0f);
            float oz = Mathf.Max(qz, 0f);
            return Mathf.Sqrt(ox * ox + oy * oy + oz * oz) + Mathf.Min(Mathf.Max(qx, Mathf.Max(qy, qz)), 0f);
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
        private bool IsTrackingValid(Vector3 pos)
        {
            // Tracking returns exactly (0,0,0) when lost; the terrain spans
            // the origin, so only the exact zero is rejected
            return pos != Vector3.zero;
        }

        // One line per event in the client log (grep "[Terrain]"), off by default
        private void LogEvent(string what)
        {
            if (logEvents) Debug.Log("[Terrain] " + what);
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
            mat.SetVectorArray("_SculptData", sculptData);
            mat.SetFloat("_SculptCount", (float)sculptCount);
            mat.SetFloat("_SculptRadius", sculptRadius);
            mat.SetFloat("_BrushShape", blockBrush ? 1f : 0f);

            // Blend factors live here, not in the material: what the player
            // stands on is exactly what is rendered
            mat.SetFloat("_AddSmooth", addSmooth);
            mat.SetFloat("_SubSmooth", subSmooth);

            mat.SetVector("_LeftHand", leftCursor);
            mat.SetVector("_RightHand", rightCursor);

            // Placement: the shader's ground plane is this object's
            mat.SetVector("_Origin", new Vector4(origin.x, origin.y, origin.z, 0f));

            if (applyColors)
            {
                mat.SetColor("_GrassColor", grassColor);
                mat.SetColor("_DirtColor", dirtColor);
                mat.SetColor("_RockColor", rockColor);
                mat.SetColor("_AddCursorColor", addCursorColor);
                mat.SetColor("_SubCursorColor", digCursorColor);
                if (groundTexture != null) mat.SetTexture("_GroundTex", groundTexture);
                mat.SetFloat("_GroundTexScale", groundTextureScale);
                mat.SetFloat("_GroundTexStrength", groundTexture != null ? groundTextureStrength : 0f);
            }
        }
    }
}
